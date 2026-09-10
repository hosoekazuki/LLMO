"""LLMO 最適化システムの画面（Streamlit）。

このファイルには表示の処理だけを書く。
評価のロジックは llmo/pipeline.py、保存は llmo/db.py にあり、
ここからはその関数を呼ぶだけにする。

表示は「DB から読んだ辞書」だけを受け取る形に統一している。
実行直後の結果も、いったん保存してから読み直して表示する。
実行直後と履歴で描画の処理を分けると、
どちらかを直したときにもう一方がずれていくため。

README の入力のパターンに応じて、表示する条件の数が変わる。

- パターン A（トピックのみ）: 条件 1 つ。単体の水準を出す。
- パターン B（既存記事）    : 条件 2 つ。書き直し前後を並べて出す。
- パターン C（改善の再評価）: 条件 1 つ。改善前との比較は improvements 経由で出す。

どちらのパターンも操作は 2 段階になる。

1. タイトル案を出す（数秒・API 2 回）
2. 案を 1 つ選んで、記事の作成と評価を実行する（数分・API 13 回程度）

1 段階目が軽いので、案が気に入らなければ何度でも出し直せる。
途中の状態は st.session_state に置く。Streamlit は操作のたびに
このファイルを最初から実行し直すため、変数に持たせると消えてしまう。

起動:
    streamlit run app.py
"""

import json
from dataclasses import asdict

import streamlit as st

from llmo import db
from llmo.core.brand import Brand, load_brand, save_brand
from llmo.config import EVAL, MODELS
from llmo.pipeline import (
    evaluate_new_article,
    evaluate_rewrite,
    make_queries,
    prepare_corpus,
    run_improvement,
)
from llmo.evaluation import compare as compare_module
from llmo.generation.title import (
    TitleCandidate,
    generate_titles_from_article,
    generate_titles_from_topic,
)
from llmo.generation.writer import (
    Article,
    from_existing,
    generate_optimized_article,
    rewrite_article,
)
from llmo.corpus.chunk import split_text
from llmo.corpus.clean import clean_documents
from llmo.corpus.fetch import (
    fetch_pages,
    has_corpus,
    list_corpora,
    load_corpus,
    load_meta,
    save_corpus,
    save_meta,
)
from llmo.generation import rules
from llmo.generation.audience import Issue, to_issues
from llmo.generation import leakage
from llmo.evaluation import weakness

st.set_page_config(page_title="LLMO 最適化システム", layout="wide")

# 条件の名前を、画面に出す見出しに対応させる。
CONDITION_LABELS = {
    "single": "評価結果",
    "before": "書き直し前",
    "after": "書き直し後",
}


# ---------------------------------------------------------------------------
# 表示のための小さな部品
# ---------------------------------------------------------------------------

def corpus_name_for(topic: str) -> str:
    """トピックからベースコーパスの保存名を作る。

    トピックごとに分けて、別のトピックの検索対象が混ざらないようにする。
    """
    return "".join(c for c in topic if c.isalnum() or c in "-_") or "corpus"


def format_rate(value: float | None) -> str:
    """割合を見やすい文字列にする。None は「該当なし」を意味する。"""
    return "—" if value is None else f"{value * 100:.0f}%"


def format_number(value: float | None) -> str:
    """平均値などを見やすい文字列にする。"""
    return "—" if value is None else f"{value:.2f}"


def condition_names(view: dict) -> list[str]:
    """この評価に含まれる条件を、表示したい順に返す。"""
    available = set(view["metrics"])
    if view["evaluation"]["pattern"] == "B":
        return [n for n in ("before", "after") if n in available]
    return ["single"]


def render_metric_row(label: str, values: list[str], help_text: str) -> None:
    """指標 1 行を、条件の数だけ横に並べて表示する。"""
    columns = st.columns([3] + [2] * len(values))
    columns[0].markdown(f"**{label}**", help=help_text)
    for column, value in zip(columns[1:], values):
        column.markdown(
            f"<div style='text-align:right'>{value}</div>", unsafe_allow_html=True
        )


# (ラベル, DB の列名, 整形する関数, 説明) の一覧。
# README の「評価に使用する基準」と同じ順に並べる。
# 記事が「検索で拾われ、回答に使われ、推される」という過程をたどる順で、
# どの段階で落ちたのかが読み取れるようにするため。
# 指標が増えたときはここへ 1 行足すだけで済む。

# 回答が作られる前。ベクトル検索で何が起きたかを見る。
SEARCH_METRICS = [
    ("検索ヒット率（Retrieval Rate）", "retrieval_rate", format_rate,
     "自社記事がベクトル検索の上位 K 件に入ったクエリの割合。"
     "最初の関門で、ここで拾われなければ回答の材料にすらならない"),
    ("検索順位（Average Search Rank）", "avg_search_rank", format_number,
     "検索で拾われたときの平均順位（1 始まり）。"
     "分母は拾われたクエリのみで、圏外のクエリは平均に含めない"),
]

# 生成された回答を見る。
ANSWER_METRICS = [
    ("言及率（Mention Rate）", "mention_rate", format_rate,
     "回答本文に自社名・製品名が登場したクエリの割合。"
     "自社記事が引用されなくても、他社記事の中で名前を出されることがあるため、"
     "引用とは独立に測る"),
    ("引用率（Citation Rate）", "citation_rate", format_rate,
     "自社記事の出典番号が回答本文に現れたクエリの割合。"
     "「名前が出たか」ではなく「自社記事が根拠として使われたか」を見る"),
    ("引用シェア（Citation Share）", "citation_share", format_rate,
     "回答中の全引用のうち、自社記事が占めた割合。"
     "引用率が 0/1 なのに対し、こちらは量を見る"),
    ("最推奨率（Top Recommendation Rate）", "top_recommendation_rate", format_rate,
     "複数社を比較する回答のうち、自社が最も推奨された割合。"
     "分母は比較形式の回答のみ。比較形式の回答が無い場合は「—」"),
    ("根拠寄与率（Answer Grounding）", "avg_grounding_rate", format_rate,
     "回答の主張のうち、自社記事の記述が根拠になっている主張の割合。"
     "唯一「記事のどこが効いたか」を返す指標で、"
     "支えられなかった主張が記事の穴になる"),
]


def render_summary(view: dict) -> None:
    """指標の表を表示する。条件が 2 つなら並べて出す。"""
    evaluation = view["evaluation"]
    names = condition_names(view)

    st.subheader("評価結果")

    if evaluation["pattern"] == "B":
        pattern_note = "書き直し前と書き直し後を比較しています"
    elif evaluation["pattern"] == "C":
        pattern_note = "改善ラウンドの再評価です（改善前との比較は下に出します）"
    else:
        pattern_note = "比較対象が無いため、記事単体の水準を示しています"
    st.caption(
        f"{evaluation['created_at']}　/　ベースコーパス {evaluation['num_corpus_docs']} 記事"
        f"　/　クエリ {evaluation['num_queries']} 本 × {evaluation['num_runs']} 回"
        f"　/　{pattern_note}"
    )

    header = st.columns([3] + [2] * len(names))
    header[0].markdown("**指標**")
    for column, name in zip(header[1:], names):
        column.markdown(
            f"<div style='text-align:right'><b>{CONDITION_LABELS[name]}</b></div>",
            unsafe_allow_html=True,
        )
    st.divider()

    st.caption("検索の段階（回答生成前）")
    for label, key, formatter, help_text in SEARCH_METRICS:
        render_metric_row(
            label,
            [formatter(view["metrics"].get(n, {}).get(key)) for n in names],
            help_text,
        )

    st.divider()
    st.caption("回答の段階")

    for label, key, formatter, help_text in ANSWER_METRICS:
        render_metric_row(
            label,
            [formatter(view["metrics"].get(n, {}).get(key)) for n in names],
            help_text,
        )


def render_conditions(evaluation: dict) -> None:
    """その評価がどの条件で測られたかを表示する。

    履歴を見返したときに、条件が違えば数字を比べても意味がない。
    どの条件だったかを結果と同じ場所に置いておく。
    """
    left, right = st.columns(2)
    with left:
        st.text(f"生成モデル　: {evaluation['model_generation']}")
        st.text(f"判定モデル　: {evaluation['model_judge']}")
        st.text(f"埋め込み　　: {evaluation['model_embedding']}")
        st.text(f"疑似ウェブ　: {evaluation['corpus_name']}"
                f"（{evaluation['num_corpus_docs']} 記事）")
    with right:
        st.text(f"クエリ数　　: {evaluation['num_queries']}")
        st.text(f"試行回数　　: {evaluation['num_runs']} 回（平均を取る）")
        st.text(f"取得件数 K　: {evaluation['top_k']}（全条件で共通）")
        st.text(f"チャンク長　: {evaluation['chunk_size']} 字"
                f"（重なり {evaluation['chunk_overlap']} 字）")
        st.text(f"条件キー　　: {evaluation['condition_key']}")


def render_details(view: dict) -> None:
    """クエリごとの詳細を表示する。"""
    st.subheader("クエリ別の詳細")

    names = condition_names(view)
    output_condition = view["evaluation"]["output_condition"]

    # answers は (クエリ番号 × 条件) で並んでいるので、クエリ単位にまとめ直す。
    by_query: dict[int, dict[str, dict]] = {}
    for answer in view["answers"]:
        by_query.setdefault(answer["query_index"], {})[answer["condition"]] = answer

    for query_index in sorted(by_query):
        answers = by_query[query_index]

        # 見出しには出力側の条件の結果を出して、
        # 開かなくても傾向が分かるようにする。
        output = answers.get(output_condition, {})
        rank = output.get("search_rank")
        summary = f"検索 {rank} 位" if rank else "検索圏外"
        summary += " / 引用あり" if output.get("cited") else " / 引用なし"

        query_text = next(iter(answers.values()))["query"]

        with st.expander(f"Q{query_index}. {query_text}　（{summary}）"):
            columns = st.columns(len(names))

            for column, name in zip(columns, names):
                with column:
                    st.markdown(f"##### {CONDITION_LABELS[name]}")
                    answer = answers.get(name)
                    if not answer:
                        st.caption("—")
                        continue

                    # どの出典が渡されたかを先に示す。
                    # 自社記事が何番の出典だったかが分かると、
                    # 回答中の [n] を追いやすい。
                    for source in answer["sources"]:
                        mark = "★ " if source["is_target"] else ""
                        st.caption(f"[{source['number']}] {mark}{source['title'][:40]}")

                    st.markdown(answer["text"])

                    if answer["top_vendor"]:
                        st.caption(f"最も推奨: {answer['top_vendor']}"
                                   f"（{answer['judge_reason']}）")

                    if answer["total_claims"]:
                        st.markdown(
                            f"**根拠寄与率**: {answer['total_claims']} 件中 "
                            f"{answer['supported_claims']} 件を自社記事が支持"
                        )
                        for claim in answer["claims"]:
                            st.markdown(f"- {claim['claim']}")
                            st.caption(f"　根拠: {claim['evidence']}")


def render_leakage(evaluation: dict) -> None:
    """転記チェックの結果と、記事に渡した読者の状況を表示する。

    転記が検出された評価は、検索順位が記事の出来ではなく
    文字列の一致で決まっている可能性がある。
    指標と同じ画面に出さないと、数字だけを見て判断してしまう。
    """
    raw = evaluation.get("leakage_report")
    if raw:
        report = json.loads(raw)
        findings = report["findings"]
        if findings:
            st.error(
                f"**見出しに検索クエリの転記が検出されています（{len(findings)} 件）。**"
                f"最も高い一致率は {report['max_score']}（閾値 {report['threshold']}）。"
                "この評価の検索順位は、記事の出来ではなく文字列の一致で"
                "決まっている可能性があります",
                icon="⚠️",
            )
            with st.expander(f"転記の内訳（{len(findings)} 件）"):
                for finding in findings:
                    mark = "逐語一致" if finding["verbatim"] else "類似"
                    st.markdown(
                        f"**{mark}**　共通 {finding['longest_common_ratio']}"
                        f" / 類似 {finding['similarity']}"
                    )
                    st.caption(f"クエリ　: {finding['query']}")
                    st.caption(f"見出し　: {finding['heading']}")
        else:
            st.success(
                f"転記チェック: 問題なし（最も高い一致率 {report['max_score']}"
                f" / 閾値 {report['threshold']}）",
                icon="✅",
            )
        if report.get("regenerations"):
            st.caption(f"転記を検出したため {report['regenerations']} 回作り直しています")

    raw_issues = evaluation.get("audience_issues")
    if raw_issues:
        issues = json.loads(raw_issues)
        with st.expander(f"記事に渡した読者の状況（{len(issues)} 件）"):
            st.caption(
                "記事生成には検索クエリではなくこちらを渡しています。"
                "クエリの文字列が記事に転記されるのを防ぐためです"
            )
            for issue in issues:
                st.markdown(f"- {issue['situation']}")
                st.caption(f"　元のクエリ: {issue['query']}")


# ---------------------------------------------------------------------------
# 改善ループ
# ---------------------------------------------------------------------------

# 改善の前後で並べる指標。上の表と同じ順・同じ整形を使う。
# 別に持つと、指標を足したときに片方だけ直して食い違う。
COMPARED_METRICS = SEARCH_METRICS + ANSWER_METRICS

# render_view() が 1 回の描画で何度呼ばれたか。
#
# render_view_if_any() はタブ A・B・履歴の 3 か所から呼ばれ、
# 同じ評価が同時に 3 回描かれる。widget のキーを評価 id だけで作ると
# 3 つとも同じキーになり、Streamlit が重複エラーで落ちる。
# 呼ばれた順番をキーに混ぜて避ける。
#
# Streamlit は操作のたびにこのファイルを最初から実行し直すので、
# この変数も毎回 0 に戻る。session_state に置く必要はない。
_view_slot = 0


def format_outcome(outcome) -> str:
    """クエリ 1 本の結果を、1 行の文字列にする。"""
    if outcome is None:
        return "—"
    rank = f"{outcome.search_rank} 位" if outcome.search_rank else "圏外"
    cited = "引用あり" if outcome.cited else "引用なし"
    return f"{rank} / {cited}"


def improvement_inputs(view: dict) -> tuple:
    """保存済みの評価から、改善に必要な入力をそろえる。

    改善は画面で開いている評価から始まる。
    画面はいつでも DB から読んだ辞書しか持っていないので、
    そこから記事・クエリ・改善前の値を組み立て直す。

    Returns:
        (記事, クエリ, 改善前の集計値, 改善前のクエリごとの結果) の組。
    """
    evaluation = view["evaluation"]
    condition = evaluation["output_condition"]

    article = Article(
        title=evaluation["article_title"],
        body=evaluation["article_body"],
        style=evaluation["article_style"],
    )

    outcomes = compare_module.outcomes_from_rows(view["answers"], condition)
    # 評価に使ったクエリを、そのときの並び順で取り出す。
    # 改善前と違うクエリで測ると、前後の差が記事の差ではなくなる。
    queries = [o.query for o in sorted(outcomes, key=lambda o: o.query_index)]

    before = compare_module.snapshot_from_row(view["metrics"].get(condition, {}))

    return article, queries, before, outcomes


def run_improvement_ui(view: dict, limit: int) -> bool:
    """画面から改善を 1 ラウンド実行して、結果を保存する。

    採用しなかった場合も保存する。
    「直したが良くならなかった」という結果も、次に何を試すかの材料になる。
    採用しない場合に元の記事へ書き戻す処理は無い。
    元の評価はそのまま残っているので、そちらを開けばよい。

    Returns:
        改善を実行して保存したか。対象が無くて何もしなかったときは False。
    """
    evaluation = view["evaluation"]
    base_id = evaluation["id"]
    corpus_name = evaluation["corpus_name"]

    article, queries, before, outcomes = improvement_inputs(view)

    with st.status("改善を実行しています", expanded=True) as status:
        def progress(message: str) -> None:
            status.write(message)

        base = prepare_corpus(corpus_name, progress=progress)

        improvement = run_improvement(
            topic=evaluation["topic"],
            article=article,
            before=before,
            before_outcomes=outcomes,
            brand=brand,
            queries=queries,
            base=base,
            limit=limit,
            round_number=db.next_round_number(base_id),
            progress=progress,
        )

        if improvement.skipped:
            status.update(
                label="弱点のあるクエリがありませんでした",
                state="complete",
                expanded=True,
            )
            return False

        progress("結果を保存しています...")
        after_id = db.save_evaluation(
            improvement.after_result,
            brand,
            corpus_name,
            note=f"改善 {improvement.round} 回目（評価 {base_id} から）",
            # 読者の状況は改善前のものをそのまま引き継ぐ。
            # 作り直すと、記事を書いた前提が前後で変わってしまう。
            issues=json.loads(evaluation["audience_issues"] or "[]"),
            leakage=improvement.leakage_report,
        )
        db.save_improvement(improvement, base_id, after_id)

        status.update(
            label=(
                "改善しました（採用）" if improvement.adopted
                else "改善しませんでした（不採用）"
            ),
            state="complete",
            expanded=False,
        )

    st.session_state["view_id"] = after_id
    return True


def render_improvement_controls(view: dict, slot: int) -> None:
    """この評価から改善を実行するための操作を出す。

    Args:
        view: db.load_evaluation() が返した辞書。
        slot: 同じ描画の中で何番目に呼ばれたか。widget のキーに混ぜる。
    """
    evaluation = view["evaluation"]

    st.divider()
    st.subheader("LLMO 改善")

    # 改善は、改善前と同じ条件で測り直せる場合にだけ意味を持つ。
    # 条件が変わっていると、前後の差が記事の差なのか条件の差なのか分からない。
    current_key = db.condition_key(
        evaluation["corpus_name"], evaluation["num_queries"]
    )
    if current_key != evaluation["condition_key"]:
        st.warning(
            "この評価は現在と違う条件（モデル・チャンク長・疑似ウェブなど）で"
            "測られています。改善しても前後を比較できないため、実行できません",
            icon="⚠️",
        )
        return

    _, _, _, outcomes = improvement_inputs(view)
    if not outcomes:
        st.caption("クエリごとの結果が残っていないため、改善対象を判定できません")
        return

    weak = weakness.identify_weak_queries(outcomes, limit=len(outcomes))

    if not weak:
        st.success(
            "すべてのクエリで検索・引用・言及のいずれも満たしています。"
            "改善対象がありません",
            icon="✅",
        )
        return

    st.caption(
        f"弱点のあるクエリが {len(weak)} 本あります。"
        "優先度の高いものから、記事の該当する節だけを書き直して測り直します"
    )

    with st.expander(f"弱点のあるクエリ（{len(weak)} 本）"):
        for item in weak:
            st.markdown(f"**{item.query}**　`{item.label}`")
            st.caption(f"　{item.reason}")

    left, right = st.columns([1, 3])
    limit = left.number_input(
        "改善するクエリ数",
        min_value=1,
        max_value=min(5, len(weak)),
        value=min(3, len(weak)),
        key=f"improve_limit_{evaluation['id']}_{slot}",
        help="分析と修正の API 呼び出しがこの本数に比例します",
    )
    with right:
        st.caption(
            f"実行すると API を約 {int(limit) * 2 + evaluation['num_queries']} 回"
            f"（生成）＋ 最大 {evaluation['num_queries'] * 2} 回（判定）呼びます。"
            "1 回押すと 1 ラウンドだけ実行します。自動では繰り返しません"
        )

    if st.button("LLMO 改善を実行", type="primary", key=f"improve_{evaluation['id']}_{slot}"):
        # 改善対象が見つからずに終わった場合は描き直さない。
        # 描き直すと status に出した理由が消えてしまう。
        if run_improvement_ui(view, int(limit)):
            st.rerun()


def render_improvement_result(improvement: dict, view: dict) -> None:
    """この評価が改善で生まれたものであるとき、改善前との比較を出す。"""
    base_id = improvement["base_evaluation_id"]
    base_view = db.load_evaluation(base_id)

    st.divider()
    st.subheader(f"改善 {improvement['round']} 回目の結果")

    if base_view is None:
        st.warning("改善前の評価が削除されているため、比較を表示できません", icon="⚠️")
        return

    base_condition = base_view["evaluation"]["output_condition"]
    after_condition = view["evaluation"]["output_condition"]

    # --- 採否 -------------------------------------------------------------
    delta = improvement["score_after"] - improvement["score_before"]
    message = (
        f"総合スコア {improvement['score_before']:.3f} → "
        f"{improvement['score_after']:.3f}（{delta:+.3f}）"
    )
    if improvement["adopted"]:
        st.success(f"**修正版を採用しました。** {message}", icon="✅")
    else:
        st.error(
            f"**修正版は採用しません。** {message}　"
            f"改善前の評価（id {base_id}）の記事を使ってください",
            icon="⚠️",
        )
    st.caption(
        "総合スコアは 検索ヒット率 0.30 / 引用率 0.30 / 引用シェア 0.15 / "
        "言及率 0.15 / 根拠寄与率 0.10 の加重和です。"
        "検索順位と最推奨率は分母が変動するため合成に含めていません"
    )

    # --- 指標の前後 -------------------------------------------------------
    st.markdown("#### 指標の変化")
    header = st.columns([3, 2, 2, 2])
    header[0].markdown("**指標**")
    for column, label in zip(header[1:], ("改善前", "改善後", "変化")):
        column.markdown(
            f"<div style='text-align:right'><b>{label}</b></div>",
            unsafe_allow_html=True,
        )

    for label, key, formatter, help_text in COMPARED_METRICS:
        before_value = base_view["metrics"].get(base_condition, {}).get(key)
        after_value = view["metrics"].get(after_condition, {}).get(key)

        if before_value is None or after_value is None:
            change = "—"
        else:
            difference = after_value - before_value
            # 検索順位だけは小さいほうが良いので、矢印の向きを反転させる。
            improved = difference < 0 if key == "avg_search_rank" else difference > 0
            mark = "▲" if improved else ("▼" if difference else "→")
            change = f"{mark} {formatter(abs(difference))}" if difference else "→"

        render_metric_row(
            label,
            [formatter(before_value), formatter(after_value), change],
            help_text,
        )

    # --- クエリごとの前後 -------------------------------------------------
    st.markdown("#### クエリごとの変化")
    before_outcomes = compare_module.outcomes_from_rows(
        base_view["answers"], base_condition
    )
    after_outcomes = compare_module.outcomes_from_rows(
        view["answers"], after_condition
    )
    comparison = compare_module.compare(
        compare_module.snapshot_from_row(base_view["metrics"].get(base_condition, {})),
        compare_module.snapshot_from_row(view["metrics"].get(after_condition, {})),
        before_outcomes,
        after_outcomes,
    )

    targeted = {w["query"] for w in improvement["weak_queries"]}
    st.dataframe(
        [
            {
                "対象": "◎" if item.query in targeted else "",
                "変化": "変化あり" if item.changed else "",
                "クエリ": item.query,
                "改善前": format_outcome(item.before),
                "改善後": format_outcome(item.after),
            }
            for item in comparison.per_query
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "「対象」が付いたクエリを狙って修正しました。"
        "それ以外のクエリが下がっていないかも確認してください。"
        "ある節を厚くすると、別のクエリで拾われなくなることがあります"
    )

    # --- 分析と修正 -------------------------------------------------------
    st.markdown("#### 改善の内容")
    for analysis in improvement["analyses"]:
        with st.expander(f"{analysis['query']} → 「{analysis['target_section']}」を修正"):
            st.markdown(f"**問題**　{analysis['problem']}")
            st.markdown("**不足していた情報**")
            for missing in analysis["missing_information"]:
                st.markdown(f"- {missing}")
            st.markdown(f"**改善の指示**　{analysis['improvement_instruction']}")

    for edit in improvement["section_edits"]:
        kind = "新しく追加した節" if edit["is_new"] else "書き直した節"
        with st.expander(f"{kind}: {edit['heading']}"):
            st.caption("対象クエリ: " + "、".join(edit["queries"]))
            if edit["is_new"]:
                st.markdown(edit["after"])
            else:
                before_tab, after_tab = st.tabs(["修正前", "修正後"])
                with before_tab:
                    st.markdown(edit["before"])
                with after_tab:
                    st.markdown(edit["after"])

    # --- 作り話の疑い -----------------------------------------------------
    warnings = improvement["fact_warnings"]
    if warnings:
        st.warning(
            f"**修正で新しく現れた数値が {len(warnings)} 件あります。**"
            "元の記事にも自社情報にも無い数値です。"
            "根拠が説明できるかを確認してください",
            icon="⚠️",
        )
        st.caption(
            "、".join(f"{w['text']}（{w['kind']}）" for w in warnings)
        )
    else:
        st.caption("修正で新しく現れた数値はありません（社数・料金・割合などを検査）")


def render_view(evaluation_id: int) -> None:
    """保存済みの評価 1 件を丸ごと表示する。"""
    global _view_slot
    _view_slot += 1
    slot = _view_slot

    view = db.load_evaluation(evaluation_id)
    if view is None:
        st.error("その評価は見つかりませんでした（削除された可能性があります）")
        return

    evaluation = view["evaluation"]

    render_summary(view)

    # この評価が改善で生まれたものなら、改善前との比較をここに出す。
    # 指標の表のすぐ下に置かないと、数字だけを見て良し悪しを判断してしまう。
    improvement = db.load_improvement_of(evaluation_id)
    if improvement:
        render_improvement_result(improvement, view)

    render_improvement_controls(view, slot)

    # パターン B では書き直し前の記事も見られるようにする。
    # 何がどう変わったのかは、指標だけでは分からないため。
    if evaluation["pattern"] == "B" and evaluation["original_body"]:
        with st.expander(f"書き直し前の記事: {evaluation['original_title']}"):
            st.markdown(evaluation["original_body"])
        label = "書き直し後の記事"
    else:
        label = "評価した記事"

    with st.expander(f"{label}: {evaluation['article_title']}"):
        st.markdown(evaluation["article_body"])

    render_leakage(evaluation)

    with st.expander("この評価の条件"):
        render_conditions(evaluation)

    render_details(view)


# ---------------------------------------------------------------------------
# サイドバー
# ---------------------------------------------------------------------------

brand = load_brand()

with st.sidebar:
    st.header("企業情報")
    st.caption("「自社情報」タブで編集できます")
    st.text(f"会社名　: {brand.company}")
    st.text(f"製品名　: {brand.product}")
    st.text(f"ドメイン: {brand.domain}")

    st.header("評価条件")
    st.caption("llmo/config.py で変更できます")
    st.text(f"クエリ数　　: {EVAL.num_queries}")
    st.text(f"試行回数　　: {EVAL.num_runs}")
    st.text(f"取得件数 K　: {EVAL.top_k}（全条件で共通）")
    st.text(f"チャンク長　: {EVAL.chunk_size} 字")
    st.text(f"生成モデル　: {MODELS.generation}")
    st.text(f"判定モデル　: {MODELS.judge}")

    st.header("実行設定")
    note = st.text_input(
        "この評価のメモ",
        help="「製品名の指示を追加した版」など。"
        "後で履歴を見返したときに、どの版だったかを思い出すために使います",
    )


# ---------------------------------------------------------------------------
# 実行
# ---------------------------------------------------------------------------

def generate_checked(build, queries: list[str], progress) -> tuple:
    """記事を作り、見出しにクエリが転記されていないかを調べる。

    転記が起きると、そのクエリで検索したときに文字列の一致だけで
    上位に入ってしまい、記事の出来を測れなくなる。
    設定で作り直しを有効にしていれば、検出時に上限回数まで作り直す。

    Args:
        build: 記事を作る関数（引数なしで Article を返す）。
        queries: 評価に使うクエリ。照合の相手。
        progress: 経過を書き出す関数。

    Returns:
        (記事, 検証結果) の組。
    """
    article = build()
    report = leakage.check(article.body, queries)

    attempts = 0
    while (
        report.has_leakage
        and EVAL.regenerate_on_leakage
        and attempts < EVAL.max_regenerations
    ):
        attempts += 1
        progress(
            f"見出しにクエリの転記を検出しました（最大 {report.max_score}）。"
            f"作り直します（{attempts}/{EVAL.max_regenerations}）..."
        )
        article = build()
        report = leakage.check(article.body, queries)

    report.regenerations = attempts

    if report.has_leakage:
        progress(
            f"⚠ 見出しにクエリの転記が残っています"
            f"（{len(report.findings)} 件 / 最大 {report.max_score}）。"
            "検索順位が記事の出来ではなく文字列の一致で決まっている可能性があります"
        )
    else:
        progress(f"転記チェック: 問題なし（最大 {report.max_score}）")

    return article, report


def propose_titles_a(topic: str, corpus_name: str, num: int) -> None:
    """パターン A の 1 段階目: クエリとタイトル案を作る。

    クエリは記事のトピックから作る。疑似ウェブに保存されているクエリは
    そのコーパスを集めるために使ったものであり、評価に使うものとは別である。
    """
    with st.status("タイトル案を作成しています", expanded=True) as status:
        def progress(message: str) -> None:
            status.write(message)

        queries = make_queries(topic, progress=progress)
        progress("クエリを読者の状況に言い換えています...")
        issues = to_issues(queries)
        progress("タイトル案を作成しています...")
        candidates = generate_titles_from_topic(topic, brand, issues, num)

        status.update(label="タイトル案ができました", state="complete", expanded=False)

    # 2 段階目で使うので、クエリと検索対象も一緒に持っておく。
    # ここで作り直すと、タイトルを決めた前提と評価の前提がずれる。
    st.session_state["stage_a"] = {
        "topic": topic,
        "corpus_name": corpus_name,
        "queries": queries,
        "issues": issues,
        "candidates": candidates,
    }


def run_pattern_a(stage: dict, title: str) -> None:
    """パターン A の 2 段階目: 記事を作って単体で評価する。"""
    topic = stage["topic"]
    queries = stage["queries"]
    # 検索対象は、トピックから導くのではなく、ユーザーが選んだものを使う。
    corpus_name = stage["corpus_name"]

    with st.status("評価を実行しています", expanded=True) as status:
        def progress(message: str) -> None:
            status.write(message)

        base = prepare_corpus(corpus_name, progress=progress)

        progress("最適化記事を生成しています...")
        article, report = generate_checked(
            lambda: generate_optimized_article(topic, title, brand, stage["issues"]),
            queries,
            progress,
        )
        progress(f"記事を生成しました（{len(article.body)} 字）")

        result = evaluate_new_article(
            topic, article, brand, queries, base, progress=progress
        )

        progress("結果を保存しています...")
        evaluation_id = db.save_evaluation(
            result,
            brand,
            corpus_name,
            note=note,
            issues=[asdict(i) for i in stage["issues"]],
            leakage=report.to_dict(),
        )
        status.update(label="評価が完了しました", state="complete", expanded=False)

    st.session_state["view_id"] = evaluation_id


def propose_titles_b(
    topic: str, corpus_name: str, title: str, body: str, num: int
) -> None:
    """パターン B の 1 段階目: クエリと、書き直し後のタイトル案を作る。"""
    with st.status("タイトル案を作成しています", expanded=True) as status:
        def progress(message: str) -> None:
            status.write(message)

        queries = make_queries(topic, progress=progress)
        progress("クエリを読者の状況に言い換えています...")
        issues = to_issues(queries)
        original = from_existing(title, body)
        progress("記事の内容からタイトル案を作成しています...")
        candidates = generate_titles_from_article(original, brand, issues, num)

        status.update(label="タイトル案ができました", state="complete", expanded=False)

    st.session_state["stage_b"] = {
        "topic": topic,
        "corpus_name": corpus_name,
        "queries": queries,
        "issues": issues,
        "original_title": title,
        "original_body": body,
        "candidates": candidates,
    }


def run_pattern_b(stage: dict, title: str) -> None:
    """パターン B の 2 段階目: 書き直して、書き直し前後を比較する。"""
    topic = stage["topic"]
    queries = stage["queries"]
    # 検索対象は、トピックから導くのではなく、ユーザーが選んだものを使う。
    corpus_name = stage["corpus_name"]

    with st.status("評価を実行しています", expanded=True) as status:
        def progress(message: str) -> None:
            status.write(message)

        base = prepare_corpus(corpus_name, progress=progress)

        original = from_existing(stage["original_title"], stage["original_body"])
        progress("記事を書き直しています...")
        rewritten, report = generate_checked(
            lambda: rewrite_article(original, title, brand, stage["issues"]),
            queries,
            progress,
        )
        progress(
            f"書き直しました（{len(original.body)} 字 → {len(rewritten.body)} 字）"
        )

        result = evaluate_rewrite(
            topic, original, rewritten, brand, queries, base, progress=progress
        )

        progress("結果を保存しています...")
        evaluation_id = db.save_evaluation(
            result,
            brand,
            corpus_name,
            note=note,
            issues=[asdict(i) for i in stage["issues"]],
            leakage=report.to_dict(),
        )
        status.update(label="評価が完了しました", state="complete", expanded=False)

    st.session_state["view_id"] = evaluation_id


def render_title_choice(
    stage: dict, key: str, button_label: str
) -> str | None:
    """タイトル案を並べて選ばせる。

    案だけを並べても何が違うのか分からないので、
    選んだ案の切り口の説明を下に出す。

    Returns:
        実行ボタンが押されたときだけ、選ばれたタイトル。それ以外は None。
    """
    candidates: list[TitleCandidate] = stage["candidates"]

    with st.expander(f"この記事が答える想定質問（{len(stage['queries'])} 本）"):
        for query in stage["queries"]:
            st.markdown(f"- {query}")

    st.markdown("#### タイトル案")
    selected = st.radio(
        "記事にするタイトルを 1 つ選んでください",
        options=range(len(candidates)),
        format_func=lambda i: candidates[i].title,
        key=f"choice_{key}",
    )
    st.caption(f"この切り口: {candidates[selected].reason}")

    if st.button(button_label, type="primary", key=f"run_{key}"):
        return candidates[selected].title
    return None


def render_view_if_any() -> None:
    """表示中の評価があれば出す。

    タブの外に置くと、Streamlit はどのタブを開いていても本文の下に描くため、
    設定用のタブ（自社情報・最適化の指示）にまで評価結果が出てしまう。
    結果を見たいタブの中からそれぞれ呼ぶ。
    """
    view_id = st.session_state.get("view_id")
    if view_id:
        st.divider()
        render_view(view_id)


# ---------------------------------------------------------------------------
# 疑似ウェブの表示
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def corpus_overview(name: str) -> list[dict]:
    """疑似ウェブの中身を、表示に必要な形にして返す。

    掃除とチャンク分割は何度やっても同じ結果になるが、
    Streamlit は操作のたびにこのファイルを最初から実行し直すため、
    記事を 1 つ開くだけで全件の処理が走ってしまう。
    コーパス名をキーにして結果を取っておく。

    ベクトルは扱わない。数千次元の数値を見ても分からないためである。
    """
    documents = load_corpus(name)
    # clean_documents は元のリストと同じ順序・同じ件数を返すので、
    # 添字で対応させて「生の本文」と「掃除後の本文」を並べられる。
    cleaned = clean_documents(documents)
    return [
        {
            "title": doc.title or "（タイトルなし）",
            "url": doc.url,
            "query": doc.found_by_query or "—",
            "raw": doc.content,
            "clean": clean.content,
            "raw_chars": len(doc.content),
            "clean_chars": len(clean.content),
            "num_chunks": len(split_text(clean.content)),
            # build_base はここで落とすので、画面でも分かるようにしておく。
            # 判定に使うのは掃除前の文字数（build_base と同じ条件にする）。
            "dropped": len(doc.content) < EVAL.min_content_chars,
        }
        for doc, clean in zip(documents, cleaned)
    ]


def render_corpus(name: str) -> None:
    """1 つの疑似ウェブの中身を表示する。"""
    meta = load_meta(name)
    if meta:
        st.caption(
            f"作成 {meta['created_at'].replace('T', ' ')}"
            f"　/　クエリ {len(meta['queries'])} 本"
        )
        if meta.get("description"):
            st.caption(meta["description"])
        with st.expander(f"取得に使ったクエリ（{len(meta['queries'])} 本）"):
            for query in meta["queries"]:
                st.markdown(f"- {query}")
    else:
        st.caption(
            "この疑似ウェブにはクエリの記録がありません"
            "（この機能より前に作られたものです）。評価に使うには作り直してください"
        )

    docs = corpus_overview(name)
    kept = [d for d in docs if not d["dropped"]]

    columns = st.columns(4)
    columns[0].metric("記事数", f"{len(kept)} 件")
    columns[1].metric("総チャンク数", f"{sum(d['num_chunks'] for d in kept)} 個")
    columns[2].metric(
        "平均文字数",
        f"{sum(d['clean_chars'] for d in kept) // len(kept):,} 字" if kept else "—",
    )
    columns[3].metric("取得クエリ数", f"{len({d['query'] for d in docs})} 本")

    if len(kept) < len(docs):
        st.caption(
            f"{len(docs) - len(kept)} 件は本文が {EVAL.min_content_chars} 字未満のため、"
            "検索対象から除かれています（本文の取得に失敗したページ）"
        )

    st.dataframe(
        [
            {
                "": "—" if d["dropped"] else "✓",
                "タイトル": d["title"],
                "掃除後": d["clean_chars"],
                "生": d["raw_chars"],
                "チャンク": d["num_chunks"],
                "取得クエリ": d["query"],
                "URL": d["url"],
            }
            for d in docs
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "「掃除後」は広告・ナビゲーションを除いた本文の文字数。"
        "検索も回答生成もこちらを使います。"
        "生との差が極端に大きい記事は、掃除が本文まで削っている可能性があります"
    )

    st.subheader("本文")
    st.caption("見出しをクリックすると開きます")
    for number, doc in enumerate(docs, start=1):
        mark = "" if not doc["dropped"] else "（対象外）"
        with st.expander(
            f"{number}. {doc['title']}{mark}"
            f"　— {doc['clean_chars']:,} 字 / {doc['num_chunks']} チャンク"
        ):
            st.caption(f"{doc['url']}　取得クエリ: {doc['query']}")
            clean_view, raw_view = st.tabs(["掃除後（実際に使うもの）", "生の本文"])
            with clean_view:
                st.text(doc["clean"])
            with raw_view:
                st.text(doc["raw"])


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

st.title("LLMO 最適化システム")
st.caption(
    "記事を疑似ウェブに差し込み、生成 AI の回答に引用されるかを定量的に評価します"
)

def corpus_label(name: str) -> str:
    """選択肢に出す名前。メタ情報があれば、そこに書かれたトピック名を使う。"""
    meta = load_meta(name)
    return meta["topic"] if meta and meta.get("topic") else name


def pick_corpus(key: str) -> str | None:
    """評価に使う疑似ウェブを選ばせる。

    自由入力ではなく選択にしているのは、存在しないトピックを打てないようにするため。
    疑似ウェブは評価より前に作っておくものなので、
    ここで打った名前から新しく作られることはない。

    Returns:
        選ばれたコーパスの保存名。1 つも無ければ None。
    """
    names = list_corpora()
    if not names:
        st.warning(
            "疑似ウェブがまだありません。"
            "「疑似ウェブ」タブで先に作成してください",
            icon="⚠️",
        )
        return None

    picked = st.selectbox(
        "トピック（疑似ウェブ）",
        options=names,
        format_func=corpus_label,
        key=key,
        help="「疑似ウェブ」タブで作成したものから選びます",
    )
    meta = load_meta(picked)
    if meta:
        st.caption(
            f"{meta.get('num_documents', '?')} 記事"
            f"　/　このコーパスを集めるのに使ったクエリ {len(meta['queries'])} 本"
            f"　/　作成 {meta['created_at'].replace('T', ' ')}"
        )
    st.caption(
        "**記事のトピックと噛み合った疑似ウェブを選んでください。**"
        "関係のないコーパスを選ぶと、そのトピックを扱っている文書が自社記事しか無くなり、"
        "記事の出来にかかわらず検索で上位に入ります"
    )
    return picked


topic_tab, article_tab, brand_tab, rules_tab, corpus_tab, history_tab = st.tabs(
    [
        "A. トピックから記事を作る",
        "B. 既存の記事を書き直す",
        "自社情報",
        "最適化の指示",
        "疑似ウェブ",
        "履歴",
    ]
)

with topic_tab:
    st.caption(
        "トピックから LLMO 最適化した記事を作成し、そのまま評価します。"
        "書き直す前にあたるものが無いため、**比較は行いません**"
    )
    topic = st.text_input(
        "記事のトピック",
        key="topic_input",
        help="このトピックから、評価用のクエリとタイトル案を作ります",
    )
    corpus_a = pick_corpus("corpus_a")
    num_titles_a = st.number_input(
        "タイトル案の数",
        min_value=3,
        max_value=15,
        value=EVAL.num_title_candidates,
        key="num_titles_a",
    )
    if st.button("タイトル案を出す", key="propose_a", disabled=corpus_a is None):
        if not topic.strip():
            st.error("記事のトピックを入力してください")
        else:
            propose_titles_a(topic.strip(), corpus_a, int(num_titles_a))

    stage_a = st.session_state.get("stage_a")
    if stage_a:
        st.divider()
        chosen = render_title_choice(
            stage_a, "a", "この案で記事を作成して評価する"
        )
        if chosen:
            run_pattern_a(stage_a, chosen)

    render_view_if_any()

with article_tab:
    existing_title = st.text_input("記事のタイトル", key="existing_title")
    existing_body = st.text_area("記事の本文", height=300, key="existing_body")
    existing_topic = st.text_input(
        "記事のトピック",
        key="existing_topic",
        help="このトピックから評価用のクエリを作ります",
    )
    corpus_b = pick_corpus("corpus_b")
    st.caption(
        "入力した記事の内容からタイトル案を作り、選んだタイトルで書き直して、"
        "**書き直す前と後を比較**します"
    )
    num_titles_b = st.number_input(
        "タイトル案の数",
        min_value=3,
        max_value=15,
        value=EVAL.num_title_candidates,
        key="num_titles_b",
    )
    if st.button("タイトル案を出す", key="propose_b", disabled=corpus_b is None):
        if not existing_body.strip():
            st.error("記事の本文を入力してください")
        elif not existing_title.strip():
            st.error("記事のタイトルを入力してください")
        elif not existing_topic.strip():
            st.error("記事のトピックを入力してください")
        else:
            propose_titles_b(
                existing_topic.strip(),
                corpus_b,
                existing_title,
                existing_body,
                int(num_titles_b),
            )

    stage_b = st.session_state.get("stage_b")
    if stage_b:
        st.divider()
        chosen = render_title_choice(
            stage_b, "b", "この案で書き直して評価する"
        )
        if chosen:
            run_pattern_b(stage_b, chosen)

    render_view_if_any()

with brand_tab:
    st.caption(
        "記事を出す企業の情報です。**記事の生成と、評価の判定の両方**で使います。"
        "ここを差し替えれば別の企業に使えます"
    )

    # 保存のあとに入力欄の中身を差し替えるため、版番号をキーに混ぜる。
    # 最適化の指示のタブと同じ理由（Streamlit は同じキーの widget の値を保持する）。
    brand_revision = st.session_state.get("brand_revision", 0)

    left_column, right_column = st.columns(2)
    with left_column:
        company_input = st.text_input(
            "企業名", value=brand.company, key=f"brand_company_{brand_revision}",
            help="回答本文にこの名前が出たかで言及率を判定します",
        )
        domain_input = st.text_input(
            "ドメイン", value=brand.domain, key=f"brand_domain_{brand_revision}",
            help="未公開記事に与える URL の組み立てに使います。"
            "https:// やスラッシュは含めないでください",
        )
    with right_column:
        product_input = st.text_input(
            "製品・サービス名", value=brand.product, key=f"brand_product_{brand_revision}",
            help="企業名と別に持ちます。回答では製品名だけが挙がることが多いためです",
        )
        audience_input = st.text_input(
            "想定読者", value=brand.audience, key=f"brand_audience_{brand_revision}",
            help="記事のトーンと前提知識の水準を決めるのに使います",
        )

    description_input = st.text_area(
        "事業内容",
        value=brand.description,
        height=120,
        key=f"brand_description_{brand_revision}",
        help="記事を書くときの前提として LLM に渡します",
    )

    strengths_input = st.text_area(
        "製品の強み（1 行に 1 つ）",
        value="\n".join(brand.strengths),
        height=140,
        key=f"brand_strengths_{brand_revision}",
        help="記事の中で自然に触れさせるために渡します",
    )

    if st.button("保存する", type="primary", key="brand_save"):
        edited_brand = Brand(
            company=company_input.strip(),
            product=product_input.strip(),
            description=description_input.strip(),
            domain=domain_input.strip(),
            # 空行を落として一覧にする。行を消すつもりで残った空行が、
            # 空文字の強みとしてプロンプトに載るのを防ぐ。
            strengths=[line.strip() for line in strengths_input.splitlines() if line.strip()],
            audience=audience_input.strip(),
            # 表記ゆれは画面から編集しない。
            # 直接 data/brand.json に書いた内容を消さないよう、そのまま持ち越す。
            aliases=brand.aliases,
        )
        try:
            save_brand(edited_brand)
        except ValueError as error:
            st.error(str(error))
        else:
            st.session_state["brand_revision"] = brand_revision + 1
            st.success("保存しました")
            st.rerun()

    st.caption(
        "言及率の照合に使う語: "
        + "、".join(f"`{term}`" for term in brand.mention_terms)
        + "（部分一致・大文字小文字は区別します）"
    )


with rules_tab:
    st.caption(
        "記事を LLMO 最適化するときに、LLM に渡している指示です。"
        "**新規作成（A）と書き直し（B）の両方**に同じものが差し込まれます。"
        "条件を揃えないと、書き直しの効果を測れなくなるためです"
    )

    if rules.is_customized():
        st.info("既定から変更されています")
    else:
        st.caption("現在は既定の内容です")

    # 保存や初期化のあとに入力欄の中身を差し替えるため、
    # 版番号をキーに混ぜる。Streamlit は同じキーの widget の値を
    # 保持し続けるので、キーを変えないと画面が古い内容のままになる。
    revision = st.session_state.get("rules_revision", 0)
    edited = st.text_area(
        "指示",
        value=rules.load(),
        height=480,
        key=f"rules_text_{revision}",
        label_visibility="collapsed",
    )

    st.caption(
        "使える差し込み変数: "
        + "、".join(f"`{{{name}}}`" for name in rules.PLACEHOLDERS)
        + "（それぞれチャンク長・製品名に置き換わります）。"
        "これ以外の `{ }` は保存時に弾かれます"
    )

    save_column, reset_column = st.columns([1, 4])

    if save_column.button("保存する", type="primary"):
        try:
            rules.save(edited)
        except ValueError as error:
            st.error(str(error))
        else:
            st.session_state["rules_revision"] = revision + 1
            st.success("保存しました。次に作成する記事から反映されます")
            st.rerun()

    if rules.is_customized():
        if reset_column.button("既定に戻す"):
            rules.reset()
            st.session_state["rules_revision"] = revision + 1
            st.rerun()

    st.warning(
        "ここを変えても、**過去の評価結果とは比較できます**。"
        "指示は測る道具ではなく測られる側であり、変えた前後を比べること自体が目的だからです。"
        "どの指示で書かれた記事かは評価結果とともに保存されます。"
        "一方、サイドバーに出ている評価条件（チャンク長・取得件数など）は測り方そのものなので、"
        "変えると過去の結果と比較できなくなります",
        icon="ℹ️",
    )

    with st.expander("既定の内容を見る"):
        st.code(rules.DEFAULT_RULES, language="markdown")


with corpus_tab:
    st.caption(
        "評価の検索対象になる疑似ウェブです。**評価より前に、ここで作っておきます。**"
        "評価の途中で検索対象が入れ替わると、観測された差が"
        "「記事を変えたから」なのか「検索対象が変わったから」なのか区別できなくなるためです"
    )

    with st.expander("新しく作る／取り直す"):
        new_topic = st.text_input(
            "トピック",
            key="new_corpus_topic",
            help="評価するときにこの名前で選びます",
        )
        new_description = st.text_area(
            "概要",
            height=120,
            key="new_corpus_description",
            help="どんな読者が何を知りたいトピックなのかを書いてください。"
            "この内容から、想定される質問をクエリ案として提示します。"
            "空欄ならトピック名だけから作ります",
        )

        if st.button("クエリ案を出す", key="propose_queries"):
            if not new_topic.strip():
                st.error("トピックを入力してください。")
            else:
                with st.status("クエリ案を作成しています", expanded=True) as status:
                    proposed = make_queries(
                        new_description.strip() or new_topic.strip(),
                        progress=status.write,
                    )
                    status.update(
                        label=f"クエリ案を {len(proposed)} 本作成しました",
                        state="complete",
                        expanded=False,
                    )
                st.session_state["query_draft"] = proposed
                # 案を出し直したら、前の案に対する承認の状態は捨てる。
                st.session_state["query_revision"] = (
                    st.session_state.get("query_revision", 0) + 1
                )

        draft = st.session_state.get("query_draft")
        if draft:
            st.divider()
            st.markdown("**このクエリで取得します。外すものはチェックを外してください**")
            st.caption(
                "文面は直せます。ただし自社記事が見つかりやすい語を入れると、"
                "検索されて当然の結果になり、評価が意味を持たなくなります。"
                "実際の読者が打ちそうな質問のままにしてください"
            )

            revision = st.session_state.get("query_revision", 0)
            approved: list[str] = []
            # 案の本数だけ行を出す。空欄にした行は捨てる。
            for index in range(len(draft) + 1):
                is_extra = index == len(draft)
                checkbox_column, text_column = st.columns([1, 12])
                keep = checkbox_column.checkbox(
                    "使う",
                    value=not is_extra,
                    key=f"query_keep_{revision}_{index}",
                    label_visibility="collapsed",
                )
                text = text_column.text_input(
                    "クエリ",
                    value="" if is_extra else draft[index],
                    placeholder="自分で追記する場合はここに書く",
                    key=f"query_text_{revision}_{index}",
                    label_visibility="collapsed",
                )
                if keep and text.strip():
                    approved.append(text.strip())

            st.caption(
                f"承認したクエリ: {len(approved)} 本"
                "（この本数が、言及率などの割合の分母になります）"
            )

            target_name = corpus_name_for(new_topic)
            if has_corpus(target_name):
                existing_meta = load_meta(target_name) or {}
                created = existing_meta.get("created_at", "不明")
                count = existing_meta.get("num_documents", "?")
                st.warning(
                    f"「{new_topic}」の疑似ウェブはすでにあります"
                    f"（{count} 記事 / 作成 {created}）。"
                    "**作り直すと上書きされ、検索対象が変わるため、"
                    "このトピックの過去の評価結果とは比較できなくなります**",
                    icon="⚠️",
                )

            if st.button(
                "このクエリで疑似ウェブを作成する",
                type="primary",
                key="build_corpus",
                disabled=not approved,
            ):
                with st.status("疑似ウェブを作成しています", expanded=True) as status:
                    status.write(
                        f"ウェブ検索で記事を取得しています（{len(approved)} クエリ）..."
                    )
                    documents = fetch_pages(approved)
                    status.write(f"{len(documents)} 件を取得しました")

                    save_corpus(documents, target_name)
                    save_meta(
                        target_name,
                        topic=new_topic.strip(),
                        description=new_description.strip(),
                        queries=approved,
                        num_documents=len(documents),
                    )
                    status.update(
                        label=f"疑似ウェブを作成しました（{len(documents)} 記事）",
                        state="complete",
                        expanded=False,
                    )
                del st.session_state["query_draft"]
                st.rerun()

    corpus_names = list_corpora()
    if not corpus_names:
        st.info(
            "保存された疑似ウェブがまだありません。"
            "タブ A か B で評価を実行すると作成されます。"
        )
    else:
        picked = st.selectbox(
            "疑似ウェブ",
            options=corpus_names,
            help="トピックごとに分けて保存しています",
        )
        render_corpus(picked)


with history_tab:
    # 現在の設定での条件キー。これと一致する行だけが比較してよい相手になる。
    # A タブで選ばれている疑似ウェブを基準にする。
    current_key = db.condition_key(corpus_a) if corpus_a else None
    if current_key:
        st.caption(f"現在の条件キー: `{current_key}`")
    else:
        st.caption("疑似ウェブが未選択のため、比較可能かどうかの判定は出しません")

    summaries = db.list_evaluations(limit=50)

    if not summaries:
        st.info("まだ評価結果がありません。")
    else:
        # 条件が違う行は消さずに残し、印を付けるだけにする。
        # 条件を変えたという事実自体が記録として必要なため。
        table = [
            {
                "比較": "✓" if s.condition_key == current_key else "⚠ 条件違い",
                "id": s.id,
                "日時": s.created_at.replace("T", " "),
                "パターン": s.pattern,
                "トピック": s.topic,
                "版": s.article_style,
                "メモ": s.note,
                "検索ヒット": format_rate(s.retrieval_rate),
                "検索順位": format_number(s.avg_search_rank),
                "言及": format_rate(s.mention_rate),
                "引用": format_rate(s.citation_rate),
                "引用シェア": format_rate(s.citation_share),
                "最推奨": format_rate(s.top_recommendation_rate),
                "根拠寄与": format_rate(s.avg_grounding_rate),
            }
            for s in summaries
        ]
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption(
            "指標はいずれも**出力側の記事**のもの"
            "（A は作成した記事、B は書き直し後の記事）。"
            "⚠ が付いた行は、モデル・チャンク長・ベースコーパスのいずれかが"
            "現在と違う条件で測られています。数字を並べて比べないでください。"
        )

        selected = st.selectbox(
            "詳しく見る評価",
            options=[s.id for s in summaries],
            format_func=lambda i: next(
                f"[{s.id}] {s.created_at.replace('T', ' ')} / パターン {s.pattern}"
                f" / {s.article_style}" + (f" / {s.note}" if s.note else "")
                for s in summaries if s.id == i
            ),
        )
        if st.button("この評価を表示する"):
            st.session_state["view_id"] = selected


    render_view_if_any()

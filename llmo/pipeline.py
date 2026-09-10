"""評価の流れ全体を 1 つにまとめるモジュール。

README の評価フロー（クエリ生成 → タイトル案の提示 → ユーザーの選択 →
ベースコーパス構築 → 記事の生成 → 条件ごとの検索 → 回答生成 → 指標の算出）
のうち、ロジック側の各段階をここから呼べるようにする。

タイトルの選択で処理が一度止まるので、入口を段階ごとに分けている。
クエリ生成（make_queries）とコーパス構築（prepare_corpus）が別の関数なのはそのため。

画面から呼ぶのはこのファイルの関数だけにする。
Streamlit を別のフロントに差し替えたくなったときに、
ロジック側を一切変えずに済むようにするため。

README の入力のパターンに対応して、入口を 2 つ用意している。

- evaluate_new_article(): パターン A。条件は 1 つ（"single"）。比較しない。
- evaluate_rewrite():     パターン B。条件は 2 つ（"before" / "after"）。比較する。

どちらも中身は同じ _evaluate_conditions() を呼ぶ。
条件が 1 つか 2 つかの違いしかなく、
指標の計算やコーパスの扱いはまったく同じであるため。

評価した後に記事を直して測り直す入口をもう 1 つ持つ。

- run_improvement(): 評価結果から弱いクエリを見つけ、該当する節だけを直し、
  もう一度評価して前後を比べる（パターン "C"）。

改善後の評価も _evaluate_conditions() をそのまま使う。条件は 1 つ（"single"）で、
改善前の値は保存済みの評価から取る。改善前を測り直さないのは API の予算のためで、
判定モデルの日次上限がこのシステムの律速になっているためである。
測り直したい場合は、改善前の記事をもう一度評価してから比べればよい。
"""

from dataclasses import dataclass, field
from typing import Callable

from llmo.retrieval.answer import Answer, build_sources, generate_answer
from llmo.core.brand import Brand
from llmo.config import EVAL
from llmo.corpus.fetch import has_corpus
from llmo.retrieval.experiment import BaseCorpus, Condition, build_base, build_condition
from llmo.corpus.embed import embed_queries
from llmo.evaluation.judge import (
    GroundingResult,
    RecommendationResult,
    judge_grounding,
    judge_recommendation,
    top_recommendation_rate,
)
from llmo.evaluation.compare import (
    Comparison,
    MetricSnapshot,
    compare,
    outcomes_from_result,
    snapshot_from_result,
)
from llmo.evaluation.metrics import (
    AggregatedMetrics,
    QueryMetrics,
    QueryOutcome,
    aggregate,
    evaluate_answer,
)
from llmo.evaluation.weakness import WeakQuery, identify_weak_queries
from llmo.generation import leakage
from llmo.generation.improve import (
    Analysis,
    FactWarning,
    SectionEdit,
    analyze_weakness,
    check_new_facts,
    revise_sections,
)
from llmo.generation.query import generate_queries
from llmo.retrieval.retrieve import find_rank
from llmo.generation.writer import Article

# 進捗を伝えるための関数の型。
# 画面側がこれを受け取って表示を更新する。
# print で書くとフロントに出せないので、呼び出し側に渡してもらう。
ProgressCallback = Callable[[str], None]

# 条件の名前。文字列を直接書くとタイプミスに気づけないので定数にする。
# 改善後の再評価も条件は 1 つなので SINGLE を使い回す。
# 名前を増やすと画面の対応表にも足す必要があり、増やす利点が無い。
SINGLE = "single"
BEFORE = "before"
AFTER = "after"


def _noop(message: str) -> None:
    """進捗を捨てる既定の実装（コマンドラインから試すとき用）。"""


@dataclass
class QueryResult:
    """クエリ 1 本について、全条件の結果をまとめたもの。

    辞書のキーはすべて条件の名前（"single" または "before" / "after"）。
    """

    query: str
    answers: dict[str, Answer] = field(default_factory=dict)
    metrics: dict[str, QueryMetrics] = field(default_factory=dict)
    recommendation: dict[str, RecommendationResult] = field(default_factory=dict)

    # 根拠判定は全条件で行う。
    # どの条件にも未公開記事が入っているため、どちらも支える側になりうる。
    # 検索で拾われず出典に入らなかった条件は None になる。
    grounding: dict[str, GroundingResult | None] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """評価 1 回分の結果全体。画面はこれを受け取って表示する。"""

    # "A"（トピックのみ・比較なし）か "B"（書き直し前後の比較）か。
    pattern: str

    topic: str

    # 条件の名前を、表示したい順に並べたもの。
    # 画面はこれを見て 1 列で出すか 2 列で出すかを決める。
    condition_names: list[str]

    # 条件ごとに差し込んだ記事。
    articles: dict[str, Article]

    queries: list[str]

    # ベースコーパスに入っている他社記事の数。
    num_corpus_docs: int

    # 各条件を何回まわして平均したか。
    # この値が違う結果同士は、数字の安定度が違うので並べて比べられない。
    num_runs: int

    per_query: list[QueryResult] = field(default_factory=list)

    # 条件ごとの集計値。
    aggregated: dict[str, AggregatedMetrics] = field(default_factory=dict)

    # Top Recommendation Rate。比較形式の回答が無ければ None。
    top_recommendation: dict[str, float | None] = field(default_factory=dict)

    # Answer Grounding の平均。条件ごとに出す。
    avg_grounding_rate: dict[str, float | None] = field(default_factory=dict)

    @property
    def output_article(self) -> Article:
        """このシステムの出力にあたる記事。

        パターン A は作成した記事、パターン B は書き直した後の記事。
        """
        return self.articles[self.condition_names[-1]]


def make_queries(topic: str, progress: ProgressCallback = _noop) -> list[str]:
    """ユーザーの入力から、評価に使うクエリを作る。

    コーパス構築とは別の関数にしている。
    タイトル案を作る段階でクエリが必要になるが、
    その時点ではまだコーパスを作る必要がないため。
    コーパス構築は初回に数分かかるので、
    タイトルを選び直すたびに待たされるのは無駄が大きい。

    ここで作ったクエリは、以降すべての段階で同じものを使い回す。
    タイトル案・記事の生成・評価で別のクエリを使うと、
    「その質問に答える記事か」という前提が揃わなくなるため。
    """
    progress("クエリを生成しています...")
    return generate_queries(topic)


def prepare_corpus(
    corpus_name: str,
    progress: ProgressCallback = _noop,
) -> BaseCorpus:
    """保存済みの疑似ウェブを読み込んで、検索できる形にする。

    ここでは取得を行わない。
    評価の途中で検索対象が入れ替わると、観測された差が
    「記事を変えたから」なのか「検索対象が変わったから」なのか
    区別できなくなるためである。
    疑似ウェブの作成は画面の「疑似ウェブ」タブで、評価とは別の操作として行う。

    Args:
        corpus_name: コーパスの保存名。

    Returns:
        ベースコーパス。

    Raises:
        FileNotFoundError: そのコーパスがまだ作られていないとき。
    """
    if not has_corpus(corpus_name):
        raise FileNotFoundError(
            f"疑似ウェブ「{corpus_name}」がまだ作られていません。"
            "「疑似ウェブ」タブで先に作成してください。"
        )

    progress("疑似ウェブをベクトル化しています（初回は数分かかります）...")
    base = build_base(corpus_name)
    progress(f"ベースコーパス: {len(base.documents)} 記事 / {len(base.chunks)} チャンク")

    return base


def _evaluate_conditions(
    pattern: str,
    topic: str,
    base: BaseCorpus,
    conditions: list[Condition],
    brand: Brand,
    queries: list[str],
    progress: ProgressCallback,
) -> EvaluationResult:
    """条件のリストを受け取って、クエリごとに評価する。

    パターン A・B の違いは、渡ってくる条件が 1 つか 2 つかだけ。
    指標の計算はまったく同じなので、ここに 1 本化している。
    """
    progress("クエリをベクトル化しています...")
    query_vectors = embed_queries(queries)

    names = [c.name for c in conditions]

    # 集計用の入れ物。試行を繰り返すぶん、
    # 1 条件あたり (クエリ数 × 試行回数) 件の観測値が入る。
    per_query: list[QueryResult] = []
    metrics: dict[str, list[QueryMetrics]] = {n: [] for n in names}
    recommendations: dict[str, list[RecommendationResult]] = {n: [] for n in names}
    grounding_rates: dict[str, list[float]] = {n: [] for n in names}

    for index, (query, query_vector) in enumerate(zip(queries, query_vectors), start=1):
        progress(f"クエリ {index}/{len(queries)} を評価しています...")
        result = QueryResult(query=query)

        for condition in conditions:
            # 検索は毎回同じ結果になる（ベクトルもコーパスも固定）ので、
            # 試行を繰り返す外には置かない。ここで 1 度だけ実行する。
            search_results = condition.search(query_vector)

            # 差し込んだ記事が検索で何位だったか。圏外なら None。
            rank = find_rank(search_results, condition.target_index)

            sources = build_sources(search_results, condition.documents)

            # 同じ入力で複数回まわす。
            # LLM の出力は毎回変わるため、1 回だけだと
            # 「プロンプトを変えたから良くなった」のか「たまたま」なのか区別できない。
            # 試行ごとの観測値をすべて集め、最後にまとめて平均する。
            for run in range(EVAL.num_runs):
                if EVAL.num_runs > 1:
                    progress(
                        f"クエリ {index}/{len(queries)}・{condition.name}"
                        f"・試行 {run + 1}/{EVAL.num_runs}"
                    )

                answer = generate_answer(query, sources, condition.name)
                query_metrics = evaluate_answer(answer, brand, rank)
                recommendation = judge_recommendation(answer, brand)

                # 根拠判定は、その条件で記事が出典に入っていたときだけ行う。
                # 入っていなければ支えようがない（judge 側が None を返す）。
                grounding = judge_grounding(answer)

                metrics[condition.name].append(query_metrics)
                recommendations[condition.name].append(recommendation)
                if grounding:
                    grounding_rates[condition.name].append(grounding.grounding_rate)

                # 画面に出すのは 1 回目の試行だけにする。
                # 平均した数字の裏付けとして回答を 1 本見せるのが目的であり、
                # 全試行を並べても読み切れないため。
                if run == 0:
                    result.answers[condition.name] = answer
                    result.metrics[condition.name] = query_metrics
                    result.recommendation[condition.name] = recommendation
                    result.grounding[condition.name] = grounding

        per_query.append(result)

    progress("集計しています...")
    return EvaluationResult(
        pattern=pattern,
        topic=topic,
        condition_names=names,
        articles={c.name: c.article for c in conditions},
        queries=queries,
        num_corpus_docs=len(base.documents),
        num_runs=EVAL.num_runs,
        per_query=per_query,
        aggregated={n: aggregate(values, n) for n, values in metrics.items()},
        top_recommendation={
            n: top_recommendation_rate(values) for n, values in recommendations.items()
        },
        avg_grounding_rate={
            n: (sum(rates) / len(rates) if rates else None)
            for n, rates in grounding_rates.items()
        },
    )


def evaluate_new_article(
    topic: str,
    article: Article,
    brand: Brand,
    queries: list[str],
    base: BaseCorpus,
    slug: str = "article",
    progress: ProgressCallback = _noop,
) -> EvaluationResult:
    """パターン A: 作成した記事を単体で評価する（比較なし）。

    条件は 1 つだけ。書き直す前にあたるものが存在しないため、
    比べる相手がいない。各指標は「その記事が単体でどの水準に達したか」を示す。
    """
    progress("記事を疑似ウェブに差し込んでいます...")
    condition = build_condition(base, article, brand.article_url(slug), SINGLE)

    return _evaluate_conditions(
        pattern="A",
        topic=topic,
        base=base,
        conditions=[condition],
        brand=brand,
        queries=queries,
        progress=progress,
    )


def evaluate_rewrite(
    topic: str,
    original: Article,
    rewritten: Article,
    brand: Brand,
    queries: list[str],
    base: BaseCorpus,
    slug: str = "article",
    progress: ProgressCallback = _noop,
) -> EvaluationResult:
    """パターン B: 書き直し前と書き直し後を比較する。

    ベースコーパスは共通のまま、差し込む記事だけを入れ替える。
    2 条件の差分は「差し込んだ記事が書き直し前か後か」だけになるので、
    観測された差は書き直しによる効果とみなせる。

    URL は両条件で同じものを与える。
    同じ記事の書き直し前後であり、URL が違うと出典の見え方が変わって、
    記事の中身以外の要因が入り込むため。
    """
    url = brand.article_url(slug)

    progress("書き直し前の記事を疑似ウェブに差し込んでいます...")
    before = build_condition(base, original, url, BEFORE)

    progress("書き直し後の記事を疑似ウェブに差し込んでいます...")
    after = build_condition(base, rewritten, url, AFTER)

    return _evaluate_conditions(
        pattern="B",
        topic=topic,
        base=base,
        conditions=[before, after],
        brand=brand,
        queries=queries,
        progress=progress,
    )


# ---------------------------------------------------------------------------
# 改善ループ（パターン C）
# ---------------------------------------------------------------------------

# 改善後の評価に付けるパターン名。
# 'A' / 'B' と区別するのは、履歴で「これは改善ラウンドの結果だ」と
# 分かるようにするため。条件は 1 つ（SINGLE）なので、
# 表示側は 'A' とまったく同じ扱いで描ける。
IMPROVED = "C"


@dataclass
class ImprovementResult:
    """改善 1 ラウンド分の結果。

    弱点の特定から再評価までを 1 つにまとめる。
    途中の判断（どのクエリを選び、何が足りないと分析し、どの節を直したか）を
    すべて持たせているのは、指標が動いたときに
    「どの修正が効いたのか」を後から辿れるようにするためである。
    """

    # 何回目の改善か（1 始まり）。
    round: int

    # 改善対象に選んだクエリ。1 本も無ければ改善は行われない。
    weak_queries: list[WeakQuery]

    # クエリごとの改善点の分析。
    analyses: list[Analysis]

    # 実際に書き直した節。
    edits: list[SectionEdit]

    # 修正で新しく現れた数値（作り話の疑い）。
    fact_warnings: list[FactWarning]

    # 修正版に対する転記チェックの結果。
    leakage_report: dict

    # 修正版の記事。採用しない場合でも、何ができたかを見せるために持つ。
    article: Article | None

    # 修正版の評価結果。改善対象が無ければ None。
    after_result: EvaluationResult | None

    # 改善前後の比較。改善対象が無ければ None。
    comparison: Comparison | None

    @property
    def skipped(self) -> bool:
        """改善すべきクエリが見つからず、何もしなかったか。"""
        return not self.weak_queries

    @property
    def adopted(self) -> bool:
        """修正版を採用してよいか。"""
        return bool(self.comparison and self.comparison.adopted)


def run_improvement(
    topic: str,
    article: Article,
    before: MetricSnapshot,
    before_outcomes: list[QueryOutcome],
    brand: Brand,
    queries: list[str],
    base: BaseCorpus,
    limit: int = 3,
    round_number: int = 1,
    slug: str = "article",
    progress: ProgressCallback = _noop,
) -> ImprovementResult:
    """評価結果をもとに記事を直し、もう一度評価して前後を比べる。

    自動では繰り返さない。1 回の呼び出しが 1 ラウンドで、
    続けるかどうかは呼び出し側（画面）が決める。
    改善は必ず良くなるとは限らず、無人で回すと悪化した版に上書きし続けるため。

    Args:
        topic: 記事のトピック。保存と表示に使う。
        article: 改善前の記事。評価済みのもの。
        before: 改善前の集計値。保存済みの評価から取ってよい。
        before_outcomes: 改善前の、クエリごとの観測結果。
            弱点の特定と、クエリ単位の前後比較の両方に使う。
        brand: 自社の情報。作ってよい事実の範囲を決める。
        queries: 評価に使うクエリ。改善前とまったく同じものを渡すこと。
            違うクエリで測ると、前後の差が記事の差なのか
            クエリの差なのか区別できなくなる。
        base: ベースコーパス。改善前と同じものを渡すこと。
        limit: 改善対象にするクエリの本数の上限。
        round_number: 何回目の改善か。記録用。
        slug: 記事に与える URL の末尾。改善前と同じにする。

    Returns:
        改善 1 ラウンド分の結果。comparison.adopted が True のときだけ
        修正版を採用してよい。
    """
    weak = identify_weak_queries(before_outcomes, limit=limit)

    if not weak:
        progress("弱点のあるクエリが見つかりませんでした。改善は行いません")
        return ImprovementResult(
            round=round_number,
            weak_queries=[],
            analyses=[],
            edits=[],
            fact_warnings=[],
            leakage_report={},
            article=None,
            after_result=None,
            comparison=None,
        )

    progress(f"改善対象のクエリを {len(weak)} 本選びました")
    for item in weak:
        progress(f"　- {item.query}（{item.label}: {item.reason}）")

    url = brand.article_url(slug)

    # 改善前の記事を差し込んだ条件を組み立て直す。
    # 分析に渡す「自社記事の該当部分」と「競合の該当部分」を取るために必要で、
    # ここで作るベクトルは手元の埋め込みモデルで計算するため API は使わない。
    progress("記事と競合の該当部分を取り出しています...")
    current = build_condition(base, article, url, SINGLE)

    # 弱いクエリのぶんだけベクトル化する。全クエリを作り直す必要はない。
    weak_vectors = embed_queries([item.query for item in weak])

    # --- 分析 -------------------------------------------------------------
    analyses: list[Analysis] = []
    for index, (item, vector) in enumerate(zip(weak, weak_vectors), start=1):
        progress(f"改善点を分析しています（{index}/{len(weak)}）: {item.query}")
        analyses.append(
            analyze_weakness(
                article=article,
                weak=item,
                brand=brand,
                own_chunks=current.target_chunks(vector),
                rival_chunks=current.rival_chunks(vector),
            )
        )

    for analysis in analyses:
        progress(f"　「{analysis.target_section}」を修正: {analysis.problem}")

    # --- 修正 -------------------------------------------------------------
    progress("該当する節を書き直しています...")
    revised, edits = revise_sections(article, analyses, brand, progress=progress)
    progress(
        f"{len(edits)} 節を修正しました"
        f"（{len(article.body)} 字 → {len(revised.body)} 字）"
    )

    # --- 検査 -------------------------------------------------------------
    # 改善ループは評価クエリを狙って書き換えるため、通常の生成より
    # 見出しへの転記が起きやすい。転記があると検索順位が記事の出来ではなく
    # 文字列の一致で決まってしまい、改善したように見えるだけになる。
    report = leakage.check(revised.body, queries)
    if report.has_leakage:
        progress(
            f"⚠ 修正後の見出しにクエリの転記があります"
            f"（{len(report.findings)} 件 / 最大 {report.max_score}）"
        )
    else:
        progress(f"転記チェック: 問題なし（最大 {report.max_score}）")

    fact_warnings = check_new_facts(article.body, revised.body, brand)
    if fact_warnings:
        progress(
            f"⚠ 修正で新しく現れた数値が {len(fact_warnings)} 件あります。"
            "根拠があるか確認してください"
        )

    # --- 再評価 -----------------------------------------------------------
    progress("修正版を疑似ウェブに差し込んでいます...")
    condition = build_condition(base, revised, url, SINGLE)

    after_result = _evaluate_conditions(
        pattern=IMPROVED,
        topic=topic,
        base=base,
        conditions=[condition],
        brand=brand,
        queries=queries,
        progress=progress,
    )

    # --- 比較 -------------------------------------------------------------
    comparison = compare(
        before=before,
        after=snapshot_from_result(after_result, SINGLE),
        before_queries=before_outcomes,
        after_queries=outcomes_from_result(after_result, SINGLE),
    )

    if comparison.adopted:
        progress(
            f"総合スコアが {comparison.score_before:.3f} → "
            f"{comparison.score_after:.3f} に上がりました。修正版を採用します"
        )
    else:
        progress(
            f"総合スコアが {comparison.score_before:.3f} → "
            f"{comparison.score_after:.3f} で改善しませんでした。"
            "修正版は採用しません"
        )

    return ImprovementResult(
        round=round_number,
        weak_queries=weak,
        analyses=analyses,
        edits=edits,
        fact_warnings=fact_warnings,
        leakage_report=report.to_dict(),
        article=revised,
        after_result=after_result,
        comparison=comparison,
    )

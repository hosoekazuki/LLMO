"""評価結果を SQLite に保存するモジュール。

目的は「記事を直しながら指標の推移を追う」こと。
そのために 2 つのことを守っている。

1. 実験条件をスナップショットで一緒に保存する。
   チャンク長や top_k、モデル、疑似ウェブの中身が変われば数字は動く。
   条件の違う結果を並べて「良くなった」と言えば嘘になるので、
   条件から作ったハッシュ（condition_key）を各行に持たせ、
   比較していい相手かどうかを機械的に判定できるようにする。

2. 数字だけでなく、回答本文・出典・根拠まで残す。
   数字だけ残しても「なぜその数字になったか」が後から分からないため。

外部ライブラリは使わない。標準の sqlite3 で足りる規模であり、
ORM を挟むとテーブルの実体が見えにくくなるため。
"""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from llmo.core.brand import Brand
from llmo.config import EVAL, MODELS, STORAGE
from llmo.corpus.fetch import load_meta
from llmo.generation import rules
from llmo.pipeline import EvaluationResult, ImprovementResult
from llmo.generation.writer import Article


# ---------------------------------------------------------------------------
# スキーマ
# ---------------------------------------------------------------------------

# ON DELETE CASCADE を付けているのは、評価 1 件を消したときに
# ぶら下がる回答・出典・根拠が孤児として残らないようにするため。
_SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluations (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at         TEXT    NOT NULL,

    -- 何を評価したか
    topic              TEXT    NOT NULL,

    -- 入力のパターン。'A' = トピックのみ（比較なし）、'B' = 書き直し前後の比較、
    -- 'C' = 改善ラウンドの再評価（条件は 'single' 1 つ。改善前は improvements 経由で辿る）。
    pattern            TEXT    NOT NULL,

    -- このシステムの出力にあたる条件の名前。
    -- パターン A と C なら 'single'、B なら 'after'。
    -- 一覧で「出力側の指標」を引くときに、この列で結合する。
    output_condition   TEXT    NOT NULL,

    -- 出力にあたる記事（A: 作成した記事 / B: 書き直した後の記事）
    article_style      TEXT    NOT NULL,
    article_title      TEXT    NOT NULL,
    article_body       TEXT    NOT NULL,

    -- 書き直し前の記事。パターン B のときだけ入る。
    original_title     TEXT,
    original_body      TEXT,

    note               TEXT    NOT NULL DEFAULT '',

    -- どの疑似ウェブで測ったか
    corpus_name        TEXT    NOT NULL,
    num_corpus_docs    INTEGER NOT NULL,

    -- 誰の記事か
    brand_company      TEXT    NOT NULL,
    brand_product      TEXT    NOT NULL,

    -- 実験条件のスナップショット
    model_generation   TEXT    NOT NULL,
    model_judge        TEXT    NOT NULL,
    model_embedding    TEXT    NOT NULL,
    num_queries        INTEGER NOT NULL,

    -- 各条件を何回まわして平均したか。
    -- 1 回だけの結果と 3 回平均の結果では数字の安定度が違うため、
    -- 比較可能かどうかの判定に含める。
    num_runs           INTEGER NOT NULL,

    top_k              INTEGER NOT NULL,

    -- 回答生成のとき、1 出典あたり LLM に渡した本文の最大文字数。
    -- 渡す本文の量が変われば、引用のされ方も変わる。
    -- チャンクだけを渡していた頃の結果と混ざらないように条件に含める。
    max_source_chars   INTEGER NOT NULL,

    chunk_size         INTEGER NOT NULL,
    chunk_overlap      INTEGER NOT NULL,
    min_content_chars  INTEGER NOT NULL,

    -- 上の条件と疑似ウェブの中身から作るハッシュ。
    -- 同じ値の行同士だけが比較可能。
    condition_key      TEXT    NOT NULL,

    -- 検索クエリを言い換えた「読者の状況」（JSON）。
    -- 記事にはクエリではなくこちらを渡しているため、
    -- どの課題記述から記事を作ったのかを残さないと再現できない。
    audience_issues    TEXT    NOT NULL DEFAULT '',

    -- 生成後の転記チェックの結果（JSON）。
    -- 見出しにクエリが写っていた評価は、検索順位が記事の出来ではなく
    -- 文字列の一致で決まっている可能性があるため、数字と一緒に残す。
    leakage_report     TEXT    NOT NULL DEFAULT '',

    -- 記事を書かせたときの LLMO 最適化の指示（画面から編集できる）。
    -- condition_key には含めない。この指示は測る道具ではなく測られる側であり、
    -- 指示を変えた前後を比べること自体が目的だからである。
    -- ただしどの指示で書かれた記事かは後から追えないと困るので、文面を残す。
    llmo_rules         TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
    evaluation_id           INTEGER NOT NULL
                            REFERENCES evaluations(id) ON DELETE CASCADE,
    -- 'single'（パターン A）または 'before' / 'after'（パターン B）
    condition               TEXT    NOT NULL,

    mention_rate            REAL,
    citation_rate           REAL,
    top_recommendation_rate REAL,
    avg_citation_position   REAL,
    avg_grounding_rate      REAL,

    retrieval_rate          REAL,
    avg_search_rank         REAL,
    citation_share          REAL,

    PRIMARY KEY (evaluation_id, condition)
);

CREATE TABLE IF NOT EXISTS answers (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id     INTEGER NOT NULL
                      REFERENCES evaluations(id) ON DELETE CASCADE,
    query_index       INTEGER NOT NULL,
    query             TEXT    NOT NULL,
    condition         TEXT    NOT NULL,
    text              TEXT    NOT NULL,

    mentioned         INTEGER NOT NULL,
    cited             INTEGER NOT NULL,
    citation_position INTEGER,
    citation_count    INTEGER NOT NULL,
    total_citations   INTEGER NOT NULL,
    search_rank       INTEGER,

    is_comparison     INTEGER NOT NULL,
    is_top            INTEGER NOT NULL,
    top_vendor        TEXT,
    judge_reason      TEXT    NOT NULL DEFAULT '',

    total_claims      INTEGER,
    supported_claims  INTEGER,

    UNIQUE (evaluation_id, query_index, condition)
);

CREATE TABLE IF NOT EXISTS sources (
    answer_id  INTEGER NOT NULL REFERENCES answers(id) ON DELETE CASCADE,
    number     INTEGER NOT NULL,
    url        TEXT    NOT NULL,
    title      TEXT    NOT NULL,
    is_target  INTEGER NOT NULL,
    PRIMARY KEY (answer_id, number)
);

CREATE TABLE IF NOT EXISTS grounding_claims (
    answer_id   INTEGER NOT NULL REFERENCES answers(id) ON DELETE CASCADE,
    claim_index INTEGER NOT NULL,
    claim       TEXT    NOT NULL,
    evidence    TEXT    NOT NULL,
    PRIMARY KEY (answer_id, claim_index)
);

-- 改善ラウンドの記録。
--
-- 改善後の評価そのものは evaluations に普通の 1 行として入る（pattern = 'C'）。
-- このテーブルが持つのは「どの評価をどう直して、どの評価になったか」という
-- 2 つの評価の間の関係と、その途中で下した判断である。
--
-- 判断を数字と別に残すのは、指標が動いた理由を後から辿るため。
-- 「引用率が上がった」だけでは、どの節をどう直したから上がったのかが分からない。
CREATE TABLE IF NOT EXISTS improvements (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT    NOT NULL,

    -- 何回目の改善か（1 始まり）。同じ記事を続けて直すと増えていく。
    round               INTEGER NOT NULL,

    -- 改善前の評価。ここから弱いクエリを選んだ。
    base_evaluation_id  INTEGER NOT NULL
                        REFERENCES evaluations(id) ON DELETE CASCADE,

    -- 改善後の評価。
    after_evaluation_id INTEGER NOT NULL
                        REFERENCES evaluations(id) ON DELETE CASCADE,

    -- compare.composite_score() の値。採否はこの 2 つの比較で決めている。
    score_before        REAL    NOT NULL,
    score_after         REAL    NOT NULL,

    -- 修正版を採用してよいと判定したか。
    -- 悪化した版も行としては残す。「直したが駄目だった」ことも記録である。
    adopted             INTEGER NOT NULL,

    -- 途中の判断（いずれも JSON）。
    weak_queries        TEXT    NOT NULL DEFAULT '',
    analyses            TEXT    NOT NULL DEFAULT '',
    section_edits       TEXT    NOT NULL DEFAULT '',
    fact_warnings       TEXT    NOT NULL DEFAULT '',

    -- 1 つの評価から改善を 2 回走らせることはできる（やり直し）が、
    -- 同じ組み合わせが二重に入ることはない。
    UNIQUE (base_evaluation_id, after_evaluation_id)
);

-- 一覧画面は「このトピックの新しい順」で引くので、その形に合わせる。
CREATE INDEX IF NOT EXISTS idx_evaluations_topic
    ON evaluations (topic, created_at DESC);

-- 評価を開いたときに「この評価から改善したか」「改善で生まれた評価か」を
-- 両方向から引くので、両方の列に索引を付ける。
CREATE INDEX IF NOT EXISTS idx_improvements_base
    ON improvements (base_evaluation_id);
CREATE INDEX IF NOT EXISTS idx_improvements_after
    ON improvements (after_evaluation_id);
"""


# ---------------------------------------------------------------------------
# 接続
# ---------------------------------------------------------------------------

@contextmanager
def connect(db_path: str | None = None):
    """DB に接続する。

    foreign_keys を明示的に有効にしている。
    SQLite は既定で外部キー制約を無視するため、
    ON DELETE CASCADE も指定しないと効かない。

    with 文を抜けるときにコミットし、例外が出ればロールバックする。
    評価 1 回分は複数テーブルにまたがって書くので、
    途中で落ちて中途半端な行が残ると、それが混ざった集計になってしまう。
    """
    path = Path(db_path or STORAGE.db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


# 後から足したカラム。テーブル名ごとに「カラム名 → 定義」で持つ。
#
# _SCHEMA は CREATE TABLE IF NOT EXISTS なので、
# すでに DB があるとカラムを足しても作り直されない。
# 気づくのは保存の時点、つまり数分かけた評価が終わった後になり、
# その回の結果が保存できずに消える。
# 起動のたびに不足を補って、そこで落ちないようにする。
#
# 既存の行に入る既定値は「その頃は記録していなかった」ことを表す。
# 実際にその値で測ったという意味ではない。
_ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "evaluations": {
        # 0 は「出典としてチャンクを渡していた頃の行」を意味する。
        "max_source_chars": "INTEGER NOT NULL DEFAULT 0",
        # 空文字は「どの指示で書かれたか記録が無い」ことを意味する。
        "llmo_rules": "TEXT NOT NULL DEFAULT ''",
        # 空文字は「クエリを言い換えずに記事を作っていた頃の行」を意味する。
        "audience_issues": "TEXT NOT NULL DEFAULT ''",
        # 空文字は「転記チェックを入れる前の行」を意味する。
        "leakage_report": "TEXT NOT NULL DEFAULT ''",
    },
}


def _add_missing_columns(connection: sqlite3.Connection) -> None:
    """既存の DB に足りないカラムを補う。

    SQLite の ALTER TABLE ADD COLUMN は既存の行を書き換えずに済むため、
    貯めた評価結果を捨てずにスキーマだけ追いつかせられる。
    """
    for table, columns in _ADDED_COLUMNS.items():
        existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        if not existing:
            # テーブル自体がまだ無い場合は _SCHEMA が作るので、ここでは触らない。
            continue
        for name, definition in columns.items():
            if name not in existing:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def init_db(db_path: str | None = None) -> None:
    """テーブルを作る。すでにあれば足りないカラムだけを補う。

    保存のたびに呼んでよい。IF NOT EXISTS なので繰り返し実行しても安全。
    """
    with connect(db_path) as connection:
        connection.executescript(_SCHEMA)
        _add_missing_columns(connection)


# ---------------------------------------------------------------------------
# 条件ハッシュ
# ---------------------------------------------------------------------------

def corpus_fingerprint(corpus_name: str) -> str:
    """疑似ウェブの中身を表す短いハッシュを返す。

    URL の一覧だけから作る。本文まで含めないのは、
    同じページを取り直したときに広告や日付で本文がわずかに変わることがあり、
    それを「別の疑似ウェブ」と見なすと比較できる相手がいなくなるため。
    検索対象として何が入っているかは URL の集合で決まる。

    Returns:
        16 桁のハッシュ。コーパスが無ければ "missing"。
    """
    path = Path(STORAGE.corpus_dir) / f"{corpus_name}.json"
    if not path.exists():
        return "missing"

    documents = json.loads(path.read_text(encoding="utf-8"))
    # 取得順は毎回変わりうるので、並べ替えてから固める。
    urls = sorted(doc["url"] for doc in documents)
    digest = hashlib.sha256("\n".join(urls).encode("utf-8")).hexdigest()
    return digest[:16]


def condition_key(corpus_name: str, num_queries: int | None = None) -> str:
    """実験条件を表す短いハッシュを返す。

    ここに含めるのは「値が変わると指標が動くもの」だけ。
    2 つの評価結果を比べてよいかは、この値が一致するかで判定する。

    Args:
        corpus_name: コーパスの保存名。
        num_queries: 実際に評価に使ったクエリの本数。
            疑似ウェブを作るときにクエリを取捨選択できるため、
            設定値（EVAL.num_queries）と一致するとは限らない。
            割合系の指標はこの本数が分母なので、実測値で区別する。
            省略時は保存済みコーパスのクエリ本数、それも無ければ設定値を使う。
    """
    if num_queries is None:
        meta = load_meta(corpus_name)
        num_queries = len(meta["queries"]) if meta else EVAL.num_queries

    parts = [
        MODELS.generation,
        MODELS.judge,
        MODELS.embedding,
        str(num_queries),
        str(EVAL.num_runs),
        str(EVAL.top_k),
        str(EVAL.max_source_chars),
        str(EVAL.chunk_size),
        str(EVAL.chunk_overlap),
        str(EVAL.min_content_chars),
        corpus_name,
        corpus_fingerprint(corpus_name),
    ]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


# ---------------------------------------------------------------------------
# 保存
# ---------------------------------------------------------------------------


@dataclass
class _OutputArticles:
    """保存する記事を、パターンによらず同じ形にまとめたもの。

    パターン A は記事が 1 本、B は 2 本（書き直し前と後）ある。
    保存のたびに if で分けると読みにくいので、ここで吸収する。
    """

    condition: str          # 出力にあたる条件の名前
    article: Article        # 出力にあたる記事
    original_title: str | None
    original_body: str | None


def _output_articles(result: EvaluationResult) -> _OutputArticles:
    """評価結果から、出力にあたる記事と書き直し前の記事を取り出す。"""
    # 条件は表示したい順に並んでいて、最後が出力にあたる
    # （A なら "single"、B なら "after"）。
    condition = result.condition_names[-1]
    original = result.articles.get("before") if result.pattern == "B" else None

    return _OutputArticles(
        condition=condition,
        article=result.articles[condition],
        original_title=original.title if original else None,
        original_body=original.body if original else None,
    )


def save_evaluation(
    result: EvaluationResult,
    brand: Brand,
    corpus_name: str,
    note: str = "",
    issues: list[dict] | None = None,
    leakage: dict | None = None,
    db_path: str | None = None,
) -> int:
    """評価 1 回分を保存する。

    テーブルをまたぐ書き込みを 1 つのトランザクションで行う。
    途中で落ちた行が残ると、それが混ざった集計になってしまうため。

    Args:
        result: 評価結果。
        brand: 自社の情報。
        corpus_name: 使った疑似ウェブの名前。条件ハッシュの材料にもなる。
        note: 「製品名の指示を追加した版」のような手書きのメモ。
            後で一覧を見たときに、どの版だったかを思い出すためのもの。
        issues: 検索クエリを言い換えた読者の状況。
            記事にはクエリではなくこれを渡しているため、
            どの課題記述から記事を作ったのかを残す。
        leakage: 生成後の転記チェックの結果（LeakageReport.to_dict()）。
            見出しにクエリが写っていた評価は、検索順位が記事の出来ではなく
            文字列の一致で決まっている可能性がある。

    Returns:
        保存した評価の id。
    """
    init_db(db_path)
    output = _output_articles(result)

    with connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO evaluations (
                created_at, topic, pattern, output_condition,
                article_style, article_title, article_body,
                original_title, original_body, note,
                corpus_name, num_corpus_docs, brand_company, brand_product,
                model_generation, model_judge, model_embedding,
                num_queries, num_runs, top_k, max_source_chars,
                chunk_size, chunk_overlap, min_content_chars, condition_key,
                llmo_rules, audience_issues, leakage_report
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                result.topic,
                result.pattern,
                output.condition,
                output.article.style,
                output.article.title,
                output.article.body,
                output.original_title,
                output.original_body,
                note,
                corpus_name,
                result.num_corpus_docs,
                brand.company,
                brand.product,
                MODELS.generation,
                MODELS.judge,
                MODELS.embedding,
                len(result.queries),
                result.num_runs,
                EVAL.top_k,
                EVAL.max_source_chars,
                EVAL.chunk_size,
                EVAL.chunk_overlap,
                EVAL.min_content_chars,
                condition_key(corpus_name, len(result.queries)),
                rules.load(),
                json.dumps(issues, ensure_ascii=False) if issues else "",
                json.dumps(leakage, ensure_ascii=False) if leakage else "",
            ),
        )
        evaluation_id = cursor.lastrowid

        # --- 集計値 -------------------------------------------------------
        for name, aggregated in result.aggregated.items():
            connection.execute(
                """
                INSERT INTO metrics (
                    evaluation_id, condition,
                    mention_rate, citation_rate, top_recommendation_rate,
                    avg_citation_position, avg_grounding_rate,
                    retrieval_rate, avg_search_rank, citation_share
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    evaluation_id,
                    name,
                    aggregated.mention_rate,
                    aggregated.citation_rate,
                    result.top_recommendation.get(name),
                    aggregated.avg_citation_position,
                    # 根拠判定は全条件で行う。
                    # その条件で記事が一度も出典に入らなければ None になる。
                    result.avg_grounding_rate.get(name),
                    aggregated.retrieval_rate,
                    aggregated.avg_search_rank,
                    aggregated.citation_share,
                ),
            )

        # --- クエリごとの回答 ---------------------------------------------
        for query_index, query_result in enumerate(result.per_query, start=1):
            for name, answer in query_result.answers.items():
                metrics = query_result.metrics[name]
                recommendation = query_result.recommendation.get(name)
                grounding = query_result.grounding.get(name)

                cursor = connection.execute(
                    """
                    INSERT INTO answers (
                        evaluation_id, query_index, query, condition, text,
                        mentioned, cited, citation_position,
                        citation_count, total_citations, search_rank,
                        is_comparison, is_top, top_vendor, judge_reason,
                        total_claims, supported_claims
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        evaluation_id,
                        query_index,
                        query_result.query,
                        name,
                        answer.text,
                        int(metrics.mentioned),
                        int(metrics.cited),
                        metrics.citation_position,
                        metrics.citation_count,
                        metrics.total_citations,
                        metrics.search_rank,
                        int(recommendation.is_comparison) if recommendation else 0,
                        int(recommendation.is_top) if recommendation else 0,
                        recommendation.top_vendor if recommendation else None,
                        recommendation.reason if recommendation else "",
                        grounding.total_claims if grounding else None,
                        grounding.supported_claims if grounding else None,
                    ),
                )
                answer_id = cursor.lastrowid

                # 出典を残すと、自社記事が拾われなかったときに
                # 代わりに誰が入っていたかが後から分かる。
                connection.executemany(
                    """
                    INSERT INTO sources (answer_id, number, url, title, is_target)
                    VALUES (?,?,?,?,?)
                    """,
                    [
                        (answer_id, s.number, s.url, s.title, int(s.is_target))
                        for s in answer.sources
                    ],
                )

                # 自社記事が支えた主張。記事のどの記述が効いたかが貯まる。
                if grounding:
                    connection.executemany(
                        """
                        INSERT INTO grounding_claims
                            (answer_id, claim_index, claim, evidence)
                        VALUES (?,?,?,?)
                        """,
                        [
                            (answer_id, i, claim, evidence)
                            for i, (claim, evidence) in enumerate(
                                grounding.supported, start=1
                            )
                        ],
                    )

    return evaluation_id


def save_improvement(
    improvement: ImprovementResult,
    base_evaluation_id: int,
    after_evaluation_id: int,
    db_path: str | None = None,
) -> int:
    """改善 1 ラウンド分の記録を保存する。

    改善後の評価そのものは、先に save_evaluation() で普通の 1 行として
    保存しておくこと。ここが保存するのは 2 つの評価の関係と、
    その途中で下した判断だけである。

    採用しなかったラウンドも保存する。
    「直したが良くならなかった」ことも、次に何を試すかを決める材料になるため。

    Args:
        improvement: pipeline.run_improvement() の結果。
        base_evaluation_id: 改善前の評価の id。
        after_evaluation_id: 改善後の評価の id。

    Returns:
        保存した改善の id。

    Raises:
        ValueError: 改善対象が 1 本も無く、実際には何もしていないとき。
    """
    if improvement.comparison is None:
        raise ValueError("改善が行われていないため保存できません")

    init_db(db_path)

    def dump(items) -> str:
        return json.dumps([i.to_dict() for i in items], ensure_ascii=False)

    with connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO improvements (
                created_at, round, base_evaluation_id, after_evaluation_id,
                score_before, score_after, adopted,
                weak_queries, analyses, section_edits, fact_warnings
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                improvement.round,
                base_evaluation_id,
                after_evaluation_id,
                improvement.comparison.score_before,
                improvement.comparison.score_after,
                int(improvement.comparison.adopted),
                dump(improvement.weak_queries),
                dump(improvement.analyses),
                dump(improvement.edits),
                dump(improvement.fact_warnings),
            ),
        )
        return cursor.lastrowid


# ---------------------------------------------------------------------------
# 読み出し
# ---------------------------------------------------------------------------

@dataclass
class EvaluationSummary:
    """一覧に出す 1 行。

    指標は出力側の条件のものだけを持つ
    （パターン A なら "single"、B なら "after"）。
    一覧で見たいのは「そのとき出した記事がどうだったか」の推移だからである。
    書き直し前との差を見たいときは、その評価を開いて確認する。
    """

    id: int
    created_at: str
    topic: str
    pattern: str
    article_style: str
    article_title: str
    note: str
    condition_key: str

    # README の評価基準と同じ順（検索 → 回答）で並べる。
    retrieval_rate: float | None
    avg_search_rank: float | None
    mention_rate: float | None
    citation_rate: float | None
    citation_share: float | None
    top_recommendation_rate: float | None
    avg_grounding_rate: float | None

    # 画面には出さないが、記録として保持する。
    avg_citation_position: float | None


def list_evaluations(
    topic: str | None = None,
    limit: int = 50,
    db_path: str | None = None,
) -> list[EvaluationSummary]:
    """保存済みの評価を新しい順に返す。

    condition_key をそのまま返している。
    呼び出し側が現在の条件と突き合わせて、
    比較してよい行かどうかを判定できるようにするため。

    Args:
        topic: 指定するとそのトピックだけに絞る。
        limit: 返す件数の上限。
    """
    init_db(db_path)

    # after の集計値だけを横に並べたいので、metrics を条件付きで結合する。
    sql = """
        SELECT e.id, e.created_at, e.topic, e.pattern, e.article_style,
               e.article_title, e.note, e.condition_key,
               m.retrieval_rate, m.avg_search_rank,
               m.mention_rate, m.citation_rate, m.citation_share,
               m.top_recommendation_rate, m.avg_grounding_rate,
               m.avg_citation_position
        FROM evaluations e
        LEFT JOIN metrics m
               ON m.evaluation_id = e.id AND m.condition = e.output_condition
    """
    params: list = []
    if topic:
        sql += " WHERE e.topic = ?"
        params.append(topic)
    sql += " ORDER BY e.created_at DESC, e.id DESC LIMIT ?"
    params.append(limit)

    with connect(db_path) as connection:
        rows = connection.execute(sql, params).fetchall()

    return [EvaluationSummary(**dict(row)) for row in rows]


def load_evaluation(
    evaluation_id: int,
    db_path: str | None = None,
) -> dict | None:
    """評価 1 件を、画面に出せる形で丸ごと読み出す。

    dataclass ではなく辞書で返している。
    ここで返すのは表示のためのデータで、
    EvaluationResult（実行中の型）とは役割が違うため、
    無理に同じ型に戻さない。

    Returns:
        評価の内容。見つからなければ None。
    """
    init_db(db_path)

    with connect(db_path) as connection:
        evaluation = connection.execute(
            "SELECT * FROM evaluations WHERE id = ?", (evaluation_id,)
        ).fetchone()
        if evaluation is None:
            return None

        metrics = {
            row["condition"]: dict(row)
            for row in connection.execute(
                "SELECT * FROM metrics WHERE evaluation_id = ?", (evaluation_id,)
            )
        }

        answers = []
        for row in connection.execute(
            """
            SELECT * FROM answers
            WHERE evaluation_id = ?
            ORDER BY query_index, condition DESC
            """,
            (evaluation_id,),
        ):
            answer = dict(row)
            answer["sources"] = [
                dict(s)
                for s in connection.execute(
                    "SELECT * FROM sources WHERE answer_id = ? ORDER BY number",
                    (row["id"],),
                )
            ]
            answer["claims"] = [
                dict(c)
                for c in connection.execute(
                    """
                    SELECT * FROM grounding_claims
                    WHERE answer_id = ? ORDER BY claim_index
                    """,
                    (row["id"],),
                )
            ]
            answers.append(answer)

    return {
        "evaluation": dict(evaluation),
        "metrics": metrics,
        "answers": answers,
    }


def delete_evaluation(evaluation_id: int, db_path: str | None = None) -> None:
    """評価 1 件を消す。

    ぶら下がる回答・出典・根拠は ON DELETE CASCADE で一緒に消える
    （connect() で foreign_keys を有効にしているため）。
    """
    init_db(db_path)
    with connect(db_path) as connection:
        connection.execute("DELETE FROM evaluations WHERE id = ?", (evaluation_id,))


def _decode_improvement(row: sqlite3.Row) -> dict:
    """improvements の 1 行を、画面に出せる形に開く。

    JSON で入れている列をその場で辞書に戻す。
    表示側で毎回 json.loads を書くと、列を増やしたときに書き漏らすため。
    """
    improvement = dict(row)
    for key in ("weak_queries", "analyses", "section_edits", "fact_warnings"):
        raw = improvement.get(key)
        improvement[key] = json.loads(raw) if raw else []
    improvement["adopted"] = bool(improvement["adopted"])
    return improvement


def load_improvement_of(
    after_evaluation_id: int,
    db_path: str | None = None,
) -> dict | None:
    """その評価が「改善で生まれたもの」なら、改善の記録を返す。

    評価を開いたときに、改善前と比べた表を出すために使う。

    Returns:
        改善の記録。改善で生まれた評価でなければ None。
    """
    init_db(db_path)
    with connect(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM improvements WHERE after_evaluation_id = ?",
            (after_evaluation_id,),
        ).fetchone()

    return _decode_improvement(row) if row else None


def list_improvements_from(
    base_evaluation_id: int,
    db_path: str | None = None,
) -> list[dict]:
    """その評価を出発点にした改善の一覧を、新しい順に返す。

    同じ評価から複数回やり直せるので、リストで返す。

    Returns:
        改善の記録の一覧。1 度も改善していなければ空リスト。
    """
    init_db(db_path)
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT * FROM improvements
            WHERE base_evaluation_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (base_evaluation_id,),
        ).fetchall()

    return [_decode_improvement(row) for row in rows]


def next_round_number(base_evaluation_id: int, db_path: str | None = None) -> int:
    """次に行う改善が何回目にあたるかを返す。

    その評価自体が改善で生まれたものなら、その回数に 1 を足す。
    改善を重ねたときに「改善 3 回目」と数え続けられるようにするため。
    """
    previous = load_improvement_of(base_evaluation_id, db_path)
    return (previous["round"] + 1) if previous else 1

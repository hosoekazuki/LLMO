"""改善の前後を比べて、修正版を採用してよいかを判定するモジュール。

記事を直せば必ず良くなるとは限らない。
ある節を厚くした結果、別のクエリで拾われなくなることは普通に起こる。
そのため修正版を無条件に採用せず、総合的に良くなった場合だけ採用する。

ここでも API は呼ばない。測り終えた値を比べるだけである。

## 比較の相手をどこから取るか

改善前の値は 2 通りの経路で入ってくる。

- 実行中の評価結果（pipeline.EvaluationResult）
- 保存済みの評価（db.load_evaluation() が返す辞書）

どちらも同じ数字なので、いったん MetricSnapshot に揃えてから比べる。
経路ごとに比較の処理を書くと、片方を直したときにもう片方がずれる。
"""

from dataclasses import dataclass

from llmo.evaluation.metrics import QueryOutcome

# 総合スコアの重み。合計 1.0 になるようにしている。
#
# 記事がたどる漏斗（拾われる → 引用される → 名前が出る）のうち、
# 前半ほど重くする。拾われなければ後ろの段階は起こりようがないためである。
#
# ここに入れていない指標が 2 つある。理由は分母にある。
#
# - avg_search_rank / avg_citation_position:
#   分母が「拾われたクエリのみ」なので、1 本しか拾われない記事ほど
#   平均順位が良く見える。総合スコアに入れると
#   「拾われる本数を減らすほど有利」という逆向きの誘因が生まれる。
# - top_recommendation_rate:
#   分母が「比較形式になった回答の本数」で、実行のたびに変動し、
#   0 本なら None になる。前後で分母が違う値を足し合わせても意味を持たない。
#
# どちらも比較表には出す。合成に入れないだけである。
WEIGHTS = {
    "retrieval_rate": 0.30,
    "citation_rate": 0.30,
    "citation_share": 0.15,
    "mention_rate": 0.15,
    "avg_grounding_rate": 0.10,
}


@dataclass
class MetricSnapshot:
    """1 つの条件の集計値を、経路によらない形にまとめたもの。

    値が None なのは「測れなかった」ことを意味する
    （一度も引用されなければ根拠寄与率は出ない、など）。
    """

    retrieval_rate: float | None
    avg_search_rank: float | None
    mention_rate: float | None
    citation_rate: float | None
    citation_share: float | None
    top_recommendation_rate: float | None
    avg_grounding_rate: float | None

    def to_dict(self) -> dict:
        return {
            "retrieval_rate": self.retrieval_rate,
            "avg_search_rank": self.avg_search_rank,
            "mention_rate": self.mention_rate,
            "citation_rate": self.citation_rate,
            "citation_share": self.citation_share,
            "top_recommendation_rate": self.top_recommendation_rate,
            "avg_grounding_rate": self.avg_grounding_rate,
        }


@dataclass
class QueryComparison:
    """クエリ 1 本の、改善前と改善後。"""

    query: str
    before: QueryOutcome | None
    after: QueryOutcome | None

    @property
    def changed(self) -> bool:
        """検索・引用のどちらかが変わったか。画面で目立たせるのに使う。"""
        if self.before is None or self.after is None:
            return False
        return (
            self.before.search_rank != self.after.search_rank
            or self.before.cited != self.after.cited
        )


@dataclass
class Comparison:
    """改善前後の比較結果。"""

    before: MetricSnapshot
    after: MetricSnapshot

    score_before: float
    score_after: float

    # 修正版を採用してよいか。総合スコアが上がったときだけ True。
    adopted: bool

    per_query: list[QueryComparison]

    @property
    def score_delta(self) -> float:
        return self.score_after - self.score_before

    def deltas(self) -> dict[str, float | None]:
        """指標ごとの増減。どちらかが None の指標は None を返す。"""
        result: dict[str, float | None] = {}
        for key in self.before.to_dict():
            before = getattr(self.before, key)
            after = getattr(self.after, key)
            result[key] = None if before is None or after is None else after - before
        return result

    def to_dict(self) -> dict:
        return {
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "score_before": self.score_before,
            "score_after": self.score_after,
            "adopted": self.adopted,
            "deltas": self.deltas(),
        }


def composite_score(snapshot: MetricSnapshot) -> float:
    """総合スコアを 0.0〜1.0 で返す。

    採否の判定に使う唯一の数値。
    指標ごとに上がり下がりが混ざるのが普通なので、
    どれか 1 つを見て決めると「引用率は上がったが検索から消えた」ような
    修正を採用してしまう。

    None は 0 として扱う。
    「測れなかった」のは、その段階まで到達しなかったということであり、
    到達した記事より低く評価するのが妥当なためである。
    """
    total = 0.0
    for key, weight in WEIGHTS.items():
        value = getattr(snapshot, key)
        total += weight * (value if value is not None else 0.0)
    return total


def snapshot_from_result(result, condition: str) -> MetricSnapshot:
    """実行中の評価結果から集計値を取り出す。

    Args:
        result: pipeline.EvaluationResult。
            型注釈を付けないのは循環 import を避けるため。
        condition: 条件の名前（"single" / "after" など）。
    """
    aggregated = result.aggregated[condition]
    return MetricSnapshot(
        retrieval_rate=aggregated.retrieval_rate,
        avg_search_rank=aggregated.avg_search_rank,
        mention_rate=aggregated.mention_rate,
        citation_rate=aggregated.citation_rate,
        citation_share=aggregated.citation_share,
        top_recommendation_rate=result.top_recommendation.get(condition),
        avg_grounding_rate=result.avg_grounding_rate.get(condition),
    )


def snapshot_from_row(row: dict) -> MetricSnapshot:
    """保存済みの metrics テーブルの 1 行から集計値を取り出す。

    列名は MetricSnapshot の項目名と揃えてあるので、そのまま引ける。
    """
    return MetricSnapshot(
        retrieval_rate=row.get("retrieval_rate"),
        avg_search_rank=row.get("avg_search_rank"),
        mention_rate=row.get("mention_rate"),
        citation_rate=row.get("citation_rate"),
        citation_share=row.get("citation_share"),
        top_recommendation_rate=row.get("top_recommendation_rate"),
        avg_grounding_rate=row.get("avg_grounding_rate"),
    )


def outcomes_from_result(result, condition: str) -> list[QueryOutcome]:
    """実行中の評価結果から、クエリごとの結果を取り出す。"""
    outcomes = []
    for query_index, query_result in enumerate(result.per_query, start=1):
        metrics = query_result.metrics.get(condition)
        if metrics is None:
            continue
        grounding = query_result.grounding.get(condition)
        outcomes.append(
            QueryOutcome(
                query_index=query_index,
                query=query_result.query,
                search_rank=metrics.search_rank,
                cited=metrics.cited,
                mentioned=metrics.mentioned,
                grounding_rate=grounding.grounding_rate if grounding else None,
            )
        )
    return outcomes


def outcomes_from_rows(rows: list[dict], condition: str) -> list[QueryOutcome]:
    """保存済みの answers テーブルの行から、クエリごとの結果を取り出す。

    Args:
        rows: db.load_evaluation() が返す "answers" のリスト。
        condition: 取り出す条件の名前。
    """
    outcomes = []
    for row in rows:
        if row["condition"] != condition:
            continue

        # 根拠寄与率は保存時に分子と分母で持っている（率では持っていない）。
        # 引用されなかった回答は判定自体を行わないので None のままになる。
        total = row.get("total_claims")
        supported = row.get("supported_claims")
        rate = supported / total if total else None

        outcomes.append(
            QueryOutcome(
                query_index=row["query_index"],
                query=row["query"],
                search_rank=row["search_rank"],
                cited=bool(row["cited"]),
                mentioned=bool(row["mentioned"]),
                grounding_rate=rate,
            )
        )
    return outcomes


def compare(
    before: MetricSnapshot,
    after: MetricSnapshot,
    before_queries: list[QueryOutcome],
    after_queries: list[QueryOutcome],
) -> Comparison:
    """改善前後を比べ、修正版を採用してよいかを判定する。

    クエリは番号ではなく文字列で突き合わせる。
    改善の前後で同じクエリを使うのが前提だが、
    番号で結ぶと並びがずれたときに気づかないまま別のクエリと比べてしまう。

    Returns:
        比較結果。adopted が True のときだけ修正版を採用してよい。
    """
    score_before = composite_score(before)
    score_after = composite_score(after)

    by_query_before = {o.query: o for o in before_queries}
    by_query_after = {o.query: o for o in after_queries}

    # 並び順は改善前の順を保つ。画面で前回と同じ順に読めるようにするため。
    ordered = [o.query for o in before_queries]
    ordered += [q for q in by_query_after if q not in by_query_before]

    per_query = [
        QueryComparison(
            query=query,
            before=by_query_before.get(query),
            after=by_query_after.get(query),
        )
        for query in ordered
    ]

    return Comparison(
        before=before,
        after=after,
        score_before=score_before,
        score_after=score_after,
        # 同点は採用しない。API を使った以上わずかでも上がっているべきで、
        # 同じなら元の記事を保つほうが安全である。
        adopted=score_after > score_before,
        per_query=per_query,
    )

"""生成された回答を定量的に評価するモジュール。

README の「評価に使用する基準」を実装する。

このファイルでは、文字列処理だけで機械的に出せる 3 指標を扱う。
判断を伴う 2 指標（Top Recommendation Rate と Answer Grounding）は
LLM による判定が必要なため、別に用意する。

すべての指標は条件ごとに計算する。

- パターン A は条件が 1 つ（"single"）なので、その記事単体の水準を示す。
- パターン B は条件が 2 つ（"before" / "after"）で、その差を見る。
  ベースコーパスもクエリも共通で、差し込んだ記事が書き直し前か後かだけが違うため、
  差が出たらそれは書き直しによる差である。
"""

from dataclasses import dataclass

from llmo.retrieval.answer import Answer
from llmo.core.brand import Brand


@dataclass
class QueryMetrics:
    """クエリ 1 本ぶんの評価結果。

    割合（Rate）は複数クエリを集計して初めて出るので、
    ここでは 1 本ごとの生の観測値を持つ。
    """

    query: str
    condition: str

    # 回答本文に自社の名前が出たか。Mention Rate の materials。
    mentioned: bool

    # 自社記事が出典として引用されたか。Citation Rate の材料。
    cited: bool

    # 回答の中で、自社記事の引用が何番目に登場したか（1 始まり）。
    # 引用されていなければ None。Citation Position の材料。
    #
    # 「回答の上部に表示されているか」を見る指標なので、
    # 出典リストの並び順ではなく、回答本文での登場順で数える。
    citation_position: int | None

    # 自社記事が引用された回数。回答のうちどれだけを占めたかを見る。
    citation_count: int

    # 回答中の引用の総数。上の割合を出すための分母。
    total_citations: int

    # 検索で自社記事が何位だったか（1 始まり）。圏外なら None。
    #
    # これは README の評価基準には無いが、必要な指標として足している。
    # 記事がそもそも検索に拾われなければ引用されるはずもないので、
    # 「拾われなかった」のか「拾われたが引用されなかった」のかを
    # 区別できないと、改善の方向が決まらない。
    search_rank: int | None


@dataclass
class QueryOutcome:
    """クエリ 1 本の観測結果を、経路によらない形にまとめたもの。

    QueryMetrics との違いは出どころにある。
    QueryMetrics は評価を実行した直後にしか存在しないが、こちらは
    保存済みの DB からも組み立てられる形にしてある（根拠寄与率も含む）。

    改善ループは「実行直後の結果」からも「保存済みの評価」からも始められる。
    経路ごとに別の型を持つと、弱点の判定と前後の比較を二重に書くことになる。
    組み立ては evaluation/compare.py の outcomes_from_* が行う。
    """

    # 何本目のクエリか（1 始まり）。
    query_index: int

    query: str

    # 検索での順位。圏外なら None。
    search_rank: int | None

    cited: bool
    mentioned: bool

    # 回答の主張のうち自社記事が支えた割合。
    # 引用されなかった回答は判定自体を行わないので None になる。
    grounding_rate: float | None


@dataclass
class AggregatedMetrics:
    """1 つの条件の集計結果。

    条件の名前は "single"（パターン A）、
    または "before" / "after"（パターン B）。
    """

    condition: str

    # 集計に使った観測値の数。
    # 試行を複数回まわす場合は (クエリ数 × 試行回数) になる。
    # 割合の分母はこれであり、クエリ数そのものではない。
    num_observations: int

    # 全クエリのうち、回答に自社名が出たクエリの割合。
    mention_rate: float

    # 全クエリのうち、自社記事が引用されたクエリの割合。
    citation_rate: float

    # 引用されたときの、回答内での平均登場順位。
    # 一度も引用されなければ None。
    avg_citation_position: float | None

    # 回答中の全引用のうち、自社記事が占めた割合。
    citation_share: float

    # 全クエリのうち、検索で上位に入ったクエリの割合。
    retrieval_rate: float

    # 検索で拾われたときの平均順位。
    avg_search_rank: float | None


def evaluate_answer(
    answer: Answer,
    brand: Brand,
    search_rank: int | None,
) -> QueryMetrics:
    """回答 1 件を評価する。

    Args:
        answer: 生成された回答。
        brand: 自社の情報。名前の照合に使う。
        search_rank: 検索での自社記事の順位。圏外なら None。
    """
    # 自社記事に割り当てられた出典番号を調べる。
    # どの条件にも記事は差し込まれているが、検索で拾われなければ
    # 出典に入らないので None になる。
    target_number = next(
        (s.number for s in answer.sources if s.is_target),
        None,
    )

    # 回答本文に自社の名前（表記ゆれを含む）が出たか。
    #
    # 引用とは別に測る必要がある。
    # 出典として番号を付けられなくても本文で名前を出されることはあるし、
    # 逆に引用されても社名に触れられないこともある。
    mentioned = any(term in answer.text for term in brand.mention_terms)

    if target_number is None:
        citation_position = None
        citation_count = 0
    else:
        # citations は回答本文に登場した順に並んでいるので、
        # 最初に自社記事が現れた位置がそのまま「何番目の引用か」になる。
        citation_position = (
            answer.citations.index(target_number) + 1
            if target_number in answer.citations
            else None
        )
        citation_count = answer.citations.count(target_number)

    return QueryMetrics(
        query=answer.query,
        condition=answer.condition,
        mentioned=mentioned,
        cited=citation_count > 0,
        citation_position=citation_position,
        citation_count=citation_count,
        total_citations=len(answer.citations),
        search_rank=search_rank,
    )


def aggregate(metrics: list[QueryMetrics], condition: str) -> AggregatedMetrics:
    """複数クエリの結果を 1 つにまとめる。

    Rate 系の指標は「複数のクエリのうちどの程度の割合か」なので、
    ここで初めて意味のある値になる。
    """
    total = len(metrics)

    # 分母が 0 のときに 0 除算にならないようにする補助。
    def rate(count: int) -> float:
        return count / total if total else 0.0

    positions = [m.citation_position for m in metrics if m.citation_position]
    ranks = [m.search_rank for m in metrics if m.search_rank]

    total_citations = sum(m.total_citations for m in metrics)
    own_citations = sum(m.citation_count for m in metrics)

    return AggregatedMetrics(
        condition=condition,
        num_observations=total,
        mention_rate=rate(sum(1 for m in metrics if m.mentioned)),
        citation_rate=rate(sum(1 for m in metrics if m.cited)),
        avg_citation_position=sum(positions) / len(positions) if positions else None,
        citation_share=own_citations / total_citations if total_citations else 0.0,
        retrieval_rate=rate(sum(1 for m in metrics if m.search_rank)),
        avg_search_rank=sum(ranks) / len(ranks) if ranks else None,
    )

"""評価結果から「改善すべきクエリ」を選び出すモジュール。

評価して終わりにせず、結果を次の記事修正に渡すための最初の段階にあたる。
ここでは API を一切呼ばない。すでに測り終えた観測値を読み直すだけである。

## 単一のスコアで並べない理由

「弱さ」を 1 つの数値に潰すと、なぜそのクエリが選ばれたのかが説明できなくなる。
記事の改善は人が読んで納得できないと運用に乗らないため、
**どの段階で落ちたか**という分類を残したまま優先度をつける。

段階は README の評価基準と同じ順、つまり記事がたどる漏斗の順である。

    検索で拾われる → 回答に引用される → 根拠として効く → 社名が出る

早い段階で落ちているほど、後ろの段階を直しても意味がない。
検索で拾われない記事は、どれだけ良い内容を書いても引用されようがない。
そのため落ちた段階が早いものほど優先度を高くする。
"""

from dataclasses import dataclass

from llmo.config import EVAL
from llmo.evaluation.metrics import QueryOutcome

# 弱点の種類。落ちた段階が早いものほど優先度が高い。
#
#   not_retrieved  … 検索で圏外だった。最も重い。回答の材料にすらなっていない
#   low_rank       … 拾われたが下位。出典には入るが読まれ方が弱い
#   not_cited      … 拾われたが引用されなかった。中身がクエリに答えていない
#   weak_grounding … 引用されたが、支えた主張が少ない。記述が薄い
#   not_mentioned  … 引用されたが社名が出ない。製品名の書き方の問題
NOT_RETRIEVED = "not_retrieved"
LOW_RANK = "low_rank"
NOT_CITED = "not_cited"
WEAK_GROUNDING = "weak_grounding"
NOT_MENTIONED = "not_mentioned"

# 種類ごとの優先度。大きいほど先に直す。
_PRIORITY = {
    NOT_RETRIEVED: 4,
    LOW_RANK: 3,
    NOT_CITED: 2,
    WEAK_GROUNDING: 1,
    NOT_MENTIONED: 1,
}

# 画面と、分析プロンプトに渡す説明。
# LLM に「なぜ弱いのか」を推測させず、こちらで判定した事実として渡す。
KIND_LABELS = {
    NOT_RETRIEVED: "検索圏外",
    LOW_RANK: "検索下位",
    NOT_CITED: "引用されず",
    WEAK_GROUNDING: "根拠が薄い",
    NOT_MENTIONED: "社名が出ない",
}

# 「下位」とみなす順位のしきい値。
# top_k の下半分に入っていたら下位とする。
# 固定値にしないのは、top_k を変えたときに意味がずれるため。
def _low_rank_threshold() -> int:
    return max(2, EVAL.top_k // 2 + 1)


# 根拠寄与率がこれ未満なら「薄い」とみなす。
# 引用されている以上ゼロではないはずで、
# 3 割を切るなら回答のほとんどを他社記事が支えていることになる。
_WEAK_GROUNDING_THRESHOLD = 0.3


@dataclass
class WeakQuery:
    """改善対象に選ばれたクエリ 1 本。

    分析プロンプトへの入力になるので、
    「なぜ弱いと判定したか」を再構成できるだけの観測値を持たせる。
    """

    # 何本目のクエリか（1 始まり）。評価結果の並びと対応する。
    query_index: int

    query: str

    # 弱点の種類（上の定数のいずれか）。
    kind: str

    # 種類から決まる優先度。大きいほど先に直す。
    priority: int

    # 判定の材料になった観測値。
    search_rank: int | None
    cited: bool
    mentioned: bool

    # 根拠寄与率。引用されなかった場合は None。
    grounding_rate: float | None

    @property
    def label(self) -> str:
        """画面に出す弱点の名前。"""
        return KIND_LABELS[self.kind]

    @property
    def reason(self) -> str:
        """なぜ改善対象に選ばれたかの説明。画面と分析プロンプトの両方で使う。"""
        if self.kind == NOT_RETRIEVED:
            return "ベクトル検索の上位に入らず、回答の材料になっていない"
        if self.kind == LOW_RANK:
            return f"検索で {self.search_rank} 位と下位にとどまっている"
        if self.kind == NOT_CITED:
            return (
                f"検索では {self.search_rank} 位に入ったが、"
                "回答の根拠として引用されなかった"
            )
        if self.kind == WEAK_GROUNDING:
            rate = f"{(self.grounding_rate or 0) * 100:.0f}%"
            return f"引用はされたが、回答の主張を支えた割合が {rate} と低い"
        return "引用はされたが、回答本文に自社名・製品名が出なかった"

    def to_dict(self) -> dict:
        """保存用。DB には JSON で入れる。"""
        return {
            "query_index": self.query_index,
            "query": self.query,
            "kind": self.kind,
            "label": self.label,
            "reason": self.reason,
            "search_rank": self.search_rank,
            "cited": self.cited,
            "mentioned": self.mentioned,
            "grounding_rate": self.grounding_rate,
        }


def classify(outcome: QueryOutcome) -> str | None:
    """クエリ 1 本の観測値から、弱点の種類を判定する。

    漏斗の順に見て、最初に落ちた段階を返す。
    複数の弱点を持つクエリでも、最も早い段階のものだけを返すのは、
    そこを直さない限り後ろの段階は動かないためである。

    Returns:
        弱点の種類。どの段階でも落ちていなければ None。
    """
    if outcome.search_rank is None:
        return NOT_RETRIEVED

    if outcome.search_rank >= _low_rank_threshold():
        return LOW_RANK

    if not outcome.cited:
        return NOT_CITED

    # ここから先は「拾われて、上位で、引用もされた」クエリ。
    # 残る弱点は中身の問題になる。
    if (
        outcome.grounding_rate is not None
        and outcome.grounding_rate < _WEAK_GROUNDING_THRESHOLD
    ):
        return WEAK_GROUNDING

    if not outcome.mentioned:
        return NOT_MENTIONED

    return None


def _sort_key(weak: WeakQuery) -> tuple:
    """優先度の高い順に並べるための鍵。

    同じ種類の中では、
    「検索順位が悪いもの」→「根拠寄与率が低いもの」の順に前へ出す。
    圏外（順位 None）は最も悪い扱いにするため、大きな値に置き換える。
    """
    rank = weak.search_rank if weak.search_rank is not None else 10**6
    grounding = weak.grounding_rate if weak.grounding_rate is not None else 0.0
    return (-weak.priority, -rank, grounding, weak.query_index)


def identify_weak_queries(
    outcomes: list[QueryOutcome],
    limit: int = 3,
) -> list[WeakQuery]:
    """クエリごとの観測結果から、改善優先度の高いものを選ぶ。

    API は呼ばない。すでに測り終えた観測値を読み直すだけである。

    入力を QueryOutcome にしているのは、改善ループを
    「実行直後の評価結果」からも「保存済みの評価」からも始められるようにするため。
    組み立ては evaluation/compare.py の outcomes_from_result() /
    outcomes_from_rows() が受け持つ。

    Args:
        outcomes: 1 つの条件についての、クエリごとの観測結果。
        limit: 選ぶ本数の上限。
            分析と修正の API 呼び出しが本数に比例するため、既定は 3 本にしている。

    Returns:
        優先度の高い順に並んだ、最大 limit 本のクエリ。
        弱点が 1 つも無ければ空リスト。
    """
    weak: list[WeakQuery] = []

    for outcome in outcomes:
        kind = classify(outcome)
        if kind is None:
            continue

        weak.append(
            WeakQuery(
                query_index=outcome.query_index,
                query=outcome.query,
                kind=kind,
                priority=_PRIORITY[kind],
                search_rank=outcome.search_rank,
                cited=outcome.cited,
                mentioned=outcome.mentioned,
                grounding_rate=outcome.grounding_rate,
            )
        )

    weak.sort(key=_sort_key)
    return weak[:limit]

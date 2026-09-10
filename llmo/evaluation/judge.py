"""LLM による判定が必要な評価指標を扱うモジュール。

metrics.py の 3 指標は文字列処理だけで出せるが、
残りの 2 つは回答の中身を読んで判断する必要がある。

- Top Recommendation Rate:
  複数社が紹介されたとき、自社が最もおすすめとして扱われたか。
  「最もおすすめ」は文面から読み取るしかない。

- Answer Grounding:
  回答のどの主張を自社記事が支えているか。
  何の内容のおかげで引用されたのかを知るための指標で、
  記事のどこを厚くすべきかの手がかりになる。

判定には config の judge モデルを使う。
記事を書いたモデルにそのまま評価させると、
自分の文章を甘く採点する偏りが入るため、枠を分けている。
"""

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from llmo.retrieval.answer import Answer
from llmo.core.brand import Brand
from llmo.config import MODELS
from llmo.core.gemini import generate


class _RecommendationJudgement(BaseModel):
    """Top Recommendation Rate の判定結果を受け取る型。"""

    mentioned_vendors: list[str] = Field(
        description="回答の中で紹介されている企業名・製品名をすべて挙げる"
    )
    is_comparison: bool = Field(
        description="複数の選択肢を比較・紹介する回答になっているか"
    )
    top_vendor: str | None = Field(
        description="最も推奨されている企業名・製品名。特定の一社を推していなければ null"
    )
    reason: str = Field(description="そう判断した理由を一文で")


class _GroundingItem(BaseModel):
    """回答中の主張 1 件と、その根拠の対応。"""

    claim: str = Field(description="回答の中の主張（一文程度に要約）")
    is_supported_by_target: bool = Field(
        description="この主張が、対象記事の内容によって支えられているか"
    )
    evidence: str = Field(
        description="支えているとした場合、対象記事のどの記述が根拠か。無ければ空文字"
    )


class _GroundingJudgement(BaseModel):
    """Answer Grounding の判定結果を受け取る型。"""

    claims: list[_GroundingItem] = Field(description="回答の主要な主張ごとの判定")


@dataclass
class RecommendationResult:
    """1 つの回答についての推奨判定。"""

    # 複数の選択肢を比較する回答だったか。
    # そうでなければ Top Recommendation Rate の対象外になる
    # （比較していない回答で「1 位でない」と数えるのは不当なため）。
    is_comparison: bool

    # 自社が最も推奨されていたか。
    is_top: bool

    # 回答に登場した企業・製品の一覧。競合の可視化に使う。
    mentioned_vendors: list[str] = field(default_factory=list)

    top_vendor: str | None = None
    reason: str = ""


@dataclass
class GroundingResult:
    """1 つの回答についての根拠判定。"""

    # 回答から抽出された主張の総数。
    total_claims: int

    # そのうち、自社記事が支えている主張の数。
    supported_claims: int

    # 自社記事が支えた主張と、その根拠。
    # 「何のおかげで引用されたか」がここに出る。
    supported: list[tuple[str, str]] = field(default_factory=list)

    @property
    def grounding_rate(self) -> float:
        """回答の主張のうち、自社記事が支えた割合。"""
        return self.supported_claims / self.total_claims if self.total_claims else 0.0


_RECOMMENDATION_PROMPT = """以下は、ある質問に対して生成 AI が返した回答です。
この回答の中で、どの企業・製品が推奨されているかを判定してください。

## 質問
{query}

## 回答
{answer}

## 判定の条件
- 回答に登場する企業名・製品名をすべて挙げること。
- 複数の選択肢を比較・紹介する回答になっているかを判定すること。
- 最も強く推奨されている企業・製品を 1 つ挙げること。
  どれか 1 つを特に推しているわけではない場合は null にすること。
  「一般論として説明しているだけ」の回答で無理に 1 つ選ばないこと。
"""


_GROUNDING_PROMPT = """以下は、ある質問に対して生成 AI が返した回答と、
その回答の材料として渡された「対象記事」の本文です。

回答の中の主要な主張を挙げ、それぞれが対象記事の内容によって
支えられているかどうかを判定してください。

## 質問
{query}

## 回答
{answer}

## 対象記事の本文
{target_text}

## 判定の条件
- 回答を主要な主張に分解すること（10 個以内）。
- 各主張について、対象記事の記述が根拠になっているかを判定すること。
- 対象記事に書かれていない主張は is_supported_by_target を false にすること。
  他の資料にも同じ内容が書かれていそうだという理由で true にしないこと。
- 支えていると判定した場合は、対象記事のどの記述が根拠かを示すこと。
"""


def judge_recommendation(answer: Answer, brand: Brand) -> RecommendationResult:
    """回答の中で自社が最も推奨されているかを判定する。

    Args:
        answer: 判定対象の回答。
        brand: 自社の情報。名前の照合に使う。
    """
    response = generate(
        model=MODELS.judge,
        contents=_RECOMMENDATION_PROMPT.format(query=answer.query, answer=answer.text),
        config={
            "response_mime_type": "application/json",
            "response_schema": _RecommendationJudgement,
        },
    )
    judgement: _RecommendationJudgement = response.parsed

    # LLM が返した企業名が自社を指しているかを、表記ゆれを含めて照合する。
    # 「SignFlow」「サインフロー株式会社」など、どの表記で返ってきても
    # 同じ企業として扱えるようにする。
    top = judgement.top_vendor or ""
    is_top = any(term in top for term in brand.mention_terms)

    return RecommendationResult(
        is_comparison=judgement.is_comparison,
        is_top=is_top,
        mentioned_vendors=judgement.mentioned_vendors,
        top_vendor=judgement.top_vendor,
        reason=judgement.reason,
    )


def judge_grounding(answer: Answer) -> GroundingResult | None:
    """回答のどの主張を自社記事が支えているかを判定する。

    Returns:
        判定結果。自社記事が出典に含まれていなければ None
        （支えようがないため、判定する意味がない）。
    """
    target_sources = [s for s in answer.sources if s.is_target]
    if not target_sources:
        return None

    target_text = "\n…\n".join(s.text for s in target_sources)

    response = generate(
        model=MODELS.judge,
        contents=_GROUNDING_PROMPT.format(
            query=answer.query,
            answer=answer.text,
            target_text=target_text,
        ),
        config={
            "response_mime_type": "application/json",
            "response_schema": _GroundingJudgement,
        },
    )
    judgement: _GroundingJudgement = response.parsed

    supported = [
        (item.claim, item.evidence)
        for item in judgement.claims
        if item.is_supported_by_target
    ]

    return GroundingResult(
        total_claims=len(judgement.claims),
        supported_claims=len(supported),
        supported=supported,
    )


def top_recommendation_rate(results: list[RecommendationResult]) -> float | None:
    """Top Recommendation Rate を集計する。

    分母は「複数社を比較している回答」だけにする。
    比較していない回答を分母に入れると、
    そもそも 1 位を決める場面がないのに負けとして数えてしまうため。

    Returns:
        割合。比較回答が 1 つも無ければ None。
    """
    comparisons = [r for r in results if r.is_comparison]
    if not comparisons:
        return None
    return sum(1 for r in comparisons if r.is_top) / len(comparisons)

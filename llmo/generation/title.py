"""タイトル案を作るモジュール。

README の流れでは、記事を書く前にここが入る。
ユーザーが案の中から 1 つ選び、そのタイトルで記事を作成・書き直しする。

タイトルを人間に選ばせるのは、タイトルが検索クエリと最も直接ぶつかる要素だから。
本文の書き方は基準を決めて機械的に指示できるが、
「どの切り口で書くか」は事業の判断であり、自動で決めてしまうべきではない。

パターン A（トピック）とパターン B（既存記事）で入力が違うので、
入口を 2 つに分けている。ただしタイトルに求める性質は同じなので、
条件は _TITLE_RULES に 1 か所だけ書く。
"""

from dataclasses import dataclass

from pydantic import BaseModel, Field

from llmo.core.brand import Brand
from llmo.generation.audience import Issue
from llmo.config import EVAL, MODELS
from llmo.core.gemini import generate
from llmo.generation.writer import Article


class _TitleOutput(BaseModel):
    """Gemini に JSON で返させるための型。"""

    title: str = Field(description="記事のタイトル")
    reason: str = Field(
        description="この切り口がどの想定質問に効くかを一文で"
    )


class _TitleListOutput(BaseModel):
    """タイトル案の一覧。"""

    candidates: list[_TitleOutput] = Field(description="タイトル案")


@dataclass
class TitleCandidate:
    """タイトル案 1 件。"""

    title: str

    # なぜこの切り口なのか。ユーザーが選ぶときの手がかりにする。
    # 案だけを並べても、何が違うのか分からず選べないため。
    reason: str


# タイトルに求める条件。パターン A・B の両方から使う。
#
# 「案ごとに切り口を変える」を入れているのが要点。
# 同じ内容の言い換えを 8 個並べても選ぶ意味がないため。
_TITLE_RULES = """- 想定質問に近い言い回しにすること。
  読者がその質問を投げたときに、答えが載っていそうだと分かるタイトルにする。
- 具体的にすること。対象・条件・数値のいずれかを含める。
  「〜とは」「〜のすべて」のような漠然としたタイトルにしないこと。
- 案ごとに切り口を変えること。
  同じ内容の言い換えを並べないこと。読者の状況のどれに答えるかを案ごとに変える。
- 日本語で、40 文字程度までにすること。
- 宣伝文句にしないこと。製品名を入れるかどうかは切り口次第で判断してよいが、
  すべての案に入れないこと。比較・調査の段階の読者には選ばれにくくなる。"""


_FROM_TOPIC_PROMPT = """あなたは BtoB SaaS 企業のオウンドメディアの編集者です。
これから書く記事のタイトル案を {num} 個出してください。
この記事は、人間の読者だけでなく、生成 AI の検索結果に引用されることを狙っています。

## トピック
{topic}

## 自社について
- 企業名: {company}
- 製品名: {product}
- 事業内容: {description}

## 想定読者
{audience}

## 読者の状況
この記事を読む読者が置かれている状況です。
以下の記述の語句をそのまま並べたタイトルにせず、自分の言葉で書いてください。
{issues}

## タイトルの条件
{rules}
"""


_FROM_ARTICLE_PROMPT = """あなたは BtoB SaaS 企業のオウンドメディアの編集者です。
以下の記事を、生成 AI の検索結果に引用されやすくなるように書き直します。
その書き直し後のタイトル案を {num} 個出してください。

## 現在の記事

### タイトル
{original_title}

### 本文
{original_body}

## 自社について
- 企業名: {company}
- 製品名: {product}
- 事業内容: {description}

## 想定読者
{audience}

## 読者の状況
この記事を読む読者が置かれている状況です。
以下の記述の語句をそのまま並べたタイトルにせず、自分の言葉で書いてください。
{issues}

## タイトルの条件
- 元の記事が扱っている内容の範囲を保つこと。
  記事の中身と合わないタイトルを付けると、検索で拾われても回答に使われない。
{rules}
"""


def _format_issues(issues: list[Issue]) -> str:
    """読者の状況をプロンプトに埋め込む形に整える。

    タイトルは本文の先頭に見出しとして載るため、
    記事本文と同じく、検索クエリの文字列を渡さない。
    """
    return "\n".join(f"- {issue.situation}" for issue in issues)


def _parse(response) -> list[TitleCandidate]:
    """Gemini の返答を TitleCandidate の一覧にする。"""
    output: _TitleListOutput = response.parsed
    return [
        TitleCandidate(title=c.title, reason=c.reason) for c in output.candidates
    ]


def generate_titles_from_topic(
    topic: str,
    brand: Brand,
    issues: list[Issue],
    num: int | None = None,
) -> list[TitleCandidate]:
    """トピックからタイトル案を作る（パターン A）。

    Args:
        topic: 記事のトピック。
        brand: 記事を出す企業の情報。
        issues: 読者の状況。評価に使うクエリを言い換えたもので、
            クエリの文字列そのものは渡らない。
        num: 作る案の数。省略すると config の既定値を使う。
    """
    prompt = _FROM_TOPIC_PROMPT.format(
        num=num or EVAL.num_title_candidates,
        topic=topic,
        company=brand.company,
        product=brand.product,
        description=brand.description,
        audience=brand.audience,
        issues=_format_issues(issues),
        rules=_TITLE_RULES,
    )

    return _parse(generate(
        model=MODELS.generation,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": _TitleListOutput,
        },
    ))


def generate_titles_from_article(
    original: Article,
    brand: Brand,
    issues: list[Issue],
    num: int | None = None,
) -> list[TitleCandidate]:
    """既存の記事の内容からタイトル案を作る（パターン B）。

    Args:
        original: 書き直す前の記事。
        brand: 記事を出す企業の情報。
        issues: 読者の状況（クエリを言い換えたもの）。
        num: 作る案の数。省略すると config の既定値を使う。
    """
    prompt = _FROM_ARTICLE_PROMPT.format(
        num=num or EVAL.num_title_candidates,
        original_title=original.title,
        original_body=original.body,
        company=brand.company,
        product=brand.product,
        description=brand.description,
        audience=brand.audience,
        issues=_format_issues(issues),
        rules=_TITLE_RULES,
    )

    return _parse(generate(
        model=MODELS.generation,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": _TitleListOutput,
        },
    ))

"""記事を生成するモジュール。

このシステムの出力そのもの。README でいう「未公開記事」を作る。

記事の書き方はルールベースにせず、プロンプトで指示する方針。
「見出しを N 個入れる」のような機械的な規則で良い記事は書けないため。
プロンプトはロジックの一部なので、関数の外に定数として置き、
変更の履歴を追えるようにしている。

README の入力のパターンに対応して、記事の作り方を分けている。

- generate_optimized_article(): パターン A。トピックから最適化記事を新規作成する。
- rewrite_article():            パターン B。既存の記事を最適化して書き直す。
- from_existing():              パターン B の書き直し前の記事を受け取る。

A と B は入口が違うだけで、目指す記事の性質は同じである。
そのため最適化の条件は llmo/generation/rules.py に 1 か所だけ置き、両方から使う。
この条件は画面から編集できる。
別々に書くと、片方を直したときにもう片方の基準がずれていくため。
"""

from dataclasses import dataclass

from pydantic import BaseModel, Field

from llmo.core.brand import Brand
from llmo.generation import rules
from llmo.generation.audience import Issue
from llmo.config import EVAL, MODELS
from llmo.core.gemini import generate


class _ArticleOutput(BaseModel):
    """Gemini に JSON で返させるための型。"""

    title: str = Field(description="記事のタイトル")
    body: str = Field(description="記事の本文（マークダウン）")


@dataclass
class Article:
    """生成された記事。"""

    title: str
    body: str

    # どの方針で書かれたか。
    #   "optimized" … パターン A で新規作成した最適化記事
    #   "existing"  … パターン B の書き直し前（人間が書いた記事）
    #   "rewritten" … パターン B の書き直し後
    #   "improved"  … 評価結果をもとに弱い節だけを直した版（改善ループ）
    # 履歴を見返すときに、どの版の結果だったかを見分ける目印になる。
    style: str


def _format_strengths(strengths: list[str]) -> str:
    """製品の強みをプロンプトに埋め込む形に整える。"""
    return "\n".join(f"  - {s}" for s in strengths)


# 自社の情報。両方のプロンプトで同じ書式にする。
_BRAND_BLOCK = """## 自社について
- 企業名: {company}
- 製品名: {product}
- 事業内容: {description}
- 製品の強み:
{strengths}

## 想定読者
{audience}

## 読者の状況
この記事を読む読者が置かれている状況です。
記事はこれらの状況にある読者の役に立つ内容にしてください。

見出しは、これらの状況に答える内容を**自分の言葉で**書いてください。
以下の記述の語句をそのまま並べた見出しにしないこと。
{issues}"""


# パターン A: トピックから最適化記事を新規作成する。
_OPTIMIZED_PROMPT = """あなたは BtoB SaaS 企業のオウンドメディアの編集者です。
この記事は、人間の読者だけでなく、生成 AI の検索結果に引用されることを狙って書きます。

以下のトピックとタイトルで記事を書いてください。

## トピック
{topic}

## タイトル
{title}

このタイトルはユーザーが選んだものです。**一字も変えずにそのまま使ってください。**
本文はこのタイトルが約束している内容に答えるものにしてください。

""" + _BRAND_BLOCK + """

## 書き方の条件

{rules}
"""


# パターン B: 既存の記事を最適化して書き直す。
#
# 「元記事の主張と事実を保つ」と明示しているのが要点。
# 中身まで書き換えてしまうと、書き直し前後で比べているものが
# 「同じ記事の書き方の違い」ではなく「別の記事」になってしまい、
# 書き直しの効果を測るという目的が果たせなくなる。
_REWRITE_PROMPT = """あなたは BtoB SaaS 企業のオウンドメディアの編集者です。
以下の記事を、生成 AI の検索結果に引用されやすくなるように書き直してください。

## 書き直す記事

### タイトル
{original_title}

### 本文
{original_body}

## 書き直し後のタイトル
{title}

このタイトルはユーザーが選んだものです。**一字も変えずにそのまま使ってください。**
本文はこのタイトルが約束している内容に答えるものにしてください。

""" + _BRAND_BLOCK + """

## 書き直しの条件

### 守ること
- 元の記事が述べている主張と事実を保つこと。
  書かれていない事実を新たに作り出さないこと。
- 元の記事が扱っている範囲を大きく超えないこと。
  別の記事にしてしまうと、書き直しの効果を測れなくなる。

### 書き方
{rules}
"""


def _format_issues(issues: list[Issue]) -> str:
    """読者の状況をプロンプトに埋め込む形に整える。

    渡すのは言い換えた状況だけで、元の検索クエリは渡さない。
    クエリの文字列が記事に転記されると、そのクエリで検索したときに
    文字列の一致だけで上位に入り、評価が成り立たなくなるためである。
    """
    return "\n".join(f"- {issue.situation}" for issue in issues)


def _format_rules(brand: Brand) -> str:
    """LLMO 最適化の指示を、プロンプトに差し込める形にする。

    指示の文面に含まれる {chunk_size} や {product} は、ここで埋める。
    str.format は差し込んだ先の文字列をもう一度処理しないため、
    プロンプト本体に入れる前に済ませておく必要がある。

    指示そのものは llmo/generation/rules.py が持つ。
    画面から編集されている場合は、その内容が返る。
    """
    return rules.load().format(chunk_size=EVAL.chunk_size, product=brand.product)


def generate_optimized_article(
    topic: str, title: str, brand: Brand, issues: list[Issue]
) -> Article:
    """トピックと選ばれたタイトルから、LLMO 最適化された記事を生成する。

    rewrite_article() とはモデルも出力の型も揃えてある。
    結果に差が出たとき、原因をプロンプトに絞れるようにするため。

    Args:
        topic: 記事のトピック。
        title: ユーザーが選んだタイトル。llmo/title.py が出した案の 1 つ。
            タイトルは検索クエリと最も直接ぶつかる要素であり、
            どの切り口で書くかは事業の判断なので、自動で決めずに固定する。
        brand: 記事を出す企業の情報。data/brand.json から読む。
        issues: 読者の状況。評価に使うクエリを言い換えたもので、
            クエリの文字列そのものは渡らない（audience.to_issues が作る）。
            評価と同じクエリを渡すのは、評価に合わせて記事を書かせているのではなく、
            「読者がその質問をする」という前提を両者で揃えるため。

    Returns:
        生成された記事。
    """
    prompt = _OPTIMIZED_PROMPT.format(
        topic=topic,
        title=title,
        company=brand.company,
        product=brand.product,
        description=brand.description,
        strengths=_format_strengths(brand.strengths),
        audience=brand.audience,
        issues=_format_issues(issues),
        rules=_format_rules(brand),
    )

    response = generate(
        model=MODELS.generation,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": _ArticleOutput,
        },
    )
    output: _ArticleOutput = response.parsed

    # タイトルはユーザーが選んだものを使う。
    # プロンプトで指示していても、モデルが言い換えて返すことがあるため、
    # ここで確定させる。
    return Article(title=title, body=output.body, style="optimized")


def from_existing(title: str, body: str) -> Article:
    """人間が書いた記事をそのまま Article にする（パターン B の書き直し前）。

    ここでは書き直しをしない。
    パターン B は「書き直し前」と「書き直し後」を比べるので、
    書き直し前の記事も条件の 1 つとして評価に渡す必要がある。
    書き直しは rewrite_article() が行う。

    Args:
        title: 記事のタイトル。
        body: 記事の本文。

    Returns:
        評価に渡せる形にした記事。
    """
    return Article(title=title, body=body, style="existing")


def rewrite_article(
    original: Article, title: str, brand: Brand, issues: list[Issue]
) -> Article:
    """既存の記事を LLMO 最適化して書き直す（パターン B）。

    generate_optimized_article() と同じ最適化条件（rules モジュール）を使う。
    入口が違うだけで、目指す記事の性質は同じであるため。

    Args:
        original: 書き直す前の記事。from_existing() で作ったもの。
        title: ユーザーが選んだ書き直し後のタイトル。
        brand: 記事を出す企業の情報。
        issues: 読者の状況。評価に使うクエリを言い換えたもので、
            クエリの文字列そのものは渡らない（audience.to_issues が作る）。

    Returns:
        書き直した記事。
    """
    prompt = _REWRITE_PROMPT.format(
        original_title=original.title,
        original_body=original.body,
        title=title,
        company=brand.company,
        product=brand.product,
        description=brand.description,
        strengths=_format_strengths(brand.strengths),
        audience=brand.audience,
        issues=_format_issues(issues),
        rules=_format_rules(brand),
    )

    response = generate(
        model=MODELS.generation,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": _ArticleOutput,
        },
    )
    output: _ArticleOutput = response.parsed

    # 同上。選ばれたタイトルで固定する。
    return Article(title=title, body=output.body, style="rewritten")

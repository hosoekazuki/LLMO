"""ユーザー入力から、評価用の検索クエリを複数生成するモジュール。

README の評価フローの 1 番目にあたる。

ここで作るのは「エンドユーザーが生成 AI に実際に打ち込みそうな質問文」であり、
「自社記事を見つけるためのクエリ」ではない。
後者を作ってしまうと自社記事が必ずヒットして、評価が意味を持たなくなる。
"""

from pydantic import BaseModel, Field

from llmo.config import EVAL, MODELS
from llmo.core.gemini import generate


class GeneratedQueries(BaseModel):
    """Gemini に JSON で返させるための型。

    LLM に自由文で答えさせると「1. ○○」のような箇条書きが返ってきて、
    それを正規表現で切り出す処理が必要になる。
    型を渡して JSON で返させれば、そのパース処理が丸ごと不要になる。
    """

    queries: list[str] = Field(description="検索クエリの一覧")


# LLM への指示文。プロンプトはロジックの一部なので、
# 関数の中に埋めずに定数として外に出し、変更履歴を追えるようにしておく。
_PROMPT = """あなたは、生成 AI の検索経由でどんな質問が来るかを想定する担当者です。

以下の「対象トピック」について書かれた記事が、生成 AI の回答に引用されるかを
検証したいと考えています。そのために、**実際のユーザーが ChatGPT や Gemini に
打ち込みそうな質問文**を {num_queries} 個作ってください。

## 対象トピック
{user_input}

## 条件
- 実在のユーザーが自然言語で打ち込む形にすること（検索エンジン用のキーワード列にしない）。
- 特定の企業名や記事名を含めないこと。指名検索ではなく、比較・調査の段階の質問にする。
- 5 個が互いに似すぎないように、聞き方や観点を変えること。
  （例: 概要を尋ねる / 比較を求める / 選び方を尋ねる / 具体的な課題の解決策を尋ねる）
- 日本語で書くこと。
"""


def generate_queries(user_input: str, num_queries: int | None = None) -> list[str]:
    """ユーザー入力から評価用クエリを生成して返す。

    Args:
        user_input: 記事のトピック、または既存記事の本文。
        num_queries: 生成する本数。省略時は config の値（既定 5 本）を使う。

    Returns:
        クエリ文字列のリスト。この本数が Mention Rate などの分母になる。
    """
    if num_queries is None:
        num_queries = EVAL.num_queries

    prompt = _PROMPT.format(num_queries=num_queries, user_input=user_input)

    response = generate(
        model=MODELS.generation,
        contents=prompt,
        config={
            # 出力を JSON に固定し、上で定義した型に沿わせる。
            "response_mime_type": "application/json",
            "response_schema": GeneratedQueries,
        },
    )

    # .parsed に、型に変換済みのオブジェクトが入っている。
    result: GeneratedQueries = response.parsed

    # 指示しても本数がずれることがあるため、念のため切り詰める。
    return result.queries[:num_queries]

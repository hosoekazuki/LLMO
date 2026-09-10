"""検索クエリを「読者の状況・課題」に言い換えるモジュール。

## なぜ必要か

記事生成に検索クエリをそのまま渡すと、LLM は見出しにその文字列を
書き写す。埋め込みベクトルは文字列の一致に強く反応するため、
クエリと同じ文字列を含むチャンクは、そのクエリで検索したときに
ほぼ最大の類似度を返す。結果として記事の出来にかかわらず上位に入り、
評価が自己成就する。

そこでクエリと記事生成の間に言い換えを挟み、文字列が直接渡る経路を断つ。
記事生成に渡すのは「その質問をする読者が置かれている状況」であって、
質問文そのものではない。

## 限界

意味の近さによる有利は原理的に残る。同じことを尋ねている以上、
言い換えても記事の内容はクエリに近づく。
ここで取り除けるのは文字列の直接転記という明らかなバイアスだけである。
残りは llmo/generation/leakage.py の検証で観測する。
"""

from dataclasses import dataclass

from pydantic import BaseModel, Field

from llmo.config import MODELS
from llmo.core.gemini import generate


class _IssueOutput(BaseModel):
    """Gemini に JSON で返させるための型。"""

    situations: list[str] = Field(
        description="各クエリに対応する読者の状況・課題。入力と同じ順序・同じ個数で返す"
    )


@dataclass
class Issue:
    """1 本のクエリと、それを言い換えた読者の状況。"""

    # 元になった検索クエリ。記事生成には渡さない。
    # 生成後の検証（leakage.py）で照合するために保持する。
    query: str

    # 記事生成に渡す、読者の状況・課題の記述。
    situation: str


_PROMPT = """あなたは、BtoB SaaS のオウンドメディアで読者像を整理する担当者です。

以下は、ある製品カテゴリについて調べている読者が、生成 AI に打ち込んだ
検索クエリの一覧です。それぞれについて、**その質問をする読者が
置かれている状況**を、自然な日本語の文で書いてください。

## 検索クエリ
{queries}

## 条件
- 入力と**同じ順序・同じ個数**で返すこと。
- **検索クエリの語句をそのまま使わないこと。** 同じ意味でも、別の言い方に置き換える。
  この記述は記事の材料になるため、クエリの文字列がそのまま残ると、
  検索の評価が成り立たなくなる。
- 疑問文にしないこと。「〜を知りたい読者」ではなく、
  「〜という状態にあり、〜を判断できずにいる読者」のように、
  読者が何に困っているかが分かる書き方にする。
- その読者が置かれている業務上の状況、すでに試したこと、
  判断できずにいることのいずれかに触れること。
- 1 件あたり 60〜120 字程度にすること。

## 例
クエリ: 稟議 時間がかかる 原因
状況　: 申請してから承認が下りるまでに時間がかかっており、
　　　　どの工程で滞留しているかを把握できていない読者。
　　　　部署ごとに運用が異なるため、原因の切り分けができずにいる。
"""


def to_issues(queries: list[str]) -> list[Issue]:
    """検索クエリを読者の状況に言い換える。

    Args:
        queries: generate_queries() が作ったクエリ。

    Returns:
        クエリと言い換えの組。入力と同じ順序・同じ本数。

    Raises:
        ValueError: 言い換えの本数が入力と合わないとき。
            順序で対応させているため、個数がずれると
            別のクエリの状況を記事に渡すことになる。
    """
    prompt = _PROMPT.format(queries="\n".join(f"- {q}" for q in queries))

    response = generate(
        model=MODELS.transform,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": _IssueOutput,
        },
    )
    situations = _IssueOutput.model_validate_json(response.text).situations

    if len(situations) != len(queries):
        raise ValueError(
            f"言い換えの本数が合いません（クエリ {len(queries)} 本に対して "
            f"{len(situations)} 件）。順序で対応させているため処理を止めます。"
        )

    return [
        Issue(query=query, situation=situation.strip())
        for query, situation in zip(queries, situations)
    ]

"""検索結果をもとに回答を生成するモジュール。

README の評価フローの 5 番目にあたる。
実際の生成 AI が「検索してきたページを読んで回答を書く」段階を再現する。

ここで重要なのは、どの記事を引用したかを機械的に判定できる形にすること。
自由に書かせると「クラウドサインによると…」のような書き方になり、
どの出典を使ったのかを後から数えられない。
実際の生成 AI と同じく [1] [2] の形で出典番号を付けさせ、それを数える。
"""

import re
from dataclasses import dataclass, field

from llmo.config import EVAL, MODELS
from llmo.corpus.fetch import Document
from llmo.core.gemini import generate
from llmo.retrieval.retrieve import SearchResult


@dataclass
class Source:
    """回答生成に渡した出典 1 件。

    番号は 1 始まり。回答中の [1] [2] がこの番号に対応する。
    """

    number: int
    doc_index: int
    title: str
    url: str
    text: str
    is_target: bool


@dataclass
class Answer:
    """生成された回答 1 件分。"""

    # どの条件で生成したか（"single" / "before" / "after"）。
    condition: str

    query: str
    text: str

    # 回答生成に渡した出典の一覧。
    sources: list[Source] = field(default_factory=list)

    # 回答中に登場した出典番号を、登場した順に並べたもの。
    # 同じ番号が複数回出ることもある（引用の量を測るのに使う）。
    citations: list[int] = field(default_factory=list)


# 回答生成のプロンプト。
# 実際の生成 AI のウェブ検索モードに近い指示にする。
_PROMPT = """あなたは、ウェブ検索の結果をもとにユーザーの質問に答えるアシスタントです。

以下の「検索結果」だけを根拠にして、ユーザーの質問に日本語で答えてください。

## ユーザーの質問
{query}

## 検索結果
{sources}

## 条件
- あなたはGoogle検索アシスタントであり、ユーザーの求める内容を推論し、出力する生成AIである。
- 検索結果に書かれていないことは書かないこと。
- 根拠にした箇所には、必ず出典番号を [1] のように付けること。
  複数の出典を根拠にした文には [1][3] のように並べて付けること。
- 有用な検索結果だけを使うこと。すべての出典を無理に使う必要はない。
- 800 文字程度でまとめること。
"""


def _format_sources(sources: list[Source]) -> str:
    """出典をプロンプトに埋め込む形に整える。

    タイトルと URL も一緒に渡す。
    実際の生成 AI も出典の見た目を手がかりに信頼性を判断しているため、
    本文だけを渡すと実態から離れる。
    """
    blocks = []
    for source in sources:
        blocks.append(
            f"[{source.number}] {source.title}\n"
            f"URL: {source.url}\n"
            f"{source.text}"
        )
    return "\n\n---\n\n".join(blocks)


def _truncate(text: str, limit: int) -> str:
    """本文を上限の文字数で切る。

    途中で切れたことが LLM に分かるように印を付ける。
    印が無いと、切れた末尾を「そこで話が終わっている」と読んでしまう。

    切る位置は、上限の手前で最後に現れる改行に寄せる。
    文の途中で切ると、壊れた文をそのまま根拠として引用されることがあるため。
    """
    if len(text) <= limit:
        return text

    head = text[:limit]
    # 上限の 8 割より後ろに改行があれば、そこで切る。
    # 8 割で線を引くのは、改行がずっと手前にしかない場合に
    # 本文を大きく削ってしまうのを防ぐため。
    boundary = head.rfind("\n")
    if boundary > limit * 0.8:
        head = head[:boundary]

    return head.rstrip() + "\n…（以下略）"


def build_sources(results: list[SearchResult], documents: list[Document]) -> list[Source]:
    """検索結果を、回答生成に渡す出典の形に変換する。

    渡すのは検索で当たったチャンクではなく、記事の本文である。
    実際の生成 AI は、検索エンジンが返したページの本文をそのまま
    コンテキストに入れて回答を書く。チャンクだけを渡すと、
    記事のどこを読ませるかをこちら側が先回りして決めてしまうことになり、
    引用先の選択が実態から離れる。
    ベクトル検索は「どの記事を読ませるか」を決めるところまでで役目を終える。

    本文は EVAL.max_source_chars で切り揃える。
    数万字の未公開記事と数千字の他社記事をそのまま並べると、
    自社記事だけがコンテキストを占有して有利になるため。

    matched_chunks は LLM には渡さないが、
    「どのチャンクが検索に当たったか」の記録として SearchResult に残る。
    """
    sources = []
    for number, result in enumerate(results, start=1):
        doc = documents[result.doc_index]
        sources.append(
            Source(
                number=number,
                doc_index=result.doc_index,
                title=doc.title,
                url=doc.url,
                text=_truncate(doc.content, EVAL.max_source_chars),
                is_target=doc.is_target,
            )
        )
    return sources


def _extract_citations(text: str, max_number: int) -> list[int]:
    """回答本文から出典番号を、登場した順に抜き出す。

    "[1][3]" のように連続する場合も個別に拾う。
    存在しない番号を書いてくることがあるため、範囲外は捨てる。
    """
    numbers = [int(n) for n in re.findall(r"\[(\d+)\]", text)]
    return [n for n in numbers if 1 <= n <= max_number]


def generate_answer(query: str, sources: list[Source], condition: str) -> Answer:
    """出典をもとに回答を生成する。

    Args:
        query: ユーザーの質問（検索クエリ）。
        sources: build_sources() が作った出典。
        condition: 条件の名前（"single" / "before" / "after"）。

    Returns:
        生成された回答と、そこに含まれる引用の記録。
    """
    response = generate(
        model=MODELS.generation,
        contents=_PROMPT.format(query=query, sources=_format_sources(sources)),
    )
    text = response.text or ""

    return Answer(
        condition=condition,
        query=query,
        text=text,
        sources=sources,
        citations=_extract_citations(text, max_number=len(sources)),
    )

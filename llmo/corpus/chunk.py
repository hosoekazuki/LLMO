"""文書をチャンク（検索の最小単位）に分割するモジュール。

なぜ分割するのか:
47,000 字の記事を丸ごと 1 つのベクトルにすると、記事全体の平均的な意味しか残らず、
「この記事のどの部分がクエリに答えているか」が消えてしまう。
段落くらいの粒度に割ることで、クエリに直接答えている箇所を拾えるようになり、
Answer Grounding（どの主張が引用を生んだか）の分析にもつながる。
"""

from dataclasses import dataclass

from llmo.config import EVAL
from llmo.corpus.fetch import Document


@dataclass
class Chunk:
    """検索対象になる文書の断片。"""

    # このチャンクの元になった文書が、コーパスの何番目か。
    # 検索後に「どの記事から来たチャンクか」を辿るために持つ。
    # 記事単位で集約するときに使う。
    doc_index: int

    # その文書の中で何番目のチャンクか。本文中の位置の目安になる。
    chunk_index: int

    # チャンクの本文。これをベクトル化する。
    text: str


def split_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """テキストをチャンクに分割する。

    段落（空行）の境目を優先して切る。
    単純に N 文字ごとに切ると文の途中で分断され、意味が壊れたチャンクができるため。

    Args:
        text: 分割する本文。
        chunk_size: 1 チャンクの目安の文字数。省略時は config の値（既定 500）。
        overlap: チャンク同士を重ねる文字数。省略時は config の値（既定 100）。

    Returns:
        チャンク本文のリスト。
    """
    if chunk_size is None:
        chunk_size = EVAL.chunk_size
    if overlap is None:
        overlap = EVAL.chunk_overlap

    # 空行で段落に割る。空だけの段落は捨てる。
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # 1 段落だけで既に大きすぎる場合は、その段落を単独で扱う。
        # （見出しの無い長文などで起きる）
        if len(para) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            # 長すぎる段落は、やむを得ず文字数で機械的に割る。
            for i in range(0, len(para), chunk_size):
                chunks.append(para[i : i + chunk_size])
            continue

        # この段落を足すと目安を超えるなら、今のチャンクを確定して次に移る。
        if current and len(current) + len(para) > chunk_size:
            chunks.append(current)
            # 直前のチャンクの末尾を次の先頭に重ねる。
            # 境目で話が切れて、前後の文脈を失ったチャンクができるのを防ぐ。
            current = current[-overlap:] if overlap > 0 else ""

        current = f"{current}\n{para}" if current else para

    if current:
        chunks.append(current)

    return chunks


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    """コーパス全体をチャンクに分割する。

    doc_index には、引数 documents の中での位置（0 始まり）を入れる。
    検索結果を記事単位に集約するときに、この番号で元の文書に戻る。

    Args:
        documents: 疑似ウェブの文書一覧。

    Returns:
        全文書のチャンクを 1 本のリストにまとめたもの。
    """
    chunks: list[Chunk] = []

    for doc_index, doc in enumerate(documents):
        for chunk_index, text in enumerate(split_text(doc.content)):
            chunks.append(Chunk(doc_index=doc_index, chunk_index=chunk_index, text=text))

    return chunks

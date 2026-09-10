"""疑似ウェブに対してベクトル検索を行うモジュール。

README の評価フローの 4 番目「検索の再現」にあたる。

検索の単位は「記事」にする。
チャンク単位で上位 K 件を取ると、長い記事はチャンク数が多いぶん当選しやすく、
上位 5 件がすべて同じ 1 記事から来ることも起こる。
実際の検索エンジンは複数のページを返すため、それでは現実から離れてしまう。

そこでチャンク単位で類似度を計算した後、記事ごとに集約してから順位をつける。
"""

from dataclasses import dataclass

import numpy as np

from llmo.corpus.chunk import Chunk
from llmo.corpus.fetch import Document


@dataclass
class SearchResult:
    """検索で選ばれた記事 1 件分の結果。"""

    # コーパスの中での記事の位置。Document を引くのに使う。
    doc_index: int

    # この記事の代表スコア（最も高かったチャンクの類似度）。
    score: float

    # 実際にクエリに近かったチャンクの本文。
    # 回答生成のときは記事全文ではなくこれを渡す。
    # 実際の LLM も、ページ全体ではなく関連箇所を読んで回答するため。
    matched_chunks: list[str]


def search(
    query_vector: np.ndarray,
    chunk_vectors: np.ndarray,
    chunks: list[Chunk],
    top_k: int,
    chunks_per_doc: int = 2,
) -> list[SearchResult]:
    """1 つのクエリで検索し、上位 top_k 件の記事を返す。

    Args:
        query_vector: クエリのベクトル（1 次元、正規化済み）。
        chunk_vectors: 全チャンクのベクトル（チャンク数 × 次元数、正規化済み）。
        chunks: chunk_vectors と同じ並び順のチャンク。
        top_k: 返す記事の件数。
        chunks_per_doc: 1 記事あたり、回答生成に渡すチャンクの最大数。

    Returns:
        スコアの高い順に並んだ記事のリスト。
    """
    # 全チャンクとの類似度を一度に計算する。
    # ベクトルは長さ 1 に正規化済みなので、内積がそのままコサイン類似度になる。
    # ループを書かずに行列の掛け算 1 回で済むため、445 チャンクでも一瞬で終わる。
    similarities = chunk_vectors @ query_vector

    # 記事ごとに、その記事のチャンクのスコアを集める。
    per_doc: dict[int, list[tuple[float, str]]] = {}
    for chunk, score in zip(chunks, similarities):
        per_doc.setdefault(chunk.doc_index, []).append((float(score), chunk.text))

    results: list[SearchResult] = []
    for doc_index, scored_chunks in per_doc.items():
        # スコアの高い順に並べる。
        scored_chunks.sort(key=lambda item: item[0], reverse=True)

        # 記事の代表スコアは「最も高かったチャンク」の値。
        #
        # 平均にしないのは、長い記事ほど本題と関係ない部分に薄められて
        # 不利になるため。実際の検索も「そのページにクエリへ答える箇所があるか」
        # で評価するので、最高値のほうが実態に近い。
        results.append(
            SearchResult(
                doc_index=doc_index,
                score=scored_chunks[0][0],
                matched_chunks=[text for _, text in scored_chunks[:chunks_per_doc]],
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


def find_rank(results: list[SearchResult], doc_index: int) -> int | None:
    """検索結果の中で、指定した記事が何位だったかを返す。

    未公開記事が上位に入れたかどうかを見るために使う。
    README でいう「検索結果に含まれるか」の観測にあたる。

    Returns:
        1 始まりの順位。入っていなければ None。
    """
    for rank, result in enumerate(results, start=1):
        if result.doc_index == doc_index:
            return rank
    return None


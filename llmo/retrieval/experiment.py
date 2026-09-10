"""ベースコーパスに未公開記事を差し込んで、検索できる条件を組み立てるモジュール。

README の「疑似ウェブによる評価環境」にあたる。

用語を 2 つに分ける。

- **ベースコーパス** — Tavily で取得した他社記事だけの集合。一度作ったら固定して使い回す。
  これ自体は検索の対象にしない。
- **条件（Condition）** — ベースコーパスに未公開記事を 1 本差し込んだもの。
  実際に検索するのはこちら。

README の前提として、**どの条件でも未公開記事は必ず 1 本だけ差し込む**。
差し込む本数が条件によって変わると、検索の枠を奪い合う数が変わってしまい、
記事そのものの効果と混ざるためである。
そのため取得件数 K もすべての条件で共通にしている。

パターンごとの条件の作り方は以下のとおり。

- パターン A（トピックのみ）: 条件は 1 つ。作成した記事を差し込む。
- パターン B（既存記事）: 条件は 2 つ。書き直し前の記事と、書き直し後の記事を
  それぞれ差し込む。ベースコーパスは共通なので、差は記事だけになる。
"""


from dataclasses import dataclass

import numpy as np

from llmo.corpus.chunk import Chunk, chunk_documents
from llmo.corpus.clean import clean_documents
from llmo.config import EVAL
from llmo.corpus.fetch import Document, load_corpus
from llmo.corpus.embed import embed_documents
from llmo.retrieval.retrieve import SearchResult, search
from llmo.generation.writer import Article


@dataclass
class BaseCorpus:
    """他社記事だけの疑似ウェブ。すべての条件で共通の土台になる。

    これ単体では検索しない。
    README の評価はどのパターンでも未公開記事を差し込んだ状態で行うため、
    記事の入っていないコーパスを検索する場面が存在しない。
    """

    name: str
    documents: list[Document]
    chunks: list[Chunk]
    vectors: np.ndarray


@dataclass
class Condition:
    """検索できる状態になった 1 つの条件。

    name は "single"（パターン A）、"before" / "after"（パターン B）のいずれか。
    """

    name: str
    documents: list[Document]
    chunks: list[Chunk]
    vectors: np.ndarray

    # 差し込んだ未公開記事がコーパスの何番目か。
    # どの条件にも必ず 1 本入るので、None にはならない。
    target_index: int

    # この条件で差し込んだ記事そのもの。
    # 「どの記事を入れた条件だったか」を後から辿れるようにするため。
    article: Article

    def search(self, query_vector: np.ndarray) -> list[SearchResult]:
        """この条件のコーパスを検索する。

        取得件数は EVAL.top_k で固定する。
        条件ごとに変えると、記事の差ではなく件数の差を見ることになってしまう。
        """
        return search(query_vector, self.vectors, self.chunks, EVAL.top_k)

    def target_chunks(self, query_vector: np.ndarray, limit: int = 2) -> list[str]:
        """このクエリに最も近い、差し込んだ記事の部分を返す。

        search() では取れない。検索で圏外になった記事は結果に現れないが、
        改善の分析では「圏外だった記事のどこが最も近かったか」こそ知りたい。
        検索の順位に関係なく、差し込んだ記事のチャンクだけを見て類似度を計算する。

        ベクトルは計算済みのものを使うので、API も埋め込みも走らない。

        Args:
            query_vector: クエリのベクトル（正規化済み）。
            limit: 返すチャンクの数。

        Returns:
            類似度の高い順に並んだチャンク本文。
        """
        positions = [
            index
            for index, chunk in enumerate(self.chunks)
            if chunk.doc_index == self.target_index
        ]
        if not positions:
            return []

        scores = self.vectors[positions] @ query_vector
        # スコアの高い順に並べ替えて、上から limit 件を返す。
        order = np.argsort(scores)[::-1][:limit]
        return [self.chunks[positions[i]].text for i in order]

    def rival_chunks(
        self,
        query_vector: np.ndarray,
        num_docs: int = 2,
        per_doc: int = 1,
    ) -> list[str]:
        """このクエリで上位に入った他社記事の該当部分を返す。

        改善の分析で「競合には書かれていて自社に無い情報」を挙げさせるのに使う。
        検索は評価と同じ search() を通すので、実際に上位を占めた記事と一致する。

        Args:
            query_vector: クエリのベクトル（正規化済み）。
            num_docs: 何記事ぶん取るか。
            per_doc: 1 記事あたり何チャンク取るか。

        Returns:
            上位の他社記事から取ったチャンク本文。
        """
        texts: list[str] = []
        for result in self.search(query_vector):
            if result.doc_index == self.target_index:
                continue
            texts.extend(result.matched_chunks[:per_doc])
            if len(texts) >= num_docs * per_doc:
                break
        return texts[: num_docs * per_doc]


def build_base(corpus_name: str) -> BaseCorpus:
    """保存済みの疑似ウェブからベースコーパスを組み立てる。

    本文の掃除・チャンク分割・ベクトル化までを行う。
    ベクトルはキャッシュから読むので、2 回目以降は API を叩かない。
    """
    documents = [
        doc
        for doc in load_corpus(corpus_name)
        if len(doc.content) >= EVAL.min_content_chars
    ]
    documents = clean_documents(documents)
    chunks = chunk_documents(documents)
    vectors = embed_documents([c.text for c in chunks], cache_name=corpus_name)

    return BaseCorpus(
        name=corpus_name,
        documents=documents,
        chunks=chunks,
        vectors=vectors,
    )


def build_condition(
    base: BaseCorpus,
    article: Article,
    url: str,
    name: str,
) -> Condition:
    """ベースコーパスに未公開記事を 1 本差し込んで、条件を作る。

    ベースコーパスのベクトルはそのまま再利用し、
    差し込む記事のチャンクぶんだけをベクトル化して後ろに連結する。
    数百件を計算し直す必要はない。

    Args:
        base: build_base() が作ったベースコーパス。
        article: 差し込む未公開記事。
        url: 記事に与える URL。
            他社記事と形式を揃えるために必要。
            LLM は出典の見た目でも引用の判断を変えるため、
            URL やタイトルが無い文書だけ扱いが変わってしまうのを避ける。
        name: 条件の名前（"single" / "before" / "after"）。
    """
    target = Document(
        url=url,
        title=article.title,
        content=article.body,
        found_by_query=None,
        is_target=True,
    )

    # 他社記事と同じ掃除・分割の手順を通す。
    # 前処理が条件によって変わると、それ自体が検索順位の差を生んでしまう。
    target = clean_documents([target])[0]

    documents = base.documents + [target]
    target_index = len(documents) - 1

    # 差し込む記事のチャンクだけを作る。
    # doc_index を新しい記事の位置に付け替える必要があるため、
    # 単独で分割してから番号を振り直す。
    target_chunks = [
        Chunk(doc_index=target_index, chunk_index=c.chunk_index, text=c.text)
        for c in chunk_documents([target])
    ]

    # 差し込むぶんだけベクトル化する（キャッシュ名を渡さないので毎回計算する）。
    # 記事は毎回変わるためキャッシュしても当たらない。
    target_vectors = embed_documents([c.text for c in target_chunks])

    return Condition(
        name=name,
        documents=documents,
        chunks=base.chunks + target_chunks,
        vectors=np.vstack([base.vectors, target_vectors]),
        target_index=target_index,
        article=article,
    )

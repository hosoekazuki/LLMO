"""テキストをベクトルに変換するモジュール。

ベクトル検索は「意味の近さ」を数値で測る仕組み。
テキストを数百次元の数値の並び（ベクトル）に変換し、
その向きがどれだけ揃っているかで近さを判断する。

## なぜ手元のモデルを使うのか

疑似ウェブを作り直すたびにコーパス全体を埋め込むため、
API では無料枠の上限に当たって評価が完走しない。
手元で動かせば回数の制限が無く、同じ入力に対して常に同じベクトルが出る。
API のモデルは更新されるとベクトルが変わるので、再現性の面でも都合がよい。

記事の生成・回答の生成・指標の判定では引き続き Gemini を使う。
置き換えたのは埋め込みだけである。

## 文書とクエリで変換の仕方を変える

「電子署名の自動化とは？」という質問文と、それに答える記事本文は、
文章としては似ていない（片方は疑問文、片方は説明文）。
素朴に変換すると、質問文には別の質問文が最も近いと判定されてしまう。

ruri v3 は接頭辞でこれを区別する。文書には「検索文書: 」、
クエリには「検索クエリ: 」を付けると、質問と答えが近づくように変換される。
接頭辞は変換の直前にだけ付け、保存するチャンク本文には含めない。
本文は画面表示と引用の判定に使うため、汚さないようにする。
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from llmo.config import EMBEDDING, MODELS, STORAGE

# 読み込んだモデルを入れておく場所。
#
# モデルの読み込みには 20 秒ほどかかる。
# Streamlit は操作のたびに app.py を最初から実行し直すが、
# import 済みのモジュールは読み直されないため、
# ここに持たせておけば読み込みは 1 回で済む。
_model = None


def get_model():
    """埋め込みモデルを返す。最初の 1 回だけ読み込む。

    初回は Hugging Face からモデルを取得するため、
    ダウンロードの時間がかかる（約 130MB）。2 回目以降は
    ~/.cache/huggingface から読むので通信は発生しない。
    """
    global _model
    if _model is None:
        # import をここに置くのは、読み込みに数秒かかるため。
        # 埋め込みを使わない画面操作にまで待ち時間を広げない。
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODELS.embedding)
    return _model


def _embed(texts: list[str], prefix: str, verbose: bool = False) -> np.ndarray:
    """テキストをまとめてベクトル化する。

    Args:
        texts: 変換するテキスト。
        prefix: 先頭に付ける接頭辞（文書用かクエリ用か）。
        verbose: 進捗を出すか。

    Returns:
        (テキスト数, 次元数) の配列。長さは 1 に正規化済み。
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    if verbose:
        print(f"  {len(texts)} 件をベクトル化しています...", flush=True)

    # 正規化を必ず有効にする。
    # retrieve.py はコサイン類似度を内積 1 回で計算しており、
    # 長さが 1 に揃っていないと、意味の近さではなく
    # 文章の長さで順位がついてしまう。
    vectors = get_model().encode(
        [prefix + text for text in texts],
        normalize_embeddings=True,
        batch_size=EMBEDDING.batch_size,
        show_progress_bar=verbose,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_queries(queries: list[str]) -> np.ndarray:
    """検索クエリをベクトル化する。"""
    return _embed(queries, EMBEDDING.query_prefix)


def embed_documents(
    texts: list[str], cache_name: str | None = None, verbose: bool = False
) -> np.ndarray:
    """検索対象の文書（チャンク）をベクトル化する。

    cache_name を渡すと結果をファイルに保存し、次回はそれを読む。
    コーパスは固定されていて中身が変わらないため、毎回やり直す必要がない。

    Args:
        texts: チャンク本文のリスト。接頭辞は付けずに渡すこと。
        cache_name: 保存名。省略するとキャッシュを使わない。

    Returns:
        (チャンク数, 次元数) の配列。
    """
    if cache_name is None:
        return _embed(texts, EMBEDDING.document_prefix, verbose=verbose)

    cache_path = Path(STORAGE.corpus_dir) / f"{cache_name}.npy"
    fingerprint_path = Path(STORAGE.corpus_dir) / f"{cache_name}.fingerprint.json"

    # 入力テキストと変換の条件から指紋（ハッシュ値）を作る。
    # 指紋が一致したときだけキャッシュを使う。
    #
    # モデル名と接頭辞を含めているのが要点。
    # これらが変わるとベクトルの意味する空間そのものが変わるため、
    # 古いベクトルを新しいクエリのベクトルと突き合わせると、
    # 数字は出るのに意味の無い類似度になる。
    # 次元数が同じ別モデルに替えた場合、計算は成功してしまうので、
    # 指紋で弾かないと誤りに気づけない。
    fingerprint = {
        "model": MODELS.embedding,
        "document_prefix": EMBEDDING.document_prefix,
        "count": len(texts),
        "hash": hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest(),
    }

    if cache_path.exists() and fingerprint_path.exists():
        saved = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if saved == fingerprint:
            return np.load(cache_path)

        changed = [
            key
            for key in ("model", "document_prefix", "count", "hash")
            if saved.get(key) != fingerprint[key]
        ]
        print(
            f"  保存済みのベクトルは使えません（{'・'.join(changed)} が変わっています）。"
            "作り直します",
            flush=True,
        )

    vectors = _embed(texts, EMBEDDING.document_prefix, verbose=verbose)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, vectors)
    fingerprint_path.write_text(
        json.dumps(fingerprint, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return vectors

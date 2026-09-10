"""疑似ウェブ（コーパス）を構築するモジュール。

README の「疑似ウェブによる評価環境」の中核。

LLM に自律的にウェブ検索させると、まだ公開していない記事を検索結果に含められない。
そこで一度 Tavily で取得した結果を固定してコーパス化し、
そこに未公開記事を「入れた場合／入れない場合」でベクトル検索することで、
公開前に効果を比較できるようにする。

一度取得したら保存して固定するのが重要。
前後の比較で検索対象が変わってしまうと、記事の効果を測っていることにならない。
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from tavily import TavilyClient

from llmo.config import EVAL, STORAGE, get_tavily_api_key


@dataclass
class Document:
    """疑似ウェブに入る 1 件の文書。

    Tavily から取得した他社のページも、システムが作った未公開記事も、
    同じこの型で扱う。ベクトル検索の際に両者を区別なく並べるため、
    形式を揃えておく必要がある。
    """

    url: str
    title: str
    content: str

    # この文書がどのクエリで取れたか。取得元をたどるための記録用。
    # 未公開記事の場合は None。
    found_by_query: str | None = None

    # 未公開記事かどうかの目印。
    # 評価のときに「どれが自社記事か」を判定するのに使う。
    is_target: bool = False


def fetch_pages(queries: list[str], results_per_query: int | None = None) -> list[Document]:
    """クエリごとに Tavily で検索し、ページ本文を集めて返す。

    Args:
        queries: generate_queries() が作ったクエリのリスト。
        results_per_query: 1 クエリあたりの取得件数。省略時は config の値。

    Returns:
        URL で重複を除いた Document のリスト。これが疑似ウェブの材料になる。
    """
    if results_per_query is None:
        results_per_query = EVAL.tavily_results_per_query

    client = TavilyClient(api_key=get_tavily_api_key())

    # URL をキーにして重複を除く。
    # 複数のクエリで同じページが返るのは普通に起きるため。
    # dict を使うことで、取得した順序も保たれる。
    collected: dict[str, Document] = {}

    for query in queries:
        response = client.search(
            query=query,
            max_results=results_per_query,
            # ページ本文まで取得する。
            # これが無いと数行のスニペットしか手に入らず、
            # 疑似ウェブが薄くなって引用の再現性が落ちる。
            include_raw_content=True,
        )

        for item in response.get("results", []):
            url = item.get("url", "")
            if not url or url in collected:
                continue

            # raw_content が本文。取れないサイトもあるので、
            # その場合は content（要約スニペット）で代替する。
            body = item.get("raw_content") or item.get("content") or ""
            body = body.strip()

            # 本文の取得に失敗した文書を捨てる。
            # 数十字しか無いものが混ざると、中身が無いのに検索候補の枠を潰す。
            if len(body) < EVAL.min_content_chars:
                continue

            collected[url] = Document(
                url=url,
                title=item.get("title", ""),
                content=body,
                found_by_query=query,
            )

    return list(collected.values())


def save_corpus(documents: list[Document], name: str) -> Path:
    """コーパスを JSON ファイルに保存する。

    保存する目的は、検索対象を固定すること。
    評価のたびに Tavily を叩き直すと、そのときどきで結果が変わり、
    「差し込んだ記事が変わったから変わった」のか「検索対象が変わったから変わった」のか
    区別できなくなる。

    Args:
        documents: 保存する文書。
        name: 保存名。トピックごとに分けるための識別子。

    Returns:
        保存したファイルのパス。
    """
    path = Path(STORAGE.corpus_dir) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    # ensure_ascii=False で日本語をそのまま書き出す（中身を目で確認できるように）。
    path.write_text(
        json.dumps([asdict(d) for d in documents], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_corpus(name: str) -> list[Document]:
    """保存したコーパスを読み込む。

    2 回目以降の評価では Tavily を叩かずにこれを読む。
    """
    path = Path(STORAGE.corpus_dir) / f"{name}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Document(**item) for item in raw]


def meta_path(name: str) -> Path:
    """コーパスのメタ情報ファイルの場所を返す。"""
    return Path(STORAGE.corpus_dir) / f"{name}.meta.json"


def save_meta(
    name: str,
    topic: str,
    description: str,
    queries: list[str],
    num_documents: int,
) -> None:
    """コーパスを何から作ったかを保存する。

    本文の JSON には、どのクエリで取れたかが文書ごとにしか残らない。
    1 件も結果を返さなかったクエリは痕跡が消えるため、
    承認したクエリの一覧はここに丸ごと持つ。

    このクエリは評価でもそのまま使う。評価のたびに作り直すと、
    同じトピックでも実行ごとに分母が揺れてしまうためである。
    """
    path = meta_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "topic": topic,
                "description": description,
                "queries": queries,
                # 取得できた記事の数。
                # 画面で件数を出すためだけに本文の JSON を読むと、
                # 数百 KB の解析が操作のたびに走るので、ここに写しておく。
                "num_documents": num_documents,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_meta(name: str) -> dict | None:
    """コーパスのメタ情報を読む。無ければ None。

    メタ情報を持たない古いコーパスがありうるので、
    呼び出し側が「作り直しが要る」と判断できるように None を返す。
    """
    path = meta_path(name)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def has_corpus(name: str) -> bool:
    """保存済みのコーパスがあるか。"""
    return (Path(STORAGE.corpus_dir) / f"{name}.json").exists()


def list_corpora() -> list[str]:
    """保存済みのコーパスの名前を返す。

    画面で「どの疑似ウェブを見るか」を選ばせるために使う。
    app.py からディレクトリを直接走査させると、
    保存先の決め方が画面側に漏れてしまうため、ここに置く。

    Returns:
        保存名のリスト（名前順）。1 つも無ければ空リスト。
    """
    directory = Path(STORAGE.corpus_dir)
    if not directory.exists():
        return []
    # .npy や .fingerprint.json、.meta.json は除き、本文の .json だけを拾う。
    #
    # Path.stem は末尾の拡張子しか落とさないため、
    # "esign.meta.json" の stem は "esign.meta" になる。
    # 名前で弾かずに stem だけで判定すると、実体の無いコーパスが一覧に並ぶ。
    return sorted(
        path.stem
        for path in directory.glob("*.json")
        if not path.name.endswith((".fingerprint.json", ".meta.json"))
    )

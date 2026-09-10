"""記事を書く主体である企業・製品の情報を扱うモジュール。

このシステムは特定の 1 社のためのものではないので、
企業名や製品名をコードやプロンプトに直接書かない。
JSON ファイルとして外に置き、差し替えるだけで別の企業に使えるようにする。

この情報は 2 か所で使われる。

1. 記事生成: 自社製品に触れた、実際のオウンドメディアに近い記事を書くため。
2. 評価: Mention Rate（企業が回答に出現した割合）や
   Top Recommendation Rate（最もおすすめと紹介された割合）は、
   回答の中に企業名が出たかどうかで判定する。その照合対象になる。
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# 企業情報の既定の置き場所。
DEFAULT_BRAND_PATH = "data/brand.json"


@dataclass
class Brand:
    """記事を出す企業と、その製品の情報。"""

    # 企業名。回答中にこの名前が出たかで Mention Rate を判定する。
    company: str

    # 製品・サービス名。企業名と別に持つのは、
    # 回答では製品名だけが挙がることが多いため
    # （「クラウドサインが便利」のように、企業名は出ないことがある）。
    product: str

    # 事業内容の説明。記事を書くときの前提として LLM に渡す。
    description: str

    # 自社サイトのドメイン。未公開記事に与える URL の組み立てに使う。
    domain: str

    # 製品の強み。記事の中で自然に触れさせるために渡す。
    strengths: list[str] = field(default_factory=list)

    # 想定読者。記事のトーンや前提知識の水準を決めるのに使う。
    audience: str = ""

    # 表記ゆれの一覧。
    # 回答に「サインフロー」「SignFlow」のどちらで出ても
    # 言及されたと判定できるようにする。
    aliases: list[str] = field(default_factory=list)

    @property
    def mention_terms(self) -> list[str]:
        """回答の中を探すときに使う語の一覧。

        企業名・製品名・表記ゆれをまとめたもの。
        このどれかが回答に含まれていれば「言及された」と判定する。
        """
        terms = [self.company, self.product, *self.aliases]
        # 空文字と重複を除く。
        return list(dict.fromkeys(term for term in terms if term))

    def article_url(self, slug: str) -> str:
        """未公開記事に与える URL を組み立てる。

        他社記事と形式を揃えるために必要。
        LLM は出典の見た目でも引用の判断を変えるため、
        URL の無い文書だけ扱いが変わってしまうのを避ける。
        """
        return f"https://{self.domain}/blog/{slug}"


def load_brand(path: str = DEFAULT_BRAND_PATH) -> Brand:
    """企業情報を JSON ファイルから読み込む。

    Args:
        path: 読み込むファイル。別の企業で評価したいときはここを差し替える。
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return Brand(**raw)


def validate_brand(brand: Brand) -> str | None:
    """保存してよい内容かを調べる。

    ここを検証するのは、誤りが静かに効いてしまうためである。
    企業名と製品名が空だと mention_terms が空になり、
    言及率は「一度も言及されなかった」ではなく
    「照合する語が無い」という理由で常に 0% になる。
    画面には同じ 0% としか出ないので、気づく手がかりが無い。

    Returns:
        問題があればその説明。無ければ None。
    """
    if not brand.company.strip():
        return "企業名は必須です。空だと回答に自社が出たかを判定できません。"
    if not brand.product.strip():
        return "製品名は必須です。空だと回答に自社が出たかを判定できません。"
    if not brand.description.strip():
        return "事業内容は必須です。記事を書くときの前提として渡します。"

    domain = brand.domain.strip()
    if not domain:
        return "ドメインは必須です。未公開記事に与える URL の組み立てに使います。"
    # article_url() が "https://{domain}/blog/{slug}" を組み立てるので、
    # ここにスキームや余分なスラッシュが入ると壊れた URL が出典として LLM に渡る。
    if "://" in domain or "/" in domain or " " in domain:
        return (
            "ドメインは「signflow.co.jp」のように、"
            "https:// やスラッシュを含めずに書いてください。"
        )
    return None


def save_brand(brand: Brand, path: str = DEFAULT_BRAND_PATH) -> None:
    """企業情報を JSON ファイルに保存する。

    Raises:
        ValueError: 内容に問題があるとき。
    """
    problem = validate_brand(brand)
    if problem:
        raise ValueError(problem)

    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False で日本語をそのまま書き出す。
    # ファイルを直接開いて確認・編集できる状態を保つため。
    file.write_text(
        json.dumps(asdict(brand), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

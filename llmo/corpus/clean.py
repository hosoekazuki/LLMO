"""取得した本文から、記事本文でない部分を取り除くモジュール。

Tavily の raw_content には、記事本文だけでなく、
ナビゲーションメニュー・パンくずリスト・フッターのリンク集などが混ざって入ってくる。

これを放置すると、ナビだけでできたチャンクが検索候補に混ざり、
記事本文と枠を奪い合う。本物の検索エンジンはナビと本文を区別して扱うため、
掃除しないほうが実際のウェブから離れてしまう。

チャンク分割の前に通す。
"""

import re

# マークダウンのリンク。
# 表示テキスト部分に角括弧が入れ子になることがあるため（例: "[[電子帳票]活文](/x/)"）、
# 内側の [...] を 1 段だけ許す形にしている。
_LINK = re.compile(r"!?\[((?:[^\[\]]|\[[^\[\]]*\])*)\]\([^)]*\)")

# URL だけの行。
_URL_ONLY_LINE = re.compile(r"^\s*!?\[?https?://\S+\]?\s*$")

# 1 行にリンクがこの数以上あればナビゲーション行とみなして行ごと捨てる。
# 記事本文の 1 行に 2 つ以上のリンクが並ぶことは稀だが、
# メニューやフッターのリンク集は必ずこの形になる。
_NAV_LINK_COUNT = 2

# リンク置換を繰り返す上限。入れ子は深くても数段なので、この回数で足りる。
# 上限を設けているのは、万一置換が収束しない入力でも止まるようにするため。
_MAX_LINK_PASSES = 5

# 記法の外に裸で置かれた URL。上の置換で取りきれなかったものを落とす。
_BARE_URL = re.compile(r"https?://\S+")

# リンクの表示テキストが改行を跨いでいた場合に残る、閉じ側だけの残骸。
# 例: "仕組みとメリットを徹底解説](/column/xxx/)" の "](/column/xxx/)" 部分。
_LINK_REMNANT = re.compile(r"\]\([^)]*\)")

# 記事本文としては短すぎる行の閾値（文字数）。
# メニュー項目やボタンのラベルはたいてい 10 文字未満。
_MIN_LINE_CHARS = 10


def clean_text(text: str) -> str:
    """本文からナビゲーション類を取り除く。

    行単位で判定する。ナビゲーションは 1 行 1 項目で並ぶことが多く、
    行で見るのが最も素直に効くため。

    Args:
        text: Tavily から取得した生の本文。

    Returns:
        掃除後の本文。
    """
    kept_lines: list[str] = []

    for line in text.split("\n"):
        stripped = line.strip()

        if not stripped:
            continue

        # URL がそのまま置かれただけの行は本文ではない。
        if _URL_ONLY_LINE.match(stripped):
            continue

        # リンクが 2 つ以上並ぶ行はナビゲーションとみなし、行ごと捨てる。
        if len(_LINK.findall(stripped)) >= _NAV_LINK_COUNT:
            continue

        # リンクが 1 つだけの行は本文の一部とみなし、
        # 表示テキストだけ残して URL を捨てる。
        # URL の文字列がベクトルに乗ると、意味の計算を濁らせるため。
        #
        # 画像をリンクで包んだ "[![alt](画像URL)](リンクURL)" のような入れ子があるため、
        # 変化しなくなるまで繰り返す（1 回だけだと内側が残る）。
        for _ in range(_MAX_LINK_PASSES):
            replaced = _LINK.sub(r"\1", stripped).strip()
            if replaced == stripped:
                break
            stripped = replaced

        # 上で取りきれなかった裸の URL と、閉じ側だけ残ったリンクの残骸を落とす。
        stripped = _LINK_REMNANT.sub("", stripped)
        stripped = _BARE_URL.sub("", stripped).strip()

        # 見出し記号などを落としてから長さを見る。
        # "## 電子署名とは" は見出しとして意味があるので、記号だけ外して残す。
        without_marks = stripped.lstrip("#*->|").strip()

        if len(without_marks) < _MIN_LINE_CHARS:
            continue

        kept_lines.append(without_marks)

    return "\n".join(kept_lines)


def clean_documents(documents: list) -> list:
    """コーパス全体の本文を掃除して、新しいリストを返す。

    元の Document を書き換えず、掃除後の本文を持つ複製を返す。
    元データを残しておけば、掃除が過剰だったときに元に戻せる。
    """
    from dataclasses import replace

    return [replace(doc, content=clean_text(doc.content)) for doc in documents]

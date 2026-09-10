"""生成された記事に、検索クエリが転記されていないかを調べるモジュール。

## 何を見るか

記事の見出しと、評価に使う検索クエリを突き合わせる。
見出しを見るのは、そこにクエリが写されると影響が最も大きいためである。
見出しはチャンクの先頭に来ることが多く、そのチャンクは
クエリで検索したときにほぼ最大の類似度を返す。

## なぜ形態素解析を使わないか

「クエリの構成語がそのままの並びで含まれるか」を厳密に見るには
日本語の分かち書きが要るが、辞書と依存関係が増えるわりに、
ここで判定したいのは「写したかどうか」でしかない。

写した文字列は必ず長い共通部分文字列として現れる。
逆に、自分の言葉で書き直した見出しは、助詞や語尾が一致しても
共通部分は短い断片にとどまる。
そのため最長共通部分文字列の長さで十分に見分けられる。

## 限界

意味の近さによる有利は検出できない。
「稟議が遅い原因」と「承認が滞留する要因」は文字列としてはほとんど
重ならないが、埋め込み空間では近い。
ここで取り除けるのは文字列の直接転記だけである。
"""

import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher

from llmo.config import EVAL

# マークダウンの見出し行。
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)

# 比較の前に落とす記号類。
# 「〜とは？」と「〜とは」を別物として扱うと、
# 記号を 1 つ消すだけで検出を逃れられてしまう。
_NOISE = re.compile(r"[\s　。、，．！？!?「」『』（）()｛｝\[\]【】〜~・:：;；\-—ー]+")


@dataclass
class LeakageFinding:
    """転記が疑われる、クエリと見出しの組 1 件。"""

    query: str
    heading: str

    # 見出しにクエリがそのまま含まれる（またはその逆）。
    # これが True なら言い訳の余地がない転記である。
    verbatim: bool

    # 最長共通部分文字列の長さ ÷ 短いほうの長さ。
    # 「構成語がそのままの並びで含まれるか」に対応する。
    longest_common_ratio: float

    # 文字列全体としての似かより（difflib）。
    similarity: float

    @property
    def score(self) -> float:
        """閾値と比べる値。2 つの指標のうち大きいほうを採る。"""
        return max(self.longest_common_ratio, self.similarity)


@dataclass
class LeakageReport:
    """1 つの記事についての検証結果。"""

    threshold: float

    # 記事から取り出した見出し。
    headings: list[str] = field(default_factory=list)

    # 閾値を超えた組。空なら転記は検出されていない。
    findings: list[LeakageFinding] = field(default_factory=list)

    # 閾値を超えなかったものも含めた、最も高かった値。
    # 閾値の妥当性を後から見直すために残す。
    max_score: float = 0.0

    # 何回作り直した後の結果か。作り直していなければ 0。
    regenerations: int = 0

    @property
    def has_leakage(self) -> bool:
        return bool(self.findings)

    @property
    def has_verbatim(self) -> bool:
        """そのままの転記があったか。"""
        return any(f.verbatim for f in self.findings)

    def to_dict(self) -> dict:
        """保存できる形にする。"""
        return {
            "threshold": self.threshold,
            "headings": self.headings,
            "max_score": self.max_score,
            "regenerations": self.regenerations,
            "findings": [asdict(f) for f in self.findings],
        }


def extract_headings(body: str) -> list[str]:
    """マークダウン本文から見出しの文字列を取り出す。"""
    return [m.group(1).strip() for m in _HEADING.finditer(body)]


def _normalize(text: str) -> str:
    """比較用に記号と空白を落とす。"""
    return _NOISE.sub("", text)


def _longest_common_ratio(left: str, right: str) -> float:
    """最長共通部分文字列の長さを、短いほうの長さで割った値。

    短いほうで割るのは、長い見出しの一部にクエリが丸ごと埋め込まれた場合を
    見逃さないため。全体の長さで割ると、周りに文章を足すだけで値が下がる。
    """
    if not left or not right:
        return 0.0
    match = SequenceMatcher(None, left, right, autojunk=False).find_longest_match(
        0, len(left), 0, len(right)
    )
    return match.size / min(len(left), len(right))


def check(body: str, queries: list[str], threshold: float | None = None) -> LeakageReport:
    """記事の見出しに、クエリが転記されていないかを調べる。

    Args:
        body: 生成された記事の本文（マークダウン）。
        queries: 評価に使う検索クエリ。
        threshold: 一致率の上限。省略時は config の値。

    Returns:
        検証結果。閾値を超えた組が findings に入る。
    """
    if threshold is None:
        threshold = EVAL.leakage_threshold

    headings = extract_headings(body)
    report = LeakageReport(threshold=threshold, headings=headings)

    for query in queries:
        normalized_query = _normalize(query)
        for heading in headings:
            normalized_heading = _normalize(heading)

            verbatim = (
                normalized_query in normalized_heading
                or normalized_heading in normalized_query
            )
            common = _longest_common_ratio(normalized_query, normalized_heading)
            similarity = SequenceMatcher(
                None, normalized_query, normalized_heading, autojunk=False
            ).ratio()

            finding = LeakageFinding(
                query=query,
                heading=heading,
                verbatim=verbatim,
                longest_common_ratio=round(common, 3),
                similarity=round(similarity, 3),
            )
            report.max_score = max(report.max_score, finding.score)

            # そのままの転記は、値の大小にかかわらず必ず記録する。
            if verbatim or finding.score >= threshold:
                report.findings.append(finding)

    # 疑いの強い順に並べる。画面には上から出す。
    report.findings.sort(key=lambda f: (f.verbatim, f.score), reverse=True)
    report.max_score = round(report.max_score, 3)
    return report

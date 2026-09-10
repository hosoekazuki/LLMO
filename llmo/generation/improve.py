"""評価結果をもとに、記事の弱い部分を分析して直すモジュール。

weakness.py が選んだ「弱いクエリ」を受け取り、2 段階で処理する。

1. analyze_weakness(): なぜそのクエリに対して弱いのかを分析する。
2. revise_sections(): 分析結果をもとに、該当する節だけを書き直す。

## なぜ記事全文を書き直さないのか

全文を渡して「良くしてください」と頼むと、すでに検索で拾われている節まで
変わってしまう。そうなると評価が動いても、どの変更が効いたのかが分からない。
節を単位にすれば、直した箇所と指標の変化を対応づけられる。
既に強い部分を壊さずに済み、LLM が余計に手を入れる余地も減る。

## なぜ rules.py の指示を差し込まないのか

rules.py は「記事全体をどう構成するか」の指示であり、
見出しの本数のような記事単位の条件を含む。
節を 1 つだけ渡す場面でそれを読ませると、記事全体を作り直そうとする。
ここでは節単位で意味を持つ条件だけを、このモジュールのプロンプトに書く。

## ハルシネーションについて

この処理は「情報が足りない」と判定した箇所に文章を足させるため、
記事生成の中で最も作り話が生まれやすい。3 段構えで抑える。

1. 製品の事実として渡すのは Brand（data/brand.json）だけにする。
2. プロンプトで、作ってはいけない情報の種類を具体的に列挙する。
3. 生成後に check_new_facts() で、新しく現れた数値を機械的に拾って警告する。

3 は検出であって防止ではない。止めずに警告に留めているのは、
一般論としての数値（法定保存期間など）まで弾くと書けなくなるためである。
判断は画面を見る人に委ねる。
"""

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from llmo.config import MODELS
from llmo.core.brand import Brand
from llmo.core.gemini import generate
from llmo.evaluation.weakness import WeakQuery
from llmo.generation.writer import Article

# 見出し行（マークダウン）。leakage.py と同じ形にしている。
_HEADING_LINE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")

# 節の照合に使うとき、意味を持たない記号と空白を落とすためのもの。
_NOISE = re.compile(r"[\s　。、，．！？!?「」『』（）()｛｝\[\]【】〜~・:：;；\-—ー]+")


# ---------------------------------------------------------------------------
# 節の分割
# ---------------------------------------------------------------------------

@dataclass
class Section:
    """記事の節 1 つ。

    body には見出し行そのものも含める。
    分割して繋ぎ直すときに、見出しを別に管理すると復元でずれるため。
    """

    # 見出しの文字列（記号と空白を除いたもの）。
    # 最初の見出しより前にある前書きは None になる。
    heading: str | None

    # 見出し行を含む、この節の本文全体。
    body: str

    @property
    def is_preamble(self) -> bool:
        """最初の見出しより前の部分か。"""
        return self.heading is None


def split_sections(body: str) -> list[Section]:
    """記事本文を見出しで節に分割する。

    見出しの階層は区別しない。'###' で始まる小見出しも節の切れ目とする。
    分析側が「どの見出しを直すか」を見出しの文字列で指すので、
    階層を意識させると指定が曖昧になる。

    Returns:
        本文中の順に並んだ節。見出しが 1 つも無ければ、全体で 1 つの節になる。
    """
    sections: list[Section] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in body.splitlines():
        match = _HEADING_LINE.match(line)
        if match:
            # ここまでを 1 つの節として確定する。
            # 空の前書き（記事がいきなり見出しで始まる場合）は作らない。
            if current_lines or current_heading is not None:
                sections.append(
                    Section(heading=current_heading, body="\n".join(current_lines))
                )
            current_heading = match.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines or current_heading is not None:
        sections.append(Section(heading=current_heading, body="\n".join(current_lines)))

    return sections


def join_sections(sections: list[Section]) -> str:
    """節を元の記事の形に繋ぎ直す。"""
    return "\n".join(section.body for section in sections)


def _normalize(text: str) -> str:
    """節の照合用に、記号と空白を落とす。"""
    return _NOISE.sub("", text)


def find_section(sections: list[Section], heading: str) -> int | None:
    """指定された見出しに対応する節の位置を返す。

    LLM が返す見出しは、記号や助詞が原文とわずかに違うことがある。
    完全一致だけで探すと、実在する節を「見つからない」と判定して
    新しい節を無駄に増やしてしまう。
    そこで記号を落とした一致、部分一致の順に緩めて探す。

    Returns:
        節の添字。見つからなければ None。
    """
    target = _normalize(heading)
    if not target:
        return None

    candidates = [
        (index, _normalize(section.heading))
        for index, section in enumerate(sections)
        if section.heading
    ]

    for index, normalized in candidates:
        if normalized == target:
            return index

    for index, normalized in candidates:
        if target in normalized or normalized in target:
            return index

    return None


# ---------------------------------------------------------------------------
# 改善点の分析
# ---------------------------------------------------------------------------

class _AnalysisOutput(BaseModel):
    """分析結果を JSON で受け取るための型。"""

    problem: str = Field(
        description="このクエリに対して記事が弱い理由を、記事の内容に即して一文で"
    )
    missing_information: list[str] = Field(
        description="記事に不足している情報の種類を 2〜5 個。"
        "「代理承認の手順」のように具体的な項目名で挙げる"
    )
    improvement_instruction: str = Field(
        description="どの情報をどう書き足すかの指示。"
        "「もっと詳しく」のような抽象的な指示にしないこと"
    )
    target_section: str = Field(
        description="修正すべき節の見出し。"
        "既存の見出しから 1 つ選ぶ。どれも当てはまらない場合だけ新しい見出しを書く"
    )


@dataclass
class Analysis:
    """クエリ 1 本についての改善点の分析結果。"""

    query: str

    # 弱点の種類（weakness.py の定数）。修正プロンプトの方針を変えるのに使う。
    kind: str

    problem: str
    missing_information: list[str]
    improvement_instruction: str
    target_section: str

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "kind": self.kind,
            "problem": self.problem,
            "missing_information": self.missing_information,
            "improvement_instruction": self.improvement_instruction,
            "target_section": self.target_section,
        }


_ANALYSIS_PROMPT = """あなたは、生成 AI の検索結果に記事が引用されるかを分析する担当者です。

ある記事を疑似的なウェブ検索に載せて評価したところ、
以下の検索クエリに対して弱いという結果が出ました。
なぜ弱いのかを、記事の内容に即して分析してください。

## 対象の検索クエリ
{query}

## 測定された結果
- 弱点: {weakness_label}
- 判定の理由: {weakness_reason}
- ベクトル検索での順位: {search_rank}
- 回答に引用されたか: {cited}
- 回答本文に自社名・製品名が出たか: {mentioned}
- 回答の主張のうち自社記事が支えた割合: {grounding_rate}

## 評価した記事
### タイトル
{title}

### 見出しの一覧
{headings}

### このクエリに最も近かった、自社記事の部分
{own_chunks}

## 同じクエリで検索上位に入った他社記事の該当部分
{rival_chunks}

## 自社について
- 企業名: {company}
- 製品名: {product}
- 事業内容: {description}

## 分析の条件
- 「なぜこの記事はこのクエリに対して弱いのか」を、記事の実際の記述に即して述べること。
  一般論ではなく、上に示した記事の内容を根拠にすること。
- 不足している情報は、項目名の形で具体的に挙げること。
  「情報が少ない」「説明が浅い」のような書き方をしないこと。
- 改善の指示は、何をどう書き足すかが分かる形にすること。
  「もっと詳しく書く」「充実させる」のような抽象的な指示を書かないこと。
- 修正すべき節は、上の「見出しの一覧」から 1 つ選ぶこと。
  クエリが扱う話題を書ける節がどこにも無い場合に限り、新しい見出しを書くこと。
- 他社記事に書かれている情報のうち、このクエリに答えるうえで
  自社記事に欠けているものがあれば指摘すること。
  ただし他社固有の事実（他社の実績・料金・機能）を写すことは求めないこと。
"""


def _format_chunks(chunks: list[str], empty: str) -> str:
    """チャンクをプロンプトに埋め込む形に整える。"""
    if not chunks:
        return empty
    return "\n\n---\n\n".join(chunks)


def _format_rate(value: float | None) -> str:
    return "測定されず（引用されなかったため）" if value is None else f"{value * 100:.0f}%"


def analyze_weakness(
    article: Article,
    weak: WeakQuery,
    brand: Brand,
    own_chunks: list[str],
    rival_chunks: list[str],
) -> Analysis:
    """弱いクエリ 1 本について、記事のどこをどう直すべきかを分析する。

    Args:
        article: 評価した記事。
        weak: weakness.identify_weak_queries() が選んだクエリ。
        brand: 自社の情報。
        own_chunks: そのクエリに最も近かった自社記事のチャンク。
            検索で圏外だった場合も、記事のどこが最も近かったかは計算できる。
        rival_chunks: 同じクエリで上位に入った他社記事のチャンク。

    Returns:
        構造化された分析結果。
    """
    sections = split_sections(article.body)
    headings = [s.heading for s in sections if s.heading]

    prompt = _ANALYSIS_PROMPT.format(
        query=weak.query,
        weakness_label=weak.label,
        weakness_reason=weak.reason,
        search_rank=f"{weak.search_rank} 位" if weak.search_rank else "圏外",
        cited="はい" if weak.cited else "いいえ",
        mentioned="はい" if weak.mentioned else "いいえ",
        grounding_rate=_format_rate(weak.grounding_rate),
        title=article.title,
        headings="\n".join(f"- {h}" for h in headings) or "（見出しなし）",
        own_chunks=_format_chunks(own_chunks, "（該当する部分がありません）"),
        rival_chunks=_format_chunks(rival_chunks, "（他社記事が取得できませんでした）"),
        company=brand.company,
        product=brand.product,
        description=brand.description,
    )

    response = generate(
        model=MODELS.generation,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": _AnalysisOutput,
        },
    )
    output: _AnalysisOutput = response.parsed

    return Analysis(
        query=weak.query,
        kind=weak.kind,
        problem=output.problem,
        missing_information=list(output.missing_information),
        improvement_instruction=output.improvement_instruction,
        target_section=output.target_section.strip(),
    )


# ---------------------------------------------------------------------------
# 記事の修正
# ---------------------------------------------------------------------------

class _SectionOutput(BaseModel):
    """書き直した節を JSON で受け取るための型。"""

    heading: str = Field(description="節の見出し（本文には含めず、文字列だけ）")
    body: str = Field(description="見出しを除いた節の本文（マークダウン）")


@dataclass
class SectionEdit:
    """節 1 つの修正の記録。

    どの節を、どのクエリのために、どう直したかを残す。
    指標が動いたときに、どの修正が効いたのかを後から辿るために使う。
    """

    heading: str

    # 新しく作った節か、既存の節を書き直したか。
    is_new: bool

    # この修正の対象になったクエリ。1 つの節に複数が集まることがある。
    queries: list[str] = field(default_factory=list)

    before: str = ""
    after: str = ""

    def to_dict(self) -> dict:
        return {
            "heading": self.heading,
            "is_new": self.is_new,
            "queries": self.queries,
            "before": self.before,
            "after": self.after,
        }


# 修正で守らせる条件。作り話を防ぐ部分と、LLMO の観点をまとめている。
#
# 禁止事項を種類で列挙しているのは、「嘘を書かないこと」とだけ書いても
# 効きが弱いためである。実際に出てきやすいものを名指しする。
_REVISION_RULES = """### 事実の扱い（最優先。他のどの条件よりも優先する）
- 自社製品について書いてよいのは、上の「自社について」に書かれていることだけです。
- 次のものは**絶対に書かないでください**。上に書かれていない限り、存在しません。
  - 導入企業数、導入社数（「○○社が導入」）
  - 導入事例、顧客名
  - 改善率、削減率、短縮率（「○○%削減」「○○倍」）
  - 独自の調査結果、アンケート結果、統計
  - 料金、価格、プラン名
  - 上に挙がっていない機能
  - 特定の法令・制度への対応をうたう記述
  - 他社サービスとの連携
- 数字を書きたいが根拠が無い場合は、数字を書かないでください。
- 他社記事に書かれていた事実を、自社の事実として書き写さないでください。
- 断定できないことは「一般的には〜」「〜という方法が考えられます」のように
  一般論として書いてください。

### 書き方（この節が単体で検索・引用されることを前提にする）
- 対象の検索クエリに対する答えが、この節の中だけで完結すること。
  この節だけが切り出されて生成 AI に渡されても、意味が通るようにします。
- 結論を節の最初の 1〜2 文に書くこと。
- 手順・条件・場合分けがあるものは、具体的に書き出すこと。
- 「上記の」「前述の」「以下のように」など、節の外を指す言葉を使わないこと。
- 製品名（{product}）を書くときは省略せず、そのまま書くこと。
  ただし製品名を出す回数を増やすことを目的にしないこと。
- 見出しは、この節の結論を含んだ文にすること。
- **検索クエリの語句をそのまま並べた見出しにしないこと。**
  クエリの文字列を写すと、記事の中身とは無関係に検索で上位に入ってしまい、
  評価が成り立たなくなります。自分の言葉で書いてください。

### 変更の範囲
- 元の節に書かれている内容を消さないこと。書き足すことを中心にします。
- 元の節が扱っている話題から大きく離れないこと。
- 節の分量は、元の節と同程度から 1.5 倍程度までにすること。"""


_REVISE_PROMPT = """あなたは BtoB SaaS 企業のオウンドメディアの編集者です。
記事の 1 つの節を、生成 AI の検索結果に引用されやすくなるように書き直してください。

## この節が答えるべき検索クエリ
{queries}

## 評価で分かっていること
{findings}

## 書き直す節
{section}

## 自社について
- 企業名: {company}
- 製品名: {product}
- 事業内容: {description}
- 製品の強み:
{strengths}

## 想定読者
{audience}

## 条件

""" + _REVISION_RULES + """

## 出力
見出しと本文を分けて返してください。本文に見出し行を含めないでください。
"""


_NEW_SECTION_PROMPT = """あなたは BtoB SaaS 企業のオウンドメディアの編集者です。
既存の記事に、新しい節を 1 つ書き足してください。

## この節が答えるべき検索クエリ
{queries}

## 評価で分かっていること
{findings}

## 追加する節の見出し（案）
{heading}

## 記事のタイトル
{title}

## 記事にすでにある見出し
{headings}

## 自社について
- 企業名: {company}
- 製品名: {product}
- 事業内容: {description}
- 製品の強み:
{strengths}

## 想定読者
{audience}

## 条件

""" + _REVISION_RULES + """

- すでにある節と内容が重ならないようにすること。
- 分量は 1000〜1500 字程度にすること。

## 出力
見出しと本文を分けて返してください。本文に見出し行を含めないでください。
"""


def _format_findings(analyses: list[Analysis]) -> str:
    """同じ節に集まった分析結果を、プロンプトに埋め込む形に整える。"""
    blocks = []
    for analysis in analyses:
        missing = "\n".join(f"  - {m}" for m in analysis.missing_information)
        blocks.append(
            f"- 対象クエリ: {analysis.query}\n"
            f"- 問題: {analysis.problem}\n"
            f"- 不足している情報:\n{missing}\n"
            f"- 改善の指示: {analysis.improvement_instruction}"
        )
    return "\n\n".join(blocks)


def _format_strengths(strengths: list[str]) -> str:
    return "\n".join(f"  - {s}" for s in strengths)


def _brand_fields(brand: Brand) -> dict:
    """両方のプロンプトで共通の差し込み値。"""
    return {
        "company": brand.company,
        "product": brand.product,
        "description": brand.description,
        "strengths": _format_strengths(brand.strengths),
        "audience": brand.audience,
    }


def _generate_section(prompt: str) -> _SectionOutput:
    response = generate(
        model=MODELS.generation,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": _SectionOutput,
        },
    )
    return response.parsed


def revise_sections(
    article: Article,
    analyses: list[Analysis],
    brand: Brand,
    progress=None,
) -> tuple[Article, list[SectionEdit]]:
    """分析結果をもとに、該当する節だけを書き直した記事を作る。

    同じ節を指す分析は 1 回にまとめて渡す。
    節ごとに別々に書き直すと、後の修正が前の修正を上書きしてしまう。

    タイトルは変えない。
    タイトルはユーザーが選んだものであり、
    ここで変えると改善前後で比べているものが別の記事になる。

    Args:
        article: 修正前の記事。
        analyses: analyze_weakness() の結果。
        brand: 自社の情報。作ってよい事実の範囲を決める。
        progress: 経過を書き出す関数（省略可）。

    Returns:
        (修正後の記事, 修正した節の記録) の組。
    """
    sections = split_sections(article.body)

    # 同じ節を指す分析をまとめる。
    # 既存の節に対応づかないものは、新しい節としてまとめて扱う。
    by_index: dict[int, list[Analysis]] = {}
    new_sections: dict[str, list[Analysis]] = {}

    for analysis in analyses:
        index = find_section(sections, analysis.target_section)
        if index is None:
            new_sections.setdefault(analysis.target_section, []).append(analysis)
        else:
            by_index.setdefault(index, []).append(analysis)

    edits: list[SectionEdit] = []
    brand_fields = _brand_fields(brand)

    # --- 既存の節を書き直す ------------------------------------------------
    for index, group in by_index.items():
        section = sections[index]
        if progress:
            progress(f"節を書き直しています: {section.heading}")

        output = _generate_section(
            _REVISE_PROMPT.format(
                queries="\n".join(f"- {a.query}" for a in group),
                findings=_format_findings(group),
                section=section.body,
                **brand_fields,
            )
        )

        # 見出しの階層を元の節と揃える。
        # LLM が返す見出しは階層を持たないので、元の '#' の数を使い回す。
        match = _HEADING_LINE.match(section.body.splitlines()[0])
        level = match.group(1) if match else "##"
        new_body = f"{level} {output.heading.strip()}\n\n{output.body.strip()}"

        edits.append(
            SectionEdit(
                heading=output.heading.strip(),
                is_new=False,
                queries=[a.query for a in group],
                before=section.body,
                after=new_body,
            )
        )
        sections[index] = Section(heading=output.heading.strip(), body=new_body)

    # --- 新しい節を足す ----------------------------------------------------
    headings = [s.heading for s in sections if s.heading]
    for heading, group in new_sections.items():
        if progress:
            progress(f"節を追加しています: {heading}")

        output = _generate_section(
            _NEW_SECTION_PROMPT.format(
                queries="\n".join(f"- {a.query}" for a in group),
                findings=_format_findings(group),
                heading=heading,
                title=article.title,
                headings="\n".join(f"- {h}" for h in headings) or "（見出しなし）",
                **brand_fields,
            )
        )

        new_body = f"## {output.heading.strip()}\n\n{output.body.strip()}"
        edits.append(
            SectionEdit(
                heading=output.heading.strip(),
                is_new=True,
                queries=[a.query for a in group],
                before="",
                after=new_body,
            )
        )
        sections.append(Section(heading=output.heading.strip(), body=new_body))

    revised = Article(
        title=article.title,
        body=join_sections(sections),
        style="improved",
    )
    return revised, edits


# ---------------------------------------------------------------------------
# 作り話の検出
# ---------------------------------------------------------------------------

# 根拠なく書かれやすい数値の形。
# 「〜%」「〜社」のように、実績や規模を示す数字だけを拾う。
# 年月日や条数まで拾うと、法令の参照まで警告になって使い物にならない。
_FACT_PATTERNS = [
    (re.compile(r"\d+(?:\.\d+)?\s*[%％]"), "割合"),
    (re.compile(r"\d+(?:,\d{3})*\s*社"), "社数"),
    (re.compile(r"\d+(?:,\d{3})*\s*[円ドル]"), "金額"),
    (re.compile(r"\d+(?:\.\d+)?\s*倍"), "倍率"),
    (re.compile(r"\d+(?:,\d{3})*\s*件"), "件数"),
    (re.compile(r"\d+(?:,\d{3})*\s*人"), "人数"),
]


@dataclass
class FactWarning:
    """修正で新しく現れた数値 1 件。"""

    kind: str
    text: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "text": self.text}


def check_new_facts(
    before_body: str,
    after_body: str,
    brand: Brand,
) -> list[FactWarning]:
    """修正で新しく現れた数値を拾う。

    元の記事にも自社情報にも無い数値は、LLM が作った可能性がある。
    ここでは止めずに警告だけを返す。
    法定保存年数のような一般論としての数値まで弾くと書けなくなるため、
    採否の判断は画面を見る人に委ねる。

    Args:
        before_body: 修正前の本文。
        after_body: 修正後の本文。
        brand: 自社の情報。強みや事業内容に書かれた数値は既知として扱う。

    Returns:
        新しく現れた数値の一覧。順序は本文中の登場順、重複は除く。
    """
    known = before_body + "\n" + brand.description + "\n" + "\n".join(brand.strengths)

    warnings: list[FactWarning] = []
    seen: set[str] = set()

    for pattern, kind in _FACT_PATTERNS:
        for match in pattern.finditer(after_body):
            text = match.group(0)
            # 空白の有無だけが違う書き方を同じものとして扱う。
            normalized = re.sub(r"\s+", "", text)
            if normalized in seen:
                continue
            if re.search(re.escape(normalized), re.sub(r"\s+", "", known)):
                continue
            seen.add(normalized)
            warnings.append(FactWarning(kind=kind, text=text))

    return warnings

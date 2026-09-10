# README

## これは何をするアプリか

未公開の記事を「疑似ウェブ」（Tavily で集めた他社記事の集合）に差し込み、
生成 AI がその記事を検索・引用するかを定量的に測るツール。
記事を公開する前に LLMO 施策の効果を比較できるようにするのが目的。

画面（Streamlit）には 3 つの使い方がある。

| パターン | 入力 | 条件の数 | 何が分かるか |
| --- | --- | --- | --- |
| A. トピックから記事を作る | トピック | 1（`single`） | 作った記事単体の水準 |
| B. 既存の記事を書き直す | タイトル + 本文 | 2（`before` / `after`） | 書き直しの前後差 |
| C. 改善の再評価 | 評価結果からの続き | 1 | 改善前との比較（`improvements` 経由） |

---

## セットアップ

```bash
# 1. 仮想環境
python3 -m venv .venv
source .venv/bin/activate

# 2. 依存ライブラリ（torch は CPU 版を先に入れる。既定だと CUDA 込みで数 GB になる）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. API キー
cat > .env <<'ENV'
GEMINI_API_KEY=...   # https://aistudio.google.com/apikey
TAVILY_API_KEY=...   # https://app.tavily.com/
ENV
```

キーが未設定だと `llmo/config.py` が起動時点で `RuntimeError` を投げる。

初回実行時に埋め込みモデル `cl-nagoya/ruri-v3-30m`（約 130MB）を Hugging Face から取得する。

## 起動

```bash
source .venv/bin/activate
streamlit run app.py
```

ブラウザで http://localhost:8501 が開く。

**最初にやること**: 「疑似ウェブ」タブでコーパスを作る。
評価タブはここで作ったコーパスを選ぶだけで、その場では作らない。

---

## ファイル構成

```
LLMO/
├── README.md                設計書。仕様・判断の根拠の正
├── OVERVIEW.md              このファイル
├── app.py                   画面（Streamlit）。表示だけ。1,576 行
├── requirements.txt         依存ライブラリ
├── .env                     API キー（.gitignore 済み）
│
├── llmo/                    ロジック本体。画面から独立
│   ├── config.py            設定値と API キー。実験条件はすべてここ
│   ├── pipeline.py          各段階を順に呼ぶ。画面が呼ぶのはここだけ
│   ├── db.py                評価結果を SQLite に保存・読み出し
│   │
│   ├── core/                どの段階からも使う下回り
│   │   ├── gemini.py        Gemini クライアント + 共通のリトライ
│   │   └── brand.py         自社情報（企業名・製品名・表記ゆれ）の読み書き
│   │
│   ├── generation/          記事を作る側
│   │   ├── query.py         ユーザー入力 → 評価用の検索クエリを複数生成
│   │   ├── audience.py      クエリ → 「読者の状況」に言い換える（クエリ漏れ対策）
│   │   ├── title.py         タイトル案を複数生成（ユーザーが 1 つ選ぶ）
│   │   ├── writer.py        記事の新規作成 / 既存記事の書き直し
│   │   ├── rules.py         LLMO 最適化の指示文（画面から編集可・data/ に保存）
│   │   ├── leakage.py       見出しにクエリが転記されていないか検査
│   │   └── improve.py       弱点の分析 → 該当する節だけを書き直す
│   │
│   ├── corpus/              疑似ウェブ（ベースコーパス）を作る側
│   │   ├── fetch.py         Tavily で本文を取得し保存（固定して使い回す）
│   │   ├── clean.py         ナビ・フッター等を除いて本文だけ残す
│   │   ├── chunk.py         本文をチャンクに分割（500 字 / 重なり 100 字）
│   │   └── embed.py         チャンク・クエリをベクトル化（手元のモデル・キャッシュあり）
│   │
│   ├── retrieval/           検索して回答を作る側
│   │   ├── experiment.py    ベースコーパスに記事を 1 本差し込んで「条件」を作る
│   │   ├── retrieve.py      ベクトル検索 → 記事単位に集約して順位付け
│   │   └── answer.py        検索結果を出典として渡し、[1][2] 付きで回答生成
│   │
│   └── evaluation/          指標を計算する側
│       ├── metrics.py       文字列処理だけで出せる 3 指標
│       ├── judge.py         LLM 判定が要る 2 指標（推奨・根拠）
│       ├── weakness.py      結果から「改善すべきクエリ」を選ぶ（API 呼ばない）
│       └── compare.py       改善の前後を比べ、修正版を採用してよいか判定
│
└── data/                    生成物（.gitignore 済み）
    ├── brand.json           自社情報。差し替えれば別の企業に使える
    ├── llmo_rules.md        画面で編集した最適化の指示
    ├── sample_article.md    動作確認用のサンプル記事
    ├── corpus/              <名前>.json（本文） / .npy（ベクトル） / .fingerprint.json
    └── llmo.db              評価結果（指標・回答・出典・根拠・実験条件）
```

---

## 処理の流れ

```
① query.py      トピック → 評価用クエリ 20 本
② title.py      タイトル案 8 件 → ユーザーが 1 つ選ぶ（ここで一度止まる）
③ fetch/clean/chunk/embed  疑似ウェブを構築（別タブで事前に実行・固定）
④ writer.py     記事を生成（audience.py で言い換えたお題を渡す）
⑤ leakage.py    見出しにクエリが写っていないか検査
⑥ experiment.py ベースコーパス + 記事 1 本 = 条件
⑦ retrieve.py   条件ごとにベクトル検索（上位 K=5 記事）
⑧ answer.py     出典を渡して回答生成
⑨ metrics/judge 指標を計算
⑩ db.py         条件のスナップショットごと保存
   ↓（改善ループ）
⑪ weakness.py   弱いクエリを選ぶ → improve.py で該当の節を書き直す
⑫ compare.py    前後を比べ、総合的に良くなったときだけ採用
```

画面から呼ぶ入口は `llmo/pipeline.py` の 5 つだけ。

| 関数 | 役割 |
| --- | --- |
| `make_queries(topic)` | 評価用クエリの生成 |
| `prepare_corpus(...)` | 疑似ウェブの構築・読み込み |
| `evaluate_new_article(...)` | パターン A の評価 |
| `evaluate_rewrite(...)` | パターン B の評価 |
| `run_improvement(...)` | 改善ループ 1 周分 |

---

## 主な設定値（`llmo/config.py`）

コード中に数値を直書きせず、実験条件はすべてここに集約している。

| 分類 | 項目 | 既定値 |
| --- | --- | --- |
| モデル | 記事・回答の生成 | `gemini-3.1-flash-lite` |
| | 評価の判定 / 言い換え | `gemini-3.5-flash-lite` |
| | 埋め込み（手元で実行） | `cl-nagoya/ruri-v3-30m` |
| 評価 | クエリ数 / タイトル案数 | 20 / 8 |
| | 検索の取得件数 `top_k` | 5 |
| | 回答の生成回数 `num_runs` | 1 |
| | チャンク長 / 重なり | 500 / 100 字 |
| | クエリ漏れの閾値 | 0.6 |
| 保存先 | DB / 指示文 / コーパス | `data/llmo.db` / `data/llmo_rules.md` / `data/corpus/` |

> ここを変えると測り方そのものが変わるため、過去の結果とは比較できなくなる。
> `db.py` は条件から `condition_key` を作って各行に持たせ、比較していい相手かを機械的に判定している。

## DB のテーブル（`data/llmo.db`）

| テーブル | 内容 |
| --- | --- |
| `evaluations` | 1 回の評価。実験条件のスナップショットを含む |
| `metrics` | 条件ごとの指標 |
| `answers` | 生成された回答の本文 |
| `sources` | 回答が引用した出典 |
| `grounding_claims` | どの主張を自社記事が支えたか |
| `improvements` | 改善ループの各周と、その採否 |

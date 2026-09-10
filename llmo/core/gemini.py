"""Gemini クライアントと、API 呼び出しの共通処理をまとめるモジュール。

クエリ生成・回答生成・埋め込み・評価と、Gemini を呼ぶ場所が複数ある。
各所で個別にクライアントを作ると API キーの読み込みが散らばるので、
入口をこの 1 か所にまとめる。

リトライもここに置く。
評価は 1 回で十数回 API を叩くため、途中の 1 回が一時的なエラーで落ちると
それまでの処理がすべて無駄になる。
"""

import random
import re
import time
from functools import lru_cache
from typing import Any, Callable, TypeVar

from google import genai
from google.genai import errors

from llmo.config import get_gemini_api_key

T = TypeVar("T")

# 待てば直る見込みのあるエラーコード。
# 429: レート上限に達した（時間を空ければ回復する）
# 503: モデル側が混雑している（一時的なもの）
# 500 / 502 / 504: サーバー側の一時的な不調
_RETRYABLE_CODES = {429, 500, 502, 503, 504}

# 再試行の回数。
_MAX_RETRIES = 5

# 待ち時間の基準（秒）。試行のたびに倍にしていく。
_BASE_WAIT = 2.0


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    """Gemini クライアントを返す。

    lru_cache を付けているので、何度呼んでも実際に作られるのは初回だけ。
    以降は同じインスタンスが返る（毎回作り直す無駄を避ける）。
    """
    return genai.Client(api_key=get_gemini_api_key())


def _wait_seconds(error: Exception, attempt: int) -> float:
    """次の再試行までに待つ秒数を決める。

    API が "Please retry in 41.8s" のように待ち時間を指示してくることがある。
    その場合は指示に従うのが最も確実。

    指示が無ければ、試行のたびに待ち時間を倍にしていく（2 秒 → 4 秒 → 8 秒…）。
    混雑が原因のときに、すぐ叩き直すとさらに混雑を悪化させるため。

    最後に少しランダムな幅を足している。
    複数の処理が同時に失敗したとき、全部が同じタイミングで再試行すると
    また同時に集中してしまうのを避けるため。
    """
    match = re.search(r"retry in ([\d.]+)s", str(error))
    if match:
        return float(match.group(1)) + 1.0

    return _BASE_WAIT * (2**attempt) + random.uniform(0, 1)


def call_with_retry(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """API 呼び出しを、一時的なエラーに耐える形で実行する。

    待っても直らないエラー（API キーの誤り、リクエストの不備など）は
    そのまま投げる。何度試しても同じ結果になるうえ、
    原因が分かりにくくなるため。

    Args:
        func: 実行する関数。API を呼ぶ処理を渡す。

    Returns:
        func の戻り値。
    """
    for attempt in range(_MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except (errors.ClientError, errors.ServerError) as error:
            is_last = attempt == _MAX_RETRIES - 1
            if error.code not in _RETRYABLE_CODES or is_last:
                raise

            wait = _wait_seconds(error, attempt)
            print(
                f"  API エラー({error.code})。{wait:.0f} 秒待って再試行します"
                f"（{attempt + 1}/{_MAX_RETRIES - 1}）",
                flush=True,
            )
            time.sleep(wait)

    raise RuntimeError("API 呼び出しの再試行回数が上限に達しました")


def generate(**kwargs: Any) -> Any:
    """テキスト生成を実行する（リトライ付き）。

    generate_content の引数をそのまま渡せる薄い包み。
    呼ぶ側がリトライを意識しなくて済むようにする。
    """
    return call_with_retry(get_client().models.generate_content, **kwargs)

import json
import os
import signal
import threading
import urllib.error
import urllib.request
from pathlib import Path


API_URL = "https://qianfan.baidubce.com/v2/chat/completions"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


class QianfanRequestTimeout(TimeoutError):
    pass


def _timeout_handler(_signum, _frame):
    raise QianfanRequestTimeout("qianfan request exceeded configured timeout")


def load_project_env(env_path=ENV_PATH):
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

    if load_dotenv is not None:
        load_dotenv(dotenv_path=env_path, override=False)
        return

    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_road_choice_prompt(samples):
    lines = []
    for idx, sample in enumerate(samples, start=1):
        text = str(sample.get("text", "")).strip()
        score = sample.get("score", None)
        score_text = "未知" if score is None else f"{float(score):.3f}"
        lines.append(f"{idx}. 文本：{text}，置信度：{score_text}")

    return (
        "你是道路选择专家，负责把智能车识别到的路牌文字转换成岔路控制指令。\n"
        "\n"
        "道路场景：车辆沿赛道顺时针行驶。赛道像一个外侧大圈套着一个内侧小圈，"
        "大圈和小圈有一段共用的重合道路；车辆从重合道路向前行驶后，会遇到一个岔路口。"
        "在这个岔路口，LEFT 表示走左侧/直行方向，继续沿外侧大圈、原路线或较长路线行驶；"
        "RIGHT 表示走右侧岔路，进入内侧小圈、小圈近路、较短路线或抄近道路线。"
        "也就是说，直道/原路线/外圈/大圈通常对应 LEFT，右道/右侧岔路/小圈/近路通常对应 RIGHT。\n"
        "\n"
        "任务：下面是看到路牌后停车期间连续多次整图 OCR 的识别结果；每次可能包含多段文本，"
        "其中有些文本可能不在路牌附近或与路牌无关。OCR 可能有错字、漏字、形近字，"
        "例如“大圈”可能被识别成“犬圈”，“小圈”可能被识别不完整。"
        "请结合道路场景和每次 OCR 的置信度综合判断路牌真实意图，不要机械匹配单个关键词。\n"
        "\n"
        "语义规则：\n"
        "1. 表示某条路不可通行的语义优先级最高，例如走不了、不能走、不通、禁止、别走、不要走，"
        "以及含义相近的表达；如果某条路不可通行，必须选择另一条路。\n"
        "2. 表示路况差或体验差的语义不等于不可通行，例如不好走、崎岖、难走、绕、麻烦、坑洼，"
        "以及含义相近的表达；如果另一条路明确不可通行，应选择这条虽然差但还能走的路。\n"
        "3. 表示更快、更短、更近、省时间、少绕路、捷径、近路、抄近道、走小圈、内圈等效率或距离优势，"
        "以及含义相近的表达，通常表示选择 RIGHT。\n"
        "4. 表示继续原路线、外圈、大圈、直行、直道，或者明确不要走近路/小圈/右侧岔路，通常表示选择 LEFT。\n"
        "5. 注意反问、反话和转折。比如“右道好走多了？才怪”表示右道并不好走，但只是路况差；"
        "“右道崎岖，但是直道走不了”表示直道不可通行，应选择 RIGHT。\n"
        "\n"
        "输出要求：只能输出一个单词 LEFT 或 RIGHT，不要输出解释、标点或其它内容。\n"
        "\n"
        "OCR结果：\n"
        + "\n".join(lines)
    )


def request_road_choice(samples, model=None, timeout=3.0):
    load_project_env()
    api_key = os.getenv("QIANFAN_API_KEY", "").strip()
    model = str(model or os.getenv("QIANFAN_MODEL", "ernie-4.5-turbo-32k")).strip()
    if not api_key:
        raise RuntimeError("missing QIANFAN_API_KEY")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": build_road_choice_prompt(samples)}],
        "temperature": 0.1,
        "max_tokens": 3,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    timeout = max(0.1, float(timeout))
    # urllib 的 timeout 在部分板端 OpenSSL TLS 握手内不会严格生效；千帆调用位于
    # 独立进程的主线程，因此用 SIGALRM 补上硬上限，防止一次握手卡住二十多秒。
    use_alarm = threading.current_thread() is threading.main_thread()
    old_handler = None
    old_timer = None
    try:
        if use_alarm:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            old_timer = signal.setitimer(signal.ITIMER_REAL, timeout)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {raw}") from exc
    except QianfanRequestTimeout as exc:
        raise RuntimeError(str(exc)) from exc
    finally:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, *(old_timer or (0.0, 0.0)))
            signal.signal(signal.SIGALRM, old_handler)

    data = json.loads(raw)
    content = str(data["choices"][0]["message"]["content"]).strip().upper()
    if "LEFT" in content and "RIGHT" not in content:
        result = "LEFT"
    elif "RIGHT" in content and "LEFT" not in content:
        result = "RIGHT"
    elif content in ("LEFT", "RIGHT"):
        result = content
    else:
        raise RuntimeError(f"invalid qianfan result: {content!r}")
    return result, raw

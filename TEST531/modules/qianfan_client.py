import json
import os
import urllib.error
import urllib.request
from pathlib import Path


API_URL = "https://qianfan.baidubce.com/v2/chat/completions"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


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
        "道路场景：车辆行驶在一段大圈和小圈共用的重合道路上，前方会出现一个岔路口。"
        "在这个岔路口，左侧道路会继续沿大圈行驶，从车的视角看近似直走或略向左；"
        "右侧道路会进入小圈，小圈是更短、更近的路线。\n"
        "\n"
        "任务：下面是同一个路牌停车后连续多次 OCR 的识别结果。OCR 可能有错字、漏字、形近字，"
        "例如“大圈”可能被识别成“犬圈”，“小圈”可能被识别不完整。"
        "请结合道路场景和每次 OCR 的置信度综合判断路牌真实意图，而不是机械匹配单个关键词。\n"
        "\n"
        "决策含义：\n"
        "- 如果路牌想让车辆继续走大圈、外圈、原路线，或明确不要走近路，输出 LEFT。\n"
        "- 如果路牌想让车辆走小圈、内圈、近路、捷径、抄近道，输出 RIGHT。\n"
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

    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {raw}") from exc

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

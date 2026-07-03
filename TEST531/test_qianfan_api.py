#!/usr/bin/env python3
"""Quick Qianfan chat-completions API smoke test.

Usage:
    cp .env.example .env
    # edit .env and fill QIANFAN_API_KEY
    python3 test_qianfan_api.py
    python3 test_qianfan_api.py --model ernie-4.5-turbo-32k --text "抄近道"
"""

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request


API_URL = "https://qianfan.baidubce.com/v2/chat/completions"
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"


def load_project_env(env_path=ENV_PATH):
    """Load project .env using python-dotenv when available, with a small fallback."""
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


load_project_env()


def get_secret_value(name, default=""):
    env_value = os.getenv(name, "").strip()
    if env_value:
        return env_value
    return default


def parse_args():
    parser = argparse.ArgumentParser(description="Test Baidu Qianfan chat-completions API.")
    parser.add_argument(
        "--model",
        default=get_secret_value("QIANFAN_MODEL", "ernie-4.5-turbo-32k"),
        help="Qianfan model id. Default: QIANFAN_MODEL or ernie-4.5-turbo-32k.",
    )
    parser.add_argument(
        "--text",
        default="走大圈",
        help="Road choice text to classify. Default: 走大圈",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout seconds. Default: 20",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    api_key = get_secret_value("QIANFAN_API_KEY")
    if not api_key:
        print("Missing QIANFAN_API_KEY. Put it in .env or export it in the shell.", file=sys.stderr)
        return 2

    prompt = (
        "你是道路选择专家，正在帮助一辆智能车理解路牌或任务文本。"
        "赛道由一个大圈和一个小圈组成，大圈和小圈有一段道路是重合的。"
        "车辆行驶在重合路段后会遇到一个岔路口："
        "左侧道路会继续沿着大圈行驶，从车的视角看接近直走或略向左；"
        "右侧道路会进入小圈，也就是更短、更近的路线。"
        "请根据输入文本表达的真实意图判断车辆在这个岔路口应该选择哪条路。"
        "如果文本想让车继续走大圈、外圈、原来的大路线、不要走近路，就选择左侧。"
        "如果文本想让车进入小圈、内圈、近路、捷径、抄近道，就选择右侧。"
        "只输出最终控制指令：LEFT 或 RIGHT。"
        "不要输出任何解释、标点、换行以外的其它内容。"
        f"输入文本：{args.text}"
    )
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
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

    print(f"Qianfan model: {args.model}")
    print(f"OCR input: {args.text}")
    print("Requesting Qianfan API...")

    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code} {exc.reason}", file=sys.stderr)
        print(raw, file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    print("\nRaw response:")
    print(raw)

    try:
        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
    except Exception:
        return 0

    print("\nParsed result:")
    print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

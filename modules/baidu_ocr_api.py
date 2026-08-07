"""Baidu OCR API client for in-memory OpenCV images.

The main project can pass a BGR numpy image directly to ``run_full_frame``.
The image is encoded in memory and is never written to disk.

Environment variables:
    BAIDU_OCR_API_KEY
    BAIDU_OCR_SECRET_KEY

Example:
    from modules.baidu_ocr_api import BaiduOCRRecognizer

    ocr = BaiduOCRRecognizer()
    results = ocr.run_full_frame(frame_bgr)
    for item in results:
        print(item["text"], item["score"], item["points"])
"""

import argparse
import base64
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
OCR_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/general"


def _load_project_env(env_path=ENV_PATH):
    """Load the project .env file when python-dotenv is available."""
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


def _env_value(name, default=""):
    return os.getenv(name, default).strip()


class BaiduOCRRecognizer:
    """Call Baidu's location-aware general OCR endpoint."""

    def __init__(
        self,
        api_key=None,
        secret_key=None,
        timeout=10.0,
        image_format=".jpg",
        jpeg_quality=90,
    ):
        _load_project_env()

        self.api_key = str(api_key or _env_value("BAIDU_OCR_API_KEY"))
        self.secret_key = str(secret_key or _env_value("BAIDU_OCR_SECRET_KEY"))
        self.timeout = float(timeout)
        self.image_format = str(image_format).lower()
        self.jpeg_quality = int(jpeg_quality)

        if not self.api_key or not self.secret_key:
            raise RuntimeError(
                "Missing BAIDU_OCR_API_KEY or BAIDU_OCR_SECRET_KEY. "
                "Put them in .env or export them in the shell."
            )

        if self.image_format not in (".jpg", ".jpeg", ".png"):
            raise ValueError("image_format must be .jpg, .jpeg, or .png")

        self._access_token = ""
        self._token_expires_at = 0.0

    def _get_access_token(self):
        """Get and cache an access token until shortly before it expires."""
        now = time.monotonic()
        if self._access_token and now < self._token_expires_at:
            return self._access_token

        response = requests.post(
            TOKEN_URL,
            params={
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.secret_key,
            },
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()

        access_token = str(payload.get("access_token", "")).strip()
        if not access_token:
            error_code = payload.get("error")
            error_description = payload.get("error_description")
            raise RuntimeError(
                "Baidu access token request failed: "
                f"error={error_code!r}, description={error_description!r}"
            )

        expires_in = float(payload.get("expires_in", 30 * 24 * 60 * 60))
        self._access_token = access_token
        self._token_expires_at = time.monotonic() + max(60.0, expires_in - 60.0)
        return self._access_token

    def _encode_bgr_image(self, image_bgr):
        """Encode an OpenCV BGR image to image bytes without creating a file."""
        if image_bgr is None:
            raise ValueError("image_bgr cannot be None")

        image = np.asarray(image_bgr)
        if image.size == 0 or image.ndim not in (2, 3):
            raise ValueError("image_bgr must be a non-empty image array")

        encode_params = []
        if self.image_format in (".jpg", ".jpeg"):
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]

        ok, encoded = cv2.imencode(self.image_format, image, encode_params)
        if not ok:
            raise RuntimeError(f"Failed to encode image as {self.image_format}")
        return encoded.tobytes()

    def recognize_image_bytes(self, image_bytes):
        """Send encoded image bytes to Baidu and return the raw JSON payload."""
        if not image_bytes:
            raise ValueError("image_bytes cannot be empty")

        access_token = self._get_access_token()
        image_base64 = base64.b64encode(image_bytes).decode("ascii")
        response = requests.post(
            OCR_URL,
            params={"access_token": access_token},
            data={
                "image": image_base64,
                "language_type": "CHN_ENG",
                "probability": "true",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()

        if "error_code" in payload:
            raise RuntimeError(
                "Baidu OCR request failed: "
                f"error_code={payload.get('error_code')}, "
                f"error_msg={payload.get('error_msg', '')}"
            )
        return payload

    def recognize_image(self, image_bgr):
        """Encode an in-memory BGR image and return Baidu's raw response."""
        image_bytes = self._encode_bgr_image(image_bgr)
        return self.recognize_image_bytes(image_bytes)

    def run_full_frame(self, image_bgr):
        """Return OCR results in the shape expected by the local OCR pipeline."""
        payload = self.recognize_image(image_bgr)
        results = []

        for item in payload.get("words_result", []):
            text = str(item.get("words", "")).strip().upper()
            if not text:
                continue

            location = item.get("location") or {}
            left = float(location.get("left", location.get("x", 0.0)))
            top = float(location.get("top", location.get("y", 0.0)))
            width = float(location.get("width", 0.0))
            height = float(location.get("height", 0.0))
            points = np.array(
                [
                    [left, top],
                    [left + width, top],
                    [left + width, top + height],
                    [left, top + height],
                ],
                dtype=np.float32,
            )

            probability = item.get("probability") or {}
            score = probability.get("average", probability.get("score", 0.0))
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0

            results.append({
                "points": points,
                "text": text,
                "score": score,
            })

        return results

    def close(self):
        """Compatibility method matching the local OCR recognizer."""
        return None


def _parse_args():
    parser = argparse.ArgumentParser(description="Smoke test Baidu OCR with an image file.")
    parser.add_argument("--image", required=True, help="Path to a test image.")
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = _parse_args()
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Failed to read image: {args.image}")

    recognizer = BaiduOCRRecognizer(timeout=args.timeout)
    payload = recognizer.recognize_image(image)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

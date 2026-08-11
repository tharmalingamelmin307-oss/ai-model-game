"""Persistent C++ RKNN segmentation worker wrapper."""

import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

import config


REQ_MAGIC = 0x4E494753   # "SGIN"
RESP_MAGIC = 0x554F4753  # "SGOU"
REQ_STRUCT = struct.Struct("<III")
RESP_STRUCT = struct.Struct("<IIiIffii")


class CppSegWorker:
    def __init__(self, core_id):
        self.frame_id = 0
        self.last_timing = {"rknn": 0.0, "decode": 0.0, "total": 0.0, "candidates": 0, "kept": 0}
        bin_path = Path(getattr(config, "CPP_SEG_WORKER_BIN", "cpp_rknn/seg_worker"))
        if not bin_path.is_absolute():
            bin_path = config.PROJECT_ROOT / bin_path
        if not bin_path.exists():
            raise FileNotFoundError(f"C++ Seg worker not found: {bin_path}")

        cmd = [
            str(bin_path),
            "--model",
            str(config.SEG_MODEL),
            "--core",
            str(core_id),
            "--conf",
            str(float(getattr(config, "SEG_CONF_THRES", 0.25))),
            "--iou",
            str(float(getattr(config, "SEG_NMS_THRES", 0.45))),
            "--max-det",
            str(int(getattr(config, "SEG_MAX_DETS", 1))),
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            bufsize=0,
        )

    def _read_exact(self, size):
        data = bytearray()
        while len(data) < size:
            chunk = self.proc.stdout.read(size - len(data))
            if not chunk:
                raise RuntimeError("C++ Seg worker closed stdout")
            data.extend(chunk)
        return bytes(data)

    def run(self, blob_rgb):
        if self.proc.poll() is not None:
            raise RuntimeError(f"C++ Seg worker exited: {self.proc.returncode}")

        blob = np.ascontiguousarray(blob_rgb, dtype=np.uint8)
        expected_w, expected_h = config.SEG_SIZE
        if blob.shape != (expected_h, expected_w, 3):
            raise ValueError(f"C++ Seg input shape mismatch: {blob.shape}")

        self.frame_id = (self.frame_id + 1) & 0xFFFFFFFF
        payload = blob.tobytes()
        self.proc.stdin.write(REQ_STRUCT.pack(REQ_MAGIC, self.frame_id, len(payload)))
        self.proc.stdin.write(payload)
        self.proc.stdin.flush()

        header = self._read_exact(RESP_STRUCT.size)
        magic, frame_id, status, mask_bytes, run_ms, post_ms, candidates, kept = RESP_STRUCT.unpack(header)
        if magic != RESP_MAGIC or frame_id != self.frame_id:
            raise RuntimeError(f"C++ Seg protocol mismatch: magic={magic:x} frame={frame_id}")
        if status != 0:
            raise RuntimeError(f"C++ Seg worker status={status}")
        mask_data = self._read_exact(mask_bytes)
        mask = np.frombuffer(mask_data, dtype=np.uint8).reshape((expected_h, expected_w))
        total_s = (float(run_ms) + float(post_ms)) / 1000.0
        self.last_timing = {
            "rknn": float(run_ms) / 1000.0,
            "decode": float(post_ms) / 1000.0,
            "total": total_s,
            "candidates": int(candidates),
            "kept": int(kept),
        }
        return (mask > 0).astype(np.uint8), total_s

    def close(self):
        proc = getattr(self, "proc", None)
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=0.5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        self.proc = None

    def __del__(self):
        self.close()

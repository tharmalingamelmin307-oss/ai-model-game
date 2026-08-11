"""Persistent C++ RKNN detection worker wrapper."""

import struct
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

import config
from utils.yolo_rknn_post import letterbox_bgr_or_rgb, scale_boxes_from_letterbox


REQ_MAGIC = 0x4E494459   # "YDIN"
RESP_MAGIC = 0x554F4459  # "YDOU"
REQ_STRUCT = struct.Struct("<III")
RESP_STRUCT = struct.Struct("<IIiIffii")
DET_STRUCT = struct.Struct("<fffffi")


class CppDetWorker:
    def __init__(self, core_id):
        self.frame_id = 0
        self.last_timing = {"preprocess": 0.0, "rknn": 0.0, "decode": 0.0, "total": 0.0}
        self.last_counts = {"raw": 0, "topk": 0, "nms": 0, "result": 0}
        bin_path = Path(getattr(config, "CPP_DET_WORKER_BIN", "cpp_rknn/det_worker"))
        if not bin_path.is_absolute():
            bin_path = config.PROJECT_ROOT / bin_path
        if not bin_path.exists():
            raise FileNotFoundError(f"C++ Det worker not found: {bin_path}")

        cmd = [
            str(bin_path),
            "--model",
            str(config.YOLO_MODEL),
            "--core",
            str(core_id),
            "--conf",
            str(float(getattr(config, "YOLO_CONF_THRES", 0.5))),
            "--iou",
            str(float(getattr(config, "YOLO_NMS_THRES", 0.45))),
            "--max-det",
            str(int(getattr(config, "YOLO_MAX_DETS", 50))),
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
                raise RuntimeError("C++ Det worker closed stdout")
            data.extend(chunk)
        return bytes(data)

    def _class_name_from_id(self, cls_id):
        cls_id = int(cls_id)
        if 0 <= cls_id < len(config.CLASS_NAMES):
            return config.CLASS_NAMES[cls_id]
        return str(cls_id)

    def run(self, frame_bgr, output_size=None):
        if self.proc.poll() is not None:
            raise RuntimeError(f"C++ Det worker exited: {self.proc.returncode}")

        if output_size is None:
            output_size = (frame_bgr.shape[1], frame_bgr.shape[0])
        t0 = cv2.getTickCount()
        blob_bgr, transform = letterbox_bgr_or_rgb(
            frame_bgr,
            config.YOLO_SIZE,
            pad_value=int(getattr(config, "YOLO_LETTERBOX_PAD_VALUE", 114)),
            scaleup=bool(getattr(config, "YOLO_LETTERBOX_SCALEUP", False)),
        )
        blob_rgb = cv2.cvtColor(blob_bgr, cv2.COLOR_BGR2RGB)
        blob_rgb = np.ascontiguousarray(blob_rgb, dtype=np.uint8)
        t1 = cv2.getTickCount()

        self.frame_id = (self.frame_id + 1) & 0xFFFFFFFF
        payload = blob_rgb.tobytes()
        self.proc.stdin.write(REQ_STRUCT.pack(REQ_MAGIC, self.frame_id, len(payload)))
        self.proc.stdin.write(payload)
        self.proc.stdin.flush()

        header = self._read_exact(RESP_STRUCT.size)
        magic, frame_id, status, count, run_ms, post_ms, candidates, kept = RESP_STRUCT.unpack(header)
        if magic != RESP_MAGIC or frame_id != self.frame_id:
            raise RuntimeError(f"C++ Det protocol mismatch: magic={magic:x} frame={frame_id}")
        if status != 0:
            raise RuntimeError(f"C++ Det worker status={status}")

        records = []
        if count > 0:
            raw = self._read_exact(int(count) * DET_STRUCT.size)
            for i in range(int(count)):
                records.append(DET_STRUCT.unpack_from(raw, i * DET_STRUCT.size))

        boxes = np.array([[r[0], r[1], r[2], r[3]] for r in records], dtype=np.float32)
        if len(boxes) > 0:
            boxes = scale_boxes_from_letterbox(boxes, transform, output_size)

        results = []
        output_w, output_h = int(output_size[0]), int(output_size[1])
        for i, rec in enumerate(records):
            x1, y1, x2, y2 = boxes[i]
            w = max(0.0, float(x2 - x1))
            h = max(0.0, float(y2 - y1))
            if w < float(getattr(config, "YOLO_BOX_MIN_SIZE", 3)) or h < float(getattr(config, "YOLO_BOX_MIN_SIZE", 3)):
                continue
            cls_id = int(rec[5])
            cls_name = self._class_name_from_id(cls_id)
            score = float(rec[4])
            frame_area = float(max(output_w * output_h, 1))
            max_area_ratio = getattr(config, "YOLO_MAX_AREA_RATIO_BY_CLASS", {}).get(cls_name)
            if max_area_ratio is not None and (w * h) / frame_area > float(max_area_ratio):
                continue
            results.append({
                "rect": [int(x1), int(y1), int(w), int(h)],
                "class_id": cls_id,
                "class_name": cls_name,
                "score": score,
            })

        t2 = cv2.getTickCount()
        tick_freq = cv2.getTickFrequency()
        self.last_timing = {
            "preprocess": (t1 - t0) / tick_freq,
            "rknn": float(run_ms) / 1000.0,
            "decode": float(post_ms) / 1000.0,
            "total": (t2 - t0) / tick_freq,
        }
        self.last_counts = {
            "raw": int(candidates),
            "topk": int(candidates),
            "nms": int(kept),
            "result": int(len(results)),
        }
        return results

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

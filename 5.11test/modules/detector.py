# modules/detector.py
import cv2
import numpy as np
from rknnlite.api import RKNNLite
import config


class YOLODetector:
    def __init__(self, core_id):
        self.rknn = RKNNLite()
        print(f"--> [Detector] 加载模型: {config.YOLO_MODEL}", flush=True)
        if self.rknn.load_rknn(config.YOLO_MODEL) != 0:
            raise RuntimeError("YOLO 加载失败")
        if self.rknn.init_runtime(core_mask=core_id) != 0:
            raise RuntimeError("YOLO 初始化失败")

        self.conf_thres = 0.25
        self.nms_thres = 0.45
        self.debug_once = True

    def _preprocess(self, frame):
        input_w, input_h = config.YOLO_SIZE  # (768, 576)

        img = cv2.resize(frame, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 关键修正：RKNNLite 这里需要 4 维输入 [1, H, W, C]
        img = np.expand_dims(img, axis=0).astype(np.uint8)
        return img

    def _normalize_preds(self, outputs):
        if outputs is None or len(outputs) == 0:
            return None

        preds = np.array(outputs[0])

        if preds.ndim == 3 and preds.shape[0] == 1:
            preds = preds[0]  # [1, N, C] -> [N, C]

        if preds.ndim != 2:
            raise RuntimeError(f"Unexpected pred shape: {preds.shape}")

        feat_dim = 4 + len(config.CLASS_NAMES)  # 17

        if preds.shape[1] == feat_dim:
            return preds.astype(np.float32)

        if preds.shape[0] == feat_dim:
            return preds.transpose(1, 0).astype(np.float32)

        raise RuntimeError(
            f"Unexpected pred shape after squeeze: {preds.shape}, expect [N, {feat_dim}]"
        )

    def _decode_xyxy(self, boxes, class_ids, scores, orig_w, orig_h, input_w, input_h):
        scale_x = orig_w / float(input_w)
        scale_y = orig_h / float(input_h)

        nms_boxes = []
        valid_class_ids = []
        valid_scores = []

        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = b

            x1 = float(np.clip(x1, 0, input_w - 1))
            y1 = float(np.clip(y1, 0, input_h - 1))
            x2 = float(np.clip(x2, 0, input_w - 1))
            y2 = float(np.clip(y2, 0, input_h - 1))

            w = x2 - x1
            h = y2 - y1
            if w < 2 or h < 2:
                continue

            x = x1 * scale_x
            y = y1 * scale_y
            w = w * scale_x
            h = h * scale_y

            nms_boxes.append([x, y, w, h])
            valid_class_ids.append(int(class_ids[i]))
            valid_scores.append(float(scores[i]))

        if not nms_boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            nms_boxes,
            valid_scores,
            self.conf_thres,
            self.nms_thres
        )

        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                x, y, w, h = nms_boxes[idx]
                cls_id = valid_class_ids[idx]
                cls_name = (
                    config.CLASS_NAMES[cls_id]
                    if 0 <= cls_id < len(config.CLASS_NAMES)
                    else str(cls_id)
                )

                results.append({
                    "rect": [int(x), int(y), int(w), int(h)],
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "score": valid_scores[idx],
                })

        return results

    def run(self, frame_data):
        try:
            orig_h, orig_w = frame_data.shape[:2]
            input_w, input_h = config.YOLO_SIZE

            blob = self._preprocess(frame_data)

            outputs = self.rknn.inference(
                inputs=[blob],
                data_format=['nhwc']
            )
            if outputs is None or len(outputs) == 0:
                return []

            preds = self._normalize_preds(outputs)
            if preds is None or len(preds) == 0:
                return []

            boxes = preds[:, :4].astype(np.float32)
            scores = preds[:, 4:].astype(np.float32)
            scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

            class_ids = np.argmax(scores, axis=1)
            max_scores = scores[np.arange(scores.shape[0]), class_ids]

            if self.debug_once:
                print(f"YOLO input shape: {blob.shape}, dtype: {blob.dtype}", flush=True)
                print("YOLO output num:", len(outputs), flush=True)
                for idx, out in enumerate(outputs):
                    arr = np.array(out)
                    print(f"output[{idx}] shape: {arr.shape}, dtype: {arr.dtype}", flush=True)
                print(
                    f"YOLO score range: min={float(np.min(max_scores)):.6f}, "
                    f"max={float(np.max(max_scores)):.6f}, "
                    f"mean={float(np.mean(max_scores)):.6f}",
                    flush=True
                )
                self.debug_once = False

            keep = max_scores > self.conf_thres
            if not np.any(keep):
                return []

            boxes = boxes[keep]
            class_ids = class_ids[keep]
            max_scores = max_scores[keep]

            return self._decode_xyxy(
                boxes, class_ids, max_scores,
                orig_w, orig_h, input_w, input_h
            )

        except Exception as e:
            print(f"YOLO 解析异常: {e}", flush=True)
            return []
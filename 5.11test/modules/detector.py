# modules/detector.py
import cv2
import numpy as np
from rknnlite.api import RKNNLite
import config


class YOLODetector:
    def __init__(self, core_id):
        self.rknn = RKNNLite()
        print(f"--> [Detector] 加载模型: {config.YOLO_MODEL}")
        if self.rknn.load_rknn(config.YOLO_MODEL) != 0:
            raise RuntimeError("YOLO 加载失败")
        if self.rknn.init_runtime(core_mask=core_id) != 0:
            raise RuntimeError("YOLO 初始化失败")

        self.conf_thres = 0.25
        self.nms_thres = 0.45
        self.debug_once = True

    def _preprocess(self, frame):
        input_w, input_h = config.YOLO_SIZE  # (w, h)

        img = cv2.resize(frame, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        img = img / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        # RKNNLite 按 NHWC 喂
        img = np.expand_dims(img, axis=0)  # [1, H, W, C]
        return img

    def run(self, frame_data):
        try:
            orig_h, orig_w = frame_data.shape[:2]
            input_w, input_h = config.YOLO_SIZE  # (w, h)

            # 1. 预处理
            blob = self._preprocess(frame_data)

            # 2. 推理
            outputs = self.rknn.inference(inputs=[blob])
            if outputs is None or len(outputs) == 0:
                return []

            preds = np.array(outputs[0])

            if self.debug_once:
                print("YOLO output num:", len(outputs))
                for idx, out in enumerate(outputs):
                    arr = np.array(out)
                    print(f"output[{idx}] shape: {arr.shape}, dtype: {arr.dtype}")
                self.debug_once = False

            # 3. 兼容可能的输出布局
            if preds.ndim == 3 and preds.shape[0] == 1:
                preds = preds[0]  # [1, N, C] -> [N, C]

            # 正确模型应输出:
            # [9072, 17] 或 [17, 9072]
            if preds.shape == (9072, 17):
                pass
            elif preds.shape == (17, 9072):
                preds = preds.transpose(1, 0)
            elif preds.shape == (8400, 11) or preds.shape == (11, 8400):
                raise RuntimeError(
                    f"当前加载的不是 576x768 13类 raw 模型，输出 shape={preds.shape}"
                )
            else:
                raise RuntimeError(
                    f"Unexpected pred shape after squeeze: {preds.shape}, expect [9072, 17]"
                )

            preds = preds.astype(np.float32)

            # 4. 拆框和类别分数
            # 前4维: x1, y1, x2, y2
            boxes = preds[:, :4]

            # 后13维: 各类别分数
            scores = preds[:, 4:]

            class_ids = np.argmax(scores, axis=1)
            max_scores = scores[np.arange(scores.shape[0]), class_ids]

            # 5. 阈值过滤
            mask = max_scores > self.conf_thres
            if not np.any(mask):
                return []

            boxes = boxes[mask]
            class_ids = class_ids[mask]
            max_scores = max_scores[mask]

            # 6. 裁剪到模型输入图范围
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, input_w - 1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, input_h - 1)

            # 7. 映射回原图尺寸
            scale_x = orig_w / float(input_w)
            scale_y = orig_h / float(input_h)

            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

            # 8. 转成 OpenCV NMSBoxes 需要的 [x, y, w, h]
            nms_boxes = []
            valid_class_ids = []
            valid_scores = []

            for i, b in enumerate(boxes):
                x1, y1, x2, y2 = b
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)

                if w < 1 or h < 1:
                    continue

                nms_boxes.append([float(x1), float(y1), float(w), float(h)])
                valid_class_ids.append(int(class_ids[i]))
                valid_scores.append(float(max_scores[i]))

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
                for i in indices.flatten():
                    x, y, w, h = nms_boxes[i]
                    cls_id = valid_class_ids[i]
                    cls_name = (
                        config.CLASS_NAMES[cls_id]
                        if 0 <= cls_id < len(config.CLASS_NAMES)
                        else str(cls_id)
                    )

                    results.append({
                        'rect': [int(x), int(y), int(w), int(h)],
                        'class_id': cls_id,
                        'class_name': cls_name,
                        'score': valid_scores[i],
                    })

            return results

        except Exception as e:
            print(f"YOLO 解析异常: {e}")
            return []
# modules/ocr_system.py
"""OCR 检测 + 识别模块.

这里尽量贴近项目里已经验证过的一版处理流程：

1. det 在整张图上找文字区域
2. 用透视变换把文字区域拉正
3. rec 对拉正后的文本图做识别

YOLO 的 `sign / limit_sign` 现在只负责触发 OCR 和做结果关联，
不再直接把 YOLO 框拿来当最终识别 ROI。

当前模块本身不关心“这是普通语义牌还是限速牌”。
类别语义差异由 main.py 在回匹配后决定：
- `sign` 关注 LEFT / RIGHT
- `limit_sign` 只提取其中的数字字符参与限速确认
"""

import cv2
import numpy as np
from pathlib import Path
from rknnlite.api import RKNNLite

import config
from utils.image_proc import get_rotate_crop_image


class OCRRecognizer:
    def __init__(self, core_id):
        """初始化 OCR 检测模型与识别模型."""
        self.rknn_det = RKNNLite()
        self.rknn_rec = RKNNLite()

        if self.rknn_det.load_rknn(config.OCR_DET_MODEL_PATH) != 0:
            raise RuntimeError("OCR det 模型加载失败")

        if self.rknn_rec.load_rknn(config.REC_MODEL_PATH) != 0:
            raise RuntimeError("OCR rec 模型加载失败")

        # 当前主程序里 OCR 是单线程顺序执行 det + rec，所以两者绑定到同一核即可。
        if self.rknn_det.init_runtime(core_mask=core_id) != 0:
            raise RuntimeError("OCR det 初始化失败")
        if self.rknn_rec.init_runtime(core_mask=core_id) != 0:
            raise RuntimeError("OCR rec 初始化失败")

        self.char_list = []
        with open(config.DICT_PATH, 'r', encoding='utf-8') as f:
            self.char_list = [line.strip('\n') for line in f.readlines()]
        self.char_list.append(' ')  # CTC blank

        self.det_input_size = int(config.OCR_DET_INPUT_SIZE)
        self.det_binary_thresh = float(config.OCR_DET_BINARY_THRESH)
        self.det_min_contour_area = float(config.OCR_DET_MIN_CONTOUR_AREA)
        self.last_det_box_count = 0
        self.last_rec_empty_count = 0
        self.last_rec_exception_count = 0
        self.last_rec_valid_count = 0
        self.debug_saved_count = 0

    def _save_debug_crop(self, crop, box, text_tag):
        """保存 OCR det 裁出的文本图，用于排查 rec 空文本。"""
        if not bool(getattr(config, "OCR_DEBUG_SAVE_EMPTY_CROPS", False)):
            return
        max_images = int(getattr(config, "OCR_DEBUG_SAVE_MAX_IMAGES", 30))
        if self.debug_saved_count >= max_images:
            return
        if crop is None or crop.size == 0:
            return

        debug_dir = Path(getattr(config, "OCR_DEBUG_SAVE_DIR", "debug_ocr"))
        debug_dir.mkdir(parents=True, exist_ok=True)

        pts = np.array(box, dtype=np.float32)
        x_min = int(np.min(pts[:, 0])) if pts.size else 0
        y_min = int(np.min(pts[:, 1])) if pts.size else 0
        h, w = crop.shape[:2]
        filename = f"{self.debug_saved_count:03d}_{text_tag}_x{x_min}_y{y_min}_w{w}_h{h}.jpg"
        cv2.imwrite(str(debug_dir / filename), crop)
        self.debug_saved_count += 1

    def _decode(self, preds):
        """把识别模型输出按 CTC 规则解码成字符串和平均置信度。"""
        preds_idx = preds.argmax(axis=1)
        text = ""
        conf = 0.0
        count = 0

        for i in range(len(preds_idx)):
            idx = int(preds_idx[i])
            if idx > 0 and not (i > 0 and idx == int(preds_idx[i - 1])):
                char_pos = idx - 1
                if char_pos < len(self.char_list):
                    text += self.char_list[char_pos]
                    conf += float(preds[i][idx])
                    count += 1

        return text, (conf / count if count > 0 else 0.0)

    def run_text_detection(self, image_bgr):
        """在整张图上执行 OCR 检测，返回文本框四点列表.

        这里采用轻量化的后处理：
        - 二值化概率图
        - 适度膨胀
        - 直接找轮廓并拟合旋转矩形
        目标是优先满足板端实时性，而不是追求最复杂的 DB 后处理。
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        src_h, src_w = image_bgr.shape[:2]
        det_size = int(self.det_input_size)
        img_det = cv2.resize(image_bgr, (det_size, det_size), interpolation=cv2.INTER_LINEAR)
        img_det = cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB)

        outputs = self.rknn_det.inference(inputs=[np.expand_dims(img_det, 0)])
        if outputs is None or len(outputs) == 0:
            return []

        pred = np.array(outputs[0])
        if pred.ndim == 4:
            mask = pred[0, 0, :, :]
        elif pred.ndim == 3:
            mask = pred[0]
        else:
            mask = pred.squeeze()

        mask = (mask > self.det_binary_thresh).astype(np.uint8) * 255

        # 这一步是项目里已经试过的简化版 unclip，先轻量膨胀再找轮廓。
        kernel_size = int(config.OCR_DET_DILATE_KERNEL_SIZE)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=int(config.OCR_DET_DILATE_ITERATIONS))

        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        scale_x = src_w / float(det_size)
        scale_y = src_h / float(det_size)

        for cnt in contours:
            if cnt is None or cv2.contourArea(cnt) < self.det_min_contour_area:
                continue

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect).astype(np.float32)
            box[:, 0] *= scale_x
            box[:, 1] *= scale_y
            boxes.append(box)

        boxes = sorted(boxes, key=lambda pts: (pts[0][1], pts[0][0]))
        return boxes

    def run_single_crop(self, crop):
        """对单个裁图执行识别.

        裁图会先按固定高度缩放，再在右侧补零到 REC_WIDTH，
        以适配 PaddleOCR 风格的 rec 输入。
        """
        if crop is None or crop.size == 0:
            return "", 0.0

        h, w = crop.shape[:2]
        if h < 2 or w < 2:
            return "", 0.0

        ratio = w / float(h)
        new_w = int(config.REC_HEIGHT * ratio)
        new_w = min(max(1, new_w), config.REC_WIDTH)

        img_rec = cv2.resize(crop, (new_w, config.REC_HEIGHT), interpolation=cv2.INTER_LINEAR)

        padded_img = np.zeros((config.REC_HEIGHT, config.REC_WIDTH, 3), dtype=np.uint8)
        padded_img[:, :new_w, :] = img_rec

        img_input = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        outputs = self.rknn_rec.inference(inputs=[np.expand_dims(img_input, 0)])
        text, score = self._decode(outputs[0][0])
        return text, score

    def run_full_frame(self, image_bgr):
        """对整张图执行 det + rec，返回识别结果列表.

        返回结果里的 `points` 会保留原图坐标四点框，
        供上层按中心点回匹配到 sign / limit_sign 检测框。
        """
        boxes = self.run_text_detection(image_bgr)
        self.last_det_box_count = len(boxes)
        self.last_rec_empty_count = 0
        self.last_rec_exception_count = 0
        self.last_rec_valid_count = 0
        results = []

        for box in boxes:
            try:
                crop = get_rotate_crop_image(image_bgr, box)
                text, score = self.run_single_crop(crop)
                text = text.strip().upper()
                if not text:
                    self.last_rec_empty_count += 1
                    self._save_debug_crop(crop, box, "empty")
                    continue

                self.last_rec_valid_count += 1
                results.append({
                    "points": np.array(box, dtype=np.float32),
                    "text": text,
                    "score": float(score),
                })
            except Exception:
                self.last_rec_exception_count += 1
                continue

        return results

# modules/ocr_system.py
"""OCR 识别模块.

当前版本只保留文本识别 (rec) 模型，不跑 OCR 检测模型。
因此它依赖外部先提供裁剪好的 ROI，比如主流程里由 YOLO 检出 sign 后，
再把 sign 对应的小图送到这里识别 LEFT / RIGHT。
"""

import cv2
import numpy as np
from rknnlite.api import RKNNLite
import config

class OCRRecognizer:
    def __init__(self, core_id):
        """初始化 OCR 识别模型并绑定到独立 NPU 核."""
        self.rknn_rec = RKNNLite()
        print(f"--> 加载识别模型: {config.REC_MODEL_PATH}")
        self.rknn_rec.load_rknn(config.REC_MODEL_PATH)
        self.rknn_rec.init_runtime(core_mask=core_id)
        
        # PaddleOCR 的识别模型输出是字符索引序列，这里加载字典用于解码。
        self.char_list = []
        with open(config.DICT_PATH, 'r', encoding='utf-8') as f:
            self.char_list = [line.strip('\n') for line in f.readlines()]
        self.char_list.append(' ')  # 对应 CTC blank

    def _decode(self, preds):
        """将网络输出按 CTC 规则解码成字符串和平均置信度."""
        preds_idx = preds.argmax(axis=1)
        text = ""
        conf = 0.0
        count = 0
        for i in range(len(preds_idx)):
            # CTC 中 0 一般代表 blank，同时需要去掉相邻重复字符。
            if preds_idx[i] > 0 and (not (i > 0 and preds_idx[i] == preds_idx[i - 1])):
                if preds_idx[i] - 1 < len(self.char_list):
                    text += self.char_list[preds_idx[i] - 1]
                    conf += preds[i][preds_idx[i]]
                    count += 1
        return text, (conf / count if count > 0 else 0.0)

    def run_single_crop(self, crop):
        """对单个裁图执行识别.

        参数:
            crop: BGR 格式 ROI 图像。

        返回:
            (text, score)
        """
        if crop is None or crop.size == 0:
            return "", 0.0
            
        # PaddleOCR 识别模型通常固定高度、宽度自适应。
        # 这里先按比例缩放到固定高度，再在右侧补零到固定宽度。
        h, w = crop.shape[:2]
        ratio = w / float(h)
        new_w = int(config.REC_HEIGHT * ratio)
        new_w = min(new_w, config.REC_WIDTH)
        
        img_rec = cv2.resize(crop, (new_w, config.REC_HEIGHT))
        
        padded_img = np.zeros((config.REC_HEIGHT, config.REC_WIDTH, 3), dtype=np.uint8)
        padded_img[:, :new_w, :] = img_rec
        
        # 当前 RKNN 版本直接喂 RGB uint8，归一化由模型导出配置决定。
        img_input = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        
        outputs = self.rknn_rec.inference(inputs=[np.expand_dims(img_input, 0)])
        
        text, score = self._decode(outputs[0][0])
        return text, score

# modules/ocr_system.py
import cv2
import numpy as np
from rknnlite.api import RKNNLite
import config

class OCRRecognizer:
    def __init__(self, core_id):  # 👈 修改 1：加上 core_id 参数
        # 仅加载识别模型 (Rec)
        self.rknn_rec = RKNNLite()
        print(f"--> 加载识别模型: {config.REC_MODEL_PATH}")
        self.rknn_rec.load_rknn(config.REC_MODEL_PATH)
        self.rknn_rec.init_runtime(core_mask=core_id) # 👈 修改 2：这里使用传入的 core_id
        
        # 加载字典
        self.char_list = []
        with open(config.DICT_PATH, 'r', encoding='utf-8') as f:
            self.char_list = [line.strip('\n') for line in f.readlines()]
        self.char_list.append(' ') # 对应 CTC 的空白符

    def _decode(self, preds):
        """ CTC 解码逻辑 """
        preds_idx = preds.argmax(axis=1)
        text = ""; conf = 0.0; count = 0
        for i in range(len(preds_idx)):
            # 0 通常是 CTC 的 blank 标签
            if preds_idx[i] > 0 and (not (i > 0 and preds_idx[i] == preds_idx[i - 1])):
                if preds_idx[i] - 1 < len(self.char_list):
                    text += self.char_list[preds_idx[i] - 1]
                    conf += preds[i][preds_idx[i]]
                    count += 1
        return text, (conf / count if count > 0 else 0.0)

    def run_single_crop(self, crop):
        """ 接收一个 ROI 图像，返回识别文字和置信度 """
        if crop is None or crop.size == 0:
            return "", 0.0
            
        # 1. 预处理 (Resize & Padding)
        h, w = crop.shape[:2]
        ratio = w / float(h)
        # 按照 Rec 模型的输入要求缩放（通常高度固定 48）
        new_w = int(config.REC_HEIGHT * ratio)
        new_w = min(new_w, config.REC_WIDTH)
        
        img_rec = cv2.resize(crop, (new_w, config.REC_HEIGHT))
        
        # 创建画布并填充（Padding）
        padded_img = np.zeros((config.REC_HEIGHT, config.REC_WIDTH, 3), dtype=np.uint8)
        padded_img[:, :new_w, :] = img_rec
        
        # 转换色彩空间并归一化（1/255 是 PaddleOCR 的默认做法）
        img_input = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        
        # 2. 推理
        outputs = self.rknn_rec.inference(inputs=[np.expand_dims(img_input, 0)])
        
        # 3. 解码
        text, score = self._decode(outputs[0][0])
        return text, score
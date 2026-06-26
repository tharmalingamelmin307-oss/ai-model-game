import cv2
import numpy as np
import os
import time
from rknnlite.api import RKNNLite

# ==========================================
# 1. 配置参数 (匹配你提供的官方 Config)
# ==========================================
DET_MODEL_PATH = "models/ppocrv4_det_int8.rknn"
REC_MODEL_PATH = "models/ppocrv4_rec_fp16.rknn"
DICT_PATH = "keys.txt" # 你的字典文件
IMAGE_DIR = "./test.picture"
SAVE_DIR = "./ocr_results"

# 官方配置参数
DET_INPUT_SIZE = 480
REC_HEIGHT = 48
REC_WIDTH = 320

# ==========================================
# 2. 核心工具类：预处理与后处理
# ==========================================

def get_rotate_crop_image(img, points):
    """ 官方 TextSystem 中的图像矫正逻辑 """
    points = np.array(points, dtype=np.float32)
    img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), 
                             np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), 
                              np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(img, M, (img_crop_width, img_crop_height),
                                 borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if dst_img.shape[0] * 1.0 / dst_img.shape[1] >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img

class CTCDecoder:
    """ 替代 utils.rec_postprocess 中的 CTCLabelDecode """
    def __init__(self, dict_path):
        self.char_list = []
        if os.path.exists(dict_path):
            with open(dict_path, 'r', encoding='utf-8') as f:
                self.char_list = [line.strip('\n') for line in f.readlines()]
        self.char_list.append(' ') # space

    def decode(self, preds):
        preds_idx = preds.argmax(axis=1)
        text = ""
        conf = 0.0
        count = 0
        for i in range(len(preds_idx)):
            if preds_idx[i] > 0 and (not (i > 0 and preds_idx[i] == preds_idx[i - 1])):
                if preds_idx[i] - 1 < len(self.char_list):
                    text += self.char_list[preds_idx[i] - 1]
                    conf += preds[i][preds_idx[i]]
                    count += 1
        return text, (conf / count if count > 0 else 0.0)

# ==========================================
# 3. 推理封装
# ==========================================

class RKNNOCRSystem:
    def __init__(self):
        # 初始化检测器 (Core 0)
        self.rknn_det = RKNNLite()
        self.rknn_det.load_rknn(DET_MODEL_PATH)
        self.rknn_det.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        
        # 初始化识别器 (Core 1)
        self.rknn_rec = RKNNLite()
        self.rknn_rec.load_rknn(REC_MODEL_PATH)
        self.rknn_rec.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
        
        self.decoder = CTCDecoder(DICT_PATH)

    def run_det(self, img):
        ori_h, ori_w = img.shape[:2]
        # 1. 预处理 (匹配你给出的 RKNN_PRE_PROCESS_CONFIG)
        # 注意：RKNN通常在导出时做了归一化，这里只需Resize
        img_det = cv2.resize(img, (DET_INPUT_SIZE, DET_INPUT_SIZE))
        img_det = cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB)
        
        # 2. 推理
        outputs = self.rknn_det.inference(inputs=[np.expand_dims(img_det, 0)])
        
        # 3. 后处理 (DBPostProcess 简化版)
        mask = outputs[0][0, 0, :, :]
        mask = (mask > 0.3).astype(np.uint8) * 255
        
        # 官方 Unclip 逻辑简化：膨胀处理
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 100: continue
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            # 还原坐标到原图
            box[:, 0] = box[:, 0] * (ori_w / DET_INPUT_SIZE)
            box[:, 1] = box[:, 1] * (ori_h / DET_INPUT_SIZE)
            boxes.append(box.astype(np.int32))
        return boxes

    def run_rec(self, img_crop_list):
        results = []
        for crop in img_crop_list:
            # 1. 预处理 (匹配 REC_INPUT_SHAPE [48, 320])
            h, w = crop.shape[:2]
            ratio = w / float(h)
            new_w = int(REC_HEIGHT * ratio)
            if new_w > REC_WIDTH: new_w = REC_WIDTH
            
            img_rec = cv2.resize(crop, (new_w, REC_HEIGHT))
            # Padding
            padding_img = np.zeros((REC_HEIGHT, REC_WIDTH, 3), dtype=np.uint8)
            padding_img[:, :new_w, :] = img_rec
            
            # 归一化 (匹配你代码中的 1./255. scale)
            padding_img = cv2.cvtColor(padding_img, cv2.COLOR_BGR2RGB)
            
            # 2. 推理
            rec_outputs = self.rknn_rec.inference(inputs=[np.expand_dims(padding_img, 0)])
            
            # 3. 解码
            text, score = self.decoder.decode(rec_outputs[0][0])
            results.append((text, score))
        return results

# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    
    engine = RKNNOCRSystem()
    print("✅ RKNN OCR 系统已启动")

    for name in sorted(os.listdir(IMAGE_DIR)):
        if not name.lower().endswith(('.jpg', '.png')): continue
        
        path = os.path.join(IMAGE_DIR, name)
        frame = cv2.imread(path)
        if frame is None: continue

        # --- 步骤 1: 检测 ---
        boxes = engine.run_det(frame)
        
        # --- 步骤 2: 矫正 (你要求的核心逻辑) ---
        img_crop_list = []
        # 按坐标排序
        boxes = sorted(boxes, key=lambda x: (x[0][1], x[0][0]))
        for box in boxes:
            crop = get_rotate_crop_image(frame, box)
            img_crop_list.append(crop)

        # --- 步骤 3: 识别 ---
        rec_results = engine.run_rec(img_crop_list)

        # 可视化
        for box, (text, score) in zip(boxes, rec_results):
            if score < 0.5: continue
            print(f"图片: {name} | 识别: {text} ({score:.2f})")
            cv2.polylines(frame, [box], True, (0, 0, 255), 2)

        cv2.imwrite(os.path.join(SAVE_DIR, f"res_{name}"), frame)

    print(f"🏁 处理完毕，查看 {SAVE_DIR}")
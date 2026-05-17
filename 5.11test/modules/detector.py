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

    def run(self, frame_data):
        # 1. 预处理
        blob = cv2.resize(frame_data, config.YOLO_SIZE, interpolation=cv2.INTER_NEAREST)
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
        
        # 2. 推理
        out = self.rknn.inference(inputs=[np.expand_dims(blob, axis=0)])
        
        # 3. 后处理 (直接使用你原本的 NMS 逻辑)
        try:
            preds = out[0][0].transpose(1, 0)
            boxes, scores = preds[:, :4], preds[:, 4:]
            class_ids = np.argmax(scores, axis=1)
            max_scores = scores[np.arange(len(scores)), class_ids]
            
            mask = max_scores > 0.1
            if not np.any(mask): return []
            
            boxes = boxes[mask]
            class_ids = class_ids[mask]
            max_scores = max_scores[mask]
            
            x, y = boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2
            cv_boxes = np.stack((x, y, boxes[:, 2], boxes[:, 3]), axis=-1).tolist()
            indices = cv2.dnn.NMSBoxes(cv_boxes, max_scores.tolist(), 0.3, 0.45)
            
            scale_x = config.TARGET_RES[0] / config.YOLO_SIZE[0]
            scale_y = config.TARGET_RES[1] / config.YOLO_SIZE[1]
            
            results = []
            if len(indices) > 0:
                for i in indices.flatten():
                    bx, by, bw, bh = cv_boxes[i]
                    results.append({
                        'rect': [int(bx*scale_x), int(by*scale_y), int(bw*scale_x), int(bh*scale_y)], 
                        'class_id': int(class_ids[i])
                    })
            return results
        except Exception as e:
            print(f"YOLO 解析异常: {e}")
            return []
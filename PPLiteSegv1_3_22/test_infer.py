import os
import cv2
import numpy as np
import time
from rknnlite.api import RKNNLite

MODEL_PATH = './model/ppliteseg_3_22.rknn'
TEST_DIR = './test_picture'
RESULT_DIR = './result_picture'

def main():
    rknn = RKNNLite()
    if rknn.load_rknn(MODEL_PATH) != 0:
        print("❌ 加载失败！"); return
    if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
        print("❌ 初始化失败！"); return

    valid_exts = ('.jpg', '.jpeg', '.png')
    for filename in os.listdir(TEST_DIR):
        if not filename.lower().endswith(valid_exts): continue
            
        img = cv2.imread(os.path.join(TEST_DIR, filename))
        if img is None: continue
            
        # 1. 极简预处理：只需 Resize 和 BGR 转 RGB
        img_resized = cv2.resize(img, (960, 720))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # 【极其重要】不要除以 255，不要转 float32！直接喂 uint8 数据
        # NPU 内部会根据我们 config 里的 mean/std 自动处理数据
        input_data = np.expand_dims(img_rgb, axis=0) # 形状为 (1, 720, 960, 3)

        # 2. 推理
        t0 = time.time()
        outputs = rknn.inference(inputs=[input_data])
        duration = (time.time() - t0) * 1000
        
        # 3. 维度兼容性解析
        raw_out = outputs[0]
        # 如果输出是 (1, 2, 720, 960)，说明 Argmax 丢了，我们手动补
        if len(raw_out.shape) == 4 and raw_out.shape[1] == 2:
            pred_mask = np.argmax(raw_out, axis=1).squeeze()
        else:
            pred_mask = np.squeeze(raw_out)
        
        # 4. 可视化
        vis_img = img_resized.copy()
        road_mask = (pred_mask == 1)
        
        print(f"📷 {filename} | 探针: {np.unique(pred_mask)} | 耗时: {duration:.1f}ms")
        
        red_overlay = np.zeros_like(img_resized)
        red_overlay[road_mask] = [0, 0, 255]
        blended = cv2.addWeighted(vis_img, 0.6, red_overlay, 0.4, 0)
        
        cv2.putText(blended, f"{duration:.1f} ms", (30, 50), 2, 1, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(RESULT_DIR, f"fix_red_{filename}"), blended)

    rknn.release()

if __name__ == '__main__':
    main()
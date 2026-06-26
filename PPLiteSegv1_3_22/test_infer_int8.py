import os
import cv2
import numpy as np
import time
from rknnlite.api import RKNNLite

MODEL_PATH = './model/ppliteseg_int8.rknn' 
TEST_DIR = './test_picture'
RESULT_DIR = './result_picture_int8'

if not os.path.exists(RESULT_DIR): os.makedirs(RESULT_DIR)

def main():
    rknn = RKNNLite()
    
    # 1. 加载模型
    if rknn.load_rknn(MODEL_PATH) != 0: return

    # 2. 开启三核并行模式 (RK3588 特有)
    print("--> 正在开启 NPU 三核全速模式...")
    if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
        print("❌ 初始化失败！")
        return

    valid_exts = ('.jpg', '.jpeg', '.png')
    all_files = [f for f in os.listdir(TEST_DIR) if f.endswith(valid_exts)]
    
    print(f"🚀 开始极致性能测试 (图片总数: {len(all_files)})...")

    for filename in all_files:
        img = cv2.imread(os.path.join(TEST_DIR, filename))
        if img is None: continue
            
        # --- 预处理 (CPU) ---
        img_resized = cv2.resize(img, (960, 720))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img_rgb, axis=0)

        # --- 核心推理 (仅统计 NPU 耗时) ---
        t_start = time.time()
        outputs = rknn.inference(inputs=[input_data])
        npu_time_ms = (time.time() - t_start) * 1000
        
        # --- 后处理 (CPU) ---
        pred_mask = np.squeeze(outputs[0])
        if len(pred_mask.shape) == 3:
            pred_mask = np.argmax(pred_mask, axis=0)
            
        # 可视化 (如果觉得画图太慢，可以把这部分注释掉只测速度)
        vis_img = img_resized.copy()
        road_mask = (pred_mask == 1)
        
        # 优化画图逻辑：直接操作像素值比 addWeighted 快
        vis_img[road_mask] = vis_img[road_mask] * 0.5 + np.array([0, 0, 127], dtype=np.uint8)
        
        print(f"📷 {filename} | 🚀 NPU 净耗时: {npu_time_ms:.1f}ms")
        
        cv2.putText(vis_img, f"NPU: {npu_time_ms:.1f}ms", (30, 50), 2, 1, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(RESULT_DIR, f"pro_{filename}"), vis_img)

    rknn.release()
    print(f"\n✅ 优化版测试完成！结果文件夹: {RESULT_DIR}")

if __name__ == '__main__':
    main()
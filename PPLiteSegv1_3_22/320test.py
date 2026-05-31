import os
import time
import glob
import numpy as np
import cv2
from rknnlite.api import RKNNLite

# ==============================================================================
# 1. 配置信息 (专为 320x320 模型与本地图片测试精简)
# ==============================================================================
class Config:
    # ====== 🚀 模型与路径配置 ======
    SEG_MODEL = "model/ppliteseg_320_320_int8.rknn"
    TEST_DIR = "test_picture"
    OUTPUT_DIR = "test_output" # 处理后的图片会保存在这里

    TARGET_RES = (960, 720) 
    # ⚠️ 必须修改为 320x320 以匹配新模型
    SEG_SIZE = (320, 320)   
    
    MASK_ALPHA = 0.4
    
    # ====== 🚀 IPM 标定参数 ======
    SRC_PTS = np.float32([
        [0.393, 0.639],
        [0.603, 0.636],
        [0.683, 0.799],
        [0.310, 0.794],
    ])

    DST_PTS = np.float32([
        [0.385, 0.870],
        [0.615, 0.870],
        [0.615, 1.000],
        [0.385, 1.000],
    ])
    
    SMOOTH_WINDOW = 5       

# ==============================================================================
# 2. 核心测试逻辑
# ==============================================================================
def main():
    # 创建输出文件夹
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # 搜索测试图片
    image_paths = glob.glob(os.path.join(Config.TEST_DIR, "*.*"))
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_paths:
        print(f"❌ 在 {Config.TEST_DIR} 中没有找到图片！")
        return

    print(f"🚀 找到 {len(image_paths)} 张测试图片，正在初始化 NPU...")

    # 初始化 RKNN
    rknn = RKNNLite()
    ret = rknn.load_rknn(Config.SEG_MODEL)
    if ret != 0:
        print("❌ 模型加载失败！"); return
        
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    if ret != 0:
        print("❌ NPU 初始化失败！"); return

    print("✅ NPU 初始化成功，开始批量测试...\n" + "-"*40)

    # 预计算 IPM 矩阵
    w, h = Config.TARGET_RES
    src_points = np.float32([[x * w, y * h] for x, y in Config.SRC_PTS])
    dst_points = np.float32([[x * w, y * h] for x, y in Config.DST_PTS])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    M_inv = cv2.getPerspectiveTransform(dst_points, src_points)

    for img_path in image_paths:
        file_name = os.path.basename(img_path)
        img_raw = cv2.imread(img_path)
        if img_raw is None: continue
        
        # 记录推理耗时
        start_time = time.time()

        # 统一缩放到目标分辨率
        vis_img = cv2.resize(img_raw, Config.TARGET_RES, interpolation=cv2.INTER_NEAREST)
        
        # --- 预处理 ---
        blob = cv2.resize(vis_img, Config.SEG_SIZE, interpolation=cv2.INTER_NEAREST)
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)

        # --- 推理 ---
        outputs = rknn.inference(inputs=[np.expand_dims(blob, axis=0)])
        out = outputs[0]
        
        # 兼容不同导出方式的后处理 (带 ArgMax 或 不带 ArgMax)
        if len(out.shape) == 4 and out.shape[1] > 1:
            mask = (out[0][1] > out[0][0]).astype(np.uint8)
        else:
            mask = out.squeeze().astype(np.uint8)
            
        infer_time = (time.time() - start_time) * 1000 # 毫秒
        
        # --- 1. 将 Mask 放大到屏幕分辨率 ---
        mask_full = cv2.resize(mask, Config.TARGET_RES, interpolation=cv2.INTER_NEAREST)

        # --- 2. 逆透视变换 ---
        bird_eye_mask = cv2.warpPerspective(mask_full, M, (w, h), flags=cv2.INTER_NEAREST)

        # --- 3. 路径规划 (上帝视角) ---
        roi_start_y = int(h * 0.1) 
        pred_mask_roi = bird_eye_mask[roi_start_y:, :]
        white_pts = np.column_stack(np.where(pred_mask_roi == 1))
        
        err_x, l_k = 0, 0
        pts_final_bird = None
        pts_final_orig = None
        
        if len(white_pts) > 50:
            white_pts = white_pts[::8]
            ys = white_pts[:, 0] + roi_start_y
            xs = white_pts[:, 1]
            
            if len(np.unique(ys)) > 5:
                max_y, min_y = np.max(ys), np.min(ys)
                y_range = max_y - min_y

                line_bound = int(max_y - y_range * 0.7)
                line_mask = ys >= line_bound
                
                if np.sum(line_mask) > 5:
                    l_k, l_b = np.polyfit(ys[line_mask], xs[line_mask], 1)
                else:
                    l_k, l_b = 0, w // 2
                
                poly_coeffs = np.polyfit(ys, xs, 2)
                plot_y = np.linspace(max_y, min_y, num=40)
                plot_x_line = l_k * plot_y + l_b
                plot_x_curve = np.polyval(poly_coeffs, plot_y)

                t_arr = (max_y - plot_y) / (y_range + 0.1)
                alpha = t_arr ** 2
                plot_x_final = (1 - alpha) * plot_x_line + alpha * plot_x_curve

                padded_x = np.pad(plot_x_final, (Config.SMOOTH_WINDOW//2, Config.SMOOTH_WINDOW//2), mode='edge')
                plot_x_final = np.convolve(padded_x, np.ones(Config.SMOOTH_WINDOW)/Config.SMOOTH_WINDOW, mode='valid')

                err_x = plot_x_final[0] - (w // 2)
                
                pts_final_bird = np.vstack((plot_x_final, plot_y)).astype(np.int32).T.reshape((-1, 1, 2))
                pts_final_orig = cv2.perspectiveTransform(np.float32(pts_final_bird), M_inv).astype(np.int32)
            
        # --- 渲染可视化 ---
        colored_roi = np.zeros_like(vis_img)
        colored_roi[mask_full == 1] = [0, 0, 255] # 红色蒙版
        vis_img = cv2.addWeighted(vis_img, 1 - Config.MASK_ALPHA, colored_roi, Config.MASK_ALPHA, 0)
        
        # 原图上画拟合线
        if pts_final_orig is not None:
            cv2.polylines(vis_img, [pts_final_orig], False, (255, 0, 255), 4)

        # PIP 画中画显示上帝视角
        pip_img = cv2.cvtColor(bird_eye_mask * 255, cv2.COLOR_GRAY2BGR)
        if pts_final_bird is not None:
            cv2.polylines(pip_img, [pts_final_bird], False, (0, 255, 0), 4)
        
        pip_h, pip_w = h // 3, w // 3
        pip_resized = cv2.resize(pip_img, (pip_w, pip_h))
        vis_img[0:pip_h, w-pip_w:w] = pip_resized
        cv2.rectangle(vis_img, (w-pip_w, 0), (w, pip_h), (255, 255, 255), 2)

        # 打印耗时等信息
        cv2.putText(vis_img, f"Infer: {infer_time:.1f}ms | Err: {err_x:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # 保存图片
        save_path = os.path.join(Config.OUTPUT_DIR, file_name)
        cv2.imwrite(save_path, vis_img)
        print(f"✨ 成功处理: {file_name} -> 耗时 {infer_time:.1f} ms")

    rknn.release()
    print("-" * 40 + f"\n✅ 所有图片处理完毕！请检查 {Config.OUTPUT_DIR} 文件夹。")

if __name__ == "__main__":
    main()
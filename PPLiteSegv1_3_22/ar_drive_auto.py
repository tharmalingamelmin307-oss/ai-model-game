import os
import cv2
import numpy as np
import time
import threading
from rknnlite.api import RKNNLite

# ================= 1. 配置信息 =================
MODEL_PATH = './model/ppliteseg_int8.rknn' 
STREAM_URL = "http://192.168.31.189:8080/ar_feed"
TARGET_RES = (960, 720)
TARGET_CLASS_ID = 1 # 假设你要找的线/路面类别ID是1

# ================= 2. 低延迟视频流类 =================
class VideoStreamWidget(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        self.status, self.frame = self.capture.read()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
            time.sleep(0.01)

    def read(self):
        return self.status, self.frame

    def release(self):
        self.capture.release()

# ================= 3. 主流程 =================
def main():
    print("🛠️ 正在初始化 RKNN NPU 模型...")
    rknn = RKNNLite()
    if rknn.load_rknn(MODEL_PATH) != 0:
        print("❌ 模型加载失败")
        return
    if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
        print("❌ NPU 初始化失败")
        return

    print("--> 正在启动低延迟视频流读取线程...")
    video_stream = VideoStreamWidget(STREAM_URL)
    time.sleep(1) # 等待摄像头缓冲
    print("🚀 自动驾驶寻线系统已启动！按 'q' 键退出...")

    while True:
        loop_start = time.time()
        
        # --- 获取最新一帧 ---
        ret, frame = video_stream.read()
        if not ret or frame is None:
            continue
            
        # --- 预处理 ---
        img_resized = cv2.resize(frame, TARGET_RES)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img_rgb, axis=0)

        # --- NPU 推理 ---
        t_infer_start = time.time()
        outputs = rknn.inference(inputs=[input_data])
        npu_time_ms = (time.time() - t_infer_start) * 1000
        
        # --- 后处理提取 Mask ---
        pred_mask = np.squeeze(outputs[0])
        if len(pred_mask.shape) == 3:
            pred_mask = np.argmax(pred_mask, axis=0)
            
        # 构建二值化掩膜 (同你代码中的 mask_255)
        mask_255 = np.zeros_like(pred_mask, dtype=np.uint8)
        mask_255[pred_mask == TARGET_CLASS_ID] = 255

        # ====================================================
        # --- 阶段 2: 放宽限制版拟合算法 (直接嵌入) ---
        # ====================================================
        vis_img = img_resized.copy()
        height, width = vis_img.shape[:2]

        # 优化版紫色半透明覆盖 (比 addWeighted 快)
        is_target = (mask_255 == 255)
        # BGR格式：Purple = (255, 0, 255)
        vis_img[is_target] = vis_img[is_target] * 0.5 + np.array([255, 0, 255], dtype=np.uint8) * 0.5

        white_pts = np.column_stack(np.where(is_target)) 

        if len(white_pts) > 50:
            ys = white_pts[:, 0]
            xs = white_pts[:, 1]

            max_y = np.max(ys)
            min_y = np.min(ys)
            y_range = max_y - min_y
            
            if y_range >= 20: 
                # 1. 质心提取边界
                top_bound = int(min_y + y_range * 0.02)
                bottom_bound = int(max_y - y_range * 0.02)
                
                # 2. 直线拟合边界
                line_bound = int(max_y - y_range * 0.75)

                # 画出区域分割线
                cv2.line(vis_img, (0, top_bound), (width, top_bound), (255, 255, 255), 1) 
                cv2.line(vis_img, (0, bottom_bound), (width, bottom_bound), (255, 255, 255), 1) 
                cv2.line(vis_img, (0, line_bound), (width, line_bound), (0, 255, 255), 1) 

                # --- 提取质心 ---
                top_mask = ys <= top_bound
                if np.sum(top_mask) > 0:
                    tip_y = int(np.mean(ys[top_mask]))
                    tip_x = int(np.mean(xs[top_mask]))
                else:
                    tip_y, tip_x = ys[np.argmin(ys)], xs[np.argmin(ys)]

                bottom_mask = ys >= bottom_bound
                if np.sum(bottom_mask) > 0:
                    base_y = int(np.mean(ys[bottom_mask]))
                    base_x = int(np.mean(xs[bottom_mask]))
                else:
                    base_y, base_x = max_y, int(np.mean(xs[ys == max_y]))

                # 防止除零错误保护
                if base_y != tip_y:
                    # --- 拟合算法 ---
                    line_mask = ys >= line_bound
                    if np.sum(line_mask) > 20:
                        line_k, line_b = np.polyfit(ys[line_mask], xs[line_mask], 1)
                    else:
                        line_k, line_b = np.polyfit(ys, xs, 1)

                    poly_coeffs = np.polyfit(ys, xs, 2)

                    plot_y = np.linspace(base_y, tip_y, num=50)
                    plot_x_line = line_k * plot_y + line_b
                    plot_x_curve = np.polyval(poly_coeffs, plot_y)

                    # 融合计算
                    t = (base_y - plot_y) / (base_y - tip_y)
                    alpha = t ** 2  
                    plot_x_final = (1 - alpha) * plot_x_line + alpha * plot_x_curve

                    # --- 转换格式用于绘制 ---
                    pts_line = np.vstack((plot_x_line, plot_y)).astype(np.int32).T.reshape((-1, 1, 2))
                    pts_curve = np.vstack((plot_x_curve, plot_y)).astype(np.int32).T.reshape((-1, 1, 2))
                    pts_final = np.vstack((plot_x_final, plot_y)).astype(np.int32).T.reshape((-1, 1, 2))

                    # 画出内部的三根线
                    cv2.polylines(vis_img, [pts_line], False, (0, 255, 255), 2)  # 黄线
                    cv2.polylines(vis_img, [pts_curve], False, (255, 255, 0), 2) # 青线
                    cv2.polylines(vis_img, [pts_final], False, (0, 0, 255), 6)   # 红线

                    # 画出质心
                    cv2.circle(vis_img, (base_x, base_y), 8, (0, 255, 0), -1) 
                    cv2.circle(vis_img, (tip_x, tip_y), 8, (255, 0, 0), -1)

        # ====================================================

        # --- 性能统计与图例绘制 ---
        fps = 1000 / ((time.time() - loop_start) * 1000 + 0.001)
        
        cv2.putText(vis_img, f"NPU: {npu_time_ms:.1f}ms | FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis_img, "- Purple: Raw Mask", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(vis_img, "- Yellow: Base Line", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vis_img, "- Cyan: Pure Curve", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(vis_img, "- Red: Blended Result", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # --- 显示画面 ---
        cv2.imshow("Auto Drive Vision Engine", vis_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    video_stream.release()
    rknn.release()
    cv2.destroyAllWindows()
    print("✅ 系统已安全退出。")

if __name__ == '__main__':
    main()
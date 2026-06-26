import os
import cv2
import numpy as np
import time
import threading # 导入多线程库
from rknnlite.api import RKNNLite

# ... [保留你之前的配置信息] ...
MODEL_PATH = './model/ppliteseg_int8.rknn' 
STREAM_URL = "http://192.168.31.189:8080/ar_feed"
TARGET_RES = (960, 720)

# ==========================================
# 新增：一个专门抓取最新视频帧的类
# ==========================================
class VideoStreamWidget(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        # 尝试强制将缓冲区设为1（对某些网络流有效）
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        # 启动后台线程不断读取视频流
        self.status, self.frame = self.capture.read()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # 这个循环在后台疯狂运行，确保自我更新到最新帧
        while True:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
            time.sleep(0.01) # 极短的休眠，防止 CPU 占用 100%

    def read(self):
        # 主程序调用时，直接返回内存中最新的那一帧
        return self.status, self.frame

    def release(self):
        self.capture.release()

# ==========================================

def main():
    # ... [保留你之前的 rknn 初始化代码] ...
    rknn = RKNNLite()
    rknn.load_rknn(MODEL_PATH)
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)

    # 替换原本的 cap = cv2.VideoCapture(STREAM_URL)
    print("--> 正在启动低延迟视频流读取线程...")
    video_stream = VideoStreamWidget(STREAM_URL)
    
    # 稍微等一下确保线程已经拿到第一帧
    time.sleep(1)
    
    print("🚀 实时低延迟推理已启动！")

    while True:
        loop_start = time.time()
        
        # --- 获取最新一帧 (从后台线程拿，没有延迟) ---
        ret, frame = video_stream.read()
        
        if not ret or frame is None:
            continue # 如果还没准备好，跳过这次循环
            
        # --- 预处理 ---
        img_resized = cv2.resize(frame, TARGET_RES)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img_rgb, axis=0)

        # --- 核心推理 ---
        t_infer_start = time.time()
        outputs = rknn.inference(inputs=[input_data])
        npu_time_ms = (time.time() - t_infer_start) * 1000
        
        # --- 后处理 (不变) ---
        pred_mask = np.squeeze(outputs[0])
        if len(pred_mask.shape) == 3:
            pred_mask = np.argmax(pred_mask, axis=0)
            
        road_mask = (pred_mask == 1)
        vis_img = img_resized.copy()
        vis_img[road_mask] = vis_img[road_mask] * 0.5 + np.array([0, 0, 127], dtype=np.uint8)
        
        # --- 计算帧率和显示 ---
        fps = 1000 / ((time.time() - loop_start) * 1000)
        cv2.putText(vis_img, f"NPU: {npu_time_ms:.1f}ms | FPS: {fps:.1f}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("RK3588 Real-time AR Segmentation", vis_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    video_stream.release()
    rknn.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
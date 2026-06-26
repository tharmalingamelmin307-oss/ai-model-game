import time
import struct
import numpy as np
import cv2
import threading
from queue import Queue
from multiprocessing import shared_memory, resource_tracker
from flask import Flask, Response, render_template_string
from rknnlite.api import RKNNLite

# ==============================================================================
# 1. 配置信息
# ==============================================================================
SHM_NAME = "shm_ar_video"
SHM_HEADER_SIZE = 16
STREAM_PORT = 5003

# --- AI 模型配置 ---
# 确保路径指向你最新生成的 8MB-9MB 纯血 INT8 模型
MODEL_PATH = "models/ppliteseg_576_final_int8.rknn"
MODEL_W = 576
MODEL_H = 416
ROI_WIDTH_RATIO = 0.6
MASK_ALPHA = 0.5

# --- 三核并发配置 ---
CORE_MASKS = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]

# --- 推流优化配置 ---
PREVIEW_SCALE = 0.5
JPEG_QUALITY = 75

app = Flask(__name__)

# ==============================================================================
# 2. 全局状态与多线程组件
# ==============================================================================
global_preview_frame = None
frame_lock = threading.Lock()

# 任务队列 (maxsize=1 确保 NPU 永远处理最新帧，不排队产生积压延时)
input_queue = Queue(maxsize=1) 

# --- FPS 统计全局变量 ---
total_frames_processed = 0
fps_start_time = time.time()
display_fps = 0.0
fps_lock = threading.Lock()

# ==============================================================================
# 3. 核心修复：防止客户端退出时销毁主系统共享内存
# ==============================================================================
def remove_shm_from_resource_tracker():
    try:
        resource_tracker.unregister('/' + SHM_NAME, 'shared_memory')
    except Exception:
        pass

# ==============================================================================
# 4. NPU 工作线程 (消费者 - 推理与后处理)
# ==============================================================================
def npu_worker(core_id):
    global global_preview_frame, total_frames_processed, fps_start_time, display_fps
    
    rknn = RKNNLite()
    print(f"--> 核心 {core_id} 正在加载模型...")
    if rknn.load_rknn(MODEL_PATH) != 0 or rknn.init_runtime(core_mask=CORE_MASKS[core_id]) != 0:
        print(f"❌ NPU Core {core_id} 初始化失败！")
        return
    print(f"✅ NPU Core {core_id} 已就绪")

    while True:
        # 获取生产者放入的原始帧和坐标信息
        frame_data = input_queue.get()
        if frame_data is None: break
        
        orig_frame, x1, y1, x2, y2, roi_w, roi_h = frame_data
        
        # --- A. 预处理 ---
        roi = orig_frame[y1:y2, x1:x2]
        # 注意：NPU Config 已包含归一化，这里只需 resize + RGB 转换
        input_data = cv2.resize(roi, (MODEL_W, MODEL_H))
        input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(input_data, axis=0)

        # --- B. 纯 NPU 推理 (会释放 GIL 锁) ---
        outputs = rknn.inference(inputs=[input_data])
        
        # --- C. 极速后处理 (针对双通道 INT8) ---
        # outputs[0] 的 shape 为 [1, 2, 416, 576]
        out_tensor = outputs[0][0] # 得到 [2, 416, 576]
        
        # 向量化比较：通道 1(赛道) 概率 > 通道 0(背景) 概率
        mask = (out_tensor[1] > out_tensor[0]).astype(np.uint8)
        
        # --- D. 渲染可视化 ---
        color_mask = cv2.resize(mask, (roi_w, y2 - y1), interpolation=cv2.INTER_NEAREST)
        colored_roi = np.zeros_like(roi)
        # 将识别到的赛道区域涂成红色 [B, G, R]
        colored_roi[color_mask == 1] = [0, 0, 255]
        
        # 混合原图与红色遮罩
        orig_frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - MASK_ALPHA, colored_roi, MASK_ALPHA, 0)
        cv2.rectangle(orig_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # --- E. FPS 统计 ---
        with fps_lock:
            total_frames_processed += 1
            now = time.time()
            duration = now - fps_start_time
            if duration >= 1.0:
                display_fps = total_frames_processed / duration
                total_frames_processed = 0
                fps_start_time = now

        # --- F. 最终画面交付 ---
        preview = cv2.resize(orig_frame, (0, 0), fx=PREVIEW_SCALE, fy=PREVIEW_SCALE) if PREVIEW_SCALE != 1.0 else orig_frame
        
        # 在预览图上打上 FPS 标签和处理核心 ID
        cv2.putText(preview, f"Throughput FPS: {display_fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(preview, f"NPU Core: {core_id}", (10, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        with frame_lock:
            global_preview_frame = preview
            
    rknn.release()

# ==============================================================================
# 5. 共享内存拉取线程 (生产者)
# ==============================================================================
def ai_producer_thread():
    print(f"--> 📡 启动共享内存拉取线程...")
    
    while True:
        shm = None
        try:
            print("⏳ 正在搜寻共享内存...")
            while True:
                try:
                    shm = shared_memory.SharedMemory(name=SHM_NAME)
                    remove_shm_from_resource_tracker()
                    print("✅ 成功接入共享内存！")
                    break
                except FileNotFoundError:
                    time.sleep(1.0)

            last_fid = 0
            while True:
                # 读取 FID 头部，判断是否为新帧
                header = bytes(shm.buf[:SHM_HEADER_SIZE])
                fid, w, h = struct.unpack('QII', header)
                
                if fid == last_fid:
                    time.sleep(0.002) # 极小等待减少 CPU 空转
                    continue
                last_fid = fid
                
                # 读取图像数据
                size = w * h * 3
                img_view = np.ndarray((h, w, 3), dtype=np.uint8, buffer=shm.buf[SHM_HEADER_SIZE : SHM_HEADER_SIZE+size])
                frame = img_view.copy() # 深拷贝防止内存竞争
                del img_view
                
                # 基础图像翻转与颜色空间还原 (适应 Aero-Twin 推流格式)
                frame = cv2.flip(frame, 0)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # 计算 ROI 逻辑坐标 (在原图上操作)
                roi_w = int(w * ROI_WIDTH_RATIO)
                roi_h = int(roi_w * (MODEL_H / MODEL_W)) 
                x1 = (w - roi_w) // 2
                y1 = max(0, h - roi_h)
                x2 = x1 + roi_w
                y2 = h

                # 如果队列已满，强行丢弃旧图塞入新图
                if input_queue.full():
                    try: input_queue.get_nowait()
                    except: pass
                
                input_queue.put((frame, x1, y1, x2, y2, roi_w, roi_h))

        except Exception as e:
            print(f"⚠️ 生产者线程异常: {e}，正在重连...")
            time.sleep(1.0)
        finally:
            if shm: 
                try: shm.close()
                except: pass

# ==============================================================================
# 6. Web 推流服务 (Flask)
# ==============================================================================
def generate_web_stream():
    while True:
        with frame_lock:
            current_frame = None if global_preview_frame is None else global_preview_frame.copy()
        
        if current_frame is None:
            time.sleep(0.01); continue
            
        ret, buffer = cv2.imencode('.jpg', current_frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ret: continue
            
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.02) # 控制 Web 刷新率

@app.route('/')
def index():
    html = f"""
    <html>
      <body style="background-color: #121212; color: white; text-align: center; font-family: sans-serif;">
        <h2>Receiver: 三核异步 NPU 分割流</h2>
        <img src="/video_feed" style="border: 2px solid #00FF00; max-width: 100%;">
      </body>
    </html>
    """
    return render_template_string(html)

@app.route('/video_feed')
def video_feed():
    return Response(generate_web_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ==============================================================================
# 7. 启动入口
# ==============================================================================
if __name__ == "__main__":
    print("======================================================")
    print(" 🚀 Aero-Twin 三核并发 AI 服务启动 ")
    print(f" 📍 模型路径: {MODEL_PATH}")
    print(f" 📺 远程观看: http://<板子IP>:{STREAM_PORT}")
    print("======================================================")
    
    # A. 启动生产者 (拉取共享内存)
    threading.Thread(target=ai_producer_thread, daemon=True).start()
    
    # B. 启动 3 个 NPU 工作核心 (异步并行消费)
    for i in range(3):
        threading.Thread(target=npu_worker, args=(i,), daemon=True).start()
    
    # C. 启动 Flask 服务
    app.run(host='0.0.0.0', port=STREAM_PORT, threaded=True)
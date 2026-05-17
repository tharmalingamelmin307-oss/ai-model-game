import time
import struct
import numpy as np
import cv2
import threading
from multiprocessing import shared_memory, resource_tracker
from flask import Flask, Response, render_template_string

# ==============================================================================
# 配置信息
# ==============================================================================
SHM_NAME = "shm_ar_video"
SHM_HEADER_SIZE = 16
STREAM_PORT = 5003  # 独立推流端口

# --- 推流优化配置 ---
PREVIEW_SCALE = 1.0  # 纯视频模式下，如果不卡可以设为 1.0 看原画质；如果嫌带宽大可以改为 0.5
JPEG_QUALITY = 100    # 纯视频质量可以稍微调高一点

app = Flask(__name__)

# ==============================================================================
# 全局状态与多线程锁
# ==============================================================================
global_preview_frame = None
frame_lock = threading.Lock()

# ==============================================================================
# 核心修复：防止客户端退出时误删服务器的 SHM
# ==============================================================================
def remove_shm_from_resource_tracker():
    try:
        resource_tracker.unregister('/' + SHM_NAME, 'shared_memory')
    except Exception:
        pass 

# ==============================================================================
# 纯粹的视频抓取线程 (无 AI)
# ==============================================================================
def video_worker_thread():
    global global_preview_frame
    
    print(f"--> 🚀 启动后台视频拉取线程 (纯视频模式)")
    
    while True: # 断线重连外层循环
        shm = None
        try:
            print("⏳ 正在等待主系统启动共享内存...")
            while True:
                try:
                    shm = shared_memory.SharedMemory(name=SHM_NAME)
                    remove_shm_from_resource_tracker()
                    print("✅ 成功接入主系统共享内存！")
                    break
                except FileNotFoundError:
                    time.sleep(1.0)

            last_fid = 0
            fps_t = time.time()
            fps_n = 0
            cur_fps = 0.0

            while True:
                try:
                    # 读取头部
                    header = bytes(shm.buf[:SHM_HEADER_SIZE])
                    fid, w, h = struct.unpack('QII', header)
                    
                    if fid == last_fid:
                        time.sleep(0.002)
                        continue
                    last_fid = fid
                    
                    # 极速拷贝数据
                    size = w * h * 3
                    img_view = np.ndarray((h, w, 3), dtype=np.uint8, buffer=shm.buf[SHM_HEADER_SIZE : SHM_HEADER_SIZE+size])
                    frame = img_view.copy() 
                    del img_view 
                    
                    # 基础图像处理
                    frame = cv2.flip(frame, 0)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # 画面缩放 (如果设置了的话)
                    if PREVIEW_SCALE != 1.0:
                        preview_frame = cv2.resize(frame, (0, 0), fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)
                    else:
                        preview_frame = frame
                    
                    # 计算内部拉取帧率
                    fps_n += 1
                    if time.time() - fps_t >= 1.0:
                        cur_fps = fps_n / (time.time() - fps_t)
                        fps_n = 0; fps_t = time.time()
                    
                    cv2.putText(preview_frame, f"Raw SHM FPS: {cur_fps:.1f} (No AI)", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    # 更新全局变量供 Web 线程提取
                    with frame_lock:
                        global_preview_frame = preview_frame.copy()

                except (ValueError, struct.error, BufferError):
                    print("⚠️ 共享内存读取异常，准备重连...")
                    break 

        except Exception as e:
            print(f"❌ 发生异常: {e}")
            time.sleep(1.0)
        finally:
            if shm: 
                try: shm.close()
                except: pass

# ==============================================================================
# Web 推流生成器 (消费者)
# ==============================================================================
def generate_web_stream():
    global global_preview_frame
    while True:
        with frame_lock:
            current_frame = None if global_preview_frame is None else global_preview_frame.copy()
        
        if current_frame is None:
            time.sleep(0.01)
            continue
            
        ret, buffer = cv2.imencode('.jpg', current_frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ret: continue
            
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        # 控制最高推流帧率 (~33 FPS)，避免不必要的网络拥堵
        time.sleep(0.03)

# ==============================================================================
# Flask 路由
# ==============================================================================
@app.route('/')
def index():
    html = f"""
    <html>
      <head><title>Raw Video Receiver</title></head>
      <body style="background-color: #121212; color: white; text-align: center; font-family: sans-serif;">
        <h2>Receiver: 纯视频流 (无 AI)</h2>
        <img src="/video_feed" style="border: 2px solid #00FFFF; max-width: 100%;">
      </body>
    </html>
    """
    return render_template_string(html)

@app.route('/video_feed')
def video_feed():
    return Response(generate_web_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("======================================================")
    print(" 📡 独立 Receiver 服务启动 (纯净视频模式)")
    print(" ⚠️ 请确保主系统的 UI 预览窗口处于关闭状态")
    print(f" 👉 远程观看地址: http://<板子IP>:{STREAM_PORT}")
    print("======================================================")
    
    # 启动后台视频拉取线程
    threading.Thread(target=video_worker_thread, daemon=True).start()
    
    # 启动 Web 推流服务
    app.run(host='0.0.0.0', port=STREAM_PORT, threaded=True)
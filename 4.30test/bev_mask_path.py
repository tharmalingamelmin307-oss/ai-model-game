import time
import struct
import numpy as np
import cv2
import threading
import serial
from queue import Queue
from multiprocessing import shared_memory, resource_tracker
from flask import Flask, Response, render_template_string
from rknnlite.api import RKNNLite

# ==============================================================================
# 1. 配置信息 (已更新最新的 IPM 标定参数)
# ==============================================================================
class Config:
    SHM_NAME = "shm_ar_video"
    SHM_HEADER_SIZE = 16
    STREAM_PORT = 5003
    
    # 路径配置
    SEG_MODEL = "models/ppliteseg_576_final_int8.rknn"

    TARGET_RES = (960, 720) 
    SEG_SIZE = (576, 416)   
    
    ROI_TOP_CUT_RATIO = 0.3 
    MASK_ALPHA = 0.4
    
    # ====== 🚀 刚刚更新的标定参数 ======
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
    # =================================

    SERIAL_PORT = '/dev/ttyS2'
    BAUD_RATE = 115200
    SERVO_CENTER = 750
    SERVO_MIN, SERVO_MAX = 590, 910
    MOTOR_MIN_SPEED = 10
    MOTOR_MAX_SPEED = 50 
    
    SMOOTH_WINDOW = 5       
    KP = 0.16               
    KD = 160.0              
    
    JPEG_QUALITY = 75

app = Flask(__name__)

# ==============================================================================
# 2. 全局状态
# ==============================================================================
global_preview_frame = None
frame_lock = threading.Lock()
init_lock = threading.Lock() 

seg_queue = Queue(maxsize=1) 

global_control_data = {"error_x": 0, "line_k": 0} 
data_lock = threading.Lock()

fps_stats = {"seg_frames": 0, "seg_fps": 0.0}
fps_start_time = time.time()

def remove_shm_from_resource_tracker():
    try: resource_tracker.unregister('/' + Config.SHM_NAME, 'shared_memory')
    except: pass


# ==============================================================================
# 3. PPLiteSeg 路径规划工作线程 (基于新 IPM 参数)
# ==============================================================================
def seg_worker(core_id):
    global global_preview_frame, fps_start_time
    try:
        rknn = RKNNLite()
        with init_lock: 
            if rknn.load_rknn(Config.SEG_MODEL) != 0 or rknn.init_runtime(core_mask=core_id) != 0:
                return
        print(f"✅ Seg Core {core_id} 已就绪", flush=True)
    except: return

    while True:
        try:
            vis_img = seg_queue.get()
            if vis_img is None: break
            
            # --- 预处理 ---
            blob = cv2.resize(vis_img, Config.SEG_SIZE, interpolation=cv2.INTER_NEAREST)
            blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)

            # --- 推理与掩码 ---
            out = rknn.inference(inputs=[np.expand_dims(blob, axis=0)])[0][0]
            mask = (out[1] > out[0]).astype(np.uint8) 
            
            # --- 1. 将 Mask 放大到屏幕分辨率 ---
            mask_full = cv2.resize(mask, Config.TARGET_RES, interpolation=cv2.INTER_NEAREST)

            # --- 2. 逆透视变换 ---
            w, h = Config.TARGET_RES
            src_points = np.float32([[x * w, y * h] for x, y in Config.SRC_PTS])
            dst_points = np.float32([[x * w, y * h] for x, y in Config.DST_PTS])
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            M_inv = cv2.getPerspectiveTransform(dst_points, src_points)
            
            bird_eye_mask = cv2.warpPerspective(mask_full, M, (w, h), flags=cv2.INTER_NEAREST)

            # --- 3. 路径规划 (上帝视角) ---
            # 现在的视野很大，我们主要看画面下半部分的路径
            roi_start_y = int(h * 0.1) # 几乎看全图
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

                    # 拟合与平滑
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

                    # 误差计算
                    err_x = plot_x_final[0] - (w // 2)
                    
                    # 坐标转换用于渲染
                    pts_final_bird = np.vstack((plot_x_final, plot_y)).astype(np.int32).T.reshape((-1, 1, 2))
                    pts_final_orig = cv2.perspectiveTransform(np.float32(pts_final_bird), M_inv).astype(np.int32)
                
            # --- 状态更新 ---
            with data_lock:
                global_control_data["error_x"] = err_x
                global_control_data["line_k"] = l_k
                fps_stats["seg_frames"] += 1
                now = time.time()
                if now - fps_start_time >= 1.0:
                    fps_stats["seg_fps"] = fps_stats["seg_frames"] / (now - fps_start_time)
                    fps_stats["seg_frames"] = 0
                    fps_start_time = now

            # --- 渲染可视化 ---
            colored_roi = np.zeros_like(vis_img)
            colored_roi[mask_full == 1] = [0, 0, 255]
            vis_img = cv2.addWeighted(vis_img, 1 - Config.MASK_ALPHA, colored_roi, Config.MASK_ALPHA, 0)
            
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

            servo_pwm = int(Config.SERVO_CENTER + (err_x * Config.KP) - (l_k * Config.KD))
            cv2.putText(vis_img, f"Seg FPS:{fps_stats['seg_fps']:.1f} PWM:{servo_pwm}", (20, 30), 1, 1.5, (0, 255, 0), 2)

            with frame_lock:
                global_preview_frame = vis_img
                
        except Exception as e:
            pass

# ==============================================================================
# 4. 串口控制线程 
# ==============================================================================
def serial_control_thread():
    try:
        ser = serial.Serial(Config.SERIAL_PORT, Config.BAUD_RATE, timeout=0.1)
    except: ser = None
        
    while True:
        with data_lock:
            err_x = global_control_data["error_x"]
            l_k = global_control_data["line_k"]
            
        servo_pwm = int(Config.SERVO_CENTER + (err_x * Config.KP) - (l_k * Config.KD))
        motor_pwm = Config.MOTOR_MAX_SPEED - int(abs(l_k) * 15)
        
        servo_pwm = max(Config.SERVO_MIN, min(Config.SERVO_MAX, servo_pwm))
        motor_pwm = max(Config.MOTOR_MIN_SPEED, min(Config.MOTOR_MAX_SPEED, motor_pwm))
        
        if ser:
            packet = struct.pack('<BBhhBB', 0xAA, 0x55, motor_pwm, servo_pwm, 0x0D, 0x0A)
            ser.write(packet)
            
        time.sleep(0.01) 

# ==============================================================================
# 5. 共享内存拉取与其余逻辑保持不变
# ==============================================================================
def ai_producer_thread():
    while True:
        shm = None
        try:
            while True:
                try:
                    shm = shared_memory.SharedMemory(name=Config.SHM_NAME)
                    remove_shm_from_resource_tracker()
                    break
                except FileNotFoundError:
                    time.sleep(1.0)
            last_fid = 0
            while True:
                header = bytes(shm.buf[:Config.SHM_HEADER_SIZE])
                fid, w, h = struct.unpack('QII', header)
                if fid == last_fid:
                    time.sleep(0.002); continue
                last_fid = fid
                img_view = np.ndarray((h, w, 3), dtype=np.uint8, buffer=shm.buf[Config.SHM_HEADER_SIZE : Config.SHM_HEADER_SIZE+w*h*3])
                frame = img_view.copy()
                frame = cv2.flip(frame, 0)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                vis_img = cv2.resize(frame, Config.TARGET_RES, interpolation=cv2.INTER_NEAREST)
                if seg_queue.full():
                    try: seg_queue.get_nowait()
                    except: pass
                seg_queue.put(vis_img.copy())
        except Exception:
            time.sleep(1.0)
        finally:
            if shm: shm.close()

def generate_web_stream():
    while True:
        with frame_lock:
            current_frame = None if global_preview_frame is None else global_preview_frame.copy()
        if current_frame is None:
            time.sleep(0.01); continue
        ret, buffer = cv2.imencode('.jpg', current_frame, [int(cv2.IMWRITE_JPEG_QUALITY), Config.JPEG_QUALITY])
        if not ret: continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.02) 

@app.route('/')
def index():
    return render_template_string('<html><body style="background:#000;text-align:center;margin:0;"><img src="/video_feed" style="max-width:100%;"></body></html>')

@app.route('/video_feed')
def video_feed():
    return Response(generate_web_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    threading.Thread(target=ai_producer_thread, daemon=True).start()
    threading.Thread(target=serial_control_thread, daemon=True).start()
    time.sleep(0.1)
    threading.Thread(target=seg_worker, args=(RKNNLite.NPU_CORE_0,), daemon=True).start()
    time.sleep(0.1)
    threading.Thread(target=seg_worker, args=(RKNNLite.NPU_CORE_1,), daemon=True).start()
    app.run(host='0.0.0.0', port=Config.STREAM_PORT, threaded=True)
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
# 1. 配置信息 (极简纯循线版)
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
    
    SERIAL_PORT = '/dev/ttyS2'
    BAUD_RATE = 115200
    SERVO_CENTER = 750
    SERVO_MIN, SERVO_MAX = 590, 910
    
    # ======== 修改点：速度范围限制为 10 ~ 50 ========
    MOTOR_MIN_SPEED = 10
    MOTOR_MAX_SPEED = 50 
    # ================================================
    
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
# 3. PPLiteSeg 路径规划工作线程 (单线计算)
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

    roi_start_y_seg = int(Config.SEG_SIZE[1] * Config.ROI_TOP_CUT_RATIO)

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
            
            # --- 核心路径规划 ---
            pred_mask_roi = mask[roi_start_y_seg:, :]
            white_pts = np.column_stack(np.where(pred_mask_roi == 1))
            
            err_x, l_k = 0, 0
            pts_final = None
            
            if len(white_pts) > 50:
                white_pts = white_pts[::8] # 降采样加速计算
                
                scale_h = Config.TARGET_RES[1] / Config.SEG_SIZE[1]
                scale_w = Config.TARGET_RES[0] / Config.SEG_SIZE[0]
                
                ys = (white_pts[:, 0] + roi_start_y_seg) * scale_h
                xs = white_pts[:, 1] * scale_w
                
                # ====== 核心控制计算：单一主路径 ======
                if len(np.unique(ys)) > 5:
                    max_y, min_y = np.max(ys), np.min(ys)
                    y_range = max_y - min_y

                    # 1. 拟合
                    line_bound = int(max_y - y_range * 0.7)
                    line_mask = ys >= line_bound
                    
                    if np.sum(line_mask) > 5:
                        l_k, l_b = np.polyfit(ys[line_mask], xs[line_mask], 1)
                    else:
                        l_k, l_b = 0, Config.TARGET_RES[0] // 2
                    
                    poly_coeffs = np.polyfit(ys, xs, 2)
                    plot_y = np.linspace(max_y, min_y, num=40)
                    plot_x_line = l_k * plot_y + l_b
                    plot_x_curve = np.polyval(poly_coeffs, plot_y)

                    # 线性与二项式融合
                    t_arr = (max_y - plot_y) / (y_range + 0.1)
                    alpha = t_arr ** 2
                    plot_x_final = (1 - alpha) * plot_x_line + alpha * plot_x_curve

                    # 2. 平滑处理
                    padded_x = np.pad(plot_x_final, (Config.SMOOTH_WINDOW//2, Config.SMOOTH_WINDOW//2), mode='edge')
                    plot_x_final = np.convolve(padded_x, np.ones(Config.SMOOTH_WINDOW)/Config.SMOOTH_WINDOW, mode='valid')

                    # 3. 误差计算 (以画面最下方的点为基准)
                    err_x = plot_x_final[0] - (Config.TARGET_RES[0] // 2)
                    pts_final = np.vstack((plot_x_final, plot_y)).astype(np.int32).T.reshape((-1, 1, 2))
                
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
            color_mask = cv2.resize(mask, Config.TARGET_RES, interpolation=cv2.INTER_NEAREST)
            colored_roi = np.zeros_like(vis_img)
            colored_roi[color_mask == 1] = [0, 0, 255]
            vis_img = cv2.addWeighted(vis_img, 1 - Config.MASK_ALPHA, colored_roi, Config.MASK_ALPHA, 0)
            
            if pts_final is not None:
                # 仅保留单一的粉色/紫红色引导线
                cv2.polylines(vis_img, [pts_final], False, (255, 0, 255), 4)

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
        
        # ======== 修改点：弯道减速逻辑及钳制区间 ========
        # 因为速度总跨度从 350 缩小到了 40，所以计算曲率影响的系数从 120 缩小到了 15
        motor_pwm = Config.MOTOR_MAX_SPEED - int(abs(l_k) * 15)
        
        servo_pwm = max(Config.SERVO_MIN, min(Config.SERVO_MAX, servo_pwm))
        motor_pwm = max(Config.MOTOR_MIN_SPEED, min(Config.MOTOR_MAX_SPEED, motor_pwm))
        # ================================================
        
        if ser:
            packet = struct.pack('<BBhhBB', 0xAA, 0x55, motor_pwm, servo_pwm, 0x0D, 0x0A)
            ser.write(packet)
            
        time.sleep(0.01) 
        # if ser:
        #     packet = struct.pack('<BBhhBB', 0xAA, 0x55, 0, servo_pwm, 0x0D, 0x0A)
        #     ser.write(packet)
            
        # time.sleep(0.01) 

# ==============================================================================
# 5. 共享内存拉取
# ==============================================================================
def ai_producer_thread():
    print("--> 📡 启动共享内存拉取线程...", flush=True)
    while True:
        shm = None
        try:
            while True:
                try:
                    shm = shared_memory.SharedMemory(name=Config.SHM_NAME)
                    remove_shm_from_resource_tracker()
                    print("✅ 成功接入共享内存！", flush=True)
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

        except Exception as e:
            time.sleep(1.0)
        finally:
            if shm: 
                try: shm.close()
                except: pass

# ==============================================================================
# 6. Web 推流服务
# ==============================================================================
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
    html = '<html><body style="background:#000;text-align:center;margin:0;"><img src="/video_feed" style="max-width:100%;"></body></html>'
    return render_template_string(html)

@app.route('/video_feed')
def video_feed():
    return Response(generate_web_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("======================================================", flush=True)
    print(" 🚀 Aero-Twin [单线极简纯循线控制版 - 速度10~50] 启动 ", flush=True)
    print("======================================================", flush=True)
    
    threading.Thread(target=ai_producer_thread, daemon=True).start()
    threading.Thread(target=serial_control_thread, daemon=True).start()
    
    time.sleep(0.1)
    threading.Thread(target=seg_worker, args=(RKNNLite.NPU_CORE_0,), daemon=True).start()
    time.sleep(0.1)
    threading.Thread(target=seg_worker, args=(RKNNLite.NPU_CORE_1,), daemon=True).start()
    
    app.run(host='0.0.0.0', port=Config.STREAM_PORT, threaded=True)
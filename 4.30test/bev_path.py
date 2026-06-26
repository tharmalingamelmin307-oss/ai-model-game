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
# 1. 配置信息 (加入 IPM 逆透视配置)
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
    
    # ====== 逆透视映射矩阵配置 (由标定工具生成) ======
    # 这组参数需要根据你的实际摄像头角度微调，直到小窗里的直道看起来是平行的
    SRC_PTS = np.float32([
        [0.300, 0.600],
        [0.700, 0.600],
        [0.800, 0.900],
        [0.200, 0.900],
    ])

    DST_PTS = np.float32([
        [0.300, 0.500],
        [0.700, 0.500],
        [0.700, 1.000],
        [0.300, 1.000],
    ])
    # =================================================

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
# 3. PPLiteSeg 路径规划工作线程 (基于鸟瞰图计算)
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
            # vis_img 是我们需要渲染并最终显示的彩色原图 (BGR)
            vis_img = seg_queue.get()
            if vis_img is None: break
            
            # --- 预处理 ---
            blob = cv2.resize(vis_img, Config.SEG_SIZE, interpolation=cv2.INTER_NEAREST)
            blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)

            # --- 推理与掩码 ---
            out = rknn.inference(inputs=[np.expand_dims(blob, axis=0)])[0][0]
            # mask 是 576x416 的单通道 0/1 掩码
            mask = (out[1] > out[0]).astype(np.uint8) 
            
            # --- 1. 将 Mask 放大到屏幕分辨率 (960x720) ---
            mask_full = cv2.resize(mask, Config.TARGET_RES, interpolation=cv2.INTER_NEAREST)

            # --- 2. 准备逆透视变换矩阵 ---
            w, h = Config.TARGET_RES
            src_points = np.float32([[x * w, y * h] for x, y in Config.SRC_PTS])
            dst_points = np.float32([[x * w, y * h] for x, y in Config.DST_PTS])
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            M_inv = cv2.getPerspectiveTransform(dst_points, src_points) # 用于将线画回原图
            
            # 这一步是为了算法计算路径，依然使用 Mask 进行变换
            bird_eye_mask = cv2.warpPerspective(mask_full, M, (w, h), flags=cv2.INTER_NEAREST)

            # --- 3. 核心路径规划 (完全在鸟瞰图坐标系下进行) ---
            roi_start_y = int(h * Config.ROI_TOP_CUT_RATIO)
            pred_mask_roi = bird_eye_mask[roi_start_y:, :]
            white_pts = np.column_stack(np.where(pred_mask_roi == 1))
            
            err_x, l_k = 0, 0
            pts_final_bird = None
            pts_final_orig = None
            
            if len(white_pts) > 50:
                white_pts = white_pts[::8] # 降采样加速计算
                
                # y 和 x 已经是 960x720 坐标系下的物理位置
                ys = white_pts[:, 0] + roi_start_y
                xs = white_pts[:, 1]
                
                if len(np.unique(ys)) > 5:
                    max_y, min_y = np.max(ys), np.min(ys)
                    y_range = max_y - min_y

                    # 1. 拟合
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

                    # 线性与二项式融合
                    t_arr = (max_y - plot_y) / (y_range + 0.1)
                    alpha = t_arr ** 2
                    plot_x_final = (1 - alpha) * plot_x_line + alpha * plot_x_curve

                    # 2. 平滑处理
                    padded_x = np.pad(plot_x_final, (Config.SMOOTH_WINDOW//2, Config.SMOOTH_WINDOW//2), mode='edge')
                    plot_x_final = np.convolve(padded_x, np.ones(Config.SMOOTH_WINDOW)/Config.SMOOTH_WINDOW, mode='valid')

                    # 3. 误差计算 (在平坦的鸟瞰图中，车位于画面正中间)
                    err_x = plot_x_final[0] - (w // 2)
                    
                    # 生成鸟瞰图中的点集 (用于并在小窗中显示)
                    pts_final_bird = np.vstack((plot_x_final, plot_y)).astype(np.int32).T.reshape((-1, 1, 2))
                    # 转换回原视角的点集 (用于主画面显示)
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
            
            # ====== 🌟 新增/修改：准备小窗（PIP）内容 - 原画鸟瞰图 ======
            # 对彩色原图 `vis_img` 应用逆透视变换 `M`
            # 这里的 flags 使用 INTER_LINEAR 保证原图变换后质量较好
            bird_eye_org_img = cv2.warpPerspective(vis_img, M, (w, h), flags=cv2.INTER_LINEAR)
            
            if pts_final_bird is not None:
                # 在小窗的【上帝视角原图】上画绿色路径线，直观验证标定和路径规划是否准确
                cv2.polylines(bird_eye_org_img, [pts_final_bird], False, (0, 255, 0), 4)
            # ============================================================

            # 1. 在主图上绘制半透明赛道蒙版 (依然基于原视角)
            colored_roi = np.zeros_like(vis_img)
            colored_roi[mask_full == 1] = [0, 0, 255] # 红色蒙版
            vis_img = cv2.addWeighted(vis_img, 1 - Config.MASK_ALPHA, colored_roi, Config.MASK_ALPHA, 0)
            
            # 2. 在主图上绘制扭曲回原视角的路径线 (粉色)
            if pts_final_orig is not None:
                cv2.polylines(vis_img, [pts_final_orig], False, (255, 0, 255), 4)

            # 3. 将准备好的上帝视角原图 (PIP) 嵌入主图右上角
            # 缩小为原画面的 1/3
            pip_h, pip_w = h // 3, w // 3
            # 使用之前准备好的彩色鸟瞰图 bird_eye_org_img
            pip_resized = cv2.resize(bird_eye_org_img, (pip_w, pip_h))
            # 嵌入
            vis_img[0:pip_h, w-pip_w:w] = pip_resized
            # 加个白边框
            cv2.rectangle(vis_img, (w-pip_w, 0), (w, pip_h), (255, 255, 255), 2)
            cv2.putText(vis_img, "God View (Org)", (w-pip_w+5, 20), 1, 1.2, (0, 255, 255), 2)

            # 4. 文本信息
            servo_pwm = int(Config.SERVO_CENTER + (err_x * Config.KP) - (l_k * Config.KD))
            cv2.putText(vis_img, f"Seg FPS:{fps_stats['seg_fps']:.1f} PWM:{servo_pwm}", (20, 30), 1, 1.5, (0, 255, 0), 2)

            with frame_lock:
                global_preview_frame = vis_img
                
        except Exception as e:
            pass

# ==============================================================================
# 4-7 部分 (串口、共享内存、Web服务、主函数) 保持不变
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
    print(" 🚀 Aero-Twin [逆透视 IPM 原画Debug版] 启动 ", flush=True)
    print("======================================================", flush=True)
    
    threading.Thread(target=ai_producer_thread, daemon=True).start()
    threading.Thread(target=serial_control_thread, daemon=True).start()
    
    time.sleep(0.1)
    threading.Thread(target=seg_worker, args=(RKNNLite.NPU_CORE_0,), daemon=True).start()
    time.sleep(0.1)
    threading.Thread(target=seg_worker, args=(RKNNLite.NPU_CORE_1,), daemon=True).start()
    
    app.run(host='0.0.0.0', port=Config.STREAM_PORT, threaded=True)
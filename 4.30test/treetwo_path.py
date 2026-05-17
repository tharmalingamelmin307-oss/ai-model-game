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
# 1. 配置信息 
# ==============================================================================
class Config:
    SHM_NAME = "shm_ar_video"
    SHM_HEADER_SIZE = 16
    STREAM_PORT = 5003
    
    TARGET_BRANCH = 'LEFT'  

    SEG_MODEL = "models/ppliteseg_576_final_int8.rknn"
    TARGET_RES = (960, 720) 
    SEG_SIZE = (576, 416)   
    
    ROI_TOP_CUT_RATIO = 0.3 
    MASK_ALPHA = 0.4
    
    SRC_PTS = np.float32([[0.393, 0.639], [0.603, 0.636], [0.683, 0.799], [0.310, 0.794]])
    DST_PTS = np.float32([[0.385, 0.870], [0.615, 0.870], [0.615, 1.000], [0.385, 1.000]])

    SERIAL_PORT = '/dev/ttyS2'
    BAUD_RATE = 115200
    SERVO_CENTER = 750
    SERVO_MIN, SERVO_MAX = 550, 950 
    
    # 🚀 最低速度加 3
    MOTOR_MIN_SPEED = 13
    MOTOR_MAX_SPEED = 15 
    
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
# 3. PPLiteSeg 路径规划工作线程 
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

    morph_kernel = np.ones((7, 7), np.uint8)

    while True:
        try:
            vis_img = seg_queue.get()
            if vis_img is None: break
            
            blob = cv2.resize(vis_img, Config.SEG_SIZE, interpolation=cv2.INTER_NEAREST)
            blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)

            out = rknn.inference(inputs=[np.expand_dims(blob, axis=0)])[0][0]
            mask = (out[1] > out[0]).astype(np.uint8) 
            
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_kernel, iterations=2)
            mask = cv2.dilate(mask, morph_kernel, iterations=1)

            mask_full = cv2.resize(mask, Config.TARGET_RES, interpolation=cv2.INTER_NEAREST)

            w, h = Config.TARGET_RES
            src_points = np.float32([[x * w, y * h] for x, y in Config.SRC_PTS])
            dst_points = np.float32([[x * w, y * h] for x, y in Config.DST_PTS])
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            M_inv = cv2.getPerspectiveTransform(dst_points, src_points)
            bird_eye_mask = cv2.warpPerspective(mask_full, M, (w, h), flags=cv2.INTER_NEAREST)

            white_ys, white_xs = np.where(bird_eye_mask == 1)
            err_x, l_k = 0, 0
            pts_final_bird = None
            pts_final_orig = None
            debug_alternative_paths = [] 
            
            if len(white_ys) > 50:
                max_y = np.max(white_ys)
                search_min_y = max(int(h * 0.05), np.min(white_ys)) 
                
                STEP_Y = 40        
                DX_OPTIONS = [-120, -80, -40, -15, 0, 15, 40, 80, 120] 
                BEAM_WIDTH = 30    
                
                bottom_mask = (white_ys > max_y - 30)
                start_x = white_xs[bottom_mask][np.argmin(np.abs(white_xs[bottom_mask] - (w // 2)))] if np.any(bottom_mask) else w // 2
                    
                active_paths = [{'x': start_x, 'y': max_y, 'cost': 0.0, 'path': [(start_x, max_y)]}]
                
                curr_y = max_y
                while curr_y > search_min_y and active_paths:
                    next_y = curr_y - STEP_Y
                    if next_y < 0: break
                    new_branches = []
                    for branch in active_paths:
                        for dx in DX_OPTIONS:
                            next_x = branch['x'] + dx
                            if next_x < 0 or next_x >= w: continue
                            
                            y_min_b, y_max_b = max(0, next_y - 25), min(h, next_y + 25) 
                            x_min_b, x_max_b = max(0, next_x - 30), min(w, next_x + 30)
                            
                            if np.sum(bird_eye_mask[y_min_b:y_max_b, x_min_b:x_max_b]) > 3:
                                turn_cost = abs(dx) * 0.05 
                                new_branches.append({
                                    'x': next_x, 'y': next_y,
                                    'cost': branch['cost'] + turn_cost,
                                    'path': branch['path'] + [(next_x, next_y)]
                                })
                    
                    new_branches = sorted(new_branches, key=lambda b: b['cost'])[:BEAM_WIDTH]
                    if not new_branches: break
                    active_paths = new_branches
                    curr_y = next_y
                    
                if active_paths:
                    active_paths.sort(key=lambda b: b['x'])
                    best_branch = active_paths[0] if Config.TARGET_BRANCH == 'LEFT' else active_paths[-1]

                    for b in active_paths:
                        if len(b['path']) > 2: 
                            pts_bird_debug = np.array(b['path']).astype(np.int32).reshape((-1, 1, 2))
                            debug_alternative_paths.append((pts_bird_debug, cv2.perspectiveTransform(np.float32(pts_bird_debug), M_inv).astype(np.int32)))

                    path_pts = np.array(best_branch['path'])
                    if len(path_pts) > 2:
                        node_x, node_y = path_pts[:, 0], path_pts[:, 1]
                        
                        if len(np.unique(node_y)) > 2: 
                            poly_coeffs = np.polyfit(node_y, node_x, 2)
                        else:
                            poly_coeffs = np.polyfit(node_y, node_x, 1)
                            poly_coeffs = np.insert(poly_coeffs, 0, 0) 
                            
                        dense_y = np.linspace(node_y[0], node_y[-1], num=40)
                        dense_x = np.polyval(poly_coeffs, dense_y)
                        
                        car_x = w // 2
                        car_y = max_y
                        sum_slope = 0
                        valid_points = 0
                        for i in range(len(dense_x)):
                            dx = dense_x[i] - car_x
                            dy = car_y - dense_y[i]
                            if dy > 10: 
                                slope = dx / dy
                                sum_slope += slope
                                valid_points += 1
                                
                        avg_slope = sum_slope / valid_points if valid_points > 0 else 0
                        err_x = avg_slope * 100 

                        lookahead_idx = min(15, len(dense_y) - 1)
                        dy = dense_y[0] - dense_y[lookahead_idx]
                        dx = dense_x[0] - dense_x[lookahead_idx]
                        l_k = dx / dy if dy != 0 else 0
                        
                        pts_final_bird = np.vstack((dense_x, dense_y)).astype(np.int32).T.reshape((-1, 1, 2))
                        pts_final_orig = cv2.perspectiveTransform(np.float32(pts_final_bird), M_inv).astype(np.int32)
                
            # --- 状态与可视化 ---
            with data_lock:
                global_control_data["error_x"] = err_x
                global_control_data["line_k"] = l_k
                fps_stats["seg_frames"] += 1
                now = time.time()
                if now - fps_start_time >= 1.0:
                    fps_stats["seg_fps"] = fps_stats["seg_frames"] / (now - fps_start_time)
                    fps_stats["seg_frames"], fps_start_time = 0, now

            # 渲染基础画面
            colored_roi = np.zeros_like(vis_img)
            colored_roi[mask_full == 1] = [0, 0, 255]
            vis_img = cv2.addWeighted(vis_img, 1 - Config.MASK_ALPHA, colored_roi, Config.MASK_ALPHA, 0)
            
            pip_img = cv2.cvtColor(bird_eye_mask * 255, cv2.COLOR_GRAY2BGR)

            for bird_pts, orig_pts in debug_alternative_paths:
                cv2.polylines(vis_img, [orig_pts], False, (255, 255, 0), 1)
                cv2.polylines(pip_img, [bird_pts], False, (255, 255, 0), 1)
            
            if pts_final_orig is not None:
                cv2.polylines(vis_img, [pts_final_orig], False, (255, 0, 255), 5)
            if pts_final_bird is not None: 
                cv2.polylines(pip_img, [pts_final_bird], False, (0, 255, 0), 4)

            pip_h, pip_w = h // 3, w // 3
            vis_img[0:pip_h, w-pip_w:w] = cv2.resize(pip_img, (pip_w, pip_h))
            cv2.rectangle(vis_img, (w-pip_w, 0), (w, pip_h), (255, 255, 255), 2)

            # 🛡️ 渲染UI计算
            try:
                if np.isnan(err_x) or np.isinf(err_x): err_x = 0
                if np.isnan(l_k) or np.isinf(l_k): l_k = 0

                motor_speed = Config.MOTOR_MAX_SPEED - int(abs(l_k) * 15)
                motor_speed = max(Config.MOTOR_MIN_SPEED, min(Config.MOTOR_MAX_SPEED, motor_speed))
                
                base_p = err_x * Config.KP 
                extra_p = 0
                if abs(err_x) > 80: extra_p = (abs(err_x) - 80) * 0.4 * np.sign(err_x) 
                d_term = l_k * Config.KD
                
                raw_pwm = Config.SERVO_CENTER - base_p - extra_p + d_term
                servo_pwm = int(max(Config.SERVO_MIN, min(Config.SERVO_MAX, raw_pwm)))
            except Exception:
                motor_speed = Config.MOTOR_MIN_SPEED
                servo_pwm = Config.SERVO_CENTER

            overlay = vis_img.copy()
            cv2.rectangle(overlay, (10, 10), (450, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, vis_img, 0.4, 0, vis_img)
            
            info_text_1 = f"FPS: {fps_stats['seg_fps']:.1f}   MODE: {Config.TARGET_BRANCH}"
            info_text_2 = f"SPD: {motor_speed}   PWM: {servo_pwm}"
            
            cv2.putText(vis_img, info_text_1, (20, 45), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(vis_img, info_text_2, (20, 85), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)

            with frame_lock: global_preview_frame = vis_img
        except Exception as e: 
            print(f"Frame loop error: {e}", flush=True)

# ==============================================================================
# 4. 串口与 Web 逻辑
# ==============================================================================
def serial_control_thread():
    try: ser = serial.Serial(Config.SERIAL_PORT, Config.BAUD_RATE, timeout=0.1)
    except: ser = None
    while True:
        with data_lock:
            err_x, l_k = global_control_data["error_x"], global_control_data["line_k"]
            
        try:
            if np.isnan(err_x) or np.isinf(err_x): err_x = 0
            if np.isnan(l_k) or np.isinf(l_k): l_k = 0
            
            motor_pwm = Config.MOTOR_MAX_SPEED - int(abs(l_k) * 15)
            motor_pwm = max(Config.MOTOR_MIN_SPEED, min(Config.MOTOR_MAX_SPEED, motor_pwm))
            
            base_p = err_x * Config.KP 
            extra_p = 0
            if abs(err_x) > 80: extra_p = (abs(err_x) - 80) * 0.4 * np.sign(err_x) 
            d_term = l_k * Config.KD
            
            raw_pwm = Config.SERVO_CENTER - base_p - extra_p + d_term
            servo_pwm = int(max(Config.SERVO_MIN, min(Config.SERVO_MAX, raw_pwm)))
            
        except:
            motor_pwm = Config.MOTOR_MIN_SPEED
            servo_pwm = Config.SERVO_CENTER
            
        if ser: ser.write(struct.pack('<BBhhBB', 0xAA, 0x55, motor_pwm, servo_pwm, 0x0D, 0x0A))
        time.sleep(0.01) 

def ai_producer_thread():
    while True:
        try:
            shm = shared_memory.SharedMemory(name=Config.SHM_NAME)
            remove_shm_from_resource_tracker()
            last_fid = 0
            while True:
                fid, w, h = struct.unpack('QII', bytes(shm.buf[:Config.SHM_HEADER_SIZE]))
                if fid == last_fid: time.sleep(0.002); continue
                last_fid = fid
                frame = np.ndarray((h, w, 3), dtype=np.uint8, buffer=shm.buf[Config.SHM_HEADER_SIZE : Config.SHM_HEADER_SIZE+w*h*3]).copy()
                vis_img = cv2.resize(cv2.cvtColor(cv2.flip(frame, 0), cv2.COLOR_RGB2BGR), Config.TARGET_RES, interpolation=cv2.INTER_NEAREST)
                if seg_queue.full(): seg_queue.get_nowait()
                seg_queue.put(vis_img)
        except: time.sleep(1.0)

@app.route('/')
def index(): return render_template_string('<html><body style="background:#000;text-align:center;margin:0;"><img src="/video_feed" style="max-width:100%;"></body></html>')
@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            with frame_lock: 
                if global_preview_frame is None: time.sleep(0.01); continue
                _, buffer = cv2.imencode('.jpg', global_preview_frame, [int(cv2.IMWRITE_JPEG_QUALITY), Config.JPEG_QUALITY])
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.02)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    threading.Thread(target=ai_producer_thread, daemon=True).start()
    threading.Thread(target=serial_control_thread, daemon=True).start()
    time.sleep(0.1)
    threading.Thread(target=seg_worker, args=(RKNNLite.NPU_CORE_0,), daemon=True).start()
    threading.Thread(target=seg_worker, args=(RKNNLite.NPU_CORE_1,), daemon=True).start()
    app.run(host='0.0.0.0', port=Config.STREAM_PORT, threaded=True)
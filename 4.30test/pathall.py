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
    
    # 路径配置
    SEG_MODEL = "models/ppliteseg_576_final_int8.rknn"
    YOLO_MODEL = "models/yolov8_1.rknn" 

    TARGET_RES = (960, 720) 
    YOLO_SIZE = (640, 640)  
    SEG_SIZE = (576, 416)   
    
    ROI_TOP_CUT_RATIO = 0.3 
    MASK_ALPHA = 0.4
    
    # 分叉判定双重阈值
    FORK_WIDTH_RATIO = 0.35 # 1. 赛道总跨度阈值 (画面宽度的35%)
    FORK_GAP_RATIO = 0.15   # 2. 赛道中间断层(空隙)阈值，防止单条宽路被误判 (画面宽度的15%)
    
    SERIAL_PORT = '/dev/ttyS2'
    BAUD_RATE = 115200
    SERVO_CENTER = 750
    SERVO_MIN, SERVO_MAX = 590, 910
    MOTOR_STOP = 2000
    MOTOR_MAX_SPEED = 2350 
    
    GAUSSIAN_SIGMA = 35.0   
    SAFETY_MARGIN = 25      
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
yolo_queue = Queue(maxsize=1)

global_yolo_boxes = []      
# turn_intent: -1(左边), 1(右边), 0(不干预)
global_control_data = {"error_x": 0, "line_k": 0, "turn_intent": -1} 
data_lock = threading.Lock()

fps_stats = {"seg_frames": 0, "yolo_frames": 0, "seg_fps": 0.0, "yolo_fps": 0.0}
fps_start_time = time.time()

def remove_shm_from_resource_tracker():
    try: resource_tracker.unregister('/' + Config.SHM_NAME, 'shared_memory')
    except: pass

# ==============================================================================
# 3. YOLO 工作线程
# ==============================================================================
def process_yolo(output, orig_shape):
    try:
        preds = output[0][0].transpose(1, 0)
        boxes, scores = preds[:, :4], preds[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        max_scores = scores[np.arange(len(scores)), class_ids]
        
        mask = max_scores > 0.1
        if not np.any(mask): return []
        
        boxes, class_ids, max_scores = boxes[mask], class_ids[mask], max_scores[mask]
        x, y = boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2
        cv_boxes = np.stack((x, y, boxes[:, 2], boxes[:, 3]), axis=-1).tolist()
        indices = cv2.dnn.NMSBoxes(cv_boxes, max_scores.tolist(), 0.3, 0.45)
        
        scale_x, scale_y = orig_shape[0] / Config.YOLO_SIZE[0], orig_shape[1] / Config.YOLO_SIZE[1]
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                bx, by, bw, bh = cv_boxes[i]
                results.append({
                    'rect': [int(bx*scale_x), int(by*scale_y), int(bw*scale_x), int(bh*scale_y)], 
                    'class_id': int(class_ids[i])
                })
        return results
    except Exception as e:
        return []

def yolo_worker():
    try:
        rknn = RKNNLite()
        with init_lock: 
            if rknn.load_rknn(Config.YOLO_MODEL) != 0 or rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_2) != 0:
                return
        print("✅ YOLO Core 2 已就绪", flush=True)
    except: return

    while True:
        try:
            frame_data = yolo_queue.get()
            if frame_data is None: break
            
            blob = cv2.resize(frame_data, Config.YOLO_SIZE, interpolation=cv2.INTER_NEAREST)
            blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
            out = rknn.inference(inputs=[np.expand_dims(blob, axis=0)])
            res = process_yolo(out, Config.TARGET_RES)
            
            with data_lock:
                global global_yolo_boxes
                global_yolo_boxes = res
                fps_stats["yolo_frames"] += 1
        except: pass

# ==============================================================================
# 4. PPLiteSeg 路径规划工作线程 
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
            
            # --- 核心路径规划与融合 ---
            pred_mask_roi = mask[roi_start_y_seg:, :]
            white_pts = np.column_stack(np.where(pred_mask_roi == 1))
            
            err_x, l_k = 0, 0
            pts_final = None
            alt_pts_final = None  
            
            with data_lock:
                current_yolo_boxes = global_yolo_boxes.copy() 
                turn_intent = global_control_data.get("turn_intent", -1) 

            if len(white_pts) > 50:
                white_pts = white_pts[::8] 
                
                scale_h = Config.TARGET_RES[1] / Config.SEG_SIZE[1]
                scale_w = Config.TARGET_RES[0] / Config.SEG_SIZE[0]
                
                ys = (white_pts[:, 0] + roi_start_y_seg) * scale_h
                xs = white_pts[:, 1] * scale_w
                
                alt_ys, alt_xs = np.array([]), np.array([])
                
                # -------------------------------------------------------------
                # 分叉路口断层检测逻辑
                # -------------------------------------------------------------
                if len(ys) > 0:
                    y_min_val, y_max_val = np.min(ys), np.max(ys)
                    y_range_val = y_max_val - y_min_val
                    
                    fork_th = Config.TARGET_RES[0] * Config.FORK_WIDTH_RATIO 
                    gap_th = Config.TARGET_RES[0] * Config.FORK_GAP_RATIO
                    
                    NUM_BINS = 10  
                    valid_indices = []
                    alt_indices = []
                    
                    for i in range(NUM_BINS):
                        b_max = y_max_val - i * (y_range_val / NUM_BINS)
                        b_min = y_max_val - (i + 1) * (y_range_val / NUM_BINS)
                        
                        in_band = (ys <= b_max) & (ys > b_min)
                        idx = np.where(in_band)[0]
                        
                        if len(idx) == 0: continue
                        
                        band_xs = xs[idx]
                        b_x_min, b_x_max = np.min(band_xs), np.max(band_xs)
                        
                        sorted_xs = np.sort(band_xs)
                        max_gap = np.max(np.diff(sorted_xs)) if len(sorted_xs) > 1 else 0
                        
                        if (b_x_max - b_x_min) > fork_th and max_gap > gap_th:
                            gap_idx = np.argmax(np.diff(sorted_xs))
                            x_mid = (sorted_xs[gap_idx] + sorted_xs[gap_idx+1]) / 2.0
                            
                            if turn_intent == -1:    
                                keep_idx = idx[xs[idx] < x_mid]
                                drop_idx = idx[xs[idx] >= x_mid]
                            elif turn_intent == 1:   
                                keep_idx = idx[xs[idx] >= x_mid]
                                drop_idx = idx[xs[idx] < x_mid]
                            else:                    
                                keep_idx = idx
                                drop_idx = []
                                
                            valid_indices.extend(keep_idx)
                            alt_indices.extend(drop_idx)
                        else:
                            valid_indices.extend(idx)
                            
                    alt_ys = np.array(ys[alt_indices])
                    alt_xs = np.array(xs[alt_indices])
                    
                    ys = np.array(ys[valid_indices])
                    xs = np.array(xs[valid_indices])
                # -------------------------------------------------------------
                
                # ====== 渲染计算：备选路径 ======
                if len(np.unique(alt_ys)) > 5:
                    a_max_y, a_min_y = np.max(alt_ys), np.min(alt_ys)
                    a_y_range = a_max_y - a_min_y
                    
                    a_line_bound = int(a_max_y - a_y_range * 0.7)
                    a_line_mask = alt_ys >= a_line_bound
                    
                    if np.sum(a_line_mask) > 5:
                        a_l_k, a_l_b = np.polyfit(alt_ys[a_line_mask], alt_xs[a_line_mask], 1)
                    else:
                        a_l_k, a_l_b = 0, Config.TARGET_RES[0] // 2
                    
                    a_poly_coeffs = np.polyfit(alt_ys, alt_xs, 2)
                    a_plot_y = np.linspace(a_max_y, a_min_y, num=40)
                    a_plot_x_line = a_l_k * a_plot_y + a_l_b
                    a_plot_x_curve = np.polyval(a_poly_coeffs, a_plot_y)
                    
                    a_t_arr = (a_max_y - a_plot_y) / (a_y_range + 0.1)
                    a_alpha = a_t_arr ** 2
                    a_plot_x_final = (1 - a_alpha) * a_plot_x_line + a_alpha * a_plot_x_curve
                    
                    a_padded_x = np.pad(a_plot_x_final, (Config.SMOOTH_WINDOW//2, Config.SMOOTH_WINDOW//2), mode='edge')
                    a_plot_x_final = np.convolve(a_padded_x, np.ones(Config.SMOOTH_WINDOW)/Config.SMOOTH_WINDOW, mode='valid')
                    alt_pts_final = np.vstack((a_plot_x_final, a_plot_y)).astype(np.int32).T.reshape((-1, 1, 2))


                # ====== 核心控制计算：主路径 ======
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

                    t_arr = (max_y - plot_y) / (y_range + 0.1)
                    alpha = t_arr ** 2
                    plot_x_final = (1 - alpha) * plot_x_line + alpha * plot_x_curve

                    # 2. YOLO 避障与引导融合
                    for obj in current_yolo_boxes:
                        ox, oy, ow, oh = obj['rect']
                        cx, cy = ox + ow/2.0, oy + oh/2.0
                        
                        # ==========================================================
                        # 🌟 新增：过滤赛道掩码外的金币
                        # ==========================================================
                        # 映射回 576x416 的 Mask 坐标系，并用 np.clip 防止越界报错
                        mask_x = int(np.clip(cx * Config.SEG_SIZE[0] / Config.TARGET_RES[0], 0, Config.SEG_SIZE[0] - 1))
                        mask_y = int(np.clip(cy * Config.SEG_SIZE[1] / Config.TARGET_RES[1], 0, Config.SEG_SIZE[1] - 1))
                        
                        # 如果是金币(class_id==1)，并且它的中心点在掩码上是0（黑色/非赛道），则忽略它！
                        if obj['class_id'] == 1 and mask[mask_y, mask_x] == 0:
                            continue
                        # ==========================================================
                        
                        if not (min_y <= cy <= max_y): continue
                        idx = np.argmin(np.abs(plot_y - cy))
                        
                        if obj['class_id'] == 1: 
                            plot_x_final += (cx - plot_x_final[idx]) * np.exp(-((plot_y - cy)**2) / (2 * Config.GAUSSIAN_SIGMA**2))
                        elif obj['class_id'] == 0: 
                            for i in range(len(plot_y)):
                                if oy <= plot_y[i] <= oy + oh:
                                    if ox - Config.SAFETY_MARGIN < plot_x_final[i] < ox + ow + Config.SAFETY_MARGIN:
                                        plot_x_final[i] = (ox - Config.SAFETY_MARGIN) if plot_x_final[idx] < cx else (ox + ow + Config.SAFETY_MARGIN)

                    # 3. 平滑
                    padded_x = np.pad(plot_x_final, (Config.SMOOTH_WINDOW//2, Config.SMOOTH_WINDOW//2), mode='edge')
                    plot_x_final = np.convolve(padded_x, np.ones(Config.SMOOTH_WINDOW)/Config.SMOOTH_WINDOW, mode='valid')

                    # 4. 误差计算
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
                    fps_stats["yolo_fps"] = fps_stats["yolo_frames"] / (now - fps_start_time)
                    fps_stats["seg_frames"] = fps_stats["yolo_frames"] = 0
                    fps_start_time = now

            # --- 渲染可视化 ---
            color_mask = cv2.resize(mask, Config.TARGET_RES, interpolation=cv2.INTER_NEAREST)
            colored_roi = np.zeros_like(vis_img)
            colored_roi[color_mask == 1] = [0, 0, 255]
            vis_img = cv2.addWeighted(vis_img, 1 - Config.MASK_ALPHA, colored_roi, Config.MASK_ALPHA, 0)
            
            if alt_pts_final is not None:
                cv2.polylines(vis_img, [alt_pts_final], False, (0, 255, 255), 2)
                
            if pts_final is not None:
                cv2.polylines(vis_img, [pts_final], False, (255, 0, 255), 4)
            
            for box in current_yolo_boxes:
                bx, by, bw, bh = box['rect']
                color = (0, 255, 0) if box['class_id'] == 1 else (0, 0, 255)
                # 画框不受影响，仍会画出所有识别到的金币（方便排错），但赛道外的金币已经不再参与上述路径计算
                cv2.rectangle(vis_img, (bx, by), (bx+bw, by+bh), color, 2)

            servo_pwm = int(Config.SERVO_CENTER + (err_x * Config.KP) - (l_k * Config.KD))
            cv2.putText(vis_img, f"FPS:{fps_stats['seg_fps']:.1f} PWM:{servo_pwm}", (20, 30), 1, 1.5, (0, 255, 0), 2)
            cv2.putText(vis_img, f"YOLO FPS:{fps_stats['yolo_fps']:.1f} Intent:{turn_intent}", (20, 70), 1, 1.5, (0, 255, 255), 2)

            with frame_lock:
                global_preview_frame = vis_img
                
        except Exception as e:
            pass

# ==============================================================================
# 5. 串口控制线程 
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
        motor_pwm = Config.MOTOR_MAX_SPEED - int(abs(l_k) * 120)
        
        servo_pwm = max(Config.SERVO_MIN, min(Config.SERVO_MAX, servo_pwm))
        motor_pwm = max(Config.MOTOR_STOP, min(Config.MOTOR_MAX_SPEED, motor_pwm))
        
        if ser:
            packet = struct.pack('<BBhhBB', 0xAA, 0x55, motor_pwm, servo_pwm, 0x0D, 0x0A)
            ser.write(packet)
            
        time.sleep(0.01) 

# ==============================================================================
# 6. 共享内存拉取
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
                
                if yolo_queue.full():
                    try: yolo_queue.get_nowait()
                    except: pass
                yolo_queue.put(vis_img.copy())

        except Exception as e:
            time.sleep(1.0)
        finally:
            if shm: 
                try: shm.close()
                except: pass

# ==============================================================================
# 7. Web 推流服务
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
    print(" 🚀 Aero-Twin [双模型+路径规划融合 - 赛道外金币过滤版] 启动 ", flush=True)
    print("======================================================", flush=True)
    
    threading.Thread(target=ai_producer_thread, daemon=True).start()
    threading.Thread(target=serial_control_thread, daemon=True).start()
    
    time.sleep(0.1)
    threading.Thread(target=seg_worker, args=(RKNNLite.NPU_CORE_0,), daemon=True).start()
    time.sleep(0.1)
    threading.Thread(target=seg_worker, args=(RKNNLite.NPU_CORE_1,), daemon=True).start()
    time.sleep(0.1)
    threading.Thread(target=yolo_worker, daemon=True).start()
    
    app.run(host='0.0.0.0', port=Config.STREAM_PORT, threaded=True)
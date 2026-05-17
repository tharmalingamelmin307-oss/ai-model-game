# 文件路径: main.py
import time
import struct
import numpy as np
import cv2
import threading
import serial
from queue import Queue
from multiprocessing import shared_memory, resource_tracker
from flask import Flask, Response, render_template_string

import config
from modules.segmentor import RoadSegmentor
from modules.detector import YOLODetector
from modules.ocr_system import OCRRecognizer

app = Flask(__name__)

# ==============================================================================
# 全局状态与资源锁
# ==============================================================================
global_preview_frame = None
global_control_data = {"error_x": 0, "line_k": 0, "turn_intent": -1} 
fps_stats = {"seg_frames": 0, "yolo_frames": 0, "seg_fps": 0.0, "yolo_fps": 0.0}

frame_lock = threading.Lock()
data_lock = threading.Lock()

seg_queue = Queue(maxsize=1) 
yolo_queue = Queue(maxsize=1) 
global_yolo_boxes = []        

def remove_shm_from_resource_tracker():
    try: resource_tracker.unregister('/' + config.SHM_NAME, 'shared_memory')
    except: pass

# ==============================================================================
# 🎯 核心线程 1：YOLO 与 OCR 联合线程 (驻扎在 Core 2)
# ==============================================================================
def yolo_and_ocr_worker():
    try:
        det = YOLODetector(core_id=config.YOLO_CORE)
        ocr = OCRRecognizer(core_id=config.REC_CORE) 
        print(f"✅ YOLO + OCR (Core 2) 已就绪", flush=True)
    except Exception as e: 
        print(f"❌ YOLO/OCR 启动失败: {e}", flush=True)
        return

    while True:
        try:
            frame_data = yolo_queue.get()
            if frame_data is None: break
            
            # 1. 目标检测
            objs = det.run(frame_data)
            
            # 2. 仅检测到路牌时，裁图跑 OCR
            for obj in objs:
                if obj.get('class_id') == getattr(config, 'SIGN_CLASS_ID', 2):
                    bx, by, bw, bh = obj['rect']
                    x1, y1 = max(0, bx), max(0, by)
                    x2, y2 = min(frame_data.shape[1], bx+bw), min(frame_data.shape[0], by+bh)
                    roi = frame_data[y1:y2, x1:x2]
                    
                    if roi.size > 0:
                        text, score = ocr.run_single_crop(roi) 
                        obj['text'] = text
                        
                        # 意图联动逻辑
                        if text == 'LEFT':
                            with data_lock: global_control_data['turn_intent'] = -1
                        elif text == 'RIGHT':
                            with data_lock: global_control_data['turn_intent'] = 1

            # 更新全局框，供 Seg 线程渲染
            with data_lock:
                global global_yolo_boxes
                global_yolo_boxes = objs
                fps_stats["yolo_frames"] += 1
        except Exception as e:
            pass

# ==============================================================================
# 🎯 核心线程 2：分割与路径规划线程 (支持多核并发)
# ==============================================================================
def seg_worker(core_id):
    global global_preview_frame
    fps_start_time = time.time()
    
    try:
        seg = RoadSegmentor(core_id=core_id)
        print(f"✅ Seg(Core {core_id}) 已就绪", flush=True)
    except Exception as e:
        print(f"Seg 启动失败: {e}")
        return

    while True:
        # 这里拿到的已经是 320x320 的 RGB 小图了！
        blob_rgb_320 = seg_queue.get()
        if blob_rgb_320 is None: break

        with data_lock:
            current_yolo_boxes = global_yolo_boxes.copy() 
            turn_intent = global_control_data.get("turn_intent", -1) 

        err_x, l_k, rendered_img = seg.run(blob_rgb_320, current_yolo_boxes, turn_intent, fps_stats)

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

        with frame_lock:
            # 存入的 global_preview_frame 是 320x320 的，极大地减轻了 Flask 的编码负担
            global_preview_frame = rendered_img

# ==============================================================================
# 🎯 基础支撑线程：下位机通信与内存拉流
# ==============================================================================
def serial_control_thread():
    try: ser = serial.Serial(config.SERIAL_PORT, config.BAUD_RATE, timeout=0.1)
    except: ser = None
    while True:
        with data_lock:
            err_x = global_control_data.get("error_x", 0)
            l_k = global_control_data.get("line_k", 0)
            
        try:
            if np.isnan(err_x) or np.isinf(err_x): err_x = 0
            if np.isnan(l_k) or np.isinf(l_k): l_k = 0
            motor_pwm = getattr(config, 'MOTOR_MAX_SPEED', 2350) - int(abs(l_k) * 15)
            motor_pwm = max(getattr(config, 'MOTOR_MIN_SPEED', 2000), min(getattr(config, 'MOTOR_MAX_SPEED', 2350), motor_pwm))
            
            base_p = err_x * getattr(config, 'KP', 0.16)
            extra_p = 0
            if abs(err_x) > 80: extra_p = (abs(err_x) - 80) * 0.4 * np.sign(err_x) 
            d_term = l_k * getattr(config, 'KD', 160.0)
            
            raw_pwm = getattr(config, 'SERVO_CENTER', 750) - base_p - extra_p + d_term
            servo_pwm = int(max(getattr(config, 'SERVO_MIN', 590), min(getattr(config, 'SERVO_MAX', 910), raw_pwm)))
        except:
            motor_pwm = getattr(config, 'MOTOR_MIN_SPEED', 2000)
            servo_pwm = getattr(config, 'SERVO_CENTER', 750)
            
        if ser: ser.write(struct.pack('<BBhhBB', 0xAA, 0x55, motor_pwm, servo_pwm, 0x0D, 0x0A))
        time.sleep(0.01) 

def ai_producer_thread():
    print("--> 📡 启动拉流...", flush=True)
    while True:
        try:
            shm = shared_memory.SharedMemory(name=config.SHM_NAME)
            remove_shm_from_resource_tracker()
            last_fid = 0
            while True:
                header = bytes(shm.buf[:config.SHM_HEADER_SIZE])
                fid, w, h = struct.unpack('QII', header)
                if fid == last_fid: time.sleep(0.002); continue
                last_fid = fid
                
                # 获取原生图像
                img_view = np.ndarray((h, w, 3), dtype=np.uint8, buffer=shm.buf[config.SHM_HEADER_SIZE : config.SHM_HEADER_SIZE+w*h*3])
                
                # 🚀 神级优化：翻转后直接保持 RGB，并缩放到 320x320 喂给 Seg 队列
                frame_rgb = cv2.flip(img_view.copy(), 0)
                seg_blob = cv2.resize(frame_rgb, config.SEG_SIZE, interpolation=cv2.INTER_NEAREST)
                
                if seg_queue.full():
                    try: seg_queue.get_nowait()
                    except: pass
                seg_queue.put(seg_blob) # Seg 得到最纯净的 320x320 RGB 图像
                
                # 喂给 YOLO 的保持 960x720 BGR 高清大图 (因为需要看清路牌上的字)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                vis_img_large = cv2.resize(frame_bgr, config.TARGET_RES, interpolation=cv2.INTER_NEAREST)
                
                if yolo_queue.full():
                    try: yolo_queue.get_nowait()
                    except: pass
                yolo_queue.put(vis_img_large)

        except:
            time.sleep(1.0)

# ==============================================================================
# Flask 推流：利用前端 HTML 进行无损拉伸，解放后端算力！
# ==============================================================================
@app.route('/')
def index(): 
    # CSS image-rendering: pixelated 可以让 320x320 拉伸后不模糊，保持赛博硬核风
    return render_template_string('<html><body style="background:#000;text-align:center;margin:0;"><img src="/video_feed" style="max-width:100%; height:100vh; image-rendering: pixelated;"></body></html>')

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            with frame_lock:
                current_frame = None if global_preview_frame is None else global_preview_frame.copy()
            if current_frame is None:
                time.sleep(0.01); continue
            # 此时的编码对象只有 320x320 大小，CPU 耗时几乎为 0
            ret, buffer = cv2.imencode('.jpg', current_frame, [int(cv2.IMWRITE_JPEG_QUALITY), getattr(config, 'JPEG_QUALITY', 75)])
            if not ret: continue
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.02) 
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print(" 🚀 Aero-Twin [极限性能版] 启动 ", flush=True)
    threading.Thread(target=ai_producer_thread, daemon=True).start()
    threading.Thread(target=serial_control_thread, daemon=True).start()
    time.sleep(0.1)
    
    # 根据你的 config.py 里的 SEG_CORES 动态拉起 NPU
    # 如果你想跑满，在 config 里填入 [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1]
    for core_id in config.SEG_CORES:
        threading.Thread(target=seg_worker, args=(core_id,), daemon=True).start()
        time.sleep(0.2)
    
    threading.Thread(target=yolo_and_ocr_worker, daemon=True).start()
    
    app.run(host='0.0.0.0', port=config.STREAM_PORT, threaded=True)
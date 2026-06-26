# main.py
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
global_control_data = {
    "error_x": 0,
    "line_k": 0,
    "turn_intent": -1,
    "actual_servo_pwm": 750,
    "target_speed": 10
}
fps_stats = {"seg_frames": 0, "yolo_frames": 0, "seg_fps": 0.0, "yolo_fps": 0.0}

frame_lock = threading.Lock()
data_lock = threading.Lock()

seg_queue = Queue(maxsize=1)
yolo_queue = Queue(maxsize=1)
global_yolo_boxes = []


def remove_shm_from_resource_tracker():
    try:
        resource_tracker.unregister('/' + config.SHM_NAME, 'shared_memory')
    except:
        pass


# ==============================================================================
# YOLO框绘制
# ==============================================================================
def draw_yolo_boxes(image, boxes):
    if image is None or len(boxes) == 0:
        return image

    img_h, img_w = image.shape[:2]
    src_w, src_h = config.TARGET_RES  # (960, 720)

    scale_x = img_w / float(src_w)
    scale_y = img_h / float(src_h)

    for obj in boxes:
        rect = obj.get("rect", [0, 0, 0, 0])
        if len(rect) != 4:
            continue

        x, y, w, h = rect
        cls_id = obj.get("class_id", -1)
        cls_name = obj.get("class_name", str(cls_id))
        score = obj.get("score", 0.0)
        text = obj.get("text", "")

        x1 = int(np.clip(round(x * scale_x), 0, img_w - 1))
        y1 = int(np.clip(round(y * scale_y), 0, img_h - 1))
        x2 = int(np.clip(round((x + w) * scale_x), 0, img_w - 1))
        y2 = int(np.clip(round((y + h) * scale_y), 0, img_h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        color = (0, 0, 255)
        if cls_id == getattr(config, "SIGN_CLASS_ID", 9):
            color = (0, 255, 255)
        elif cls_id == getattr(config, "LIMIT_SIGN_CLASS_ID", 10):
            color = (255, 0, 0)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = f"{cls_name}:{score:.2f}"
        if text:
            label += f" [{text}]"

        text_y = y1 - 8 if y1 > 20 else y1 + 18
        cv2.putText(
            image,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA
        )

    return image


# ==============================================================================
# 核心线程 1：YOLO 与 OCR 联合线程
# ==============================================================================
def yolo_and_ocr_worker():
    try:
        det = YOLODetector(core_id=config.YOLO_CORE)
        ocr = OCRRecognizer(core_id=config.REC_CORE)
        print("✅ YOLO + OCR (Core 2) 已就绪", flush=True)
    except Exception as e:
        print(f"❌ YOLO/OCR 启动失败: {e}", flush=True)
        return

    while True:
        try:
            frame_data = yolo_queue.get()
            if frame_data is None:
                break

            objs = det.run(frame_data)

            for obj in objs:
                if obj.get("class_id") != getattr(config, "SIGN_CLASS_ID", 9):
                    continue

                try:
                    bx, by, bw, bh = obj["rect"]
                    x1 = max(0, int(bx))
                    y1 = max(0, int(by))
                    x2 = min(frame_data.shape[1], int(bx + bw))
                    y2 = min(frame_data.shape[0], int(by + bh))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    roi = frame_data[y1:y2, x1:x2]
                    if roi is None or roi.size == 0:
                        continue

                    text, score = ocr.run_single_crop(roi)
                    text = text.strip().upper()
                    obj["text"] = text
                    obj["ocr_score"] = float(score)

                    if text == "LEFT":
                        with data_lock:
                            global_control_data["turn_intent"] = -1
                    elif text == "RIGHT":
                        with data_lock:
                            global_control_data["turn_intent"] = 1
                except Exception as e:
                    print(f"OCR单框异常: {e}", flush=True)

            with data_lock:
                global global_yolo_boxes
                global_yolo_boxes = objs
                fps_stats["yolo_frames"] += 1

        except Exception as e:
            print(f"YOLO/OCR线程异常: {e}", flush=True)

# ==============================================================================
# 核心线程 2：分割与路径规划线程
# ==============================================================================
def seg_worker(core_id):
    global global_preview_frame
    fps_start_time = time.time()

    try:
        seg = RoadSegmentor(core_id=core_id)
        print(f"✅ Seg(Core {core_id}) 已就绪", flush=True)
    except Exception as e:
        print(f"Seg 启动失败: {e}", flush=True)
        return

    while True:
        blob_rgb_320 = seg_queue.get()
        if blob_rgb_320 is None:
            break

        with data_lock:
            current_yolo_boxes = [obj.copy() for obj in global_yolo_boxes]
            turn_intent = global_control_data.get("turn_intent", -1)

        err_x, l_k, rendered_img = seg.run(
            blob_rgb_320,
            current_yolo_boxes,
            turn_intent,
            fps_stats
        )

        # 关键修正：先放大回 960x720，再叠加 YOLO 框
        if rendered_img is not None:
            if rendered_img.shape[1] != config.TARGET_RES[0] or rendered_img.shape[0] != config.TARGET_RES[1]:
                rendered_img = cv2.resize(
                    rendered_img,
                    config.TARGET_RES,
                    interpolation=cv2.INTER_NEAREST
                )
            rendered_img = draw_yolo_boxes(rendered_img, current_yolo_boxes)

        with data_lock:
            global_control_data["error_x"] = err_x
            global_control_data["line_k"] = l_k

            actual_servo = global_control_data.get("actual_servo_pwm", 750)
            actual_speed = global_control_data.get("target_speed", 10)

            fps_stats["seg_frames"] += 1
            now = time.time()
            if now - fps_start_time >= 1.0:
                fps_stats["seg_fps"] = fps_stats["seg_frames"] / (now - fps_start_time)
                fps_stats["yolo_fps"] = fps_stats["yolo_frames"] / (now - fps_start_time)
                fps_stats["seg_frames"] = 0
                fps_stats["yolo_frames"] = 0
                fps_start_time = now

            current_seg_fps = fps_stats["seg_fps"]
            current_yolo_fps = fps_stats["yolo_fps"]

        if rendered_img is not None:
            cv2.rectangle(rendered_img, (2, 102), (230, 154), (0, 0, 0), -1)
            cv2.rectangle(rendered_img, (2, 102), (230, 154), (0, 255, 255), 1)

            cv2.putText(
                rendered_img,
                f"Seg:{current_seg_fps:.1f} YOL:{current_yolo_fps:.1f}",
                (6, 116),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            cv2.putText(
                rendered_img,
                f"Err:{err_x:.1f}cm Srv:{actual_servo}",
                (6, 132),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            cv2.putText(
                rendered_img,
                f"Target Spd:{actual_speed}",
                (6, 148),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )

        with frame_lock:
            global_preview_frame = rendered_img

# ==============================================================================
# 基础支撑线程：串口控制
# ==============================================================================
def serial_control_thread():
    try:
        ser = serial.Serial(config.SERIAL_PORT, config.BAUD_RATE, timeout=0.1)
    except:
        ser = None

    while True:
        with data_lock:
            err_x = global_control_data.get("error_x", 0)
            l_k = global_control_data.get("line_k", 0)

        try:
            if np.isnan(err_x) or np.isinf(err_x):
                err_x = 0
            if np.isnan(l_k) or np.isinf(l_k):
                l_k = 0

            max_speed = getattr(config, 'MOTOR_MAX_SPEED', 30)
            min_speed = 10

            target_speed = 30 - int(abs(l_k) * 10)
            target_speed = int(max(min_speed, min(30, target_speed)))

            base_p = err_x * getattr(config, 'KP', 0.16)
            extra_p = 0
            if abs(err_x) > 80:
                extra_p = (abs(err_x) - 80) * 0.4 * np.sign(err_x)
            d_term = l_k * getattr(config, 'KD', 160.0)

            raw_pwm = getattr(config, 'SERVO_CENTER', 750) - base_p - extra_p + d_term
            servo_pwm = int(max(getattr(config, 'SERVO_MIN', 590), min(getattr(config, 'SERVO_MAX', 910), raw_pwm)))
        except:
            target_speed = 10
            servo_pwm = getattr(config, 'SERVO_CENTER', 750)

        with data_lock:
            global_control_data["actual_servo_pwm"] = servo_pwm
            global_control_data["target_speed"] = target_speed

        if ser:
            ser.write(struct.pack('<BBhhBB', 0xAA, 0x55, target_speed, servo_pwm, 0x0D, 0x0A))

        time.sleep(0.01)


# ==============================================================================
# 基础支撑线程：共享内存拉流
# ==============================================================================
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
                if fid == last_fid:
                    time.sleep(0.002)
                    continue
                last_fid = fid

                img_view = np.ndarray(
                    (h, w, 3),
                    dtype=np.uint8,
                    buffer=shm.buf[config.SHM_HEADER_SIZE: config.SHM_HEADER_SIZE + w * h * 3]
                )

                frame_rgb = cv2.flip(img_view.copy(), 0)

                # 分割输入
                seg_blob = cv2.resize(frame_rgb, config.SEG_SIZE, interpolation=cv2.INTER_NEAREST)
                if seg_queue.full():
                    try:
                        seg_queue.get_nowait()
                    except:
                        pass
                seg_queue.put(seg_blob)

                # YOLO 输入
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                vis_img_large = cv2.resize(frame_bgr, config.TARGET_RES, interpolation=cv2.INTER_NEAREST)

                if yolo_queue.full():
                    try:
                        yolo_queue.get_nowait()
                    except:
                        pass
                yolo_queue.put(vis_img_large)

        except Exception:
            time.sleep(1.0)


# ==============================================================================
# Flask 推流
# ==============================================================================
@app.route('/')
def index():
    return render_template_string('''
    <html>
    <body style="background:#000;text-align:center;margin:0;">
        <img src="/video_feed" style="max-width:100%; height:100vh; image-rendering: pixelated;">
    </body>
    </html>
    ''')


@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            with frame_lock:
                current_frame = None if global_preview_frame is None else global_preview_frame.copy()

            if current_frame is None:
                time.sleep(0.01)
                continue

            ret, buffer = cv2.imencode(
                '.jpg',
                current_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), getattr(config, 'JPEG_QUALITY', 75)]
            )
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.02)

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    print("🚀 Aero-Twin [极限性能版] 启动", flush=True)

    threading.Thread(target=ai_producer_thread, daemon=True).start()
    threading.Thread(target=serial_control_thread, daemon=True).start()
    time.sleep(0.1)

    for core_id in config.SEG_CORES:
        threading.Thread(target=seg_worker, args=(core_id,), daemon=True).start()
        time.sleep(0.2)

    threading.Thread(target=yolo_and_ocr_worker, daemon=True).start()

    app.run(host='0.0.0.0', port=config.STREAM_PORT, threaded=True)
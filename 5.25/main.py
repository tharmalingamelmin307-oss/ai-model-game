"""系统主入口.

整体流程:
1. ai_producer_thread 从共享内存读取最新图像。
2. 图像被拆成两条支路:
   - seg_queue: 送给分割 / 路径规划线程
   - yolo_queue: 送给目标检测线程
3. yolo_worker 只负责检测；如果检测到 sign，再把 OCR 任务异步送入 ocr_queue。
4. ocr_worker 单独占用一个 NPU 核，只负责识别 sign 里的文字。
5. seg_worker 读取当前最新的检测结果与 turn_intent，生成控制量和预览图。
6. serial_control_thread 将控制量打包后发给下位机。
7. Flask 将 global_preview_frame 编码成 MJPEG 提供网页预览。
"""

import time
import struct
import numpy as np
import cv2
import threading
import serial
from queue import Queue
from collections import deque
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
# 当前网页预览的最新画面，只由 seg_worker 写，由 Flask 推流读取。
global_preview_frame = None

# 供分割线程、串口线程、OCR 线程共享的控制状态。
global_control_data = {
    "error_x": 0,
    "line_k": 0,
    "turn_intent": -1,
    "actual_servo_pwm": 750,
    "target_speed": 10
}

# 用于在页面上显示 Seg / YOLO 处理频率。
fps_stats = {"seg_frames": 0, "yolo_frames": 0, "seg_fps": 0.0, "yolo_fps": 0.0}

frame_lock = threading.Lock()
data_lock = threading.Lock()

# 三条工作队列:
# - seg_queue: 最新一帧分割输入
# - yolo_queue: 最新一帧检测输入 (fid, det_img)
# - ocr_queue: 检测线程生成的 (fid, sign 原图 ROI 任务)
seg_queue = Queue(maxsize=1)
yolo_queue = Queue(maxsize=1)
ocr_queue = Queue(maxsize=2)

# 当前最新一帧的检测结果。
global_yolo_boxes = []
global_yolo_frame_id = -1

raw_frame_lock = threading.Lock()
raw_frame_cache = {}
raw_frame_order = deque()
RAW_FRAME_CACHE_SIZE = 4


def remove_shm_from_resource_tracker():
    """避免 Python 退出时错误回收外部创建的共享内存对象."""
    try:
        resource_tracker.unregister('/' + config.SHM_NAME, 'shared_memory')
    except:
        pass


# ==============================================================================
# YOLO框绘制
# ==============================================================================
def draw_yolo_boxes(image, boxes):
    """在最终显示图上叠加检测框和文字标签.

    说明:
    - 检测框统一按照 TARGET_RES 坐标系保存；
    - 如果当前显示图尺寸不是 TARGET_RES，这里会做一次比例映射。
    """
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

        # 在主预览图上把检测框画粗一点，方便快速确认检测是否生效。
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

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


def summarize_yolo_boxes(boxes):
    """生成一行简短检测摘要，便于在预览页直接确认 YOLO 是否有输出。"""
    if not boxes:
        return "YOLO:0"

    parts = [f"YOLO:{len(boxes)}"]
    for obj in boxes[:4]:
        cls_name = obj.get("class_name", "?")
        score = float(obj.get("score", 0.0))
        parts.append(f"{cls_name}:{score:.2f}")
    if len(boxes) > 4:
        parts.append("...")
    return " | ".join(parts)


def cache_raw_frame(fid, raw_frame):
    """缓存少量原始分辨率帧，供 sign 命中后按 fid 回取 ROI。"""
    if raw_frame is None:
        return

    with raw_frame_lock:
        if fid in raw_frame_cache:
            try:
                raw_frame_order.remove(fid)
            except ValueError:
                pass
        raw_frame_cache[fid] = raw_frame
        raw_frame_order.append(fid)

        while len(raw_frame_order) > RAW_FRAME_CACHE_SIZE:
            old_fid = raw_frame_order.popleft()
            raw_frame_cache.pop(old_fid, None)


def get_cached_raw_frame(fid):
    """按 fid 取回原始分辨率帧；返回的是同一块内存引用，不做整帧复制。"""
    with raw_frame_lock:
        return raw_frame_cache.get(fid)


def crop_target_rect_from_raw_frame(raw_frame, rect):
    """将 TARGET_RES 坐标系下的检测框映射回原图并裁出 ROI。"""
    if raw_frame is None or len(rect) != 4:
        return None

    raw_h, raw_w = raw_frame.shape[:2]
    target_w, target_h = config.TARGET_RES
    if raw_w <= 0 or raw_h <= 0 or target_w <= 0 or target_h <= 0:
        return None

    bx, by, bw, bh = rect
    scale_x = raw_w / float(target_w)
    scale_y = raw_h / float(target_h)

    x1 = max(0, int(np.floor(bx * scale_x)))
    y1 = max(0, int(np.floor(by * scale_y)))
    x2 = min(raw_w, int(np.ceil((bx + bw) * scale_x)))
    y2 = min(raw_h, int(np.ceil((by + bh) * scale_y)))

    if x2 <= x1 or y2 <= y1:
        return None

    roi = raw_frame[y1:y2, x1:x2]
    if roi is None or roi.size == 0:
        return None
    return roi.copy()


# ==============================================================================
# 核心线程 1：YOLO 检测线程
# ==============================================================================
def yolo_worker():
    """纯检测线程.

    这个线程只做两件事:
    1. 执行 YOLO 推理并更新 global_yolo_boxes
    2. 将 sign 类别的 ROI 任务投递给 OCR 线程

    这样 OCR 不会反过来阻塞 YOLO，能显著降低检测延迟。
    """
    try:
        det = YOLODetector(core_id=config.YOLO_CORE)
        print(f"✅ YOLO (Core {config.YOLO_CORE}) 已就绪", flush=True)
    except Exception as e:
        print(f"❌ YOLO 启动失败: {e}", flush=True)
        return

    while True:
        try:
            frame_data = yolo_queue.get()
            if frame_data is None:
                break

            frame_fid, det_frame = frame_data
            objs = det.run(det_frame, output_size=config.TARGET_RES)

            # 新一帧检测结果先清掉旧的 OCR 文本，避免沿用上一帧残留内容。
            for obj in objs:
                try:
                    obj.pop("text", None)
                    obj.pop("ocr_score", None)
                except Exception:
                    pass

            with data_lock:
                global global_yolo_boxes
                global global_yolo_frame_id
                global_yolo_boxes = objs
                global_yolo_frame_id = frame_fid
                fps_stats["yolo_frames"] += 1

            # 只把需要 OCR 的 sign 框送到 OCR 队列，减少额外开销。
            raw_frame = get_cached_raw_frame(frame_fid)
            sign_crops = []
            for idx, obj in enumerate(objs):
                if obj.get("class_id") != getattr(config, "SIGN_CLASS_ID", 9):
                    continue
                roi = crop_target_rect_from_raw_frame(raw_frame, obj.get("rect", [0, 0, 0, 0]))
                if roi is None:
                    continue
                sign_crops.append((idx, roi))

            if sign_crops:
                # OCR 只保留较新的任务，过旧的任务直接丢掉。
                if ocr_queue.full():
                    try:
                        ocr_queue.get_nowait()
                    except:
                        pass
                ocr_queue.put((frame_fid, sign_crops))

        except Exception as e:
            print(f"YOLO线程异常: {e}", flush=True)


# ==============================================================================
# 核心线程 1.5：OCR 识别线程
# ==============================================================================
def ocr_worker():
    """纯 OCR 线程.

    输入:
        yolo_worker 投递的 (fid, sign 原图 ROI 列表)

    输出:
        将识别出的 text / ocr_score 回写到 global_yolo_boxes，
        同时根据 LEFT / RIGHT 更新 turn_intent。
    """
    try:
        ocr = OCRRecognizer(core_id=config.REC_CORE)
        print(f"✅ OCR (Core {config.REC_CORE}) 已就绪", flush=True)
    except Exception as e:
        print(f"❌ OCR 启动失败: {e}", flush=True)
        return

    while True:
        try:
            job = ocr_queue.get()
            if job is None:
                break

            frame_fid, sign_crops = job
            updates = []

            for idx, roi in sign_crops:
                try:
                    if roi is None or roi.size == 0:
                        continue

                    raw_text, score = ocr.run_single_crop(roi)
                    text = raw_text.strip().upper()
                    print(
                        f"OCR fid={frame_fid} idx={idx} roi={roi.shape[1]}x{roi.shape[0]} "
                        f"raw={raw_text!r} norm={text!r} score={float(score):.3f}",
                        flush=True
                    )
                    updates.append((idx, text, float(score)))
                except Exception as e:
                    print(f"OCR单框异常: {e}", flush=True)

            if not updates:
                continue

            # 回写时再做一次 class_id 检查，避免队列延迟导致“框已经换帧”的情况。
            with data_lock:
                if global_yolo_frame_id != frame_fid:
                    continue
                for idx, text, score in updates:
                    if idx >= len(global_yolo_boxes):
                        continue
                    if global_yolo_boxes[idx].get("class_id") != getattr(config, "SIGN_CLASS_ID", 9):
                        continue

                    global_yolo_boxes[idx]["text"] = text
                    global_yolo_boxes[idx]["ocr_score"] = score

                    if text == "LEFT":
                        global_control_data["turn_intent"] = -1
                    elif text == "RIGHT":
                        global_control_data["turn_intent"] = 1

        except Exception as e:
            print(f"OCR线程异常: {e}", flush=True)

# ==============================================================================
# 核心线程 2：分割与路径规划线程
# ==============================================================================
def seg_worker(core_id):
    """分割 + 路径规划线程.

    它是控制闭环的核心：
    - 从 seg_queue 取最新 320x320 RGB 图
    - 结合当前检测结果和转向意图生成 err_x / l_k
    - 渲染最终预览画面
    """
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

        # segmentor 输出是 320 空间画面，这里统一放大回 TARGET_RES，
        # 让检测框和页面显示都处于同一坐标系。
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

            cv2.rectangle(rendered_img, (2, 156), (520, 178), (0, 0, 0), -1)
            cv2.putText(
                rendered_img,
                summarize_yolo_boxes(current_yolo_boxes),
                (6, 172),
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
    """根据 err_x / l_k 持续输出底层控制命令."""
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

            # 弯越大，目标速度越低，避免高速出弯失控。
            target_speed = 30 - int(abs(l_k) * 10)
            target_speed = int(max(min_speed, min(30, target_speed)))

            # 一个简单但够用的串级控制：
            # 横向误差负责“拉回中心”，斜率负责“提前预瞄修正”。
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
    """从共享内存读取最新帧，并分发给 Seg / YOLO 两条支路.

    注意:
    - 这里的设计是“永远只保留最新帧”，所以两个队列都是小容量；
    - 如果下游处理不过来，会主动丢弃旧帧，优先保证实时性。
    """
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

                # 分割分支直接使用 320x320 RGB 小图，尽可能减轻主控制链路负担。
                seg_blob = cv2.resize(frame_rgb, config.SEG_SIZE, interpolation=cv2.INTER_NEAREST)
                if seg_queue.full():
                    try:
                        seg_queue.get_nowait()
                    except:
                        pass
                seg_queue.put(seg_blob)

                # 检测分支直接生成 YOLO 输入尺寸的小图，避免大图先放大再缩小。
                # 原始分辨率 BGR 图只放进一个很小的缓存里，等 sign 命中后再按 fid 取回。
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                det_img = cv2.resize(frame_bgr, config.YOLO_SIZE, interpolation=cv2.INTER_LINEAR)
                cache_raw_frame(fid, frame_bgr)

                if yolo_queue.full():
                    try:
                        yolo_queue.get_nowait()
                    except:
                        pass
                yolo_queue.put((fid, det_img))

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
    """MJPEG 推流接口."""
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
    """按模块顺序启动所有线程和 Flask 服务."""
    print("🚀 Aero-Twin [极限性能版] 启动", flush=True)

    threading.Thread(target=ai_producer_thread, daemon=True).start()
    threading.Thread(target=serial_control_thread, daemon=True).start()
    time.sleep(0.1)

    for core_id in config.SEG_CORES:
        threading.Thread(target=seg_worker, args=(core_id,), daemon=True).start()
        time.sleep(0.2)

    threading.Thread(target=yolo_worker, daemon=True).start()
    threading.Thread(target=ocr_worker, daemon=True).start()

    app.run(host='0.0.0.0', port=config.STREAM_PORT, threaded=True)

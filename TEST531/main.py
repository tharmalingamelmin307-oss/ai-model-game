"""系统主入口.

整体流程:
1. ai_producer_thread 从共享内存读取最新图像。
2. 图像被拆成两条支路:
   - seg_queue: 送给分割 / 路径规划线程
   - yolo_queue: 送给目标检测线程
3. yolo_worker 只负责检测；如果检测到 sign / limit_sign，再把 OCR 任务异步送入 ocr_queue。
4. ocr_worker 单独占用一个 NPU 核，对整张 TARGET_RES 图执行 OCR det + rec，
   再把识别结果按中心点回匹配到 sign / limit_sign 检测框。
5. seg_worker 读取当前最新的检测结果与 turn_intent，生成控制量和预览图。
6. serial_control_thread 将控制量打包后发给下位机。
7. Flask 将 global_preview_frame 编码成 MJPEG 提供网页预览。
"""

import time
import struct
import copy
import socket
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
try:
    from utils.rknn_quiet import install_rknn_warning_filter
except ImportError:
    def install_rknn_warning_filter():
        return

app = Flask(__name__)

# ==============================================================================
# 全局状态与资源锁
# ==============================================================================
# 当前网页预览的最新画面，只由 seg_worker 写，由 Flask 推流读取。
global_preview_frame = None

# 供分割线程、串口线程、OCR 线程共享的控制状态。
# 这份状态是系统里最核心的一块“跨线程控制面板”：
# - seg_worker 写入 steer_signal / 红绿灯 / 停止线
# - ocr_worker 写入 turn_intent / speed_limit
# - serial_control_thread 读取这些状态并生成最终底层控制命令
global_control_data = copy.deepcopy(config.DEFAULT_CONTROL_DATA)

# 用于在页面上显示 Seg / YOLO 处理频率。
fps_stats = copy.deepcopy(config.DEFAULT_FPS_STATS)

frame_lock = threading.Lock()
data_lock = threading.Lock()
log_lock = threading.Lock()

# 三条工作队列:
# - seg_queue: 最新一帧分割输入
# - yolo_queue: 最新一帧检测输入
# - ocr_queue: 检测线程生成的 sign / limit_sign OCR 任务
seg_queue = Queue(maxsize=config.SEG_QUEUE_MAXSIZE)
yolo_queue = Queue(maxsize=config.YOLO_QUEUE_MAXSIZE)
ocr_queue = Queue(maxsize=config.OCR_QUEUE_MAXSIZE)

# 当前最新一帧的检测结果。
global_yolo_boxes = []
global_yolo_frame_id = -1
log_cache = {}
profile_cache = {}


def remove_shm_from_resource_tracker():
    """避免 Python 退出时错误回收外部创建的共享内存对象."""
    try:
        resource_tracker.unregister('/' + config.SHM_NAME, 'shared_memory')
    except:
        pass


def throttled_log(key, message, state=None, min_interval=None):
    """按状态变化或最小时间间隔打印日志，避免终端刷屏."""
    now = time.time()
    if min_interval is None:
        min_interval = config.LOG_INTERVAL_DEFAULT
    with log_lock:
        prev = log_cache.get(key)
        should_print = False
        if prev is None:
            should_print = True
        else:
            prev_state = prev.get("state")
            prev_time = prev.get("time", 0.0)
            if state is not None and state != prev_state:
                should_print = True
            elif now - prev_time >= float(min_interval):
                should_print = True

        if should_print:
            print(message, flush=True)
            log_cache[key] = {"time": now, "state": state}


def log_once(key, message):
    """同一类错误只打印一次，避免异常反复刷屏。"""
    with log_lock:
        if key in log_cache:
            return
        print(message, flush=True)
        log_cache[key] = {"time": time.time(), "state": "__once__"}


def profile_log(key, label, metrics, min_interval=None):
    """用 EMA 节流打印主流程耗时，数值单位统一按毫秒展示."""
    if not bool(getattr(config, "MAIN_PROFILE_LOG_ENABLED", False)):
        return

    now = time.time()
    if min_interval is None:
        min_interval = float(getattr(config, "MAIN_PROFILE_LOG_INTERVAL", 2.0))

    with log_lock:
        item = profile_cache.get(key)
        if item is None:
            ema = {name: float(value) for name, value in metrics.items()}
            item = {"ema": ema, "time": 0.0}
            profile_cache[key] = item
        else:
            alpha = 0.85
            ema = item["ema"]
            for name, value in metrics.items():
                ema[name] = alpha * float(ema.get(name, 0.0)) + (1.0 - alpha) * float(value)

        if now - float(item.get("time", 0.0)) < min_interval:
            return
        item["time"] = now

        parts = []
        for name, value in item["ema"].items():
            value = float(value)
            if name.endswith("_fps"):
                parts.append(f"{name}={value:.1f}")
            else:
                parts.append(f"{name}={value * 1000.0:.1f}ms")
        print(f"{label} " + " ".join(parts), flush=True)


def get_preview_host():
    """返回适合局域网浏览器访问的预览主机地址."""
    bind_host = str(config.FLASK_HOST)
    if bind_host and bind_host not in ("0.0.0.0", "::"):
        return bind_host

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"
    finally:
        sock.close()


def print_preview_url():
    """启动时主动打印一条可点击的网页推流地址."""
    host = get_preview_host()
    print(f"AI推流网页: http://{host}:{config.STREAM_PORT}/", flush=True)


def make_seg_input(frame_rgb):
    """按当前分割模型约定生成 RGB 输入图."""
    crop_ratio = float(getattr(config, "SEG_INPUT_CROP_TOP_RATIO", 0.0))
    crop_ratio = max(0.0, min(0.95, crop_ratio))
    if crop_ratio > 0.0:
        h = frame_rgb.shape[0]
        crop_y = int(round(h * crop_ratio))
        frame_rgb = frame_rgb[crop_y:, :, :]

    return cv2.resize(frame_rgb, config.SEG_SIZE, interpolation=cv2.INTER_LINEAR)


def expand_seg_render_to_target(rendered_img, base_frame=None):
    """把裁剪坐标系的 Seg 调试图贴回 TARGET_RES 预览画布."""
    target_w, target_h = config.TARGET_RES
    crop_ratio = float(getattr(config, "SEG_INPUT_CROP_TOP_RATIO", 0.0))
    crop_ratio = max(0.0, min(0.95, crop_ratio))
    if crop_ratio <= 0.0:
        if rendered_img.shape[1] == target_w and rendered_img.shape[0] == target_h:
            return rendered_img
        return cv2.resize(rendered_img, config.TARGET_RES, interpolation=cv2.INTER_NEAREST)

    crop_y = int(round(target_h * crop_ratio))
    crop_y = max(0, min(target_h - 1, crop_y))
    bottom_h = target_h - crop_y
    seg_view = cv2.resize(rendered_img, (target_w, bottom_h), interpolation=cv2.INTER_NEAREST)
    if base_frame is not None:
        if base_frame.shape[1] != target_w or base_frame.shape[0] != target_h:
            canvas = cv2.resize(base_frame, config.TARGET_RES, interpolation=cv2.INTER_LINEAR)
        else:
            canvas = base_frame.copy()
        base_bottom = canvas[crop_y:, :]
        diff = cv2.absdiff(seg_view, base_bottom)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        overlay_mask = diff_gray > int(getattr(config, "SEG_PREVIEW_OVERLAY_DIFF_THRESH", 24))
        base_bottom[overlay_mask] = seg_view[overlay_mask]
    else:
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[crop_y:, :] = seg_view
    return canvas


def unpack_seg_item(item):
    """兼容旧的纯 seg_blob 队列项和新的 (seg_blob, preview_frame) 队列项."""
    if isinstance(item, tuple) and len(item) == 2:
        return item
    return item, None


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

        color = config.YOLO_DEFAULT_BOX_COLOR
        if cls_id == config.SIGN_CLASS_ID:
            color = config.YOLO_SIGN_BOX_COLOR
        elif cls_id == config.LIMIT_SIGN_CLASS_ID:
            color = config.YOLO_LIMIT_SIGN_BOX_COLOR

        # 在主预览图上把检测框画粗一点，方便快速确认检测是否生效。
        cv2.rectangle(image, (x1, y1), (x2, y2), color, config.YOLO_BOX_THICKNESS)

        label = f"{cls_name}:{score:.2f}"
        if text:
            label += f" [{text}]"

        text_y = (
            y1 - config.YOLO_LABEL_TOP_OFFSET
            if y1 > config.YOLO_LABEL_TOP_MARGIN
            else y1 + config.YOLO_LABEL_BOTTOM_OFFSET
        )
        cv2.putText(
            image,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.YOLO_LABEL_FONT_SCALE,
            color,
            config.YOLO_LABEL_THICKNESS,
            cv2.LINE_AA
        )

    return image


def summarize_yolo_boxes(boxes):
    """生成一行简短检测摘要，便于在预览页直接确认 YOLO 是否有输出。"""
    if not boxes:
        return "YOLO:0"

    parts = [f"YOLO:{len(boxes)}"]
    for obj in boxes[:config.YOLO_SUMMARY_MAX_ITEMS]:
        cls_name = obj.get("class_name", "?")
        score = float(obj.get("score", 0.0))
        parts.append(f"{cls_name}:{score:.2f}")
    if len(boxes) > config.YOLO_SUMMARY_MAX_ITEMS:
        parts.append("...")
    return " | ".join(parts)


def extract_scene_control_signals(boxes):
    """从检测结果中提取斑马线停止线和当前交通灯状态."""
    zebra_cls_id = (
        config.CLASS_NAMES.index(config.ZEBRA_CROSSING_CLASS_NAME)
        if config.ZEBRA_CROSSING_CLASS_NAME in config.CLASS_NAMES
        else config.ZEBRA_CROSSING_CLASS_ID_FALLBACK
    )
    red_cls_id = (
        config.CLASS_NAMES.index(config.TRAFFIC_LIGHT_RED_CLASS_NAME)
        if config.TRAFFIC_LIGHT_RED_CLASS_NAME in config.CLASS_NAMES
        else config.TRAFFIC_LIGHT_RED_CLASS_ID_FALLBACK
    )
    green_cls_id = (
        config.CLASS_NAMES.index(config.TRAFFIC_LIGHT_GREEN_CLASS_NAME)
        if config.TRAFFIC_LIGHT_GREEN_CLASS_NAME in config.CLASS_NAMES
        else config.TRAFFIC_LIGHT_GREEN_CLASS_ID_FALLBACK
    )
    yellow_cls_id = (
        config.CLASS_NAMES.index(config.TRAFFIC_LIGHT_YELLOW_CLASS_NAME)
        if config.TRAFFIC_LIGHT_YELLOW_CLASS_NAME in config.CLASS_NAMES
        else config.TRAFFIC_LIGHT_YELLOW_CLASS_ID_FALLBACK
    )

    zebra_stopline = None
    zebra_bottom_y = -1
    red_score = -1.0
    yellow_score = -1.0
    green_score = -1.0

    for obj in boxes:
        rect = obj.get("rect", [0, 0, 0, 0])
        if len(rect) != 4:
            continue

        x, y, w, h = rect
        cls_id = obj.get("class_id", -1)
        score = float(obj.get("score", 0.0))

        if cls_id == zebra_cls_id:
            bottom_y = y + h
            if bottom_y > zebra_bottom_y:
                zebra_bottom_y = bottom_y
                zebra_stopline = (x, y + h, w)
        elif cls_id == red_cls_id:
            red_score = max(red_score, score)
        elif cls_id == yellow_cls_id:
            yellow_score = max(yellow_score, score)
        elif cls_id == green_cls_id:
            green_score = max(green_score, score)

    if red_score >= 0:
        best_light_state = "red"
    elif yellow_score >= 0:
        best_light_state = "yellow"
    elif green_score >= 0:
        best_light_state = "green"
    else:
        best_light_state = ""

    return zebra_stopline, best_light_state


def extract_person_stop_candidate(boxes, frame_id):
    """取最靠近车身的 person 框底边，用于行人停车状态机."""
    person_cls_id = (
        config.CLASS_NAMES.index(config.PERSON_CLASS_NAME)
        if config.PERSON_CLASS_NAME in config.CLASS_NAMES
        else config.PERSON_CLASS_ID_FALLBACK
    )

    best = None
    best_bottom_y = -1.0
    target_w, target_h = config.TARGET_RES

    for obj in boxes:
        if obj.get("class_id", -1) != person_cls_id:
            continue
        rect = obj.get("rect", [0, 0, 0, 0])
        if len(rect) != 4:
            continue

        x, y, w, h = [float(v) for v in rect]
        if w <= 0.0 or h <= 0.0:
            continue

        bottom_y = float(np.clip(y + h, 0.0, float(target_h - 1)))
        if bottom_y <= best_bottom_y:
            continue

        x1 = float(np.clip(x, 0.0, float(target_w - 1)))
        x2 = float(np.clip(x + w, 0.0, float(target_w - 1)))
        best_bottom_y = bottom_y
        best = {
            "frame_id": int(frame_id),
            "bottom_y": bottom_y,
            "bottom_center_x": 0.5 * (x1 + x2),
            "bottom_right_x": max(x1, x2),
            "dist_to_bottom": float(target_h) - bottom_y,
            "score": float(obj.get("score", 0.0)),
        }

    return best


def update_person_stop_state(state, person_info, left_boundary_x, right_boundary_x, yolo_frame_id):
    """行人靠近时停车；连续左移且越过道路释放线后放行."""
    active = bool(state.get("person_stop_active", False))
    prev_active = active
    released_outside_left = bool(state.get("person_released_outside_left", False))
    yolo_frame_id = int(yolo_frame_id)
    last_frame_id = int(state.get("person_last_frame_id", -1))
    if yolo_frame_id == last_frame_id:
        if person_info is not None:
            state["person_left_boundary_x"] = left_boundary_x
            state["person_right_boundary_x"] = right_boundary_x
        return active

    clear_frames = int(state.get("person_clear_frames", 0))
    miss_frames = int(state.get("person_miss_frames", 0))

    if person_info is None:
        miss_frames += 1
        clear_frames = 0
        if active and miss_frames >= int(config.PERSON_STOP_MISS_RELEASE_FRAMES):
            active = False
        if miss_frames >= int(config.PERSON_STOP_MISS_RELEASE_FRAMES):
            released_outside_left = False
        state["person_stop_active"] = active
        state["person_bottom_y"] = None
        state["person_bottom_center_x"] = None
        state["person_bottom_right_x"] = None
        state["person_dist_to_bottom"] = None
        state["person_left_boundary_x"] = None
        state["person_right_boundary_x"] = None
        state["person_clear_line_x"] = None
        state["person_miss_frames"] = miss_frames
        state["person_clear_frames"] = clear_frames
        state["person_last_frame_id"] = yolo_frame_id
        state["person_last_bottom_center_x"] = None
        state["person_released_outside_left"] = released_outside_left
        state["person_stop_event"] = "release_missing" if prev_active and not active else ""
        return active

    bottom_center_x = float(person_info["bottom_center_x"])
    bottom_right_x = float(person_info["bottom_right_x"])
    bottom_y = float(person_info["bottom_y"])
    dist_to_bottom = float(person_info["dist_to_bottom"])
    last_center = state.get("person_last_bottom_center_x")
    moving_left = (
        last_center is not None and
        bottom_center_x <= float(last_center) - float(config.PERSON_CLEAR_MIN_LEFT_DX)
    )
    clear_line_x = None
    if left_boundary_x is not None and right_boundary_x is not None:
        lane_ratio = float(getattr(config, "PERSON_CLEAR_LANE_RATIO", 0.50))
        lane_ratio = max(0.0, min(1.0, lane_ratio))
        clear_line_x = float(left_boundary_x) + (float(right_boundary_x) - float(left_boundary_x)) * lane_ratio
    elif left_boundary_x is not None:
        clear_line_x = float(left_boundary_x)
    crossed_clear_line = clear_line_x is not None and bottom_center_x <= float(clear_line_x)
    trigger_dist = float(config.PERSON_STOP_TRIGGER_DIST)
    near_bottom = dist_to_bottom <= trigger_dist

    if not crossed_clear_line:
        released_outside_left = False

    if near_bottom and not released_outside_left:
        active = True
    miss_frames = 0

    if active and moving_left and crossed_clear_line:
        clear_frames += 1
    else:
        clear_frames = 0

    if active and clear_frames >= int(config.PERSON_CLEAR_MOVE_FRAMES):
        active = False
        clear_frames = 0
        released_outside_left = True

    state["person_stop_active"] = active
    state["person_bottom_y"] = bottom_y
    state["person_bottom_center_x"] = bottom_center_x
    state["person_bottom_right_x"] = bottom_right_x
    state["person_dist_to_bottom"] = dist_to_bottom
    state["person_left_boundary_x"] = left_boundary_x
    state["person_right_boundary_x"] = right_boundary_x
    state["person_clear_line_x"] = clear_line_x
    state["person_clear_frames"] = clear_frames
    state["person_miss_frames"] = miss_frames
    state["person_last_frame_id"] = yolo_frame_id
    state["person_last_bottom_center_x"] = bottom_center_x
    state["person_released_outside_left"] = released_outside_left
    if not prev_active and active:
        state["person_stop_event"] = "stop"
    elif prev_active and not active:
        state["person_stop_event"] = "release_clear_line"
    else:
        state["person_stop_event"] = ""
    return active


def draw_zebra_stopline(image, zebra_stopline):
    """把斑马线框底部当作截至线画出来，并向左右延长."""
    if image is None or zebra_stopline is None:
        return image

    img_h, img_w = image.shape[:2]
    src_w, src_h = config.TARGET_RES
    scale_x = img_w / float(src_w)
    scale_y = img_h / float(src_h)

    x, line_y, w = zebra_stopline
    extend_w = w * float(config.ZEBRA_STOPLINE_EXTEND_RATIO)

    x1 = int(np.clip(round((x - extend_w) * scale_x), 0, img_w - 1))
    x2 = int(np.clip(round((x + w + extend_w) * scale_x), 0, img_w - 1))
    y = int(np.clip(round(line_y * scale_y), 0, img_h - 1))

    cv2.line(
        image,
        (x1, y),
        (x2, y),
        config.ZEBRA_STOPLINE_COLOR,
        config.ZEBRA_STOPLINE_THICKNESS,
        cv2.LINE_AA,
    )
    return image


def should_enqueue_ocr_job(cls_id, rect):
    """判断某个 sign / limit_sign 是否值得触发一次整图 OCR.

    当前不是“直接裁检测框做 OCR”，而是:
    1. YOLO 先给出路牌框
    2. 只有框足够大、且没有明显贴边截断风险时，才触发一次整图 OCR
    3. OCR det + rec 的结果再回匹配到这个框

    额外约束:
    - 对 limit_sign 来说，如果框面积已经达到 LIMIT_SIGN_APPLY_MIN_AREA，
      就不再继续 OCR，而是转去使用历史聚合结果做正式生效判定。

    返回:
        (should_enqueue, reason)
        - should_enqueue: 是否允许进入 OCR
        - reason: 便于日志打印的跳过原因
    """
    if len(rect) != 4:
        return False, "invalid_rect"

    x, y, w, h = rect
    if w < 2 or h < 2:
        return False, "invalid_rect"

    area = float(w * h)
    if cls_id == config.SIGN_CLASS_ID:
        min_area = float(config.OCR_MIN_SIGN_BOX_AREA)
    elif cls_id == config.LIMIT_SIGN_CLASS_ID:
        if not bool(getattr(config, "LIMIT_SIGN_ENABLED", True)):
            return False, "limit_sign_disabled"
        min_area = float(config.OCR_MIN_LIMIT_SIGN_BOX_AREA)
        apply_min_area = float(config.LIMIT_SIGN_APPLY_MIN_AREA)
        if area >= apply_min_area:
            return False, f"ready_to_apply_history({int(area)}>={int(apply_min_area)})"
    else:
        return False, "non_ocr_class"

    if area < min_area:
        return False, f"area_too_small({int(area)}<{int(min_area)})"

    frame_w, frame_h = config.TARGET_RES
    edge_margin_ratio = float(config.OCR_SIGN_EDGE_MARGIN_RATIO)
    edge_margin_x = frame_w * edge_margin_ratio
    edge_margin_y = frame_h * edge_margin_ratio

    if (
        x <= edge_margin_x or
        y <= edge_margin_y or
        (x + w) >= (frame_w - edge_margin_x) or
        (y + h) >= (frame_h - edge_margin_y)
    ):
        return False, "too_close_to_edge"

    return True, "ok"


def class_name_from_id(cls_id):
    """把类别 id 转成类别名，便于终端调试打印。"""
    if 0 <= int(cls_id) < len(config.CLASS_NAMES):
        return config.CLASS_NAMES[int(cls_id)]
    return str(cls_id)


def rect_center(rect):
    """返回 [x, y, w, h] 框的中心点."""
    if len(rect) != 4:
        return (0.0, 0.0)
    x, y, w, h = rect
    return (x + w * 0.5, y + h * 0.5)


def points_center(points):
    """返回四点框的中心点."""
    pts = np.array(points, dtype=np.float32)
    if pts.size == 0:
        return (0.0, 0.0)
    center = np.mean(pts, axis=0)
    return (float(center[0]), float(center[1]))


def rect_area(rect):
    """返回 [x, y, w, h] 框面积."""
    if len(rect) != 4:
        return 0.0
    return float(max(0, rect[2]) * max(0, rect[3]))


def reset_limit_sign_history(state):
    """清空当前限速牌历史聚合统计."""
    state["limit_sign_history"] = {}


def update_limit_sign_history(state, digit_text, score):
    """按数字聚合累计限速牌 OCR 结果，只保留次数和分数和."""
    history = state.setdefault("limit_sign_history", {})
    item = history.setdefault(str(digit_text), {"count": 0, "score_sum": 0.0})
    item["count"] = int(item.get("count", 0)) + 1
    item["score_sum"] = float(item.get("score_sum", 0.0)) + float(score)

    count = int(item["count"])
    avg_score = float(item["score_sum"]) / float(max(count, 1))
    return count, avg_score


def select_best_limit_sign_candidate(state):
    """从限速牌历史聚合统计里选当前最优候选。

    规则:
    1. 优先从出现次数达到 LIMIT_SIGN_CONFIRM_FRAMES 的候选里挑
    2. 若都没达到，再从全部候选里挑
    3. 同一池子里先比 count，再比平均置信度，再比分数和
    """
    history = dict(state.get("limit_sign_history", {}))
    if not history:
        return None

    confirm_needed = max(1, int(config.LIMIT_SIGN_CONFIRM_FRAMES))
    candidates = []
    for digit_text, item in history.items():
        count = int(item.get("count", 0))
        score_sum = float(item.get("score_sum", 0.0))
        if count <= 0:
            continue
        avg_score = score_sum / float(count)
        candidates.append({
            "digit_text": str(digit_text),
            "count": count,
            "score_sum": score_sum,
            "avg_score": avg_score,
        })

    if not candidates:
        return None

    stable_candidates = [c for c in candidates if c["count"] >= confirm_needed]
    pool = stable_candidates if stable_candidates else candidates
    return max(pool, key=lambda c: (c["count"], c["avg_score"], c["score_sum"]))


# ==============================================================================
# 核心线程 1：YOLO 检测线程
# ==============================================================================
def yolo_worker():
    """纯检测线程.

    这个线程只做两件事:
    1. 执行 YOLO 推理并更新 global_yolo_boxes
    2. 将 sign / limit_sign 类别的候选任务连同同帧 TARGET_RES 图一起投递给 OCR 线程

    这样 OCR 不会反过来阻塞 YOLO，能显著降低检测延迟。
    """
    try:
        det = YOLODetector(core_id=config.YOLO_CORE)
    except Exception as e:
        log_once("yolo_init_error", f"YOLO启动失败: {e}")
        return

    while True:
        try:
            frame_data = yolo_queue.get()
            if frame_data is None:
                break

            det_frame, vis_frame, src_size, frame_id = frame_data
            objs = det.run(det_frame, output_size=config.TARGET_RES)

            # 新一帧检测结果先清掉旧的 OCR 文本，避免沿用上一帧残留内容。
            for obj in objs:
                try:
                    obj.pop("text", None)
                    obj.pop("ocr_score", None)
                except Exception:
                    pass

            with data_lock:
                global global_yolo_boxes, global_yolo_frame_id
                global_yolo_boxes = objs
                global_yolo_frame_id = int(frame_id)
                fps_stats["yolo_frames"] += 1

            # 只把通过门槛的 sign / limit_sign 送去 OCR，减少不必要的整图文字检测开销。
            # 其中 limit_sign 一旦面积达到“生效阈值”，就不再继续 OCR，
            # 而是直接从历史聚合统计里挑选最优数字。
            sign_jobs = []
            apply_limit_jobs = []
            limit_sign_seen = False
            for idx, obj in enumerate(objs):
                cls_id = obj.get("class_id")
                if cls_id not in (
                    config.SIGN_CLASS_ID,
                    config.LIMIT_SIGN_CLASS_ID,
                ):
                    continue
                rect = obj.get("rect", [0, 0, 0, 0])
                if cls_id == config.LIMIT_SIGN_CLASS_ID:
                    limit_sign_seen = True
                should_enqueue, skip_reason = should_enqueue_ocr_job(cls_id, rect)
                if not should_enqueue:
                    if (
                        cls_id == config.LIMIT_SIGN_CLASS_ID and
                        skip_reason.startswith("ready_to_apply_history")
                    ):
                        apply_limit_jobs.append((idx, rect))
                        continue
                    continue
                sign_jobs.append((idx, cls_id, rect))

            limit_enabled = bool(getattr(config, "LIMIT_SIGN_ENABLED", True))
            limit_applied_from_history = None
            with data_lock:
                if limit_enabled and limit_sign_seen:
                    global_control_data["limit_sign_last_detect_fid"] = int(frame_id)
                elif limit_enabled:
                    last_limit_detect_fid = int(global_control_data.get("limit_sign_last_detect_fid", -1))
                    max_miss = max(1, int(config.LIMIT_SIGN_HISTORY_MAX_MISS_FRAMES))
                    if last_limit_detect_fid >= 0 and int(frame_id) - last_limit_detect_fid > max_miss:
                        if global_control_data.get("limit_sign_history"):
                            reset_limit_sign_history(global_control_data)
                        global_control_data["limit_sign_last_detect_fid"] = -1

                if limit_enabled and apply_limit_jobs:
                    best_apply_idx, best_apply_rect = max(
                        apply_limit_jobs,
                        key=lambda item: rect_area(item[1]),
                    )
                    best_candidate = select_best_limit_sign_candidate(global_control_data)
                    if best_candidate is not None:
                        prev_effective_limit = global_control_data.get("speed_limit")
                        last_speed_fid = int(global_control_data.get("speed_limit_fid", -1))
                        if int(frame_id) >= last_speed_fid:
                            speed_limit = int(best_candidate["digit_text"])
                            effective_limit = max(0, speed_limit - config.LIMIT_SIGN_EFFECTIVE_SPEED_OFFSET)
                            global_control_data["speed_limit"] = effective_limit
                            global_control_data["speed_limit_fid"] = int(frame_id)
                            reset_limit_sign_history(global_control_data)
                            limit_applied_from_history = {
                                "digit_text": best_candidate["digit_text"],
                                "count": int(best_candidate["count"]),
                                "avg_score": float(best_candidate["avg_score"]),
                                "effective_limit": effective_limit,
                                "area": int(rect_area(best_apply_rect)),
                                "changed": effective_limit != prev_effective_limit,
                            }

                            if best_apply_idx < len(global_yolo_boxes):
                                if global_yolo_boxes[best_apply_idx].get("class_id") == config.LIMIT_SIGN_CLASS_ID:
                                    global_yolo_boxes[best_apply_idx]["text"] = best_candidate["digit_text"]
                                    global_yolo_boxes[best_apply_idx]["ocr_score"] = float(best_candidate["avg_score"])

            if limit_enabled and limit_applied_from_history is not None:
                throttled_log(
                    "speed_limit_effective",
                    "限速生效: "
                    f"历史最优={limit_applied_from_history['digit_text']} "
                    f"次数={limit_applied_from_history['count']} "
                    f"平均置信度={limit_applied_from_history['avg_score']:.3f} "
                    f"牌面面积={limit_applied_from_history['area']} "
                    f"实际上限={limit_applied_from_history['effective_limit']}",
                    state=(
                        limit_applied_from_history["digit_text"],
                        limit_applied_from_history["count"],
                        round(limit_applied_from_history["avg_score"], 3),
                        limit_applied_from_history["effective_limit"],
                    ),
                    min_interval=config.LOG_INTERVAL_SPEED_LIMIT_EFFECTIVE
                )

            if sign_jobs:
                job_names = tuple(class_name_from_id(cls_id) for _, cls_id, _ in sign_jobs)
                throttled_log(
                    "ocr_enter",
                    f"路牌达到识别条件: 数量={len(sign_jobs)} 类型={list(job_names)}",
                    state=job_names,
                    min_interval=config.LOG_INTERVAL_OCR_ENTER
                )
                # OCR 只保留较新的任务，过旧的任务直接丢掉。
                if ocr_queue.full():
                    try:
                        ocr_queue.get_nowait()
                    except:
                        pass
                ocr_queue.put((vis_frame.copy(), sign_jobs, int(frame_id)))

        except Exception as e:
            log_once("yolo_worker_error", f"YOLO线程异常: {e}")


# ==============================================================================
# 核心线程 1.5：OCR 识别线程
# ==============================================================================
def ocr_worker():
    """纯 OCR 线程.

    输入:
        yolo_worker 投递的 (frame_data, sign_jobs, frame_id)

    输出:
        将识别出的 text / ocr_score 回写到 global_yolo_boxes，
        同时根据 LEFT / RIGHT 更新 turn_intent，根据限速牌更新 speed_limit。

    关键约束:
    - OCR 在整张 TARGET_RES 图上执行 det + rec，不是直接拿检测框裁图识别
    - 同一帧里的多个 OCR 结果，会按“中心点最近”去匹配各个路牌框
    - 匹配结果回写时还会再核对 frame_id，避免旧帧 OCR 迟到污染新状态
    """
    try:
        ocr = OCRRecognizer(core_id=config.REC_CORE)
    except Exception as e:
        log_once("ocr_init_error", f"OCR启动失败: {e}")
        return

    while True:
        try:
            job = ocr_queue.get()
            if job is None:
                break

            frame_data, sign_jobs, frame_id = job
            updates = []
            # 这里跑的是整图 OCR，再把结果按中心点回匹配给 sign_jobs。
            ocr_results = ocr.run_full_frame(frame_data)
            if not ocr_results:
                continue

            used_result_ids = set()
            for idx, cls_id, rect in sign_jobs:
                try:
                    det_cx, det_cy = rect_center(rect)
                    best_match = None
                    best_dist = float(config.OCR_MATCH_INIT_DIST)

                    for result_id, result in enumerate(ocr_results):
                        if result_id in used_result_ids:
                            continue
                        ocr_cx, ocr_cy = points_center(result.get("points"))
                        dist = (ocr_cx - det_cx) ** 2 + (ocr_cy - det_cy) ** 2
                        if dist < best_dist:
                            best_dist = dist
                            best_match = (result_id, result)

                    if best_match is None:
                        continue

                    result_id, result = best_match
                    used_result_ids.add(result_id)
                    text = str(result.get("text", "")).strip().upper()
                    score = float(result.get("score", 0.0))

                    if not text:
                        continue

                    if score < float(config.OCR_MIN_SCORE):
                        continue

                    updates.append((idx, cls_id, text, score))
                except Exception as e:
                    log_once("ocr_single_box_error", f"OCR单框处理异常: {e}")

            if not updates:
                continue

            # 回写时再做一次 class_id 检查，避免队列延迟导致“框已经换帧”的情况。
            with data_lock:
                current_yolo_frame_id = int(global_yolo_frame_id)
                sign_class_id = config.SIGN_CLASS_ID
                limit_sign_class_id = config.LIMIT_SIGN_CLASS_ID

                for idx, job_cls_id, text, score in updates:
                    # 只有 OCR 结果仍对应当前显示的检测帧时，才把文字回写到显示框上。
                    if (
                        int(frame_id) == current_yolo_frame_id and
                        idx < len(global_yolo_boxes) and
                        global_yolo_boxes[idx].get("class_id") == job_cls_id
                    ):
                        global_yolo_boxes[idx]["text"] = text
                        global_yolo_boxes[idx]["ocr_score"] = score

                    if job_cls_id == sign_class_id:
                        last_turn_fid = int(global_control_data.get("turn_intent_fid", -1))
                        if int(frame_id) >= last_turn_fid:
                            throttled_log(
                                f"ocr_sign_result_{text}",
                                f"路牌识别结果: 文本={text} 置信度={score:.3f}",
                                state=text,
                                min_interval=config.LOG_INTERVAL_OCR_ENTER
                            )
                            if text == "LEFT":
                                global_control_data["turn_intent"] = -1
                                global_control_data["turn_intent_fid"] = int(frame_id)
                                throttled_log(
                                    "turn_intent",
                                    "语义路牌生效: 分叉意图=LEFT",
                                    state="LEFT",
                                    min_interval=config.LOG_INTERVAL_TURN_INTENT
                                )
                            elif text == "RIGHT":
                                global_control_data["turn_intent"] = 1
                                global_control_data["turn_intent_fid"] = int(frame_id)
                                throttled_log(
                                    "turn_intent",
                                    "语义路牌生效: 分叉意图=RIGHT",
                                    state="RIGHT",
                                    min_interval=config.LOG_INTERVAL_TURN_INTENT
                                )
                    elif job_cls_id == limit_sign_class_id:
                        try:
                            digit_text = "".join(ch for ch in text if ch.isdigit())
                            if not digit_text:
                                continue

                            throttled_log(
                                f"ocr_limit_result_{digit_text}",
                                f"限速牌识别结果: 文本={digit_text} 置信度={score:.3f}",
                                state=digit_text,
                                min_interval=config.LOG_INTERVAL_OCR_ENTER
                            )

                            update_limit_sign_history(
                                global_control_data,
                                digit_text,
                                score,
                            )
                        except Exception:
                            pass

        except Exception as e:
            log_once("ocr_worker_error", f"OCR线程异常: {e}")

# ==============================================================================
# 核心线程 2：分割与路径规划线程
# ==============================================================================
def seg_worker(core_id):
    """分割 + 路径规划线程.

    默认启用流水线:
    - 当前线程持有 RKNNLite，只做 NPU 推理并输出最新 mask
    - 内部后处理线程消费最新 mask，生成 steer_signal 和预览画面
    """
    global global_preview_frame
    fps_start_time = time.time()

    try:
        seg = RoadSegmentor(core_id=core_id)
    except Exception as e:
        log_once(f"seg_init_error_{core_id}", f"Seg启动失败(Core {core_id}): {e}")
        return

    def publish_seg_result(
        steer_signal,
        rendered_img,
        current_yolo_boxes,
        fps_start_holder,
        current_yolo_frame_id=-1,
        preview_frame=None,
    ):
        global global_preview_frame

        zebra_stopline, traffic_light_state = extract_scene_control_signals(current_yolo_boxes)
        person_info = extract_person_stop_candidate(current_yolo_boxes, current_yolo_frame_id)
        person_left_boundary_x = None
        person_right_boundary_x = None
        if person_info is not None:
            person_left_boundary_x = seg.selected_left_boundary_x_at_target_y(person_info["bottom_y"])
            person_right_boundary_x = seg.selected_right_boundary_x_at_target_y(person_info["bottom_y"])

        if rendered_img is not None:
            if rendered_img.shape[1] != config.TARGET_RES[0] or rendered_img.shape[0] != config.TARGET_RES[1]:
                rendered_img = expand_seg_render_to_target(rendered_img, preview_frame)
            rendered_img = draw_yolo_boxes(rendered_img, current_yolo_boxes)
            rendered_img = draw_zebra_stopline(rendered_img, zebra_stopline)

        with data_lock:
            global_control_data["steer_signal"] = steer_signal
            global_control_data["traffic_light_state"] = traffic_light_state
            global_control_data["zebra_stopline_y"] = None if zebra_stopline is None else int(zebra_stopline[1])
            person_stop_active = update_person_stop_state(
                global_control_data,
                person_info,
                person_left_boundary_x,
                person_right_boundary_x,
                current_yolo_frame_id,
            )

            actual_servo = global_control_data.get("actual_servo_pwm", config.SERVO_CENTER)
            actual_speed = global_control_data.get("target_speed", config.CONTROL_MIN_SPEED)
            speed_limit = (
                global_control_data.get("speed_limit")
                if bool(getattr(config, "LIMIT_SIGN_ENABLED", True))
                else None
            )
            traffic_stop_active = global_control_data.get("traffic_stop_active", False)

            fps_stats["seg_frames"] += 1
            now = time.time()
            if now - fps_start_holder[0] >= config.FPS_STATS_UPDATE_INTERVAL:
                fps_stats["seg_fps"] = fps_stats["seg_frames"] / (now - fps_start_holder[0])
                fps_stats["yolo_fps"] = fps_stats["yolo_frames"] / (now - fps_start_holder[0])
                fps_stats["seg_frames"] = 0
                fps_stats["yolo_frames"] = 0
                fps_start_holder[0] = now

            current_seg_fps = fps_stats["seg_fps"]
            current_yolo_fps = fps_stats["yolo_fps"]

        if rendered_img is not None:
            cv2.rectangle(
                rendered_img,
                config.PREVIEW_STATUS_PANEL_TOP_LEFT,
                config.PREVIEW_STATUS_PANEL_BOTTOM_RIGHT,
                config.PREVIEW_PANEL_BG_COLOR,
                -1,
            )
            cv2.rectangle(
                rendered_img,
                config.PREVIEW_STATUS_PANEL_TOP_LEFT,
                config.PREVIEW_STATUS_PANEL_BOTTOM_RIGHT,
                config.PREVIEW_PANEL_BORDER_COLOR,
                config.PREVIEW_TEXT_THICKNESS,
            )

            cv2.putText(
                rendered_img,
                f"Seg:{current_seg_fps:.1f} YOL:{current_yolo_fps:.1f}",
                config.PREVIEW_TEXT_POS_FPS,
                cv2.FONT_HERSHEY_SIMPLEX,
                config.PREVIEW_TEXT_FONT_SCALE,
                config.PREVIEW_TEXT_COLOR,
                config.PREVIEW_TEXT_THICKNESS,
                cv2.LINE_AA
            )
            cv2.putText(
                rendered_img,
                f"Ctrl:{steer_signal:.1f} Srv:{actual_servo}",
                config.PREVIEW_TEXT_POS_CTRL,
                cv2.FONT_HERSHEY_SIMPLEX,
                config.PREVIEW_TEXT_FONT_SCALE,
                config.PREVIEW_TEXT_COLOR,
                config.PREVIEW_TEXT_THICKNESS,
                cv2.LINE_AA
            )
            cv2.putText(
                rendered_img,
                f"Target Spd:{actual_speed}",
                config.PREVIEW_TEXT_POS_SPEED,
                cv2.FONT_HERSHEY_SIMPLEX,
                config.PREVIEW_TEXT_FONT_SCALE,
                config.PREVIEW_TEXT_ACCENT_COLOR,
                config.PREVIEW_TEXT_THICKNESS,
                cv2.LINE_AA
            )
            if speed_limit is not None:
                cv2.putText(
                    rendered_img,
                    f"Limit:{speed_limit}",
                    config.PREVIEW_TEXT_POS_LIMIT,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    config.PREVIEW_TEXT_FONT_SCALE,
                    config.PREVIEW_TEXT_LIMIT_COLOR,
                    config.PREVIEW_TEXT_THICKNESS,
                    cv2.LINE_AA
                )
            if traffic_light_state:
                light_color = config.PREVIEW_LIGHT_RED_COLOR
                if traffic_light_state == "yellow":
                    light_color = config.PREVIEW_LIGHT_YELLOW_COLOR
                elif traffic_light_state == "green":
                    light_color = config.PREVIEW_LIGHT_GREEN_COLOR
                cv2.putText(
                    rendered_img,
                    f"Light:{traffic_light_state}",
                    config.PREVIEW_TEXT_POS_LIGHT,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    config.PREVIEW_TEXT_FONT_SCALE,
                    light_color,
                    config.PREVIEW_TEXT_THICKNESS,
                    cv2.LINE_AA
                )
            stop_text = ""
            if traffic_stop_active:
                stop_text = "STOP_BY_LIGHT"
            elif person_stop_active:
                stop_text = "STOP_BY_PERSON"
            if stop_text:
                cv2.putText(
                    rendered_img,
                    stop_text,
                    config.PREVIEW_TEXT_POS_STOP,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    config.PREVIEW_TEXT_FONT_SCALE,
                    config.PREVIEW_TEXT_STOP_COLOR,
                    config.PREVIEW_TEXT_THICKNESS,
                    cv2.LINE_AA
                )

            cv2.rectangle(
                rendered_img,
                config.PREVIEW_YOLO_PANEL_TOP_LEFT,
                config.PREVIEW_YOLO_PANEL_BOTTOM_RIGHT,
                config.PREVIEW_PANEL_BG_COLOR,
                -1,
            )
            cv2.putText(
                rendered_img,
                summarize_yolo_boxes(current_yolo_boxes),
                config.PREVIEW_TEXT_POS_YOLO_SUMMARY,
                cv2.FONT_HERSHEY_SIMPLEX,
                config.PREVIEW_TEXT_FONT_SCALE,
                config.PREVIEW_TEXT_ACCENT_COLOR,
                config.PREVIEW_TEXT_THICKNESS,
                cv2.LINE_AA
            )

        with frame_lock:
            global_preview_frame = rendered_img

    if not bool(getattr(config, "SEG_PIPELINE_ENABLED", True)):
        fps_start_holder = [fps_start_time]
        while True:
            seg_item = seg_queue.get()
            if seg_item is None:
                break
            blob_rgb_320, preview_frame = unpack_seg_item(seg_item)

            with data_lock:
                current_yolo_boxes = [obj.copy() for obj in global_yolo_boxes]
                current_yolo_frame_id = int(global_yolo_frame_id)
                turn_intent = global_control_data.get("turn_intent", -1)

            try:
                steer_signal, rendered_img = seg.run(
                    blob_rgb_320,
                    current_yolo_boxes,
                    turn_intent,
                    fps_stats
                )
                publish_seg_result(
                    steer_signal,
                    rendered_img,
                    current_yolo_boxes,
                    fps_start_holder,
                    current_yolo_frame_id=current_yolo_frame_id,
                    preview_frame=preview_frame,
                )
            except Exception as e:
                throttled_log(
                    f"seg_serial_error_{core_id}",
                    f"Seg串行处理异常(Core {core_id}): {e}",
                    state=str(e),
                )
        return

    mask_queue = Queue(maxsize=max(1, int(getattr(config, "SEG_PIPELINE_QUEUE_MAXSIZE", 1))))
    fps_start_holder = [fps_start_time]

    def postprocess_loop():
        while True:
            t_get_start = time.perf_counter()
            item = mask_queue.get()
            t_after_get = time.perf_counter()
            if item is None:
                break

            blob_rgb_320, preview_frame, mask, infer_s, total_start, current_yolo_boxes, current_yolo_frame_id, turn_intent = item

            try:
                t_post_start = time.perf_counter()
                steer_signal, rendered_img = seg.postprocess_mask(
                    blob_rgb_320,
                    mask,
                    current_yolo_boxes,
                    turn_intent,
                    fps_stats,
                    infer_s=infer_s,
                    total_start=total_start,
                    preview_frame=preview_frame,
                )
                t_post_end = time.perf_counter()
                publish_seg_result(
                    steer_signal,
                    rendered_img,
                    current_yolo_boxes,
                    fps_start_holder,
                    current_yolo_frame_id=current_yolo_frame_id,
                    preview_frame=preview_frame,
                )
                t_publish_end = time.perf_counter()
                profile_log(
                    "seg_post_loop",
                    "SegPostLoop",
                    {
                        "wait_mask": t_after_get - t_get_start,
                        "post": t_post_end - t_post_start,
                        "publish": t_publish_end - t_post_end,
                        "loop": t_publish_end - t_get_start,
                    },
                )
            except Exception as e:
                throttled_log(
                    f"seg_postprocess_error_{core_id}",
                    f"Seg后处理异常(Core {core_id}): {e}",
                    state=str(e),
                )

    threading.Thread(target=postprocess_loop, daemon=True).start()

    while True:
        t_get_start = time.perf_counter()
        seg_item = seg_queue.get()
        t_after_get = time.perf_counter()
        if seg_item is None:
            mask_queue.put(None)
            break
        blob_rgb_320, preview_frame = unpack_seg_item(seg_item)

        total_start = time.perf_counter()
        t_lock_start = time.perf_counter()
        with data_lock:
            current_yolo_boxes = [obj.copy() for obj in global_yolo_boxes]
            current_yolo_frame_id = int(global_yolo_frame_id)
            turn_intent = global_control_data.get("turn_intent", -1)
        t_lock_end = time.perf_counter()

        try:
            t_infer_start = time.perf_counter()
            mask, infer_s = seg.infer_mask(blob_rgb_320)
            t_infer_end = time.perf_counter()
        except Exception as e:
            throttled_log(
                f"seg_infer_error_{core_id}",
                f"Seg推理异常(Core {core_id}): {e}",
                state=str(e),
            )
            continue

        dropped_post = 0
        if mask_queue.full():
            try:
                mask_queue.get_nowait()
                dropped_post = 1
            except:
                pass
        t_put_start = time.perf_counter()
        mask_queue.put((blob_rgb_320, preview_frame, mask, infer_s, total_start, current_yolo_boxes, current_yolo_frame_id, turn_intent))
        t_put_end = time.perf_counter()
        profile_log(
            "seg_infer_loop",
            "SegInferLoop",
            {
                "wait_input": t_after_get - t_get_start,
                "lock": t_lock_end - t_lock_start,
                "infer": t_infer_end - t_infer_start,
                "put_mask": t_put_end - t_put_start,
                "loop": t_put_end - t_get_start,
                "drop_post_fps": float(dropped_post),
            },
        )

# ==============================================================================
# 基础支撑线程：串口控制
# ==============================================================================
def serial_control_thread():
    """根据视觉状态持续输出底层控制命令.

    当前速度控制是三层叠加关系:
    1. 先根据 steer_signal 幅度得到 dynamic_target_speed
    2. 如果 speed_limit 已生效，再把它作为速度上限
    3. 如果红/黄灯且停止线已接近，或行人停车状态激活，则强制把速度打到 0
    """
    try:
        ser = serial.Serial(config.SERIAL_PORT, config.BAUD_RATE, timeout=config.SERIAL_TIMEOUT)
        log_once(
            "serial_open_ok",
            f"串口已打开: {config.SERIAL_PORT} @ {config.BAUD_RATE}",
        )
    except Exception as e:
        ser = None
        log_once(
            "serial_open_error",
            f"串口打开失败: {config.SERIAL_PORT} @ {config.BAUD_RATE}, 错误: {e}",
        )
    serial_first_packet_logged = False
    last_output_speed = int(config.CONTROL_MIN_SPEED)

    while True:
        with data_lock:
            steer_signal = global_control_data.get("steer_signal", 0.0)
            speed_limit = (
                global_control_data.get("speed_limit")
                if bool(getattr(config, "LIMIT_SIGN_ENABLED", True))
                else None
            )
            zebra_stopline_y = global_control_data.get("zebra_stopline_y")
            traffic_light_state = str(global_control_data.get("traffic_light_state", ""))
            traffic_stop_active = bool(global_control_data.get("traffic_stop_active", False))
            person_stop_active = bool(global_control_data.get("person_stop_active", False))
            person_dist_to_bottom = global_control_data.get("person_dist_to_bottom")
            person_left_boundary_x = global_control_data.get("person_left_boundary_x")
            person_right_boundary_x = global_control_data.get("person_right_boundary_x")
            person_clear_line_x = global_control_data.get("person_clear_line_x")
            person_bottom_center_x = global_control_data.get("person_bottom_center_x")
            person_bottom_right_x = global_control_data.get("person_bottom_right_x")
            person_clear_frames = int(global_control_data.get("person_clear_frames", 0))
            person_stop_event = str(global_control_data.get("person_stop_event", ""))

        try:
            if np.isnan(steer_signal) or np.isinf(steer_signal):
                steer_signal = 0.0

            # 转向量幅度越大，目标速度越低，避免高速出弯失控。
            dynamic_target_speed = config.CONTROL_MAX_SPEED - int(
                abs(steer_signal) * config.STEER_SIGNAL_SPEED_GAIN
            )
            dynamic_target_speed = int(
                max(
                    config.CONTROL_MIN_SPEED,
                    min(config.CONTROL_MAX_SPEED, dynamic_target_speed),
                )
            )
            target_speed = dynamic_target_speed
            if speed_limit is not None:
                target_speed = min(target_speed, int(speed_limit))

            stop_trigger_dist = int(config.ZEBRA_STOPLINE_TRIGGER_DIST)
            stopline_dist_to_bottom = None
            if zebra_stopline_y is not None:
                stopline_dist_to_bottom = config.TARGET_RES[1] - int(zebra_stopline_y)
            stop_ready = (
                zebra_stopline_y is not None and
                stopline_dist_to_bottom <= stop_trigger_dist
            )

            if traffic_light_state in ("red", "yellow", "green"):
                zebra_dist_text = "无" if stopline_dist_to_bottom is None else str(stopline_dist_to_bottom)
                throttled_log(
                    "traffic_light_state_detail",
                    "红绿灯状态: "
                    f"灯色={traffic_light_state} "
                    f"停车线到底部距离={zebra_dist_text} "
                    f"阈值={stop_trigger_dist} "
                    f"stop_ready={'是' if stop_ready else '否'} "
                    f"已停车={'是' if traffic_stop_active else '否'}",
                    state=(traffic_light_state, zebra_dist_text, int(stop_trigger_dist), bool(stop_ready), bool(traffic_stop_active)),
                    min_interval=config.LOG_INTERVAL_TRAFFIC_STOP_DETAIL
                )

            if traffic_light_state == "green":
                traffic_stop_active = False
            elif traffic_light_state in ("red", "yellow") and stop_ready:
                traffic_stop_active = True

            if traffic_stop_active:
                target_speed = 0
            if person_stop_active:
                target_speed = 0

            limit_applied = speed_limit is not None and dynamic_target_speed > int(speed_limit)
            if bool(getattr(config, "CONTROL_SPEED_SMOOTH_ENABLED", True)):
                if target_speed <= 0:
                    last_output_speed = 0
                else:
                    if last_output_speed <= 0:
                        last_output_speed = min(int(target_speed), int(config.CONTROL_MIN_SPEED))
                    elif int(target_speed) > last_output_speed:
                        step_up = max(1, int(getattr(config, "CONTROL_SPEED_MAX_STEP_UP", 1)))
                        last_output_speed = min(int(target_speed), last_output_speed + step_up)
                    elif int(target_speed) < last_output_speed:
                        step_down = max(1, int(getattr(config, "CONTROL_SPEED_MAX_STEP_DOWN", 4)))
                        last_output_speed = max(int(target_speed), last_output_speed - step_down)
                    else:
                        last_output_speed = int(target_speed)
                target_speed = int(last_output_speed)

            raw_pwm = (
                config.SERVO_CENTER
                - steer_signal * config.STEER_SIGNAL_PWM_GAIN
            )
            servo_pwm = int(max(config.SERVO_MIN, min(config.SERVO_MAX, raw_pwm)))
        except:
            target_speed = config.CONTROL_MIN_SPEED
            servo_pwm = config.SERVO_CENTER
            dynamic_target_speed = target_speed
            stop_ready = False
            limit_applied = False
            steer_signal = 0.0
            stopline_dist_to_bottom = None
            person_dist_to_bottom = None
            person_left_boundary_x = None
            person_right_boundary_x = None
            person_clear_line_x = None
            person_bottom_center_x = None
            person_bottom_right_x = None
            person_clear_frames = 0
            person_stop_event = ""
            if traffic_stop_active or person_stop_active:
                target_speed = 0

        with data_lock:
            global_control_data["traffic_stop_active"] = traffic_stop_active
            global_control_data["person_stop_active"] = person_stop_active
            global_control_data["actual_servo_pwm"] = servo_pwm
            global_control_data["target_speed"] = target_speed
            if person_stop_event:
                global_control_data["person_stop_event"] = ""

        if traffic_stop_active:
            light_text = traffic_light_state or "无"
            zebra_dist_text = "无" if stopline_dist_to_bottom is None else str(stopline_dist_to_bottom)
            throttled_log(
                "traffic_stop_detail",
                "红绿灯停车条件触发: "
                f"灯色={light_text} 停车线到底部距离={zebra_dist_text} "
                f"阈值={stop_trigger_dist} 已停车=是",
                state=(light_text, zebra_dist_text, int(stop_trigger_dist)),
                min_interval=config.LOG_INTERVAL_TRAFFIC_STOP_DETAIL
            )

        person_dist_text = "无" if person_dist_to_bottom is None else f"{float(person_dist_to_bottom):.1f}"
        left_boundary_text = "无" if person_left_boundary_x is None else f"{float(person_left_boundary_x):.1f}"
        right_boundary_text = "无" if person_right_boundary_x is None else f"{float(person_right_boundary_x):.1f}"
        clear_line_text = "无" if person_clear_line_x is None else f"{float(person_clear_line_x):.1f}"
        center_x_text = "无" if person_bottom_center_x is None else f"{float(person_bottom_center_x):.1f}"
        right_x_text = "无" if person_bottom_right_x is None else f"{float(person_bottom_right_x):.1f}"
        stop_text = "是" if person_stop_active else "否"

        if person_stop_event == "stop":
            throttled_log(
                "person_stop_event",
                ">>> 行人: 停",
                state=("stop",),
                min_interval=0.0,
            )
        elif person_stop_event == "release_clear_line":
            throttled_log(
                "person_stop_event",
                ">>> 行人: 走",
                state=("release_clear_line",),
                min_interval=0.0,
            )
        elif person_stop_event == "release_missing":
            throttled_log(
                "person_stop_event",
                ">>> 行人: 走",
                state=("release_missing",),
                min_interval=0.0,
            )

        if ser:
            packet = struct.pack(
                '<BBhhBB',
                config.SERIAL_PACKET_HEADER[0],
                config.SERIAL_PACKET_HEADER[1],
                target_speed,
                servo_pwm,
                config.SERIAL_PACKET_TAIL[0],
                config.SERIAL_PACKET_TAIL[1],
            )
            try:
                written = ser.write(packet)
                if not serial_first_packet_logged:
                    log_once(
                        "serial_first_packet",
                        f"串口首包已发送: bytes={packet.hex(' ')} written={written}/{len(packet)}",
                    )
                    serial_first_packet_logged = True
                if written != len(packet):
                    throttled_log(
                        "serial_write_incomplete",
                        f"串口发送不完整: {written}/{len(packet)}",
                        state=(int(written), len(packet)),
                        min_interval=config.LOG_INTERVAL_SERIAL_ERROR,
                    )
            except Exception as e:
                throttled_log(
                    "serial_write_error",
                    f"串口发送失败: {e}",
                    state=str(e),
                    min_interval=config.LOG_INTERVAL_SERIAL_ERROR,
                )

        time.sleep(config.CONTROL_LOOP_SLEEP)


# ==============================================================================
# 基础支撑线程：共享内存拉流
# ==============================================================================
def ai_producer_thread():
    """从共享内存读取最新帧，并分发给 Seg / YOLO 两条支路.

    注意:
    - 这里的设计是“永远只保留最新帧”，所以两个队列都是小容量；
    - 如果下游处理不过来，会主动丢弃旧帧，优先保证实时性。
    """
    while True:
        try:
            shm = shared_memory.SharedMemory(name=config.SHM_NAME)
            remove_shm_from_resource_tracker()
            last_fid = 0

            while True:
                t_frame_start = time.perf_counter()
                header = bytes(shm.buf[:config.SHM_HEADER_SIZE])
                fid, w, h = struct.unpack('QII', header)
                if fid == last_fid:
                    time.sleep(config.SHM_FRAME_POLL_SLEEP)
                    continue
                last_fid = fid

                img_view = np.ndarray(
                    (h, w, 3),
                    dtype=np.uint8,
                    buffer=shm.buf[config.SHM_HEADER_SIZE: config.SHM_HEADER_SIZE + w * h * 3]
                )

                t_copy_start = time.perf_counter()
                frame_rgb = cv2.flip(img_view.copy(), 0)
                t_copy_end = time.perf_counter()
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                vis_img_large = cv2.resize(frame_bgr, config.TARGET_RES, interpolation=cv2.INTER_LINEAR)
                t_vis_end = time.perf_counter()

                # 分割分支按当前模型约定裁剪/缩放 RGB 小图，尽可能减轻主控制链路负担。
                seg_blob = make_seg_input(frame_rgb)
                t_seg_end = time.perf_counter()
                dropped_seg = 0
                if seg_queue.full():
                    try:
                        seg_queue.get_nowait()
                        dropped_seg = 1
                    except:
                        pass
                seg_queue.put((seg_blob, vis_img_large))
                t_seg_put_end = time.perf_counter()

                # 检测分支直接生成 YOLO 输入尺寸的小图，避免大图先放大再缩小。
                # 同时保留一份 TARGET_RES 大图，供:
                # - 检测框绘制
                # - OCR 整图 det + rec
                # - OCR 结果回写后的页面可视化
                # 注意: 检测框最终也会映射到 TARGET_RES 坐标系，所以这里要保持一致。
                det_img = cv2.resize(frame_bgr, config.YOLO_SIZE, interpolation=cv2.INTER_LINEAR)
                t_yolo_end = time.perf_counter()

                dropped_yolo = 0
                if yolo_queue.full():
                    try:
                        yolo_queue.get_nowait()
                        dropped_yolo = 1
                    except:
                        pass
                yolo_queue.put((det_img, vis_img_large, (w, h), int(fid)))
                t_frame_end = time.perf_counter()
                profile_log(
                    "producer_loop",
                    "ProducerProfile",
                    {
                        "copy_flip": t_copy_end - t_copy_start,
                        "vis_resize": t_vis_end - t_copy_end,
                        "seg_prep": t_seg_end - t_vis_end,
                        "seg_put": t_seg_put_end - t_seg_end,
                        "yolo_prep": t_yolo_end - t_seg_put_end,
                        "total": t_frame_end - t_frame_start,
                        "drop_seg_fps": float(dropped_seg),
                        "drop_yolo_fps": float(dropped_yolo),
                    },
                )

        except Exception as e:
            log_once("shm_attach_error", f"共享内存拉流异常: {e}")
            time.sleep(config.SHM_RETRY_SLEEP)


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
                time.sleep(config.VIDEO_FEED_IDLE_SLEEP)
                continue

            t_encode_start = time.perf_counter()
            ret, buffer = cv2.imencode(
                '.jpg',
                current_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY]
            )
            t_encode_end = time.perf_counter()
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            t_yield_end = time.perf_counter()
            profile_log(
                "video_feed_loop",
                "VideoFeedProfile",
                {
                    "jpeg": t_encode_end - t_encode_start,
                    "yield": t_yield_end - t_encode_end,
                    "frame": t_yield_end - t_encode_start,
                },
            )
            time.sleep(config.VIDEO_FEED_FRAME_SLEEP)

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    """按模块顺序启动所有线程和 Flask 服务."""
    install_rknn_warning_filter()
    print_preview_url()

    threading.Thread(target=ai_producer_thread, daemon=True).start()
    threading.Thread(target=serial_control_thread, daemon=True).start()
    time.sleep(config.STARTUP_SHARED_THREAD_SLEEP)

    for core_id in config.SEG_CORES:
        threading.Thread(target=seg_worker, args=(core_id,), daemon=True).start()
        time.sleep(config.STARTUP_SEG_THREAD_SLEEP)

    threading.Thread(target=yolo_worker, daemon=True).start()
    threading.Thread(target=ocr_worker, daemon=True).start()

    app.run(host=config.FLASK_HOST, port=config.STREAM_PORT, threaded=True)

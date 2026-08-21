"""系统主入口.

整体流程:
1. ai_producer_thread 从共享内存读取最新图像。
2. 图像被拆成两条支路:
   - seg_queues: 按赛程阶段送给一个或多个分割 / 路径规划线程
   - yolo_queues: 轮流送给多个目标检测线程
3. yolo_worker 只负责检测；如果检测到 sign，再把 OCR 任务异步送入 ocr_queue。
4. ocr_worker 对整张 TARGET_RES 图执行 OCR，再把识别结果按中心点回匹配到 sign 检测框。
   OCR 后端可在本地 RKNN 和百度云 API 之间切换。
5. seg_worker 读取当前最新的检测结果与 turn_intent，生成控制量和预览图。
6. serial_control_thread 将控制量打包后发给下位机。
7. Flask 读取 global_preview_frame，并调用 debug_tools 编码成 MJPEG 提供网页预览。
"""

import time
import struct
import copy
import logging
import numpy as np
import cv2
import threading
import serial
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from multiprocessing import Process, Queue as MPQueue, shared_memory, resource_tracker
from flask import Flask, Response, render_template_string, request

import config
from modules.debug_tools import (
    DebugLogger,
    draw_preview_status_panel,
    draw_yolo_boxes,
    encode_mjpeg_frame,
    get_debug_drive_keyboard_state,
    get_preview_host,
    preview_index_html,
    set_debug_drive_manual_stop,
    start_debug_drive_keyboard_control,
)
from modules.segmentor import RoadSegmentor
from modules.detector import YOLODetector
from modules.baidu_ocr_api import BaiduOCRRecognizer
from modules.ocr_system import OCRRecognizer as LocalOCRRecognizer
from modules.qianfan_client import request_road_choice
from utils.yolo_rknn_post import letterbox_bgr_or_rgb
try:
    from utils.rknn_quiet import install_rknn_warning_filter, suppress_rknn_init_output
except ImportError:
    def install_rknn_warning_filter():
        return

    from contextlib import nullcontext

    def suppress_rknn_init_output():
        return nullcontext()

app = Flask(__name__)
logging.getLogger("werkzeug").disabled = True
LOG_GREEN = "\033[92m"
LOG_RESET = "\033[0m"

# ==============================================================================
# 全局状态与资源锁
# ==============================================================================
# 当前网页预览的最新画面，只由 seg_worker 写，由 Flask 推流读取。
global_preview_frame = None

# 供分割线程、串口线程、OCR 线程共享的控制状态。
# 这份状态是系统里最核心的一块“跨线程控制面板”：
# - seg_worker 写入 steer_signal
# - ocr_worker 写入 turn_intent
# - serial_control_thread 读取这些状态并生成最终底层控制命令
global_control_data = copy.deepcopy(config.DEFAULT_CONTROL_DATA)

# 用于在页面上显示 Seg / YOLO 处理频率。
fps_stats = copy.deepcopy(config.DEFAULT_FPS_STATS)

frame_lock = threading.Lock()
data_lock = threading.Lock()
debug_logger = DebugLogger()

# 三条工作队列:
# - seg_queues: 每个 Seg worker 一条最新帧分割输入队列
# - yolo_queues: 每个 YOLO worker 一条最新帧检测输入队列
# - ocr_queue: 检测线程生成的 sign OCR 任务
seg_core_ids = list(getattr(config, "SEG_CORES_AFTER_SIGN", getattr(config, "SEG_CORES", [0])))
if not seg_core_ids:
    seg_core_ids = list(getattr(config, "SEG_CORES", [0]))
seg_initial_worker_count = max(1, len(getattr(config, "SEG_CORES", [0])))
seg_queues = [Queue(maxsize=config.SEG_QUEUE_MAXSIZE) for _ in seg_core_ids]
yolo_core_ids = list(getattr(config, "YOLO_CORES", [getattr(config, "YOLO_CORE", 2)]))
if not yolo_core_ids:
    yolo_core_ids = [getattr(config, "YOLO_CORE", 2)]
yolo_queues = [Queue(maxsize=config.YOLO_QUEUE_MAXSIZE) for _ in yolo_core_ids]
ocr_queue = Queue(maxsize=config.OCR_QUEUE_MAXSIZE)
# 本地 OCR 与百度 OCR 分开排队：慢网络请求不能占住本地 OCR 的工作线程。
ocr_local_executor = ThreadPoolExecutor(max_workers=1)
ocr_api_executor = ThreadPoolExecutor(max_workers=1)
# OCR executor 各自单线程，后端实例可安全复用。尤其本地 RKNN 不应每次采样都
# 重载 det/rec 模型，否则会刷静态 shape 提示并浪费数百毫秒初始化时间。
ocr_recognizer_lock = threading.Lock()
ocr_recognizers = {}
llm_queue = MPQueue(maxsize=2)
llm_result_queue = MPQueue(maxsize=2)

# 当前最新一帧的检测结果。
global_yolo_boxes = []
global_yolo_frame_id = -1


def remove_shm_from_resource_tracker():
    """避免 Python 退出时错误回收外部创建的共享内存对象."""
    try:
        resource_tracker.unregister('/' + config.SHM_NAME, 'shared_memory')
    except:
        pass


def throttled_log(key, message, state=None, min_interval=None):
    """按状态变化或最小时间间隔打印日志，避免终端刷屏."""
    debug_logger.throttled_log(key, message, state=state, min_interval=min_interval)


def log_once(key, message):
    """同一类错误只打印一次，避免异常反复刷屏。"""
    debug_logger.log_once(key, message)


def drain_latest_queue_item(input_queue, first_item):
    """从最新帧队列里清到最后一个 item，避免处理排队旧帧。"""
    latest_item = first_item
    dropped = 0
    while True:
        try:
            latest_item = input_queue.get_nowait()
            dropped += 1
            if latest_item is None:
                break
        except Empty:
            break
    return latest_item, dropped


def profile_log(key, label, metrics, min_interval=None):
    """用 EMA 节流打印主流程耗时，数值单位统一按毫秒展示."""
    debug_logger.profile_log(key, label, metrics, min_interval=min_interval)


def print_preview_url():
    """启动时主动打印一条可点击的网页推流地址."""
    host = get_preview_host()
    print(f"AI推流网页: http://{host}:{config.STREAM_PORT}/", flush=True)


def print_runtime_config_summary():
    """启动时打印关键运行路径和 OCR 参数，避免现场跑到旧目录还没发现."""
    print(
        "运行配置: "
        f"main={__file__} "
        f"config={getattr(config, '__file__', 'unknown')} "
        f"OCR_MIN_SIGN_BOX_AREA={config.OCR_MIN_SIGN_BOX_AREA} "
        f"OCR_DET_INPUT_SIZE={config.OCR_DET_INPUT_SIZE} "
        f"OCR_DET_BINARY_THRESH={config.OCR_DET_BINARY_THRESH} "
        f"OCR_DET_MIN_CONTOUR_AREA={config.OCR_DET_MIN_CONTOUR_AREA} "
        f"OCR_MIN_SCORE={config.OCR_MIN_SCORE} "
        f"SIGN_ROUTE_DECISION_MODE={getattr(config, 'SIGN_ROUTE_DECISION_MODE', None)} "
        f"SIGN_ROUTE_SKIP_FIRST_PASS={getattr(config, 'SIGN_ROUTE_SKIP_FIRST_PASS', None)} "
        f"SIGN_ROUTE_FIXED_FIRST_CHOICE={getattr(config, 'SIGN_ROUTE_FIXED_FIRST_CHOICE', None)} "
        f"SIGN_LLM_FORK_POINT_TRIGGER_ROWS={getattr(config, 'SIGN_LLM_FORK_POINT_TRIGGER_ROWS', None)} "
        f"CAR_AVOIDANCE_SWITCH_MIN_BOTTOM_Y={getattr(config, 'CAR_AVOIDANCE_SWITCH_MIN_BOTTOM_Y', None)} "
        f"CAR_AVOIDANCE_SIDE_MIN_ROAD_WIDTH={getattr(config, 'CAR_AVOIDANCE_SIDE_MIN_ROAD_WIDTH', None)} "
        f"CAR_AVOIDANCE_SIDE_ROAD_CENTER_Y={getattr(config, 'CAR_AVOIDANCE_SIDE_ROAD_CENTER_Y', None)} "
        f"CAR_AVOIDANCE_NEAR_BOUNDARY_ROWS={getattr(config, 'CAR_AVOIDANCE_NEAR_BOUNDARY_ROWS', None)}",
        flush=True,
    )


def car_avoidance_updates_enabled(control_state):
    """行人/路牌停车时冻结避车状态；手动未发车仍允许预判左右。"""
    return not (
        bool(control_state.get("person_stop_active", False)) or
        bool(control_state.get("sign_llm_stop_active", False))
    )


def make_seg_input(frame_rgb):
    """按当前分割模型约定生成 RGB 输入图."""
    crop_ratio = float(getattr(config, "SEG_INPUT_CROP_TOP_RATIO", 0.0))
    crop_ratio = max(0.0, min(0.95, crop_ratio))
    if crop_ratio > 0.0:
        h = frame_rgb.shape[0]
        crop_y = int(round(h * crop_ratio))
        frame_rgb = frame_rgb[crop_y:, :, :]

    # argmax 赛道模型的旧训练/部署链路是直接拉伸到 416x160，
    # 不能套 YOLO letterbox，否则会改变赛道在模型坐标中的几何比例。
    if str(getattr(config, "SEG_OUTPUT_ROUTE", "")).lower() in ("argmax", "class_map"):
        return cv2.resize(frame_rgb, config.SEG_SIZE, interpolation=cv2.INTER_LINEAR)

    seg_img, _ = letterbox_bgr_or_rgb(
        frame_rgb,
        config.SEG_SIZE,
        pad_value=int(getattr(config, "SEG_LETTERBOX_PAD_VALUE", 114)),
        scaleup=bool(getattr(config, "SEG_LETTERBOX_SCALEUP", False)),
    )
    return seg_img


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
        dist_to_bottom = float(target_h) - bottom_y
        if bottom_y <= best_bottom_y:
            continue

        x1 = float(np.clip(x, 0.0, float(target_w - 1)))
        x2 = float(np.clip(x + w, 0.0, float(target_w - 1)))
        area = max(0.0, x2 - x1) * max(0.0, float(h))
        ground_distance_m = obj.get("ground_distance_m")
        best_bottom_y = bottom_y
        best = {
            "frame_id": int(frame_id),
            "bottom_y": bottom_y,
            "bottom_left_x": min(x1, x2),
            "bottom_center_x": 0.5 * (x1 + x2),
            "bottom_right_x": max(x1, x2),
            "dist_to_bottom": dist_to_bottom,
            "area": float(area),
            "score": float(obj.get("score", 0.0)),
            "ground_distance_m": None if ground_distance_m is None else float(ground_distance_m),
        }

    return best


def has_car_on_left(boxes, left_boundary_x=None, right_boundary_x=None):
    """判断当前画面左侧是否有 car，保留给行人状态显示和兼容调用."""
    car_cls_id = config.CLASS_NAMES.index("car") if "car" in config.CLASS_NAMES else 0
    target_w = config.TARGET_RES[0]
    if left_boundary_x is not None and right_boundary_x is not None:
        ref_x = 0.5 * (float(left_boundary_x) + float(right_boundary_x))
    else:
        ref_x = 0.5 * float(target_w)
    min_score = float(getattr(config, "CAR_AVOIDANCE_MIN_SCORE", 0.0))

    for obj in boxes:
        if obj.get("class_id", -1) != car_cls_id and obj.get("class_name", "") != "car":
            continue
        if float(obj.get("score", 0.0)) < min_score:
            continue
        rect = obj.get("rect", [0, 0, 0, 0])
        if len(rect) != 4:
            continue

        x, y, w, h = [float(v) for v in rect]
        if w <= 0.0 or h <= 0.0:
            continue

        x1 = float(np.clip(x, 0.0, float(target_w - 1)))
        x2 = float(np.clip(x + w, 0.0, float(target_w - 1)))
        car_center_x = 0.5 * (x1 + x2)
        if car_center_x < ref_x:
            return True

    return False


def update_person_stop_state(state, person_info, left_boundary_x, right_boundary_x, yolo_frame_id, car_on_left=False):
    """行人靠近时先停车；确认其沿当前方向越过放行线后，直接恢复正常巡线."""
    active = bool(state.get("person_stop_active", False))
    prev_active = active
    yolo_frame_id = int(yolo_frame_id)
    if not bool(getattr(config, "PERSON_STOP_ENABLED", True)):
        state["person_stop_active"] = False
        state["person_bottom_y"] = None
        state["person_bottom_left_x"] = None
        state["person_bottom_center_x"] = None
        state["person_bottom_right_x"] = None
        state["person_bottom_area"] = None
        state["person_dist_to_bottom"] = None
        state["person_ground_distance_m"] = None
        state["person_stop_trigger_source"] = ""
        state["person_car_on_left"] = False
        state["person_left_boundary_x"] = None
        state["person_right_boundary_x"] = None
        state["person_road_center_x"] = None
        state["person_clear_line_x"] = None
        state["person_clear_line_side"] = ""
        state["person_stop_cutoff_y"] = None
        state["person_miss_frames"] = 0
        state["person_clear_frames"] = 0
        state["person_move_direction"] = 0
        state["person_current_direction"] = 0
        state["person_release_direction"] = 0
        state["person_move_dx"] = 0.0
        state["person_abs_move_dx"] = 0.0
        state["person_min_move_dx"] = float(getattr(
            config,
            "PERSON_CLEAR_MIN_MOVE_DX",
            getattr(config, "PERSON_CLEAR_MIN_RIGHT_DX", 3.0),
        ))
        state["person_line_reached"] = False
        state["person_movement_confirmed"] = False
        state["person_near_bottom"] = False
        state["person_enough_area"] = False
        state["person_missing_started_at"] = None
        state["person_stop_started_at"] = None
        state["person_stop_max_released"] = False
        state["person_lock_bottom_y"] = None
        state["person_last_frame_id"] = yolo_frame_id
        state["person_last_bottom_center_x"] = None
        state["person_stop_event"] = "disabled_release" if prev_active else ""
        return False
    last_frame_id = int(state.get("person_last_frame_id", -1))
    if yolo_frame_id == last_frame_id:
        if person_info is not None:
            state["person_left_boundary_x"] = left_boundary_x
            state["person_right_boundary_x"] = right_boundary_x
        return active

    clear_frames = int(state.get("person_clear_frames", 0))
    miss_frames = int(state.get("person_miss_frames", 0))
    move_direction = int(state.get("person_move_direction", 0))
    missing_started_at = state.get("person_missing_started_at")
    stop_started_at = state.get("person_stop_started_at")
    max_released = bool(state.get("person_stop_max_released", False))
    lock_bottom_y = state.get("person_lock_bottom_y")
    now = time.monotonic()
    released_by_line = False
    released_by_missing = False

    if person_info is not None and active:
        bottom_y = float(person_info.get("bottom_y", 0.0))
        row_margin = max(0.0, float(getattr(config, "PERSON_STOP_LOCK_ROW_MARGIN", 45.0)))
        if lock_bottom_y is not None and bottom_y < float(lock_bottom_y) - row_margin:
            person_info = None

    if person_info is None:
        miss_frames += 1
        if active:
            if stop_started_at is None:
                stop_started_at = now
            if missing_started_at is None:
                missing_started_at = now
            missing_timeout = max(0.0, float(getattr(config, "PERSON_STOP_MISSING_TIMEOUT_SECONDS", 2.0)))
            if missing_timeout > 0.0 and now - float(missing_started_at) >= missing_timeout:
                active = False
                clear_frames = 0
                move_direction = 0
                missing_started_at = None
                stop_started_at = None
                max_released = True
                released_by_missing = True
        state["person_stop_active"] = active
        state["person_bottom_y"] = None
        state["person_bottom_left_x"] = None
        state["person_bottom_center_x"] = None
        state["person_bottom_right_x"] = None
        state["person_bottom_area"] = None
        state["person_dist_to_bottom"] = None
        state["person_ground_distance_m"] = None
        state["person_stop_trigger_source"] = ""
        state["person_car_on_left"] = False
        state["person_left_boundary_x"] = None
        state["person_right_boundary_x"] = None
        state["person_road_center_x"] = None
        state["person_clear_line_x"] = None
        state["person_clear_line_side"] = ""
        state["person_stop_cutoff_y"] = None
        state["person_miss_frames"] = miss_frames
        state["person_clear_frames"] = clear_frames
        state["person_move_direction"] = move_direction if active else 0
        state["person_current_direction"] = 0
        state["person_release_direction"] = move_direction if active else 0
        state["person_move_dx"] = 0.0
        state["person_abs_move_dx"] = 0.0
        state["person_min_move_dx"] = float(getattr(
            config,
            "PERSON_CLEAR_MIN_MOVE_DX",
            getattr(config, "PERSON_CLEAR_MIN_RIGHT_DX", 3.0),
        ))
        state["person_line_reached"] = False
        state["person_movement_confirmed"] = False
        state["person_near_bottom"] = False
        state["person_enough_area"] = False
        state["person_missing_started_at"] = missing_started_at if active else None
        state["person_stop_started_at"] = stop_started_at if active else None
        state["person_stop_max_released"] = max_released
        state["person_lock_bottom_y"] = lock_bottom_y if active else None
        state["person_last_frame_id"] = yolo_frame_id
        if not active:
            state["person_last_bottom_center_x"] = None
        if prev_active and not active and released_by_missing:
            state["person_stop_event"] = "release_missing"
        elif prev_active and not active:
            state["person_stop_event"] = "release_timeout"
        else:
            state["person_stop_event"] = ""
        return active

    bottom_left_x = float(person_info.get("bottom_left_x", person_info["bottom_center_x"]))
    bottom_center_x = float(person_info["bottom_center_x"])
    bottom_right_x = float(person_info["bottom_right_x"])
    bottom_y = float(person_info["bottom_y"])
    dist_to_bottom = float(person_info["dist_to_bottom"])
    ground_distance_m = person_info.get("ground_distance_m")
    ground_distance_m = None if ground_distance_m is None else float(ground_distance_m)
    person_area = float(person_info.get("area", 0.0))
    last_center = state.get("person_last_bottom_center_x")
    min_move_dx = float(getattr(
        config,
        "PERSON_CLEAR_MIN_MOVE_DX",
        getattr(config, "PERSON_CLEAR_MIN_RIGHT_DX", 3.0),
    ))
    dx = 0.0 if last_center is None else bottom_center_x - float(last_center)
    abs_dx = abs(float(dx))
    current_direction = 0
    if dx >= min_move_dx:
        current_direction = 1
    elif dx <= -min_move_dx:
        current_direction = -1

    image_center_x = float(config.TARGET_RES[0]) / 2.0
    road_center_x = image_center_x
    if left_boundary_x is not None and right_boundary_x is not None:
        road_center_x = 0.5 * (float(left_boundary_x) + float(right_boundary_x))
    target_h = float(config.TARGET_RES[1])
    trigger_dist = float(config.PERSON_STOP_TRIGGER_DIST)
    stop_cutoff_y = max(0.0, min(target_h - 1.0, target_h - trigger_dist))
    min_area = float(getattr(config, "PERSON_STOP_MIN_AREA", 0.0))
    pixel_trigger_enabled = bool(getattr(config, "PERSON_STOP_PIXEL_TRIGGER_ENABLED", True))
    distance_trigger_enabled = bool(getattr(config, "PERSON_STOP_DISTANCE_ENABLED", True))
    distance_trigger_m = float(getattr(config, "PERSON_STOP_DISTANCE_M", 0.0))
    near_bottom_by_pixel = bool(pixel_trigger_enabled and dist_to_bottom <= trigger_dist)
    near_bottom_by_distance = bool(
        distance_trigger_enabled and
        distance_trigger_m > 0.0 and
        ground_distance_m is not None and
        ground_distance_m <= distance_trigger_m
    )
    near_bottom = bool(near_bottom_by_pixel or near_bottom_by_distance)
    enough_area = person_area >= min_area if near_bottom else False
    if near_bottom_by_distance and near_bottom_by_pixel:
        stop_trigger_source = "distance+pixel"
    elif near_bottom_by_distance:
        stop_trigger_source = "distance"
    elif near_bottom_by_pixel:
        stop_trigger_source = "pixel"
    else:
        stop_trigger_source = ""

    missing_started_at = None
    if not near_bottom:
        max_released = False

    release_direction = current_direction if current_direction != 0 else move_direction
    clear_line_x = None
    clear_line_side = ""
    if release_direction > 0:
        clear_line_side = "right"
        if right_boundary_x is not None:
            clear_line_x = float(right_boundary_x)
    elif release_direction < 0:
        clear_line_side = "left"
        if left_boundary_x is not None:
            clear_line_x = float(left_boundary_x)

    line_reached = False
    if clear_line_x is not None:
        if release_direction > 0:
            line_reached = bottom_left_x >= clear_line_x
        elif release_direction < 0:
            line_reached = bottom_right_x <= clear_line_x

    if near_bottom:
        if enough_area and (not max_released or not line_reached):
            if not active:
                stop_started_at = now
            active = True
            if not line_reached:
                max_released = False
    elif not active:
        stop_started_at = None
    miss_frames = 0

    movement_confirmed = (
        active and
        current_direction != 0 and
        release_direction != 0 and
        current_direction == release_direction
    )
    if movement_confirmed and line_reached:
        clear_frames += 1
    else:
        clear_frames = 0
        if current_direction != 0:
            move_direction = current_direction
        elif not active:
            move_direction = 0

    if active and clear_frames >= int(config.PERSON_CLEAR_MOVE_FRAMES):
        active = False
        clear_frames = 0
        stop_started_at = None
        max_released = True
        released_by_line = True

    if active:
        if stop_started_at is None:
            stop_started_at = now
        lock_bottom_y = bottom_y
    elif not near_bottom:
        stop_started_at = None
        lock_bottom_y = None
    else:
        lock_bottom_y = None

    state["person_stop_active"] = active
    state["person_bottom_y"] = bottom_y
    state["person_bottom_left_x"] = bottom_left_x
    state["person_bottom_center_x"] = bottom_center_x
    state["person_bottom_right_x"] = bottom_right_x
    state["person_bottom_area"] = person_area
    state["person_dist_to_bottom"] = dist_to_bottom
    state["person_ground_distance_m"] = ground_distance_m
    state["person_stop_trigger_source"] = stop_trigger_source
    state["person_car_on_left"] = bool(car_on_left)
    state["person_left_boundary_x"] = left_boundary_x
    state["person_right_boundary_x"] = right_boundary_x
    state["person_road_center_x"] = road_center_x
    state["person_clear_line_x"] = clear_line_x
    state["person_clear_line_side"] = clear_line_side
    state["person_stop_cutoff_y"] = stop_cutoff_y
    state["person_clear_frames"] = clear_frames
    state["person_miss_frames"] = miss_frames
    state["person_move_direction"] = move_direction
    state["person_current_direction"] = current_direction
    state["person_release_direction"] = release_direction
    state["person_move_dx"] = float(dx)
    state["person_abs_move_dx"] = float(abs_dx)
    state["person_min_move_dx"] = float(min_move_dx)
    state["person_line_reached"] = bool(line_reached)
    state["person_movement_confirmed"] = bool(movement_confirmed)
    state["person_near_bottom"] = bool(near_bottom)
    state["person_enough_area"] = bool(enough_area)
    state["person_missing_started_at"] = missing_started_at
    state["person_stop_started_at"] = stop_started_at
    state["person_stop_max_released"] = max_released
    state["person_lock_bottom_y"] = lock_bottom_y
    state["person_last_frame_id"] = yolo_frame_id
    state["person_last_bottom_center_x"] = bottom_center_x
    if not prev_active and active:
        state["person_stop_event"] = "stop"
    elif prev_active and not active and released_by_line:
        state["person_stop_event"] = "release_line"
    elif prev_active and not active:
        state["person_stop_event"] = "release_timeout"
    else:
        state["person_stop_event"] = ""
    return active


def should_enqueue_ocr_job(cls_id, rect):
    """判断某个 sign 是否值得触发一次整图 OCR.

    当前不是“直接裁检测框做 OCR”，而是:
    1. YOLO 先给出路牌框
    2. 只有框足够大、且没有明显贴边截断风险时，才触发一次整图 OCR
    3. OCR det + rec 的结果再回匹配到这个框

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
    if cls_id != config.SIGN_CLASS_ID:
        return False, "non_ocr_class"
    if not sign_route_uses_llm():
        return False, "sign_route_no_llm"

    min_area = float(config.OCR_MIN_SIGN_BOX_AREA)
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


def should_trigger_sign_route(cls_id, rect):
    """判断是否启动一次完整的语义路牌路线状态机."""
    if cls_id != config.SIGN_CLASS_ID:
        return False, "non_sign"
    if len(rect) != 4:
        return False, "invalid_rect"
    x, y, w, h = rect
    if w < 2 or h < 2:
        return False, "invalid_rect"
    if not sign_rect_edge_safe(rect):
        return False, "too_close_to_edge"

    sign_distance_m = estimate_sign_distance_m(rect)
    trigger_distance_m = float(getattr(config, "SIGN_LLM_TRIGGER_DISTANCE_M", 0.55))
    if sign_distance_m is None:
        return False, "distance_unavailable"
    if sign_distance_m > trigger_distance_m:
        return False, f"distance_too_far({sign_distance_m:.2f}>{trigger_distance_m:.2f})"

    # 旧条件暂不参与触发:
    # area = rect_area(rect)
    # dist_to_bottom = max(0.0, float(config.TARGET_RES[1]) - float(y + h))
    return True, "ok"


def sign_rect_edge_safe(rect):
    """路牌框是否满足不贴边条件."""
    if len(rect) != 4:
        return False
    x, y, w, h = rect
    if w < 2 or h < 2:
        return False
    frame_w, frame_h = config.TARGET_RES
    edge_margin_ratio = max(
        float(config.OCR_SIGN_EDGE_MARGIN_RATIO),
        float(getattr(config, "SIGN_LLM_TRIGGER_EDGE_MARGIN_RATIO", config.OCR_SIGN_EDGE_MARGIN_RATIO)),
    )
    edge_margin_x = frame_w * edge_margin_ratio
    edge_margin_y = frame_h * edge_margin_ratio
    if (
        x <= edge_margin_x or
        y <= edge_margin_y or
        (x + w) >= (frame_w - edge_margin_x) or
        (y + h) >= (frame_h - edge_margin_y)
    ):
        return False
    return True


def estimate_sign_distance_m(rect):
    """用固定真实高度和检测框高度估算 sign 到相机的距离，单位米."""
    if not bool(getattr(config, "SIGN_MONOCULAR_DISTANCE_ENABLED", False)):
        return None
    if rect is None or len(rect) != 4:
        return None
    try:
        _x, _y, _w, h = [float(v) for v in rect]
    except (TypeError, ValueError):
        return None
    min_h = max(1.0, float(getattr(config, "SIGN_DISTANCE_MIN_BOX_HEIGHT_PX", 8.0)))
    if h < min_h:
        return None
    try:
        source_h = float(getattr(config, "IPM_SOURCE_SIZE", (640, 480))[1])
        target_h = float(config.TARGET_RES[1])
        fy_source = float(config.IPM_CAMERA_K[1][1])
        fy_target = fy_source * (target_h / max(source_h, 1.0))
        real_h = float(getattr(config, "SIGN_REAL_HEIGHT_M", 0.0))
        if real_h <= 0.0:
            return None
        return float(fy_target * real_h / h)
    except Exception:
        return None


def estimate_ground_object_distance_m(rect):
    """用检测框底边中点的地面反投估算贴地目标距离，单位米。"""
    if rect is None or len(rect) != 4:
        return None
    try:
        x, y, w, h = [float(v) for v in rect]
    except (TypeError, ValueError):
        return None
    if w <= 0.0 or h <= 0.0:
        return None
    try:
        target_w, target_h = map(float, config.TARGET_RES)
        source_w, source_h = map(float, getattr(config, "IPM_SOURCE_SIZE", (640, 480)))
        fy = float(config.IPM_CAMERA_K[1][1])
        cy = float(config.IPM_CAMERA_K[1][2])
        camera_height = float(getattr(config, "IPM_CAMERA_HEIGHT", 0.14))
        forward_offset = float(getattr(config, "IPM_CAMERA_FORWARD_OFFSET", 0.0))
        _ = (x + w * 0.5) * (source_w / max(target_w, 1.0))
        v = (y + h) * (source_h / max(target_h, 1.0))
        denom = v - cy
        if denom <= 1e-4:
            return None
        return float(fy * camera_height / denom - forward_offset)
    except Exception:
        return None


def sign_route_fork_point_trigger_active(state):
    """Y 岔特征点足够靠近底部时，强制启动路牌采样."""
    if not sign_route_uses_llm():
        return False, None
    if not bool(state.get("sign_route_y_fork_active", False)):
        return False, None
    rows_to_bottom = state.get("sign_route_fork_rows_to_bottom")
    if rows_to_bottom is None:
        return False, None
    try:
        rows_to_bottom = float(rows_to_bottom)
    except (TypeError, ValueError):
        return False, None
    trigger_rows = float(getattr(config, "SIGN_LLM_FORK_POINT_TRIGGER_ROWS", 0.0))
    if trigger_rows <= 0.0:
        return False, rows_to_bottom
    return rows_to_bottom <= trigger_rows, rows_to_bottom


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


def expanded_rect(rect, expand_ratio):
    """按宽高比例向外扩展 [x, y, w, h] 框，并裁剪到 TARGET_RES 内."""
    if len(rect) != 4:
        return [0.0, 0.0, 0.0, 0.0]
    x, y, w, h = [float(v) for v in rect]
    pad_x = max(0.0, w * float(expand_ratio))
    pad_y = max(0.0, h * float(expand_ratio))
    frame_w, frame_h = config.TARGET_RES
    x1 = float(np.clip(x - pad_x, 0.0, float(frame_w)))
    y1 = float(np.clip(y - pad_y, 0.0, float(frame_h)))
    x2 = float(np.clip(x + w + pad_x, 0.0, float(frame_w)))
    y2 = float(np.clip(y + h + pad_y, 0.0, float(frame_h)))
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def point_in_rect(point, rect):
    """判断点是否在 [x, y, w, h] 框内."""
    if len(rect) != 4:
        return False
    px, py = point
    x, y, w, h = [float(v) for v in rect]
    return x <= float(px) <= x + w and y <= float(py) <= y + h


def pack_ocr_matches(matches):
    """把多条 OCR 结果合成一个给显示/LLM 使用的文本和平均置信度."""
    texts = [str(result.get("text", "")).strip().upper() for _, result, _, _ in matches]
    scores = [float(result.get("score", 0.0)) for _, result, _, _ in matches]
    text = "；".join([item for item in texts if item])
    score = float(sum(scores) / max(len(scores), 1))
    detail = [
        f"{str(result.get('text', '')).strip()}:{float(result.get('score', 0.0)):.3f}"
        for _, result, _, _ in matches
    ]
    return text, score, detail


def rect_area(rect):
    """返回 [x, y, w, h] 框面积."""
    if len(rect) != 4:
        return 0.0
    return float(max(0, rect[2]) * max(0, rect[3]))


def rect_center_distance(a, b):
    """返回两个 [x,y,w,h] 框中心点距离."""
    if a is None or b is None or len(a) != 4 or len(b) != 4:
        return float("inf")
    ax, ay = rect_center(a)
    bx, by = rect_center(b)
    return float(np.hypot(float(ax) - float(bx), float(ay) - float(by)))


def best_matching_sign_rect(sign_rects, locked_rect):
    """从当前帧 sign 框里找最接近锁定路牌的那个框."""
    if not sign_rects:
        return None
    if locked_rect is None:
        return list(sign_rects[0])
    return list(min(sign_rects, key=lambda rect: rect_center_distance(rect, locked_rect)))


def update_sign_llm_rect_stability(state, current_rect):
    """停车后延时一小段时间，再允许 OCR."""
    if not (
        sign_route_uses_llm() and
        bool(state.get("sign_llm_stop_active", False)) and
        bool(state.get("sign_llm_collecting", False)) and
        str(state.get("sign_route_state", "IDLE")) == "SIGN_STOP_COLLECT"
    ):
        return False

    delay_s = max(0.0, float(getattr(config, "SIGN_LLM_OCR_START_DELAY_SECONDS", 0.3)))
    started_at = state.get("sign_llm_started_at")
    if (
        bool(state.get("sign_llm_ocr_ready", False)) or
        bool(state.get("sign_llm_ocr_started", False))
    ):
        if current_rect is not None:
            state["sign_llm_stable_rect"] = list(current_rect)
        return True

    if current_rect is not None:
        state["sign_llm_stable_rect"] = list(current_rect)
    ready = started_at is not None and (time.monotonic() - float(started_at)) >= delay_s
    was_ready = bool(state.get("sign_llm_ocr_ready", False))
    state["sign_llm_ocr_ready"] = ready
    if ready and not was_ready:
        state["sign_llm_ocr_started"] = True
        ready_rect = current_rect if current_rect is not None else state.get("sign_route_locked_rect")
        throttled_log(
            "sign_llm_delay_ready",
            f"语义路牌停车后延时到点，开始OCR采样: delay={delay_s:.1f}s rect={ready_rect}",
            state=(
                round(delay_s, 2),
                None if ready_rect is None else tuple(int(float(v)) for v in ready_rect),
            ),
            min_interval=0.0,
        )
    return ready


def clear_sign_llm_state(state, keep_completed=False):
    """清理语义路牌 LLM 状态；可选择保留本次路牌已处理标记."""
    state["sign_llm_stop_active"] = False
    state["sign_llm_collecting"] = False
    state["sign_llm_waiting_result"] = False
    state["sign_llm_samples"] = []
    state["sign_llm_attempts"] = 0
    state["sign_llm_started_at"] = None
    state["sign_llm_frame_id"] = -1
    state["sign_llm_ocr_inflight"] = False
    state["sign_llm_ocr_inflight_started_at"] = None
    state["sign_llm_ocr_active_backends"] = []
    state["sign_llm_ocr_started"] = False
    state["sign_llm_ocr_job_frame_id"] = -1
    state["sign_llm_stable_rect"] = None
    state["sign_llm_ocr_ready"] = False
    state["sign_llm_error"] = ""
    if not keep_completed:
        state["sign_llm_completed_hold"] = False


def reset_sign_route_state(state, next_state="IDLE"):
    """释放一次路牌路线任务，恢复默认分支策略."""
    state["turn_intent"] = -1
    state["turn_intent_fid"] = -1
    state["sign_route_state"] = str(next_state)
    state["sign_route_choice"] = 0
    state["sign_route_locked_rect"] = None
    state["sign_route_drive_started_at"] = None
    state["sign_route_fork_entered_at"] = None
    state["sign_route_single_road_frames"] = 0
    state["sign_route_sign_gone_frames"] = 0
    state["sign_route_api_submitted"] = False
    state["sign_llm_ocr_inflight"] = False
    state["sign_llm_ocr_inflight_started_at"] = None
    state["sign_llm_ocr_active_backends"] = []
    state["sign_llm_ocr_started"] = False
    state["sign_llm_ocr_job_frame_id"] = -1
    state["sign_llm_stable_rect"] = None
    state["sign_llm_ocr_ready"] = False
    state["sign_llm_completed_hold"] = False


def sign_route_label(choice):
    """把路线数值编码转成显示/日志用的 LEFT/RIGHT."""
    return "LEFT" if int(choice) == -1 else "RIGHT"


def sign_route_choice_from_label(label, default=1):
    """把 LEFT/RIGHT 配置值转成内部路线编码."""
    text = str(label or "").strip().upper()
    if text == "LEFT":
        return -1
    if text == "RIGHT":
        return 1
    return int(default)


def sign_route_decision_mode():
    """当前语义岔路决策模式."""
    return str(getattr(config, "SIGN_ROUTE_DECISION_MODE", "llm_once")).strip().lower()


def sign_route_uses_llm():
    """是否使用停车 OCR + 千帆做第二圈岔路牌判定."""
    return sign_route_decision_mode() == "llm_once" and bool(getattr(config, "SIGN_LLM_ENABLED", True))


def sign_route_fixed_mode():
    """是否使用纯固定序列模式."""
    return sign_route_decision_mode() == "fixed_sequence"


def sign_route_skip_first_pass():
    """fixed_sequence 模式下是否忽略第一次分割岔路事件."""
    return bool(getattr(config, "SIGN_ROUTE_SKIP_FIRST_PASS", True))


def fixed_route_visible_sign(current_yolo_boxes):
    """fixed_sequence 模式下，当前 YOLO 框里是否有 sign；不看面积大小."""
    if not fixed_route_requires_sign():
        return True, None
    for obj in current_yolo_boxes or []:
        cls_id = obj.get("class_id", -1)
        cls_name = str(obj.get("class_name", ""))
        if cls_id != config.SIGN_CLASS_ID and cls_name != "sign":
            continue
        rect = obj.get("rect", [0, 0, 0, 0])
        if len(rect) != 4:
            continue
        x, y, w, h = rect
        if float(w) < 2.0 or float(h) < 2.0:
            continue
        return True, [x, y, w, h]
    return False, None


def current_frame_has_visible_sign(current_yolo_boxes):
    """当前帧是否真的看见了可用 sign 框."""
    for obj in current_yolo_boxes or []:
        cls_id = obj.get("class_id", -1)
        cls_name = str(obj.get("class_name", ""))
        if cls_id != config.SIGN_CLASS_ID and cls_name != "sign":
            continue
        rect = obj.get("rect", [0, 0, 0, 0])
        if len(rect) != 4:
            continue
        _x, _y, w, h = rect
        if float(w) >= 2.0 and float(h) >= 2.0:
            return True
    return False


def sign_route_pending_centerline_active(route_state, current_yolo_boxes):
    """路牌岔路尚未决策时，当前看见路牌才临时走分叉中心线."""
    if str(route_state) not in (
        "APPROACH_SIGN",
        "REVERSE_APPROACH_SIGN",
        "SIGN_STOP_COLLECT",
        "WAIT_API",
    ):
        return False
    return current_frame_has_visible_sign(current_yolo_boxes)


def fixed_route_requires_sign():
    """fixed_sequence 是否只在路牌岔路口推进序列."""
    return bool(getattr(config, "SIGN_ROUTE_FIXED_REQUIRE_SIGN", True))


def sign_route_pass_index(state):
    try:
        return max(0, int(state.get("sign_route_pass_index", 0)))
    except (TypeError, ValueError):
        return 0


def activate_sign_route_choice(state, choice, frame_id, locked_rect=None):
    """写入一次有效的语义路牌路线选择."""
    choice = int(choice)
    if choice not in (-1, 1):
        return False
    state["turn_intent"] = choice
    state["turn_intent_fid"] = int(frame_id)
    state["sign_llm_result"] = sign_route_label(choice)
    state["sign_route_state"] = "CHOICE_READY"
    state["sign_route_choice"] = choice
    if locked_rect is not None:
        state["sign_route_locked_rect"] = list(locked_rect)
    state["sign_route_drive_started_at"] = time.monotonic()
    state["sign_route_fork_entered_at"] = None
    state["sign_route_single_road_frames"] = 0
    state["sign_llm_completed_hold"] = False
    state["post_sign_phase"] = True
    return True


def activate_fixed_route_on_fork_if_needed(state, y_fork_active, current_yolo_boxes=None, yolo_frame_id=None):
    """fixed_sequence 模式下，只在当前同时看见路牌和 Y 岔时推进固定路线序列."""
    if not sign_route_fixed_mode():
        return False

    route_state = str(state.get("sign_route_state", "IDLE"))
    if route_state == "IGNORE_FORK":
        if not bool(y_fork_active):
            state["sign_route_state"] = "IDLE"
        return True

    if not bool(y_fork_active) or route_state != "IDLE":
        return False
    sign_visible, sign_rect = fixed_route_visible_sign(current_yolo_boxes)
    if not sign_visible:
        throttled_log(
            "sign_route_fixed_fork_without_sign",
            "固定序列模式忽略普通岔路: 当前未同时看见路牌",
            state=(
                sign_route_pass_index(state),
                int(yolo_frame_id) if yolo_frame_id is not None else int(global_yolo_frame_id),
            ),
            min_interval=1.0,
        )
        return False

    pass_index = sign_route_pass_index(state)
    now_frame_id = int(yolo_frame_id) if yolo_frame_id is not None else int(global_yolo_frame_id)
    state["sign_route_fixed_sign_frame_id"] = now_frame_id
    state["sign_route_fixed_sign_rect"] = None if sign_rect is None else list(sign_rect)
    throttled_log(
        "sign_route_fixed_sign_fork_seen",
        f"固定序列模式检测到路牌+岔路: frame={now_frame_id} rect={sign_rect}",
        state=(
            now_frame_id,
            None if sign_rect is None else tuple(int(float(v)) for v in sign_rect),
        ),
        min_interval=0.5,
    )

    if pass_index <= 0 and sign_route_skip_first_pass() and not fixed_route_requires_sign():
        state["sign_route_pass_index"] = 1
        clear_sign_llm_state(state, keep_completed=True)
        reset_sign_route_state(state, next_state="IGNORE_FORK")
        throttled_log(
            "sign_route_fixed_first_pass_skip",
            "固定序列模式第一圈岔路忽略，不触发路线决策",
            state=("skip_first",),
            min_interval=0.0,
        )
        return True

    if pass_index <= 0:
        state["sign_route_pass_index"] = 1
        pass_index = 1

    if pass_index == 1:
        fixed_choice = sign_route_choice_from_label(
            getattr(config, "SIGN_ROUTE_FIXED_FIRST_CHOICE", "LEFT"),
            default=-1,
        )
        state["sign_route_first_choice"] = fixed_choice
        state["sign_route_pass_index"] = 2
        clear_sign_llm_state(state, keep_completed=True)
        if activate_sign_route_choice(state, fixed_choice, now_frame_id):
            state["sign_route_state"] = "IN_FORK"
            state["sign_route_fork_entered_at"] = time.monotonic()
            throttled_log(
                "sign_route_fixed_choice",
                f"固定序列模式第二圈岔路直接选择: current={sign_route_label(fixed_choice)}",
                state=(fixed_choice, pass_index),
                min_interval=0.0,
            )
        return True

    first_choice = int(state.get("sign_route_first_choice", 0))
    if pass_index == 2 and first_choice in (-1, 1):
        reverse_choice = -first_choice
        state["sign_route_pass_index"] = 3
        clear_sign_llm_state(state, keep_completed=True)
        if activate_sign_route_choice(state, reverse_choice, now_frame_id):
            state["sign_route_state"] = "IN_FORK"
            state["sign_route_fork_entered_at"] = time.monotonic()
            throttled_log(
                "sign_route_fixed_reverse_choice",
                f"固定序列模式第三圈岔路按第二圈取反: "
                f"first={sign_route_label(first_choice)} current={sign_route_label(reverse_choice)}",
                state=(first_choice, reverse_choice),
                min_interval=0.0,
            )
        return True

    reset_sign_route_state(state, next_state="IGNORE_FORK")
    throttled_log(
        "sign_route_fixed_done_ignore",
        "固定序列模式第二/三圈已完成，后续路牌岔路默认走左",
        state=(pass_index, first_choice),
        min_interval=0.0,
    )
    return True


def pending_fixed_route_choice(state, current_yolo_boxes=None):
    """fixed_sequence 模式下，当前看见 sign 时预先传给分割的目标方向."""
    if not sign_route_fixed_mode():
        return 0
    sign_visible, _ = fixed_route_visible_sign(current_yolo_boxes)
    if not sign_visible:
        return 0
    if int(state.get("sign_route_choice", 0)) in (-1, 1):
        return 0
    if str(state.get("sign_route_state", "IDLE")) != "IDLE":
        return 0
    pass_index = sign_route_pass_index(state)
    if pass_index <= 0 and fixed_route_requires_sign():
        return sign_route_choice_from_label(
            getattr(config, "SIGN_ROUTE_FIXED_FIRST_CHOICE", "LEFT"),
            default=-1,
        )
    if pass_index == 1:
        return sign_route_choice_from_label(
            getattr(config, "SIGN_ROUTE_FIXED_FIRST_CHOICE", "LEFT"),
            default=-1,
        )
    first_choice = int(state.get("sign_route_first_choice", 0))
    if pass_index == 2 and first_choice in (-1, 1):
        return -first_choice
    return 0


def reset_sign_route_after_drive(state):
    """一次路线锁定完成后释放；路牌模式需等当前路牌离开画面再接下一次."""
    locked_rect = state.get("sign_route_locked_rect")
    if locked_rect is not None:
        state["sign_route_blocked_rect"] = list(locked_rect)
    next_state = "WAIT_SIGN_GONE" if sign_route_uses_llm() else "IDLE"
    reset_sign_route_state(state, next_state=next_state)


def block_sign_until_gone(state, rect=None):
    """失败/完成后锁住当前路牌，直到它连续离开画面."""
    blocked_rect = rect if rect is not None else state.get("sign_route_locked_rect")
    if blocked_rect is not None and len(blocked_rect) == 4:
        state["sign_route_blocked_rect"] = list(blocked_rect)
    state["sign_route_sign_gone_frames"] = 0
    state["sign_route_state"] = "WAIT_SIGN_GONE"


def update_wait_sign_gone_state(state, route_trigger_rect, visible_sign_rects=None):
    """等待当前路牌稳定离开，避免一帧漏检导致同一块牌重复触发."""
    if str(state.get("sign_route_state", "IDLE")) != "WAIT_SIGN_GONE":
        return str(state.get("sign_route_state", "IDLE"))
    blocked_rect = state.get("sign_route_blocked_rect")
    visible_sign_rects = visible_sign_rects or []
    same_sign_visible = False
    if blocked_rect is not None:
        same_sign_visible = any(
            rect_center_distance(blocked_rect, rect) <= 120.0
            for rect in visible_sign_rects
            if rect is not None and len(rect) == 4
        )
    elif route_trigger_rect is not None:
        same_sign_visible = True

    if same_sign_visible:
        state["sign_route_sign_gone_frames"] = 0
        return "WAIT_SIGN_GONE"

    gone_frames = int(state.get("sign_route_sign_gone_frames", 0)) + 1
    state["sign_route_sign_gone_frames"] = gone_frames
    need_frames = max(1, int(getattr(config, "SIGN_ROUTE_SIGN_GONE_EXIT_FRAMES", 8)))
    if gone_frames >= need_frames:
        state["sign_route_state"] = "IDLE"
        state["sign_route_sign_gone_frames"] = 0
        state["sign_route_blocked_rect"] = None
        return "IDLE"
    return "WAIT_SIGN_GONE"


def update_sign_route_after_seg(state, y_fork_active, current_yolo_boxes=None, yolo_frame_id=None):
    """根据分割岔路状态推进路牌路线生命周期."""
    now = time.monotonic()
    route_state = str(state.get("sign_route_state", "IDLE"))

    if activate_fixed_route_on_fork_if_needed(
        state,
        y_fork_active,
        current_yolo_boxes=current_yolo_boxes,
        yolo_frame_id=yolo_frame_id,
    ):
        route_state = str(state.get("sign_route_state", "IDLE"))

    # 第二次路牌由 YOLO 先置为 REVERSE_APPROACH_SIGN，再由这里确认 Y 岔。
    # 取反必须保留“当前仍检出路牌”这个门槛，普通无路牌岔路不能消耗取反机会。
    if (
        sign_route_uses_llm() and
        bool(y_fork_active) and
        route_state == "REVERSE_APPROACH_SIGN" and
        sign_route_pass_index(state) == 2 and
        int(state.get("sign_route_first_choice", 0)) in (-1, 1)
    ):
        if current_frame_has_visible_sign(current_yolo_boxes):
            first_choice = int(state.get("sign_route_first_choice"))
            reverse_choice = -first_choice
            reverse_rect = state.get("sign_route_locked_rect")
            state["sign_route_pass_index"] = 3
            clear_sign_llm_state(state, keep_completed=True)
            activate_sign_route_choice(
                state,
                reverse_choice,
                int(yolo_frame_id) if yolo_frame_id is not None else int(global_yolo_frame_id),
                locked_rect=reverse_rect,
            )
            state["sign_route_state"] = "IN_FORK"
            state["sign_route_fork_entered_at"] = now
            throttled_log(
                "sign_route_reverse_choice_seg",
                f"分割确认第二次路牌岔路，执行取反: "
                f"first={sign_route_label(first_choice)} current={sign_route_label(reverse_choice)} "
                f"rect={reverse_rect}",
                state=(first_choice, reverse_choice),
                min_interval=0.0,
            )
            return

    if route_state == "CHOICE_READY":
        drive_started_at = state.get("sign_route_drive_started_at")
        if drive_started_at is None:
            return
        max_hold = float(getattr(config, "SIGN_ROUTE_MAX_DRIVE_HOLD_SECONDS", 10.0))
        if now - float(drive_started_at) >= max_hold:
            throttled_log("sign_route_timeout", "语义路牌路线超时释放: 未稳定通过岔路", state="choice_ready", min_interval=0.0)
            reset_sign_route_after_drive(state)
            return
        if y_fork_active:
            state["sign_route_state"] = "IN_FORK"
            state["sign_route_fork_entered_at"] = now
            state["sign_route_single_road_frames"] = 0
            throttled_log("sign_route_in_fork", "语义路牌路线进入岔路，开始保持方向", state=int(state.get("sign_route_choice", 0)), min_interval=0.0)
        return

    if route_state == "IN_FORK":
        drive_started_at = state.get("sign_route_drive_started_at")
        fork_entered_at = state.get("sign_route_fork_entered_at")
        if drive_started_at is None:
            state["sign_route_drive_started_at"] = now
            drive_started_at = now
        if fork_entered_at is None:
            state["sign_route_fork_entered_at"] = now
            fork_entered_at = now

        max_hold = float(getattr(config, "SIGN_ROUTE_MAX_DRIVE_HOLD_SECONDS", 10.0))
        if now - float(drive_started_at) >= max_hold:
            throttled_log("sign_route_timeout", "语义路牌路线超时释放: 已到最长保持时间", state="in_fork", min_interval=0.0)
            reset_sign_route_after_drive(state)
            return

        if y_fork_active:
            state["sign_route_single_road_frames"] = 0
            return

        single_frames = int(state.get("sign_route_single_road_frames", 0)) + 1
        state["sign_route_single_road_frames"] = single_frames
        min_hold = float(getattr(config, "SIGN_ROUTE_MIN_FORK_HOLD_SECONDS", 1.0))
        need_frames = max(1, int(getattr(config, "SIGN_ROUTE_SINGLE_ROAD_EXIT_FRAMES", 20)))
        if now - float(fork_entered_at) >= min_hold and single_frames >= need_frames:
            throttled_log(
                "sign_route_done",
                f"语义路牌路线完成释放: 单路连续{single_frames}帧",
                state=(single_frames, int(state.get("sign_route_choice", 0))),
                min_interval=0.0,
            )
            reset_sign_route_after_drive(state)


def enqueue_sign_llm_job(frame_id, samples):
    """只保留最新一份语义路牌 LLM 任务，避免旧路牌请求排队积压."""
    if not sign_route_uses_llm():
        return False
    task = {
        "frame_id": int(frame_id),
        "samples": list(samples),
        "created_at": time.time(),
    }
    if llm_queue.full():
        try:
            llm_queue.get_nowait()
        except Exception:
            pass
    llm_queue.put(task)
    return True


def maybe_submit_sign_llm_job(state, frame_id, force=False):
    """两边各收满 3 条时提前提交，否则在采样总时长结束时提交已有文本。"""
    if (
        state.get("sign_llm_waiting_result", False) or
        state.get("sign_route_api_submitted", False) or
        str(state.get("sign_route_state", "IDLE")) != "SIGN_STOP_COLLECT"
    ):
        return False
    samples = list(state.get("sign_llm_samples", []))
    valid_samples = [sample for sample in samples if str(sample.get("text", "")).strip()]
    attempts = int(state.get("sign_llm_attempts", len(samples)))
    sample_need = max(1, int(getattr(config, "SIGN_LLM_OCR_SAMPLES", 3)))
    local_count = sum(1 for sample in valid_samples if sample.get("backend") == "local")
    api_count = sum(1 for sample in valid_samples if sample.get("backend") == "api")
    both_full = local_count >= sample_need and api_count >= sample_need
    if not force and not both_full:
        return False
    if not valid_samples:
        locked_rect = state.get("sign_route_locked_rect")
        clear_sign_llm_state(state, keep_completed=False)
        state["sign_llm_error"] = "collect_timeout_no_valid_samples"
        state["sign_route_pass_index"] = 1
        state["sign_route_llm_used"] = False
        state["sign_route_first_choice"] = 0
        reset_sign_route_state(state, next_state="WAIT_SIGN_GONE")
        block_sign_until_gone(state, locked_rect)
        throttled_log(
            "sign_llm_collect_timeout_empty",
            "语义路牌采样总时长结束但没有有效文本，放弃本次会话",
            state=(int(state.get("sign_llm_frame_id", -1)),),
            min_interval=0.0,
        )
        return False
    if not enqueue_sign_llm_job(frame_id, valid_samples):
        return False
    state["sign_llm_collecting"] = False
    state["sign_llm_waiting_result"] = True
    state["sign_llm_samples"] = valid_samples
    state["sign_route_state"] = "WAIT_API"
    state["sign_route_api_submitted"] = True
    throttled_log(
        "sign_route_api_submit",
        "语义路牌采样提交千帆: "
        f"frame={frame_id} local={local_count} api={api_count} total={len(valid_samples)} "
        f"force={int(force)}",
        state=(int(frame_id), local_count, api_count, len(valid_samples), int(force)),
        min_interval=0.0,
    )
    return True


def record_sign_llm_sample(state, frame_id, text="", score=0.0, reason="", backend="unknown"):
    """记录一次停车 OCR 采样尝试；空文本也计入尝试次数，便于诊断进度."""
    if not (
        sign_route_uses_llm() and
        bool(state.get("sign_llm_stop_active", False)) and
        bool(state.get("sign_llm_collecting", False)) and
        not bool(state.get("sign_route_api_submitted", False)) and
        str(state.get("sign_route_state", "IDLE")) == "SIGN_STOP_COLLECT"
    ):
        return False

    sample_need = max(1, int(getattr(config, "SIGN_LLM_OCR_SAMPLES", 10)))
    attempts = int(state.get("sign_llm_attempts", 0)) + 1
    state["sign_llm_attempts"] = attempts

    text = str(text or "").strip().upper()
    backend = backend_label(backend)
    if text:
        samples = list(state.get("sign_llm_samples", []))
        backend_count = sum(1 for sample in samples if sample.get("backend") == backend)
        if backend_count < sample_need:
            samples.append({
                "text": text,
                "score": float(score),
                "frame_id": int(frame_id),
                "backend": backend,
            })
            state["sign_llm_samples"] = samples

    valid_count = len([sample for sample in state.get("sign_llm_samples", []) if str(sample.get("text", "")).strip()])
    local_count = sum(1 for sample in state.get("sign_llm_samples", []) if sample.get("backend") == "local")
    api_count = sum(1 for sample in state.get("sign_llm_samples", []) if sample.get("backend") == "api")
    throttled_log(
        "sign_llm_sample",
        f"语义路牌OCR采样: attempts={attempts} local={local_count}/{sample_need} "
        f"api={api_count}/{sample_need} valid={valid_count} 文本={text or '<空>'} "
        f"置信度={float(score):.3f} 原因={reason or 'ok'}",
        state=(attempts, local_count, api_count, valid_count, text, reason),
        min_interval=0.0,
    )
    return maybe_submit_sign_llm_job(state, int(frame_id))


def mark_sign_ocr_done(frame_id=None, backend=None):
    with data_lock:
        active_backends = set(global_control_data.get("sign_llm_ocr_active_backends", []))
        label = backend_label(backend) if backend else ""
        if label:
            active_backends.discard(label)
        else:
            active_backends.clear()
        global_control_data["sign_llm_ocr_active_backends"] = sorted(active_backends)
        # 百度网络请求不该暂停 YOLO；只有本地 RKNN OCR 会与检测共享 NPU。
        local_busy = "local" in active_backends
        global_control_data["sign_llm_ocr_inflight"] = local_busy
        global_control_data["sign_llm_ocr_inflight_started_at"] = (
            global_control_data.get("sign_llm_ocr_inflight_started_at") if local_busy else None
        )
        if not active_backends:
            global_control_data["sign_llm_ocr_job_frame_id"] = -1


def create_ocr_recognizer(backend=None):
    """Create an OCR backend while keeping the worker interface stable."""
    backend = str(backend or getattr(config, "OCR_BACKEND", "local")).strip().lower()
    if backend in ("baidu", "baidu_api", "api", "cloud"):
        return BaiduOCRRecognizer(
            timeout=float(getattr(config, "BAIDU_OCR_TIMEOUT", 10.0)),
            image_format=str(getattr(config, "BAIDU_OCR_IMAGE_FORMAT", ".jpg")),
            jpeg_quality=int(getattr(config, "BAIDU_OCR_JPEG_QUALITY", 90)),
        )
    if backend in ("local", "rknn", "rknn_local"):
        return LocalOCRRecognizer(core_id=config.REC_CORE)
    raise RuntimeError(f"unsupported OCR_BACKEND: {backend}")


def get_ocr_recognizer(backend):
    """每个 OCR 后端只初始化一次，采样期间复用其模型/云端 token 缓存。"""
    normalized = str(backend or getattr(config, "OCR_BACKEND", "local")).strip().lower()
    with ocr_recognizer_lock:
        recognizer = ocr_recognizers.get(normalized)
        if recognizer is None:
            # RKNN 静态模型的 native 初始化提示没有诊断价值，且只应发生一次。
            with suppress_rknn_init_output():
                recognizer = create_ocr_recognizer(normalized)
            ocr_recognizers[normalized] = recognizer
        return recognizer


def sign_ocr_backends():
    """返回当前语义路牌采样要运行的 OCR 后端列表."""
    if bool(getattr(config, "SIGN_LLM_OCR_LOCAL_ONLY", True)):
        return [str(getattr(config, "OCR_LOCAL_BACKEND", "local")).strip().lower()]
    if not bool(getattr(config, "OCR_DUAL_BACKEND_ENABLED", False)):
        return [str(getattr(config, "OCR_BACKEND", "local")).strip().lower()]

    backends = [
        str(getattr(config, "OCR_LOCAL_BACKEND", "local")).strip().lower(),
        str(getattr(config, "OCR_API_BACKEND", "baidu_api")).strip().lower(),
    ]
    unique = []
    for backend in backends:
        if backend and backend not in unique:
            unique.append(backend)
    return unique or [str(getattr(config, "OCR_BACKEND", "local")).strip().lower()]


def backend_label(backend):
    backend = str(backend or "").strip().lower()
    if backend in ("baidu", "baidu_api", "api", "cloud"):
        return "api"
    if backend in ("local", "rknn", "rknn_local"):
        return "local"
    return backend or "unknown"


def run_ocr_backend_once(backend, frame_data):
    label = backend_label(backend)
    try:
        recognizer = get_ocr_recognizer(backend)
        results = recognizer.run_full_frame(frame_data)
        return {
            "backend": label,
            "results": results or [],
            "error": "",
            "det_box_count": int(getattr(recognizer, "last_det_box_count", 0)),
            "rec_empty_count": int(getattr(recognizer, "last_rec_empty_count", 0)),
            "rec_exception_count": int(getattr(recognizer, "last_rec_exception_count", 0)),
            "rec_valid_count": int(getattr(recognizer, "last_rec_valid_count", 0)),
        }
    except Exception as exc:
        return {
            "backend": label,
            "results": [],
            "error": str(exc),
            "det_box_count": 0,
            "rec_empty_count": 0,
            "rec_exception_count": 0,
            "rec_valid_count": 0,
        }
def submit_sign_ocr_backends(frame_data, sign_jobs, frame_id):
    """分别提交本地/API OCR，让任一后端完成时立即回写对应样本。"""
    with data_lock:
        if not bool(global_control_data.get("sign_llm_collecting", False)):
            return
        sample_need = max(1, int(getattr(config, "SIGN_LLM_OCR_SAMPLES", 3)))
        samples = list(global_control_data.get("sign_llm_samples", []))
        active_backends = set(global_control_data.get("sign_llm_ocr_active_backends", []))
        backends = []
        for backend in sign_ocr_backends():
            label = backend_label(backend)
            completed = sum(1 for sample in samples if sample.get("backend") == label)
            if completed < sample_need and label not in active_backends:
                backends.append(backend)
                active_backends.add(label)
        if not backends:
            return
        global_control_data["sign_llm_ocr_active_backends"] = sorted(active_backends)
        if "local" in {backend_label(backend) for backend in backends}:
            global_control_data["sign_llm_ocr_inflight"] = True
            global_control_data["sign_llm_ocr_inflight_started_at"] = time.monotonic()
            global_control_data["sign_llm_ocr_job_frame_id"] = int(frame_id)

    for backend in backends:
        label = backend_label(backend)
        executor = ocr_api_executor if label == "api" else ocr_local_executor
        future = executor.submit(run_ocr_backend_once, backend, frame_data)

        def publish_result(completed, backend_label_value=label, jobs=list(sign_jobs), result_frame_id=int(frame_id)):
            try:
                output = completed.result()
            except Exception as exc:
                output = {
                    "backend": backend_label_value,
                    "results": [],
                    "error": str(exc),
                    "det_box_count": 0,
                    "rec_empty_count": 0,
                    "rec_exception_count": 0,
                    "rec_valid_count": 0,
                }
            ocr_queue.put(("backend_result", output, jobs, result_frame_id))

        future.add_done_callback(publish_result)


def sign_llm_worker():
    """千帆语义判定独立进程，避免网络请求阻塞视觉/串口线程."""
    while True:
        task = llm_queue.get()
        if task is None:
            break
        frame_id = int(task.get("frame_id", -1))
        samples = list(task.get("samples", []))
        created_at = float(task.get("created_at", time.time()))
        sample_text = [(s.get("text"), round(float(s.get("score", 0.0)), 3)) for s in samples]
        worker_started_at = time.time()
        queue_wait_s = max(0.0, worker_started_at - created_at)
        print(f"千帆路牌任务开始: frame={frame_id} queue={queue_wait_s:.2f}s samples={sample_text}", flush=True)
        try:
            api_started_at = time.time()
            result, raw = request_road_choice(
                samples,
                timeout=float(getattr(config, "SIGN_LLM_API_TIMEOUT", 3.0)),
            )
            api_elapsed_s = max(0.0, time.time() - api_started_at)
            total_elapsed_s = max(0.0, time.time() - created_at)
            payload = {
                "frame_id": frame_id,
                "result": result,
                "raw": raw,
                "error": "",
                "samples": samples,
                "created_at": created_at,
                "queue_wait_s": queue_wait_s,
                "api_elapsed_s": api_elapsed_s,
                "total_elapsed_s": total_elapsed_s,
            }
            print(
                f"{LOG_GREEN}千帆路牌任务返回: frame={frame_id} result={result} "
                f"api={api_elapsed_s:.2f}s total={total_elapsed_s:.2f}s raw={raw}{LOG_RESET}",
                flush=True,
            )
        except Exception as e:
            api_elapsed_s = max(0.0, time.time() - worker_started_at)
            total_elapsed_s = max(0.0, time.time() - created_at)
            payload = {
                "frame_id": frame_id,
                "result": "",
                "raw": "",
                "error": str(e),
                "samples": samples,
                "created_at": created_at,
                "queue_wait_s": queue_wait_s,
                "api_elapsed_s": api_elapsed_s,
                "total_elapsed_s": total_elapsed_s,
            }
            print(f"千帆路牌任务异常: frame={frame_id} api={api_elapsed_s:.2f}s total={total_elapsed_s:.2f}s error={e}", flush=True)
        if llm_result_queue.full():
            try:
                llm_result_queue.get_nowait()
            except Exception:
                pass
        llm_result_queue.put(payload)


def drain_sign_llm_results():
    """在主进程内消费 LLM 结果，写入 turn_intent 并释放停车状态."""
    while True:
        try:
            item = llm_result_queue.get_nowait()
        except Exception:
            break
        frame_id = int(item.get("frame_id", -1))
        result = str(item.get("result", "")).strip().upper()
        error = str(item.get("error", "")).strip()
        samples = list(item.get("samples", []))
        created_at = float(item.get("created_at", time.time()))
        queue_wait_s = max(0.0, float(item.get("queue_wait_s", 0.0)))
        api_elapsed_s = max(0.0, float(item.get("api_elapsed_s", 0.0)))
        total_elapsed_s = max(0.0, float(item.get("total_elapsed_s", time.time() - created_at)))

        with data_lock:
            waiting = bool(global_control_data.get("sign_llm_waiting_result", False))
            active_fid = int(global_control_data.get("sign_llm_frame_id", -1))
            active_rect = global_control_data.get("sign_route_locked_rect")
            current_fid = int(global_yolo_frame_id)
            max_age = max(1, int(getattr(config, "SIGN_LLM_RESULT_MAX_AGE_FRAMES", 60)))
            too_old = current_fid >= 0 and frame_id >= 0 and current_fid - frame_id > max_age
            if not waiting:
                print(
                    "千帆路牌结果丢弃: "
                    f"waiting={waiting} result_frame={frame_id} active_frame={active_fid} "
                    f"current_frame={current_fid} max_age={max_age} too_old={too_old}",
                    flush=True,
                )
                continue

            if result in ("LEFT", "RIGHT"):
                choice = -1 if result == "LEFT" else 1
                if int(global_control_data.get("sign_route_first_choice", 0)) not in (-1, 1):
                    global_control_data["sign_route_first_choice"] = choice
                global_control_data["sign_route_pass_index"] = max(2, sign_route_pass_index(global_control_data))
                activate_sign_route_choice(global_control_data, choice, max(frame_id, active_fid))
                clear_sign_llm_state(global_control_data, keep_completed=True)
            else:
                fallback_error = error or "invalid_empty_result"
                global_control_data["sign_llm_result"] = ""
                clear_sign_llm_state(global_control_data, keep_completed=True)
                global_control_data["sign_llm_error"] = fallback_error
                # 千帆网络失败时，当前岔路按既有默认 LEFT 放行；它仍是本次路线
                # 的首次选择，下一次“带路牌的 Y 岔”必须以此为基准走 RIGHT。
                fallback_choice = -1
                global_control_data["sign_route_first_choice"] = fallback_choice
                global_control_data["sign_route_pass_index"] = 2
                reset_sign_route_state(global_control_data, next_state="WAIT_SIGN_GONE")
                block_sign_until_gone(global_control_data, active_rect)

        if result in ("LEFT", "RIGHT"):
            sample_text = [(s.get("text"), round(float(s.get("score", 0.0)), 3)) for s in samples]
            print(
                f"{LOG_GREEN}千帆路牌最终结果: {result} "
                f"耗时={total_elapsed_s:.2f}s API={api_elapsed_s:.2f}s 排队={queue_wait_s:.2f}s "
                f"samples={sample_text}{LOG_RESET}",
                flush=True,
            )
        else:
            print(
                f"千帆路牌识别失败，当前按默认LEFT放行；下一次带路牌岔路将取反为RIGHT: {error} "
                f"耗时={total_elapsed_s:.2f}s API={api_elapsed_s:.2f}s 排队={queue_wait_s:.2f}s",
                flush=True,
            )


# ==============================================================================
# 核心线程 1：YOLO 检测线程
# ==============================================================================
def yolo_worker(core_id=None, worker_id=0):
    """纯检测线程.

    这个线程只做两件事:
    1. 执行 YOLO 推理并更新 global_yolo_boxes
    2. 将 sign 类别的候选任务连同同帧 TARGET_RES 图一起投递给 OCR 线程

    这样 OCR 不会反过来阻塞 YOLO，能显著降低检测延迟。
    """
    if core_id is None:
        core_id = config.YOLO_CORE
    worker_id = int(worker_id)
    input_queue = yolo_queues[worker_id % len(yolo_queues)]
    try:
        det = YOLODetector(core_id=core_id)
    except Exception as e:
        log_once(f"yolo_init_error_{worker_id}", f"YOLO启动失败(worker={worker_id}, core={core_id}): {e}")
        return

    while True:
        try:
            t_loop_start = time.perf_counter()
            if bool(getattr(config, "YOLO_PAUSE_DURING_OCR", True)):
                with data_lock:
                    pause_for_ocr = bool(global_control_data.get("sign_llm_ocr_inflight", False))
                    pause_started_at = global_control_data.get("sign_llm_ocr_inflight_started_at")
                    if pause_for_ocr and pause_started_at is not None:
                        pause_timeout = float(getattr(config, "YOLO_PAUSE_OCR_TIMEOUT", 1.5))
                        if time.monotonic() - float(pause_started_at) >= pause_timeout:
                            pause_for_ocr = False
                            global_control_data["sign_llm_ocr_inflight"] = False
                            global_control_data["sign_llm_ocr_inflight_started_at"] = None
                            throttled_log(
                                "yolo_pause_ocr_timeout",
                                f"YOLO等待OCR超时，恢复检测: timeout={pause_timeout:.1f}s",
                                state="timeout",
                                min_interval=0.0,
                            )
                if pause_for_ocr:
                    time.sleep(0.005)
                    continue

            frame_data = input_queue.get()
            t_after_get = time.perf_counter()
            drained_input = 0
            if bool(getattr(config, "YOLO_WORKER_DRAIN_LATEST_QUEUE", True)):
                frame_data, drained_input = drain_latest_queue_item(input_queue, frame_data)
            t_after_drain = time.perf_counter()
            if frame_data is None:
                break
            if drained_input > 0:
                throttled_log(
                    f"yolo_drain_latest_{worker_id}",
                    f"YOLO[{worker_id}]丢弃排队旧输入: dropped={drained_input}",
                    state=int(drained_input),
                    min_interval=1.0,
                )

            det_frame, vis_frame, src_size, frame_id = frame_data
            t_det_start = time.perf_counter()
            objs = det.run(det_frame, output_size=config.TARGET_RES)
            t_det_end = time.perf_counter()
            det_timing = getattr(det, "last_timing", {})
            det_counts = getattr(det, "last_decode_counts", {})

            # 新一帧检测结果先清掉旧的 OCR 文本，避免沿用上一帧残留内容。
            for obj in objs:
                try:
                    obj.pop("text", None)
                    obj.pop("ocr_score", None)
                    obj.pop("sign_distance_m", None)
                    obj.pop("ground_distance_m", None)
                    if obj.get("class_id") == config.SIGN_CLASS_ID:
                        sign_distance_m = estimate_sign_distance_m(obj.get("rect", [0, 0, 0, 0]))
                        if sign_distance_m is not None:
                            obj["sign_distance_m"] = float(sign_distance_m)
                    obj_class_name = str(obj.get("class_name", ""))
                    obj_class_id = int(obj.get("class_id", -1))
                    person_cls_id = (
                        config.CLASS_NAMES.index(config.PERSON_CLASS_NAME)
                        if config.PERSON_CLASS_NAME in config.CLASS_NAMES
                        else config.PERSON_CLASS_ID_FALLBACK
                    )
                    car_cls_id = config.CLASS_NAMES.index("car") if "car" in config.CLASS_NAMES else -1
                    if (
                        obj_class_name in ("car", "person") or
                        obj_class_id in (car_cls_id, person_cls_id)
                    ):
                        ground_distance_m = estimate_ground_object_distance_m(obj.get("rect", [0, 0, 0, 0]))
                        if ground_distance_m is not None:
                            obj["ground_distance_m"] = float(ground_distance_m)
                except Exception:
                    pass

            with data_lock:
                global global_yolo_boxes, global_yolo_frame_id
                frame_is_current = int(frame_id) >= int(global_yolo_frame_id)
                if frame_is_current:
                    global_yolo_boxes = objs
                    global_yolo_frame_id = int(frame_id)
                fps_stats["yolo_frames"] += 1
            t_update_end = time.perf_counter()
            if not frame_is_current:
                continue

            # 只把通过门槛的 sign 送去 OCR，减少不必要的整图文字检测开销。
            sign_jobs = []
            pending_sign_jobs = []
            visible_sign_rects = []
            visible_sign_rect = None
            route_trigger_rect = None
            route_skip_debug = None
            # 距离、贴边门槛只服务于首次停车 OCR。首次结果已确定后，第二次
            # 路牌只用于武装取反，不能再被这些门槛拦住或输出“未达标”。
            with data_lock:
                route_pass_index_for_scan = sign_route_pass_index(global_control_data)
            route_sign_gate_enabled = (
                sign_route_uses_llm() and
                route_pass_index_for_scan <= 1
            )
            for idx, obj in enumerate(objs):
                cls_id = obj.get("class_id")
                if cls_id != config.SIGN_CLASS_ID:
                    continue
                rect = obj.get("rect", [0, 0, 0, 0])
                if len(rect) == 4 and float(rect[2]) >= 2.0 and float(rect[3]) >= 2.0:
                    visible_sign_rects.append(list(rect))
                    if visible_sign_rect is None or rect_area(rect) > rect_area(visible_sign_rect):
                        visible_sign_rect = list(rect)
                if route_sign_gate_enabled and route_trigger_rect is None:
                    should_trigger, trigger_skip_reason = should_trigger_sign_route(cls_id, rect)
                    if should_trigger:
                        route_trigger_rect = list(rect)
                    elif route_skip_debug is None:
                        x, y, w, h = rect if len(rect) == 4 else (0, 0, 0, 0)
                        frame_w, frame_h = config.TARGET_RES
                        route_skip_debug = {
                            "reason": trigger_skip_reason,
                            "area": rect_area(rect),
                            "dist": max(0.0, float(frame_h) - float(y + h)),
                            "sign_distance_m": estimate_sign_distance_m(rect),
                            "rect": list(rect),
                        }
                should_enqueue, skip_reason = should_enqueue_ocr_job(cls_id, rect)
                if not should_enqueue:
                    continue
                pending_sign_jobs.append((idx, cls_id, rect))
            if (
                route_sign_gate_enabled and
                route_trigger_rect is None and
                route_skip_debug is not None
            ):
                # 与取反状态在同一把锁内复核，避免旧 YOLO 帧在取反之后才把
                # 已经无意义的贴边诊断打印出来。
                with data_lock:
                    if sign_route_pass_index(global_control_data) < 3:
                        throttled_log(
                            "sign_route_trigger_skip",
                            "语义路牌未达标: "
                            f"原因={route_skip_debug['reason']} "
                            f"area={route_skip_debug['area']:.0f} "
                            f"dist={route_skip_debug['dist']:.0f} "
                            f"sign_z={route_skip_debug.get('sign_distance_m')} "
                            f"rect={route_skip_debug['rect']}",
                            state=route_skip_debug["reason"],
                            min_interval=2.0,
                        )

            sign_llm_collecting_now = False
            with data_lock:
                route_state = str(global_control_data.get("sign_route_state", "IDLE"))
                if sign_route_uses_llm():
                    route_state = update_wait_sign_gone_state(
                        global_control_data,
                        route_trigger_rect,
                        visible_sign_rects=visible_sign_rects,
                    )
                pass_index = sign_route_pass_index(global_control_data)
                first_choice = int(global_control_data.get("sign_route_first_choice", 0))
                llm_used = bool(global_control_data.get("sign_route_llm_used", False))
                fork_visible_now = bool(global_control_data.get("sign_route_y_fork_active", False))
                if (
                    route_state == "IDLE" and
                    pass_index <= 1 and
                    not llm_used and
                    visible_sign_rect is not None
                ):
                    # YOLO 与分割是异步线程，不能要求 YOLO 先读到上一帧的 Y 岔结果。
                    # 第一次看到路牌就把中线优先权交给分割；分割在当前帧检测到 Y 岔时
                    # 立即拉“底部中点 -> 分叉点”的临时线。停车仍由下面的距离+Y 岔门控决定。
                    global_control_data["sign_route_state"] = "APPROACH_SIGN"
                    global_control_data["sign_route_locked_rect"] = list(visible_sign_rect)
                    route_state = "APPROACH_SIGN"
                    throttled_log(
                        "sign_route_approach_start",
                        "首次看到路牌，等待分割检测Y岔后沿中线接近并等待距离达标: "
                        f"reason={route_skip_debug['reason'] if route_skip_debug is not None else 'visible'} "
                        f"frame={frame_id} rect={visible_sign_rect}",
                        state=(
                            int(frame_id),
                            route_skip_debug["reason"] if route_skip_debug is not None else "visible",
                        ),
                        min_interval=0.0,
                    )
                elif pass_index == 2 and first_choice in (-1, 1):
                    # 第二次看见路牌先记录“待取反”，不要求此刻 YOLO 已经读到 Y 岔。
                    # 分割线程稍后确认 Y 岔且仍看见路牌时，才真正执行取反。
                    if route_state == "IDLE" and visible_sign_rect is not None:
                        global_control_data["sign_route_state"] = "REVERSE_APPROACH_SIGN"
                        global_control_data["sign_route_locked_rect"] = list(visible_sign_rect)
                        route_state = "REVERSE_APPROACH_SIGN"
                        throttled_log(
                            "sign_route_reverse_approach",
                            f"第二次看到路牌，已准备取反并等待Y岔: "
                            f"first={sign_route_label(first_choice)} frame={frame_id} "
                            f"rect={visible_sign_rect}",
                            state=(first_choice, int(frame_id)),
                            min_interval=0.0,
                        )
                if (
                    route_trigger_rect is not None and
                    route_state in ("IDLE", "APPROACH_SIGN") and
                    fork_visible_now
                ):
                    if pass_index == 2 and first_choice in (-1, 1):
                        reverse_choice = -first_choice
                        global_control_data["sign_route_pass_index"] = 3
                        clear_sign_llm_state(global_control_data, keep_completed=True)
                        activate_sign_route_choice(
                            global_control_data,
                            reverse_choice,
                            int(frame_id),
                            locked_rect=route_trigger_rect,
                        )
                        throttled_log(
                            "sign_route_reverse_choice",
                            f"第二次岔路路牌与岔路同时出现，直接取反: "
                            f"first={sign_route_label(first_choice)} current={sign_route_label(reverse_choice)} "
                            f"frame={frame_id} rect={route_trigger_rect}",
                            state=(first_choice, reverse_choice, int(frame_id)),
                            min_interval=0.0,
                        )
                    elif llm_used or pass_index >= 3:
                        locked_rect = list(route_trigger_rect)
                        clear_sign_llm_state(global_control_data, keep_completed=True)
                        reset_sign_route_state(global_control_data, next_state="WAIT_SIGN_GONE")
                        block_sign_until_gone(global_control_data, locked_rect)
                        throttled_log(
                            "sign_route_llm_done_ignore",
                            f"路牌识别已使用过，不再启动OCR/千帆: frame={frame_id} rect={route_trigger_rect}",
                            state=(pass_index, llm_used, int(frame_id)),
                            min_interval=0.0,
                        )
                    else:
                        if pass_index <= 0:
                            global_control_data["sign_route_pass_index"] = 1
                        global_control_data["sign_route_llm_used"] = True
                        global_control_data["sign_llm_frame_id"] = int(frame_id)
                        global_control_data["sign_route_state"] = "SIGN_STOP_COLLECT"
                        global_control_data["sign_route_locked_rect"] = route_trigger_rect
                        global_control_data["sign_route_choice"] = 0
                        global_control_data["sign_route_drive_started_at"] = None
                        global_control_data["sign_route_fork_entered_at"] = None
                        global_control_data["sign_route_single_road_frames"] = 0
                        global_control_data["sign_route_api_submitted"] = False
                        global_control_data["sign_llm_stop_active"] = True
                        global_control_data["sign_llm_collecting"] = True
                        global_control_data["sign_llm_waiting_result"] = False
                        global_control_data["sign_llm_completed_hold"] = False
                        global_control_data["sign_llm_samples"] = []
                        global_control_data["sign_llm_attempts"] = 0
                        global_control_data["sign_llm_ocr_inflight"] = False
                        global_control_data["sign_llm_ocr_inflight_started_at"] = None
                        global_control_data["sign_llm_ocr_active_backends"] = []
                        global_control_data["sign_llm_ocr_started"] = False
                        global_control_data["sign_llm_ocr_job_frame_id"] = -1
                        global_control_data["sign_llm_stable_rect"] = None
                        global_control_data["sign_llm_ocr_ready"] = False
                        global_control_data["sign_llm_started_at"] = time.monotonic()
                        global_control_data["sign_llm_result"] = ""
                        global_control_data["sign_llm_error"] = ""
                        throttled_log(
                            "sign_llm_stop_start",
                            f"第一次岔路路牌停车采集OCR并等待千帆: "
                            f"frame={frame_id} rect={route_trigger_rect}",
                            state=("start", int(frame_id)),
                            min_interval=0.0,
                        )
                sign_llm_collecting_now = bool(global_control_data.get("sign_llm_collecting", False))
                locked_rect = global_control_data.get("sign_route_locked_rect")

            if sign_llm_collecting_now:
                current_sign_rect = best_matching_sign_rect(visible_sign_rects, locked_rect)
                with data_lock:
                    ocr_ready = update_sign_llm_rect_stability(global_control_data, current_sign_rect)
                    locked_rect = global_control_data.get("sign_route_locked_rect")
                if ocr_ready:
                    sample_rect = current_sign_rect if current_sign_rect is not None else locked_rect
                    if sample_rect is not None:
                        sign_jobs.append((-1, config.SIGN_CLASS_ID, list(sample_rect)))

            if sign_jobs:
                if sign_llm_collecting_now:
                    with data_lock:
                        # 具体后端的并发占用由 submit_sign_ocr_backends 管理：本地和
                        # 百度各最多一条在途，慢百度不会阻止本地继续补足三个样本。
                        global_control_data["sign_llm_ocr_started"] = True

            if sign_jobs:
                job_names = tuple(class_name_from_id(cls_id) for _, cls_id, _ in sign_jobs)
                throttled_log(
                    "ocr_enter",
                    f"路牌停车后延时开始OCR: 数量={len(sign_jobs)} 类型={list(job_names)} backends={sign_ocr_backends()}",
                    state=(job_names, tuple(sign_ocr_backends())),
                    min_interval=config.LOG_INTERVAL_OCR_ENTER
                )
                # OCR 只保留较新的任务，过旧的任务直接丢掉。
                if ocr_queue.full():
                    try:
                        ocr_queue.get_nowait()
                    except:
                        pass
                ocr_queue.put((vis_frame.copy(), sign_jobs, int(frame_id)))
            t_loop_end = time.perf_counter()
            profile_log(
                f"yolo_worker_loop_{worker_id}",
                f"YoloProfile[{worker_id}]",
                {
                    "wait_input": t_after_get - t_loop_start,
                    "drain_input": t_after_drain - t_after_get,
                    "det_run": t_det_end - t_det_start,
                    "det_pre": float(det_timing.get("preprocess", 0.0)),
                    "det_rknn": float(det_timing.get("rknn", 0.0)),
                    "det_decode": float(det_timing.get("decode", 0.0)),
                    "det_raw_count": float(det_counts.get("raw", 0)),
                    "det_topk_count": float(det_counts.get("topk", 0)),
                    "det_nms_count": float(det_counts.get("nms", 0)),
                    "update": t_update_end - t_det_end,
                    "post": t_loop_end - t_update_end,
                    "loop": t_loop_end - t_loop_start,
                },
            )

        except Exception as e:
            if "worker closed stdout" in str(e) or "Broken pipe" in str(e):
                return
            log_once(f"yolo_worker_error_{worker_id}", f"YOLO线程异常(worker={worker_id}, core={core_id}): {e}")


# ==============================================================================
# 核心线程 1.5：OCR 识别线程
# ==============================================================================
def ocr_worker():
    """纯 OCR 线程.

    输入:
        yolo_worker 投递的 (frame_data, sign_jobs, frame_id)

    输出:
        将识别出的 text / ocr_score 回写到 global_yolo_boxes，
        同时根据 LEFT / RIGHT 更新 turn_intent。

    关键约束:
    - OCR 在整张 TARGET_RES 图上执行 det + rec，不是直接拿检测框裁图识别
    - 同一帧里的多个 OCR 结果，会按“中心点最近”去匹配各个路牌框
    - 匹配结果回写时还会再核对 frame_id，避免旧帧 OCR 迟到污染新状态
    """
    job_frame_id = None
    sample_backend = None
    while True:
        try:
            job_frame_id = None
            sample_backend = None
            job = ocr_queue.get()
            if job is None:
                break

            backend_result_job = (
                isinstance(job, tuple) and
                len(job) == 4 and
                job[0] == "backend_result"
            )
            if backend_result_job:
                _, backend_output, sign_jobs, frame_id = job
                frame_data = None
                backend_outputs = [backend_output]
            else:
                frame_data, sign_jobs, frame_id = job
                backend_outputs = None
            job_frame_id = int(frame_id)
            with data_lock:
                sign_collecting = (
                    sign_route_uses_llm() and
                    bool(global_control_data.get("sign_llm_collecting", False))
                )
            sign_jobs = [
                item for item in sign_jobs
                if item[1] != config.SIGN_CLASS_ID or sign_collecting
            ]
            if not sign_jobs:
                mark_sign_ocr_done(job_frame_id)
                continue

            if not backend_result_job:
                # 两个后端分别异步执行；本地完成后即可开启下一轮，API 慢不会堵住采样窗口。
                submit_sign_ocr_backends(frame_data, sign_jobs, frame_id)
                continue

            updates = []
            # 当前队列项只对应一个已经完成的 OCR 后端结果。
            sample_backend = str(backend_outputs[0].get("backend", "unknown"))
            ocr_results = []
            for backend_output in backend_outputs:
                backend_name = str(backend_output.get("backend", "unknown"))
                error = str(backend_output.get("error", ""))
                if error:
                    throttled_log(
                        f"ocr_backend_error_{backend_name}",
                        f"OCR后端失败({backend_name}): {error}",
                        state=(backend_name, error),
                        min_interval=config.LOG_INTERVAL_OCR_RAW,
                    )
                for result in backend_output.get("results", []) or []:
                    item = dict(result)
                    item["backend"] = backend_name
                    ocr_results.append(item)

            if not ocr_results:
                with data_lock:
                    submitted = record_sign_llm_sample(
                        global_control_data,
                        int(frame_id),
                        "",
                        0.0,
                        "no_ocr_results",
                        backend=sample_backend,
                    )
                    if submitted:
                        throttled_log(
                            "sign_llm_submit",
                            f"语义路牌OCR样本已提交千帆: samples={len(global_control_data.get('sign_llm_samples', []))}",
                            state=("submit", int(frame_id)),
                            min_interval=0.0,
                        )
                throttled_log(
                    "ocr_no_results",
                    "OCR无有效文本: "
                    f"backends={[(b.get('backend'), len(b.get('results', []) or []), b.get('error', '')) for b in backend_outputs]} "
                    f"jobs={len(sign_jobs)}",
                    state=tuple((b.get("backend"), len(b.get("results", []) or []), str(b.get("error", ""))[:40]) for b in backend_outputs),
                    min_interval=config.LOG_INTERVAL_OCR_RAW
                )
                mark_sign_ocr_done(job_frame_id, sample_backend)
                continue

            min_ocr_score = float(config.OCR_MIN_SCORE)
            raw_texts = [
                f"{str(result.get('text', '')).strip()}:{float(result.get('score', 0.0)):.3f}"
                for result in ocr_results
                if float(result.get("score", 0.0)) >= min_ocr_score
            ]
            if raw_texts:
                throttled_log(
                    "ocr_raw_results",
                    f"OCR整图原始结果: {raw_texts}",
                    state=tuple(raw_texts),
                    min_interval=config.LOG_INTERVAL_OCR_RAW
                )

            full_frame_matches = []
            for result_id, result in enumerate(ocr_results):
                ocr_cx, ocr_cy = points_center(result.get("points"))
                text = str(result.get("text", "")).strip().upper()
                score = float(result.get("score", 0.0))
                if not text or score < min_ocr_score:
                    continue
                full_frame_matches.append((result_id, result, ocr_cx, ocr_cy))

            full_frame_matches.sort(key=lambda item: (float(item[3]), float(item[2])))
            full_frame_text = ""
            full_frame_score = 0.0
            full_frame_sample_recorded = False
            if full_frame_matches:
                full_frame_text, full_frame_score, full_frame_detail = pack_ocr_matches(full_frame_matches)
                with data_lock:
                    sign_llm_active = (
                        sign_route_uses_llm() and
                        bool(global_control_data.get("sign_llm_stop_active", False)) and
                        bool(global_control_data.get("sign_llm_collecting", False))
                    )
                    if sign_llm_active:
                        full_frame_sample_recorded = True
                        submitted = record_sign_llm_sample(
                            global_control_data,
                            int(frame_id),
                            full_frame_text,
                            full_frame_score,
                            "full_frame_ocr",
                            backend=sample_backend,
                        )
                        if submitted:
                            throttled_log(
                                "sign_llm_submit",
                                f"语义路牌OCR样本已提交千帆: samples={len(global_control_data.get('sign_llm_samples', []))}",
                                state=("submit", int(frame_id)),
                                min_interval=0.0,
                            )
                throttled_log(
                    "ocr_full_frame_llm_sample",
                    f"OCR整图样本送LLM: 文本={full_frame_text or '<空>'} "
                    f"置信度={full_frame_score:.3f} 条数={len(full_frame_matches)} matches={full_frame_detail}",
                    state=(full_frame_text, round(full_frame_score, 3), len(full_frame_matches)),
                    min_interval=config.LOG_INTERVAL_OCR_RAW,
                )

            used_result_ids = set()
            for idx, cls_id, rect in sign_jobs:
                try:
                    match_expand_ratio = float(getattr(config, "SIGN_OCR_MATCH_EXPAND_RATIO", 0.20))
                    match_rect = expanded_rect(rect, match_expand_ratio)
                    grouped_matches = []

                    for result_id, result, ocr_cx, ocr_cy in full_frame_matches:
                        if result_id in used_result_ids:
                            continue
                        match_item = (result_id, result, ocr_cx, ocr_cy)
                        if point_in_rect((ocr_cx, ocr_cy), match_rect):
                            grouped_matches.append(match_item)

                    if grouped_matches:
                        grouped_matches.sort(key=lambda item: (float(item[3]), float(item[2])))
                        used_result_ids.update(result_id for result_id, _, _, _ in grouped_matches)
                        text, score, match_detail = pack_ocr_matches(grouped_matches)
                        throttled_log(
                            "ocr_sign_match_raw",
                            f"OCR匹配到sign扩展框: 文本={text or '<空>'} 置信度={score:.3f} "
                            f"条数={len(grouped_matches)} 扩展比例={match_expand_ratio:.2f} rect={rect} matches={match_detail}",
                            state=(idx, text, round(score, 3), len(grouped_matches)),
                            min_interval=config.LOG_INTERVAL_OCR_RAW
                        )
                    else:
                        throttled_log(
                            "ocr_match_missing",
                            f"OCR没有文本落在sign扩展框内: sign_rect={rect} 达标文本框={len(full_frame_matches)}",
                            state=(idx, len(ocr_results)),
                            min_interval=config.LOG_INTERVAL_OCR_RAW
                        )
                        continue

                    if not text:
                        throttled_log(
                            "ocr_empty_text",
                            f"OCR匹配结果为空文本: 置信度={score:.3f} rect={rect}",
                            state=idx,
                            min_interval=config.LOG_INTERVAL_OCR_RAW
                        )
                        continue

                    updates.append((idx, cls_id, text, score))
                except Exception as e:
                    log_once("ocr_single_box_error", f"OCR单框处理异常: {e}")

            if not updates:
                if not full_frame_sample_recorded:
                    with data_lock:
                        submitted = record_sign_llm_sample(
                            global_control_data,
                            int(frame_id),
                            "",
                            0.0,
                            "no_valid_update",
                            backend=sample_backend,
                        )
                        if submitted:
                            throttled_log(
                                "sign_llm_submit",
                                f"语义路牌OCR样本已提交千帆: samples={len(global_control_data.get('sign_llm_samples', []))}",
                                state=("submit", int(frame_id)),
                                min_interval=0.0,
                            )
                throttled_log(
                    "ocr_no_updates",
                    f"OCR未形成有效更新: 文本框={len(ocr_results)} jobs={len(sign_jobs)}",
                    state=(len(ocr_results), len(sign_jobs)),
                    min_interval=config.LOG_INTERVAL_OCR_RAW
                )
                mark_sign_ocr_done(job_frame_id, sample_backend)
                continue

            # 回写时再做一次 class_id 检查，避免队列延迟导致“框已经换帧”的情况。
            with data_lock:
                current_yolo_frame_id = int(global_yolo_frame_id)
                sign_class_id = config.SIGN_CLASS_ID

                for idx, job_cls_id, text, score in updates:
                    # 只有 OCR 结果仍对应当前显示的检测帧时，才把文字回写到显示框上。
                    if (
                        int(frame_id) == current_yolo_frame_id and
                        idx >= 0 and
                        idx < len(global_yolo_boxes) and
                        global_yolo_boxes[idx].get("class_id") == job_cls_id
                    ):
                        global_yolo_boxes[idx]["text"] = text
                        global_yolo_boxes[idx]["ocr_score"] = score

                    if job_cls_id == sign_class_id:
                        last_turn_fid = int(global_control_data.get("turn_intent_fid", -1))
                        if int(frame_id) >= last_turn_fid:
                            matched_class = class_name_from_id(job_cls_id)
                            throttled_log(
                                "ocr_sign_result",
                                f"路牌识别结果: 类型={matched_class} 文本={text} 置信度={score:.3f}",
                                state=(matched_class, text),
                                min_interval=config.LOG_INTERVAL_OCR_ENTER
                            )
                            sign_llm_active = (
                                sign_route_uses_llm() and
                                bool(global_control_data.get("sign_llm_stop_active", False)) and
                                bool(global_control_data.get("sign_llm_collecting", False))
                            )
                            if sign_llm_active:
                                if full_frame_text:
                                    continue
                                if record_sign_llm_sample(
                                    global_control_data,
                                    int(frame_id),
                                    text,
                                    score,
                                    "sign_box_ocr",
                                    backend=sample_backend,
                                ):
                                    throttled_log(
                                        "sign_llm_submit",
                                        f"语义路牌OCR样本已提交千帆: samples={len(global_control_data.get('sign_llm_samples', []))}",
                                        state=("submit", int(frame_id)),
                                        min_interval=0.0,
                                    )
                            elif not bool(getattr(config, "SIGN_LLM_ENABLED", True)) and text == "LEFT":
                                global_control_data["turn_intent"] = -1
                                global_control_data["turn_intent_fid"] = int(frame_id)
                                throttled_log(
                                    "turn_intent",
                                    "语义路牌生效: 分叉意图=LEFT",
                                    state="LEFT",
                                    min_interval=config.LOG_INTERVAL_TURN_INTENT
                                )
                            elif not bool(getattr(config, "SIGN_LLM_ENABLED", True)) and text == "RIGHT":
                                global_control_data["turn_intent"] = 1
                                global_control_data["turn_intent_fid"] = int(frame_id)
                                throttled_log(
                                    "turn_intent",
                                    "语义路牌生效: 分叉意图=RIGHT",
                                    state="RIGHT",
                                    min_interval=config.LOG_INTERVAL_TURN_INTENT
                                )
            mark_sign_ocr_done(job_frame_id, sample_backend)

        except Exception as e:
            mark_sign_ocr_done(job_frame_id, sample_backend)
            log_once("ocr_worker_error", f"OCR线程异常: {e}")

# ==============================================================================
# 核心线程 2：分割与路径规划线程
# ==============================================================================
def seg_worker(core_id, worker_id=0):
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
        y_fork_active=False,
    ):
        global global_preview_frame

        person_info = extract_person_stop_candidate(current_yolo_boxes, current_yolo_frame_id)
        person_left_boundary_x = None
        person_right_boundary_x = None
        if person_info is not None:
            person_left_boundary_x = seg.selected_left_boundary_x_at_target_y(person_info["bottom_y"])
            person_right_boundary_x = seg.selected_right_boundary_x_at_target_y(person_info["bottom_y"])
        person_car_on_left = has_car_on_left(
            current_yolo_boxes,
            person_left_boundary_x,
            person_right_boundary_x,
        )

        if rendered_img is not None:
            if rendered_img.shape[1] != config.TARGET_RES[0] or rendered_img.shape[0] != config.TARGET_RES[1]:
                rendered_img = expand_seg_render_to_target(rendered_img, preview_frame)
            rendered_img = draw_yolo_boxes(rendered_img, current_yolo_boxes)

        car_stats = getattr(seg, "last_branch_stats", {}) or {}
        with data_lock:
            global_control_data["steer_signal"] = steer_signal
            global_control_data["car_avoidance_active"] = bool(car_stats.get("car_active", False))
            global_control_data["car_avoidance_state"] = str(car_stats.get("car_state", "FOLLOW_LANE"))
            global_control_data["car_avoidance_rows_to_bottom"] = car_stats.get("car_rows_to_bottom")
            global_control_data["car_avoidance_ground_distance_m"] = car_stats.get("car_ground_distance_m")
            global_control_data["car_avoidance_miss_frames"] = int(car_stats.get("car_miss_frames", 0))
            global_control_data["car_avoidance_clear_frames"] = int(car_stats.get("car_clear_frames", 0))
            global_control_data["car_avoidance_state_paused"] = bool(car_stats.get("car_state_paused", False))
            global_control_data["car_avoidance_boundary_side"] = str(car_stats.get("car_boundary_side", ""))
            global_control_data["car_avoidance_nearest_boundary_side"] = str(car_stats.get("car_nearest_boundary_side", ""))
            global_control_data["car_avoidance_boundary_x"] = car_stats.get("car_boundary_x")
            global_control_data["car_avoidance_boundary_error"] = car_stats.get("car_boundary_error")
            global_control_data["car_avoidance_left_boundary_error"] = car_stats.get("car_left_boundary_error")
            global_control_data["car_avoidance_left_boundary_x"] = car_stats.get("car_left_boundary_x")
            global_control_data["car_avoidance_control_path_error"] = car_stats.get("car_control_path_error")
            global_control_data["car_avoidance_boundary_inset_x"] = float(car_stats.get("car_boundary_inset_x", 0.0))
            global_control_data["car_avoidance_control_error_offset_x"] = float(car_stats.get("car_control_error_offset_x", 0.0))
            global_control_data["car_avoidance_detected_cars"] = int(car_stats.get("car_detected_cars", 0))
            global_control_data["car_avoidance_locked_confirmed"] = bool(car_stats.get("car_locked_confirmed", False))
            global_control_data["car_avoidance_locked_hit_frames"] = int(car_stats.get("car_locked_hit_frames", 0))
            global_control_data["car_avoidance_boundary_side_ready"] = bool(car_stats.get("car_boundary_side_ready", False))
            global_control_data["car_avoidance_boundary_path_active"] = bool(car_stats.get("car_boundary_path_active", False))
            global_control_data["car_avoidance_avoid_weight"] = float(car_stats.get("car_avoid_weight", 0.0))
            global_control_data["car_avoidance_event"] = str(car_stats.get("car_event", ""))
            global_control_data["car_avoidance_cycle_id"] = int(car_stats.get("car_cycle_id", 0))
            global_control_data["post_car_control_active"] = bool(car_stats.get("post_car_control_active", False))
            global_control_data["car_control_profile"] = str(car_stats.get("car_control_profile", "NORMAL"))
            global_control_data["car_avoidance_control_kp"] = car_stats.get("car_control_kp")
            global_control_data["car_avoidance_control_kd"] = car_stats.get("car_control_kd")
            global_control_data["car_avoidance_control_psi"] = car_stats.get("car_control_psi")
            global_control_data["car_avoidance_control_speed_estimate"] = car_stats.get("car_control_speed_estimate")
            y_fork_point = car_stats.get("y_fork_point")
            y_fork_rows_to_bottom = None
            if bool(y_fork_active) and y_fork_point is not None:
                try:
                    y_fork_rows_to_bottom = max(
                        0.0,
                        float(config.SEG_SIZE[1] - 1) - float(y_fork_point[1]),
                    )
                except Exception:
                    y_fork_rows_to_bottom = None
            global_control_data["sign_route_y_fork_active"] = bool(y_fork_active)
            global_control_data["sign_route_fork_point"] = y_fork_point if bool(y_fork_active) else None
            global_control_data["sign_route_fork_rows_to_bottom"] = y_fork_rows_to_bottom
            person_stop_active = update_person_stop_state(
                global_control_data,
                person_info,
                person_left_boundary_x,
                person_right_boundary_x,
                current_yolo_frame_id,
                person_car_on_left,
            )
            person_clear_line_x = global_control_data.get("person_clear_line_x")
            person_clear_line_side = str(global_control_data.get("person_clear_line_side", ""))
            person_stop_cutoff_y = global_control_data.get("person_stop_cutoff_y")
            person_road_center_x = global_control_data.get("person_road_center_x")

            actual_servo = global_control_data.get("actual_servo_pwm", config.SERVO_CENTER)
            actual_speed = global_control_data.get("target_speed", config.CONTROL_MIN_SPEED)
            sign_llm_stop_active = global_control_data.get("sign_llm_stop_active", False)
            sign_llm_waiting_result = global_control_data.get("sign_llm_waiting_result", False)
            debug_keyboard_stop_active = bool(global_control_data.get("debug_keyboard_stop_active", False))
            update_sign_route_after_seg(
                global_control_data,
                bool(y_fork_active),
                current_yolo_boxes=current_yolo_boxes,
                yolo_frame_id=current_yolo_frame_id,
            )
            route_state = str(global_control_data.get("sign_route_state", "IDLE"))
            route_choice = int(global_control_data.get("sign_route_choice", 0))

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
            if (
                bool(getattr(config, "CAR_AVOIDANCE_ENABLED", True)) and
                bool(getattr(config, "CAR_AVOIDANCE_DISTANCE_LINE_ENABLED", True))
            ):
                target_w, target_h = config.TARGET_RES
                crop_ratio = float(getattr(config, "SEG_INPUT_CROP_TOP_RATIO", 0.0))
                crop_ratio = max(0.0, min(0.95, crop_ratio))
                crop_y = float(target_h) * crop_ratio
                crop_h = max(1.0, float(target_h) - crop_y)
                seg_h = float(config.SEG_SIZE[1])
                switch_rows = max(0.0, float(getattr(
                    config,
                    "CAR_AVOIDANCE_SWITCH_MIN_BOTTOM_Y",
                    seg_h - 1.0,
                )))
                line_seg_y = float(np.clip((seg_h - 1.0) - switch_rows, 0.0, seg_h - 1.0))
                line_target_y = crop_y + (line_seg_y / seg_h) * crop_h
                line_y = int(round(line_target_y * rendered_img.shape[0] / float(target_h)))
                line_y = max(0, min(rendered_img.shape[0] - 1, line_y))
                cv2.line(
                    rendered_img,
                    (0, line_y),
                    (rendered_img.shape[1] - 1, line_y),
                    getattr(config, "CAR_AVOIDANCE_DISTANCE_LINE_COLOR", (255, 0, 0)),
                    max(1, int(getattr(config, "CAR_AVOIDANCE_DISTANCE_LINE_THICKNESS", 3))),
                    cv2.LINE_AA,
                )
            if bool(getattr(config, "PERSON_STOP_ENABLED", True)) and person_info is not None:
                target_w, target_h = config.TARGET_RES
                scale_y = rendered_img.shape[0] / float(target_h)
                default_cutoff_y = float(target_h) - float(getattr(config, "PERSON_STOP_TRIGGER_DIST", 0.0))
                cutoff_y = default_cutoff_y if person_stop_cutoff_y is None else float(person_stop_cutoff_y)
                line_y = int(round(float(cutoff_y) * scale_y))
                line_y = max(0, min(rendered_img.shape[0] - 1, line_y))
                cv2.line(
                    rendered_img,
                    (0, line_y),
                    (rendered_img.shape[1] - 1, line_y),
                    getattr(config, "PERSON_STOP_CUTOFF_LINE_COLOR", (0, 255, 255)),
                    max(1, int(getattr(config, "PERSON_STOP_CUTOFF_LINE_THICKNESS", 2))),
                    cv2.LINE_AA,
                )
                if bool(getattr(config, "PERSON_DEBUG_DRAW_RELEASE_LINE", False)):
                    scale_x = rendered_img.shape[1] / float(target_w)
                    release_xs = []
                    if person_clear_line_x is not None and person_clear_line_side in ("left", "right"):
                        release_xs.append(float(person_clear_line_x))
                    for release_x in release_xs:
                        line_x = int(round(float(release_x) * scale_x))
                        line_x = max(0, min(rendered_img.shape[1] - 1, line_x))
                        cv2.line(
                            rendered_img,
                            (line_x, 0),
                            (line_x, rendered_img.shape[0] - 1),
                            getattr(config, "PERSON_CLEAR_LINE_COLOR", (0, 255, 255)),
                            max(1, int(getattr(config, "PERSON_CLEAR_LINE_THICKNESS", 2))),
                            cv2.LINE_AA,
                        )
            draw_preview_status_panel(
                rendered_img,
                current_seg_fps=current_seg_fps,
                current_yolo_fps=current_yolo_fps,
                steer_signal=steer_signal,
                actual_servo=actual_servo,
                actual_speed=actual_speed,
                sign_llm_waiting_result=sign_llm_waiting_result,
                sign_llm_stop_active=sign_llm_stop_active,
                person_stop_active=person_stop_active,
                debug_keyboard_stop_active=debug_keyboard_stop_active,
                route_state=route_state,
                route_choice=route_choice,
                yolo_boxes=current_yolo_boxes,
            )

        if rendered_img is not None:
            with frame_lock:
                global_preview_frame = rendered_img

    if not bool(getattr(config, "SEG_PIPELINE_ENABLED", True)):
        fps_start_holder = [fps_start_time]
        while True:
            seg_item = seg_queues[int(worker_id) % len(seg_queues)].get()
            if seg_item is None:
                break
            blob_rgb_320, preview_frame = unpack_seg_item(seg_item)

            drain_sign_llm_results()
            with data_lock:
                current_yolo_boxes = [obj.copy() for obj in global_yolo_boxes]
                current_yolo_frame_id = int(global_yolo_frame_id)
                turn_intent = global_control_data.get("turn_intent", -1)
                sign_route_choice = int(global_control_data.get("sign_route_choice", 0))
                if sign_route_choice not in (-1, 1):
                    sign_route_choice = pending_fixed_route_choice(global_control_data, current_yolo_boxes)
                route_state = str(global_control_data.get("sign_route_state", "IDLE"))
                sign_route_pending = sign_route_pending_centerline_active(route_state, current_yolo_boxes)
                external_boundary_inset_x = 0.0
                external_boundary_side = "left"
                debug_keyboard_state = get_debug_drive_keyboard_state()
                drive_stop_active = (
                    bool(debug_keyboard_state.get("manual_stop_active", False)) or
                    bool(global_control_data.get("person_stop_active", False)) or
                    bool(global_control_data.get("sign_llm_stop_active", False))
                )
                debug_drive_active = not drive_stop_active
                car_avoidance_state_updates_enabled = car_avoidance_updates_enabled(global_control_data)

            try:
                steer_signal, rendered_img = seg.run(
                    blob_rgb_320,
                    current_yolo_boxes,
                    turn_intent,
                    fps_stats,
                    sign_route_choice=sign_route_choice,
                    external_boundary_inset_x=external_boundary_inset_x,
                    external_boundary_side=external_boundary_side,
                    sign_route_pending=sign_route_pending,
                    debug_drive_active=debug_drive_active,
                    car_avoidance_state_updates_enabled=car_avoidance_state_updates_enabled,
                    preview_frame=preview_frame,
                )
                publish_seg_result(
                    steer_signal,
                    rendered_img,
                    current_yolo_boxes,
                    fps_start_holder,
                    current_yolo_frame_id=current_yolo_frame_id,
                    preview_frame=preview_frame,
                    y_fork_active=bool(seg.last_branch_stats.get("y_fork_active", False)),
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
        render_counter = 0
        while True:
            t_get_start = time.perf_counter()
            item = mask_queue.get()
            t_after_get = time.perf_counter()
            if item is None:
                break

            (
                blob_rgb_320,
                preview_frame,
                mask,
                infer_s,
                total_start,
                current_yolo_boxes,
                current_yolo_frame_id,
                turn_intent,
                sign_route_choice,
                external_boundary_inset_x,
                external_boundary_side,
                sign_route_pending,
                debug_drive_active,
                car_avoidance_state_updates_enabled,
            ) = item

            if bool(getattr(config, "SEG_POSTPROCESS_REFRESH_YOLO", True)):
                drain_sign_llm_results()
                with data_lock:
                    latest_yolo_frame_id = int(global_yolo_frame_id)
                    if latest_yolo_frame_id >= int(current_yolo_frame_id):
                        current_yolo_boxes = [obj.copy() for obj in global_yolo_boxes]
                        current_yolo_frame_id = latest_yolo_frame_id
                    turn_intent = global_control_data.get("turn_intent", -1)
                    sign_route_choice = int(global_control_data.get("sign_route_choice", 0))
                    if sign_route_choice not in (-1, 1):
                        sign_route_choice = pending_fixed_route_choice(global_control_data, current_yolo_boxes)
                    route_state = str(global_control_data.get("sign_route_state", "IDLE"))
                    sign_route_pending = sign_route_pending_centerline_active(route_state, current_yolo_boxes)
                    debug_keyboard_state = get_debug_drive_keyboard_state()
                    drive_stop_active = (
                        bool(debug_keyboard_state.get("manual_stop_active", False)) or
                        bool(global_control_data.get("person_stop_active", False)) or
                        bool(global_control_data.get("sign_llm_stop_active", False))
                    )
                    debug_drive_active = not drive_stop_active
                    car_avoidance_state_updates_enabled = car_avoidance_updates_enabled(global_control_data)

            try:
                render_counter += 1
                render_interval = max(1, int(getattr(config, "SEG_RENDER_INTERVAL", 1)))
                render_enabled = (render_counter % render_interval) == 0
                t_post_start = time.perf_counter()
                steer_signal, rendered_img = seg.postprocess_mask(
                    blob_rgb_320,
                    mask,
                    current_yolo_boxes,
                    turn_intent,
                    fps_stats,
                    sign_route_choice=sign_route_choice,
                    infer_s=infer_s,
                    total_start=total_start,
                    preview_frame=preview_frame,
                    external_boundary_inset_x=external_boundary_inset_x,
                    external_boundary_side=external_boundary_side,
                    sign_route_pending=sign_route_pending,
                    debug_drive_active=debug_drive_active,
                    car_avoidance_state_updates_enabled=car_avoidance_state_updates_enabled,
                    render_enabled=render_enabled,
                )
                t_post_end = time.perf_counter()
                publish_seg_result(
                    steer_signal,
                    rendered_img,
                    current_yolo_boxes,
                    fps_start_holder,
                    current_yolo_frame_id=current_yolo_frame_id,
                    preview_frame=preview_frame,
                    y_fork_active=bool(seg.last_branch_stats.get("y_fork_active", False)),
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
        seg_item = seg_queues[int(worker_id) % len(seg_queues)].get()
        t_after_get = time.perf_counter()
        if seg_item is None:
            mask_queue.put(None)
            break
        blob_rgb_320, preview_frame = unpack_seg_item(seg_item)

        total_start = time.perf_counter()
        t_lock_start = time.perf_counter()
        drain_sign_llm_results()
        with data_lock:
            current_yolo_boxes = [obj.copy() for obj in global_yolo_boxes]
            current_yolo_frame_id = int(global_yolo_frame_id)
            turn_intent = global_control_data.get("turn_intent", -1)
            sign_route_choice = int(global_control_data.get("sign_route_choice", 0))
            if sign_route_choice not in (-1, 1):
                sign_route_choice = pending_fixed_route_choice(global_control_data, current_yolo_boxes)
            route_state = str(global_control_data.get("sign_route_state", "IDLE"))
            sign_route_pending = sign_route_pending_centerline_active(route_state, current_yolo_boxes)
            external_boundary_inset_x = 0.0
            external_boundary_side = "left"
            debug_keyboard_state = get_debug_drive_keyboard_state()
            drive_stop_active = (
                bool(debug_keyboard_state.get("manual_stop_active", False)) or
                bool(global_control_data.get("person_stop_active", False)) or
                bool(global_control_data.get("sign_llm_stop_active", False))
            )
            debug_drive_active = not drive_stop_active
            car_avoidance_state_updates_enabled = car_avoidance_updates_enabled(global_control_data)
        t_lock_end = time.perf_counter()

        try:
            t_infer_start = time.perf_counter()
            mask, infer_s = seg.infer_mask(blob_rgb_320)
            t_infer_end = time.perf_counter()
            infer_timing = getattr(seg, "last_infer_timing", {})
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
        mask_queue.put((
            blob_rgb_320,
            preview_frame,
            mask,
            infer_s,
            total_start,
            current_yolo_boxes,
            current_yolo_frame_id,
            turn_intent,
            sign_route_choice,
            external_boundary_inset_x,
            external_boundary_side,
            sign_route_pending,
            debug_drive_active,
            car_avoidance_state_updates_enabled,
        ))
        t_put_end = time.perf_counter()
        profile_log(
            "seg_infer_loop",
            "SegInferLoop",
            {
                "wait_input": t_after_get - t_get_start,
                "lock": t_lock_end - t_lock_start,
                "infer": t_infer_end - t_infer_start,
                "rknn": float(infer_timing.get("rknn", 0.0)),
                "decode": float(infer_timing.get("decode", 0.0)),
                "seg_candidates_count": float(infer_timing.get("candidates", 0)),
                "seg_kept_count": float(infer_timing.get("kept", 0)),
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
    2. 如果语义路牌采样停车或行人停车状态激活，则强制把速度打到 0
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
    last_output_servo = int(config.SERVO_CENTER)

    while True:
        drain_sign_llm_results()
        with data_lock:
            steer_signal = global_control_data.get("steer_signal", 0.0)
            car_avoidance_active = bool(global_control_data.get("car_avoidance_active", False))
            car_avoidance_state = str(global_control_data.get("car_avoidance_state", "FOLLOW_LANE"))
            car_avoidance_rows_to_bottom = global_control_data.get("car_avoidance_rows_to_bottom")
            car_avoidance_ground_distance_m = global_control_data.get("car_avoidance_ground_distance_m")
            car_avoidance_miss_frames = int(global_control_data.get("car_avoidance_miss_frames", 0))
            car_avoidance_clear_frames = int(global_control_data.get("car_avoidance_clear_frames", 0))
            car_avoidance_state_paused = bool(global_control_data.get("car_avoidance_state_paused", False))
            car_avoidance_boundary_side = str(global_control_data.get("car_avoidance_boundary_side", ""))
            car_avoidance_nearest_boundary_side = str(global_control_data.get("car_avoidance_nearest_boundary_side", ""))
            car_avoidance_boundary_x = global_control_data.get("car_avoidance_boundary_x")
            car_avoidance_boundary_error = global_control_data.get("car_avoidance_boundary_error")
            car_avoidance_left_boundary_error = global_control_data.get("car_avoidance_left_boundary_error")
            car_avoidance_left_boundary_x = global_control_data.get("car_avoidance_left_boundary_x")
            car_avoidance_control_path_error = global_control_data.get("car_avoidance_control_path_error")
            car_avoidance_boundary_inset_x = float(global_control_data.get("car_avoidance_boundary_inset_x", 0.0))
            car_avoidance_control_error_offset_x = float(global_control_data.get("car_avoidance_control_error_offset_x", 0.0))
            car_avoidance_detected_cars = int(global_control_data.get("car_avoidance_detected_cars", 0))
            car_avoidance_locked_confirmed = bool(global_control_data.get("car_avoidance_locked_confirmed", False))
            car_avoidance_locked_hit_frames = int(global_control_data.get("car_avoidance_locked_hit_frames", 0))
            car_avoidance_boundary_side_ready = bool(global_control_data.get("car_avoidance_boundary_side_ready", False))
            car_avoidance_boundary_path_active = bool(global_control_data.get("car_avoidance_boundary_path_active", False))
            car_avoidance_avoid_weight = float(global_control_data.get("car_avoidance_avoid_weight", 0.0))
            car_avoidance_event = str(global_control_data.get("car_avoidance_event", ""))
            car_avoidance_cycle_id = int(global_control_data.get("car_avoidance_cycle_id", 0))
            post_car_control_active = bool(global_control_data.get("post_car_control_active", False))
            car_control_profile = str(global_control_data.get("car_control_profile", "NORMAL"))
            car_avoidance_control_kp = global_control_data.get("car_avoidance_control_kp")
            car_avoidance_control_kd = global_control_data.get("car_avoidance_control_kd")
            car_avoidance_control_psi = global_control_data.get("car_avoidance_control_psi")
            car_avoidance_control_speed_estimate = global_control_data.get("car_avoidance_control_speed_estimate")
            latest_yolo_boxes_for_car_log = [obj.copy() for obj in global_yolo_boxes]
            sign_llm_stop_active = bool(global_control_data.get("sign_llm_stop_active", False))
            sign_llm_collecting = bool(global_control_data.get("sign_llm_collecting", False))
            sign_llm_frame_id = int(global_control_data.get("sign_llm_frame_id", -1))
            person_stop_active = bool(global_control_data.get("person_stop_active", False))
            person_dist_to_bottom = global_control_data.get("person_dist_to_bottom")
            person_ground_distance_m = global_control_data.get("person_ground_distance_m")
            person_stop_trigger_source = str(global_control_data.get("person_stop_trigger_source", ""))
            person_area = global_control_data.get("person_bottom_area")
            person_left_boundary_x = global_control_data.get("person_left_boundary_x")
            person_right_boundary_x = global_control_data.get("person_right_boundary_x")
            person_road_center_x = global_control_data.get("person_road_center_x")
            person_clear_line_x = global_control_data.get("person_clear_line_x")
            person_clear_line_side = str(global_control_data.get("person_clear_line_side", ""))
            person_stop_cutoff_y = global_control_data.get("person_stop_cutoff_y")
            person_bottom_center_x = global_control_data.get("person_bottom_center_x")
            person_bottom_right_x = global_control_data.get("person_bottom_right_x")
            person_lock_bottom_y = global_control_data.get("person_lock_bottom_y")
            person_clear_frames = int(global_control_data.get("person_clear_frames", 0))
            person_stop_event = str(global_control_data.get("person_stop_event", ""))
            person_move_dx = float(global_control_data.get("person_move_dx", 0.0))
            person_abs_move_dx = float(global_control_data.get("person_abs_move_dx", 0.0))
            person_min_move_dx = float(global_control_data.get("person_min_move_dx", 0.0))
            person_move_direction = int(global_control_data.get("person_move_direction", 0))
            person_current_direction = int(global_control_data.get("person_current_direction", 0))
            person_release_direction = int(global_control_data.get("person_release_direction", 0))
            person_line_reached = bool(global_control_data.get("person_line_reached", False))
            person_movement_confirmed = bool(global_control_data.get("person_movement_confirmed", False))
            person_near_bottom = bool(global_control_data.get("person_near_bottom", False))
            person_enough_area = bool(global_control_data.get("person_enough_area", False))
        debug_keyboard_state = get_debug_drive_keyboard_state()
        debug_keyboard_stop_active = bool(debug_keyboard_state.get("manual_stop_active", False))

        try:
            if sign_llm_collecting:
                with data_lock:
                    started_at = global_control_data.get("sign_llm_started_at")
                    collect_timeout = float(getattr(config, "SIGN_LLM_COLLECT_TIMEOUT", 3.0))
                    force_submit = (
                        started_at is not None and
                        collect_timeout > 0.0 and
                        time.monotonic() - float(started_at) >= collect_timeout
                    )
                    submitted = maybe_submit_sign_llm_job(global_control_data, sign_llm_frame_id, force=force_submit)
                    sign_llm_collecting = bool(global_control_data.get("sign_llm_collecting", False))
                    sign_llm_stop_active = bool(global_control_data.get("sign_llm_stop_active", False))
                    if submitted:
                        sign_llm_collecting = False
                        sign_llm_stop_active = True
                        throttled_log(
                            "sign_llm_submit_timeout",
                            f"语义路牌OCR采集提交千帆: samples={len(global_control_data.get('sign_llm_samples', []))} force={int(force_submit)}",
                            state=("submit", sign_llm_frame_id, int(force_submit)),
                            min_interval=0.0,
                        )

            if np.isnan(steer_signal) or np.isinf(steer_signal):
                steer_signal = 0.0

            # 转向量幅度越大，目标速度越低，避免高速出弯失控。
            speed_ceiling = (
                int(round(float(getattr(config, "POST_CAR_TARGET_SPEED", config.CONTROL_MAX_SPEED))))
                if post_car_control_active else
                int(config.CONTROL_MAX_SPEED)
            )
            dynamic_target_speed = speed_ceiling - int(
                abs(steer_signal) * config.STEER_SIGNAL_SPEED_GAIN
            )
            dynamic_target_speed = int(
                max(
                    config.CONTROL_MIN_SPEED,
                    min(speed_ceiling, dynamic_target_speed),
                )
            )
            target_speed = dynamic_target_speed
            if car_avoidance_active:
                target_speed = min(
                    int(target_speed),
                    int(round(float(getattr(config, "CAR_AVOIDANCE_TARGET_SPEED", target_speed)))),
                )

            stop_ready = False
            if sign_llm_stop_active:
                target_speed = 0
            if person_stop_active:
                target_speed = 0
            if debug_keyboard_stop_active:
                target_speed = 0

            limit_applied = False
            if bool(getattr(config, "CONTROL_SPEED_SMOOTH_ENABLED", True)):
                if target_speed < 0:
                    last_output_speed = int(target_speed)
                elif target_speed == 0:
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

            pwm_gain = float(config.STEER_SIGNAL_PWM_GAIN)
            control_mode = str(getattr(config, "STEER_CONTROL_MODE", "weighted_slope")).lower()
            if control_mode == "stanley_band":
                pwm_gain = float(getattr(config, "STANLEY_PWM_GAIN", 0.012))
            elif control_mode == "control_c":
                pwm_gain = float(getattr(config, "CONTROL_C_PWM_GAIN", 12.0))
            raw_pwm = (
                config.SERVO_CENTER
                - steer_signal * pwm_gain
            )
            target_servo_pwm = int(max(config.SERVO_MIN, min(config.SERVO_MAX, raw_pwm)))
            servo_pwm = target_servo_pwm
            if bool(getattr(config, "SERVO_OUTPUT_FILTER_ENABLED", False)):
                alpha = float(getattr(config, "SERVO_OUTPUT_EMA_ALPHA", 0.0))
                alpha = float(np.clip(alpha, 0.0, 0.98))
                filtered_servo = (
                    alpha * float(last_output_servo) +
                    (1.0 - alpha) * float(target_servo_pwm)
                )

                deadband = max(0, int(getattr(config, "SERVO_OUTPUT_DEADBAND_PWM", 0)))
                delta = float(filtered_servo) - float(last_output_servo)
                if abs(delta) <= float(deadband):
                    filtered_servo = float(last_output_servo)

                max_step = max(0, int(getattr(config, "SERVO_OUTPUT_MAX_STEP", 0)))
                if max_step > 0:
                    step_delta = float(filtered_servo) - float(last_output_servo)
                    step_delta = float(np.clip(step_delta, -float(max_step), float(max_step)))
                    filtered_servo = float(last_output_servo) + step_delta

                servo_pwm = int(round(max(config.SERVO_MIN, min(config.SERVO_MAX, filtered_servo))))
            last_output_servo = int(servo_pwm)
        except:
            target_speed = config.CONTROL_MIN_SPEED
            servo_pwm = config.SERVO_CENTER
            last_output_servo = int(servo_pwm)
            dynamic_target_speed = target_speed
            stop_ready = False
            limit_applied = False
            steer_signal = 0.0
            car_avoidance_active = False
            car_avoidance_state = "FOLLOW_LANE"
            car_avoidance_rows_to_bottom = None
            car_avoidance_miss_frames = 0
            car_avoidance_clear_frames = 0
            car_avoidance_state_paused = False
            car_avoidance_boundary_side = ""
            car_avoidance_nearest_boundary_side = ""
            car_avoidance_boundary_x = None
            car_avoidance_boundary_error = None
            car_avoidance_left_boundary_error = None
            car_avoidance_left_boundary_x = None
            car_avoidance_control_path_error = None
            car_avoidance_boundary_inset_x = 0.0
            car_avoidance_control_error_offset_x = 0.0
            car_avoidance_detected_cars = 0
            car_avoidance_locked_confirmed = False
            car_avoidance_locked_hit_frames = 0
            car_avoidance_boundary_side_ready = False
            car_avoidance_boundary_path_active = False
            car_avoidance_avoid_weight = 0.0
            car_avoidance_event = ""
            car_avoidance_cycle_id = 0
            post_car_control_active = False
            car_control_profile = "NORMAL"
            car_avoidance_control_kp = None
            car_avoidance_control_kd = None
            car_avoidance_control_psi = None
            car_avoidance_control_speed_estimate = None
            person_dist_to_bottom = None
            person_ground_distance_m = None
            person_stop_trigger_source = ""
            person_area = None
            person_left_boundary_x = None
            person_right_boundary_x = None
            person_road_center_x = None
            person_clear_line_x = None
            person_bottom_center_x = None
            person_bottom_right_x = None
            person_lock_bottom_y = None
            person_stop_cutoff_y = None
            person_clear_frames = 0
            person_stop_event = ""
            person_move_dx = 0.0
            person_abs_move_dx = 0.0
            person_min_move_dx = 0.0
            person_move_direction = 0
            person_current_direction = 0
            person_release_direction = 0
            person_line_reached = False
            person_movement_confirmed = False
            person_near_bottom = False
            person_enough_area = False
            debug_keyboard_state = get_debug_drive_keyboard_state()
            debug_keyboard_stop_active = bool(debug_keyboard_state.get("manual_stop_active", False))
            if sign_llm_stop_active or person_stop_active:
                target_speed = 0
            if debug_keyboard_stop_active:
                target_speed = 0

        with data_lock:
            global_control_data["person_stop_active"] = person_stop_active
            global_control_data["actual_servo_pwm"] = servo_pwm
            global_control_data["target_speed"] = target_speed
            global_control_data["debug_keyboard_enabled"] = bool(debug_keyboard_state.get("enabled", False))
            global_control_data["debug_keyboard_stop_active"] = bool(debug_keyboard_stop_active)
            global_control_data["debug_keyboard_message"] = str(debug_keyboard_state.get("message", ""))

        person_area_text = "无" if 'person_area' not in locals() or person_area is None else f"{float(person_area):.0f}"
        person_dist_text = "无" if person_dist_to_bottom is None else f"{float(person_dist_to_bottom):.1f}"
        person_z_text = "无" if person_ground_distance_m is None else f"{float(person_ground_distance_m):.2f}m"
        person_trigger_text = str(person_stop_trigger_source or "无")
        left_boundary_text = "无" if person_left_boundary_x is None else f"{float(person_left_boundary_x):.1f}"
        right_boundary_text = "无" if person_right_boundary_x is None else f"{float(person_right_boundary_x):.1f}"
        road_center_text = "无" if person_road_center_x is None else f"{float(person_road_center_x):.1f}"
        clear_line_text = "无" if person_clear_line_x is None else f"{float(person_clear_line_x):.1f}"
        lock_bottom_text = "无" if person_lock_bottom_y is None else f"{float(person_lock_bottom_y):.1f}"
        cutoff_line_text = "无" if person_stop_cutoff_y is None else f"{float(person_stop_cutoff_y):.1f}"
        center_x_text = "无" if person_bottom_center_x is None else f"{float(person_bottom_center_x):.1f}"
        right_x_text = "无" if person_bottom_right_x is None else f"{float(person_bottom_right_x):.1f}"
        stop_text = "是" if person_stop_active else "否"
        person_dx_text = f"{float(person_move_dx):+.1f}"
        person_abs_dx_text = f"{float(person_abs_move_dx):.1f}"
        person_min_dx_text = f"{float(person_min_move_dx):.1f}"
        person_dir_text = "右" if int(person_current_direction) > 0 else ("左" if int(person_current_direction) < 0 else "无")
        person_release_dir_text = "右" if int(person_release_direction) > 0 else ("左" if int(person_release_direction) < 0 else "无")
        person_locked_dir_text = "右" if int(person_move_direction) > 0 else ("左" if int(person_move_direction) < 0 else "无")
        person_line_text = "1" if bool(person_line_reached) else "0"
        person_move_ok_text = "1" if bool(person_movement_confirmed) else "0"
        person_near_text = "1" if bool(person_near_bottom) else "0"
        person_area_ok_text = "1" if bool(person_enough_area) else "0"
        if car_avoidance_ground_distance_m is None:
            car_candidates = [
                obj for obj in latest_yolo_boxes_for_car_log
                if str(obj.get("class_name", "")) == "car"
            ]
            if car_candidates:
                nearest_car = min(
                    car_candidates,
                    key=lambda obj: float(obj.get("ground_distance_m", 1e9))
                    if obj.get("ground_distance_m") is not None else 1e9,
                )
                car_avoidance_ground_distance_m = nearest_car.get("ground_distance_m")
        car_rows_text = "无" if car_avoidance_rows_to_bottom is None else f"{float(car_avoidance_rows_to_bottom):.1f}"
        car_z_text = "无" if car_avoidance_ground_distance_m is None else f"{float(car_avoidance_ground_distance_m):.2f}m"
        car_cut_trigger_z = float(getattr(config, "CAR_AVOIDANCE_SWITCH_DISTANCE_M", 0.0))
        car_side_trigger_z = float(getattr(config, "CAR_AVOIDANCE_SIDE_DECISION_DISTANCE_M", car_cut_trigger_z))
        car_cut_trigger_rows = max(0.0, float(getattr(config, "CAR_AVOIDANCE_SWITCH_MIN_BOTTOM_Y", 160.0)))
        car_side_trigger_rows = max(0.0, float(getattr(
            config,
            "CAR_AVOIDANCE_SIDE_DECISION_MIN_BOTTOM_Y",
            car_cut_trigger_rows,
        )))
        car_cut_trigger_z_text = (
            f"{car_cut_trigger_z:.2f}m/{car_cut_trigger_rows:.0f}行"
            if car_cut_trigger_z > 0.0 else
            f"关闭/{car_cut_trigger_rows:.0f}行"
        )
        car_side_trigger_z_text = (
            f"{car_side_trigger_z:.2f}m/{car_side_trigger_rows:.0f}行"
            if car_side_trigger_z > 0.0 else
            f"关闭/{car_side_trigger_rows:.0f}行"
        )
        car_cut_ready_z = (
            car_cut_trigger_z > 0.0 and
            car_avoidance_ground_distance_m is not None and
            float(car_avoidance_ground_distance_m) <= car_cut_trigger_z
        )
        car_side_ready_z = (
            car_side_trigger_z > 0.0 and
            car_avoidance_ground_distance_m is not None and
            float(car_avoidance_ground_distance_m) <= car_side_trigger_z
        )
        car_cut_ready_rows = (
            car_avoidance_rows_to_bottom is not None and
            float(car_avoidance_rows_to_bottom) <= car_cut_trigger_rows
        )
        car_side_ready_rows = (
            car_avoidance_rows_to_bottom is not None and
            float(car_avoidance_rows_to_bottom) <= car_side_trigger_rows
        )
        car_cut_ready_z_text = f"{int(car_cut_ready_z or car_cut_ready_rows)}(z={int(car_cut_ready_z)},r={int(car_cut_ready_rows)})"
        car_side_ready_z_text = f"{int(car_side_ready_z or car_side_ready_rows)}(z={int(car_side_ready_z)},r={int(car_side_ready_rows)})"

        def _car_trigger_source(z_ready, row_ready):
            if z_ready and row_ready:
                return "距离+行距"
            if z_ready:
                return "距离"
            if row_ready:
                return "行距"
            return "未触发"

        car_side_ready_source_text = (
            f"{_car_trigger_source(car_side_ready_z, car_side_ready_rows)}"
            f"(距离={int(car_side_ready_z)},行距={int(car_side_ready_rows)})"
        )
        car_cut_ready_source_text = (
            f"{_car_trigger_source(car_cut_ready_z, car_cut_ready_rows)}"
            f"(距离={int(car_cut_ready_z)},行距={int(car_cut_ready_rows)})"
        )
        car_miss_text = f"{int(car_avoidance_miss_frames)}"
        car_clear_text = f"{int(car_avoidance_clear_frames)}"
        car_pause_text = "1" if car_avoidance_state_paused else "0"
        car_locked_text = "1" if car_avoidance_locked_confirmed else "0"
        car_path_text = "1" if car_avoidance_boundary_path_active else "0"
        car_side_ready_text = "1" if car_avoidance_boundary_side_ready else "0"
        car_side_text = str(car_avoidance_boundary_side or "").upper() or "无"
        car_boundary_x_text = "无" if car_avoidance_boundary_x is None else f"{float(car_avoidance_boundary_x):.1f}"
        car_boundary_error_text = "无" if car_avoidance_boundary_error is None else f"{float(car_avoidance_boundary_error):.1f}"
        car_control_error_text = "无" if car_avoidance_control_path_error is None else f"{float(car_avoidance_control_path_error):.1f}"
        car_boundary_inset_text = f"{float(car_avoidance_boundary_inset_x):.1f}"
        car_control_error_offset_text = f"{float(car_avoidance_control_error_offset_x):.1f}"
        car_kp_text = "无" if car_avoidance_control_kp is None else f"{float(car_avoidance_control_kp):.2f}"
        car_kd_text = "无" if car_avoidance_control_kd is None else f"{float(car_avoidance_control_kd):.2f}"
        car_psi_text = "无" if car_avoidance_control_psi is None else f"{float(car_avoidance_control_psi):.2f}"
        car_v_text = "无" if car_avoidance_control_speed_estimate is None else f"{float(car_avoidance_control_speed_estimate):.0f}"

        def _car_side_cn(side):
            side = str(side or "").lower()
            if side == "left":
                return "左侧"
            if side == "right":
                return "右侧"
            return "未知侧"

        car_avoidance_detail_log_enabled = bool(getattr(config, "CAR_AVOIDANCE_DETAIL_LOG_ENABLED", False))
        car_avoidance_visible_or_event = (
            bool(car_avoidance_active) or
            bool(car_avoidance_event) or
            bool(car_avoidance_state_paused) or
            str(car_avoidance_state) != "FOLLOW_LANE" or
            int(car_avoidance_detected_cars) > 0 or
            car_avoidance_ground_distance_m is not None
        )
        if car_avoidance_detail_log_enabled and car_avoidance_visible_or_event:
            throttled_log(
                "car_avoid_detail",
                f">>> 避车#{car_avoidance_cycle_id}(B): event={car_avoidance_event or '-'} state={car_avoidance_state} "
                f"车底距底={car_rows_text}行 car_z={car_z_text} "
                f"判左右阈值={car_side_trigger_z_text} 触发={car_side_ready_source_text} "
                f"绕车阈值={car_cut_trigger_z_text} 触发={car_cut_ready_source_text} "
                f"det={int(car_avoidance_detected_cars)} locked={car_locked_text} hits={int(car_avoidance_locked_hit_frames)} "
                f"miss={car_miss_text} clear={car_clear_text} pause={car_pause_text} path={car_path_text} "
                f"inset={car_boundary_inset_text} boundary_side_ready={car_side_ready_text} ctrl_off={car_control_error_offset_text} weight={float(car_avoidance_avoid_weight):.2f} "
                f"side={car_side_text} boundary_x={car_boundary_x_text} boundary_e={car_boundary_error_text} "
                f"ctrl_e={car_control_error_text} "
                f"B=({car_kp_text},{car_kd_text},{car_psi_text}) v={car_v_text}",
                state=(
                    car_avoidance_cycle_id,
                    car_avoidance_event,
                    car_avoidance_state,
                    car_rows_text,
                    car_z_text,
                    car_side_trigger_z_text,
                    car_side_ready_z_text,
                    car_cut_trigger_z_text,
                    car_cut_ready_z_text,
                    int(car_avoidance_detected_cars),
                    car_locked_text,
                    int(car_avoidance_locked_hit_frames),
                    car_miss_text,
                    car_clear_text,
                    car_pause_text,
                    car_path_text,
                    car_side_ready_text,
                    car_boundary_inset_text,
                    car_control_error_offset_text,
                    int(float(car_avoidance_avoid_weight) * 100),
                    car_side_text,
                    car_boundary_x_text,
                    car_boundary_error_text,
                    car_control_error_text,
                    car_kp_text,
                    car_kd_text,
                    car_psi_text,
                    car_v_text,
                ),
                min_interval=float(getattr(config, "LOG_INTERVAL_CAR_AVOIDANCE_DETAIL", 1.0)),
            )
        if person_stop_event == "stop":
            throttled_log(
                "person_stop_event",
                f">>> 行人: 停 area={person_area_text} dist={person_dist_text} "
                f"person_z={person_z_text} 触发={person_trigger_text} "
                f"dx={person_dx_text}/帧 min={person_min_dx_text}",
                state=("stop", person_area_text, person_dist_text, person_z_text, person_trigger_text, person_dx_text, person_min_dx_text),
                min_interval=0.0,
            )
        elif person_stop_event == "release_line":
            line_side_text = "右侧" if person_clear_line_side == "right" else "左侧"
            throttled_log(
                "person_stop_event",
                f">>> 行人: 过线放行({line_side_text}) cutoff_y={cutoff_line_text} "
                f"dx={person_dx_text}/帧 clear={person_clear_frames}",
                state=("release_line", person_clear_line_side, clear_line_text, person_dx_text, person_clear_frames),
                min_interval=0.0,
            )
        elif person_stop_event == "release_missing":
            throttled_log(
                "person_stop_event",
                f">>> 行人: 漏检{float(getattr(config, 'PERSON_STOP_MISSING_TIMEOUT_SECONDS', 1.0)):.1f}秒放行",
                state=("release_missing",),
                min_interval=0.0,
            )
        elif person_stop_event == "release_timeout":
            throttled_log(
                "person_stop_event",
                ">>> 行人: 等待超时，走",
                state=("release_timeout",),
                min_interval=0.0,
            )
        elif person_stop_event == "disabled_release":
            throttled_log(
                "person_stop_event",
                ">>> 行人: 开关关闭，放行",
                state=("disabled_release",),
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
            yolo_frame_counter = 0
            yolo_dispatch_idx = 0

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
                with data_lock:
                    post_sign_phase = bool(global_control_data.get("post_sign_phase", False))
                active_seg_workers = len(seg_queues) if post_sign_phase else seg_initial_worker_count
                active_seg_workers = max(1, min(active_seg_workers, len(seg_queues)))
                dropped_seg = 0
                for seg_queue in seg_queues[:active_seg_workers]:
                    if seg_queue.full():
                        try:
                            seg_queue.get_nowait()
                            dropped_seg += 1
                        except:
                            pass
                    seg_queue.put((seg_blob, vis_img_large))
                t_seg_put_end = time.perf_counter()

                # 检测分支交给 detector 内部按模型契约做 letterbox。
                # 同时保留一份 TARGET_RES 大图，供:
                # - 检测框绘制
                # - OCR 整图 det + rec
                # - OCR 结果回写后的页面可视化
                # 注意: 检测框最终也会映射到 TARGET_RES 坐标系，所以这里要保持一致。
                yolo_frame_counter += 1
                yolo_interval = max(1, int(getattr(config, "YOLO_PRODUCER_FRAME_INTERVAL", 1)))
                submit_yolo = (yolo_frame_counter % yolo_interval) == 0
                if submit_yolo:
                    det_img = vis_img_large
                t_yolo_end = time.perf_counter()

                dropped_yolo = 0
                if submit_yolo:
                    active_yolo_workers = len(yolo_queues)
                    if post_sign_phase:
                        active_yolo_workers = max(
                            0,
                            min(
                                int(getattr(config, "YOLO_ACTIVE_WORKERS_AFTER_SIGN", 1)),
                                len(yolo_queues),
                            ),
                        )
                    if active_yolo_workers > 0:
                        yolo_queue = yolo_queues[yolo_dispatch_idx % active_yolo_workers]
                        yolo_dispatch_idx += 1
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
def _debug_control_param_values():
    """返回网页调试面板当前使用的 B 控制参数和速度."""
    return {
        "kp": float(getattr(config, "STANLEY_LATERAL_GAIN", 0.0)),
        "kd": float(getattr(config, "STANLEY_LATERAL_D_GAIN", 0.0)),
        "psi": float(getattr(config, "STANLEY_HEADING_GAIN", 0.0)),
        "speed": int(round(float(getattr(config, "CONTROL_MAX_SPEED", 0.0)))),
        "person_stop_enabled": bool(getattr(config, "PERSON_STOP_ENABLED", True)),
        "car_avoidance_enabled": bool(getattr(config, "CAR_AVOIDANCE_ENABLED", True)),
    }


def _apply_debug_control_params(payload):
    """校验并应用网页调试面板提交的参数."""
    if not isinstance(payload, dict):
        raise ValueError("参数格式错误")

    values = payload.get("values", payload)
    if not isinstance(values, dict):
        raise ValueError("参数格式错误")

    limits = {
        "kp": (0.0, 3.0),
        "kd": (0.0, 1.0),
        "psi": (0.0, 5.0),
        "speed": (0.0, 100.0),
    }
    parsed = {}
    for name, (lower, upper) in limits.items():
        if name not in values:
            continue
        try:
            value = float(values[name])
        except (TypeError, ValueError):
            raise ValueError(f"{name} 不是数字")
        if not np.isfinite(value):
            raise ValueError(f"{name} 不是有效数字")
        parsed[name] = max(lower, min(upper, value))

    bool_updates = {}
    for name in ("person_stop_enabled", "car_avoidance_enabled"):
        if name not in values:
            continue
        value = values[name]
        if isinstance(value, str):
            value = value.strip().lower() in ("1", "true", "yes", "on", "开", "开启")
        else:
            value = bool(value)
        bool_updates[name] = value

    if not parsed and not bool_updates:
        raise ValueError("没有可更新的参数")

    if "kp" in parsed:
        config.STANLEY_LATERAL_GAIN = float(parsed["kp"])
    if "kd" in parsed:
        config.STANLEY_LATERAL_D_GAIN = float(parsed["kd"])
    if "psi" in parsed:
        config.STANLEY_HEADING_GAIN = float(parsed["psi"])
    if "speed" in parsed:
        speed = int(round(parsed["speed"]))
        config.CONTROL_MIN_SPEED = speed
        config.CONTROL_MAX_SPEED = speed
        # B 算法的速度估计同步更新，否则调速后横向纠偏比例会不一致。
        config.STANLEY_SPEED_ESTIMATE = float(speed)
    if "person_stop_enabled" in bool_updates:
        config.PERSON_STOP_ENABLED = bool(bool_updates["person_stop_enabled"])
    if "car_avoidance_enabled" in bool_updates:
        config.CAR_AVOIDANCE_ENABLED = bool(bool_updates["car_avoidance_enabled"])

    return _debug_control_param_values()


@app.route('/')
def index():
    return render_template_string(preview_index_html())


@app.route('/debug_drive/start', methods=['POST'])
def debug_drive_start():
    state = set_debug_drive_manual_stop(False, key="b", label="发车")
    return {
        "ok": True,
        "manual_stop_active": bool(state.get("manual_stop_active", False)),
        "message": str(state.get("message", "")),
    }


@app.route('/debug_drive/stop', methods=['POST'])
def debug_drive_stop():
    state = set_debug_drive_manual_stop(True, key="e", label="停车")
    return {
        "ok": True,
        "manual_stop_active": bool(state.get("manual_stop_active", False)),
        "message": str(state.get("message", "")),
    }


@app.route('/debug_control_params', methods=['GET', 'POST'])
def debug_control_params():
    if request.method == 'GET':
        return {"ok": True, "values": _debug_control_param_values()}

    payload = request.get_json(silent=True)
    try:
        values = _apply_debug_control_params(payload)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "values": _debug_control_param_values()}, 400
    return {"ok": True, "values": values}


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
            chunk = encode_mjpeg_frame(current_frame)
            t_encode_end = time.perf_counter()
            if chunk is None:
                continue

            yield chunk
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
    start_debug_drive_keyboard_control()
    print_preview_url()
    print_runtime_config_summary()

    threading.Thread(target=ai_producer_thread, daemon=True).start()
    threading.Thread(target=serial_control_thread, daemon=True).start()
    time.sleep(config.STARTUP_SHARED_THREAD_SLEEP)

    for worker_id, core_id in enumerate(seg_core_ids):
        threading.Thread(target=seg_worker, args=(core_id, worker_id), daemon=True).start()
        time.sleep(config.STARTUP_SEG_THREAD_SLEEP)

    if sign_route_uses_llm():
        Process(target=sign_llm_worker, daemon=True).start()
        threading.Thread(target=ocr_worker, daemon=True).start()
    else:
        print("路牌 OCR/千帆线程未启动: 当前为固定序列或未启用 LLM 路牌模式", flush=True)
    for worker_id, core_id in enumerate(yolo_core_ids):
        threading.Thread(target=yolo_worker, args=(core_id, worker_id), daemon=True).start()

    app.run(host=config.FLASK_HOST, port=config.STREAM_PORT, threaded=True)

# modules/debug_tools.py
"""网页预览、终端日志和画面叠加等调试工具."""

import atexit
import select
import socket
import sys
import termios
import threading
import time
import tty

import cv2
import numpy as np

import config


class DebugDriveKeyboardControl:
    """用终端键盘做调试发车/停车控制."""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None
        self._fd = None
        self._old_termios = None
        self.enabled = bool(getattr(config, "DEBUG_DRIVE_CONTROL_ENABLED", True))
        self.listening = False
        self.manual_stop_active = bool(getattr(
            config,
            "DEBUG_DRIVE_INITIAL_STOPPED",
            getattr(config, "DEBUG_KEYBOARD_DRIVE_INITIAL_STOPPED", True),
        ))
        self.last_key = ""
        self.last_event_at = None
        self.message = "未启动"

    def start(self):
        if not bool(getattr(config, "DEBUG_DRIVE_CONTROL_ENABLED", True)):
            with self._lock:
                self.enabled = False
                self.listening = False
                self.manual_stop_active = False
                self.message = "调试发车/停车已关闭"
            return self

        with self._lock:
            self.enabled = True
            if not self.message or self.message == "未启动":
                self.message = "等待B发车" if self.manual_stop_active else "运行中"

        if not bool(getattr(config, "DEBUG_KEYBOARD_DRIVE_ENABLED", True)):
            with self._lock:
                self.listening = False
                self.message = "网页B/E控制，终端监听关闭"
            print("调试发车/停车已启用: 网页预览页面按 B 发车，按 E 停车；终端键盘监听关闭", flush=True)
            return self

        with self._lock:
            if self._thread is not None:
                return self

        if not sys.stdin or not sys.stdin.isatty():
            with self._lock:
                self.enabled = True
                self.listening = False
                self.message = "stdin不是TTY，键盘发车/停车未启用"
            print("调试键盘控制未启用: stdin不是TTY", flush=True)
            return self

        try:
            self._fd = sys.stdin.fileno()
            self._old_termios = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            atexit.register(self.restore_terminal)
        except Exception as e:
            with self._lock:
                self.enabled = True
                self.listening = False
                self.message = f"TTY初始化失败: {e}"
            print(f"调试键盘控制启动失败: {e}", flush=True)
            return self

        with self._lock:
            self.enabled = True
            self.listening = True
            self.message = "等待B发车" if self.manual_stop_active else "运行中"

        self._thread = threading.Thread(
            target=self._listen_loop,
            name="debug-drive-keyboard",
            daemon=True,
        )
        self._thread.start()

        start_key = str(getattr(config, "DEBUG_KEYBOARD_DRIVE_START_KEY", "b")).lower()
        stop_key = str(getattr(config, "DEBUG_KEYBOARD_DRIVE_STOP_KEY", "e")).lower()
        initial = "停车" if self.manual_stop_active else "运行"
        print(
            f"调试键盘控制已启动: 按 {start_key.upper()} 发车，按 {stop_key.upper()} 停车，当前={initial}",
            flush=True,
        )
        return self

    def restore_terminal(self):
        if self._fd is None or self._old_termios is None:
            return
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_termios)
        except Exception:
            pass
        self._old_termios = None

    def _listen_loop(self):
        start_key = str(getattr(config, "DEBUG_KEYBOARD_DRIVE_START_KEY", "b")).lower()
        stop_key = str(getattr(config, "DEBUG_KEYBOARD_DRIVE_STOP_KEY", "e")).lower()
        poll_interval = float(getattr(config, "DEBUG_KEYBOARD_DRIVE_POLL_INTERVAL", 0.05))

        while True:
            try:
                readable, _, _ = select.select([sys.stdin], [], [], poll_interval)
                if not readable:
                    continue
                ch = sys.stdin.read(1)
            except Exception as e:
                with self._lock:
                    self.enabled = False
                    self.listening = False
                    self.manual_stop_active = False
                    self.message = f"监听异常: {e}"
                print(f"调试键盘控制停止: {e}", flush=True)
                return

            key = str(ch).lower()
            if key == start_key:
                self.set_manual_stop(False, key, "发车")
            elif key == stop_key:
                self.set_manual_stop(True, key, "停车")

    def set_manual_stop(self, active, key="", label=None):
        if label is None:
            label = "停车" if active else "发车"
        with self._lock:
            changed = bool(self.manual_stop_active) != bool(active)
            self.manual_stop_active = bool(active)
            self.last_key = str(key)
            self.last_event_at = time.time()
            self.message = str(label)
        if changed:
            print(f">>> 调试键盘: {label}", flush=True)

    def snapshot(self):
        with self._lock:
            return {
                "enabled": bool(self.enabled),
                "listening": bool(self.listening),
                "manual_stop_active": bool(self.enabled and self.manual_stop_active),
                "last_key": self.last_key,
                "last_event_at": self.last_event_at,
                "message": self.message,
            }

    def is_manual_stop_active(self):
        with self._lock:
            return bool(self.enabled and self.manual_stop_active)


_debug_drive_keyboard_control = DebugDriveKeyboardControl()


def start_debug_drive_keyboard_control():
    """启动调试键盘发车/停车监听线程."""
    return _debug_drive_keyboard_control.start()


def get_debug_drive_keyboard_state():
    """返回调试键盘控制状态快照."""
    return _debug_drive_keyboard_control.snapshot()


def set_debug_drive_manual_stop(active, key="", label=None):
    """设置调试发车/停车状态，供网页按钮或键盘事件调用."""
    _debug_drive_keyboard_control.set_manual_stop(active, key=key, label=label)
    return _debug_drive_keyboard_control.snapshot()


class DebugLogger:
    """线程安全的节流日志和性能日志."""

    def __init__(self):
        self._lock = threading.Lock()
        self._log_cache = {}
        self._profile_cache = {}

    def throttled_log(self, key, message, state=None, min_interval=None):
        now = time.time()
        if min_interval is None:
            min_interval = config.LOG_INTERVAL_DEFAULT
        with self._lock:
            prev = self._log_cache.get(key)
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
                self._log_cache[key] = {"time": now, "state": state}

    def log_once(self, key, message):
        with self._lock:
            if key in self._log_cache:
                return
            print(message, flush=True)
            self._log_cache[key] = {"time": time.time(), "state": "__once__"}

    def profile_log(self, key, label, metrics, min_interval=None):
        if not bool(getattr(config, "MAIN_PROFILE_LOG_ENABLED", False)):
            return

        now = time.time()
        if min_interval is None:
            min_interval = float(getattr(config, "MAIN_PROFILE_LOG_INTERVAL", 2.0))

        with self._lock:
            item = self._profile_cache.get(key)
            if item is None:
                ema = {name: float(value) for name, value in metrics.items()}
                item = {"ema": ema, "time": 0.0}
                self._profile_cache[key] = item
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


class SegProfileLogger:
    """Seg 后处理链路耗时 EMA 与终端打印."""

    def __init__(self):
        self.profile_ema = None
        self.profile_last_log = 0.0

    def add(self, infer_s, preprocess_s, search_s, fit_s, render_s, total_s, queue_wait_s=None):
        if not bool(getattr(config, "SEG_PROFILE_LOG_ENABLED", False)):
            return

        current = {
            "infer": float(infer_s),
            "prep": float(preprocess_s),
            "search": float(search_s),
            "fit": float(fit_s),
            "render": float(render_s),
            "total": float(total_s),
            "queue_wait": float(queue_wait_s) if queue_wait_s is not None else 0.0,
        }
        stage_sum = (
            current["infer"] + current["prep"] + current["search"] + current["fit"] + current["render"]
        )
        current["overhead"] = float(current["total"] - stage_sum)

        if self.profile_ema is None:
            self.profile_ema = current
        else:
            alpha = 0.85
            keys = set(list(self.profile_ema.keys()) + list(current.keys()))
            self.profile_ema = {
                key: alpha * float(self.profile_ema.get(key, 0.0)) + (1.0 - alpha) * float(current.get(key, 0.0))
                for key in keys
            }

        now = time.time()
        interval = float(getattr(config, "SEG_PROFILE_LOG_INTERVAL", 2.0))
        if now - self.profile_last_log < interval:
            return
        self.profile_last_log = now

        avg = self.profile_ema
        total_for_print = (
            float(avg.get("infer", 0.0))
            + float(avg.get("prep", 0.0))
            + float(avg.get("search", 0.0))
            + float(avg.get("fit", 0.0))
            + float(avg.get("render", 0.0))
            + float(avg.get("overhead", 0.0))
        )
        queue_wait_avg = float(avg.get("queue_wait", 0.0))
        fps_est = 1.0 / max(float(total_for_print), 1e-6)
        print(
            "SegProfile "
            f"infer={avg['infer'] * 1000.0:.1f}ms "
            f"prep={avg['prep'] * 1000.0:.1f}ms "
            f"search={avg['search'] * 1000.0:.1f}ms "
            f"fit={avg['fit'] * 1000.0:.1f}ms "
            f"render={avg['render'] * 1000.0:.1f}ms "
            f"queue_wait={queue_wait_avg * 1000.0:.1f}ms "
            f"total={total_for_print * 1000.0:.1f}ms "
            f"est={fps_est:.1f}fps",
            flush=True,
        )


class SegDebugOverlay:
    """缓存并绘制 Seg 路径、边界、控制取样线等调试叠加层."""

    def __init__(self, base_size):
        self.overlay = self._empty_overlay(base_size)

    def _empty_overlay(self, base_size):
        return {
            "path": None,
            "left": None,
            "right": None,
            "candidate_left": None,
            "candidate_right": None,
            "merge_guide": None,
            "fork_point": None,
            "control_band": None,
            "bottom_mid": (0.0, 0.0),
            "base_size": tuple(base_size),
        }

    def store(
        self,
        path_pts,
        left_pts,
        right_pts,
        img_w,
        img_h,
        candidate_left_pts=None,
        candidate_right_pts=None,
        merge_guide_pts=None,
        fork_point=None,
        control_band=None,
        bottom_mid=None,
    ):
        if bottom_mid is None:
            bottom_mid = (float(img_w) / 2.0, float(img_h) - 1.0)
        self.overlay = {
            "path": None if path_pts is None else np.array(path_pts, dtype=np.float32).copy(),
            "left": None if left_pts is None else np.array(left_pts, dtype=np.float32).copy(),
            "right": None if right_pts is None else np.array(right_pts, dtype=np.float32).copy(),
            "candidate_left": None if candidate_left_pts is None else np.array(candidate_left_pts, dtype=np.float32).copy(),
            "candidate_right": None if candidate_right_pts is None else np.array(candidate_right_pts, dtype=np.float32).copy(),
            "merge_guide": None if merge_guide_pts is None else np.array(merge_guide_pts, dtype=np.float32).copy(),
            "fork_point": None if fork_point is None else (float(fork_point[0]), float(fork_point[1])),
            "control_band": control_band,
            "bottom_mid": (float(bottom_mid[0]), float(bottom_mid[1])),
            "base_size": (int(img_w), int(img_h)),
        }

    def draw(self, image):
        if image is None:
            return image

        overlay = self.overlay
        base_w, base_h = overlay.get("base_size", tuple(config.SEG_SIZE))
        if base_w <= 0 or base_h <= 0:
            return image

        img_h, img_w = image.shape[:2]
        scale_x = img_w / float(base_w)
        scale_y = img_h / float(base_h)
        scale = max(scale_x, scale_y)

        def _scaled_polyline(polyline):
            if polyline is None:
                return None
            pts = np.array(polyline, dtype=np.float32).copy().reshape((-1, 1, 2))
            pts[:, 0, 0] *= scale_x
            pts[:, 0, 1] *= scale_y
            return pts.astype(np.int32)

        def _scale_point(pt):
            return (
                int(round(float(pt[0]) * scale_x)),
                int(round(float(pt[1]) * scale_y)),
            )

        path_poly = _scaled_polyline(overlay.get("path"))
        left_poly = _scaled_polyline(overlay.get("left"))
        right_poly = _scaled_polyline(overlay.get("right"))
        candidate_left_poly = _scaled_polyline(overlay.get("candidate_left"))
        candidate_right_poly = _scaled_polyline(overlay.get("candidate_right"))
        merge_guide_poly = _scaled_polyline(overlay.get("merge_guide"))

        path_thickness = max(1, int(round(config.SEG_DEBUG_PATH_THICKNESS * scale)))
        boundary_thickness = max(1, int(round(config.SEG_DEBUG_BOUNDARY_THICKNESS * scale)))
        candidate_path_thickness = max(1, int(round(config.SEG_DEBUG_CANDIDATE_PATH_THICKNESS * scale)))
        merge_guide_thickness = max(1, int(round(config.SEG_DEBUG_MERGE_GUIDE_THICKNESS * scale)))
        bottom_mid_radius = max(1, int(round(config.SEG_DEBUG_BOTTOM_MID_RADIUS * scale)))
        control_band_thickness = max(1, int(round(getattr(config, "SEG_DEBUG_CONTROL_BAND_THICKNESS", 2) * scale)))

        if bool(getattr(config, "SEG_DEBUG_DRAW_CANDIDATE_PATHS", False)) and candidate_left_poly is not None:
            cv2.polylines(image, [candidate_left_poly], False, config.SEG_DEBUG_LEFT_PATH_COLOR, candidate_path_thickness)
        if bool(getattr(config, "SEG_DEBUG_DRAW_CANDIDATE_PATHS", False)) and candidate_right_poly is not None:
            cv2.polylines(image, [candidate_right_poly], False, config.SEG_DEBUG_RIGHT_PATH_COLOR, candidate_path_thickness)
        if path_poly is not None:
            cv2.polylines(image, [path_poly], False, config.SEG_DEBUG_PATH_COLOR, path_thickness)
        if bool(getattr(config, "SEG_DEBUG_DRAW_BOUNDARIES", True)) and left_poly is not None:
            cv2.polylines(image, [left_poly], False, config.SEG_DEBUG_LEFT_BOUNDARY_COLOR, boundary_thickness)
        if bool(getattr(config, "SEG_DEBUG_DRAW_BOUNDARIES", True)) and right_poly is not None:
            cv2.polylines(image, [right_poly], False, config.SEG_DEBUG_RIGHT_BOUNDARY_COLOR, boundary_thickness)
        if bool(getattr(config, "SEG_DEBUG_DRAW_MERGE_GUIDE", True)) and merge_guide_poly is not None:
            cv2.polylines(image, [merge_guide_poly], False, config.SEG_DEBUG_MERGE_GUIDE_COLOR, merge_guide_thickness, cv2.LINE_AA)

        control_band = overlay.get("control_band")
        if bool(getattr(config, "SEG_DEBUG_CONTROL_BAND_ENABLED", True)) and control_band is not None:
            try:
                y_min = float(control_band[0])
                y_max = float(control_band[1])
                y1 = int(round(np.clip(y_min * scale_y, 0, img_h - 1)))
                y2 = int(round(np.clip(y_max * scale_y, 0, img_h - 1)))
                color = getattr(config, "SEG_DEBUG_CONTROL_BAND_COLOR", (255, 0, 255))
                cv2.line(image, (0, y1), (img_w - 1, y1), color, control_band_thickness, cv2.LINE_AA)
                cv2.line(image, (0, y2), (img_w - 1, y2), color, control_band_thickness, cv2.LINE_AA)
            except Exception:
                pass

        bottom_mid = overlay.get("bottom_mid", (float(base_w) / 2.0, float(base_h) - 1.0))
        fork_point = overlay.get("fork_point")
        if fork_point is not None:
            fork_pt = _scale_point(fork_point)
            fork_bottom_pt = _scale_point(bottom_mid)
            cv2.line(
                image,
                fork_pt,
                fork_bottom_pt,
                config.SEG_DEBUG_FORK_DIVIDER_COLOR,
                max(1, int(round(config.SEG_DEBUG_FORK_DIVIDER_THICKNESS * scale))),
                cv2.LINE_AA,
            )
        cv2.circle(image, _scale_point(bottom_mid), bottom_mid_radius, config.SEG_DEBUG_BOTTOM_MID_COLOR, -1)
        return image


def draw_seg_status_text(ai_view, fps_stats, steer_signal, servo_pwm, branch_stats, stone_branch_side=None):
    """绘制 Seg 调试文字."""
    cv2.putText(
        ai_view,
        f"Seg FPS:{fps_stats.get('seg_fps', 0):.1f} YOLO:{fps_stats.get('yolo_fps', 0):.1f}",
        config.SEG_DEBUG_TEXT_POS_FPS,
        1,
        config.SEG_DEBUG_TEXT_FONT_SCALE,
        config.SEG_DEBUG_TEXT_COLOR_FPS,
        config.SEG_DEBUG_TEXT_THICKNESS,
    )
    cv2.putText(
        ai_view,
        f"Ctrl:{steer_signal:.1f} PWM:{servo_pwm}",
        config.SEG_DEBUG_TEXT_POS_CTRL,
        1,
        config.SEG_DEBUG_TEXT_FONT_SCALE,
        config.SEG_DEBUG_TEXT_COLOR_CTRL,
        config.SEG_DEBUG_TEXT_THICKNESS,
    )
    stone_side_text = "UNK"
    if stone_branch_side == -1:
        stone_side_text = "LEFT"
    elif stone_branch_side == 1:
        stone_side_text = "RIGHT"
    elif stone_branch_side is not None:
        stone_side_text = "NONE"
    cv2.putText(
        ai_view,
        f"Stone:{stone_side_text}",
        config.SEG_DEBUG_TEXT_POS_STONE,
        1,
        config.SEG_DEBUG_TEXT_FONT_SCALE,
        config.SEG_DEBUG_TEXT_COLOR_STONE,
        config.SEG_DEBUG_TEXT_THICKNESS,
    )
    cv2.putText(
        ai_view,
        (
            f"PairsMax:{branch_stats.get('branch_pair_count_max', 0)} "
            f"Rows2+:{branch_stats.get('branch_support_rows', 0)} "
            f"Y:{int(branch_stats.get('y_fork_active', False))} "
            f"Merge:{branch_stats.get('merge_side') or 'NONE'}"
        ),
        config.SEG_DEBUG_TEXT_POS_BRANCH,
        1,
        config.SEG_DEBUG_TEXT_FONT_SCALE,
        config.SEG_DEBUG_TEXT_COLOR_BRANCH,
        config.SEG_DEBUG_TEXT_THICKNESS,
    )
    return ai_view


def draw_yolo_boxes(image, boxes):
    """在主预览图上绘制 YOLO 检测框和标签."""
    if image is None or not boxes:
        return image

    img_h, img_w = image.shape[:2]
    target_w, target_h = config.TARGET_RES
    scale_x = img_w / float(target_w)
    scale_y = img_h / float(target_h)

    for obj in boxes:
        rect = obj.get("rect", obj.get("box", (0, 0, 0, 0)))
        if len(rect) != 4:
            continue
        x, y, w, h = rect
        cls_id = obj.get("class_id", -1)
        cls_name = obj.get("class_name", "?")
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
            cv2.LINE_AA,
        )

    return image


def summarize_yolo_boxes(boxes):
    """生成一行简短检测摘要，便于在预览页确认 YOLO 输出."""
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


def draw_preview_status_panel(
    rendered_img,
    *,
    current_seg_fps,
    current_yolo_fps,
    steer_signal,
    actual_servo,
    actual_speed,
    sign_llm_waiting_result,
    sign_llm_stop_active,
    person_stop_active,
    person_avoid_active=False,
    person_avoid_boundary_inset_x=0.0,
    person_avoid_boundary_side="none",
    debug_keyboard_stop_active=False,
    route_state,
    route_choice,
    yolo_boxes,
):
    """绘制网页主预览上的状态面板和 YOLO 摘要."""
    if rendered_img is None:
        return rendered_img

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
        cv2.LINE_AA,
    )
    cv2.putText(
        rendered_img,
        f"Ctrl:{steer_signal:.1f} Srv:{actual_servo}",
        config.PREVIEW_TEXT_POS_CTRL,
        cv2.FONT_HERSHEY_SIMPLEX,
        config.PREVIEW_TEXT_FONT_SCALE,
        config.PREVIEW_TEXT_COLOR,
        config.PREVIEW_TEXT_THICKNESS,
        cv2.LINE_AA,
    )
    cv2.putText(
        rendered_img,
        f"Target Spd:{actual_speed}",
        config.PREVIEW_TEXT_POS_SPEED,
        cv2.FONT_HERSHEY_SIMPLEX,
        config.PREVIEW_TEXT_FONT_SCALE,
        config.PREVIEW_TEXT_ACCENT_COLOR,
        config.PREVIEW_TEXT_THICKNESS,
        cv2.LINE_AA,
    )

    stop_text = ""
    if debug_keyboard_stop_active:
        stop_text = "STOP_BY_KEY"
    elif sign_llm_waiting_result:
        stop_text = "SIGN_LLM_WAIT"
    elif sign_llm_stop_active:
        stop_text = "SIGN_LLM_OCR"
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
            cv2.LINE_AA,
        )

    if route_state not in ("IDLE",):
        route_choice_text = "R" if route_choice == 1 else ("L" if route_choice == -1 else "-")
        cv2.putText(
            rendered_img,
            f"Route:{route_state} {route_choice_text}",
            (config.PREVIEW_TEXT_POS_STOP[0], config.PREVIEW_TEXT_POS_STOP[1] + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.PREVIEW_TEXT_FONT_SCALE,
            config.PREVIEW_TEXT_ACCENT_COLOR,
            config.PREVIEW_TEXT_THICKNESS,
            cv2.LINE_AA,
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
        summarize_yolo_boxes(yolo_boxes),
        config.PREVIEW_TEXT_POS_YOLO_SUMMARY,
        cv2.FONT_HERSHEY_SIMPLEX,
        config.PREVIEW_TEXT_FONT_SCALE,
        config.PREVIEW_TEXT_ACCENT_COLOR,
        config.PREVIEW_TEXT_THICKNESS,
        cv2.LINE_AA,
    )
    return rendered_img


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


def preview_index_html():
    return '''
    <html>
    <body style="background:#000;text-align:center;margin:0;" tabindex="0">
        <img src="/video_feed" style="max-width:100%; height:100vh; image-rendering: pixelated;">
        <script>
        document.body.focus();
        document.addEventListener('keydown', function (event) {
            const key = String(event.key || '').toLowerCase();
            if (key === 'b') {
                fetch('/debug_drive/start', { method: 'POST' });
            } else if (key === 'e') {
                fetch('/debug_drive/stop', { method: 'POST' });
            }
        });
        </script>
    </body>
    </html>
    '''


def encode_mjpeg_frame(frame):
    """把预览图编码成 MJPEG chunk."""
    ret, buffer = cv2.imencode(
        '.jpg',
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY],
    )
    if not ret:
        return None
    return (
        b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
    )

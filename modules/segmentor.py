# modules/segmentor.py
"""赛道分割与路径规划模块.

这个模块负责:
1. 调用分割模型得到二值赛道 mask
2. 在 mask 空间中搜索一条可跟踪路径（引入分支局部中心约束，抑制切内线）
3. 调用 path_controller 将路径点转换成单一转向控制量
4. 引入多项式时域低通滤波 (EMA)，提升路径稳定性
5. 返回用于控制的 steer_signal，以及一张调试渲染图

当前路径选择策略的核心优先级是:
1. 先在 mask 中寻找可行路径；若存在明显岔路，再先做分叉分区
2. 路径必须从图像底部触达区域起步，悬空候选会被直接丢弃
3. 优先使用语义路牌或固定策略给出的分支方向
4. 没有有效分支方向时默认偏向左支
"""

import cv2
import numpy as np
import time
from rknnlite.api import RKNNLite
import config
from modules.debug_tools import SegDebugOverlay, SegProfileLogger, draw_seg_status_text
from modules.path_controller import PathController
try:
    from utils.rknn_quiet import suppress_rknn_init_output
except ImportError:
    from contextlib import nullcontext as suppress_rknn_init_output

class RoadSegmentor:
    def __init__(self, core_id):
        """初始化分割模型与分割空间坐标映射."""
        with suppress_rknn_init_output():
            self.rknn = RKNNLite()

            if self.rknn.load_rknn(config.SEG_MODEL) != 0 or self.rknn.init_runtime(core_mask=core_id) != 0:
                raise RuntimeError("Seg 模型加载或初始化失败")
            
        w_seg, h_seg = config.SEG_SIZE
        target_w, target_h = config.TARGET_RES
        self.seg_crop_top_ratio = float(getattr(config, "SEG_INPUT_CROP_TOP_RATIO", 0.0))
        self.seg_crop_top_ratio = max(0.0, min(0.95, self.seg_crop_top_ratio))
        self.seg_crop_top_target_y = float(target_h) * self.seg_crop_top_ratio
        self.seg_crop_target_h = max(1.0, float(target_h) - self.seg_crop_top_target_y)
        self.scale_x_to_seg = w_seg / float(target_w)
        self.scale_y_to_seg = h_seg / self.seg_crop_target_h
        self.planning_class_names = set(
            getattr(config, "PLANNING_CLASS_NAMES", ())
        )
        self.fixed_track_widths = np.array(
            getattr(config, "SEG_FIXED_WIDTHS_320_SMOOTH", getattr(config, "SEG_FIXED_WIDTHS_320", ())),
            dtype=np.float32,
        )
        self.fixed_track_width_indices = np.where(self.fixed_track_widths > 0)[0]
        self.fixed_width_source_size = tuple(
            getattr(config, "SEG_FIXED_WIDTH_SOURCE_SIZE", config.SEG_SIZE)
        )
        self.fixed_width_source_crop_top_ratio = float(
            getattr(config, "SEG_FIXED_WIDTH_SOURCE_CROP_TOP_RATIO", 0.0)
        )

        # -------------------------------------------------------------------
        # 时域滤波历史记忆
        # -------------------------------------------------------------------
        self.last_poly_coeffs = None
        self.last_path_points_orig = None
        self.path_controller = PathController()
        self.missing_path_frames = 0
        self.ema_alpha = float(config.SEG_EMA_ALPHA)
        self.last_branch_stats = {
            "branch_pair_count_max": 0,
            "branch_support_rows": 0,
            "fork_active": False,
            "y_fork_active": False,
            "merge_side": None,
            "car_active": False,
            "car_state": "FOLLOW_LANE",
            "car_rows_to_bottom": None,
            "car_boundary_side": "",
            "car_boundary_x": None,
            "car_boundary_error": None,
        }
        self.merge_state_active = False
        self.merge_state_hit_frames = 0
        self.merge_state_hit_times = []
        self.merge_state_miss_frames = 0
        self.merge_state_exit_frames = 0
        self.merge_state_enter_time = None
        self.merge_state_info = None
        self.merge_state_side = None
        self.merge_edge_trace_debug_counter = 0
        self.locked_car = None
        self.locked_car_miss_frames = 0
        self.car_avoidance_state = "FOLLOW_LANE"
        self.car_avoidance_cycle_id = 0
        self.car_avoidance_boundary_side = None
        self.car_clearing_frames = 0
        self.car_last_avoid_path = None
        self.car_last_avoid_path_is_boundary = False
        self.car_last_boundary_inset_x = 0.0
        self.last_control_path_source = "normal"
        self.last_car_avoid_log_at = 0.0
        self.last_car_avoid_log_state = None
        self.debug_overlay = SegDebugOverlay(tuple(config.SEG_SIZE))
        self.seg_profile_logger = SegProfileLogger()
        self.last_control_c_debug_log_at = 0.0

    def _log_control_c_debug(self, steer_signal, servo_pwm, car_active=False):
        """低频打印 C 控制内部量，便于试车后反推小弯/大弯参数."""
        if not bool(getattr(config, "CONTROL_C_DEBUG_LOG_ENABLED", False)):
            return
        if str(getattr(config, "STEER_CONTROL_MODE", "weighted_slope")).lower() != "control_c":
            return

        debug = getattr(self.path_controller, "last_control_c_debug", None)
        if not debug:
            return

        now = time.monotonic()
        interval = max(0.05, float(getattr(config, "CONTROL_C_DEBUG_LOG_INTERVAL", 0.5)))
        if now - float(self.last_control_c_debug_log_at) < interval:
            return
        self.last_control_c_debug_log_at = now

        heading_deg = float(np.degrees(debug.get("heading_error", 0.0)))
        filtered_heading_deg = float(np.degrees(debug.get("filtered_heading_error", 0.0)))
        ff_heading_deg = float(np.degrees(debug.get("ff_heading_error", 0.0)))
        print(
            "C调参: "
            f"pwm={int(servo_pwm)} steer={float(steer_signal):.2f} "
            f"e={float(debug.get('lateral_error', 0.0)):.1f}px "
            f"e_f={float(debug.get('filtered_error', 0.0)):.1f}px "
            f"de={float(debug.get('error_delta', 0.0)):.1f}px "
            f"psi={heading_deg:.1f}deg "
            f"psi_f={filtered_heading_deg:.1f}deg "
            f"psi_ff={ff_heading_deg:.1f}deg "
            f"level={float(debug.get('curve_level', 0.0)):.2f} "
            f"raw={float(debug.get('raw_curve_level', 0.0)):.2f} "
            f"h={float(debug.get('curve_from_heading', 0.0)):.2f} "
            f"d={float(debug.get('curve_from_delta', 0.0)):.2f} "
            f"Kp={float(debug.get('lateral_gain', 0.0)):.3f} "
            f"Kd={float(debug.get('d_gain', 0.0)):.3f} "
            f"Kyaw={float(debug.get('heading_gain', 0.0)):.2f} "
            f"Kff={float(debug.get('ff_gain', 0.0)):.2f} "
            f"terms=({float(debug.get('lateral_term', 0.0)):.2f},"
            f"{float(debug.get('d_term', 0.0)):.2f},"
            f"{float(debug.get('heading_term', 0.0)):.2f},"
            f"{float(debug.get('ff_term', 0.0)):.2f}) "
            f"car={int(bool(car_active))}",
            flush=True,
        )

    def _log_car_avoidance_process(self, debug, event="", force=False):
        """打印避车状态机过程，用来定位躲车后丢线前的状态流转."""
        if not bool(getattr(config, "CAR_AVOIDANCE_PROCESS_LOG_ENABLED", True)):
            return

        state = str(debug.get("state", self.car_avoidance_state))
        event = str(event or debug.get("event", ""))
        active = bool(debug.get("active", False))
        cycle_id = int(debug.get("cycle_id", self.car_avoidance_cycle_id))
        detected = int(debug.get("detected_cars", 0))
        boundary_strength = float(debug.get("boundary_strength_x", 0.0))
        signature = (
            state,
            event,
            active,
            detected,
            int(debug.get("miss_frames", 0)),
            int(debug.get("clear_frames", 0)),
            bool(debug.get("boundary_path_active", False)),
            str(debug.get("boundary_side", "") or ""),
            round(boundary_strength, 1),
        )

        now = time.monotonic()
        interval = max(0.05, float(getattr(config, "LOG_INTERVAL_CAR_AVOIDANCE_PROCESS", 0.25)))
        if not force and signature == self.last_car_avoid_log_state and now - float(self.last_car_avoid_log_at) < interval:
            return
        if not force and not active and detected <= 0 and state == "FOLLOW_LANE":
            return

        self.last_car_avoid_log_state = signature
        self.last_car_avoid_log_at = now

        rows = debug.get("rows_to_bottom")
        rows_text = "无" if rows is None else f"{float(rows):.1f}"
        center = debug.get("locked_center")
        center_text = "无" if center is None else f"({float(center[0]):.1f},{float(center[1]):.1f})"
        boundary_side = str(debug.get("boundary_side", "") or "").upper() or "无"
        nearest_side = str(debug.get("nearest_boundary_side", "") or "").upper() or "无"
        boundary_x = debug.get("boundary_x")
        boundary_x_text = "无" if boundary_x is None else f"{float(boundary_x):.1f}"
        boundary_error = debug.get("boundary_error")
        boundary_error_text = "无" if boundary_error is None else f"{float(boundary_error):.1f}"
        control_error = debug.get("control_path_error")
        control_error_text = "无" if control_error is None else f"{float(control_error):.1f}"
        stage_label = "避车过程"
        stage_note = ""
        line_suffix = ""
        if event == "enter_avoiding":
            stage_label = "避车开始"
            stage_note = "开始"
            line_suffix = "\033[0m"
        elif event == "clearing":
            stage_label = "避车回正"
            stage_note = "回正"
        elif event == "clear_done":
            stage_label = "避车结束"
            stage_note = "结束"
            line_suffix = "\033[0m"
        print(
            ("\033[92m" if event in ("enter_avoiding", "clear_done") else "") +
            f"避车#{cycle_id} {stage_label}: {stage_note} "
            f"state={state} det={detected} locked={int(bool(debug.get('locked_confirmed', False)))} "
            f"hits={int(debug.get('locked_hit_frames', 0))} miss={int(debug.get('miss_frames', 0))} "
            f"clear={int(debug.get('clear_frames', 0))} rows={rows_text} center={center_text} "
            f"inset={float(debug.get('boundary_inset_x', 0.0)):.1f} weight={float(debug.get('avoid_weight', 0.0)):.2f} "
            f"path={int(bool(debug.get('boundary_path_active', False)))} ready={int(bool(debug.get('boundary_ready', False)))} "
            f"active={int(active)} "
            f"side={boundary_side} nearest={nearest_side} boundary_x={boundary_x_text} "
            f"boundary_e={boundary_error_text} ctrl_e={control_error_text}" +
            line_suffix,
            flush=True,
        )

    def selected_left_boundary_x_at_target_y(self, target_y):
        """返回当前选中路径左边界在 TARGET_RES 坐标系下的 x."""
        left_boundary = self.debug_overlay.overlay.get("left")
        if left_boundary is None:
            return None

        y_seg = (float(target_y) - self.seg_crop_top_target_y) * self.scale_y_to_seg
        if y_seg < 0.0 or y_seg > float(config.SEG_SIZE[1] - 1):
            return None

        x_seg = self._boundary_x_at_y(left_boundary, y_seg)
        if x_seg is None:
            return None
        if self.scale_x_to_seg <= 0.0:
            return None
        return float(x_seg) / self.scale_x_to_seg

    def selected_right_boundary_x_at_target_y(self, target_y):
        """返回当前选中路径右边界在 TARGET_RES 坐标系下的 x."""
        right_boundary = self.debug_overlay.overlay.get("right")
        if right_boundary is None:
            return None

        y_seg = (float(target_y) - self.seg_crop_top_target_y) * self.scale_y_to_seg
        if y_seg < 0.0 or y_seg > float(config.SEG_SIZE[1] - 1):
            return None

        x_seg = self._boundary_x_at_y(right_boundary, y_seg)
        if x_seg is None:
            return None
        if self.scale_x_to_seg <= 0.0:
            return None
        return float(x_seg) / self.scale_x_to_seg

    def _store_main_overlay(
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
        """缓存主图路径叠加层，供主线程在其它元素之上重绘."""
        self.debug_overlay.store(
            path_pts,
            left_pts,
            right_pts,
            img_w,
            img_h,
            candidate_left_pts=candidate_left_pts,
            candidate_right_pts=candidate_right_pts,
            merge_guide_pts=merge_guide_pts,
            fork_point=fork_point,
            control_band=control_band,
            bottom_mid=bottom_mid,
        )

    def draw_path_overlay(self, image):
        """把最近一次搜索得到的主图路径/边界叠加到任意尺寸的画面最上层."""
        return self.debug_overlay.draw(image)

    def _profile_add(self, infer_s, preprocess_s, search_s, fit_s, render_s, total_s, queue_wait_s=None):
        """按阶段统计分割链路耗时，节流打印用于定位掉帧瓶颈。"""
        self.seg_profile_logger.add(
            infer_s=infer_s,
            preprocess_s=preprocess_s,
            search_s=search_s,
            fit_s=fit_s,
            render_s=render_s,
            total_s=total_s,
            queue_wait_s=queue_wait_s,
        )

    def _path_stability_enabled(self):
        return bool(getattr(config, "SEG_PATH_STABILITY_ENABLED", True))

    def _fit_path_poly_coeffs(self, node_y, node_x):
        """拟合 x=f(y) 的路径多项式，统一二次系数格式."""
        if len(np.unique(node_y)) > 2:
            return np.polyfit(node_y, node_x, 2)

        coeffs = np.polyfit(node_y, node_x, 1)
        return np.insert(coeffs, 0, 0)

    def _interp_path_xs(self, path_points, target_ys):
        """按 y 坐标在路径上插值得到 x，用于跨帧同高度比较."""
        pts = np.array(path_points, dtype=np.float32).reshape((-1, 2))
        if pts.size == 0:
            return None

        target_ys = np.array(target_ys, dtype=np.float32).reshape((-1,))
        if target_ys.size == 0:
            return np.array([], dtype=np.float32)

        order = np.argsort(pts[:, 1])
        ys = pts[order, 1]
        xs = pts[order, 0]
        unique_ys, unique_indices = np.unique(ys, return_index=True)
        unique_xs = xs[unique_indices]

        if len(unique_ys) == 1:
            return np.full_like(target_ys, float(unique_xs[0]), dtype=np.float32)

        return np.interp(target_ys, unique_ys, unique_xs).astype(np.float32)

    def _temporal_path_stats(self, path_points, reference_points=None):
        """计算当前候选路径相对上一帧输出路径的横向跳变量."""
        empty_stats = {
            "penalty": 0.0,
            "mean_jump": 0.0,
            "max_jump": 0.0,
            "overlap_points": 0,
        }
        if not self._path_stability_enabled():
            return empty_stats

        ref_points = self.last_path_points_orig if reference_points is None else reference_points
        if ref_points is None:
            return empty_stats

        curr = np.array(path_points, dtype=np.float32).reshape((-1, 2))
        ref = np.array(ref_points, dtype=np.float32).reshape((-1, 2))
        if len(curr) == 0 or len(ref) == 0:
            return empty_stats

        y_min = max(float(np.min(curr[:, 1])), float(np.min(ref[:, 1])))
        y_max = min(float(np.max(curr[:, 1])), float(np.max(ref[:, 1])))
        if y_max < y_min:
            return empty_stats

        min_overlap = max(1, int(getattr(config, "SEG_PATH_TEMPORAL_MIN_OVERLAP_POINTS", 4)))
        curr_ys = curr[:, 1]
        overlap_ys = curr_ys[(curr_ys >= y_min) & (curr_ys <= y_max)]
        if len(overlap_ys) < min_overlap:
            overlap_ys = np.linspace(y_min, y_max, num=min_overlap, dtype=np.float32)

        curr_x = self._interp_path_xs(curr, overlap_ys)
        ref_x = self._interp_path_xs(ref, overlap_ys)
        if curr_x is None or ref_x is None or len(curr_x) == 0:
            return empty_stats

        jumps = np.abs(curr_x - ref_x)
        mean_jump = float(np.mean(jumps))
        max_jump = float(np.max(jumps))
        soft_max_jump = max(0.0, float(getattr(config, "SEG_PATH_TEMPORAL_SOFT_MAX_JUMP", 0.0)))
        jump_excess = np.maximum(jumps - soft_max_jump, 0.0)

        penalty = (
            mean_jump * float(getattr(config, "SEG_PATH_TEMPORAL_SCORE_GAIN", 0.0)) +
            float(np.max(jump_excess)) * float(getattr(config, "SEG_PATH_TEMPORAL_EXCESS_SCORE_GAIN", 0.0))
        )
        return {
            "penalty": float(penalty),
            "mean_jump": mean_jump,
            "max_jump": max_jump,
            "overlap_points": int(len(overlap_ys)),
        }

    def _limit_path_frame_jump(self, path_points):
        """硬限制最终输出路径相对上一帧的横向位移."""
        pts = np.array(path_points, dtype=np.float32).reshape((-1, 2)).copy()
        if not self._path_stability_enabled() or self.last_path_points_orig is None:
            return pts, False

        max_jump = float(getattr(config, "SEG_PATH_MAX_FRAME_X_JUMP", 0.0))
        if max_jump <= 0.0 or len(pts) == 0:
            return pts, False

        ref = np.array(self.last_path_points_orig, dtype=np.float32).reshape((-1, 2))
        if len(ref) == 0:
            return pts, False

        ref_y_min = float(np.min(ref[:, 1]))
        ref_y_max = float(np.max(ref[:, 1]))
        overlap_mask = (pts[:, 1] >= ref_y_min) & (pts[:, 1] <= ref_y_max)
        if not np.any(overlap_mask):
            return pts, False

        ref_x = self._interp_path_xs(ref, pts[overlap_mask, 1])
        if ref_x is None or len(ref_x) == 0:
            return pts, False

        before_x = pts[overlap_mask, 0].copy()
        pts[overlap_mask, 0] = np.clip(before_x, ref_x - max_jump, ref_x + max_jump)
        limited = bool(np.any(np.abs(pts[overlap_mask, 0] - before_x) > 1e-3))
        return pts, limited

    def _hold_last_path(self):
        """路径短暂丢失时沿用上一帧输出，避免控制量瞬间归零."""
        if not self._path_stability_enabled() or self.last_path_points_orig is None:
            return None

        hold_frames = max(0, int(getattr(config, "SEG_PATH_HOLD_MISSING_FRAMES", 0)))
        self.missing_path_frames += 1
        if self.missing_path_frames > hold_frames:
            self.last_path_points_orig = None
            self.last_poly_coeffs = None
            return None

        return np.array(self.last_path_points_orig, dtype=np.float32).reshape((-1, 2)).copy()

    def _prepare_search_mask(self, mask, active_height=None):
        """为路径搜索准备更连贯的 mask，并把路径计算限制在底部窗口."""
        kernel_size = max(1, int(config.SEG_PATH_DILATE_KERNEL))
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        search_mask = mask.astype(np.uint8).copy()
        dilate_iter = max(0, int(config.SEG_PATH_DILATE_ITER))
        bottom_height = max(0, int(config.SEG_PATH_DILATE_BOTTOM_HEIGHT))

        if dilate_iter > 0 and bottom_height > 0:
            h = search_mask.shape[0]
            y0 = max(0, h - bottom_height)
            bottom_roi = search_mask[y0:h, :]
            search_mask[y0:h, :] = cv2.dilate(
                bottom_roi,
                kernel,
                iterations=dilate_iter,
            )

        if active_height is None:
            active_height = getattr(config, "SEG_PATH_ACTIVE_HEIGHT", 0)
        active_height = max(0, int(active_height))
        if active_height > 0:
            h = search_mask.shape[0]
            y0 = max(0, h - active_height)
            if y0 > 0:
                search_mask[:y0, :] = 0
        search_mask = (search_mask > 0).astype(np.uint8)
        if bool(getattr(config, "SEG_KEEP_BOTTOM_COMPONENTS", False)):
            search_mask = self._keep_bottom_components(search_mask)
        return search_mask

    def _keep_bottom_components(self, mask):
        """保留触达底部的连通白区；无触底区域时回退到最大连通区."""
        if mask is None or mask.size == 0:
            return mask

        mask_u8 = (mask > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if num_labels <= 1:
            return mask_u8

        h = mask_u8.shape[0]
        bottom_touch_height = max(1, int(getattr(config, "SEG_PATH_BOTTOM_TOUCH_HEIGHT", 20)))
        y0 = max(0, h - bottom_touch_height)
        bottom_labels = np.unique(labels[y0:h, :][mask_u8[y0:h, :] > 0])
        keep_labels = bottom_labels[bottom_labels > 0]
        if len(keep_labels) > 0:
            return np.isin(labels, keep_labels).astype(np.uint8)

        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) == 0:
            return np.zeros_like(mask_u8, dtype=np.uint8)

        largest_label = int(np.argmax(areas)) + 1
        return (labels == largest_label).astype(np.uint8)

    def _extract_edge_mask(self, mask):
        """按八邻域定义提取道路边界点."""
        if mask is None or mask.size == 0:
            return np.zeros_like(mask, dtype=np.uint8)

        mask_u8 = (mask > 0).astype(np.uint8)
        # 8 邻域边界等价于：mask 中减去“3x3 腐蚀后仍保留下来的内部区域”。
        # 这样能保持原先的边界定义，同时避免 Python 双层像素循环拖慢帧率。
        interior = cv2.erode(
            mask_u8,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return cv2.subtract(mask_u8, interior)

    def _find_mask_runs(self, mask_row, gap_thresh, min_pixels):
        """在单行 mask 上寻找若干横向白区段，供分叉几何判定使用."""
        xs = np.where(mask_row > 0)[0]
        if len(xs) == 0:
            return []

        splits = np.split(xs, np.where(np.diff(xs) > int(gap_thresh))[0] + 1)
        runs = []
        for split in splits:
            pixel_count = int(len(split))
            if pixel_count < int(min_pixels):
                continue

            left_x = int(split[0])
            right_x = int(split[-1])
            runs.append({
                "left_x": float(left_x),
                "right_x": float(right_x),
                "center_x": 0.5 * float(left_x + right_x),
                "width": float(right_x - left_x),
                "pixel_count": pixel_count,
            })
        return runs

    def _pick_dual_branch_runs(self, runs):
        """从一组横向白区段里挑出最像左右两支的组合."""
        if len(runs) < 2:
            return None

        best_pair = None
        for left_idx in range(len(runs) - 1):
            for right_idx in range(left_idx + 1, len(runs)):
                left_run = runs[left_idx]
                right_run = runs[right_idx]
                separation = float(right_run["center_x"]) - float(left_run["center_x"])
                if separation < float(config.FORK_MIN_BRANCH_SEP):
                    continue

                score = (
                    min(int(left_run["pixel_count"]), int(right_run["pixel_count"])),
                    int(left_run["pixel_count"]) + int(right_run["pixel_count"]),
                    separation,
                )
                if best_pair is None or score > best_pair["score"]:
                    best_pair = {
                        "left_run": left_run,
                        "right_run": right_run,
                        "separation": separation,
                        "score": score,
                    }
        return best_pair

    def _select_inner_gap_opening_run(self, branch_rows):
        """挑出一段“中间缺口双边张开”的有效连续区域，用来确认真实 Y 型岔路.

        这里要求的是“总体张开”而不是“逐行严格单调张开”：
        - 局部几行持平是允许的
        - 小幅回退也允许
        - 但整段累计张开量要达标
        """
        if len(branch_rows) < 2:
            return None

        min_rows = max(2, int(config.FORK_INNER_OPEN_MIN_ROWS))
        min_gap_growth = float(config.FORK_INNER_OPEN_MIN_GAP_GROWTH)
        min_side_growth = float(config.FORK_INNER_OPEN_MIN_SIDE_GROWTH)
        min_step_gain = float(config.FORK_INNER_OPEN_MIN_STEP_GAIN)
        min_positive_gap_rows = max(1, int(config.FORK_INNER_OPEN_MIN_POSITIVE_GAP_ROWS))
        min_positive_side_rows = max(1, int(config.FORK_INNER_OPEN_MIN_POSITIVE_SIDE_ROWS))
        max_step_regression = float(config.FORK_INNER_OPEN_MAX_STEP_REGRESSION)
        max_miss_rows = max(0, int(config.FORK_INNER_OPEN_MAX_MISS_ROWS))

        # 按“从近到远（底部到上方）”排序，检查缺口是否向前方持续张开。
        rows = sorted(branch_rows, key=lambda row: int(row["y"]), reverse=True)

        def _run_metrics(run_rows):
            if len(run_rows) < min_rows:
                return None

            bottom_row = run_rows[0]
            top_row = run_rows[-1]
            left_growth = float(bottom_row["left_inner_x"]) - float(top_row["left_inner_x"])
            right_growth = float(top_row["right_inner_x"]) - float(bottom_row["right_inner_x"])
            gap_growth = float(top_row["gap_width"]) - float(bottom_row["gap_width"])
            positive_gap_rows = 0
            positive_left_rows = 0
            positive_right_rows = 0

            for prev_row, curr_row in zip(run_rows[:-1], run_rows[1:]):
                left_step = float(prev_row["left_inner_x"]) - float(curr_row["left_inner_x"])
                right_step = float(curr_row["right_inner_x"]) - float(prev_row["right_inner_x"])
                gap_step = float(curr_row["gap_width"]) - float(prev_row["gap_width"])

                if gap_step >= min_step_gain:
                    positive_gap_rows += 1
                if left_step >= min_step_gain:
                    positive_left_rows += 1
                if right_step >= min_step_gain:
                    positive_right_rows += 1

            if (
                left_growth < min_side_growth or
                right_growth < min_side_growth or
                gap_growth < min_gap_growth or
                positive_gap_rows < min_positive_gap_rows or
                min(positive_left_rows, positive_right_rows) < min_positive_side_rows
            ):
                return None

            return {
                "run_rows": run_rows,
                "score": (
                    len(run_rows),
                    positive_gap_rows,
                    gap_growth,
                    min(left_growth, right_growth),
                ),
            }

        best_run = None
        curr_run = [rows[0]]

        for row in rows[1:]:
            prev_row = curr_run[-1]
            row_gap = int(prev_row["y"]) - int(row["y"]) - 1
            left_step = float(prev_row["left_inner_x"]) - float(row["left_inner_x"])
            right_step = float(row["right_inner_x"]) - float(prev_row["right_inner_x"])
            gap_step = float(row["gap_width"]) - float(prev_row["gap_width"])

            if (
                row_gap > max_miss_rows or
                left_step < -max_step_regression or
                right_step < -max_step_regression or
                gap_step < -max_step_regression
            ):
                candidate = _run_metrics(curr_run)
                if candidate is not None and (
                    best_run is None or candidate["score"] > best_run["score"]
                ):
                    best_run = candidate
                curr_run = [row]
                continue

            curr_run.append(row)

        candidate = _run_metrics(curr_run)
        if candidate is not None and (
            best_run is None or candidate["score"] > best_run["score"]
        ):
            best_run = candidate

        if best_run is None:
            return None
        return best_run["run_rows"]

    def _detect_y_fork(self, search_mask):
        """识别 Y 型路口，并估计用于切分左右区域的分叉点."""
        if search_mask is None or search_mask.size == 0:
            return {"active": False, "fork_point": None, "split_rows": 0}

        h, w = search_mask.shape[:2]
        top_y = int(np.clip(int(getattr(config, "FORK_SCAN_Y_TOP", 160)), 0, h - 1))
        bottom_y = int(np.clip(int(getattr(config, "FORK_SCAN_Y_BOTTOM", h)), 0, h - 1))
        if bottom_y < top_y:
            top_y, bottom_y = bottom_y, top_y

        # 分叉改为在整个有效高度范围里全局检查，而不是只盯住某个固定高度带。
        branch_rows = []
        for sample_y in range(top_y, bottom_y + 1):
            runs = self._find_mask_runs(
                search_mask[sample_y],
                config.FORK_MASK_GAP_THRESH,
                config.FORK_MASK_MIN_BRANCH_PIXELS,
            )
            dual_branch = self._pick_dual_branch_runs(runs)
            if dual_branch is None:
                continue

            branch_rows.append({
                "left_run": dual_branch["left_run"],
                "right_run": dual_branch["right_run"],
                "separation": dual_branch["separation"],
                "y": int(sample_y),
                "fork_x": 0.5 * (
                    float(dual_branch["left_run"]["right_x"]) +
                    float(dual_branch["right_run"]["left_x"])
                ),
                "left_inner_x": float(dual_branch["left_run"]["right_x"]),
                "right_inner_x": float(dual_branch["right_run"]["left_x"]),
                "gap_width": float(
                    dual_branch["right_run"]["left_x"] - dual_branch["left_run"]["right_x"]
                ),
            })

        if not branch_rows:
            return {
                "active": False,
                "fork_point": None,
                "split_rows": 0,
            }

        opening_run = self._select_inner_gap_opening_run(branch_rows)
        if opening_run is None:
            return {
                "active": False,
                "fork_point": None,
                "split_rows": 0,
            }

        lowest_split = opening_run[0]
        fork_point = (float(lowest_split["fork_x"]), float(lowest_split["y"]))
        if not self._fork_trunk_support_ok(search_mask, fork_point):
            return {
                "active": False,
                "fork_point": None,
                "split_rows": 0,
            }

        return {
            "active": True,
            "fork_point": fork_point,
            "split_rows": max(1, int(len(opening_run))),
        }

    def _fork_trunk_support_ok(self, search_mask, fork_point):
        """分叉点以下的公共主干应有 mask 支撑，避免分界线长距离悬空."""
        if not bool(getattr(config, "FORK_TRUNK_SUPPORT_CHECK_ENABLED", True)):
            return True
        if search_mask is None or search_mask.size == 0 or fork_point is None:
            return False

        h, w = search_mask.shape[:2]
        fork_x = float(fork_point[0])
        fork_y = float(fork_point[1])
        start_y = int(np.clip(np.floor(fork_y) + 1, 0, h - 1))
        end_y = int(h - 1)
        if end_y < start_y:
            return False

        radius = max(0, int(getattr(config, "FORK_TRUNK_SUPPORT_RADIUS", 5)))
        min_ratio = float(getattr(config, "FORK_TRUNK_SUPPORT_MIN_RATIO", 0.55))
        max_miss_rows = max(0, int(getattr(config, "FORK_TRUNK_SUPPORT_MAX_MISS_ROWS", 18)))
        min_rows = max(1, int(getattr(config, "FORK_TRUNK_SUPPORT_MIN_ROWS", 18)))

        checked_rows = 0
        hit_rows = 0
        miss_run = 0
        max_miss_run = 0

        for y in range(start_y, end_y + 1):
            center_col = int(np.clip(round(fork_x), 0, w - 1))
            left = max(0, center_col - radius)
            right = min(w - 1, center_col + radius)
            has_support = bool(np.any(search_mask[y, left:right + 1] > 0))

            checked_rows += 1
            if has_support:
                hit_rows += 1
                miss_run = 0
            else:
                miss_run += 1
                max_miss_run = max(max_miss_run, miss_run)

        if checked_rows < min_rows:
            return False
        if max_miss_run > max_miss_rows:
            return False
        return (float(hit_rows) / float(checked_rows)) >= min_ratio

    def _collect_branch_rows(self, search_mask, edge_mask, top_y, bottom_y):
        """收集指定 y 范围内的双白区行，并统计是否满足汇合入口条件."""
        branch_rows = []
        edge_rows = []
        trigger_row_width = float(config.MERGE_GUIDE_MIN_ROW_WIDTH)
        trigger_rows_need = max(1, int(config.MERGE_GUIDE_MIN_WIDE_ROWS))
        curr_trigger_streak = 0
        max_trigger_streak = 0

        for sample_y in range(top_y, bottom_y + 1):
            row = search_mask[sample_y]
            xs = np.where(search_mask[sample_y] > 0)[0]
            left_edge_touch = bool(row[0] > 0)
            right_edge_touch = bool(row[-1] > 0)
            edge_touch = bool(left_edge_touch or right_edge_touch)
            if edge_touch:
                edge_rows.append({
                    "y": int(sample_y),
                    "left_edge_touch": left_edge_touch,
                    "right_edge_touch": right_edge_touch,
                })
            wide_enough = len(xs) > 0 and float(xs[-1] - xs[0]) >= trigger_row_width
            if edge_touch or wide_enough:
                curr_trigger_streak += 1
                max_trigger_streak = max(max_trigger_streak, curr_trigger_streak)
            else:
                curr_trigger_streak = 0

            row_segments = self._build_row_segments(search_mask[sample_y], edge_mask[sample_y])
            if len(row_segments) < 2:
                continue

            left_seg = min(row_segments, key=lambda seg: float(seg["center_x"]))
            right_seg = max(row_segments, key=lambda seg: float(seg["center_x"]))
            if left_seg is right_seg:
                continue

            separation = float(right_seg["center_x"]) - float(left_seg["center_x"])
            if separation < float(config.FORK_MIN_BRANCH_SEP):
                continue

            branch_rows.append({
                "left_run": left_seg,
                "right_run": right_seg,
                "separation": separation,
                "y": int(sample_y),
                "fork_x": 0.5 * (float(left_seg["right_x"]) + float(right_seg["left_x"])),
                "left_inner_x": float(left_seg["right_x"]),
                "right_inner_x": float(right_seg["left_x"]),
                "gap_width": float(right_seg["left_x"] - left_seg["right_x"]),
                "left_edge_touch": left_edge_touch,
                "right_edge_touch": right_edge_touch,
            })

        return branch_rows, edge_rows, max_trigger_streak >= trigger_rows_need

    def _select_merge_run(self, branch_rows, side_name, edge_rows=None):
        """在全图双白区行里寻找单侧向下扩张的汇合尖角."""
        if len(branch_rows) < 2:
            return None
        edge_rows = [] if edge_rows is None else edge_rows

        max_miss_rows = max(0, int(config.MERGE_GUIDE_MAX_MISS_ROWS))
        min_side_delta = float(config.MERGE_GUIDE_MIN_SIDE_DELTA)
        min_inner_angle_deg = float(getattr(config, "MERGE_GUIDE_MIN_INNER_ANGLE_DEG", 0.0))
        min_inner_sharpness = float(getattr(config, "MERGE_GUIDE_MIN_INNER_SHARPNESS", 0.0))
        require_edge_above = bool(getattr(config, "MERGE_GUIDE_REQUIRE_EDGE_ABOVE_INNER", True))
        min_edge_above_rows = max(1, int(getattr(config, "MERGE_GUIDE_MIN_EDGE_ABOVE_ROWS", 1)))
        opposite_max_drift = float(config.MERGE_GUIDE_OPPOSITE_MAX_DRIFT)
        opposite_max_step_jump = float(getattr(config, "MERGE_GUIDE_OPPOSITE_MAX_STEP_JUMP", opposite_max_drift))
        rows = sorted(branch_rows, key=lambda row: int(row["y"]), reverse=True)

        def _run_metrics(run_rows):
            if len(run_rows) < 2:
                return None

            bottom_row = run_rows[0]
            top_row = run_rows[-1]
            top_y = int(top_row["y"])
            gap_shrink = float(bottom_row["gap_width"]) - float(top_row["gap_width"])

            if side_name == "left":
                primary_collapse = float(top_row["left_inner_x"]) - float(bottom_row["left_inner_x"])
                primary_values = [float(row["left_inner_x"]) for row in run_rows]
                opposite_values = [float(row["right_inner_x"]) for row in run_rows]
                edge_key = "left_edge_touch"
            else:
                primary_collapse = float(bottom_row["right_inner_x"]) - float(top_row["right_inner_x"])
                primary_values = [float(row["right_inner_x"]) for row in run_rows]
                opposite_values = [float(row["left_inner_x"]) for row in run_rows]
                edge_key = "right_edge_touch"

            # 第一阶段：先确认当前侧确实存在汇合塌陷/收口特征。
            if primary_collapse < min_side_delta or gap_shrink < min_side_delta:
                return None

            y_span = max(1.0, abs(float(bottom_row["y"]) - float(top_row["y"])))
            inner_angle_deg = float(np.degrees(np.arctan2(max(0.0, primary_collapse), y_span)))
            if inner_angle_deg < min_inner_angle_deg:
                return None
            primary_steps = np.abs(np.diff(np.array(primary_values, dtype=np.float32)))
            inner_sharpness = float(np.max(primary_steps) / max(primary_collapse, 1e-6)) if len(primary_steps) > 0 else 0.0
            if inner_sharpness < min_inner_sharpness:
                return None

            if require_edge_above:
                edge_above_rows = [
                    row for row in edge_rows
                    if bool(row.get(edge_key, False)) and int(row["y"]) <= top_y
                ]
                if len(edge_above_rows) < min_edge_above_rows:
                    return None

            # 第二阶段：只有塌陷成立后，才检查对侧可信边界是否连续、没有大跳变。
            opposite_drift = abs(opposite_values[-1] - opposite_values[0])
            opposite_step_jump = 0.0
            if len(opposite_values) > 1:
                opposite_step_jump = float(np.max(np.abs(np.diff(opposite_values))))
            if opposite_drift > opposite_max_drift or opposite_step_jump > opposite_max_step_jump:
                return None

            return {
                "run_rows": run_rows,
                # 先过塌陷/收口，再过对侧连续性；通过后按塌陷/收口强度排序。
                "score": (
                    primary_collapse + gap_shrink,
                    inner_angle_deg,
                    inner_sharpness,
                    len(run_rows),
                    -opposite_drift,
                    -opposite_step_jump,
                ),
            }

        best_run = None
        curr_run = [rows[0]]

        for row in rows[1:]:
            prev_row = curr_run[-1]
            row_gap = int(prev_row["y"]) - int(row["y"]) - 1
            if row_gap > max_miss_rows:
                candidate = _run_metrics(curr_run)
                if candidate is not None and (
                    best_run is None or candidate["score"] > best_run["score"]
                ):
                    best_run = candidate
                curr_run = [row]
                continue

            curr_run.append(row)

        candidate = _run_metrics(curr_run)
        if candidate is not None and (
            best_run is None or candidate["score"] > best_run["score"]
        ):
            best_run = candidate

        if best_run is None:
            return None
        return best_run

    def _detect_merge_guide(self, search_mask, edge_mask):
        """在指定 y 范围的宽带区域中搜索单侧汇合尖角，命中后返回引导线和命中侧."""
        if search_mask is None or search_mask.size == 0:
            return None

        h, _ = search_mask.shape[:2]
        scene_bottom_height = max(1, int(getattr(config, "SEG_SCENE_SCAN_BOTTOM_HEIGHT", h)))
        scene_top_y = max(0, h - scene_bottom_height)

        cond_top_y = int(np.clip(int(config.MERGE_GUIDE_SCAN_Y_TOP), 0, h - 1))
        cond_bottom_y = int(np.clip(int(config.MERGE_GUIDE_SCAN_Y_BOTTOM), 0, h - 1))
        cond_top_y = max(cond_top_y, scene_top_y)
        if cond_bottom_y < cond_top_y:
            cond_top_y, cond_bottom_y = cond_bottom_y, cond_top_y

        cond_branch_rows, cond_edge_rows, merge_trigger_ok = self._collect_branch_rows(
            search_mask,
            edge_mask,
            cond_top_y,
            cond_bottom_y,
        )
        branch_rows = cond_branch_rows if merge_trigger_ok else []
        edge_rows = cond_edge_rows if merge_trigger_ok else []

        free_top_y = int(np.clip(int(getattr(config, "MERGE_GUIDE_FREE_SCAN_Y_TOP", h)), 0, h - 1))
        free_bottom_y = int(np.clip(int(getattr(config, "MERGE_GUIDE_FREE_SCAN_Y_BOTTOM", h)), 0, h - 1))
        free_top_y = max(free_top_y, scene_top_y)
        if free_bottom_y < free_top_y:
            free_top_y, free_bottom_y = free_bottom_y, free_top_y
        free_branch_rows, free_edge_rows, _ = self._collect_branch_rows(
            search_mask,
            edge_mask,
            free_top_y,
            free_bottom_y,
        )
        if free_edge_rows:
            edge_rows.extend(free_edge_rows)
        if free_branch_rows:
            rows_by_y = {int(row["y"]): row for row in branch_rows}
            for row in free_branch_rows:
                rows_by_y[int(row["y"])] = row
            branch_rows = [rows_by_y[y] for y in sorted(rows_by_y.keys())]

        if not branch_rows:
            return None

        left_run = self._select_merge_run(branch_rows, "left", edge_rows)
        right_run = self._select_merge_run(branch_rows, "right", edge_rows)

        best_side = None
        best_run = None
        if left_run is not None:
            best_side = "left"
            best_run = left_run
        if right_run is not None:
            if best_run is None or right_run["score"] > best_run["score"]:
                best_side = "right"
                best_run = right_run

        if best_run is None:
            return None

        guide_polyline = self._build_merge_guide_line(
            best_run["run_rows"],
            best_side,
            search_mask,
            edge_mask,
        )
        if guide_polyline is None:
            return None

        return {
            "side": best_side,
            "guide_polyline": guide_polyline,
        }

    def _detect_edge_trace_merge_guide(self, search_mask, edge_mask):
        """沿左右贴边八连通边缘的连续生长方向找汇合补线触发特征."""
        if not bool(getattr(config, "MERGE_EDGE_TRACE_ENABLED", True)):
            return None
        if search_mask is None or search_mask.size == 0:
            return None

        h, w = search_mask.shape[:2]
        scan_y_top = int(np.clip(int(getattr(config, "MERGE_EDGE_TRACE_SCAN_Y_TOP", 10)), 0, h - 1))
        scan_y_bottom = int(np.clip(int(getattr(config, "MERGE_EDGE_TRACE_SCAN_Y_BOTTOM", 130)), 0, h - 1))
        if scan_y_bottom < scan_y_top:
            scan_y_top, scan_y_bottom = scan_y_bottom, scan_y_top
        touch_distance = max(0, int(getattr(config, "MERGE_EDGE_TRACE_TOUCH_DISTANCE", 10)))
        min_touch_rows = max(1, int(getattr(config, "MERGE_EDGE_TRACE_MIN_TOUCH_ROWS", 3)))
        start_below_rows = max(0, int(getattr(config, "MERGE_EDGE_TRACE_START_BELOW_ROWS", 20)))
        max_steps = max(1, int(getattr(config, "MERGE_EDGE_TRACE_WALK_MAX_STEPS", 260)))
        debug_interval = int(getattr(config, "MERGE_EDGE_TRACE_WALK_DEBUG_INTERVAL", 0))
        match_check_step = max(1, int(getattr(config, "MERGE_EDGE_TRACE_MATCH_CHECK_STEP", 8)))
        debug_max_dirs = max(1, int(getattr(config, "MERGE_EDGE_TRACE_DEBUG_MAX_DIRS", 96)))
        debug_max_runs = max(1, int(getattr(config, "MERGE_EDGE_TRACE_DEBUG_MAX_RUNS", 32)))

        def _compress_runs(codes):
            runs = []
            for code in codes:
                if code is None:
                    continue
                if runs and runs[-1][0] == code:
                    runs[-1][1] += 1
                else:
                    runs.append([int(code), 1])
            return runs

        def _consume_pattern(codes, first_codes, first_count_codes, turn_codes, down_codes):
            min_first = max(1, int(getattr(config, "MERGE_EDGE_TRACE_MIN_LEFT_RUN", 12)))
            min_turn = max(1, int(getattr(config, "MERGE_EDGE_TRACE_MIN_TURN_RUN", 4)))
            min_down = max(1, int(getattr(config, "MERGE_EDGE_TRACE_MIN_DOWN_RUN", 12)))
            max_noise = max(0, int(getattr(config, "MERGE_EDGE_TRACE_PATTERN_MAX_NOISE", 3)))
            n = len(codes)

            def _consume(start_idx, good_codes, count_codes, need):
                idx = start_idx
                count = 0
                noise = 0
                while idx < n:
                    code = codes[idx]
                    if code in good_codes:
                        if code in count_codes:
                            count += 1
                        noise = 0
                        idx += 1
                        continue
                    if count >= need:
                        break
                    if noise >= max_noise:
                        return None
                    noise += 1
                    idx += 1
                if count < need:
                    return None
                return idx

            for i in range(n):
                j = _consume(i, first_codes, first_count_codes, min_first)
                if j is None:
                    continue
                k = _consume(j, turn_codes, turn_codes, min_turn)
                if k is None:
                    continue
                m = _consume(k, down_codes, down_codes, min_down)
                if m is not None:
                    return True
            return False

        def _match_merge_pattern(side, codes):
            """从下方向上爬：右侧匹配 3/2 -> 2/1 -> 0，左侧镜像匹配 1/2 -> 2/3 -> 4。"""
            if not codes:
                return False
            if side == "left":
                return _consume_pattern(codes, (1,), (1,), (2, 3), (4,))
            return _consume_pattern(codes, (3,), (3,), (2, 1), (0,))

        def _walk_edge_codes(side):
            touch_points = []
            for y in range(scan_y_top, scan_y_bottom + 1):
                xs = np.where(edge_mask[y] > 0)[0]
                if len(xs) == 0:
                    continue
                if side == "left":
                    left_x = int(xs[0])
                    if left_x < touch_distance:
                        touch_points.append((left_x, int(y)))
                else:
                    right_x = int(xs[-1])
                    if (w - 1 - right_x) < touch_distance:
                        touch_points.append((right_x, int(y)))
            if len(touch_points) < min_touch_rows:
                return None, f"touch_rows={len(touch_points)}<{min_touch_rows}"

            seed_x, seed_y = max(
                touch_points,
                key=(lambda pt: (pt[1], -pt[0])) if side == "left" else (lambda pt: (pt[1], pt[0])),
            )
            fill_mask = (edge_mask > 0).astype(np.uint8)
            cv2.floodFill(fill_mask, None, (int(seed_x), int(seed_y)), 2)
            component = fill_mask == 2
            ys, xs = np.where(component)
            if len(xs) == 0:
                return None, f"touch_rows={len(touch_points)} empty_component"

            touch_set = {(int(x), int(y)) for x, y in touch_points}
            component_touch_rows = [
                y for x, y in touch_set
                if 0 <= y < h and 0 <= x < w and bool(component[y, x])
            ]
            if len(component_touch_rows) < min_touch_rows:
                return None, f"component_touch_rows={len(component_touch_rows)}<{min_touch_rows}"

            pixels = {(int(x), int(y)) for x, y in zip(xs, ys)}
            bottom_touch_y = max(component_touch_rows)
            target_start_y = min(h - 1, int(bottom_touch_y) + start_below_rows)
            start_candidates = []
            for radius in range(0, start_below_rows + 6):
                row_candidates = []
                for yy in (target_start_y - radius, target_start_y + radius):
                    if yy < 0 or yy >= h:
                        continue
                    xs_on_row = xs[ys == yy]
                    if len(xs_on_row) == 0:
                        continue
                    if side == "left":
                        row_candidates.append((int(np.min(xs_on_row)), int(yy)))
                    else:
                        row_candidates.append((int(np.max(xs_on_row)), int(yy)))
                if row_candidates:
                    start_candidates = row_candidates
                    break
            if not start_candidates:
                return None, f"touch_rows={len(touch_points)} no_start_y={target_start_y}"
            if side == "left":
                curr = min(start_candidates, key=lambda pt: pt[0])
                neighbor_order = [
                    (1, -1, 1), (0, -1, 2), (-1, -1, 3), (-1, 0, 4),
                    (1, 0, 0), (-1, 1, 5), (0, 1, 6), (1, 1, 7),
                ]
            else:
                curr = max(start_candidates, key=lambda pt: pt[0])
                neighbor_order = [
                    (-1, -1, 3), (0, -1, 2), (1, -1, 1), (1, 0, 0),
                    (-1, 0, 4), (1, 1, 7), (0, 1, 6), (-1, 1, 5),
                ]

            start_point = curr
            visited = {curr}
            dirs = []
            matched = False
            for step_idx in range(max_steps):
                cx, cy = curr
                choices = []
                for rank, (dx, dy, code) in enumerate(neighbor_order):
                    nxt = (cx + dx, cy + dy)
                    if nxt in pixels and nxt not in visited:
                        choices.append((rank, abs(dx), code, nxt))
                if not choices:
                    break
                _, _, code, curr = min(choices, key=lambda item: (item[0], item[1]))
                visited.add(curr)
                dirs.append(int(code))
                if len(dirs) >= match_check_step and len(dirs) % match_check_step == 0:
                    matched = _match_merge_pattern(side, dirs)
                elif step_idx + 1 >= max_steps:
                    matched = _match_merge_pattern(side, dirs)
                if matched:
                    break
            return dirs, f"touch_rows={len(touch_points)} start_y={start_point[1]}"

        for side in ("right", "left"):
            dirs, walk_reason = _walk_edge_codes(side)
            if dirs is None:
                if debug_interval > 0:
                    self.merge_edge_trace_debug_counter += 1
                    if self.merge_edge_trace_debug_counter % debug_interval == 0:
                        print(
                            f"[merge_edge_trace][{side}_walk] "
                            f"matched=0 steps=0 reason={walk_reason}",
                            flush=True,
                        )
                continue

            matched = _match_merge_pattern(side, dirs)
            if debug_interval > 0:
                self.merge_edge_trace_debug_counter += 1
                if self.merge_edge_trace_debug_counter % debug_interval == 0:
                    runs = _compress_runs(dirs)
                    dir_text = "".join(str(code) for code in dirs[:debug_max_dirs])
                    if len(dirs) > debug_max_dirs:
                        dir_text += f"...(+{len(dirs) - debug_max_dirs})"
                    shown_runs = runs[:debug_max_runs]
                    run_text = " ".join(f"{code}x{count}" for code, count in shown_runs)
                    if len(runs) > debug_max_runs:
                        run_text += f" ...(+{len(runs) - debug_max_runs})"
                    print(
                        f"[merge_edge_trace][{side}_walk] "
                        f"matched={int(bool(matched))} steps={len(dirs)} reason={walk_reason} "
                        f"dirs={dir_text} runs={run_text}",
                        flush=True,
                    )
            if not matched:
                continue

            guide_polyline = self._build_merge_guide_line([0, 1], side, search_mask, edge_mask)
            if guide_polyline is None:
                continue
            return {
                "side": side,
                "guide_polyline": guide_polyline,
                "source": "edge_trace_walk",
                "direction_codes": [int(code) for code in dirs],
            }
        return None

    def _merge_bottom_width_exit_ready(self, search_mask):
        """底部连续若干行总白区宽度小于阈值时，认为汇合补线可以退出."""
        if search_mask is None or search_mask.size == 0:
            return False

        h, _ = search_mask.shape[:2]
        rows_need = max(1, int(getattr(config, "MERGE_STATE_EXIT_BOTTOM_ROWS", 5)))
        width_thresh = float(getattr(config, "MERGE_STATE_EXIT_WIDTH_THRESH", 340.0))
        start_y = max(0, h - rows_need)
        for y in range(start_y, h):
            xs = np.where(search_mask[y] > 0)[0]
            if len(xs) == 0:
                return False
            row_width = float(xs[-1] - xs[0])
            if row_width >= width_thresh:
                return False
        no_edge_top = int(np.clip(int(getattr(config, "MERGE_STATE_EXIT_NO_EDGE_Y_TOP", 40)), 0, h - 1))
        no_edge_bottom = int(np.clip(int(getattr(config, "MERGE_STATE_EXIT_NO_EDGE_Y_BOTTOM", 150)), 0, h - 1))
        if no_edge_bottom < no_edge_top:
            no_edge_top, no_edge_bottom = no_edge_bottom, no_edge_top
        for y in range(no_edge_top, no_edge_bottom + 1):
            row = search_mask[y]
            if row[0] > 0 or row[-1] > 0:
                return False
        return True

    def _update_merge_state(self, merge_detect_info, search_mask):
        """汇合补线状态机：时间窗口内累计命中确认，底部宽度连续恢复后退出."""
        confirm_frames = max(1, int(getattr(config, "MERGE_STATE_CONFIRM_FRAMES", 3)))
        confirm_window_s = max(0.05, float(getattr(config, "MERGE_STATE_CONFIRM_WINDOW_SECONDS", 0.5)))
        miss_tolerance = max(0, int(getattr(config, "MERGE_STATE_MISS_TOLERANCE_FRAMES", 2)))
        min_hold_s = max(0.0, float(getattr(config, "MERGE_STATE_MIN_HOLD_SECONDS", 2.0)))
        exit_confirm_frames = max(1, int(getattr(config, "MERGE_STATE_EXIT_CONFIRM_FRAMES", 2)))
        now_s = time.perf_counter()

        def _drop_old_hit_times():
            self.merge_state_hit_times = [
                t for t in self.merge_state_hit_times
                if now_s - float(t) <= confirm_window_s
            ]

        detected_side = None if merge_detect_info is None else merge_detect_info.get("side")
        if self.merge_state_active and self.merge_state_side in ("left", "right"):
            if detected_side == self.merge_state_side:
                self.merge_state_hit_frames += 1
                self.merge_state_hit_times.append(now_s)
                _drop_old_hit_times()
                self.merge_state_miss_frames = 0
                self.merge_state_info = merge_detect_info
            elif detected_side is None:
                self.merge_state_miss_frames += 1
                _drop_old_hit_times()
                if self.merge_state_miss_frames > miss_tolerance:
                    self.merge_state_hit_frames = 0
                    self.merge_state_hit_times = []
            else:
                # 当前边还在，不允许直接跳到另一边；先按退出流程走。
                detected_side = None
                merge_detect_info = None
                self.merge_state_hit_frames = 0
                self.merge_state_hit_times = []
                self.merge_state_miss_frames = 0
        else:
            if merge_detect_info is not None:
                self.merge_state_hit_frames += 1
                self.merge_state_hit_times.append(now_s)
                _drop_old_hit_times()
                self.merge_state_miss_frames = 0
                self.merge_state_info = merge_detect_info
            else:
                self.merge_state_miss_frames += 1
                _drop_old_hit_times()
                if self.merge_state_miss_frames > miss_tolerance:
                    self.merge_state_hit_frames = 0
                    self.merge_state_hit_times = []
                    self.merge_state_info = None

        if not self.merge_state_active:
            self.merge_state_exit_frames = 0
            if len(self.merge_state_hit_times) >= confirm_frames and self.merge_state_info is not None:
                self.merge_state_active = True
                self.merge_state_side = self.merge_state_info.get("side")
                self.merge_state_enter_time = now_s
            else:
                return None

        hold_elapsed_s = (
            None if self.merge_state_enter_time is None
            else max(0.0, float(now_s - self.merge_state_enter_time))
        )
        hold_ready = hold_elapsed_s is None or hold_elapsed_s >= min_hold_s
        if hold_ready and self._merge_bottom_width_exit_ready(search_mask):
            self.merge_state_exit_frames += 1
        else:
            self.merge_state_exit_frames = 0

        if self.merge_state_exit_frames >= exit_confirm_frames:
            self.merge_state_active = False
            self.merge_state_hit_frames = 0
            self.merge_state_hit_times = []
            self.merge_state_miss_frames = 0
            self.merge_state_exit_frames = 0
            self.merge_state_enter_time = None
            self.merge_state_info = None
            self.merge_state_side = None
            return None

        if self.merge_state_info is not None:
            self.merge_state_side = self.merge_state_info.get("side")
        return self.merge_state_info

    def _split_mask_by_fork(self, search_mask, fork_point):
        """按“分叉特征点垂直向下”的分界线，把 mask 切成左右两大区域."""
        h, w = search_mask.shape[:2]
        fork_x = float(fork_point[0])

        left_mask = np.zeros_like(search_mask, dtype=np.uint8)
        right_mask = np.zeros_like(search_mask, dtype=np.uint8)

        split_col = int(np.clip(round(fork_x), 0, w - 1))
        for y in range(h):
            left_mask[y, :split_col + 1] = search_mask[y, :split_col + 1]
            right_mask[y, split_col:] = search_mask[y, split_col:]

        return left_mask, right_mask

    def _road_bottom_midpoint(self, search_mask):
        """取道路在画面底部附近的左右中点，作为 fork 拉线的下端点."""
        if search_mask is None or search_mask.size == 0:
            w_seg, h_seg = config.SEG_SIZE
            return (float(w_seg) / 2.0, float(h_seg) - 1.0)

        h, w = search_mask.shape[:2]
        bottom_touch_height = max(1, int(getattr(config, "SEG_PATH_BOTTOM_TOUCH_HEIGHT", 20)))
        min_y = max(0, h - bottom_touch_height)
        min_pixels = max(1, int(getattr(config, "SEG_PATH_MIN_SLICE_PIXELS", 4)))

        for y in range(h - 1, min_y - 1, -1):
            xs = np.where(search_mask[y] > 0)[0]
            if len(xs) >= min_pixels:
                return (0.5 * float(xs[0] + xs[-1]), float(y))

        return (float(w) / 2.0, float(h) - 1.0)

    def _build_fork_centerline_candidate(self, search_mask, fork_point, bottom_mid):
        """构造“岔路点 -> 道路底部中点”的临时中线候选."""
        if search_mask is None or search_mask.size == 0 or fork_point is None or bottom_mid is None:
            return None

        h, w = search_mask.shape[:2]
        fork_x = float(np.clip(float(fork_point[0]), 0.0, float(w - 1)))
        fork_y = float(np.clip(float(fork_point[1]), 0.0, float(h - 1)))
        bottom_x = float(np.clip(float(bottom_mid[0]), 0.0, float(w - 1)))
        bottom_y = float(np.clip(float(bottom_mid[1]), 0.0, float(h - 1)))
        if bottom_y <= fork_y:
            return None

        step_y = max(1, int(getattr(config, "SEG_PATH_SEARCH_STEP_Y", 4)))
        min_nodes = max(2, int(getattr(config, "SEG_PATH_MIN_LENGTH", 3)))
        node_count = max(min_nodes, int(np.ceil((bottom_y - fork_y) / float(step_y))) + 1)
        y_values = np.linspace(bottom_y, fork_y, num=node_count)
        denom = max(1e-6, bottom_y - fork_y)

        nodes = []
        for y in y_values:
            t = (bottom_y - float(y)) / denom
            center_x = (1.0 - t) * bottom_x + t * fork_x
            row_y = int(np.clip(round(float(y)), 0, h - 1))
            xs = np.where(search_mask[row_y] > 0)[0]
            observed_width = 0.0
            if len(xs) >= 2:
                observed_width = float(xs[-1] - xs[0])
            width = self._fixed_track_width_at_y(float(y), observed_width)
            if width <= 0.0:
                width = max(float(getattr(config, "SEG_PATH_MIN_PAIR_WIDTH", 8)) * 2.0, float(w) * 0.25)
            half_width = max(1.0, float(width) * 0.5)
            left_x = float(np.clip(center_x - half_width, 0.0, float(w - 1)))
            right_x = float(np.clip(center_x + half_width, 0.0, float(w - 1)))
            if right_x <= left_x:
                continue
            nodes.append({
                "pt": (float(center_x), float(y)),
                "left_x": left_x,
                "right_x": right_x,
                "width": max(0.0, right_x - left_x),
                "local_center": float(center_x),
                "pair_count": 1,
            })

        if len(nodes) < 2:
            return None

        return {
            "path": np.array([node["pt"] for node in nodes], dtype=np.float32),
            "nodes": nodes,
            "score": 0.0,
            "avg_x": float(np.mean([node["pt"][0] for node in nodes])),
        }

    def _build_row_segments(self, mask_row, edge_row):
        """把单行白区解析成若干个左右边界配对后的通道片段."""
        xs = np.where(mask_row > 0)[0]
        if len(xs) < int(config.SEG_PATH_MIN_SLICE_PIXELS):
            return []

        gap_thresh = int(config.SEG_PATH_GAP_THRESH)
        splits = np.split(xs, np.where(np.diff(xs) > gap_thresh)[0] + 1)
        segments = []
        min_branch_points = int(config.SEG_PATH_MIN_BRANCH_POINTS)
        min_pair_width = int(config.SEG_PATH_MIN_PAIR_WIDTH)

        for segment_idx, split in enumerate(splits):
            if len(split) < min_branch_points:
                continue

            base_left_x = int(split[0])
            base_right_x = int(split[-1])
            left_x = base_left_x
            right_x = base_right_x

            edge_xs = np.where(edge_row[base_left_x:base_right_x + 1] > 0)[0]
            if len(edge_xs) >= 2:
                left_x = int(base_left_x + edge_xs[0])
                right_x = int(base_left_x + edge_xs[-1])

            width = right_x - left_x
            if width < min_pair_width:
                continue

            center_x = 0.5 * (left_x + right_x)
            segments.append({
                "segment_idx": segment_idx,
                "left_x": float(left_x),
                "right_x": float(right_x),
                "center_x": float(center_x),
                "width": float(width),
            })

        max_segments = int(getattr(config, "SEG_PATH_MAX_ROW_SEGMENTS", 0))
        if max_segments > 0 and len(segments) > max_segments:
            segments = sorted(segments, key=lambda seg: float(seg["width"]), reverse=True)[:max_segments]
            segments = sorted(segments, key=lambda seg: float(seg["center_x"]))

        return segments

    def _fixed_track_width_at_y(self, y, fallback_width):
        """读取当前 y 行的固定赛道全宽；无有效标定时回退到当前观测宽度."""
        fallback_width = max(0.0, float(fallback_width))
        if self.fixed_track_widths.size == 0:
            return fallback_width

        idx_float = float(y)
        width_scale = 1.0
        curr_w, curr_h = config.SEG_SIZE
        if self.fixed_track_widths.size != int(curr_h):
            source_w = float(self.fixed_width_source_size[0])
            source_h = float(self.fixed_width_source_size[1])
            source_crop_y = source_h * max(0.0, min(0.95, self.fixed_width_source_crop_top_ratio))
            source_bottom_h = max(1.0, source_h - source_crop_y)
            curr_h_float = max(1.0, float(curr_h))
            if curr_h_float > 1.0:
                idx_float = source_crop_y + (float(y) / (curr_h_float - 1.0)) * (source_bottom_h - 1.0)
            else:
                idx_float = source_crop_y
            if source_w > 0.0:
                width_scale = float(curr_w) / source_w

        idx = int(np.clip(round(idx_float), 0, self.fixed_track_widths.size - 1))
        width = float(self.fixed_track_widths[idx]) * width_scale
        if width > 0.0:
            return width

        if self.fixed_track_width_indices.size > 0:
            nearest_idx = int(
                self.fixed_track_width_indices[
                    int(np.argmin(np.abs(self.fixed_track_width_indices - idx)))
                ]
            )
            nearest_width = float(self.fixed_track_widths[nearest_idx]) * width_scale
            if nearest_width > 0.0:
                return nearest_width

        return fallback_width

    def _build_lateral_fusion_points(self, raw_points, left_boundary, right_boundary, w_seg, trusted_side=None):
        """用当前帧边界宽度估计融合横向误差点，不引入历史帧滞后."""
        if not bool(getattr(config, "PATH_LATERAL_FUSION_ENABLED", False)):
            return None
        if raw_points is None or left_boundary is None or right_boundary is None:
            return None

        raw = np.array(raw_points, dtype=np.float32).reshape((-1, 2))
        if len(raw) < 2:
            return None

        alpha = float(np.clip(float(getattr(config, "PATH_LATERAL_FUSION_ALPHA", 0.0)), 0.0, 1.0))
        if alpha <= 0.0:
            return None
        half_width_ratio = max(0.0, float(getattr(config, "PATH_LATERAL_FUSION_HALF_WIDTH_RATIO", 1.0)))
        max_delta = max(0.0, float(getattr(config, "PATH_LATERAL_FUSION_MAX_DELTA", 0.0)))

        side = str(trusted_side or "left").lower()
        if side not in ("left", "right"):
            side = "left"

        ys = raw[:, 1]
        raw_x = raw[:, 0]
        left_xs = self._interp_path_xs(left_boundary, ys)
        right_xs = self._interp_path_xs(right_boundary, ys)
        if left_xs is None or right_xs is None:
            return None

        boundary_xs = right_xs if side == "right" else left_xs
        observed_widths = np.maximum(0.0, right_xs - left_xs)
        fixed_widths = np.array(
            [
                self._fixed_track_width_at_y(float(y), float(width))
                for y, width in zip(ys, observed_widths)
            ],
            dtype=np.float32,
        )
        valid = fixed_widths > 0.0
        if not np.any(valid):
            return None

        half_widths = 0.5 * fixed_widths * half_width_ratio
        if side == "right":
            boundary_center_x = boundary_xs - half_widths
        else:
            boundary_center_x = boundary_xs + half_widths

        if max_delta > 0.0:
            valid &= np.abs(boundary_center_x - raw_x) <= max_delta

        fused_x = raw_x.copy()
        fused_x[valid] = (1.0 - alpha) * raw_x[valid] + alpha * boundary_center_x[valid]
        fused_x = np.clip(fused_x, 0.0, float(w_seg - 1))
        return np.vstack((fused_x, ys)).astype(np.float32).T

    def _apply_merge_boundary_width(self, nodes, merge_side, guide_polyline=None):
        """单侧出现汇合尖角时，用补线重算通道边界和中心."""
        if merge_side not in ("left", "right") or not nodes:
            return nodes

        guide_points = None
        guide_y_min = None
        guide_y_max = None
        if guide_polyline is not None:
            guide_points = np.array(guide_polyline, dtype=np.float32).reshape((-1, 2))
            if len(guide_points) < 2:
                guide_points = None
            else:
                guide_y_min = float(np.min(guide_points[:, 1]))
                guide_y_max = float(np.max(guide_points[:, 1]))

        node_ys = np.array([float(node["pt"][1]) for node in nodes], dtype=np.float32)
        guide_xs = None
        if guide_points is not None and len(node_ys) > 0:
            guide_xs = self._interp_path_xs(guide_points, node_ys)

        corrected_nodes = []
        for idx, node in enumerate(nodes):
            y = float(node["pt"][1])
            left_x = float(node["left_x"])
            right_x = float(node["right_x"])
            guide_x = None if guide_xs is None else float(guide_xs[idx])
            fixed_width = self._fixed_track_width_at_y(y, right_x - left_x)
            guide_in_range = (
                guide_x is not None and
                guide_y_min is not None and
                guide_y_max is not None and
                guide_y_min <= y <= guide_y_max
            )
            if fixed_width <= 0.0 or not guide_in_range:
                corrected_nodes.append(dict(node))
                continue

            # guide_polyline 就是缺失侧补线。中心线模式下逐行最左/最右点
            # 可能把远处另一条路也包含进来，所以汇合命中后不再用节点外边界。
            if merge_side == "left":
                left_x = guide_x
                right_x = guide_x + fixed_width
            else:
                right_x = guide_x
                left_x = guide_x - fixed_width

            left_x = float(np.clip(left_x, 0.0, float(config.SEG_SIZE[0] - 1)))
            right_x = float(np.clip(right_x, 0.0, float(config.SEG_SIZE[0] - 1)))
            if right_x <= left_x:
                corrected_nodes.append(dict(node))
                continue

            center_x = 0.5 * (left_x + right_x)

            corrected = dict(node)
            corrected["pt"] = (center_x, y)
            corrected["left_x"] = left_x
            corrected["right_x"] = right_x
            corrected["width"] = max(0.0, right_x - left_x)
            corrected["local_center"] = center_x
            corrected_nodes.append(corrected)

        return corrected_nodes

    def _apply_fork_boundary_width(self, nodes, fork_side):
        """Y 岔选定单侧后，用可信外边界按固定赛道宽度补另一侧边界."""
        if fork_side not in ("left", "right") or not nodes:
            return nodes

        w_seg = float(config.SEG_SIZE[0] - 1)
        min_gap = max(0.0, float(getattr(config, "MERGE_GUIDE_LINE_MIN_GAP", 0.0)))
        width_ratio = max(0.0, float(getattr(config, "FORK_BOUNDARY_WIDTH_RATIO", 1.0)))
        corrected_nodes = []

        for node in nodes:
            y = float(node["pt"][1])
            left_x = float(node["left_x"])
            right_x = float(node["right_x"])
            observed_width = max(0.0, right_x - left_x)
            fixed_width = self._fixed_track_width_at_y(y, observed_width) * width_ratio
            if fixed_width <= 0.0:
                corrected_nodes.append(dict(node))
                continue

            if fork_side == "left":
                trusted_x = left_x
                left_x = trusted_x
                right_x = max(trusted_x + fixed_width, trusted_x + min_gap)
            else:
                trusted_x = right_x
                right_x = trusted_x
                left_x = min(trusted_x - fixed_width, trusted_x - min_gap)

            left_x = float(np.clip(left_x, 0.0, w_seg))
            right_x = float(np.clip(right_x, 0.0, w_seg))
            if right_x <= left_x:
                corrected_nodes.append(dict(node))
                continue

            center_x = 0.5 * (left_x + right_x)
            corrected = dict(node)
            corrected["pt"] = (center_x, y)
            corrected["left_x"] = left_x
            corrected["right_x"] = right_x
            corrected["width"] = max(0.0, right_x - left_x)
            corrected["local_center"] = center_x
            corrected_nodes.append(corrected)

        return corrected_nodes

    def _search_active_paths(self, search_mask, edge_mask, max_active_paths=None):
        """在给定 mask 区域内做一次自底向上的多候选路径搜索."""
        h_seg, _ = search_mask.shape[:2]
        step_y = int(config.SEG_PATH_SEARCH_STEP_Y)
        if max_active_paths is None:
            max_active_paths = int(config.SEG_PATH_MAX_ACTIVE_PATHS)
        max_active_paths = max(1, int(max_active_paths))
        active_paths = []
        branch_pair_count_max = 0
        branch_support_rows = 0

        fallback_top_y = int(h_seg * float(config.SEG_PATH_SCAN_TOP_RATIO))
        fallback_top_y = int(np.clip(fallback_top_y, 0, h_seg - 1))
        mask_rows = np.where(np.any(search_mask > 0, axis=1))[0]
        if len(mask_rows) > 0:
            top_y = int(mask_rows[0])
        else:
            top_y = fallback_top_y

        work_mask = search_mask.copy()
        work_edge = edge_mask.copy()
        if top_y > 0:
            work_mask[:top_y, :] = 0
            work_edge[:top_y, :] = 0

        bottom_touch_height = max(1, int(config.SEG_PATH_BOTTOM_TOUCH_HEIGHT))
        start_segments = []
        start_row_y = None
        min_start_y = max(0, h_seg - bottom_touch_height)
        # 只要在“图像真实底部 20 行”里触达过，就允许把它当作有效起始路径。
        # 这里不再被 BOTTOM_MARGIN 缩掉，避免“明明已经贴近底部，但因为预留边距没扫到”。
        for sample_y in range(h_seg - 1, min_start_y - 1, -1):
            row_segments = self._build_row_segments(work_mask[sample_y], work_edge[sample_y])
            if not row_segments:
                continue
            start_segments = row_segments
            start_row_y = sample_y
            break

        if not start_segments:
            return active_paths, branch_pair_count_max, branch_support_rows

        branch_pair_count_max = max(branch_pair_count_max, len(start_segments))
        if len(start_segments) >= 2:
            branch_support_rows += 1

        for segment in start_segments:
            active_paths.append([{
                "pt": (segment["center_x"], float(start_row_y)),
                "left_x": segment["left_x"],
                "right_x": segment["right_x"],
                "width": segment["width"],
                "local_center": segment["center_x"],
                "pair_count": len(start_segments),
            }])
        if len(active_paths) > max_active_paths:
            active_paths = self._prune_active_paths(active_paths, max_active_paths)

        curr_y = start_row_y - step_y
        while curr_y >= top_y:
            row_start = max(0, curr_y - step_y // 2)
            row_end = min(h_seg, curr_y + step_y // 2 + 1)

            best_row_y = None
            best_segments = []
            best_score = (-1, -1.0)
            for sample_y in range(row_start, row_end):
                row_segments = self._build_row_segments(work_mask[sample_y], work_edge[sample_y])
                score = (len(row_segments), float(sum(seg["width"] for seg in row_segments)))
                if score > best_score:
                    best_score = score
                    best_segments = row_segments
                    best_row_y = sample_y

            if not best_segments:
                curr_y -= step_y
                continue

            branch_pair_count_max = max(branch_pair_count_max, len(best_segments))
            if len(best_segments) >= 2:
                branch_support_rows += 1

            if not active_paths:
                for segment in best_segments:
                    active_paths.append([{
                        "pt": (segment["center_x"], float(best_row_y)),
                        "left_x": segment["left_x"],
                        "right_x": segment["right_x"],
                        "width": segment["width"],
                        "local_center": segment["center_x"],
                        "pair_count": len(best_segments),
                    }])
                if len(active_paths) > max_active_paths:
                    active_paths = self._prune_active_paths(active_paths, max_active_paths)
            else:
                new_paths = []

                for path in active_paths:
                    last_node = path[-1]
                    matched = False
                    for segment in best_segments:
                        if not self._segments_can_connect(last_node, segment):
                            continue

                        node = {
                            "pt": (segment["center_x"], float(best_row_y)),
                            "left_x": segment["left_x"],
                            "right_x": segment["right_x"],
                            "width": segment["width"],
                            "local_center": segment["center_x"],
                            "pair_count": len(best_segments),
                        }
                        new_paths.append(path + [node])
                        matched = True

                    if not matched:
                        new_paths.append(path)

                # 这里严格要求“候选路径必须从底部起步”：
                # 上层新出现但和底部既有路径连不上的片段，直接丢弃，
                # 不能在中途重新开一路，否则会把悬空分支误当成可走路径。
                active_paths = new_paths
                if len(active_paths) > max_active_paths:
                    active_paths = self._prune_active_paths(active_paths, max_active_paths)

            curr_y -= step_y

        return active_paths, branch_pair_count_max, branch_support_rows

    def _build_centerline_candidate(self, search_mask, edge_mask):
        """取最大连通区域，在该区域每行直接取左右边界中点."""
        if search_mask is None or search_mask.size == 0:
            return None

        work_mask = (search_mask > 0).astype(np.uint8)
        if bool(getattr(config, "SEG_CENTERLINE_LARGEST_COMPONENT_ONLY", True)):
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(work_mask, connectivity=8)
            if num_labels <= 1:
                return None

            areas = stats[1:, cv2.CC_STAT_AREA]
            if len(areas) == 0:
                return None
            largest_label = int(np.argmax(areas)) + 1
            work_mask = (labels == largest_label).astype(np.uint8)

        mask_rows = np.where(np.any(work_mask > 0, axis=1))[0]
        if len(mask_rows) == 0:
            return None

        top_y = int(mask_rows[0])
        bottom_y = int(mask_rows[-1])
        row_step = max(1, int(getattr(config, "SEG_CENTERLINE_ROW_STEP", 1)))

        nodes = []
        for sample_y in range(bottom_y, top_y - 1, -row_step):
            xs = np.where(work_mask[sample_y] > 0)[0]
            if len(xs) < int(config.SEG_PATH_MIN_BRANCH_POINTS):
                continue

            left_x = float(xs[0])
            right_x = float(xs[-1])
            width = right_x - left_x
            center_x = 0.5 * (left_x + right_x)
            nodes.append({
                "pt": (center_x, float(sample_y)),
                "left_x": left_x,
                "right_x": right_x,
                "width": width,
                "local_center": center_x,
                "pair_count": 1,
            })

        if len(nodes) < int(config.SEG_PATH_MIN_LENGTH):
            return None

        path_arr = np.array([node["pt"] for node in nodes], dtype=np.float32)
        return {
            "path": path_arr,
            "nodes": nodes,
            "score": float(len(nodes)),
            "base_score": float(len(nodes)),
            "temporal_penalty": 0.0,
            "temporal_mean_jump": 0.0,
            "temporal_max_jump": 0.0,
            "avg_x": float(np.mean(path_arr[:, 0])),
            "branch_pair_count_max": 1,
            "branch_support_rows": 0,
        }

    def _score_candidate_paths(self, active_paths):
        """对候选路径列表打分，返回可参与最终决策的候选池."""
        valid_candidates = []
        for path in active_paths:
            if len(path) < int(config.SEG_PATH_MIN_LENGTH):
                continue

            path_arr = np.array([node["pt"] for node in path], dtype=np.float32)
            px = path_arr[:, 0]
            widths = np.array([node["width"] for node in path], dtype=np.float32)

            length_score = len(path) * float(config.SEG_PATH_LENGTH_SCORE_GAIN)
            dx = np.diff(px)
            smooth_score = -np.std(dx) * float(config.SEG_PATH_SMOOTH_SCORE_GAIN)

            center_penalty = 0.0
            for node in path:
                center_penalty += abs(node["pt"][0] - node["local_center"]) * float(config.SEG_PATH_CENTER_PENALTY_GAIN)

            width_score = 0.05 * float(np.mean(widths)) if len(widths) > 0 else 0.0
            base_score = length_score + smooth_score + width_score - center_penalty
            temporal_stats = self._temporal_path_stats(path_arr)
            score = base_score - float(temporal_stats["penalty"])
            avg_x = float(np.mean(px))

            valid_candidates.append({
                "path": path_arr,
                "nodes": path,
                "score": score,
                "base_score": base_score,
                "temporal_penalty": float(temporal_stats["penalty"]),
                "temporal_mean_jump": float(temporal_stats["mean_jump"]),
                "temporal_max_jump": float(temporal_stats["max_jump"]),
                "avg_x": avg_x,
            })

        return valid_candidates

    def _build_merge_guide_line(self, merge_run, side_name, search_mask, edge_mask):
        """按完整赛道宽度，从八邻域边界提取出的可信侧边界反推缺失侧边界."""
        if merge_run is None:
            return None

        rows = list(merge_run)
        if len(rows) < 2:
            return None

        h, w = search_mask.shape[:2]
        min_gap = max(0.0, float(config.MERGE_GUIDE_LINE_MIN_GAP))

        guide_pts = []
        for y_int in range(0, h):
            xs = np.where(search_mask[y_int] > 0)[0]
            if len(xs) < int(config.SEG_PATH_MIN_BRANCH_POINTS):
                continue

            y = float(y_int)
            row_segments = self._build_row_segments(search_mask[y_int], edge_mask[y_int])
            if not row_segments:
                continue

            leftmost_seg = min(row_segments, key=lambda seg: float(seg["center_x"]))
            rightmost_seg = max(row_segments, key=lambda seg: float(seg["center_x"]))
            trusted_left_x = float(leftmost_seg["left_x"])
            trusted_right_x = float(rightmost_seg["right_x"])
            observed_width = float(xs[-1]) - float(xs[0])
            fixed_width = self._fixed_track_width_at_y(y, observed_width)
            if fixed_width <= 0.0:
                continue

            if side_name == "left":
                x = trusted_right_x - fixed_width
                x = min(x, trusted_right_x - min_gap)
            else:
                x = trusted_left_x + fixed_width
                x = max(x, trusted_left_x + min_gap)
            if x < 0.0 or x > float(w - 1):
                continue
            guide_pts.append([float(x), y])

        if len(guide_pts) < 2:
            return None
        return np.array(guide_pts, dtype=np.float32).reshape((-1, 1, 2))

    def _apply_merge_guide(self, search_mask, guide_polyline):
        """把汇合引导线补到搜索用 mask 中，帮助按单路模式继续搜路."""
        if guide_polyline is None:
            return search_mask

        guided_mask = search_mask.copy()
        thickness = max(
            int(config.MERGE_GUIDE_LINE_THICKNESS),
            int(config.SEG_PATH_MIN_PAIR_WIDTH) + 1,
        )
        cv2.polylines(
            guided_mask,
            [guide_polyline.astype(np.int32)],
            False,
            1,
            thickness,
            cv2.LINE_AA,
        )
        return (guided_mask > 0).astype(np.uint8)

    def _segments_can_connect(self, prev_node, curr_segment):
        """判断上下两层的通道片段是否属于同一条候选路径."""
        prev_center = float(prev_node["pt"][0])
        curr_center = float(curr_segment["center_x"])
        if abs(curr_center - prev_center) > int(config.SEG_PATH_CONNECT_X_THRESH):
            return False

        margin = float(config.SEG_PATH_CONNECT_OVERLAP_MARGIN)
        prev_left = float(prev_node["left_x"])
        prev_right = float(prev_node["right_x"])
        curr_left = float(curr_segment["left_x"])
        curr_right = float(curr_segment["right_x"])

        overlap = min(prev_right, curr_right) - max(prev_left, curr_left)
        if overlap >= -margin:
            return True

        return False

    def _path_x_at_y(self, candidate, target_y):
        """在候选路径上找到最接近 target_y 的横坐标，统一在分割平面里比较."""
        path = candidate.get("path")
        if path is None or len(path) == 0:
            return float(candidate.get("avg_x", 0.0))

        path = np.array(path, dtype=np.float32)
        ys = path[:, 1]
        xs = path[:, 0]
        idx = int(np.argmin(np.abs(ys - float(target_y))))
        return float(xs[idx])

    def _prune_active_paths(self, active_paths, max_paths=None):
        """裁剪候选路径数量，同时尽量保住左右代表分支."""
        if max_paths is None:
            max_paths = int(config.SEG_PATH_MAX_ACTIVE_PATHS)
        max_paths = max(1, int(max_paths))
        if len(active_paths) <= max_paths:
            return active_paths

        def _path_quality(path):
            if not path:
                return (0, -1e9, 0.0)

            xs = np.array([float(node["pt"][0]) for node in path], dtype=np.float32)
            widths = np.array([float(node["width"]) for node in path], dtype=np.float32)
            smooth_score = 0.0
            if len(xs) > 1:
                smooth_score = -float(np.std(np.diff(xs)))
            mean_width = float(np.mean(widths)) if len(widths) > 0 else 0.0
            return (len(path), smooth_score, mean_width)

        ranked_paths = sorted(active_paths, key=_path_quality, reverse=True)
        if max_paths == 1:
            return ranked_paths[:1]

        kept_paths = []
        kept_ids = set()

        def _keep(path):
            path_id = id(path)
            if path_id in kept_ids:
                return
            kept_ids.add(path_id)
            kept_paths.append(path)

        if len(ranked_paths) >= 2:
            left_path = min(ranked_paths, key=lambda p: float(p[-1]["pt"][0]))
            right_path = max(ranked_paths, key=lambda p: float(p[-1]["pt"][0]))
            if abs(float(right_path[-1]["pt"][0]) - float(left_path[-1]["pt"][0])) >= float(config.PATH_LOCK_FORK_MIN_SEP):
                _keep(left_path)
                _keep(right_path)

        for path in ranked_paths:
            if len(kept_paths) >= max_paths:
                break
            _keep(path)

        return kept_paths[:max_paths]

    def _select_fork_representatives(self, candidate_paths, branch_support_rows):
        """在明显岔路里提取左右代表候选，避免某一支在筛选前被丢掉."""
        if branch_support_rows <= 0 or len(candidate_paths) < 2:
            return None, None, False

        sorted_by_x = sorted(candidate_paths, key=lambda c: float(c["avg_x"]))
        leftmost = sorted_by_x[0]
        rightmost = sorted_by_x[-1]

        fork_sep = float(rightmost["avg_x"]) - float(leftmost["avg_x"])
        if leftmost is rightmost or fork_sep < float(config.PATH_LOCK_FORK_MIN_SEP):
            return None, None, False

        split_x = 0.5 * (float(leftmost["avg_x"]) + float(rightmost["avg_x"]))
        left_group = [c for c in candidate_paths if float(c["avg_x"]) <= split_x]
        right_group = [c for c in candidate_paths if float(c["avg_x"]) >= split_x]

        if not left_group or not right_group:
            return None, None, False

        left_repr = max(left_group, key=lambda c: (float(c["score"]), -float(c["avg_x"])))
        right_repr = max(right_group, key=lambda c: (float(c["score"]), float(c["avg_x"])))

        if left_repr is right_repr:
            return None, None, False

        return left_repr, right_repr, True

    def _resolve_preferred_turn(self, turn_intent=-1):
        """得到最终分支选择；没有有效语义方向时默认走左支."""
        if int(turn_intent) in (-1, 1):
            return int(turn_intent)
        return -1

    def _get_planning_box_points(self, obj, w_seg, h_seg):
        """把原图检测框转换成分割平面中的整框四角点."""
        rect = obj.get("rect", [0, 0, 0, 0])
        class_name = obj.get("class_name", "")

        if class_name not in self.planning_class_names or len(rect) != 4:
            return None

        x, y, w, h = rect
        if w <= 1 or h <= 1:
            return None
        if float(y) + float(h) <= self.seg_crop_top_target_y:
            return None

        corners = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ], dtype=np.float32)

        corners[:, 0] *= self.scale_x_to_seg
        corners[:, 1] = (corners[:, 1] - self.seg_crop_top_target_y) * self.scale_y_to_seg
        corners[:, 0] = np.clip(corners[:, 0], 0, w_seg - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, h_seg - 1)
        return corners

    def _project_planning_objects(self, current_yolo_boxes, w_seg, h_seg):
        """将原图中的规划相关检测框映射到分割平面.

        这些目标当前主要用于:
        - 调试显示
        暂时不会像 cost map 那样直接侵蚀主路径。
        """
        planning_items = []

        for obj in current_yolo_boxes:
            seg_box = self._get_planning_box_points(obj, w_seg, h_seg)
            if seg_box is None:
                continue

            planning_items.append({
                "class_name": obj.get("class_name", "obj"),
                "class_id": obj.get("class_id", -1),
                "score": float(obj.get("score", 0.0)),
                "seg_box": seg_box,
            })

        return planning_items

    def _path_x_at_y_points(self, path_points, y):
        """在一组路径点上按 y 插值得到 x."""
        pts = np.array(path_points, dtype=np.float32).reshape((-1, 2))
        if len(pts) == 0:
            return None

        order = np.argsort(pts[:, 1])
        ys = pts[order, 1]
        xs = pts[order, 0]
        unique_ys, unique_indices = np.unique(ys, return_index=True)
        unique_xs = xs[unique_indices]
        if len(unique_ys) == 0:
            return None
        if len(unique_ys) == 1:
            return float(unique_xs[0])
        return float(np.interp(float(y), unique_ys, unique_xs))

    def _blend_paths(self, base_path, target_path, blend):
        """按 blend 把 target_path 混回 base_path."""
        base = np.array(base_path, dtype=np.float32).reshape((-1, 2))
        target = np.array(target_path, dtype=np.float32).reshape((-1, 2))
        if len(base) == 0:
            return target
        if len(base) != len(target):
            return target
        blend = float(np.clip(blend, 0.0, 1.0))
        return (base * (1.0 - blend) + target * blend).astype(np.float32)

    def _build_car_clearing_path(self, base_path, avoid_path, clear_frames):
        """CLEARING: 把最后一次避车路径逐帧混回正常循线路径."""
        base = np.array(base_path, dtype=np.float32).reshape((-1, 2))
        if len(base) < 2:
            return base, 0.0
        avoid = np.array(avoid_path, dtype=np.float32).reshape((-1, 2))
        if len(avoid) != len(base):
            avoid = base.copy()

        hold_frames = max(0, int(getattr(
            config,
            "CAR_AVOIDANCE_CLEARING_HOLD_FRAMES",
            getattr(config, "CAR_AVOIDANCE_CLEARING_MISS_FRAMES", 0),
        )))
        return_frames = max(1, int(getattr(
            config,
            "CAR_AVOIDANCE_CLEARING_RETURN_FRAMES",
            getattr(config, "CAR_AVOIDANCE_CLEARING_DECAY_FRAMES", 5),
        )))
        if clear_frames <= hold_frames:
            avoid_weight = 1.0
        else:
            t = float(np.clip((clear_frames - hold_frames) / float(return_frames), 0.0, 1.0))
            avoid_weight = 1.0 - t

        mixed = self._blend_paths(base, avoid, avoid_weight)
        mixed[:, 0] = np.clip(mixed[:, 0], 0.0, float(config.SEG_SIZE[0] - 1))
        return mixed, float(avoid_weight)

    def _build_boundary_inset_path(self, base_path, boundary, inset_x, w_seg):
        """把控制基准切到指定边界向中线内收后的路径."""
        base = np.array(base_path, dtype=np.float32).reshape((-1, 2))
        if len(base) < 2 or boundary is None:
            return base, False

        boundary_xs = self._interp_path_xs(boundary, base[:, 1])
        if boundary_xs is None or len(boundary_xs) != len(base):
            return base, False

        inset = max(0.0, float(inset_x))
        to_center = base[:, 0] - boundary_xs
        direction = np.sign(to_center)
        direction[direction == 0.0] = 1.0
        step = np.minimum(np.abs(to_center), inset)

        planned = base.copy()
        planned[:, 0] = boundary_xs + direction * step
        planned[:, 0] = np.clip(planned[:, 0], 0.0, float(w_seg - 1))
        return planned.astype(np.float32), True

    def _build_car_boundary_path(self, base_path, boundary, inset_x, w_seg):
        """按选定边界生成避车参考线."""
        return self._build_boundary_inset_path(base_path, boundary, inset_x, w_seg)

    def _build_car_left_boundary_path(self, base_path, left_boundary, inset_x, w_seg):
        """避车时把左边线本身当作控制中线。"""
        return self._build_car_boundary_path(base_path, left_boundary, 0.0, w_seg)

    def _build_car_right_boundary_path(self, base_path, right_boundary, inset_x, w_seg):
        """避车时把右边线本身当作控制中线。"""
        return self._build_car_boundary_path(base_path, right_boundary, 0.0, w_seg)

    def _car_control_path_error_for_debug(self, path_points, w_seg):
        """返回避车参考线在 Stanley 前视行上的实际控制误差."""
        path_x = self._path_x_at_y_points(
            path_points,
            float(getattr(config, "STANLEY_LOOKAHEAD_Y", 100.0)),
        )
        if path_x is None:
            return None
        return float(path_x) - float(w_seg) * 0.5

    def _boundary_x_at_y(self, boundary_points, y):
        """按 y 在左右边界点上插值得到边界 x."""
        if boundary_points is None:
            return None
        pts = np.array(boundary_points, dtype=np.float32).reshape((-1, 2))
        if len(pts) == 0:
            return None
        xs = self._interp_path_xs(pts, np.array([float(y)], dtype=np.float32))
        if xs is None or len(xs) == 0:
            return None
        return float(xs[0])

    def _select_car_avoidance_boundary_side(self, locked_car, left_boundary, right_boundary):
        """根据车辆更靠近哪条边界，选择更空的一侧做避车参考线.

        返回:
        - avoid_side: "left" / "right"
        - boundary_x: 当前选中边界在车所在 y 的 x 值
        - boundary_error: boundary_x - 画面中线
        - nearest_side: 车辆更接近的边界侧；None 表示无法判断
        """
        if locked_car is None:
            return "left", None, None, None

        _smooth_cx, smooth_cy = locked_car.get("bottom_center", (0.0, 0.0))
        car_x = float(_smooth_cx)
        left_x = self._boundary_x_at_y(left_boundary, float(smooth_cy)) if left_boundary is not None else None
        right_x = self._boundary_x_at_y(right_boundary, float(smooth_cy)) if right_boundary is not None else None
        target_x = float(config.SEG_SIZE[0]) * 0.5

        if left_x is not None and right_x is not None:
            left_dist = abs(car_x - float(left_x))
            right_dist = abs(float(right_x) - car_x)
            if left_dist <= right_dist:
                return "right", float(right_x), float(right_x) - target_x, "left"
            return "left", float(left_x), float(left_x) - target_x, "right"

        if left_x is not None:
            return "left", float(left_x), float(left_x) - target_x, "left"
        if right_x is not None:
            return "right", float(right_x), float(right_x) - target_x, "right"
        return "left", None, None, None

    def _car_measurement_from_item(self, item, w_seg, h_seg, base):
        """从 car 检测框提取用于跟踪和避障的稳定几何量."""
        if item.get("class_name", "") != "car":
            return None
        if float(item.get("score", 0.0)) < float(getattr(config, "CAR_AVOIDANCE_MIN_SCORE", 0.0)):
            return None
        seg_box = item.get("seg_box")
        if seg_box is None:
            return None
        box = np.array(seg_box, dtype=np.float32).reshape((-1, 2))
        if len(box) < 4:
            return None

        x_min = float(np.min(box[:, 0]))
        x_max = float(np.max(box[:, 0]))
        y_min = float(np.min(box[:, 1]))
        y_max = float(np.max(box[:, 1]))
        area = float(max(0.0, x_max - x_min) * max(0.0, y_max - y_min))
        max_area = float(getattr(config, "CAR_AVOIDANCE_MAX_AREA", 0.0))
        if max_area > 0.0 and area > max_area:
            return None

        y_sorted = np.argsort(box[:, 1])
        bottom_pts = box[y_sorted[-2:]]
        top_pts = box[y_sorted[:2]]
        raw_bottom_y = float(np.max(bottom_pts[:, 1]))
        raw_top_y = float(np.min(top_pts[:, 1]))
        bottom_center_x = float(np.mean(bottom_pts[:, 0]))
        bottom_center_y = raw_bottom_y
        path_y_min = float(np.min(base[:, 1]))
        path_y_max = float(np.max(base[:, 1]))
        bottom_center_y = float(np.clip(bottom_center_y, path_y_min, path_y_max))

        return {
            "box": box,
            "score": float(item.get("score", 0.0)),
            "area": area,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "bottom_pts": bottom_pts,
            "top_pts": top_pts,
            "bottom_center": (bottom_center_x, bottom_center_y),
            "raw_bottom_y": raw_bottom_y,
            "raw_top_y": raw_top_y,
        }

    def _update_locked_car(self, measurements, base):
        """按底部中心点连续性锁定同一辆车，允许短暂遮挡/漏检."""
        max_miss = max(0, int(getattr(config, "CAR_AVOIDANCE_MISS_FRAMES", 0)))
        near_max_miss = max(0, int(getattr(config, "CAR_AVOIDANCE_NEAR_MISS_FRAMES", max_miss)))
        near_rows = max(0.0, float(getattr(config, "CAR_AVOIDANCE_NEAR_BOUNDARY_ROWS", 60.0)))
        commit_rows = max(0.0, float(getattr(config, "CAR_AVOIDANCE_COMMIT_ROWS", 0.0)))
        hit_required = max(1, int(getattr(config, "CAR_AVOIDANCE_LOCK_HIT_FRAMES", 1)))
        search_radius = max(1.0, float(getattr(config, "CAR_AVOIDANCE_SEARCH_RADIUS", 48.0)))
        miss_gain = max(0.0, float(getattr(config, "CAR_AVOIDANCE_SEARCH_RADIUS_MISS_GAIN", 0.0)))
        max_y_regression = max(0.0, float(getattr(config, "CAR_AVOIDANCE_MAX_Y_REGRESSION", 0.0)))
        ema_alpha = float(getattr(config, "CAR_AVOIDANCE_TRACK_EMA_ALPHA", 0.65))
        ema_alpha = float(np.clip(ema_alpha, 0.0, 0.98))
        miss_x_predict_gain = float(np.clip(
            float(getattr(config, "CAR_AVOIDANCE_MISS_X_PREDICT_GAIN", 0.0)),
            0.0,
            1.0,
        ))

        if self.locked_car is not None:
            last_center = self.locked_car.get("bottom_center")
            velocity = self.locked_car.get("velocity", (0.0, 0.0))
            miss_frames = int(self.locked_car_miss_frames)
            path_bottom_y = float(np.max(base[:, 1]))
            rows_to_car = max(0.0, path_bottom_y - float(last_center[1]))
            near_committed = bool(self.locked_car.get("near_committed", False)) or (
                commit_rows > 0.0 and rows_to_car <= commit_rows
            )
            predicted = (
                float(last_center[0]) + float(velocity[0]) * miss_x_predict_gain,
                float(last_center[1]) + float(velocity[1]),
            )
            radius = search_radius + miss_gain * float(miss_frames)

            local_matches = []
            for m in measurements:
                mx, my = m["bottom_center"]
                if max_y_regression > 0.0 and float(my) < float(last_center[1]) - max_y_regression:
                    continue
                dist = float(np.hypot(float(mx) - predicted[0], float(my) - predicted[1]))
                if dist <= radius:
                    local_matches.append((dist, m))

            if local_matches:
                _dist, best = min(
                    local_matches,
                    key=lambda pair: (pair[0], -float(pair[1].get("score", 0.0))),
                )
                old_center = self.locked_car["bottom_center"]
                new_center = best["bottom_center"]
                smooth_center = (
                    ema_alpha * float(old_center[0]) + (1.0 - ema_alpha) * float(new_center[0]),
                    ema_alpha * float(old_center[1]) + (1.0 - ema_alpha) * float(new_center[1]),
                )
                smooth_velocity = (
                    smooth_center[0] - float(old_center[0]),
                    smooth_center[1] - float(old_center[1]),
                )
                self.locked_car = {
                    "measurement": best,
                    "bottom_center": smooth_center,
                    "velocity": smooth_velocity,
                    "hit_frames": int(self.locked_car.get("hit_frames", 0)) + 1,
                    "confirmed": int(self.locked_car.get("hit_frames", 0)) + 1 >= hit_required,
                    "near_committed": bool(near_committed),
                }
                self.locked_car_miss_frames = 0
                if not self.locked_car["confirmed"]:
                    return None
                return self.locked_car

            if near_committed and measurements and bool(self.locked_car.get("confirmed", False)):
                path_y_min = float(np.min(base[:, 1]))
                path_y_max = float(np.max(base[:, 1]))
                predicted_center = (
                    float(np.clip(predicted[0], 0.0, float(config.SEG_SIZE[0] - 1))),
                    float(np.clip(predicted[1], path_y_min, path_y_max)),
                )
                self.locked_car = dict(self.locked_car)
                self.locked_car["bottom_center"] = predicted_center
                self.locked_car["near_committed"] = True
                self.locked_car["miss_frames"] = 0
                self.locked_car_miss_frames = 0
                return self.locked_car

            self.locked_car_miss_frames += 1
            allowed_miss = near_max_miss if rows_to_car <= near_rows else max_miss
            if self.locked_car_miss_frames <= allowed_miss:
                if not bool(self.locked_car.get("confirmed", False)):
                    return None
                path_y_min = float(np.min(base[:, 1]))
                path_y_max = float(np.max(base[:, 1]))
                predicted_center = (
                    float(np.clip(predicted[0], 0.0, float(config.SEG_SIZE[0] - 1))),
                    float(np.clip(predicted[1], path_y_min, path_y_max)),
                )
                self.locked_car = dict(self.locked_car)
                self.locked_car["bottom_center"] = predicted_center
                self.locked_car["miss_frames"] = int(self.locked_car_miss_frames)
                self.locked_car["near_committed"] = bool(near_committed)
                return self.locked_car

            self.locked_car = None
            self.locked_car_miss_frames = 0

        if not measurements:
            return None

        # 新目标优先选靠近车身底部、且离基础路径更近的 car。
        def _new_target_key(m):
            cx, cy = m["bottom_center"]
            path_x = self._path_x_at_y_points(base, cy)
            lateral = abs(float(cx) - float(path_x)) if path_x is not None else 0.0
            return (float(cy), -lateral, float(m.get("score", 0.0)))

        best = max(measurements, key=_new_target_key)
        self.locked_car = {
            "measurement": best,
            "bottom_center": tuple(map(float, best["bottom_center"])),
            "velocity": (0.0, 0.0),
            "hit_frames": 1,
            "confirmed": hit_required <= 1,
            "near_committed": False,
        }
        self.locked_car_miss_frames = 0
        if not self.locked_car["confirmed"]:
            return None
        return self.locked_car

    def _car_avoidance_boundary_inset(self, locked_car, base):
        """基于锁定目标决定左边界内收量."""
        if locked_car is None:
            return 0.0, None, False
        measurement = locked_car.get("measurement")
        if measurement is None:
            return 0.0, None, False

        path_y_min = float(np.min(base[:, 1]))
        path_y_max = float(np.max(base[:, 1]))
        _smooth_cx, smooth_cy = locked_car.get("bottom_center", measurement["bottom_center"])
        center_y = float(np.clip(smooth_cy, path_y_min, path_y_max))
        path_bottom_y = float(np.max(base[:, 1]))
        rows_to_car = max(0.0, path_bottom_y - center_y)
        start_rows = max(0.0, float(getattr(config, "CAR_AVOIDANCE_START_BOUNDARY_ROWS", 115.0)))
        if rows_to_car > start_rows:
            return 0.0, (center_y, path_bottom_y), False

        # 进入避障窗口后，把选定侧边界本身当作控制中线。
        return 0.0, (center_y, path_bottom_y), True

    def _update_car_avoidance_boundary_path(
        self,
        base_path,
        planning_items,
        w_seg,
        h_seg,
        left_boundary=None,
        right_boundary=None,
        state_updates_enabled=True,
    ):
        """检测到 car 时锁定同一辆车，并把控制参考线切到更空的一侧边界."""
        debug = {
            "blocked_y_range": None,
            "boundary_inset_x": 0.0,
            "boundary_strength_x": 0.0,
            "boundary_path_active": False,
            "boundary_ready": False,
            "boundary_side": self.car_avoidance_boundary_side,
            "boundary_x": None,
            "boundary_error": None,
            "nearest_boundary_side": None,
            "avoid_weight": 0.0,
            "rows_to_bottom": None,
            "left_boundary_error": None,
            "left_boundary_x": None,
            "right_boundary_error": None,
            "right_boundary_x": None,
            "control_path_error": None,
            "detected_cars": 0,
            "locked_confirmed": False,
            "locked_hit_frames": 0,
            "locked_center": None,
            "miss_frames": int(self.locked_car_miss_frames),
            "state": self.car_avoidance_state,
            "clear_frames": int(self.car_clearing_frames),
            "active": False,
            "event": "",
            "cycle_id": int(self.car_avoidance_cycle_id),
        }
        base = np.array(base_path, dtype=np.float32).reshape((-1, 2))
        prev_state = self.car_avoidance_state
        if len(base) < 2 or not bool(getattr(config, "CAR_AVOIDANCE_ENABLED", True)):
            return 0.0, debug, None

        max_cycles = int(getattr(config, "CAR_AVOIDANCE_MAX_CYCLES", 0))
        if (
            max_cycles > 0 and
            self.car_avoidance_state == "FOLLOW_LANE" and
            int(self.car_avoidance_cycle_id) >= max_cycles
        ):
            self.locked_car = None
            self.locked_car_miss_frames = 0
            measurements = []
            for item in planning_items:
                measurement = self._car_measurement_from_item(item, w_seg, h_seg, base)
                if measurement is not None:
                    measurements.append(measurement)
            debug["detected_cars"] = int(len(measurements))
            debug["state"] = self.car_avoidance_state
            debug["event"] = "cycle_limit"
            if measurements:
                throttled_log_key = "car_avoidance_cycle_limit"
                now = time.monotonic()
                interval = max(0.2, float(getattr(config, "LOG_INTERVAL_CAR_AVOIDANCE_PROCESS", 0.25)))
                if (
                    self.last_car_avoid_log_state != throttled_log_key or
                    now - float(self.last_car_avoid_log_at) >= interval
                ):
                    self.last_car_avoid_log_state = throttled_log_key
                    self.last_car_avoid_log_at = now
                    print(
                        f"避车次数已达上限: 已避车{int(self.car_avoidance_cycle_id)}次，忽略后续车辆 det={len(measurements)}",
                        flush=True,
                    )
            return 0.0, debug, None

        def _build_avoid_path_for_side(boundary_side, inset_value=0.0):
            if boundary_side == "right":
                if right_boundary is not None:
                    return self._build_car_right_boundary_path(base, right_boundary, inset_value, w_seg)
                if left_boundary is not None:
                    return self._build_car_left_boundary_path(base, left_boundary, inset_value, w_seg)
            if left_boundary is not None:
                return self._build_car_left_boundary_path(base, left_boundary, inset_value, w_seg)
            if right_boundary is not None:
                return self._build_car_right_boundary_path(base, right_boundary, inset_value, w_seg)
            return base, False

        if not bool(state_updates_enabled):
            debug["cycle_id"] = int(self.car_avoidance_cycle_id)
            debug["state"] = self.car_avoidance_state
            debug["clear_frames"] = int(self.car_clearing_frames)
            debug["miss_frames"] = int(self.locked_car_miss_frames)
            debug["active"] = self.car_avoidance_state != "FOLLOW_LANE"
            debug["paused"] = True
            debug["event"] = "paused"
            self._log_car_avoidance_process(debug, "paused")
            return 0.0, debug, None

        if self.car_avoidance_state == "CLEARING" and self.car_last_avoid_path is not None:
            debug["cycle_id"] = int(self.car_avoidance_cycle_id)
            self.locked_car = None
            self.locked_car_miss_frames = 0
            self.car_clearing_frames += 1
            current_avoid_path = self.car_last_avoid_path
            current_path_ok = bool(self.car_last_avoid_path_is_boundary)
            candidate_path, candidate_ok = _build_avoid_path_for_side(
                self.car_avoidance_boundary_side,
                self.car_last_boundary_inset_x,
            )
            if candidate_ok:
                current_avoid_path = candidate_path
                current_path_ok = True
                self.car_last_avoid_path = candidate_path.copy()
                self.car_last_avoid_path_is_boundary = True
            clearing_path, avoid_weight = self._build_car_clearing_path(
                base,
                current_avoid_path,
                self.car_clearing_frames,
            )
            hold_frames = max(0, int(getattr(
                config,
                "CAR_AVOIDANCE_CLEARING_HOLD_FRAMES",
                getattr(config, "CAR_AVOIDANCE_CLEARING_MISS_FRAMES", 0),
            )))
            return_frames = max(1, int(getattr(
                config,
                "CAR_AVOIDANCE_CLEARING_RETURN_FRAMES",
                getattr(config, "CAR_AVOIDANCE_CLEARING_DECAY_FRAMES", 5),
            )))
            max_clear_frames = max(
                hold_frames + return_frames,
                int(getattr(config, "CAR_AVOIDANCE_CLEARING_MAX_FRAMES", hold_frames + return_frames)),
            )
            clear_timeout = self.car_clearing_frames >= max_clear_frames
            if avoid_weight <= 0.0 or clear_timeout:
                self.car_avoidance_state = "FOLLOW_LANE"
                self.car_clearing_frames = 0
                self.car_last_avoid_path = None
                self.car_last_avoid_path_is_boundary = False
                self.car_last_boundary_inset_x = 0.0
                self.car_avoidance_boundary_side = None
                debug["state"] = self.car_avoidance_state
                debug["clear_frames"] = 0
                debug["miss_frames"] = int(self.locked_car_miss_frames)
                debug["boundary_inset_x"] = 0.0
                debug["boundary_strength_x"] = 0.0
                debug["boundary_path_active"] = False
                debug["boundary_ready"] = False
                debug["avoid_weight"] = 0.0
                debug["boundary_side"] = ""
                debug["active"] = False
                debug["event"] = "clear_done"
                debug["clear_timeout"] = bool(clear_timeout)
                self._log_car_avoidance_process(debug, "clear_done", force=True)
                return 0.0, debug, None

            debug["state"] = self.car_avoidance_state
            debug["clear_frames"] = int(self.car_clearing_frames)
            debug["miss_frames"] = int(self.locked_car_miss_frames)
            debug["boundary_inset_x"] = float(self.car_last_boundary_inset_x)
            debug["boundary_strength_x"] = float(self.car_last_boundary_inset_x) * float(avoid_weight)
            debug["boundary_side"] = self.car_avoidance_boundary_side or ""
            debug["boundary_path_active"] = bool(current_path_ok)
            debug["boundary_ready"] = bool(current_path_ok)
            debug["avoid_weight"] = float(avoid_weight)
            debug["active"] = True
            debug["event"] = "clearing"
            debug["control_path_error"] = self._car_control_path_error_for_debug(clearing_path, w_seg)
            self._log_car_avoidance_process(debug, "clearing")
            return (
                float(debug["boundary_strength_x"]),
                debug,
                clearing_path if current_path_ok else None,
            )

        measurements = []
        for item in planning_items:
            measurement = self._car_measurement_from_item(item, w_seg, h_seg, base)
            if measurement is not None:
                measurements.append(measurement)
        debug["detected_cars"] = int(len(measurements))

        if self.locked_car is None and self.car_avoidance_state == "FOLLOW_LANE":
            start_rows = max(0.0, float(getattr(config, "CAR_AVOIDANCE_START_BOUNDARY_ROWS", 115.0)))
            path_bottom_y = float(np.max(base[:, 1]))
            near_measurements = []
            far_measurements = []
            for measurement in measurements:
                _cx, cy = measurement["bottom_center"]
                rows_to_car = max(0.0, path_bottom_y - float(cy))
                if rows_to_car <= start_rows:
                    near_measurements.append(measurement)
                else:
                    far_measurements.append((rows_to_car, measurement))
            if far_measurements and not near_measurements:
                debug["state"] = self.car_avoidance_state
                debug["event"] = "distance_limit"
                nearest_rows = min(rows for rows, _measurement in far_measurements)
                now = time.monotonic()
                interval = max(0.2, float(getattr(config, "LOG_INTERVAL_CAR_AVOIDANCE_PROCESS", 0.25)))
                if (
                    self.last_car_avoid_log_state != "car_avoidance_distance_limit" or
                    now - float(self.last_car_avoid_log_at) >= interval
                ):
                    self.last_car_avoid_log_state = "car_avoidance_distance_limit"
                    self.last_car_avoid_log_at = now
                    print(
                        f"避车距离未到: rows={nearest_rows:.1f}>{start_rows:.1f}，暂不触发 det={len(measurements)}",
                        flush=True,
                    )
                return 0.0, debug, None
            measurements = near_measurements

        locked_car = self._update_locked_car(measurements, base)
        if locked_car is None:
            if self.car_avoidance_state == "AVOIDING" and self.car_last_avoid_path is not None:
                self.car_avoidance_state = "CLEARING"
                self.car_clearing_frames = 0
                debug["state"] = self.car_avoidance_state
                debug["active"] = True
                debug["event"] = "lost_enter_clearing"
                self._log_car_avoidance_process(debug, "lost_enter_clearing", force=True)
            elif measurements:
                debug["state"] = self.car_avoidance_state
                debug["event"] = "tracking_wait_confirm"
                self._log_car_avoidance_process(debug, "tracking_wait_confirm")
            return 0.0, debug, None

        _smooth_cx, smooth_cy = locked_car.get("bottom_center", (0.0, float(np.max(base[:, 1]))))
        rows_to_car = max(0.0, float(np.max(base[:, 1])) - float(smooth_cy))
        left_boundary_x = self._boundary_x_at_y(left_boundary, float(smooth_cy)) if left_boundary is not None else None
        right_boundary_x = self._boundary_x_at_y(right_boundary, float(smooth_cy)) if right_boundary is not None else None
        target_x = float(w_seg) * 0.5
        left_boundary_error = None if left_boundary_x is None else float(left_boundary_x) - target_x
        right_boundary_error = None if right_boundary_x is None else float(right_boundary_x) - target_x
        debug["rows_to_bottom"] = float(rows_to_car)
        debug["miss_frames"] = int(self.locked_car_miss_frames)
        debug["left_boundary_error"] = None if left_boundary_error is None else float(left_boundary_error)
        debug["left_boundary_x"] = None if left_boundary_x is None else float(left_boundary_x)
        debug["right_boundary_error"] = None if right_boundary_error is None else float(right_boundary_error)
        debug["right_boundary_x"] = None if right_boundary_x is None else float(right_boundary_x)
        debug["locked_confirmed"] = bool(locked_car.get("confirmed", False))
        debug["locked_hit_frames"] = int(locked_car.get("hit_frames", 0))
        debug["locked_center"] = (float(_smooth_cx), float(smooth_cy))

        resolved_side, boundary_x, boundary_error, nearest_side = self._select_car_avoidance_boundary_side(
            locked_car,
            left_boundary,
            right_boundary,
        )
        if prev_state != "AVOIDING" or self.car_avoidance_boundary_side not in ("left", "right"):
            self.car_avoidance_boundary_side = resolved_side
        selected_side = self.car_avoidance_boundary_side if self.car_avoidance_boundary_side in ("left", "right") else resolved_side
        if selected_side != resolved_side:
            boundary_x = right_boundary_x if selected_side == "right" else left_boundary_x
            boundary_error = None if boundary_x is None else float(boundary_x) - target_x
        debug["boundary_side"] = selected_side
        debug["boundary_x"] = None if boundary_x is None else float(boundary_x)
        debug["boundary_error"] = None if boundary_error is None else float(boundary_error)
        debug["nearest_boundary_side"] = nearest_side

        inset, y_range, boundary_ready = self._car_avoidance_boundary_inset(locked_car, base)
        debug["boundary_ready"] = bool(boundary_ready)
        debug["boundary_inset_x"] = float(inset)

        if not boundary_ready:
            if self.car_avoidance_state == "AVOIDING":
                self.car_avoidance_state = "FOLLOW_LANE"
                self.car_clearing_frames = 0
                self.car_last_avoid_path = None
                self.car_last_avoid_path_is_boundary = False
                self.car_last_boundary_inset_x = 0.0
                self.car_avoidance_boundary_side = None
            debug["state"] = self.car_avoidance_state
            debug["clear_frames"] = int(self.car_clearing_frames)
            debug["event"] = "locked_not_ready"
            self._log_car_avoidance_process(debug, "locked_not_ready", force=prev_state == "AVOIDING")
            return 0.0, debug, None

        if y_range is not None:
            debug["blocked_y_range"] = (float(y_range[0]), float(y_range[1]))
        avoid_path, path_ok = _build_avoid_path_for_side(selected_side, inset)
        if prev_state != "AVOIDING":
            self.car_avoidance_cycle_id += 1
        self.car_avoidance_state = "AVOIDING"
        self.car_clearing_frames = 0
        self.car_last_avoid_path = avoid_path.copy() if path_ok else base.copy()
        self.car_last_avoid_path_is_boundary = bool(path_ok)
        self.car_last_boundary_inset_x = float(inset)
        debug["state"] = self.car_avoidance_state
        debug["clear_frames"] = 0
        debug["boundary_inset_x"] = float(inset)
        debug["boundary_strength_x"] = float(inset)
        debug["avoid_weight"] = 1.0
        debug["boundary_ready"] = bool(boundary_ready)
        debug["boundary_side"] = selected_side
        debug["boundary_path_active"] = bool(path_ok)
        debug["control_path_error"] = self._car_control_path_error_for_debug(avoid_path, w_seg) if path_ok else None
        debug["active"] = True
        debug["event"] = "enter_avoiding" if prev_state != "AVOIDING" else "avoiding"
        debug["cycle_id"] = int(self.car_avoidance_cycle_id)
        self._log_car_avoidance_process(debug, debug["event"], force=prev_state != "AVOIDING")
        debug["active"] = True
        return float(inset), debug, avoid_path if path_ok else None

    def infer_mask(self, blob_rgb_320):
        """只执行分割模型推理，返回二值 mask 和推理耗时."""
        t_infer_start = time.perf_counter()
        outputs = self.rknn.inference(inputs=[np.expand_dims(blob_rgb_320, axis=0)])
        out = outputs[0]
        t_infer_end = time.perf_counter()

        if len(out.shape) == 4 and out.shape[1] > 1:
            mask = (out[0][1] > out[0][0]).astype(np.uint8)
        else:
            mask = out.squeeze().astype(np.uint8)

        return (mask > 0).astype(np.uint8), t_infer_end - t_infer_start

    def postprocess_mask(
        self,
        blob_rgb_320,
        mask,
        current_yolo_boxes,
        turn_intent,
        fps_stats,
        sign_route_choice=0,
        infer_s=0.0,
        total_start=None,
        preview_frame=None,
        external_boundary_inset_x=0.0,
        external_boundary_side="left",
        sign_route_pending=False,
        debug_drive_active=True,
    ):
        """对已推理出的 mask 做路径规划、控制器调用和调试渲染.

        输入:
        - blob_rgb_320: 分割线程当前拿到的最新 SEG_SIZE RGB 图
        - mask: infer_mask 输出的二值赛道 mask
        - current_yolo_boxes: 当前最新一帧检测框，坐标在 TARGET_RES
        - turn_intent: OCR/LLM 给出的 LEFT / RIGHT 分叉意图；石头优先，无石头时参与分支选择
        - sign_route_choice: 当前路牌路线任务确认的 LEFT / RIGHT；0 表示没有有效路牌任务

        输出:
        - steer_signal: 单一转向控制量，来自 PathController 当前选中的 A/B/C 控制器
        - ai_view: SEG_SIZE 空间调试图，主线程会再放大回 TARGET_RES
        """
        t_total_start = time.perf_counter() if total_start is None else float(total_start)
        w_seg, h_seg = config.SEG_SIZE
        
        # ai_view 是最终调试图的底板，默认保持在分割空间，后续再由主线程放大。
        # 如果主线程传入完整预览图，则后面直接在完整图下半 ROI 上叠加 mask/路径，
        # 避免把 416x160 裁剪图整块放大导致明显色差。
        blob = blob_rgb_320
        ai_view = cv2.cvtColor(blob, cv2.COLOR_RGB2BGR)
        mask = (mask > 0).astype(np.uint8)

        # 投影 YOLO 框到分割面
        t_preprocess_start = time.perf_counter()
        planning_items = self._project_planning_objects(current_yolo_boxes, w_seg, h_seg)
        t_preprocess_end = time.perf_counter()

        # -------------------------------------------------------------------
        # 3. 在 mask 空间中做自底向上的路径搜索（重构：局部中线记录版）
        # -------------------------------------------------------------------
        t_search_start = time.perf_counter()
        steer_signal = 0.0
        pts_final_orig = None
        search_mask = self._prepare_search_mask(mask)
        search_edge_mask = self._extract_edge_mask(search_mask)
        merge_detect_info = None
        if not self.merge_state_active:
            merge_detect_info = self._detect_merge_guide(search_mask, search_edge_mask)
            if merge_detect_info is None:
                merge_detect_info = self._detect_edge_trace_merge_guide(search_mask, search_edge_mask)
        merge_guide_info = self._update_merge_state(merge_detect_info, search_mask)
        merge_side = None
        if merge_guide_info is not None:
            merge_side = merge_guide_info.get("side")
            if merge_detect_info is None or merge_guide_info is not merge_detect_info:
                current_guide_polyline = self._build_merge_guide_line(
                    [0, 1],
                    merge_side,
                    search_mask,
                    search_edge_mask,
                )
                if current_guide_polyline is not None:
                    merge_guide_info = {
                        "side": merge_side,
                        "guide_polyline": current_guide_polyline,
                    }
                else:
                    merge_guide_info = None
                    merge_side = None
        if merge_guide_info is not None:
            search_mask = self._apply_merge_guide(
                search_mask,
                merge_guide_info.get("guide_polyline"),
            )
            search_edge_mask = self._extract_edge_mask(search_mask)
            y_fork_info = {"active": False, "fork_point": None, "split_rows": 0}
        else:
            y_fork_info = self._detect_y_fork(search_mask)
        fork_bottom_mid = None
        branch_pair_count_max = 0
        branch_support_rows = 0
        t_search_end = time.perf_counter()
                
        # -------------------------------------------------------------------
        # 4. 对候选路径打分，引入局部中心偏差惩罚 + EMA 滤波选择最终路径
        # -------------------------------------------------------------------
        t_fit_start = time.perf_counter()
        boundary_left_orig = None
        boundary_right_orig = None
        candidate_left_orig = None
        candidate_right_orig = None
        y_fork_active = False
        fork_active = False
        pending_centerline_active = False
        best_path = None
        best_nodes = None
        fork_selected_side = None
        try:
            route_choice_raw = int(sign_route_choice)
        except (TypeError, ValueError):
            route_choice_raw = 0
        try:
            turn_intent_raw = int(turn_intent)
        except (TypeError, ValueError):
            turn_intent_raw = -1
        route_choice = route_choice_raw if route_choice_raw in (-1, 1) else 0
        preferred_turn_default = route_choice if route_choice in (-1, 1) else turn_intent_raw
        if preferred_turn_default not in (-1, 1):
            preferred_turn_default = -1
        route_boundary_side = None
        car_path_debug = None
        car_avoid_path = None
        car_active = False
        car_boundary_strength_x = 0.0
        external_boundary_inset_x = float(external_boundary_inset_x)
        external_boundary_active = abs(external_boundary_inset_x) > 0.0
        avoid_path_source = "external" if external_boundary_active else "none"
        centerline_only_mode = bool(getattr(config, "SEG_CENTERLINE_ONLY_MODE", False))

        if y_fork_info.get("active"):
            left_mask, right_mask = self._split_mask_by_fork(search_mask, y_fork_info["fork_point"])
            left_edge = self._extract_edge_mask(left_mask)
            right_edge = self._extract_edge_mask(right_mask)
            if centerline_only_mode:
                left_best = self._build_centerline_candidate(left_mask, left_edge)
                right_best = self._build_centerline_candidate(right_mask, right_edge)
                left_pair_max = int(left_best.get("branch_pair_count_max", 0)) if left_best is not None else 0
                right_pair_max = int(right_best.get("branch_pair_count_max", 0)) if right_best is not None else 0
                left_support_rows = int(left_best.get("branch_support_rows", 0)) if left_best is not None else 0
                right_support_rows = int(right_best.get("branch_support_rows", 0)) if right_best is not None else 0
            else:
                fork_side_max_paths = max(1, int(config.SEG_PATH_MAX_FORK_SIDE_ACTIVE_PATHS))
                left_paths, left_pair_max, left_support_rows = self._search_active_paths(
                    left_mask,
                    left_edge,
                    max_active_paths=fork_side_max_paths,
                )
                right_paths, right_pair_max, right_support_rows = self._search_active_paths(
                    right_mask,
                    right_edge,
                    max_active_paths=fork_side_max_paths,
                )

                left_candidates = self._score_candidate_paths(left_paths)
                right_candidates = self._score_candidate_paths(right_paths)
                left_best = max(left_candidates, key=lambda c: float(c["score"])) if left_candidates else None
                right_best = max(right_candidates, key=lambda c: float(c["score"])) if right_candidates else None

            branch_pair_count_max = max(2, left_pair_max, right_pair_max)
            branch_support_rows = max(
                int(y_fork_info.get("split_rows", 0)),
                left_support_rows,
                right_support_rows,
            )

            if left_best is not None and right_best is not None:
                y_fork_active = True
                fork_active = True
                candidate_left_orig = np.array(left_best["path"], dtype=np.float32).reshape((-1, 1, 2))
                candidate_right_orig = np.array(right_best["path"], dtype=np.float32).reshape((-1, 1, 2))

                fork_center_candidate = None
                if bool(sign_route_pending):
                    fork_bottom_mid = self._road_bottom_midpoint(search_mask)
                    fork_center_candidate = self._build_fork_centerline_candidate(
                        search_mask,
                        y_fork_info.get("fork_point"),
                        fork_bottom_mid,
                    )
                if fork_center_candidate is not None:
                    best_candidate = fork_center_candidate
                    fork_selected_side = None
                    pending_centerline_active = True
                else:
                    preferred_turn = self._resolve_preferred_turn(preferred_turn_default)
                    best_candidate = right_best if preferred_turn == 1 else left_best
                    fork_selected_side = "right" if preferred_turn == 1 else "left"
                best_path = best_candidate["path"]
                best_nodes = best_candidate["nodes"]

        if not y_fork_active:
            if centerline_only_mode:
                best_candidate = self._build_centerline_candidate(search_mask, search_edge_mask)
                if best_candidate is not None:
                    branch_pair_count_max = int(best_candidate.get("branch_pair_count_max", 0))
                    branch_support_rows = int(best_candidate.get("branch_support_rows", 0))
                    candidate_left_orig = np.array(
                        best_candidate["path"],
                        dtype=np.float32,
                    ).reshape((-1, 1, 2))
                    best_path = best_candidate["path"]
                    best_nodes = best_candidate["nodes"]
            else:
                active_paths, branch_pair_count_max, branch_support_rows = self._search_active_paths(
                    search_mask,
                    search_edge_mask,
                )
                valid_candidates = self._score_candidate_paths(active_paths)

                if valid_candidates:
                    if len(valid_candidates) == 1:
                        candidate_left_orig = np.array(
                            valid_candidates[0]["path"],
                            dtype=np.float32,
                        ).reshape((-1, 1, 2))
                    elif len(valid_candidates) >= 2:
                        display_left = min(valid_candidates, key=lambda c: c["avg_x"])
                        display_right = max(valid_candidates, key=lambda c: c["avg_x"])
                        if display_left is not display_right:
                            candidate_left_orig = np.array(display_left["path"], dtype=np.float32).reshape((-1, 1, 2))
                            candidate_right_orig = np.array(display_right["path"], dtype=np.float32).reshape((-1, 1, 2))

                    best_candidate = max(valid_candidates, key=lambda c: float(c["score"]))
                    if merge_side not in ("left", "right"):
                        fork_left, fork_right, fork_active = self._select_fork_representatives(
                            valid_candidates,
                            branch_support_rows,
                        )
                        choice_left = None
                        choice_right = None
                        if fork_active:
                            choice_left = fork_left
                            choice_right = fork_right
                        elif len(valid_candidates) >= 2:
                            leftmost = min(valid_candidates, key=lambda c: float(c["avg_x"]))
                            rightmost = max(valid_candidates, key=lambda c: float(c["avg_x"]))
                            if (
                                leftmost is not rightmost and
                                float(rightmost["avg_x"]) - float(leftmost["avg_x"]) >= float(config.PATH_LOCK_FORK_MIN_SEP)
                            ):
                                choice_left = leftmost
                                choice_right = rightmost

                        if choice_left is not None and choice_right is not None:
                            if preferred_turn_default == -1:
                                best_candidate = choice_left
                                route_boundary_side = "left"
                            elif preferred_turn_default == 1:
                                best_candidate = choice_right
                                route_boundary_side = "right"

                    # 没有确认 Y 型分叉时，只有已经分出左右代表候选才按目标侧补线；
                    # 普通单路保持原始左右边界，避免全程变成单边补线。
                    best_path = best_candidate["path"]
                    best_nodes = best_candidate["nodes"]

        if best_path is not None:
            guide_polyline = None if merge_guide_info is None else merge_guide_info.get("guide_polyline")
            fit_nodes = self._apply_merge_boundary_width(best_nodes, merge_side, guide_polyline)
            fork_width_side = fork_selected_side or route_boundary_side
            if (
                bool(getattr(config, "FORK_BOUNDARY_WIDTH_ENABLED", True)) and
                merge_side not in ("left", "right") and
                fork_width_side in ("left", "right")
            ):
                fit_nodes = self._apply_fork_boundary_width(
                    fit_nodes,
                    fork_width_side,
                )
            fit_path = np.array([node["pt"] for node in fit_nodes], dtype=np.float32)
            node_x = fit_path[:, 0]
            node_y = fit_path[:, 1]
            left_boundary_pts = np.array(
                [[node["left_x"], node["pt"][1]] for node in fit_nodes],
                dtype=np.float32,
            )
            right_boundary_pts = np.array(
                [[node["right_x"], node["pt"][1]] for node in fit_nodes],
                dtype=np.float32,
            )
            boundary_left_orig = left_boundary_pts.reshape((-1, 1, 2))
            boundary_right_orig = right_boundary_pts.reshape((-1, 1, 2))

            current_coeffs = self._fit_path_poly_coeffs(node_y, node_x)

            # 岔路首次确认时，上一帧通常还是公共中线。若继续把公共中线
            # 以 EMA 混入目标支路，会出现“先沿分界线、再切入支路”的现象。
            # 只在进入目标支路的切换帧跳过一次 EMA，普通路径仍保持平滑。
            current_fork_side = fork_selected_side or route_boundary_side
            previous_fork_side = (
                self.last_branch_stats.get("fork_selected_side") or
                self.last_branch_stats.get("route_boundary_side")
            )
            branch_route_switch = (
                current_fork_side in ("left", "right") and
                (
                    previous_fork_side != current_fork_side or
                    (
                        bool(y_fork_active) and
                        not bool(self.last_branch_stats.get("y_fork_active", False))
                    )
                )
            )
            if (
                not branch_route_switch and
                self.last_poly_coeffs is not None and
                len(self.last_poly_coeffs) == len(current_coeffs)
            ):
                poly_coeffs = self.ema_alpha * self.last_poly_coeffs + (1.0 - self.ema_alpha) * current_coeffs
            else:
                poly_coeffs = current_coeffs

            dense_y = np.linspace(node_y[0], node_y[-1], num=int(config.SEG_PATH_DENSE_SAMPLES))
            dense_x = np.polyval(poly_coeffs, dense_y)
            dense_x = np.clip(dense_x, 0, w_seg - 1)
            dense_y = np.clip(dense_y, 0, h_seg - 1)

            path_points_orig = np.vstack((dense_x, dense_y)).astype(np.float32).T
            lateral_path_points = None
            raw_lateral_x = self._interp_path_xs(fit_path, dense_y)
            if raw_lateral_x is not None:
                raw_lateral_x = np.clip(raw_lateral_x, 0, w_seg - 1)
                lateral_path_points = np.vstack((raw_lateral_x, dense_y)).astype(np.float32).T
            lateral_fusion_side = fork_selected_side or route_boundary_side
            lateral_fusion_points = self._build_lateral_fusion_points(
                lateral_path_points,
                left_boundary_pts,
                right_boundary_pts,
                w_seg,
                trusted_side=lateral_fusion_side,
            )
            base_path_points = path_points_orig.copy()
            car_boundary_strength_x, car_path_debug, car_avoid_path = self._update_car_avoidance_boundary_path(
                path_points_orig,
                planning_items,
                w_seg,
                h_seg,
                left_boundary=left_boundary_pts,
                right_boundary=right_boundary_pts,
                state_updates_enabled=bool(debug_drive_active),
            )
            car_active = car_path_debug is not None and car_path_debug.get("active")
            external_boundary_inset_x = float(external_boundary_inset_x)
            external_boundary_active = abs(external_boundary_inset_x) > 0.0
            external_boundary_side = str(external_boundary_side).lower()
            if external_boundary_side not in ("left", "right"):
                external_boundary_side = "left"
            avoid_path_source = "car" if car_active else "none"
            if external_boundary_active:
                external_boundary_pts = right_boundary_pts if external_boundary_side == "right" else left_boundary_pts
                external_avoid_path, external_path_ok = self._build_boundary_inset_path(
                    path_points_orig,
                    external_boundary_pts,
                    external_boundary_inset_x,
                    w_seg,
                )
                if external_path_ok and (
                    (not car_active) or
                    abs(float(external_boundary_inset_x)) > abs(float(car_boundary_strength_x))
                ):
                    car_avoid_path = external_avoid_path
                    avoid_path_source = "external_car_dir" if car_active else "external"
                    car_boundary_strength_x = float(external_boundary_inset_x)
            car_state = "FOLLOW_LANE"
            if car_path_debug is not None:
                car_state = str(car_path_debug.get("state", "FOLLOW_LANE"))
            control_profile = self.path_controller.select_stanley_control_profile(
                car_active=car_active,
                car_state=car_state,
                car_cycle_id=self.car_avoidance_cycle_id,
            )
            post_car_control_active = control_profile == "POST_CAR"
            car_boundary_path_active = (
                car_avoid_path is not None and
                (
                    avoid_path_source in ("external", "external_car_dir") or
                    (
                        car_active and
                        avoid_path_source == "car" and
                        bool(car_path_debug.get("boundary_path_active", False))
                    )
                )
            )
            bypass_frame_jump = external_boundary_active or car_boundary_path_active
            if bypass_frame_jump:
                path_jump_limited = False
            else:
                path_points_orig, path_jump_limited = self._limit_path_frame_jump(path_points_orig)
            if path_jump_limited or car_active or external_boundary_active:
                poly_coeffs = self._fit_path_poly_coeffs(
                    path_points_orig[:, 1],
                    path_points_orig[:, 0],
                )

            self.last_poly_coeffs = poly_coeffs
            self.last_path_points_orig = path_points_orig.copy()
            self.missing_path_frames = 0
            control_path_points = path_points_orig
            fusion_allowed = (
                lateral_fusion_points is not None and
                not bool(car_active) and
                not bool(external_boundary_active)
            )
            lateral_control_points = lateral_fusion_points if fusion_allowed else lateral_path_points
            control_mode = str(getattr(config, "STEER_CONTROL_MODE", "weighted_slope")).lower()
            if (
                control_mode in ("stanley_band", "control_c") and
                bool(getattr(config, "PATH_LATERAL_USE_FILTERED_PATH", True)) and
                not fusion_allowed
            ):
                lateral_control_points = control_path_points
            if car_boundary_path_active:
                control_path_points = car_avoid_path
                lateral_control_points = car_avoid_path
                if control_mode == "weighted_slope":
                    weighted_points = self.path_controller.select_control_points(
                        control_mode,
                        car_avoid_path,
                        h_seg,
                    )
                    if weighted_points is not None:
                        control_path_points = weighted_points
                        lateral_control_points = weighted_points
            else:
                if control_mode == "weighted_slope":
                    weighted_points = self.path_controller.select_control_points(
                        control_mode,
                        path_points_orig,
                        h_seg,
                    )
                    if weighted_points is not None:
                        control_path_points = weighted_points
                        lateral_control_points = weighted_points
            control_path_source = "car_boundary" if car_boundary_path_active else "normal"
            if (
                bool(getattr(config, "CAR_AVOIDANCE_RESET_CONTROLLER_ON_PATH_SWITCH", False)) and
                control_path_source != self.last_control_path_source
            ):
                self.path_controller.reset()
            self.last_control_path_source = control_path_source
            control_param_overrides = (
                self.path_controller.stanley_param_overrides_for_profile(control_profile)
                if control_mode == "stanley_band" else
                None
            )
            steer_signal = self.path_controller.compute_steer_signal(
                control_path_points,
                w_seg,
                h_seg,
                center_bias_x=float(getattr(config, "CONTROL_CENTER_BIAS_X", 0.0)),
                lateral_points=lateral_control_points,
                param_overrides=control_param_overrides,
            )
            if car_active:
                steer_signal *= float(getattr(config, "STEER_SIGNAL_CAR_GAIN", 1.0))
            elif not car_active:
                steer_signal *= float(getattr(config, "STEER_SIGNAL_NO_TARGET_GAIN", 1.0))
            control_band = None
            lateral_debug_points = None
            control_mode = str(getattr(config, "STEER_CONTROL_MODE", "weighted_slope")).lower()
            control_band = self.path_controller.control_band_for_mode(
                control_mode,
                control_path_points,
                h_seg,
            )
            pts_final_orig = path_points_orig.reshape((-1, 1, 2))
        else:
            held_path = self._hold_last_path()
            if held_path is not None:
                path_points_orig = held_path
                control_path_points = path_points_orig
                lateral_control_points = path_points_orig
                control_mode = str(getattr(config, "STEER_CONTROL_MODE", "weighted_slope")).lower()
                if control_mode == "weighted_slope":
                    weighted_points = self.path_controller.select_control_points(
                        control_mode,
                        path_points_orig,
                        h_seg,
                    )
                    if weighted_points is not None:
                        control_path_points = weighted_points
                        lateral_control_points = weighted_points
                if (
                    bool(getattr(config, "CAR_AVOIDANCE_RESET_CONTROLLER_ON_PATH_SWITCH", False)) and
                    self.last_control_path_source != "normal"
                ):
                    self.path_controller.reset()
                self.last_control_path_source = "normal"
                control_profile = self.path_controller.select_stanley_control_profile(
                    car_active=False,
                    car_state=self.car_avoidance_state,
                    car_cycle_id=self.car_avoidance_cycle_id,
                )
                post_car_control_active = control_profile == "POST_CAR"
                control_param_overrides = (
                    self.path_controller.stanley_param_overrides_for_profile(control_profile)
                    if control_mode == "stanley_band" else
                    None
                )
                steer_signal = self.path_controller.compute_steer_signal(
                    control_path_points,
                    w_seg,
                    h_seg,
                    center_bias_x=float(getattr(config, "CONTROL_CENTER_BIAS_X", 0.0)),
                    lateral_points=lateral_control_points,
                    param_overrides=control_param_overrides,
                )
                control_band = self.path_controller.control_band_for_mode(
                    control_mode,
                    control_path_points,
                    h_seg,
                )
                pts_final_orig = path_points_orig.reshape((-1, 1, 2))
            else:
                self.last_poly_coeffs = None
                self.path_controller.reset()
                self.last_control_path_source = "normal"
        self.last_branch_stats = {
            "branch_pair_count_max": int(branch_pair_count_max),
            "branch_support_rows": int(branch_support_rows),
            "fork_active": bool(fork_active),
            "y_fork_active": bool(y_fork_active),
            "y_fork_point": y_fork_info.get("fork_point") if y_fork_active else None,
            "y_fork_split_rows": int(y_fork_info.get("split_rows", 0)) if y_fork_active else 0,
            "fork_selected_side": fork_selected_side,
            "route_boundary_side": route_boundary_side,
            "merge_side": merge_side,
            "merge_state_active": bool(self.merge_state_active),
            "merge_state_hit_frames": int(self.merge_state_hit_frames),
            "merge_state_exit_frames": int(self.merge_state_exit_frames),
            "car_active": bool(car_active),
            "car_state": str(car_state if 'car_state' in locals() else "FOLLOW_LANE"),
            "car_rows_to_bottom": float(car_path_debug.get("rows_to_bottom", 0.0)) if car_path_debug is not None and car_path_debug.get("rows_to_bottom") is not None else None,
            "car_miss_frames": int(car_path_debug.get("miss_frames", 0)) if car_path_debug is not None else 0,
            "car_clear_frames": int(car_path_debug.get("clear_frames", 0)) if car_path_debug is not None else 0,
            "car_state_paused": bool(car_path_debug.get("paused", False)) if car_path_debug is not None else False,
            "car_boundary_side": str(car_path_debug.get("boundary_side", "")) if car_path_debug is not None else "",
            "car_boundary_error": float(car_path_debug.get("boundary_error", 0.0)) if car_path_debug is not None and car_path_debug.get("boundary_error") is not None else None,
            "car_boundary_x": float(car_path_debug.get("boundary_x", 0.0)) if car_path_debug is not None and car_path_debug.get("boundary_x") is not None else None,
            "car_nearest_boundary_side": str(car_path_debug.get("nearest_boundary_side", "")) if car_path_debug is not None else "",
            "car_left_boundary_error": float(car_path_debug.get("left_boundary_error", 0.0)) if car_path_debug is not None and car_path_debug.get("left_boundary_error") is not None else None,
            "car_left_boundary_x": float(car_path_debug.get("left_boundary_x", 0.0)) if car_path_debug is not None and car_path_debug.get("left_boundary_x") is not None else None,
            "car_control_path_error": float(car_path_debug.get("control_path_error", 0.0)) if car_path_debug is not None and car_path_debug.get("control_path_error") is not None else None,
            "car_boundary_inset_x": float(car_path_debug.get("boundary_inset_x", 0.0)) if car_path_debug is not None else 0.0,
            "car_detected_cars": int(car_path_debug.get("detected_cars", 0)) if car_path_debug is not None else 0,
            "car_locked_confirmed": bool(car_path_debug.get("locked_confirmed", False)) if car_path_debug is not None else False,
            "car_locked_hit_frames": int(car_path_debug.get("locked_hit_frames", 0)) if car_path_debug is not None else 0,
            "car_boundary_path_active": bool(car_path_debug.get("boundary_path_active", False)) if car_path_debug is not None else False,
            "car_avoid_weight": float(car_path_debug.get("avoid_weight", 0.0)) if car_path_debug is not None else 0.0,
            "car_event": str(car_path_debug.get("event", "")) if car_path_debug is not None else "",
            "car_cycle_id": int(car_path_debug.get("cycle_id", self.car_avoidance_cycle_id)) if car_path_debug is not None else int(self.car_avoidance_cycle_id),
            "post_car_control_active": bool(post_car_control_active) if 'post_car_control_active' in locals() else False,
            "car_control_profile": str(control_profile) if 'control_profile' in locals() else "NORMAL",
            "car_control_kp": float(control_param_overrides.get("lateral_gain", 0.0)) if 'control_param_overrides' in locals() and control_param_overrides is not None else None,
            "car_control_kd": float(control_param_overrides.get("d_gain", 0.0)) if 'control_param_overrides' in locals() and control_param_overrides is not None else None,
            "car_control_psi": float(control_param_overrides.get("heading_gain", 0.0)) if 'control_param_overrides' in locals() and control_param_overrides is not None else None,
            "car_control_speed_estimate": float(control_param_overrides.get("speed_estimate", 0.0)) if 'control_param_overrides' in locals() and control_param_overrides is not None else None,
            "external_boundary_inset_x": float(external_boundary_inset_x),
            "external_boundary_side": external_boundary_side,
            "sign_route_pending_centerline": bool(pending_centerline_active),
        }
        self._store_main_overlay(
            pts_final_orig,
            boundary_left_orig,
            boundary_right_orig,
            w_seg,
            h_seg,
            candidate_left_pts=candidate_left_orig,
            candidate_right_pts=candidate_right_orig,
            merge_guide_pts=None if merge_guide_info is None else merge_guide_info.get("guide_polyline"),
            fork_point=y_fork_info.get("fork_point") if pending_centerline_active else None,
            control_band=control_band if 'control_band' in locals() else None,
            bottom_mid=fork_bottom_mid if pending_centerline_active else None,
        )
        t_fit_end = time.perf_counter()

        # -------------------------------------------------------------------
        # 5. 调试渲染
        # -------------------------------------------------------------------
        t_render_start = time.perf_counter()
        if preview_frame is not None:
            target_w, target_h = config.TARGET_RES
            if preview_frame.shape[1] != target_w or preview_frame.shape[0] != target_h:
                ai_view = cv2.resize(preview_frame, config.TARGET_RES, interpolation=cv2.INTER_LINEAR)
            else:
                ai_view = preview_frame.copy()

            crop_ratio = float(getattr(config, "SEG_INPUT_CROP_TOP_RATIO", 0.0))
            crop_ratio = max(0.0, min(0.95, crop_ratio))
            crop_y = int(round(target_h * crop_ratio))
            crop_y = max(0, min(target_h - 1, crop_y))
            bottom_h = target_h - crop_y
            bottom_roi = ai_view[crop_y:, :]

            if bool(getattr(config, "SEG_DEBUG_DRAW_MASK", True)):
                mask_bottom = cv2.resize(mask, (target_w, bottom_h), interpolation=cv2.INTER_NEAREST)
                mask_pixels = mask_bottom == 1
                if np.any(mask_pixels):
                    overlay_color = np.array([0, 255, 0], dtype=np.float32)
                    alpha = float(config.MASK_ALPHA)
                    bottom_roi[mask_pixels] = (
                        bottom_roi[mask_pixels].astype(np.float32) * (1.0 - alpha) +
                        overlay_color * alpha
                    ).astype(np.uint8)

            # 路径和边界仍按分割空间坐标存储，直接画在下半 ROI 上即可保持比例。
            self.draw_path_overlay(bottom_roi)
        else:
            if bool(getattr(config, "SEG_DEBUG_DRAW_MASK", True)):
                colored_roi = np.zeros_like(ai_view)
                colored_roi[mask == 1] = [0, 255, 0]
                ai_view = cv2.addWeighted(ai_view, 1.0 - config.MASK_ALPHA, colored_roi, config.MASK_ALPHA, 0)
            # 路径直接画在分割调试平面里，主线程只负责整体缩放和其它信息叠加。
            ai_view = self.draw_path_overlay(ai_view)

        pwm_gain = float(config.STEER_SIGNAL_PWM_GAIN)
        control_mode = str(getattr(config, "STEER_CONTROL_MODE", "weighted_slope")).lower()
        if control_mode == "stanley_band":
            pwm_gain = float(getattr(config, "STANLEY_PWM_GAIN", 0.012))
        elif control_mode == "control_c":
            pwm_gain = float(getattr(config, "CONTROL_C_PWM_GAIN", 12.0))
        servo_pwm = int(
            config.SERVO_CENTER
            - steer_signal * pwm_gain
        )
        servo_pwm = int(max(config.SERVO_MIN, min(config.SERVO_MAX, servo_pwm)))
        self._log_control_c_debug(
            steer_signal,
            servo_pwm,
            car_active=bool(car_active) if 'car_active' in locals() else False,
        )
        draw_seg_status_text(
            ai_view,
            fps_stats=fps_stats,
            steer_signal=steer_signal,
            servo_pwm=servo_pwm,
            branch_stats=self.last_branch_stats,
        )
        t_render_end = time.perf_counter()
        preprocess_s = t_preprocess_end - t_preprocess_start
        search_s = t_search_end - t_search_start
        fit_s = t_fit_end - t_fit_start
        render_s = t_render_end - t_render_start
        # 本地处理总时长（不含队列等待/排队延迟）
        local_total = float(infer_s) + float(preprocess_s) + float(search_s) + float(fit_s) + float(render_s)
        # 队列等待时间/排队延迟: 从外部传入的 total_start 到本地 preprocess 开始的间隔
        # total_start 通常是在推理开始前记录，所以剔除 infer_s 后得到真正的排队等待时间
        queue_wait = float(t_preprocess_start - (float(t_total_start) + float(infer_s)))
        if queue_wait < 0.0:
            queue_wait = 0.0
        self._profile_add(
            infer_s=float(infer_s),
            preprocess_s=preprocess_s,
            search_s=search_s,
            fit_s=fit_s,
            render_s=render_s,
            total_s=local_total,
            queue_wait_s=queue_wait,
        )

        return steer_signal, ai_view

    def run(
        self,
        blob_rgb_320,
        current_yolo_boxes,
        turn_intent,
        fps_stats,
        sign_route_choice=0,
        external_boundary_inset_x=0.0,
        external_boundary_side="left",
        sign_route_pending=False,
        debug_drive_active=True,
    ):
        """兼容旧串行调用：推理和后处理在同一个线程里连续执行."""
        t_total_start = time.perf_counter()
        mask, infer_s = self.infer_mask(blob_rgb_320)
        return self.postprocess_mask(
            blob_rgb_320,
            mask,
            current_yolo_boxes,
            turn_intent,
            fps_stats,
            sign_route_choice=sign_route_choice,
            infer_s=infer_s,
            total_start=t_total_start,
            external_boundary_inset_x=external_boundary_inset_x,
            external_boundary_side=external_boundary_side,
            sign_route_pending=sign_route_pending,
            debug_drive_active=debug_drive_active,
        )

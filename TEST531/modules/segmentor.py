# modules/segmentor.py
"""赛道分割与路径规划模块.

这个模块负责:
1. 调用分割模型得到二值赛道 mask
2. 在 mask 空间中搜索一条可跟踪路径（引入分支局部中心约束，抑制切内线）
3. 基于路径点到图像底部中点的加权斜率和，生成单一转向控制量
4. 引入多项式时域低通滤波 (EMA)，提升路径稳定性
5. 返回用于控制的 steer_signal，以及一张调试渲染图

当前路径选择策略的核心优先级是:
1. 先在 mask 中寻找可行路径；若存在明显岔路，再先做分叉分区
2. 路径必须从图像底部触达区域起步，悬空候选会被直接丢弃
3. 如果检测到 `stone`，优先绕开石头所在分支
4. 否则默认偏向左支
5. OCR 语义转向当前仅预留输入，暂不参与最终分支选择
"""

import cv2
import numpy as np
import time
from rknnlite.api import RKNNLite
import config
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
        self.missing_path_frames = 0
        self.ema_alpha = float(config.SEG_EMA_ALPHA)
        self.last_branch_stats = {
            "branch_pair_count_max": 0,
            "branch_support_rows": 0,
            "fork_active": False,
            "y_fork_active": False,
            "merge_side": None,
        }
        self.merge_state_active = False
        self.merge_state_hit_frames = 0
        self.merge_state_exit_frames = 0
        self.merge_state_info = None
        self.merge_state_side = None
        self.locked_car = None
        self.locked_car_miss_frames = 0
        self.car_avoidance_state = "FOLLOW_LANE"
        self.car_clearing_frames = 0
        self.car_last_avoid_path = None
        self.car_last_blocked_y_range = None
        self.car_last_center_bias_sign = 1.0
        self.locked_coin = None
        self.locked_coin_miss_frames = 0
        self.last_main_overlay = {
            "path": None,
            "left": None,
            "right": None,
            "candidate_left": None,
            "candidate_right": None,
            "merge_guide": None,
            "fork_point": None,
            "coin_path": None,
            "bottom_mid": (0.0, 0.0),
            "base_size": tuple(config.SEG_SIZE),
        }
        self.profile_ema = None
        self.profile_last_log = 0.0

    def selected_left_boundary_x_at_target_y(self, target_y):
        """返回当前选中路径左边界在 TARGET_RES 坐标系下的 x."""
        left_boundary = self.last_main_overlay.get("left")
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
        right_boundary = self.last_main_overlay.get("right")
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
        coin_path=None,
    ):
        """缓存主图路径叠加层，供主线程在其它元素之上重绘."""
        self.last_main_overlay = {
            "path": None if path_pts is None else np.array(path_pts, dtype=np.float32).copy(),
            "left": None if left_pts is None else np.array(left_pts, dtype=np.float32).copy(),
            "right": None if right_pts is None else np.array(right_pts, dtype=np.float32).copy(),
            "candidate_left": None if candidate_left_pts is None else np.array(candidate_left_pts, dtype=np.float32).copy(),
            "candidate_right": None if candidate_right_pts is None else np.array(candidate_right_pts, dtype=np.float32).copy(),
            "merge_guide": None if merge_guide_pts is None else np.array(merge_guide_pts, dtype=np.float32).copy(),
            "fork_point": None if fork_point is None else (float(fork_point[0]), float(fork_point[1])),
            "coin_path": coin_path,
            "bottom_mid": (float(img_w) / 2.0, float(img_h) - 1.0),
            "base_size": (int(img_w), int(img_h)),
        }

    def draw_path_overlay(self, image):
        """把最近一次搜索得到的主图路径/边界叠加到任意尺寸的画面最上层."""
        if image is None:
            return image

        overlay = self.last_main_overlay
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
        coin_dot_radius = max(1, int(round(config.SEG_DEBUG_COIN_PATH_DOT_RADIUS * scale)))
        coin_strict_line_thickness = max(1, int(round(getattr(config, "SEG_DEBUG_COIN_BOTTOM_STRICT_LINE_THICKNESS", 1) * scale)))

        def _scale_point(pt):
            return (
                int(round(float(pt[0]) * scale_x)),
                int(round(float(pt[1]) * scale_y)),
            )

        if bool(getattr(config, "SEG_DEBUG_DRAW_CANDIDATE_PATHS", False)) and candidate_left_poly is not None:
            cv2.polylines(
                image,
                [candidate_left_poly],
                False,
                config.SEG_DEBUG_LEFT_PATH_COLOR,
                candidate_path_thickness,
            )
        if bool(getattr(config, "SEG_DEBUG_DRAW_CANDIDATE_PATHS", False)) and candidate_right_poly is not None:
            cv2.polylines(
                image,
                [candidate_right_poly],
                False,
                config.SEG_DEBUG_RIGHT_PATH_COLOR,
                candidate_path_thickness,
            )
        if path_poly is not None:
            cv2.polylines(
                image,
                [path_poly],
                False,
                config.SEG_DEBUG_PATH_COLOR,
                path_thickness,
            )
        if bool(getattr(config, "SEG_DEBUG_DRAW_BOUNDARIES", True)) and left_poly is not None:
            cv2.polylines(
                image,
                [left_poly],
                False,
                config.SEG_DEBUG_LEFT_BOUNDARY_COLOR,
                boundary_thickness,
            )
        if bool(getattr(config, "SEG_DEBUG_DRAW_BOUNDARIES", True)) and right_poly is not None:
            cv2.polylines(
                image,
                [right_poly],
                False,
                config.SEG_DEBUG_RIGHT_BOUNDARY_COLOR,
                boundary_thickness,
            )
        if bool(getattr(config, "SEG_DEBUG_DRAW_MERGE_GUIDE", True)) and merge_guide_poly is not None:
            cv2.polylines(
                image,
                [merge_guide_poly],
                False,
                config.SEG_DEBUG_MERGE_GUIDE_COLOR,
                merge_guide_thickness,
                cv2.LINE_AA,
            )

        if bool(getattr(config, "SEG_DEBUG_COIN_BOTTOM_STRICT_LINE_ENABLED", True)):
            strict_rows = float(getattr(config, "COIN_PATH_ROI_BOTTOM_STRICT_ROWS", 0.0))
            if strict_rows > 0.0:
                strict_y = int(round((float(base_h) - strict_rows) * scale_y))
                strict_y = max(0, min(img_h - 1, strict_y))
                cv2.line(
                    image,
                    (0, strict_y),
                    (img_w - 1, strict_y),
                    getattr(config, "SEG_DEBUG_COIN_BOTTOM_STRICT_LINE_COLOR", (0, 0, 255)),
                    coin_strict_line_thickness,
                    cv2.LINE_AA,
                )

        coin_path_debug = overlay.get("coin_path")
        if bool(getattr(config, "SEG_DEBUG_COIN_PATH_ENABLED", True)) and coin_path_debug:
            coin_planned_poly = _scaled_polyline(coin_path_debug.get("planned_path"))
            if coin_planned_poly is not None:
                cv2.polylines(
                    image,
                    [coin_planned_poly],
                    False,
                    config.SEG_DEBUG_COIN_PATH_COLOR,
                    max(1, path_thickness),
                    cv2.LINE_AA,
                )
            for pt in coin_path_debug.get("coin_points", []):
                cv2.circle(
                    image,
                    _scale_point(pt),
                    coin_dot_radius,
                    config.SEG_DEBUG_COIN_PATH_COLOR,
                    -1,
                    cv2.LINE_AA,
                )

        bottom_mid = overlay.get("bottom_mid", (float(base_w) / 2.0, float(base_h) - 1.0))
        bottom_mid_pt = _scale_point(bottom_mid)
        fork_point = overlay.get("fork_point")
        if fork_point is not None:
            fork_pt = _scale_point(fork_point)
            cv2.line(
                image,
                fork_pt,
                bottom_mid_pt,
                config.SEG_DEBUG_FORK_DIVIDER_COLOR,
                max(1, int(round(config.SEG_DEBUG_FORK_DIVIDER_THICKNESS * scale))),
                cv2.LINE_AA,
            )
        cv2.circle(
            image,
            bottom_mid_pt,
            bottom_mid_radius,
            config.SEG_DEBUG_BOTTOM_MID_COLOR,
            -1,
        )
        return image

    def _profile_add(self, infer_s, preprocess_s, search_s, fit_s, render_s, total_s, queue_wait_s=None):
        """按阶段统计分割链路耗时，节流打印用于定位掉帧瓶颈。"""
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
        # 把 measued total 与分阶段之和的差作为 overhead，便于保持打印一致性
        stage_sum = (
            current["infer"] + current["prep"] + current["search"] + current["fit"] + current["render"]
        )
        current_overhead = float(current["total"] - stage_sum)
        current["overhead"] = current_overhead

        if self.profile_ema is None:
            self.profile_ema = current
        else:
            alpha = 0.85
            # 将 overhead 纳入 EMA，使得各项平滑后仍能恢复出一致的 total
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
        # 使用各阶段平滑后的和作为显示的 total（包含 overhead）以保证一致性
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

        return {
            "active": True,
            "fork_point": (float(lowest_split["fork_x"]), float(lowest_split["y"])),
            "split_rows": max(1, int(len(opening_run))),
        }

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
        """汇合补线状态机：三帧确认进入，底部宽度连续恢复两帧退出."""
        confirm_frames = max(1, int(getattr(config, "MERGE_STATE_CONFIRM_FRAMES", 3)))
        exit_confirm_frames = max(1, int(getattr(config, "MERGE_STATE_EXIT_CONFIRM_FRAMES", 2)))

        detected_side = None if merge_detect_info is None else merge_detect_info.get("side")
        if self.merge_state_active and self.merge_state_side in ("left", "right"):
            if detected_side == self.merge_state_side:
                self.merge_state_hit_frames += 1
                self.merge_state_info = merge_detect_info
            elif detected_side is None:
                self.merge_state_hit_frames = 0
            else:
                # 当前边还在，不允许直接跳到另一边；先按退出流程走。
                detected_side = None
                merge_detect_info = None
                self.merge_state_hit_frames = 0
        else:
            if merge_detect_info is not None:
                self.merge_state_hit_frames += 1
                self.merge_state_info = merge_detect_info
            else:
                self.merge_state_hit_frames = 0

        if not self.merge_state_active:
            self.merge_state_exit_frames = 0
            if self.merge_state_hit_frames >= confirm_frames and self.merge_state_info is not None:
                self.merge_state_active = True
                self.merge_state_side = self.merge_state_info.get("side")
            else:
                return None

        if self._merge_bottom_width_exit_ready(search_mask):
            self.merge_state_exit_frames += 1
        else:
            self.merge_state_exit_frames = 0

        if self.merge_state_exit_frames >= exit_confirm_frames:
            self.merge_state_active = False
            self.merge_state_hit_frames = 0
            self.merge_state_exit_frames = 0
            self.merge_state_info = None
            self.merge_state_side = None
            return None

        if self.merge_state_info is not None:
            self.merge_state_side = self.merge_state_info.get("side")
        return self.merge_state_info

    def _split_mask_by_fork(self, search_mask, fork_point):
        """按“分叉点到底部中点”的分界线，把 mask 切成左右两大区域."""
        h, w = search_mask.shape[:2]
        bottom_mid_x = float(w) / 2.0
        bottom_y = float(h) - 1.0
        fork_x = float(fork_point[0])
        fork_y = float(fork_point[1])

        left_mask = np.zeros_like(search_mask, dtype=np.uint8)
        right_mask = np.zeros_like(search_mask, dtype=np.uint8)

        divider_dy = max(1.0, bottom_y - fork_y)
        for y in range(h):
            if float(y) <= fork_y:
                split_x = fork_x
            else:
                t = np.clip((float(y) - fork_y) / divider_dy, 0.0, 1.0)
                split_x = fork_x + (bottom_mid_x - fork_x) * t

            split_col = int(np.clip(round(split_x), 0, w - 1))
            left_mask[y, :split_col + 1] = search_mask[y, :split_col + 1]
            right_mask[y, split_col:] = search_mask[y, split_col:]

        return left_mask, right_mask

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

    def _estimate_stone_branch_side(self, planning_items, candidate_paths):
        """估计石头更接近哪一侧候选分支.

        返回:
        -1: 更接近左支
         1: 更接近右支
         0: 无法稳定判断
        """
        stone_items = self._get_stone_avoid_items(planning_items)
        if not stone_items or len(candidate_paths) < 2:
            return 0

        left_candidate = min(candidate_paths, key=lambda c: c["avg_x"])
        right_candidate = max(candidate_paths, key=lambda c: c["avg_x"])

        if left_candidate is right_candidate:
            return 0

        vote = 0
        for item in stone_items:
            seg_box = np.array(item["seg_box"], dtype=np.float32)
            stone_center = np.mean(seg_box, axis=0)
            stone_x = float(stone_center[0])
            stone_y = float(stone_center[1])

            left_x = self._path_x_at_y(left_candidate, stone_y)
            right_x = self._path_x_at_y(right_candidate, stone_y)

            if abs(right_x - left_x) < config.STONE_BRANCH_MIN_SEP:
                continue

            if abs(stone_x - left_x) <= abs(stone_x - right_x):
                vote -= 1
            else:
                vote += 1

        if vote < 0:
            return -1
        if vote > 0:
            return 1
        return 0

    def _get_stone_avoid_items(self, planning_items):
        """筛出参与分支避让的石头检测结果."""
        return [
            item for item in planning_items
            if item.get("class_name") == "stone" and item.get("seg_box") is not None
        ]

    def _estimate_stone_side_by_fork_divider(self, planning_items, fork_point, mask_shape):
        """按 Y 分叉切分线判断石头位于左区还是右区.

        这个判断和 _split_mask_by_fork 使用同一条“fork_point -> 底部中点”分界线，
        比较适合已经确认 Y 型分叉的场景。
        """
        stone_items = self._get_stone_avoid_items(planning_items)
        if not stone_items or fork_point is None:
            return 0

        h, w = mask_shape[:2]
        bottom_mid_x = float(w) / 2.0
        bottom_y = float(h) - 1.0
        fork_x = float(fork_point[0])
        fork_y = float(fork_point[1])
        divider_dy = max(1.0, bottom_y - fork_y)
        min_sep = float(config.STONE_BRANCH_MIN_SEP)

        vote = 0
        for item in stone_items:
            seg_box = np.array(item["seg_box"], dtype=np.float32)
            stone_center = np.mean(seg_box, axis=0)
            stone_x = float(stone_center[0])
            stone_y = float(stone_center[1])

            if stone_y <= fork_y:
                split_x = fork_x
            else:
                t = np.clip((stone_y - fork_y) / divider_dy, 0.0, 1.0)
                split_x = fork_x + (bottom_mid_x - fork_x) * t

            if abs(stone_x - split_x) < min_sep:
                continue
            if stone_x < split_x:
                vote -= 1
            else:
                vote += 1

        if vote < 0:
            return -1
        if vote > 0:
            return 1
        return 0

    def _resolve_preferred_turn(self, stone_branch_side):
        """得到最终偏左/偏右选择；当前只启用石头避让，暂不使用 OCR 语义."""
        if stone_branch_side == -1:
            return 1
        if stone_branch_side == 1:
            return -1
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
        - `stone` 与分支左右关系判断
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

    def _compute_weighted_steer_signal(self, path_points, img_w, img_h, center_bias_x=0.0):
        """按“路径点到底部中点连线斜率”的加权平均聚合单一控制量."""
        if path_points is None or len(path_points) == 0:
            return 0.0

        pts = np.array(path_points, dtype=np.float32).reshape((-1, 2))
        # 以图像几何中线为 0；绕车时只临时偏移控制基准，不改普通循线路径点。
        bottom_mid_x = float(img_w) / 2.0 + float(center_bias_x)
        bottom_y = float(img_h) - 1.0
        min_dy = float(config.STEER_SIGNAL_MIN_DY)
        row_gamma = float(getattr(config, "STEER_SIGNAL_ROW_WEIGHT_GAMMA", 1.0))

        dy = np.maximum(bottom_y - pts[:, 1], min_dy)
        slopes = (pts[:, 0] - bottom_mid_x) / dy
        row_weights = np.power(np.clip(pts[:, 1], 0.0, bottom_y), row_gamma)
        weight_sum = float(np.sum(row_weights))
        if weight_sum <= 1e-6:
            return 0.0
        slope_signal = float(np.sum(slopes * row_weights) / weight_sum)
        slope_signal *= float(getattr(config, "STEER_SIGNAL_NORMALIZED_SCALE", 1.0))

        return slope_signal

    def _build_single_row_control_points(self, path_points, img_h, y_ratio=None):
        """把整条路径压成单行控制点，降低无金币时的抖动."""
        if path_points is None or len(path_points) == 0:
            return None

        if y_ratio is None:
            y_ratio = float(getattr(config, "SEG_STEER_SAMPLE_Y_RATIO", 0.6))

        pts = np.array(path_points, dtype=np.float32).reshape((-1, 2))
        sample_y = float(np.clip((float(img_h) - 1.0) * float(y_ratio), 0.0, float(img_h - 1)))
        sample_x = self._path_x_at_y_points(pts, sample_y)
        if sample_x is None:
            return None
        return np.array([[float(sample_x), sample_y]], dtype=np.float32)

    def _select_bottom_band_control_points(self, path_points, img_h, band_height=None):
        """只保留底部固定高度内的路径点参与无金币控制."""
        if path_points is None or len(path_points) == 0:
            return None

        if band_height is None:
            band_height = 96.0

        pts = np.array(path_points, dtype=np.float32).reshape((-1, 2))
        bottom_y = float(img_h) - 1.0
        y_min = max(0.0, bottom_y - float(band_height) + 1.0)
        selected = pts[pts[:, 1] >= y_min]
        if len(selected) == 0:
            return None
        return selected.astype(np.float32)

    def _select_no_target_control_points(self, path_points, img_h):
        """无金币/无车时，使用中下部指定行段参与控制，避开最底部几行."""
        if path_points is None or len(path_points) == 0:
            return None

        pts = np.array(path_points, dtype=np.float32).reshape((-1, 2))
        row_min = float(getattr(config, "STEER_SIGNAL_NO_TARGET_ROW_MIN", 60.0))
        row_max = float(getattr(config, "STEER_SIGNAL_NO_TARGET_ROW_MAX", 130.0))
        lo = max(0.0, min(row_min, row_max))
        hi = min(float(img_h) - 1.0, max(row_min, row_max))
        selected = pts[(pts[:, 1] >= lo) & (pts[:, 1] <= hi)]
        if len(selected) > 0:
            return selected.astype(np.float32)
        return self._select_bottom_band_control_points(path_points, img_h, band_height=max(1.0, float(img_h) - lo))

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

    def _point_near_mask(self, mask, point, radius):
        """判断点附近是否有赛道 mask，用来过滤不在可行区域附近的金币."""
        if mask is None or mask.size == 0:
            return True

        h, w = mask.shape[:2]
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        radius = max(0, int(radius))
        x1 = max(0, x - radius)
        x2 = min(w, x + radius + 1)
        y1 = max(0, y - radius)
        y2 = min(h, y + radius + 1)
        if x1 >= x2 or y1 >= y2:
            return False
        return bool(np.any(mask[y1:y2, x1:x2] > 0))

    def _build_segmented_coin_path(self, base_path, coin_points, w_seg, *, min_anchors=3):
        """按锚点从底部往远处分段重采样路径。"""
        planned = np.array(base_path, dtype=np.float32).reshape((-1, 2)).copy()
        if len(planned) < 2 or not coin_points:
            return planned, False

        bottom_pt = planned[int(np.argmax(planned[:, 1]))]
        far_pt = planned[int(np.argmin(planned[:, 1]))]
        y_min = float(min(bottom_pt[1], far_pt[1]))
        y_max = float(max(bottom_pt[1], far_pt[1]))

        anchors = [(float(bottom_pt[0]), float(bottom_pt[1]))]
        for cx, cy in sorted(coin_points, key=lambda pt: (-float(pt[1]), float(pt[0]))):
            cy = float(cy)
            if y_min <= cy <= y_max:
                anchors.append((float(np.clip(cx, 0.0, float(w_seg - 1))), cy))
        anchors.append((float(far_pt[0]), float(far_pt[1])))

        if len(anchors) < min_anchors:
            return planned, False

        anchors = sorted(anchors, key=lambda pt: float(pt[1]))
        unique_anchors = []
        for x, y in anchors:
            if unique_anchors and abs(float(y) - float(unique_anchors[-1][1])) < 1e-3:
                unique_anchors[-1] = (x, y)
            else:
                unique_anchors.append((x, y))

        if len(unique_anchors) < 2:
            return planned, False

        anchor_arr = np.array(unique_anchors, dtype=np.float32)
        planned[:, 0] = np.interp(planned[:, 1], anchor_arr[:, 1], anchor_arr[:, 0])
        planned[:, 0] = np.clip(planned[:, 0], 0.0, float(w_seg - 1))
        return planned, True

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

    def _build_car_clearing_path(self, base_path, avoid_path, clear_frames, fixed_bias_ready=False):
        """CLEARING: 特定贴边小车框固定左偏；车消失后再缓慢回正。"""
        base = np.array(base_path, dtype=np.float32).reshape((-1, 2))
        if len(base) < 2:
            return base, 0.0

        miss_frames = max(1, int(getattr(config, "CAR_AVOIDANCE_CLEARING_MISS_FRAMES", 2)))
        decay_frames = max(1, int(getattr(config, "CAR_AVOIDANCE_CLEARING_DECAY_FRAMES", 8)))
        residual_keep = float(getattr(config, "CAR_AVOIDANCE_CLEARING_RESIDUAL_KEEP", 0.35))
        residual_done = float(getattr(config, "CAR_AVOIDANCE_CLEARING_DONE_RESIDUAL", 0.06))
        max_left_bias = float(getattr(config, "CAR_AVOIDANCE_CLEARING_LEFT_BIAS_MAX", 10.0))

        # 只有车框贴右、贴底且高度较小时，才直接固定左偏。
        if fixed_bias_ready:
            left_bias = float(getattr(config, "CAR_AVOIDANCE_FIXED_LEFT_BIAS_VALUE", 70.0))
            bias_ratio = 1.0
        else:
            left_bias = max(0.0, max_left_bias)
            if clear_frames <= miss_frames:
                bias_ratio = 1.0
            else:
                t = float(np.clip((clear_frames - miss_frames) / float(decay_frames), 0.0, 1.0))
                bias_ratio = residual_keep + (residual_done - residual_keep) * t
            bias_ratio = float(np.clip(bias_ratio, 0.0, 1.0))

        mixed = base.copy()
        mixed[:, 0] = np.clip(mixed[:, 0] - left_bias * bias_ratio, 0.0, float(config.SEG_SIZE[0] - 1))
        return mixed, bias_ratio

    def _car_fixed_left_bias_ready(self, locked_car, w_seg, h_seg):
        """车框贴右下且高度较小时，使用基础巡线固定左偏。"""
        if locked_car is None:
            return False
        measurement = locked_car.get("measurement") or {}
        x_max = float(measurement.get("x_max", 0.0))
        y_min = float(measurement.get("y_min", 0.0))
        y_max = float(measurement.get("y_max", 0.0))
        box_h = max(0.0, y_max - y_min)
        height_thresh = float(getattr(config, "CAR_AVOIDANCE_FIXED_LEFT_BIAS_HEIGHT_THRESH", 90.0))
        right_margin = float(getattr(config, "CAR_AVOIDANCE_FIXED_LEFT_BIAS_RIGHT_MARGIN", 20.0))
        bottom_margin = float(getattr(config, "CAR_AVOIDANCE_FIXED_LEFT_BIAS_BOTTOM_MARGIN", 20.0))
        touch_right = x_max >= float(w_seg - 1) - max(0.0, right_margin)
        touch_bottom = y_max >= float(h_seg - 1) - max(0.0, bottom_margin)
        small_height = box_h <= max(0.0, height_thresh)
        return bool(touch_right and touch_bottom and small_height)

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

    def _coin_inside_selected_lane(self, coin_x, coin_y, base_path, left_boundary, right_boundary):
        """金币必须落在当前已选路径的左右边界内；边界缺失时回退到固定半宽."""
        left_x = self._boundary_x_at_y(left_boundary, coin_y)
        right_x = self._boundary_x_at_y(right_boundary, coin_y)
        if left_x is not None and right_x is not None:
            if right_x < left_x:
                left_x, right_x = right_x, left_x
            return left_x <= float(coin_x) <= right_x

        path_x = self._path_x_at_y_points(base_path, coin_y)
        if path_x is None:
            return False
        fixed_width = self._fixed_track_width_at_y(coin_y, 0.0)
        if fixed_width <= 0.0:
            return False
        half_width = 0.5 * fixed_width * float(getattr(config, "COIN_PATH_HALF_WIDTH_SCALE", 1.0))
        return abs(float(coin_x) - path_x) <= half_width

    def _coin_measurement_from_item(self, item, w_seg, h_seg):
        """从 coin 检测框取底部中心、面积和置信度."""
        seg_box = item.get("seg_box")
        if seg_box is None:
            return None
        box = np.array(seg_box, dtype=np.float32).reshape((-1, 2))
        if len(box) < 4:
            return None

        bottom_indices = np.argsort(box[:, 1])[-2:]
        bottom_pts = box[bottom_indices]
        bottom_center = np.mean(bottom_pts, axis=0)
        cx = float(np.clip(bottom_center[0], 0.0, float(w_seg - 1)))
        cy = float(np.clip(bottom_center[1], 0.0, float(h_seg - 1)))
        x_min = float(np.min(box[:, 0]))
        x_max = float(np.max(box[:, 0]))
        y_min = float(np.min(box[:, 1]))
        y_max = float(np.max(box[:, 1]))
        area = float(max(0.0, x_max - x_min) * max(0.0, y_max - y_min))
        return {
            "point": (cx, cy),
            "area": area,
            "box": (x_min, y_min, x_max, y_max),
            "score": float(item.get("score", 0.0)),
        }

    def _coin_measurement_usable(self, measurement, base, mask, left_boundary, right_boundary, *, for_new_lock):
        """过滤明显不可靠的 coin；新锁定目标额外要求在配置 ROI 内."""
        if measurement is None:
            return False

        cx, cy = measurement["point"]
        max_area = float(getattr(config, "COIN_TRACK_MAX_AREA", 0.0))
        if max_area > 0.0 and float(measurement.get("area", 0.0)) > max_area:
            return False

        if bool(getattr(config, "COIN_PATH_EDGE_REJECT_ENABLED", False)):
            box = measurement.get("box")
            if box is not None:
                x_min, _y_min, x_max, y_max = [float(v) for v in box]
                bottom_y = float(np.max(base[:, 1]))
                bottom_strict_rows = float(getattr(config, "COIN_PATH_ROI_BOTTOM_STRICT_ROWS", 0.0))
                edge_margin = max(0.0, float(getattr(config, "COIN_PATH_EDGE_REJECT_MARGIN", 0.0)))
                in_bottom_strict = bottom_strict_rows > 0.0 and y_max >= bottom_y - bottom_strict_rows
                if in_bottom_strict and (x_min <= edge_margin or x_max >= float(config.SEG_SIZE[0] - 1) - edge_margin):
                    return False

        if not self._coin_bottom_strict_point_usable(cx, cy, base):
            return False

        if for_new_lock:
            if cy < float(getattr(config, "COIN_PATH_ROI_Y_MIN", 0.0)):
                return False

        path_y_min = float(np.min(base[:, 1]))
        path_y_max = float(np.max(base[:, 1]))
        if not (path_y_min <= cy <= path_y_max):
            return False

        mask_radius = int(getattr(config, "COIN_PATH_MASK_RADIUS", 8))
        if not self._point_near_mask(mask, (cx, cy), mask_radius):
            return False
        return self._coin_inside_selected_lane(cx, cy, base, left_boundary, right_boundary)

    def _coin_bottom_strict_point_usable(self, coin_x, coin_y, base):
        """底部严格区内，coin 必须靠近当前路径中线且范围额外收窄。"""
        cx = float(coin_x)
        cy = float(coin_y)
        bottom_strict_rows = float(getattr(config, "COIN_PATH_ROI_BOTTOM_STRICT_ROWS", 0.0))
        bottom_y = float(np.max(np.array(base, dtype=np.float32).reshape((-1, 2))[:, 1]))
        if bottom_strict_rows > 0.0 and cy >= bottom_y - bottom_strict_rows:
            path_x = self._path_x_at_y_points(base, cy)
            if path_x is None:
                return False
            fixed_width = self._fixed_track_width_at_y(cy, 0.0)
            if fixed_width <= 0.0:
                return False
            half_width = 0.5 * fixed_width * float(getattr(config, "COIN_PATH_HALF_WIDTH_SCALE", 1.0))
            strict_scale = float(getattr(config, "COIN_PATH_BOTTOM_HALF_WIDTH_SCALE", 1.0))
            half_width *= strict_scale
            if half_width <= 0.0 or abs(float(cx) - path_x) > half_width:
                return False
        return True

    def _update_locked_coin(self, coin_measurements, base, h_seg, mask, left_boundary, right_boundary):
        """锁定当前金币；短暂跟丢时沿用旧点，超时后切换到新目标."""
        search_radius = float(getattr(config, "COIN_TRACK_SEARCH_RADIUS", 36.0))
        max_miss = max(0, int(getattr(config, "COIN_TRACK_MAX_MISS_FRAMES", 2)))
        eat_margin = max(0.0, float(getattr(config, "COIN_TRACK_EAT_Y_MARGIN", 6.0)))
        eat_y = float(h_seg - 1) - eat_margin

        usable_for_update = [
            m for m in coin_measurements
            if self._coin_measurement_usable(m, base, mask, left_boundary, right_boundary, for_new_lock=False)
        ]

        new_candidates = [
            m for m in coin_measurements
            if self._coin_measurement_usable(m, base, mask, left_boundary, right_boundary, for_new_lock=True)
        ]

        def _append_future_coins(current_point):
            current_x, current_y = current_point
            future = []
            for m in new_candidates:
                mx, my = m["point"]
                if abs(float(mx) - float(current_x)) <= 1e-3 and abs(float(my) - float(current_y)) <= 1e-3:
                    continue
                if float(my) < float(current_y):
                    future.append(m["point"])
            future.sort(key=lambda pt: (-float(pt[1]), float(pt[0])))
            return [current_point] + future

        if self.locked_coin is not None:
            locked_x, locked_y = self.locked_coin["point"]
            if not self._coin_bottom_strict_point_usable(locked_x, locked_y, base):
                self.locked_coin = None
                self.locked_coin_miss_frames = 0
                return []
            if locked_y >= eat_y:
                self.locked_coin = None
                self.locked_coin_miss_frames = 0
                return []

            local_matches = [
                m for m in usable_for_update
                if abs(float(m["point"][1]) - locked_y) <= search_radius and
                abs(float(m["point"][0]) - locked_x) <= search_radius
            ]
            if local_matches:
                best = min(
                    local_matches,
                    key=lambda m: (
                        abs(float(m["point"][1]) - locked_y),
                        abs(float(m["point"][0]) - locked_x),
                    ),
                )
                self.locked_coin = best
                self.locked_coin_miss_frames = 0
                best_x, best_y = best["point"]
                if not self._coin_bottom_strict_point_usable(best_x, best_y, base):
                    self.locked_coin = None
                    self.locked_coin_miss_frames = 0
                    return []
                return _append_future_coins(best["point"])

            bottom_strict_rows = float(getattr(config, "COIN_PATH_ROI_BOTTOM_STRICT_ROWS", 0.0))
            bottom_y = float(np.max(base[:, 1]))
            if bottom_strict_rows > 0.0 and locked_y >= bottom_y - bottom_strict_rows:
                self.locked_coin = None
                self.locked_coin_miss_frames = 0
                return []

            self.locked_coin_miss_frames += 1
            if self.locked_coin_miss_frames <= max_miss:
                if not self._coin_bottom_strict_point_usable(locked_x, locked_y, base):
                    self.locked_coin = None
                    self.locked_coin_miss_frames = 0
                    return []
                return _append_future_coins(self.locked_coin["point"])
            self.locked_coin = None
            self.locked_coin_miss_frames = 0

        if not new_candidates:
            return []

        sorted_candidates = sorted(
            new_candidates,
            key=lambda m: (float(m["point"][1]), float(m.get("score", 0.0))),
            reverse=True,
        )
        best = sorted_candidates[0]
        bottom_too_close_rows = float(getattr(config, "COIN_TRACK_BOTTOM_TOO_CLOSE_ROWS", 0.0))
        if bottom_too_close_rows > 0.0 and best["point"][1] >= float(np.max(base[:, 1])) - bottom_too_close_rows:
            if len(sorted_candidates) > 1:
                best = sorted_candidates[1]
        self.locked_coin = best
        self.locked_coin_miss_frames = 0
        return _append_future_coins(best["point"])

    def _apply_coin_path_planning(self, base_path, planning_items, w_seg, h_seg, mask=None,
                                  left_boundary=None, right_boundary=None, blocked_y_ranges=None,
                                  car_state="FOLLOW_LANE"):
        """只使用 coin 底部中点画点，并按这些点从底部依次分段拟合路径."""
        debug = {
            "coin_points": [],
            "control_coin": None,
            "control_points": None,
            "control_gain": 1.0,
            "raw_coin_items": 0,
            "coin_measurements": 0,
            "coin_points_count": 0,
            "path_changed": False,
            "active": False,
            "car_state": car_state,
            "car_gate_active": False,
            "measurement_none": 0,
            "reject_blocked": 0,
            "reject_car_state": 0,
        }
        base = np.array(base_path, dtype=np.float32).reshape((-1, 2))
        if (
            len(base) < 2 or
            not bool(getattr(config, "COIN_PATH_ENABLED", True))
        ):
            return base, debug

        coin_measurements = []
        base_bottom_y = float(np.max(base[:, 1]))
        blocked_ranges = list(blocked_y_ranges or [])
        debug["car_gate_active"] = bool(blocked_ranges)
        blocked_y_max = None
        for y1, y2 in blocked_ranges:
            blocked_y_max = max(float(y1), float(y2)) if blocked_y_max is None else max(
                blocked_y_max,
                float(y1),
                float(y2),
            )

        for item in planning_items:
            if item.get("class_name", "") != "coin":
                continue
            debug["raw_coin_items"] += 1
            measurement = self._coin_measurement_from_item(item, w_seg, h_seg)
            if measurement is None:
                debug["measurement_none"] += 1
                continue

            cy = float(measurement["point"][1])
            blocked = False
            for y1, y2 in blocked_ranges:
                lo = min(float(y1), float(y2))
                hi = max(float(y1), float(y2))
                if lo <= cy <= hi:
                    blocked = True
                    break
            if blocked:
                debug["reject_blocked"] += 1
                continue

            if blocked_ranges and car_state == "AVOIDING":
                allow_rows = float(getattr(config, "CAR_AVOIDANCE_COIN_ALLOW_BOTTOM_ROWS", 28.0))
                if blocked_y_max is not None and cy < float(blocked_y_max) - allow_rows:
                    debug["reject_car_state"] += 1
                    continue
            elif blocked_ranges and car_state == "CLEARING":
                safe_rows = float(getattr(config, "CAR_AVOIDANCE_CLEARING_COIN_SAFE_ROWS", 16.0))
                if cy < base_bottom_y - safe_rows:
                    debug["reject_car_state"] += 1
                    continue
                if not self._coin_bottom_strict_point_usable(measurement["point"][0], cy, base):
                    debug["reject_car_state"] += 1
                    continue

            coin_measurements.append(measurement)
        debug["coin_measurements"] = len(coin_measurements)

        coin_points = self._update_locked_coin(
            coin_measurements,
            base,
            h_seg,
            mask,
            left_boundary,
            right_boundary,
        )
        debug["coin_points_count"] = len(coin_points)

        if not coin_points:
            return base, debug

        planned, changed = self._build_segmented_coin_path(base, coin_points, w_seg)
        debug["path_changed"] = bool(changed)
        if not changed:
            return base, debug

        debug["coin_points"] = coin_points
        debug["control_coin"] = coin_points[0]
        debug["planned_path"] = planned.astype(np.float32)
        if bool(getattr(config, "COIN_PATH_CONTROL_FIRST_SEGMENT_ONLY", True)):
            nearest_coin = debug["control_coin"]
            bottom_y = float(np.max(planned[:, 1]))
            first_coin_y = float(nearest_coin[1])
            band_height = float(getattr(config, "COIN_PATH_CONTROL_BAND_HEIGHT", 96.0))
            control_top_y = max(float(np.min(planned[:, 1])), bottom_y - max(1.0, band_height) + 1.0)
            low_y = max(min(bottom_y, first_coin_y), control_top_y)
            high_y = bottom_y
            segment_mask = (planned[:, 1] >= low_y) & (planned[:, 1] <= high_y)
            control_points = planned[segment_mask]
            if len(control_points) >= 2:
                debug["control_points"] = control_points.astype(np.float32)
                coin_distance_rows = max(1.0, bottom_y - first_coin_y)
                if coin_distance_rows < band_height:
                    reference_rows = float(getattr(config, "COIN_PATH_CONTROL_REFERENCE_ROWS", band_height))
                    debug["control_gain"] = max(1.0, reference_rows / coin_distance_rows)

        debug["active"] = True
        return planned, debug

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
        hit_required = max(1, int(getattr(config, "CAR_AVOIDANCE_LOCK_HIT_FRAMES", 1)))
        search_radius = max(1.0, float(getattr(config, "CAR_AVOIDANCE_SEARCH_RADIUS", 48.0)))
        miss_gain = max(0.0, float(getattr(config, "CAR_AVOIDANCE_SEARCH_RADIUS_MISS_GAIN", 0.0)))
        ema_alpha = float(getattr(config, "CAR_AVOIDANCE_TRACK_EMA_ALPHA", 0.65))
        ema_alpha = float(np.clip(ema_alpha, 0.0, 0.98))

        if self.locked_car is not None:
            last_center = self.locked_car.get("bottom_center")
            velocity = self.locked_car.get("velocity", (0.0, 0.0))
            miss_frames = int(self.locked_car_miss_frames)
            predicted = (
                float(last_center[0]) + float(velocity[0]),
                float(last_center[1]) + float(velocity[1]),
            )
            radius = search_radius + miss_gain * float(miss_frames)

            local_matches = []
            for m in measurements:
                mx, my = m["bottom_center"]
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
                }
                self.locked_car_miss_frames = 0
                if not self.locked_car["confirmed"]:
                    return None
                return self.locked_car

            self.locked_car_miss_frames += 1
            if self.locked_car_miss_frames <= max_miss:
                if not bool(self.locked_car.get("confirmed", False)):
                    return None
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
        }
        self.locked_car_miss_frames = 0
        if not self.locked_car["confirmed"]:
            return None
        return self.locked_car

    def _car_avoidance_center_bias(self, locked_car, base, w_seg, h_seg):
        """基于锁定目标计算控制基准偏移量，不改普通循线路径点."""
        if locked_car is None:
            return 0.0, None
        measurement = locked_car.get("measurement")
        if measurement is None:
            return 0.0, None

        path_y_min = float(np.min(base[:, 1]))
        path_y_max = float(np.max(base[:, 1]))
        smooth_cx, smooth_cy = locked_car.get("bottom_center", measurement["bottom_center"])
        center_y = float(np.clip(smooth_cy, path_y_min, path_y_max))
        path_x_at_car = self._path_x_at_y_points(base, center_y)
        if path_x_at_car is None:
            bias_sign = float(getattr(self, "car_last_center_bias_sign", 1.0))
        elif float(smooth_cx) < float(path_x_at_car):
            bias_sign = -1.0
        else:
            bias_sign = 1.0
        path_bottom_y = float(np.max(base[:, 1]))
        rows_to_car = max(0.0, path_bottom_y - center_y)
        start_rows = max(0.0, float(getattr(config, "CAR_AVOIDANCE_START_OFFSET_ROWS", 100.0)))
        near_rows = max(0.0, float(getattr(config, "CAR_AVOIDANCE_NEAR_OFFSET_ROWS", 50.0)))
        start_offset = max(0.0, float(getattr(config, "CAR_AVOIDANCE_START_LEFT_OFFSET", 50.0)))
        near_offset = max(0.0, float(getattr(config, "CAR_AVOIDANCE_NEAR_LEFT_OFFSET", 70.0)))

        if start_rows > 0.0 and rows_to_car > start_rows:
            return 0.0, None
        bias = near_offset if rows_to_car <= near_rows else start_offset
        return float(bias) * float(bias_sign), (center_y, path_bottom_y)

    def _update_car_avoidance_center_bias(self, base_path, planning_items, w_seg, h_seg):
        """检测到 car 时，先锁定同一辆车，再输出控制基准偏移量."""
        debug = {
            "blocked_y_range": None,
            "center_bias_x": 0.0,
            "state": self.car_avoidance_state,
            "clear_frames": int(self.car_clearing_frames),
            "active": False,
        }
        base = np.array(base_path, dtype=np.float32).reshape((-1, 2))
        if len(base) < 2 or not bool(getattr(config, "CAR_AVOIDANCE_ENABLED", True)):
            return 0.0, debug

        measurements = []
        for item in planning_items:
            measurement = self._car_measurement_from_item(item, w_seg, h_seg, base)
            if measurement is not None:
                measurements.append(measurement)

        locked_car = self._update_locked_car(measurements, base)
        if locked_car is None:
            if self.car_avoidance_state == "AVOIDING" and self.car_last_avoid_path is not None:
                self.car_avoidance_state = "CLEARING"
                self.car_clearing_frames = 0
            if self.car_avoidance_state == "CLEARING" and self.car_last_avoid_path is not None:
                self.car_clearing_frames += 1
                _clearing_path, avoid_weight = self._build_car_clearing_path(
                    base,
                    self.car_last_avoid_path,
                    self.car_clearing_frames,
                    fixed_bias_ready=False,
                )
                done_residual = float(getattr(config, "CAR_AVOIDANCE_CLEARING_DONE_RESIDUAL", 0.06))
                done_frames = max(
                    1,
                    int(getattr(config, "CAR_AVOIDANCE_CLEARING_DECAY_FRAMES", 8)),
                )
                if self.car_clearing_frames >= done_frames and avoid_weight <= done_residual:
                    self.car_avoidance_state = "FOLLOW_LANE"
                    self.car_clearing_frames = 0
                    self.car_last_avoid_path = None
                    self.car_last_blocked_y_range = None
                    debug["state"] = self.car_avoidance_state
                    debug["clear_frames"] = 0
                    return 0.0, debug
                miss_bias = (
                    float(getattr(config, "CAR_AVOIDANCE_CLEARING_LEFT_BIAS_MAX", 30.0)) *
                    float(avoid_weight) *
                    float(getattr(self, "car_last_center_bias_sign", 1.0))
                )
                debug["state"] = self.car_avoidance_state
                debug["clear_frames"] = int(self.car_clearing_frames)
                debug["center_bias_x"] = float(miss_bias)
                debug["active"] = True
                return float(miss_bias), debug
            return 0.0, debug

        if self._car_fixed_left_bias_ready(locked_car, w_seg, h_seg):
            self.car_avoidance_state = "CLEARING"
            self.car_clearing_frames = 0
            bias = float(getattr(config, "CAR_AVOIDANCE_FIXED_LEFT_BIAS_VALUE", 50.0))
            self.car_last_center_bias_sign = 1.0
            self.car_last_avoid_path = base.copy()
            debug["state"] = self.car_avoidance_state
            debug["clear_frames"] = 0
            debug["fixed_left_bias"] = True
            debug["center_bias_x"] = float(bias)
            debug["active"] = True
            return float(bias), debug

        bias, y_range = self._car_avoidance_center_bias(locked_car, base, w_seg, h_seg)

        if abs(float(bias)) <= 0.0:
            self.locked_car = None
            self.locked_car_miss_frames = 0
            return 0.0, debug

        if y_range is not None:
            debug["blocked_y_range"] = (float(y_range[0]), float(y_range[1]))
        self.car_last_center_bias_sign = 1.0 if float(bias) >= 0.0 else -1.0
        self.car_avoidance_state = "AVOIDING"
        self.car_clearing_frames = 0
        self.car_last_avoid_path = base.copy()
        self.car_last_blocked_y_range = debug["blocked_y_range"]
        debug["state"] = self.car_avoidance_state
        debug["clear_frames"] = 0
        debug["center_bias_x"] = float(bias)
        debug["active"] = True
        return float(bias), debug

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
        infer_s=0.0,
        total_start=None,
        preview_frame=None,
    ):
        """对已推理出的 mask 做路径规划、控制量计算和调试渲染.

        输入:
        - blob_rgb_320: 分割线程当前拿到的最新 SEG_SIZE RGB 图
        - mask: infer_mask 输出的二值赛道 mask
        - current_yolo_boxes: 当前最新一帧检测框，坐标在 TARGET_RES
        - turn_intent: OCR 给出的 LEFT / RIGHT 分叉意图，当前暂不参与分支选择

        输出:
        - steer_signal: 单一转向控制量，来自路径点加权斜率和
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
        merge_detect_info = self._detect_merge_guide(search_mask, search_edge_mask)
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
        best_path = None
        best_nodes = None
        stone_branch_side = 0
        coin_path_debug = None
        car_path_debug = None
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

                candidate_pool = [left_best, right_best]
                stone_branch_side = self._estimate_stone_side_by_fork_divider(
                    planning_items,
                    y_fork_info.get("fork_point"),
                    search_mask.shape,
                )
                if stone_branch_side == 0:
                    stone_branch_side = self._estimate_stone_branch_side(planning_items, candidate_pool)
                preferred_turn = self._resolve_preferred_turn(stone_branch_side)
                best_candidate = right_best if preferred_turn == 1 else left_best
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
                        if fork_active:
                            candidate_pool = [fork_left, fork_right]
                            stone_branch_side = self._estimate_stone_branch_side(planning_items, candidate_pool)
                            if stone_branch_side == -1:
                                best_candidate = fork_right
                            elif stone_branch_side == 1:
                                best_candidate = fork_left

                    # 没有确认 Y 型分叉时，不再按 LEFT/RIGHT 去取最左/最右候选。
                    # 只有检测到石头且普通候选能稳定分成左右两支时，才按石头所在侧选对侧避让；
                    # 否则单路/汇合场景直接选路径搜索得分最高的候选。
                    best_path = best_candidate["path"]
                    best_nodes = best_candidate["nodes"]

        if best_path is not None:
            guide_polyline = None if merge_guide_info is None else merge_guide_info.get("guide_polyline")
            fit_nodes = self._apply_merge_boundary_width(best_nodes, merge_side, guide_polyline)
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

            # 4. 时域一阶低通滤波 (EMA)，赋予路径物理连贯惯性，消除分叉口反复横跳
            if self.last_poly_coeffs is not None and len(self.last_poly_coeffs) == len(current_coeffs):
                poly_coeffs = self.ema_alpha * self.last_poly_coeffs + (1.0 - self.ema_alpha) * current_coeffs
            else:
                poly_coeffs = current_coeffs

            dense_y = np.linspace(node_y[0], node_y[-1], num=int(config.SEG_PATH_DENSE_SAMPLES))
            dense_x = np.polyval(poly_coeffs, dense_y)
            dense_x = np.clip(dense_x, 0, w_seg - 1)
            dense_y = np.clip(dense_y, 0, h_seg - 1)

            path_points_orig = np.vstack((dense_x, dense_y)).astype(np.float32).T
            base_path_points = path_points_orig.copy()
            car_center_bias_x, car_path_debug = self._update_car_avoidance_center_bias(
                path_points_orig,
                planning_items,
                w_seg,
                h_seg,
            )
            car_active = car_path_debug is not None and car_path_debug.get("active")
            car_state = "FOLLOW_LANE"
            if car_path_debug is not None:
                car_state = str(car_path_debug.get("state", "FOLLOW_LANE"))
            blocked_y_ranges = []
            if car_active and car_path_debug.get("blocked_y_range") is not None:
                blocked_y_ranges.append(car_path_debug["blocked_y_range"])
            _, coin_path_debug = self._apply_coin_path_planning(
                path_points_orig,
                planning_items,
                w_seg,
                h_seg,
                mask=search_mask,
                left_boundary=left_boundary_pts,
                right_boundary=right_boundary_pts,
                blocked_y_ranges=blocked_y_ranges,
                car_state=car_state,
            )
            merged_anchor_points = []
            if coin_path_debug is not None and coin_path_debug.get("active"):
                merged_anchor_points.extend(coin_path_debug.get("coin_points", []))

            if merged_anchor_points:
                merged_path, merged_changed = self._build_segmented_coin_path(
                    base_path_points,
                    merged_anchor_points,
                    w_seg,
                    min_anchors=3,
                )
                if merged_changed:
                    path_points_orig = merged_path
                    if coin_path_debug is not None:
                        coin_path_debug["planned_path"] = merged_path.astype(np.float32)
                        coin_path_debug["control_points"] = merged_path.astype(np.float32)
                        coin_path_debug["control_coin"] = tuple(map(float, merged_path[0]))

            coin_active = coin_path_debug is not None and coin_path_debug.get("active")
            bypass_frame_jump = (
                (coin_active or car_active) and
                bool(getattr(config, "COIN_PATH_BYPASS_FRAME_JUMP", True))
            )
            if bypass_frame_jump:
                path_jump_limited = False
            else:
                path_points_orig, path_jump_limited = self._limit_path_frame_jump(path_points_orig)
            if path_jump_limited or coin_active or car_active:
                poly_coeffs = self._fit_path_poly_coeffs(
                    path_points_orig[:, 1],
                    path_points_orig[:, 0],
                )

            self.last_poly_coeffs = poly_coeffs
            self.last_path_points_orig = path_points_orig.copy()
            self.missing_path_frames = 0
            control_path_points = path_points_orig
            if coin_path_debug is not None and coin_path_debug.get("active"):
                if coin_path_debug.get("control_points") is not None:
                    control_path_points = coin_path_debug["control_points"]
            else:
                no_target_points = self._select_no_target_control_points(path_points_orig, h_seg)
                if no_target_points is not None:
                    control_path_points = no_target_points
            steer_signal = self._compute_weighted_steer_signal(
                control_path_points,
                w_seg,
                h_seg,
                center_bias_x=car_center_bias_x if car_active else 0.0,
            )
            if coin_path_debug is not None and coin_path_debug.get("active"):
                steer_signal *= float(coin_path_debug.get("control_gain", 1.0))
                steer_signal *= float(getattr(config, "STEER_SIGNAL_COIN_GAIN", 1.0))
            elif car_active:
                steer_signal *= float(getattr(config, "STEER_SIGNAL_CAR_GAIN", 1.0))
            elif not car_active:
                steer_signal *= float(getattr(config, "STEER_SIGNAL_NO_TARGET_GAIN", 1.0))
            pts_final_orig = path_points_orig.reshape((-1, 1, 2))
        else:
            held_path = self._hold_last_path()
            if held_path is not None:
                path_points_orig = held_path
                steer_signal = self._compute_weighted_steer_signal(path_points_orig, w_seg, h_seg)
                pts_final_orig = path_points_orig.reshape((-1, 1, 2))
            else:
                self.last_poly_coeffs = None
        self.last_branch_stats = {
            "branch_pair_count_max": int(branch_pair_count_max),
            "branch_support_rows": int(branch_support_rows),
            "fork_active": bool(fork_active),
            "y_fork_active": bool(y_fork_active),
            "merge_side": merge_side,
            "merge_state_active": bool(self.merge_state_active),
            "merge_state_hit_frames": int(self.merge_state_hit_frames),
            "merge_state_exit_frames": int(self.merge_state_exit_frames),
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
            fork_point=y_fork_info.get("fork_point") if y_fork_active else None,
            coin_path=coin_path_debug,
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

        servo_pwm = int(
            config.SERVO_CENTER
            - steer_signal * config.STEER_SIGNAL_PWM_GAIN
        )
        servo_pwm = int(max(config.SERVO_MIN, min(config.SERVO_MAX, servo_pwm)))
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
        if 'stone_branch_side' in locals():
            if stone_branch_side == -1:
                stone_side_text = "LEFT"
            elif stone_branch_side == 1:
                stone_side_text = "RIGHT"
            else:
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
                f"PairsMax:{self.last_branch_stats.get('branch_pair_count_max', 0)} "
                f"Rows2+:{self.last_branch_stats.get('branch_support_rows', 0)} "
                f"Y:{int(self.last_branch_stats.get('y_fork_active', False))} "
                f"Merge:{self.last_branch_stats.get('merge_side') or 'NONE'}"
            ),
            config.SEG_DEBUG_TEXT_POS_BRANCH,
            1,
            config.SEG_DEBUG_TEXT_FONT_SCALE,
            config.SEG_DEBUG_TEXT_COLOR_BRANCH,
            config.SEG_DEBUG_TEXT_THICKNESS,
        )
        if coin_path_debug is not None:
            cv2.putText(
                ai_view,
                (
                    f"Coin raw:{coin_path_debug.get('raw_coin_items', 0)} "
                    f"meas:{coin_path_debug.get('coin_measurements', 0)} "
                    f"pts:{coin_path_debug.get('coin_points_count', 0)} "
                    f"chg:{int(bool(coin_path_debug.get('path_changed', False)))} "
                    f"rej:{coin_path_debug.get('reject_car_state', 0)}/"
                    f"{coin_path_debug.get('reject_blocked', 0)} "
                    f"st:{str(coin_path_debug.get('car_state', ''))[:3]} "
                    f"gate:{int(bool(coin_path_debug.get('car_gate_active', False)))}"
                ),
                config.SEG_DEBUG_TEXT_POS_COIN,
                1,
                config.SEG_DEBUG_TEXT_FONT_SCALE,
                config.SEG_DEBUG_TEXT_COLOR_COIN,
                config.SEG_DEBUG_TEXT_THICKNESS,
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

    def run(self, blob_rgb_320, current_yolo_boxes, turn_intent, fps_stats):
        """兼容旧串行调用：推理和后处理在同一个线程里连续执行."""
        t_total_start = time.perf_counter()
        mask, infer_s = self.infer_mask(blob_rgb_320)
        return self.postprocess_mask(
            blob_rgb_320,
            mask,
            current_yolo_boxes,
            turn_intent,
            fps_stats,
            infer_s=infer_s,
            total_start=t_total_start,
        )

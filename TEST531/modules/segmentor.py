# modules/segmentor.py
"""赛道分割与路径规划模块.

这个模块负责:
1. 调用分割模型得到二值赛道 mask
2. 在 mask 空间中搜索一条可跟踪路径（引入分支局部中心约束，抑制切内线）
3. 基于路径点到图像底部中点的加权斜率和，生成单一转向控制量
4. 引入多项式时域低通滤波 (EMA)，提升路径稳定性
5. 返回用于控制的 steer_signal，以及一张调试渲染图

当前路径选择策略的核心优先级是:
1. 先从 mask 里搜索可连接的候选路径
2. 如果检测到 `stone`，优先绕开石头所在分支
3. 否则默认偏向左支
4. 如果 OCR 已经给出 `turn_intent`，再用 LEFT / RIGHT 意图覆盖默认偏向
"""

import cv2
import numpy as np
import time
from rknnlite.api import RKNNLite
import config

class RoadSegmentor:
    def __init__(self, core_id):
        """初始化分割模型与 320 空间的逆透视矩阵."""
        self.rknn = RKNNLite()
        print(f"--> [Segmentor] 正在初始化 NPU Core {core_id}...", flush=True)
        
        if self.rknn.load_rknn(config.SEG_MODEL) != 0 or self.rknn.init_runtime(core_mask=core_id) != 0:
            raise RuntimeError("Seg 模型加载或初始化失败")
            
        w_seg, h_seg = config.SEG_SIZE # 320, 320
        src_seg = np.float32([[x * w_seg, y * h_seg] for x, y in config.SRC_PTS])
        dst_seg = np.float32([[x * w_seg, y * h_seg] for x, y in config.DST_PTS])
        # 透视矩阵直接建立在分割输入尺寸上，避免每次运行都重复计算。
        self.M_seg = cv2.getPerspectiveTransform(src_seg, dst_seg)
        self.scale_x_to_seg = w_seg / float(config.TARGET_RES[0])
        self.scale_y_to_seg = h_seg / float(config.TARGET_RES[1])
        self.planning_class_names = set(
            getattr(config, "PLANNING_CLASS_NAMES", ())
        )
        self.planning_marker_styles = dict(
            getattr(config, "PLANNING_MARKER_STYLES", {})
        )
        self.planning_circle_class_names = set(
            getattr(config, "PLANNING_CIRCLE_CLASS_NAMES", ())
        )

        # -------------------------------------------------------------------
        # 时域滤波历史记忆
        # -------------------------------------------------------------------
        self.last_poly_coeffs = None
        self.ema_alpha = float(config.SEG_EMA_ALPHA)
        self.last_branch_stats = {
            "branch_pair_count_max": 0,
            "branch_support_rows": 0,
            "fork_active": False,
            "y_fork_active": False,
            "y_fork_bottom_width": 0.0,
        }
        self.last_main_overlay = {
            "path": None,
            "left": None,
            "right": None,
            "candidate_left": None,
            "candidate_right": None,
            "fork_point": None,
            "bottom_mid": (0.0, 0.0),
            "base_size": tuple(config.SEG_SIZE),
        }
        self.seg_profile_enabled = bool(getattr(config, "SEG_PROFILE_ENABLED", False))
        self.seg_profile_print_interval = float(getattr(config, "SEG_PROFILE_PRINT_INTERVAL", 1.0))
        self.seg_profile_last_print = time.perf_counter()
        self.seg_profile_acc = {
            "frames": 0,
            "infer": 0.0,
            "preprocess": 0.0,
            "search": 0.0,
            "fit": 0.0,
            "render": 0.0,
            "total": 0.0,
        }

    def _store_main_overlay(
        self,
        path_pts,
        left_pts,
        right_pts,
        img_w,
        img_h,
        candidate_left_pts=None,
        candidate_right_pts=None,
        fork_point=None,
    ):
        """缓存主图路径叠加层，供主线程在其它元素之上重绘."""
        self.last_main_overlay = {
            "path": None if path_pts is None else np.array(path_pts, dtype=np.float32).copy(),
            "left": None if left_pts is None else np.array(left_pts, dtype=np.float32).copy(),
            "right": None if right_pts is None else np.array(right_pts, dtype=np.float32).copy(),
            "candidate_left": None if candidate_left_pts is None else np.array(candidate_left_pts, dtype=np.float32).copy(),
            "candidate_right": None if candidate_right_pts is None else np.array(candidate_right_pts, dtype=np.float32).copy(),
            "fork_point": None if fork_point is None else (float(fork_point[0]), float(fork_point[1])),
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

        path_thickness = max(1, int(round(config.SEG_DEBUG_PATH_THICKNESS * scale)))
        boundary_thickness = max(1, int(round(config.SEG_DEBUG_BOUNDARY_THICKNESS * scale)))
        candidate_path_thickness = max(1, int(round(config.SEG_DEBUG_CANDIDATE_PATH_THICKNESS * scale)))
        bottom_mid_radius = max(1, int(round(config.SEG_DEBUG_BOTTOM_MID_RADIUS * scale)))

        if candidate_left_poly is not None:
            cv2.polylines(
                image,
                [candidate_left_poly],
                False,
                config.SEG_DEBUG_LEFT_PATH_COLOR,
                candidate_path_thickness,
            )
        if candidate_right_poly is not None:
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
        if left_poly is not None:
            cv2.polylines(
                image,
                [left_poly],
                False,
                config.SEG_DEBUG_LEFT_BOUNDARY_COLOR,
                boundary_thickness,
            )
        if right_poly is not None:
            cv2.polylines(
                image,
                [right_poly],
                False,
                config.SEG_DEBUG_RIGHT_BOUNDARY_COLOR,
                boundary_thickness,
            )

        bottom_mid = overlay.get("bottom_mid", (float(base_w) / 2.0, float(base_h) - 1.0))
        bottom_mid_pt = (
            int(round(float(bottom_mid[0]) * scale_x)),
            int(round(float(bottom_mid[1]) * scale_y)),
        )
        fork_point = overlay.get("fork_point")
        if fork_point is not None:
            fork_pt = (
                int(round(float(fork_point[0]) * scale_x)),
                int(round(float(fork_point[1]) * scale_y)),
            )
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

    def _profile_add(self, infer_s, preprocess_s, search_s, fit_s, render_s, total_s):
        """累计分割链路阶段耗时，并按时间窗口打印平均值."""
        if not self.seg_profile_enabled:
            return

        acc = self.seg_profile_acc
        acc["frames"] += 1
        acc["infer"] += infer_s
        acc["preprocess"] += preprocess_s
        acc["search"] += search_s
        acc["fit"] += fit_s
        acc["render"] += render_s
        acc["total"] += total_s

        now = time.perf_counter()
        interval = now - self.seg_profile_last_print
        if interval < self.seg_profile_print_interval or acc["frames"] <= 0:
            return

        frames = float(acc["frames"])
        print(
            (
                "[SegmentorProfile] "
                f"frames={int(frames)} "
                f"infer={acc['infer'] * 1000.0 / frames:.2f}ms "
                f"pre={acc['preprocess'] * 1000.0 / frames:.2f}ms "
                f"search={acc['search'] * 1000.0 / frames:.2f}ms "
                f"fit={acc['fit'] * 1000.0 / frames:.2f}ms "
                f"render={acc['render'] * 1000.0 / frames:.2f}ms "
                f"total={acc['total'] * 1000.0 / frames:.2f}ms"
            ),
            flush=True,
        )
        self.seg_profile_last_print = now
        self.seg_profile_acc = {
            "frames": 0,
            "infer": 0.0,
            "preprocess": 0.0,
            "search": 0.0,
            "fit": 0.0,
            "render": 0.0,
            "total": 0.0,
        }

    def _prepare_search_mask(self, mask):
        """为路径搜索准备更连贯的 mask，只修补底部附近的小断裂."""
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
        return (search_mask > 0).astype(np.uint8)

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

            if (
                left_growth < min_side_growth or
                right_growth < min_side_growth or
                gap_growth < min_gap_growth
            ):
                return None

            return {
                "run_rows": run_rows,
                "score": (
                    len(run_rows),
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
            return {"active": False, "bottom_width": 0.0, "fork_point": None, "split_rows": 0}

        h, w = search_mask.shape[:2]
        mask_rows = np.where(np.any(search_mask > 0, axis=1))[0]
        if len(mask_rows) > 0:
            top_y = int(mask_rows[0])
        else:
            top_y = 0

        bottom_margin = max(0, int(config.SEG_PATH_BOTTOM_MARGIN))
        bottom_band_height = max(1, int(config.FORK_BOTTOM_BAND_HEIGHT))
        bottom_y_end = max(1, h - bottom_margin)
        bottom_y_start = max(0, bottom_y_end - bottom_band_height)

        bottom_width = 0.0
        for sample_y in range(bottom_y_end - 1, bottom_y_start - 1, -1):
            runs = self._find_mask_runs(
                search_mask[sample_y],
                config.FORK_MASK_GAP_THRESH,
                config.FORK_MASK_MIN_BRANCH_PIXELS,
            )
            if not runs:
                continue
            widest_run = max(runs, key=lambda run: float(run["width"]))
            bottom_width = max(bottom_width, float(widest_run["width"]))

        # 暂时关闭“底部宽度必须足够大”的硬门槛，先观察全局分叉检测效果。
        # if bottom_width < float(config.FORK_MIN_BOTTOM_WIDTH):
        #     return {
        #         "active": False,
        #         "bottom_width": bottom_width,
        #         "fork_point": None,
        #         "split_rows": 0,
        #     }

        # 分叉改为在整个有效高度范围里全局检查，而不是只盯住某个固定高度带。
        branch_rows = []
        for sample_y in range(top_y, bottom_y_start + 1):
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
                "bottom_width": bottom_width,
                "fork_point": None,
                "split_rows": 0,
            }

        opening_run = self._select_inner_gap_opening_run(branch_rows)
        if opening_run is None:
            return {
                "active": False,
                "bottom_width": bottom_width,
                "fork_point": None,
                "split_rows": 0,
            }

        lowest_split = opening_run[0]

        return {
            "active": True,
            "bottom_width": bottom_width,
            "fork_point": (float(lowest_split["fork_x"]), float(lowest_split["y"])),
            "split_rows": max(1, int(len(opening_run))),
        }

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

        return segments

    def _search_active_paths(self, search_mask, edge_mask):
        """在给定 mask 区域内做一次自底向上的多候选路径搜索."""
        h_seg, _ = search_mask.shape[:2]
        step_y = int(config.SEG_PATH_SEARCH_STEP_Y)
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
                if len(active_paths) > int(config.SEG_PATH_MAX_ACTIVE_PATHS):
                    active_paths = self._prune_active_paths(active_paths)

            curr_y -= step_y

        return active_paths, branch_pair_count_max, branch_support_rows

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
            avg_x = float(np.mean(px))

            valid_candidates.append({
                "path": path_arr,
                "nodes": path,
                "score": base_score,
                "avg_x": avg_x,
            })

        return valid_candidates

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

    def _prune_active_paths(self, active_paths):
        """裁剪候选路径数量，同时尽量保住左右代表分支."""
        max_paths = int(config.SEG_PATH_MAX_ACTIVE_PATHS)
        if len(active_paths) <= max_paths:
            return active_paths

        ranked_paths = sorted(active_paths, key=lambda p: len(p), reverse=True)
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
        stone_items = [
            item for item in planning_items
            if item.get("class_name") == "stone" and item.get("seg_box") is not None
        ]
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

    def _resolve_preferred_turn(self, stone_branch_side, turn_intent):
        """融合石头避让和 OCR 意图，得到最终偏左/偏右选择."""
        if stone_branch_side == -1:
            return 1
        if stone_branch_side == 1:
            return -1
        return turn_intent if turn_intent in (-1, 1) else -1

    def _get_planning_box_points(self, obj, w_seg, h_seg):
        """把原图检测框转换成分割平面中的整框四角点."""
        rect = obj.get("rect", [0, 0, 0, 0])
        class_name = obj.get("class_name", "")

        if class_name not in self.planning_class_names or len(rect) != 4:
            return None

        x, y, w, h = rect
        if w <= 1 or h <= 1:
            return None

        corners = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ], dtype=np.float32)

        corners[:, 0] *= self.scale_x_to_seg
        corners[:, 1] *= self.scale_y_to_seg
        corners[:, 0] = np.clip(corners[:, 0], 0, w_seg - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, h_seg - 1)
        return corners

    def _project_planning_objects(self, current_yolo_boxes, w_seg, h_seg):
        """将原图中的规划相关检测框映射到分割平面和俯视图平面.

        这些目标当前主要用于:
        - 调试显示
        - `stone` 与分支左右关系判断
        暂时不会像 cost map 那样直接侵蚀主路径。
        """
        planning_items = []
        seg_boxes = []

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
            seg_boxes.append(seg_box)

        if not seg_boxes:
            return planning_items

        bird_boxes = cv2.perspectiveTransform(
            np.array(seg_boxes, dtype=np.float32).reshape((-1, 1, 2)),
            self.M_seg
        ).reshape((-1, 4, 2))

        for item, bird_box in zip(planning_items, bird_boxes):
            bird_box = bird_box.astype(np.float32)
            bird_box[:, 0] = np.clip(bird_box[:, 0], 0, w_seg - 1)
            bird_box[:, 1] = np.clip(bird_box[:, 1], 0, h_seg - 1)

            bird_center = np.mean(bird_box, axis=0)
            distances = np.linalg.norm(bird_box - bird_center, axis=1)
            bird_radius = float(np.max(distances)) if len(distances) > 0 else 0.0

            item["bird_box"] = bird_box
            item["bird_center"] = bird_center.astype(np.float32)
            item["bird_radius"] = bird_radius

        return planning_items

    def _compute_weighted_steer_signal(self, path_points, img_w, img_h):
        """按“路径点到底部中点连线斜率 * 行号”的方式聚合单一控制量."""
        if path_points is None or len(path_points) == 0:
            return 0.0

        pts = np.array(path_points, dtype=np.float32).reshape((-1, 2))
        # 以图像几何中线为 0，保证垂直中线上的路径控制量严格为 0。
        bottom_mid_x = float(img_w) / 2.0
        bottom_y = float(img_h) - 1.0
        min_dy = float(config.STEER_SIGNAL_MIN_DY)

        dy = np.maximum(bottom_y - pts[:, 1], min_dy)
        slopes = (pts[:, 0] - bottom_mid_x) / dy
        row_weights = np.clip(pts[:, 1], 0.0, bottom_y)
        return float(np.sum(slopes * row_weights))

    def _draw_planning_points(self, canvas, planning_items):
        """在俯视图上绘制规划相关目标的质心点，必要时附加半径圈."""
        for item in planning_items:
            bird_center = item.get("bird_center")
            bird_radius = item.get("bird_radius")
            if bird_center is None or bird_radius is None:
                continue

            class_name = item.get("class_name", "")
            style = self.planning_marker_styles.get(
                class_name,
                {"color": (0, 255, 255), "label": class_name},
            )
            color = tuple(int(v) for v in style.get("color", (0, 255, 255)))
            label = style.get("label", class_name)

            cx = int(round(float(bird_center[0])))
            cy = int(round(float(bird_center[1])))
            radius = max(config.SEG_DEBUG_PLANNING_MIN_RADIUS, int(round(float(bird_radius))))

            cv2.circle(canvas, (cx, cy), config.SEG_DEBUG_PLANNING_DOT_RADIUS, color, -1)

            text = label
            if class_name in self.planning_circle_class_names:
                cv2.circle(canvas, (cx, cy), radius, color, 1)
                text = f"{label} r={radius}"

            text_x = int(
                np.clip(
                    cx + config.SEG_DEBUG_PLANNING_TEXT_OFFSET_X,
                    0,
                    max(0, canvas.shape[1] - 1),
                )
            )
            text_y = int(
                np.clip(
                    cy + config.SEG_DEBUG_PLANNING_TEXT_OFFSET_Y,
                    config.SEG_DEBUG_PLANNING_TEXT_MIN_Y,
                    max(config.SEG_DEBUG_PLANNING_TEXT_MIN_Y, canvas.shape[0] - 4),
                )
            )
            cv2.putText(
                canvas,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.SEG_DEBUG_PLANNING_TEXT_FONT_SCALE,
                color,
                config.SEG_DEBUG_PLANNING_TEXT_THICKNESS,
                cv2.LINE_AA
            )

    def run(self, blob_rgb_320, current_yolo_boxes, turn_intent, fps_stats):
        """执行一次完整的分割和路径规划.

        输入:
        - blob_rgb_320: 分割线程当前拿到的最新 320x320 RGB 图
        - current_yolo_boxes: 当前最新一帧检测框，坐标在 TARGET_RES
        - turn_intent: OCR 给出的 LEFT / RIGHT 分叉意图

        输出:
        - steer_signal: 单一转向控制量，来自路径点加权斜率和
        - ai_view: 320 空间调试图，主线程会再放大回 TARGET_RES
        """
        t_total_start = time.perf_counter()
        w_seg, h_seg = config.SEG_SIZE   # 320, 320
        
        # ai_view 是最终调试图的底板，保持在 320 空间，后续再由主线程放大。
        blob = blob_rgb_320
        ai_view = cv2.cvtColor(blob, cv2.COLOR_RGB2BGR)

        # 分割模型推理
        t_infer_start = time.perf_counter()
        outputs = self.rknn.inference(inputs=[np.expand_dims(blob, axis=0)])
        out = outputs[0]
        t_infer_end = time.perf_counter()
        
        if len(out.shape) == 4 and out.shape[1] > 1:
            mask = (out[0][1] > out[0][0]).astype(np.uint8)
        else:
            mask = out.squeeze().astype(np.uint8)

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
        pts_final_bird = None
        search_mask = self._prepare_search_mask(mask)
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
        boundary_left_bird = None
        boundary_right_bird = None
        candidate_left_orig = None
        candidate_right_orig = None
        y_fork_active = False
        fork_active = False
        best_path = None
        best_nodes = None
        stone_branch_side = 0

        if y_fork_info.get("active"):
            left_mask, right_mask = self._split_mask_by_fork(search_mask, y_fork_info["fork_point"])
            left_paths, left_pair_max, left_support_rows = self._search_active_paths(
                left_mask,
                self._extract_edge_mask(left_mask),
            )
            right_paths, right_pair_max, right_support_rows = self._search_active_paths(
                right_mask,
                self._extract_edge_mask(right_mask),
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
                stone_branch_side = self._estimate_stone_branch_side(planning_items, candidate_pool)
                preferred_turn = self._resolve_preferred_turn(stone_branch_side, turn_intent)
                best_candidate = right_best if preferred_turn == 1 else left_best
                best_path = best_candidate["path"]
                best_nodes = best_candidate["nodes"]

        if not y_fork_active:
            active_paths, branch_pair_count_max, branch_support_rows = self._search_active_paths(
                search_mask,
                self._extract_edge_mask(search_mask),
            )
            valid_candidates = self._score_candidate_paths(active_paths)
            fork_left = None
            fork_right = None

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

                fork_left, fork_right, fork_active = self._select_fork_representatives(
                    valid_candidates,
                    branch_support_rows,
                )
                if fork_active:
                    candidate_pool = [fork_left, fork_right]
                    stone_branch_side = self._estimate_stone_branch_side(planning_items, candidate_pool)
                else:
                    stone_branch_side = self._estimate_stone_branch_side(planning_items, valid_candidates)
                    max_score = max(c["score"] for c in valid_candidates)
                    candidate_pool = [
                        c for c in valid_candidates
                        if c["score"] >= max_score - float(config.SEG_PATH_TOP_TIER_SCORE_GAP)
                    ]

                preferred_turn = self._resolve_preferred_turn(stone_branch_side, turn_intent)
                if preferred_turn == 1:
                    best_candidate = max(candidate_pool, key=lambda c: c["avg_x"])
                else:
                    best_candidate = min(candidate_pool, key=lambda c: c["avg_x"])

                best_path = best_candidate["path"]
                best_nodes = best_candidate["nodes"]

        if best_path is not None:
                node_x = best_path[:, 0]
                node_y = best_path[:, 1]
                left_boundary_pts = np.array(
                    [[node["left_x"], node["pt"][1]] for node in best_nodes],
                    dtype=np.float32,
                )
                right_boundary_pts = np.array(
                    [[node["right_x"], node["pt"][1]] for node in best_nodes],
                    dtype=np.float32,
                )
                boundary_left_orig = left_boundary_pts.reshape((-1, 1, 2))
                boundary_right_orig = right_boundary_pts.reshape((-1, 1, 2))

                if len(np.unique(node_y)) > 2:
                    current_coeffs = np.polyfit(node_y, node_x, 2)
                else:
                    current_coeffs = np.polyfit(node_y, node_x, 1)
                    current_coeffs = np.insert(current_coeffs, 0, 0)

                # 4. 时域一阶低通滤波 (EMA)，赋予路径物理连贯惯性，消除分叉口反复横跳
                if self.last_poly_coeffs is not None and len(self.last_poly_coeffs) == len(current_coeffs):
                    poly_coeffs = self.ema_alpha * self.last_poly_coeffs + (1.0 - self.ema_alpha) * current_coeffs
                else:
                    poly_coeffs = current_coeffs

                self.last_poly_coeffs = poly_coeffs

                dense_y = np.linspace(node_y[0], node_y[-1], num=int(config.SEG_PATH_DENSE_SAMPLES))
                dense_x = np.polyval(poly_coeffs, dense_y)
                dense_x = np.clip(dense_x, 0, w_seg - 1)
                dense_y = np.clip(dense_y, 0, h_seg - 1)

                path_points_orig = np.vstack((dense_x, dense_y)).astype(np.float32).T
                steer_signal = self._compute_weighted_steer_signal(path_points_orig, w_seg, h_seg)

                pts_final_orig = path_points_orig.reshape((-1, 1, 2))
                pts_final_bird = cv2.perspectiveTransform(pts_final_orig, self.M_seg)
                boundary_left_bird = cv2.perspectiveTransform(boundary_left_orig, self.M_seg)
                boundary_right_bird = cv2.perspectiveTransform(boundary_right_orig, self.M_seg)
        else:
            self.last_poly_coeffs = None
        self.last_branch_stats = {
            "branch_pair_count_max": int(branch_pair_count_max),
            "branch_support_rows": int(branch_support_rows),
            "fork_active": bool(fork_active),
            "y_fork_active": bool(y_fork_active),
            "y_fork_bottom_width": float(y_fork_info.get("bottom_width", 0.0)),
        }
        self._store_main_overlay(
            pts_final_orig,
            boundary_left_orig,
            boundary_right_orig,
            w_seg,
            h_seg,
            candidate_left_pts=candidate_left_orig,
            candidate_right_pts=candidate_right_orig,
            fork_point=y_fork_info.get("fork_point") if y_fork_active else None,
        )
        t_fit_end = time.perf_counter()

        # -------------------------------------------------------------------
        # 5. 调试渲染
        # -------------------------------------------------------------------
        t_render_start = time.perf_counter()
        colored_roi = np.zeros_like(ai_view)
        colored_roi[mask == 1] = [0, 255, 0]
        ai_view = cv2.addWeighted(ai_view, 1.0 - config.MASK_ALPHA, colored_roi, config.MASK_ALPHA, 0)
        # 路径直接画在分割调试平面里，主线程只负责整体缩放和其它信息叠加。
        ai_view = self.draw_path_overlay(ai_view)

        bird_eye_mask = cv2.warpPerspective(mask, self.M_seg, (w_seg, h_seg), flags=cv2.INTER_NEAREST)
        pip_img = cv2.cvtColor(np.where(bird_eye_mask == 1, 255, 0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        if pts_final_bird is not None:
            cv2.polylines(
                pip_img,
                [pts_final_bird.astype(np.int32)],
                False,
                config.SEG_DEBUG_BIRD_PATH_COLOR,
                config.SEG_DEBUG_BIRD_PATH_THICKNESS,
            )
        if boundary_left_bird is not None:
            cv2.polylines(
                pip_img,
                [boundary_left_bird.astype(np.int32)],
                False,
                config.SEG_DEBUG_BIRD_LEFT_BOUNDARY_COLOR,
                config.SEG_DEBUG_BIRD_BOUNDARY_THICKNESS,
            )
        if boundary_right_bird is not None:
            cv2.polylines(
                pip_img,
                [boundary_right_bird.astype(np.int32)],
                False,
                config.SEG_DEBUG_BIRD_RIGHT_BOUNDARY_COLOR,
                config.SEG_DEBUG_BIRD_BOUNDARY_THICKNESS,
            )

        self._draw_planning_points(pip_img, planning_items)
            
        pip_h, pip_w = h_seg // config.SEG_DEBUG_PIP_DIVISOR, w_seg // config.SEG_DEBUG_PIP_DIVISOR
        ai_view[0:pip_h, w_seg - pip_w:w_seg] = cv2.resize(pip_img, (pip_w, pip_h))
        cv2.rectangle(
            ai_view,
            (w_seg - pip_w, 0),
            (w_seg, pip_h),
            config.SEG_DEBUG_PIP_BORDER_COLOR,
            config.SEG_DEBUG_PIP_BORDER_THICKNESS,
        )

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
                f"Y:{int(self.last_branch_stats.get('y_fork_active', False))}"
            ),
            config.SEG_DEBUG_TEXT_POS_BRANCH,
            1,
            config.SEG_DEBUG_TEXT_FONT_SCALE,
            config.SEG_DEBUG_TEXT_COLOR_BRANCH,
            config.SEG_DEBUG_TEXT_THICKNESS,
        )
        t_render_end = time.perf_counter()
        self._profile_add(
            infer_s=t_infer_end - t_infer_start,
            preprocess_s=t_preprocess_end - t_preprocess_start,
            search_s=t_search_end - t_search_start,
            fit_s=t_fit_end - t_fit_start,
            render_s=t_render_end - t_render_start,
            total_s=t_render_end - t_total_start,
        )

        return steer_signal, ai_view

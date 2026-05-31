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
        self.ema_alpha = 0.6  # 历史权重：越接近 1.0 越稳定，越接近 0.0 响应越快

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

            if abs(right_x - left_x) < 12:
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
        min_dy = float(getattr(config, "STEER_SIGNAL_MIN_DY", 8.0))

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
            radius = max(4, int(round(float(bird_radius))))

            cv2.circle(canvas, (cx, cy), 4, color, -1)

            text = label
            if class_name in self.planning_circle_class_names:
                cv2.circle(canvas, (cx, cy), radius, color, 1)
                text = f"{label} r={radius}"

            text_x = int(np.clip(cx + 6, 0, max(0, canvas.shape[1] - 1)))
            text_y = int(np.clip(cy - 6, 12, max(12, canvas.shape[0] - 4)))
            cv2.putText(
                canvas,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
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
        w_seg, h_seg = config.SEG_SIZE   # 320, 320
        
        # ai_view 是最终调试图的底板，保持在 320 空间，后续再由主线程放大。
        blob = blob_rgb_320
        ai_view = cv2.cvtColor(blob, cv2.COLOR_RGB2BGR)

        # 分割模型推理
        outputs = self.rknn.inference(inputs=[np.expand_dims(blob, axis=0)])
        out = outputs[0]
        
        if len(out.shape) == 4 and out.shape[1] > 1:
            mask = (out[0][1] > out[0][0]).astype(np.uint8)
        else:
            mask = out.squeeze().astype(np.uint8)

        # 投影 YOLO 框到分割面
        planning_items = self._project_planning_objects(current_yolo_boxes, w_seg, h_seg)

        # -------------------------------------------------------------------
        # 3. 在 mask 空间中做自底向上的路径搜索（重构：局部中线记录版）
        # -------------------------------------------------------------------
        steer_signal = 0.0
        pts_final_orig = None
        pts_final_bird = None
        
        STEP_Y = 10        # 加密采样点，提供更好的局部边界感知
        GAP_THRESH = 15    
        active_paths = []  
        
        bottom_y = h_seg - 5
        bottom_slice = mask[h_seg-15:h_seg, :]
        white_xs = np.where(bottom_slice == 1)[1]
        
        if len(white_xs) > 8:
            start_x = int(np.median(white_xs))
            # 路径节点存储结构：包含像素点坐标 pt 和当前分支的局部几何中心 local_center
            active_paths.append([{"pt": (start_x, bottom_y), "local_center": start_x}])
            
            curr_y = bottom_y
            while curr_y >= int(h_seg * 0.1): 
                slice_mask = mask[max(0, curr_y - STEP_Y//2) : min(h_seg, curr_y + STEP_Y//2), :]
                xs = np.where(slice_mask == 1)[1]
                
                if len(xs) < 4:
                    curr_y -= STEP_Y
                    continue 
                    
                # 区分不同的局部赛道分支
                splits = np.split(xs, np.where(np.diff(xs) > GAP_THRESH)[0] + 1)
                valid_branches = []
                for s in splits:
                    if len(s) > 2:
                        branch_center = int(np.mean(s))
                        valid_branches.append(branch_center)
                
                if not valid_branches:
                    curr_y -= STEP_Y
                    continue
                    
                if not active_paths:
                    for bx in valid_branches:
                        active_paths.append([{"pt": (bx, curr_y), "local_center": bx}])
                else:
                    new_paths = []
                    for path in active_paths:
                        last_x = path[-1]["pt"][0]
                        connected = False
                        for bx in valid_branches:
                            # 关联层级：若空间距离足够近，归入同一条分支路径
                            if abs(bx - last_x) < 50:
                                new_paths.append(path + [{"pt": (bx, curr_y), "local_center": bx}])
                                connected = True
                        if not connected:
                            new_paths.append(path)
                            
                    active_paths = new_paths
                    if len(active_paths) > 15:
                        active_paths.sort(key=lambda p: len(p), reverse=True)
                        active_paths = active_paths[:15]
                        
                curr_y -= STEP_Y
                
        # -------------------------------------------------------------------
        # 4. 对候选路径打分，引入局部中心偏差惩罚 + EMA 滤波选择最终路径
        # -------------------------------------------------------------------
        right_branch_count = 0
        if active_paths:
            valid_candidates = []
            for path in active_paths:
                if len(path) < 3: 
                    continue
                
                path_arr = np.array([node["pt"] for node in path])
                px = path_arr[:, 0]
                
                # 1. 长度分
                length_score = len(path) * 50.0
                
                # 2. 平滑度分
                dx = np.diff(px)
                smooth_score = -np.std(dx) * 20.0
                
                # 3. 分支局部中心偏离惩罚（精准抑制弯道强行切直线走捷径的行为）
                center_penalty = 0.0
                for node in path:
                    x_curr = node["pt"][0]
                    x_local_center = node["local_center"]
                    center_penalty += abs(x_curr - x_local_center) * 3.5
                
                base_score = length_score + smooth_score - center_penalty
                avg_x = np.mean(px)
                
                valid_candidates.append({
                    'path': path_arr,
                    'score': base_score,
                    'avg_x': avg_x,
                })
                
            best_path = None
            stone_branch_side = 0
            if valid_candidates:
                max_score = max(c['score'] for c in valid_candidates)
                top_tier_paths = [c for c in valid_candidates if c['score'] >= max_score - 150]
                if stone_branch_side == -1:
                    preferred_turn = 1
                elif stone_branch_side == 1:
                    preferred_turn = -1
                else:
                    preferred_turn = turn_intent if turn_intent in (-1, 1) else -1

                if preferred_turn == 1:
                    best_candidate = max(top_tier_paths, key=lambda c: c['avg_x'])
                else:
                    best_candidate = min(top_tier_paths, key=lambda c: c['avg_x'])
                    
                best_path = best_candidate['path']
                    
            if best_path is not None:
                node_x = best_path[:, 0]
                node_y = best_path[:, 1]
                
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
                    
                dense_y = np.linspace(node_y[0], node_y[-1], num=30)
                dense_x = np.polyval(poly_coeffs, dense_y)
                dense_x = np.clip(dense_x, 0, w_seg - 1)
                dense_y = np.clip(dense_y, 0, h_seg - 1)

                path_points_orig = np.vstack((dense_x, dense_y)).astype(np.float32).T
                steer_signal = self._compute_weighted_steer_signal(path_points_orig, w_seg, h_seg)

                pts_final_orig = path_points_orig.reshape((-1, 1, 2))
                pts_final_bird = cv2.perspectiveTransform(pts_final_orig, self.M_seg)
            else:
                self.last_poly_coeffs = None

        # -------------------------------------------------------------------
        # 5. 调试渲染
        # -------------------------------------------------------------------
        colored_roi = np.zeros_like(ai_view)
        colored_roi[mask == 1] = [0, 255, 0] 
        ai_view = cv2.addWeighted(ai_view, 0.6, colored_roi, 0.4, 0)
        
        if pts_final_orig is not None:
            cv2.polylines(ai_view, [pts_final_orig.astype(np.int32)], False, (255, 0, 255), 2)
        bottom_mid_pt = (int(round(w_seg / 2.0)), h_seg - 1)
        cv2.circle(ai_view, bottom_mid_pt, 4, (255, 255, 0), -1)

        bird_eye_mask = cv2.warpPerspective(mask, self.M_seg, (w_seg, h_seg), flags=cv2.INTER_NEAREST)
        pip_img = cv2.cvtColor(np.where(bird_eye_mask == 1, 255, 0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        if pts_final_bird is not None:
            cv2.polylines(pip_img, [pts_final_bird.astype(np.int32)], False, (0, 0, 255), 2)

        self._draw_planning_points(pip_img, planning_items)
            
        pip_h, pip_w = h_seg // 3, w_seg // 3  
        ai_view[0:pip_h, w_seg-pip_w:w_seg] = cv2.resize(pip_img, (pip_w, pip_h))
        cv2.rectangle(ai_view, (w_seg-pip_w, 0), (w_seg, pip_h), (255, 255, 255), 1)

        servo_pwm = int(
            getattr(config, 'SERVO_CENTER', 750)
            - steer_signal * getattr(config, 'STEER_SIGNAL_PWM_GAIN', 0.03)
        )
        servo_pwm = int(max(getattr(config, 'SERVO_MIN', 590), min(getattr(config, 'SERVO_MAX', 910), servo_pwm)))
        cv2.putText(ai_view, f"Seg FPS:{fps_stats.get('seg_fps', 0):.1f} YOLO:{fps_stats.get('yolo_fps', 0):.1f}", (5, 18), 1, 0.8, (0, 255, 0), 1)
        cv2.putText(ai_view, f"Ctrl:{steer_signal:.1f} PWM:{servo_pwm}", (5, 36), 1, 0.8, (0, 255, 255), 1)
        stone_side_text = "UNK"
        if 'stone_branch_side' in locals():
            if stone_branch_side == -1:
                stone_side_text = "LEFT"
            elif stone_branch_side == 1:
                stone_side_text = "RIGHT"
            else:
                stone_side_text = "NONE"
        cv2.putText(ai_view, f"Stone:{stone_side_text}", (5, 54), 1, 0.8, (0, 200, 255), 1)

        return steer_signal, ai_view

# modules/segmentor.py
"""赛道分割与路径规划模块.

这个模块负责:
1. 调用分割模型得到二值赛道 mask
2. 在 mask 空间中搜索一条可跟踪路径
3. 将路径投影到鸟瞰图坐标系，计算横向误差和预瞄斜率
4. 返回用于控制的 err_x / l_k，以及一张调试渲染图

当前版本特点:
- 输入固定为 320x320 RGB
- 路径搜索直接在分割 mask 上进行
- 使用 turn_intent 对分叉场景做简单偏向控制
"""

import cv2
import numpy as np
from rknnlite.api import RKNNLite
import config

class RoadSegmentor:
    def __init__(self, core_id):
        """初始化分割模型与 320 空间的逆透视矩阵."""
        self.rknn = RKNNLite()
        print(f"--> [Segmentor] 正在初始化 NPU Core {core_id}...")
        
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
        """将原图中的规划相关检测框映射到分割平面和俯视图平面."""
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

        参数:
            blob_rgb_320: 来自主线程的 320x320 RGB 图像
            current_yolo_boxes: 当前最新检测框，基于原图框映射规划相关元素
            turn_intent: OCR 识别出的转向意图，-1 倾向左，1 倾向右
            fps_stats: 用于渲染调试信息
        """
        w_out, h_out = config.TARGET_RES # 960, 720
        w_seg, h_seg = config.SEG_SIZE   # 320, 320
        
        # ai_view 是最终调试图的底板，保持在 320 空间，后续再由主线程放大。
        blob = blob_rgb_320
        ai_view = cv2.cvtColor(blob, cv2.COLOR_RGB2BGR)

        # 分割模型直接输出赛道类别概率或二值图，这里统一整理成 0/1 mask。
        outputs = self.rknn.inference(inputs=[np.expand_dims(blob, axis=0)])
        out = outputs[0]
        
        if len(out.shape) == 4 and out.shape[1] > 1:
            mask = (out[0][1] > out[0][0]).astype(np.uint8)
        else:
            mask = out.squeeze().astype(np.uint8)

        # 检测结果始终来自原图，只有真正需要参与路径规划的元素才投影到俯视图。
        planning_items = self._project_planning_objects(current_yolo_boxes, w_seg, h_seg)

        # -------------------------------------------------------------------
        # 3. 在 mask 空间中做自底向上的路径搜索
        # -------------------------------------------------------------------
        err_x, l_k = 0.0, 0.0
        pts_final_orig = None
        pts_final_bird = None
        
        STEP_Y = 12        
        GAP_THRESH = 15    
        active_paths = []  
        
        # 从靠近车辆的底部区域选取起点，优先保证路径对当前控制有效。
        bottom_y = h_seg - 5
        bottom_slice = mask[h_seg-15:h_seg, :]
        white_xs = np.where(bottom_slice == 1)[1]
        
        if len(white_xs) > 8:
            start_x = int(np.median(white_xs))
            active_paths.append([(start_x, bottom_y)])
            
            curr_y = bottom_y
            while curr_y >= int(h_seg * 0.1): 
                slice_mask = mask[max(0, curr_y - STEP_Y//2) : min(h_seg, curr_y + STEP_Y//2), :]
                xs = np.where(slice_mask == 1)[1]
                
                if len(xs) < 4:
                    curr_y -= STEP_Y
                    continue 
                    
                splits = np.split(xs, np.where(np.diff(xs) > GAP_THRESH)[0] + 1)
                branch_centers = [int(np.mean(s)) for s in splits if len(s) > 2]
                
                if not branch_centers:
                    curr_y -= STEP_Y
                    continue
                    
                if not active_paths:
                    for bx in branch_centers:
                        active_paths.append([(bx, curr_y)])
                else:
                    new_paths = []
                    for path in active_paths:
                        last_x = path[-1][0]
                        connected = False
                        for bx in branch_centers:
                            # 如果当前层中心点和上一层足够接近，认为它们属于同一条路。
                            if abs(bx - last_x) < 50:
                                new_paths.append(path + [(bx, curr_y)])
                                connected = True
                        if not connected:
                            new_paths.append(path)
                            
                    active_paths = new_paths
                    if len(active_paths) > 15:
                        active_paths.sort(key=lambda p: len(p), reverse=True)
                        active_paths = active_paths[:15]
                        
                curr_y -= STEP_Y
                
        # -------------------------------------------------------------------
        # 4. 对候选路径打分，并根据 turn_intent 选择最终路径
        # -------------------------------------------------------------------
        if active_paths:
            valid_candidates = []
            for path in active_paths:
                if len(path) < 3: continue
                path_arr = np.array(path)
                px = path_arr[:, 0]
                
                # 评分思路:
                # - 越长越好
                # - 越平滑越好
                # - 横向漂移越小越好
                length_score = len(path) * 50.0
                dx = np.diff(px)
                smooth_score = -np.std(dx) * 10.0
                yaw_score = -abs(px[-1] - px[0]) * 0.5
                
                base_score = length_score + smooth_score + yaw_score
                avg_x = np.mean(px)
                
                valid_candidates.append({'path': path_arr, 'score': base_score, 'avg_x': avg_x})
                
            best_path = None
            if valid_candidates:
                max_score = max(c['score'] for c in valid_candidates)
                top_tier_paths = [c for c in valid_candidates if c['score'] >= max_score - 150]
                
                if turn_intent == 1:
                    best_candidate = max(top_tier_paths, key=lambda c: c['avg_x']) 
                else:
                    best_candidate = min(top_tier_paths, key=lambda c: c['avg_x']) 
                    
                best_path = best_candidate['path']
                    
            if best_path is not None:
                node_x = best_path[:, 0]
                node_y = best_path[:, 1]
                
                # 用多项式拟合把离散路径点变成连续曲线，便于控制和可视化。
                if len(np.unique(node_y)) > 2:
                    poly_coeffs = np.polyfit(node_y, node_x, 2)
                else:
                    poly_coeffs = np.polyfit(node_y, node_x, 1)
                    poly_coeffs = np.insert(poly_coeffs, 0, 0)
                    
                dense_y = np.linspace(node_y[0], node_y[-1], num=30)
                dense_x = np.polyval(poly_coeffs, dense_y)
                
                # OpenCV polyline / perspectiveTransform 都要求这种点集结构。
                pts_final_orig = np.vstack((dense_x, dense_y)).astype(np.float32).T.reshape((-1, 1, 2))
                
                # 把前视图路径整体投影到鸟瞰图，便于计算更稳定的控制量。
                pts_final_bird = cv2.perspectiveTransform(pts_final_orig, self.M_seg)
                
                # 真正用于控制的是鸟瞰图里的横向偏差和预瞄斜率。
                bird_nodes = pts_final_bird.squeeze()
                if bird_nodes.ndim == 2:
                    bx = bird_nodes[:, 0]
                    by = bird_nodes[:, 1]
                    
                    car_center_x_bird = w_seg / 2.0
                    # 最近端节点代表“车辆当前应跟踪的位置”，换算成厘米级误差。
                    err_x = (bx[0] - car_center_x_bird) * getattr(config, 'CM_PER_PIXEL_X', 0.109649) * (w_out / w_seg)
                    
                    # 预瞄斜率用于抑制大角度转向时的过冲。
                    lookahead_idx = min(12, len(by) - 1)
                    dy_l = by[0] - by[lookahead_idx]
                    dx_l = bx[0] - bx[lookahead_idx]
                    l_k = dx_l / dy_l if dy_l != 0 else 0.0

        # -------------------------------------------------------------------
        # 5. 调试渲染
        # -------------------------------------------------------------------
        # 主画面叠加绿色分割区域和粉色规划线。
        colored_roi = np.zeros_like(ai_view)
        colored_roi[mask == 1] = [0, 255, 0] 
        ai_view = cv2.addWeighted(ai_view, 0.6, colored_roi, 0.4, 0)
        
        if pts_final_orig is not None:
            cv2.polylines(ai_view, [pts_final_orig.astype(np.int32)], False, (255, 0, 255), 2)

        # 右上角小窗显示 BEV。这里只做控制和后续改写路径的调试，不给模型回灌。
        bird_eye_mask = cv2.warpPerspective(mask, self.M_seg, (w_seg, h_seg), flags=cv2.INTER_NEAREST)
        pip_img = cv2.cvtColor(np.where(bird_eye_mask == 1, 255, 0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        if pts_final_bird is not None:
            cv2.polylines(pip_img, [pts_final_bird.astype(np.int32)], False, (0, 0, 255), 2)

        self._draw_planning_points(pip_img, planning_items)
            
        pip_h, pip_w = h_seg // 3, w_seg // 3  
        ai_view[0:pip_h, w_seg-pip_w:w_seg] = cv2.resize(pip_img, (pip_w, pip_h))
        cv2.rectangle(ai_view, (w_seg-pip_w, 0), (w_seg, pip_h), (255, 255, 255), 1)

        # 这里单独计算一个“理论舵机值”用于页面调试显示，
        # 真正下发给下位机的值在 serial_control_thread 中统一生成。
        servo_pwm = int(getattr(config, 'SERVO_CENTER', 750) + (err_x * getattr(config, 'KP', 0.16)) - (l_k * getattr(config, 'KD', 160.0)))
        cv2.putText(ai_view, f"Seg FPS:{fps_stats.get('seg_fps', 0):.1f} YOLO:{fps_stats.get('yolo_fps', 0):.1f}", (5, 18), 1, 0.8, (0, 255, 0), 1)
        cv2.putText(ai_view, f"Err:{err_x:.1f}cm PWM:{servo_pwm}", (5, 36), 1, 0.8, (0, 255, 255), 1)

        return err_x, l_k, ai_view

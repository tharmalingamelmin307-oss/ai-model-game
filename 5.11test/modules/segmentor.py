# modules/segmentor.py
import cv2
import numpy as np
from rknnlite.api import RKNNLite
import config

class RoadSegmentor:
    def __init__(self, core_id):
        self.rknn = RKNNLite()
        print(f"--> [Segmentor] 正在初始化 NPU Core {core_id}...")
        
        if self.rknn.load_rknn(config.SEG_MODEL) != 0 or self.rknn.init_runtime(core_mask=core_id) != 0:
            raise RuntimeError("Seg 模型加载或初始化失败")
            
        w_seg, h_seg = config.SEG_SIZE # 320, 320
        src_seg = np.float32([[x * w_seg, y * h_seg] for x, y in config.SRC_PTS])
        dst_seg = np.float32([[x * w_seg, y * h_seg] for x, y in config.DST_PTS])
        # 建立 320 空间的透视变换矩阵
        self.M_seg = cv2.getPerspectiveTransform(src_seg, dst_seg)

    def run(self, blob_rgb_320, current_yolo_boxes, turn_intent, fps_stats):
        w_out, h_out = config.TARGET_RES # 960, 720
        w_seg, h_seg = config.SEG_SIZE   # 320, 320
        
        # 1. 直接承接上游 320 翻转好的小图转 BGR 作为底板
        blob = blob_rgb_320
        ai_view = cv2.cvtColor(blob, cv2.COLOR_RGB2BGR)

        # 2. NPU 前向推理拿到原生视角 Mask
        outputs = self.rknn.inference(inputs=[np.expand_dims(blob, axis=0)])
        out = outputs[0]
        
        if len(out.shape) == 4 and out.shape[1] > 1:
            mask = (out[0][1] > out[0][0]).astype(np.uint8)
        else:
            mask = out.squeeze().astype(np.uint8)

        # ==================== 3. 🌟 核心改变：直接在原图 Mask 空间寻迹 ====================
        err_x, l_k = 0.0, 0.0
        pts_final_orig = None
        pts_final_bird = None
        
        STEP_Y = 12        
        GAP_THRESH = 15    
        active_paths = []  
        
        # 底部近端车头切片
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
                            # 原图空间连续性极强，50 像素内的分叉枝干轻松锁死
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
                
        # 4️⃣ 路径评估与多分支打分过滤
        if active_paths:
            valid_candidates = []
            for path in active_paths:
                if len(path) < 3: continue
                path_arr = np.array(path)
                px = path_arr[:, 0]
                
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
                
                # 在原图空间进行密集二次曲线二次拟合
                if len(np.unique(node_y)) > 2:
                    poly_coeffs = np.polyfit(node_y, node_x, 2)
                else:
                    poly_coeffs = np.polyfit(node_y, node_x, 1)
                    poly_coeffs = np.insert(poly_coeffs, 0, 0)
                    
                dense_y = np.linspace(node_y[0], node_y[-1], num=30)
                dense_x = np.polyval(poly_coeffs, dense_y)
                
                # 重新打包为标准 OpenCV 浮点坐标点集结构 (N, 1, 2)
                pts_final_orig = np.vstack((dense_x, dense_y)).astype(np.float32).T.reshape((-1, 1, 2))
                
                # 🌟【灵魂跃迁】通过透视变换，将原图规划线坐标整体映射投影到俯视图 (BEV) 空间
                pts_final_bird = cv2.perspectiveTransform(pts_final_orig, self.M_seg)
                
                # 🌟【精准控制】在纯净、无畸变的俯视图空间计算真实的物理偏差与控制斜率
                bird_nodes = pts_final_bird.squeeze()
                if bird_nodes.ndim == 2:
                    bx = bird_nodes[:, 0]
                    by = bird_nodes[:, 1]
                    
                    car_center_x_bird = w_seg / 2.0
                    # 俯视图近端车头处的绝对厘米级偏航误差
                    err_x = (bx[0] - car_center_x_bird) * getattr(config, 'CM_PER_PIXEL_X', 0.109649) * (w_out / w_seg)
                    
                    # 俯视图空间计算预瞄斜率 (前瞻 12 个节点)
                    lookahead_idx = min(12, len(by) - 1)
                    dy_l = by[0] - by[lookahead_idx]
                    dx_l = bx[0] - bx[lookahead_idx]
                    l_k = dx_l / dy_l if dy_l != 0 else 0.0

        # ==================== 5. 跨空间可视化渲染 ====================
        # 主画面：在原图上覆盖绿色分割蒙版，画上原图视角的粉色规划线
        colored_roi = np.zeros_like(ai_view)
        colored_roi[mask == 1] = [0, 255, 0] 
        ai_view = cv2.addWeighted(ai_view, 0.6, colored_roi, 0.4, 0)
        
        if pts_final_orig is not None:
            cv2.polylines(ai_view, [pts_final_orig.astype(np.int32)], False, (255, 0, 255), 2)

        # 右上角小窗（画中画 PIP）：用于直观调试看俯视图 (BEV) 下的红线状态
        bird_eye_mask = cv2.warpPerspective(mask, self.M_seg, (w_seg, h_seg), flags=cv2.INTER_NEAREST)
        pip_img = cv2.cvtColor(np.where(bird_eye_mask == 1, 255, 0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        if pts_final_bird is not None:
            # 渲染投射到俯视图上的红色诱导控制线
            cv2.polylines(pip_img, [pts_final_bird.astype(np.int32)], False, (0, 0, 255), 2)
            
        pip_h, pip_w = h_seg // 3, w_seg // 3  
        ai_view[0:pip_h, w_seg-pip_w:w_seg] = cv2.resize(pip_img, (pip_w, pip_h))
        cv2.rectangle(ai_view, (w_seg-pip_w, 0), (w_seg, pip_h), (255, 255, 255), 1)

        # 左上角微型原始 mask 小窗
        mini_mask_size = 100
        raw_mask_vis = np.zeros((w_seg, h_seg, 3), dtype=np.uint8)
        raw_mask_vis[mask == 1] = [0, 255, 0] 
        ai_view[0:mini_mask_size, 0:mini_mask_size] = cv2.resize(raw_mask_vis, (mini_mask_size, mini_mask_size), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(ai_view, (0, 0), (mini_mask_size, mini_mask_size), (0, 255, 255), 1)

        servo_pwm = int(getattr(config, 'SERVO_CENTER', 750) + (err_x * getattr(config, 'KP', 0.16)) - (l_k * getattr(config, 'KD', 160.0)))
        cv2.putText(ai_view, f"Seg FPS:{fps_stats.get('seg_fps', 0):.1f} YOLO:{fps_stats.get('yolo_fps', 0):.1f}", (5, mini_mask_size + 15), 1, 0.8, (0, 255, 0), 1)
        cv2.putText(ai_view, f"Err:{err_x:.1f}cm PWM:{servo_pwm}", (5, mini_mask_size + 30), 1, 0.8, (0, 255, 255), 1)

        return err_x, l_k, ai_view
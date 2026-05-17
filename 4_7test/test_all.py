import os
import cv2
import numpy as np
import time
import threading
import serial
import struct
from rknnlite.api import RKNNLite
from concurrent.futures import ThreadPoolExecutor

# ==============================================================================
# 模块 1: 配置信息
# ==============================================================================
class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    YOLO_MODEL = os.path.join(BASE_DIR, "models", "yolov8_1.rknn")
    # 更新为你的新模型路径
    SEG_MODEL = "/home/orangepi/Desktop/setupUI/dist/rhr_test/4_7test/models/ppliteseg_576x416_int8.rknn"
    
    STREAM_URL = "http://192.168.31.189:8080/ar_feed"
    TARGET_RES = (960, 720) 
    YOLO_SIZE = (640, 640)  
    SEG_SIZE = (576, 416)   # 新模型输入尺寸
    
    SERIAL_PORT = '/dev/ttyS2'
    BAUD_RATE = 115200
    ENABLE_SERIAL = True
    
    SERVO_CENTER = 750
    SERVO_MIN, SERVO_MAX = 590, 910
    MOTOR_STOP = 2000
    MOTOR_MAX_SPEED = 2350 
    
    CLASSES = ("taxi", "coin", "person")
    GAUSSIAN_SIGMA = 35.0   
    SAFETY_MARGIN = 25      
    SMOOTH_WINDOW = 5       
    KP = 0.16               
    KD = 160.0              

    # 逻辑调整：裁掉顶部 30%，保留下方 70%
    ROI_TOP_CUT_RATIO = 0.3  

# ==============================================================================
# 模块 2: 硬件与视频流 (保持不变)
# ==============================================================================
class ChassisController:
    def __init__(self):
        self.ser = None
        if Config.ENABLE_SERIAL:
            try:
                self.ser = serial.Serial(Config.SERIAL_PORT, Config.BAUD_RATE, timeout=0.1)
                print(f"✅ 串口已连接: {Config.SERIAL_PORT}")
            except Exception as e:
                print(f"⚠️ 串口连接失败: {e}")

    def send_velocity(self, v, w):
        if self.ser is None: return
        v_int = max(Config.MOTOR_STOP, min(Config.MOTOR_MAX_SPEED, int(v)))
        w_int = max(Config.SERVO_MIN, min(Config.SERVO_MAX, int(w)))
        packet = struct.pack('<BBhhBB', 0xAA, 0x55, v_int, w_int, 0x0D, 0x0A)
        self.ser.write(packet)

    def stop(self):
        self.send_velocity(Config.MOTOR_STOP, Config.SERVO_CENTER)
        if self.ser: self.ser.close()

class VideoStreamWidget:
    def __init__(self, src):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.status, self.frame = self.capture.read()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
            time.sleep(0.01)

    def read(self):
        return self.status, self.frame

# ==============================================================================
# 模块 3: 后处理逻辑
# ==============================================================================
def process_yolo(output, orig_shape):
    preds = output[0][0].transpose(1, 0)
    boxes, scores = preds[:, :4], preds[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    max_scores = scores[np.arange(len(scores)), class_ids]
    
    mask = max_scores > 0.1
    if not np.any(mask): return []
    
    boxes, class_ids, max_scores = boxes[mask], class_ids[mask], max_scores[mask]
    x, y = boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2
    cv_boxes = np.stack((x, y, boxes[:, 2], boxes[:, 3]), axis=-1).tolist()
    indices = cv2.dnn.NMSBoxes(cv_boxes, max_scores.tolist(), 0.3, 0.45)
    
    scale_x, scale_y = orig_shape[0] / Config.YOLO_SIZE[0], orig_shape[1] / Config.YOLO_SIZE[1]
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            bx, by, bw, bh = cv_boxes[i]
            results.append({
                'rect': [int(bx*scale_x), int(by*scale_y), int(bw*scale_x), int(bh*scale_y)], 
                'class_id': int(class_ids[i])
            })
    return results

def main():
    rknn_yolo, rknn_seg = RKNNLite(), RKNNLite()
    
    print("--> 初始化 NPU 双模型 (Core 0/1)...")
    if rknn_yolo.load_rknn(Config.YOLO_MODEL) != 0 or rknn_yolo.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0: return
    if rknn_seg.load_rknn(Config.SEG_MODEL) != 0 or rknn_seg.init_runtime(core_mask=RKNNLite.NPU_CORE_1) != 0: return

    chassis = ChassisController()
    stream = VideoStreamWidget(Config.STREAM_URL)
    
    # 计算分割模型的 ROI 切片起始位置
    roi_start_y_seg = int(Config.SEG_SIZE[1] * Config.ROI_TOP_CUT_RATIO)
    # 对应到显示图像 720p 上的起始位置
    roi_start_y_vis = int(Config.TARGET_RES[1] * Config.ROI_TOP_CUT_RATIO)
    
    print(f"🚀 模型切换完成：PPLiteSeg 576x416 | 策略：保留底部，裁掉顶部 {Config.ROI_TOP_CUT_RATIO*100}%")

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            while True:
                loop_start = time.time()
                ret, frame = stream.read()
                if not ret or frame is None: continue

                # 极速 Resize 为两个模型的输入尺寸
                vis_img = cv2.resize(frame, Config.TARGET_RES, interpolation=cv2.INTER_NEAREST)
                img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                
                img_yolo_input = cv2.resize(img_rgb, Config.YOLO_SIZE, interpolation=cv2.INTER_NEAREST)
                img_seg_input = cv2.resize(img_rgb, Config.SEG_SIZE, interpolation=cv2.INTER_NEAREST)

                # NPU 异步推理
                future_yolo = executor.submit(rknn_yolo.inference, inputs=[np.expand_dims(img_yolo_input, axis=0)])
                future_seg = executor.submit(rknn_seg.inference, inputs=[np.expand_dims(img_seg_input, axis=0)])
                
                yolo_out = future_yolo.result()
                seg_out = future_seg.result()
                
                yolo_res = process_yolo(yolo_out, Config.TARGET_RES)
                
                # 分割后处理：裁掉顶部，保留底部
                pred_mask = np.squeeze(seg_out[0])
                if len(pred_mask.shape) == 3: # 如果输出是 (C, H, W)
                    pred_mask_roi = np.argmax(pred_mask[:, roi_start_y_seg:, :], axis=0)
                else: # 如果输出已经是 (H, W)
                    pred_mask_roi = pred_mask[roi_start_y_seg:, :]
                
                # 提取白线点
                white_pts = np.column_stack(np.where(pred_mask_roi == 1))
                
                servo_pwm, motor_pwm = Config.SERVO_CENTER, Config.MOTOR_STOP

                if len(white_pts) > 50:
                    # 采样优化
                    white_pts = white_pts[::8] 
                    
                    # 坐标转换：从 (416-ROI) 转换回 (720-ROI)
                    scale_h = Config.TARGET_RES[1] / Config.SEG_SIZE[1]
                    scale_w = Config.TARGET_RES[0] / Config.SEG_SIZE[0]
                    
                    # 先还原到分割模型全尺寸坐标，再缩放到显示坐标
                    ys = (white_pts[:, 0] + roi_start_y_seg) * scale_h
                    xs = white_pts[:, 1] * scale_w
                    
                    max_y, min_y = np.max(ys), np.min(ys)
                    y_range = max_y - min_y

                    # 拟合与控制逻辑
                    line_bound = int(max_y - y_range * 0.7)
                    line_mask = ys >= line_bound
                    
                    if np.sum(line_mask) > 5:
                        line_k, line_b = np.polyfit(ys[line_mask], xs[line_mask], 1)
                    else:
                        line_k, line_b = 0, Config.TARGET_RES[0]//2
                    
                    poly_coeffs = np.polyfit(ys, xs, 2)
                    plot_y = np.linspace(max_y, min_y, num=40)
                    plot_x_line = line_k * plot_y + line_b
                    plot_x_curve = np.polyval(poly_coeffs, plot_y)

                    # 混合曲线逻辑
                    t_arr = (max_y - plot_y) / (y_range + 0.1)
                    alpha = t_arr ** 2
                    plot_x_final = (1 - alpha) * plot_x_line + alpha * plot_x_curve

                    # 避障融合（YOLO）
                    for obj in yolo_res:
                        ox, oy, ow, oh = obj['rect']
                        cx, cy = ox + ow/2.0, oy + oh/2.0
                        if not (min_y <= cy <= max_y): continue
                        idx = np.argmin(np.abs(plot_y - cy))
                        if obj['class_id'] == 1: # 引导标
                            plot_x_final += (cx - plot_x_final[idx]) * np.exp(-((plot_y - cy)**2) / (2 * Config.GAUSSIAN_SIGMA**2))
                        elif obj['class_id'] == 0: # 障碍物
                            for i in range(len(plot_y)):
                                if oy <= plot_y[i] <= oy + oh:
                                    if ox - Config.SAFETY_MARGIN < plot_x_final[i] < ox + ow + Config.SAFETY_MARGIN:
                                        plot_x_final[i] = (ox - Config.SAFETY_MARGIN) if plot_x_final[idx] < cx else (ox + ow + Config.SAFETY_MARGIN)

                    # 平滑
                    padded_x = np.pad(plot_x_final, (Config.SMOOTH_WINDOW//2, Config.SMOOTH_WINDOW//2), mode='edge')
                    plot_x_final = np.convolve(padded_x, np.ones(Config.SMOOTH_WINDOW)/Config.SMOOTH_WINDOW, mode='valid')

                    # 计算控制量
                    error_x = plot_x_final[0] - (Config.TARGET_RES[0] // 2)
                    servo_pwm = int(Config.SERVO_CENTER + (error_x * Config.KP) - (line_k * Config.KD))
                    motor_pwm = Config.MOTOR_MAX_SPEED - int(abs(line_k) * 120)

                    # 绘制引导线
                    pts_final = np.vstack((plot_x_final, plot_y)).astype(np.int32).T.reshape((-1, 1, 2))
                    cv2.polylines(vis_img, [pts_final], False, (0, 0, 255), 4) 

                chassis.send_velocity(motor_pwm, servo_pwm)
                
                # 绘制 YOLO 框
                for res in yolo_res:
                    rx, ry, rw, rh = res['rect']
                    cv2.rectangle(vis_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 0) if res['class_id'] == 1 else (0, 0, 255), 2)
                
                # 显示 FPS
                fps = 1.0 / (time.time() - loop_start + 0.001)
                cv2.putText(vis_img, f"FPS:{fps:.1f} PWM:{servo_pwm}", (20, 30), 1, 1.2, (0, 255, 0), 2)
                cv2.imshow("Extreme Turbo AI", vis_img)
                
                if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        chassis.stop()
        rknn_yolo.release()
        rknn_seg.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
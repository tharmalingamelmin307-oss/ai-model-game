import os
import cv2
import numpy as np
import time
import threading
import serial
import struct
from rknnlite.api import RKNNLite

# ==============================================================================
# 模块 1: 全局配置信息 (Configuration)
# ==============================================================================
class Config:
    # --- 模型与处理参数 ---
    # 已更新为指定的绝对路径
    MODEL_PATH = '/home/orangepi/Desktop/setupUI/dist/rhr_test/PPLiteSegv1_3_22/model/ppliteseg_int8.rknn' 
    TARGET_RES = (960, 720)
    TARGET_CLASS_ID = 1          # 目标类别ID(如：路线为1)
    
    # --- 视频源设置 ---
    # 可选: 'STREAM' (网络串流, 代码2模式) 或 'FILE' (本地视频, 代码1模式)
    SOURCE_TYPE = 'STREAM'       
    STREAM_URL = "http://192.168.31.189:8080/ar_feed"
    VIDEO_PATH = '/home/orangepi/Desktop/hr_test/show4_6/录屏 2026-03-24 .webm'
    FAST_FORWARD_10X = False     # 仅在 'FILE' 模式下生效，是否开启10倍速跳帧
    
    # --- 串口与控制参数 ---
    SERIAL_PORT = '/dev/ttyS2'
    BAUD_RATE = 115200
    ENABLE_SERIAL = True         # 是否开启串口发送
    
    # --- PID 控制器参数 (基于图像X轴偏差) ---
    KP = 0.12                    # 比例系数: 负责纠正当前偏差
    KD = 130.0                   # 微分系数: 负责预测趋势 (根据一阶斜率)
    
    # --- 动力参数限幅 ---
    SERVO_CENTER = 750
    SERVO_MIN, SERVO_MAX = 590, 910
    MOTOR_STOP = 2000
    MOTOR_MAX_SPEED = 2600
    SPEED_DROP_FACTOR = 120      # 曲率减速系数

# ==============================================================================
# 模块 2: 硬件外设控制 (Hardware Controller)
# ==============================================================================
class ChassisController:
    def __init__(self):
        self.ser = None
        if Config.ENABLE_SERIAL:
            try:
                self.ser = serial.Serial(Config.SERIAL_PORT, Config.BAUD_RATE, timeout=0.1)
                print(f"✅ 串口连接成功 ({Config.SERIAL_PORT})")
            except Exception as e:
                print(f"⚠️ 串口连接失败: {e}，将仅运行视觉处理。")
                self.ser = None

    def send_velocity(self, v, w):
        """发送电机速度 v 和 舵机角度 w"""
        if self.ser is None:
            return
            
        v_int, w_int = int(v), int(w)
        # 限幅保护
        w_int = max(Config.SERVO_MIN, min(Config.SERVO_MAX, w_int))
        v_int = max(Config.MOTOR_STOP, min(Config.MOTOR_MAX_SPEED, v_int))
        
        # 协议组包: 帧头 0xAA 0x55, 电机(2字节), 舵机(2字节), 帧尾 0x0D 0x0A
        data_packet = struct.pack('<BBhhBB', 0xAA, 0x55, v_int, w_int, 0x0D, 0x0A)
        self.ser.write(data_packet)

    def stop(self):
        self.send_velocity(Config.MOTOR_STOP, Config.SERVO_CENTER)
        if self.ser:
            self.ser.close()

# ==============================================================================
# 模块 3: 视频流读取 (Video Capture)
# ==============================================================================
class VideoStreamWidget:
    """低延迟的独立线程读取视频流"""
    def __init__(self, src):
        self.capture = cv2.VideoCapture(src)
        if Config.SOURCE_TYPE == 'STREAM':
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
            
        self.status, self.frame = self.capture.read()
        self.stopped = False
        
        # 仅针对网络流开启独立线程以降低延迟
        if Config.SOURCE_TYPE == 'STREAM':
            self.thread = threading.Thread(target=self.update, args=())
            self.thread.daemon = True
            self.thread.start()

    def update(self):
        while not self.stopped:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
            time.sleep(0.01)

    def read(self):
        # 如果是本地视频且开启10倍速，在这里执行跳帧操作
        if Config.SOURCE_TYPE == 'FILE' and Config.FAST_FORWARD_10X:
            for _ in range(9):
                self.capture.grab()
                
        if Config.SOURCE_TYPE == 'FILE':
            self.status, self.frame = self.capture.read()
            
        return self.status, self.frame

    def release(self):
        self.stopped = True
        self.capture.release()

# ==============================================================================
# 模块 4: 核心路径算法与控制计算 (Path Planning & Control Algorithm)
# ==============================================================================
def process_frame(pred_mask, vis_img):
    """
    处理模型输出的Mask，提取拟合曲线，计算控制偏差。
    返回: (处理后的图像, 舵机PWM, 电机PWM)
    """
    height, width = vis_img.shape[:2]
    center_x_ref = width // 2  # 中轴线 (通常是 480)
    
    # 默认安全控制值
    servo_pwm = Config.SERVO_CENTER
    motor_pwm = Config.MOTOR_STOP
    error_x = 0
    npu_time_ms = 0
    
    # 提取目标掩膜 (二值化)
    mask_255 = np.zeros_like(pred_mask, dtype=np.uint8)
    mask_255[pred_mask == Config.TARGET_CLASS_ID] = 255
    
    # 优化版紫色半透明覆盖
    is_target = (mask_255 == 255)
    vis_img[is_target] = vis_img[is_target] * 0.5 + np.array([255, 0, 255], dtype=np.uint8) * 0.5
    
    white_pts = np.column_stack(np.where(is_target)) 

    if len(white_pts) > 100:
        ys, xs = white_pts[:, 0], white_pts[:, 1]
        max_y, min_y = np.max(ys), np.min(ys)
        y_range = max_y - min_y
        
        if y_range >= 20: 
            # --- 阶段 1: 算法预处理与边界划分 ---
            top_bound = int(min_y + y_range * 0.02)
            bottom_bound = int(max_y - y_range * 0.02)
            line_bound = int(max_y - y_range * 0.75) # 直线拟合区域

            # 画出区域分割线
            cv2.line(vis_img, (0, top_bound), (width, top_bound), (255, 255, 255), 1) 
            cv2.line(vis_img, (0, bottom_bound), (width, bottom_bound), (255, 255, 255), 1) 
            cv2.line(vis_img, (0, line_bound), (width, line_bound), (0, 255, 255), 1) 

            # 提取上下质心
            top_mask = ys <= top_bound
            tip_y, tip_x = (int(np.mean(ys[top_mask])), int(np.mean(xs[top_mask]))) if np.sum(top_mask) > 0 else (ys[np.argmin(ys)], xs[np.argmin(ys)])

            bottom_mask = ys >= bottom_bound
            base_y, base_x = (int(np.mean(ys[bottom_mask])), int(np.mean(xs[bottom_mask]))) if np.sum(bottom_mask) > 0 else (max_y, int(np.mean(xs[ys == max_y])))

            # --- 阶段 2: 曲线拟合 ---
            if base_y != tip_y:
                # 1. 黄线: 底部一阶拟合 (用于转向决策基准)
                line_mask = ys >= line_bound
                if np.sum(line_mask) > 20:
                    yellow_k, yellow_b = np.polyfit(ys[line_mask], xs[line_mask], 1)
                else:
                    yellow_k, yellow_b = np.polyfit(ys, xs, 1)

                # 2. 青线: 全局二阶拟合 (用于判断曲率/领航限速)
                poly_coeffs = np.polyfit(ys, xs, 2)
                kappa = abs(poly_coeffs[0]) * 10000 

                # 3. 红线: 视觉融合 (纯展示用)
                plot_y = np.linspace(base_y, tip_y, num=50)
                plot_x_line = yellow_k * plot_y + yellow_b
                plot_x_curve = np.polyval(poly_coeffs, plot_y)
                
                alpha = ((base_y - plot_y) / (base_y - tip_y)) ** 2  
                plot_x_final = (1 - alpha) * plot_x_line + alpha * plot_x_curve

                # --- 阶段 3: 绘制曲线 ---
                pts_line = np.vstack((plot_x_line, plot_y)).astype(np.int32).T.reshape((-1, 1, 2))
                pts_curve = np.vstack((plot_x_curve, plot_y)).astype(np.int32).T.reshape((-1, 1, 2))
                pts_final = np.vstack((plot_x_final, plot_y)).astype(np.int32).T.reshape((-1, 1, 2))

                cv2.polylines(vis_img, [pts_line], False, (0, 255, 255), 2)  # 黄线 (基准)
                cv2.polylines(vis_img, [pts_curve], False, (255, 255, 0), 2) # 青线 (原始二阶)
                cv2.polylines(vis_img, [pts_final], False, (0, 0, 255), 6)   # 红线 (融合结果)
                
                cv2.circle(vis_img, (base_x, base_y), 8, (0, 255, 0), -1) 
                cv2.circle(vis_img, (tip_x, tip_y), 8, (255, 0, 0), -1)

                # --- 阶段 4: 控制计算 (PID) ---
                # 目标X计算: 基于黄线在屏幕底部的外推值
                target_x_yellow = yellow_k * max_y + yellow_b
                error_x = target_x_yellow - center_x_ref
                
                cv2.circle(vis_img, (int(target_x_yellow), int(max_y)), 12, (0, 255, 0), -1) # 决策锚点
                
                # 舵机转向 PD 控制
                servo_pwm = int(Config.SERVO_CENTER + (error_x * Config.KP) - (yellow_k * Config.KD))
                
                # 动态限速: 弯道越急(kappa越大)，减速越多
                speed_drop = kappa * Config.SPEED_DROP_FACTOR 
                motor_pwm = int(max(Config.MOTOR_STOP, Config.MOTOR_MAX_SPEED - speed_drop))

    # 绘制辅助UI
    cv2.line(vis_img, (center_x_ref, 0), (center_x_ref, height), (255, 0, 0), 2) # 屏幕中轴线
    cv2.putText(vis_img, f"ErrX: {error_x:.1f}", (width - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(vis_img, f"Servo: {servo_pwm}", (width - 200, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(vis_img, f"Spd: {motor_pwm}", (width - 200, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return vis_img, servo_pwm, motor_pwm


# ==============================================================================
# 模块 5: 主循环 (Main Execution)
# ==============================================================================
def main():
    print("🛠️ 正在初始化 RKNN NPU 模型...")
    rknn = RKNNLite()
    if rknn.load_rknn(Config.MODEL_PATH) != 0:
        print("❌ 模型加载失败")
        return
    if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
        print("❌ NPU 初始化失败")
        return

    # 初始化底层驱动
    chassis = ChassisController()
    
    # 视频流初始化
    video_src = Config.STREAM_URL if Config.SOURCE_TYPE == 'STREAM' else Config.VIDEO_PATH
    print(f"--> 正在启动视频流读取... [{Config.SOURCE_TYPE} 模式]")
    video_stream = VideoStreamWidget(video_src)
    time.sleep(1) # 等待缓冲
    
    print("🚀 自动驾驶寻线系统已启动！按 'q' 键退出...")

    while True:
        loop_start = time.time()
        
        # 1. 抓取画面
        ret, frame = video_stream.read()
        if not ret or frame is None:
            if Config.SOURCE_TYPE == 'FILE':
                break # 视频结束
            continue
            
        # 2. 预处理
        img_resized = cv2.resize(frame, Config.TARGET_RES)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img_rgb, axis=0)

        # 3. NPU 推理
        t_infer_start = time.time()
        outputs = rknn.inference(inputs=[input_data])
        npu_time_ms = (time.time() - t_infer_start) * 1000
        
        pred_mask = np.squeeze(outputs[0])
        if len(pred_mask.shape) == 3:
            pred_mask = np.argmax(pred_mask, axis=0)

        # 4. 图像处理与控制量计算 (调用模块 4)
        vis_img, servo_pwm, motor_pwm = process_frame(pred_mask, img_resized)

        # 5. 执行硬件控制
        chassis.send_velocity(motor_pwm, servo_pwm)

        # 6. 计算 FPS 并显示 UI
        fps = 1000 / ((time.time() - loop_start) * 1000 + 0.001)
        
        cv2.putText(vis_img, f"NPU: {npu_time_ms:.1f}ms | FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis_img, "- Purple: Raw Mask", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(vis_img, "- Yellow: Base Line (Ctrl)", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vis_img, "- Cyan: Pure Curve (Spd)", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(vis_img, "- Red: Blended Result", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Auto Drive Vision Engine", vis_img)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ================= 清理资源 =================
    print("🛑 正在停止并释放资源...")
    chassis.stop()
    video_stream.release()
    rknn.release()
    cv2.destroyAllWindows()
    print("✅ 系统已安全退出。")

if __name__ == '__main__':
    main()
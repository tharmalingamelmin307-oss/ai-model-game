# config.py
import numpy as np

# --- 共享内存与推流 ---
SHM_NAME = "shm_ar_video"
SHM_HEADER_SIZE = 16
STREAM_PORT = 5003
JPEG_QUALITY = 75

# --- 模型路径 ---
SEG_MODEL = "models/seg/ppliteseg_320_320_int8.rknn"
YOLO_MODEL = "models/det/yolov8_1.rknn"
REC_MODEL_PATH = "models/ocr/ppocrv4_rec_fp16.rknn"
DICT_PATH = "models/ocr/keys.txt"

# --- NPU 核心分配 (2核分割，1核检测+识别) ---
# 注意：SEG_CORE 现在改成一个列表，供双线程使用
SEG_CORES = [0]        # 语义分割占用 Core 0 
YOLO_CORE = 2          # 目标检测占用 Core 2
REC_CORE = 2           # OCR 识别也占用 Core 2 (与 YOLO 分时复用)

# --- 目标检测类别定义 ---
# ⚠️ 请根据你的 YOLO 实际训练类别修改这里的数字！
# 假设 0 是障碍物，1 是金币，2 是路牌
SIGN_CLASS_ID = 2

# --- 尺寸与预处理参数 ---
TARGET_RES = (960, 720) 
YOLO_SIZE = (640, 640)  
SEG_SIZE = (320, 320)   
REC_HEIGHT = 48
REC_WIDTH = 320

ROI_TOP_CUT_RATIO = 0.3 
MASK_ALPHA = 0.4

# # --- 鸟瞰图透视变换矩阵点 ---
# SRC_PTS = np.float32([[0.393, 0.639], [0.603, 0.636], [0.683, 0.799], [0.310, 0.794]])
# DST_PTS = np.float32([[0.385, 0.870], [0.615, 0.870], [0.615, 1.000], [0.385, 1.000]])

# --- 图像与逆透视参数 ---
TARGET_RES = (960, 720) 

# 使用你最新标定的 4 个顶点
SRC_PTS = np.float32([
    [0.432, 0.546],
    [0.566, 0.547],
    [0.856, 0.967],
    [0.175, 0.960],
])

DST_PTS = np.float32([
    [0.400, 0.600],
    [0.600, 0.600],
    [0.600, 1.000],
    [0.400, 1.000],
])

# 真实物理比例尺 (单位: cm/pixel)
CM_PER_PIXEL_X = 0.208333

# --- 路径规划阈值 ---
FORK_WIDTH_RATIO = 0.35 
FORK_GAP_RATIO = 0.15   
GAUSSIAN_SIGMA = 35.0   
SAFETY_MARGIN = 25      
SMOOTH_WINDOW = 5       

# --- 串口与 PID 控制参数 ---
SERIAL_PORT = '/dev/ttyS2'
BAUD_RATE = 115200
SERVO_CENTER = 750
SERVO_MIN, SERVO_MAX = 590, 910
MOTOR_STOP = 2000
MOTOR_MAX_SPEED = 2350 
KP = 0.16               
KD = 160.0
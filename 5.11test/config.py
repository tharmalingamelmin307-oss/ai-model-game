# config.py
import numpy as np

# --- 共享内存与推流 ---
SHM_NAME = "shm_ar_video"
SHM_HEADER_SIZE = 16
STREAM_PORT = 5003
JPEG_QUALITY = 75

# --- 模型路径 ---
SEG_MODEL = "models/seg/ppliteseg_320_320_int8.rknn"
YOLO_MODEL = "models/det/ppyoloe_crn_m_80e_custom_rk3588_fp16.rknn"
REC_MODEL_PATH = "models/ocr/ppocrv4_rec_fp16.rknn"
DICT_PATH = "models/ocr/keys.txt"

# --- NPU 核心分配 ---
SEG_CORES = [0]        # 语义分割占用 Core 0
YOLO_CORE = 2          # 目标检测占用 Core 2
REC_CORE = 2           # OCR 识别与 YOLO 分时复用

# --- 目标检测类别定义 ---
CLASS_NAMES = [
    "car",                  # 0
    "coin",                 # 1
    "person",               # 2
    "door",                 # 3
    "stone",                # 4
    "zebra_crossing",       # 5
    "traffic_light_red",    # 6
    "traffic_light_green",  # 7
    "traffic_light_yellow", # 8
    "sign",                 # 9
    "limit_sign",           # 10
    "start",                # 11
    "stop",                 # 12
]

SIGN_CLASS_ID = 9          # OCR 路牌
LIMIT_SIGN_CLASS_ID = 10   # 限速牌

# --- 尺寸与预处理参数 ---
TARGET_RES = (960, 720)
YOLO_SIZE = (768, 576)     # (width, height)
SEG_SIZE = (320, 320)
REC_HEIGHT = 48
REC_WIDTH = 320

ROI_TOP_CUT_RATIO = 0.3
MASK_ALPHA = 0.4

# --- 图像与逆透视参数 ---
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
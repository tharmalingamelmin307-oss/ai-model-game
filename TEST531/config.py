# config.py
"""项目全局配置.

这个文件只放“参数”和“静态常量”，尽量不放会改变主流程控制结构的业务逻辑。
目标是让调参工作集中在这里完成：

1. 模型路径和输入尺寸改动时，不需要翻业务线程代码
2. 板卡现场调参时，可以快速看懂每个参数的作用和风险
3. README 里描述的系统行为，可以直接映射到这里的配置项
"""

import numpy as np


# ---------------------------------------------------------------------------
# 共享内存与网页推流
# ---------------------------------------------------------------------------
# 上游采集进程写入共享内存时使用的名字。
# 这个值必须和“图像生产端”完全一致，否则主程序会一直连不上共享内存。
SHM_NAME = "shm_ar_video"

# 共享内存头部字节长度，当前协议固定为:
#   uint64 frame_id   : 8 字节，帧号，用于判断是否来了新帧
#   uint32 width      : 4 字节，图像宽
#   uint32 height     : 4 字节，图像高
# 总计 16 字节，所以这里固定写成 16。
# 如果上游协议变了，这里也必须一起改。
SHM_HEADER_SIZE = 16

# Flask 预览网页暴露给外部访问的端口。
# 这个端口只影响浏览器/局域网预览，不影响底层控制线程。
STREAM_PORT = 5003

# MJPEG 推流时 JPEG 编码质量。
# 数值越大:
# - 页面越清晰
# - CPU 编码开销越大
# - 网络带宽占用越大
# 数值越小:
# - 页面更糊
# - 但通常预览会更流畅
# 如果“车控本身正常，但浏览器看起来很卡”，优先先降这个值。
JPEG_QUALITY = 75

# ---------------------------------------------------------------------------
# 模型路径
# ---------------------------------------------------------------------------
# 分割模型路径。
# 当前主控链路依赖这个模型输出赛道 mask，所以它直接影响路径规划和转向控制。
# 这里的模型默认输入尺寸是 SEG_SIZE，对应下面的 320x320。
SEG_MODEL = "models/seg/ppliteseg_320_320_int8.rknn"

# 目标检测模型路径。
# 当前使用的是 PP-YOLOE 的 RKNN 版本，输出后处理由 modules/detector.py 负责。
# 如果替换模型，除了改这里，通常还要同步检查:
# - YOLO_SIZE
# - CLASS_NAMES
# - detector.py 的输出解析逻辑
YOLO_MODEL = "models/det/detv2/ppyoloe_crn_m_80e_custom_raw_rk3588_fp16.rknn"

# OCR 检测模型路径。
# 当前改回 det + rec 全流程时使用。
# 这个模型会在整张 OCR 输入图上先找文字区域，再把文字区域交给 rec 识别。
OCR_DET_MODEL_PATH = "models/ocr/ppocrv4_det_int8.rknn"

# OCR 识别模型路径。
# det 找到文字框后，再由这个 rec 模型读出具体文本。
REC_MODEL_PATH = "models/ocr/ppocrv4_rec_fp16.rknn"

# OCR 字典路径。
# 识别模型输出的是字符索引序列，最终要靠这份字典把索引翻译成字符。
# 如果更换 OCR 模型，这份字典往往也要一起换。
DICT_PATH = "models/ocr/keys.txt"


# ---------------------------------------------------------------------------
# NPU 核心分配
# ---------------------------------------------------------------------------
# 分割线程占用的 NPU 核列表。
# 现在只开了一个分割线程，所以这里只有 [0]。
# 如果后面想尝试多分割线程并行，可以把多个 core id 放进来，
# 但要注意多线程争抢输入队列和结果覆盖的问题。
SEG_CORES = [0]

# YOLO 检测线程绑定的 NPU 核。
# 目标是让检测和 OCR 分开跑，避免互相卡住。
YOLO_CORE = 2

# OCR 识别线程绑定的 NPU 核。
# OCR 虽然不是每帧都触发，但一旦触发就可能拖慢其它推理，所以单独占核更稳。
REC_CORE = 1


# ---------------------------------------------------------------------------
# 目标检测类别定义
# ---------------------------------------------------------------------------
# 检测模型类别名列表。
# 顺序必须和训练/导出 RKNN 时的类别顺序完全一致，不能只改名字不改模型。
# 否则会出现“框是对的，但类别解释全错”的问题。
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

# 普通语义路牌类别 id。
# 这类框会送去 OCR，当前主要识别 LEFT / RIGHT。
SIGN_CLASS_ID = 9

# 限速牌类别 id。
# 这类框也会送去 OCR。
# 当前不是“识别模型只解数字”，而是先按通用字典识别，
# 再在 main.py 的限速逻辑里只提取其中的数字字符参与生效判断。
LIMIT_SIGN_CLASS_ID = 10

# 普通语义路牌在进入 OCR 前，框的最小面积阈值，单位是 TARGET_RES 坐标系像素面积。
# 这个值比单独宽高更灵活，能更直接表达“这块牌子整体够不够大”。
# 调大:
# - 误判更少
# - 但远处牌子更不容易被识别
# 调小:
# - 更早开始识别
# - 但远距离误判风险会上升
OCR_MIN_SIGN_BOX_AREA = 50 * 100

# 限速牌在进入 OCR 前，框的最小面积阈值。
# 限速牌通常数字更集中，所以可以比普通语义牌略微放宽一点。
OCR_MIN_LIMIT_SIGN_BOX_AREA = 45 * 45

# 限速牌开始“正式生效判定”的面积阈值。
# 当 limit_sign 框面积大于这个值时，系统认为牌子已经足够近：
# - 不再继续对这个牌子做 OCR
# - 转而从前面累计到的历史 OCR 结果里挑选最可靠的候选来生效
# 这个值应明显大于 OCR_MIN_LIMIT_SIGN_BOX_AREA，给系统留出“远处多帧观察”的时间。
LIMIT_SIGN_APPLY_MIN_AREA = 80 * 80

# sign / limit_sign 在进入 OCR 前，四周需要保留的最小边距比例。
# 例如 0.03 表示检测框四边都要距离画面边界至少 3% 的宽/高。
# 这样可以尽量避开“框看起来够大，但其实有一部分已经贴边截断”的情况。
OCR_SIGN_EDGE_MARGIN_RATIO = 0.03

# OCR 识别结果的最小平均置信度阈值。
# 低于这个值的 OCR 文本会在进入主逻辑前直接丢弃，避免低分脏结果参与
# sign / limit_sign 的匹配和后续状态更新。
OCR_MIN_SCORE = 0.50

# OCR 触发类别的语义说明表。
# 当前主流程里真正决定“谁能进 OCR”的是 main.py 里的 class_id 判断，
# 这里更像是一份配置侧的约定文档，方便 README 和后续改造时对齐语义。
# OCR 实际运行的是“整图 det + rec”，YOLO 的 sign / limit_sign 只负责触发和回匹配。
OCR_TRIGGER_CLASSES = ("sign", "limit_sign")

# 限速牌历史候选的“稳定次数”门槛。
# 当前逻辑会先累计同一块牌子的历史 OCR 结果；
# 当牌子面积达到 LIMIT_SIGN_APPLY_MIN_AREA 后，
# 会优先从“重复次数 >= 这个值”的候选中选最优结果生效。
# 如果没有任何候选达到这个次数，系统会退而求其次，选历史里整体最优的那个候选。
LIMIT_SIGN_CONFIRM_FRAMES = 5

# 限速牌历史结果允许保留的最大断帧数。
# 如果 limit_sign 连续很多帧都没再出现，就认为上一块牌子的观察过程结束，
# 会把历史 OCR 候选清空，避免旧牌子的数字串到下一块新牌子。
LIMIT_SIGN_HISTORY_MAX_MISS_FRAMES = 15

# 斑马线停止线相对检测框底边的延长比例。
# 例如 0.35 表示在 zebra_crossing 框底边的基础上，左右各额外延长 35% 框宽。
# 这个值只影响页面可视化和“停止线横向覆盖范围”的观感，不改变检测框本身。
ZEBRA_STOPLINE_EXTEND_RATIO = 0.35

# 当停止线距离原图底部小于这个阈值时，认为车辆已经“接近停止位置”。
# 只有同时满足:
# - 检测到斑马线停止线
# - 红灯或黄灯存在
# - 停止线足够接近画面底部
# 才会触发强制停车。
# 当前先放宽到 240，优先验证停车链路是否能触发。
ZEBRA_STOPLINE_TRIGGER_DIST = 240

# YOLO 默认置信度阈值。
# 当某个类别没有在 CLASS_MIN_SCORES 里单独指定时，就回退到这个值。
YOLO_CONF_THRES = 0.25

# YOLO NMS 阈值。
YOLO_NMS_THRES = 0.45

# 类别级最小置信度覆盖表。
# 这里现在是真正参与 PP-YOLOE 前置候选过滤的，可直接拿来逐类试水。
# 哪个类别误检多，就单独把它抬高；哪个类别太难出框，就单独放低。
# 例如:
# CLASS_MIN_SCORES = {
#     "car": 0.45,
#     "person": 0.40,
#     "traffic_light_red": 0.35,
#     "limit_sign": 0.30,
# }
CLASS_MIN_SCORES = {
    "car": 0.45,
}

# ---------------------------------------------------------------------------
# 输入尺寸与基础预处理参数
# ---------------------------------------------------------------------------
# 统一显示/检测坐标系尺寸，格式为 (width, height)。
# 检测框最终会被映射到这个尺寸，再用于:
# - 页面叠框显示
# - 分割线程读取检测框
# - OCR 把 TARGET_RES 坐标系还原回原图坐标系时的基准
# 这个值不是共享内存原始分辨率，而是系统内部统一参照坐标。
TARGET_RES = (960, 720)

# YOLO 模型输入尺寸，格式为 (width, height)。
# 主线程会把原图缩放到这个尺寸供 detector.py 推理。
# 调大:
# - 远处小目标更容易被看见
# - 检测延迟更高
# 调小:
# - 延迟更低
# - 但远处小目标和细节更容易丢
YOLO_SIZE = (768, 576)

# 分割模型输入尺寸，格式为 (width, height)。
# 分割线程直接用这张小图做主控路径搜索。
# 这是实时控制链路的关键性能点之一。
SEG_SIZE = (320, 320)

# OCR 识别模型输入高度。
# PaddleOCR 一类识别模型通常固定高度，再按宽高比自适应宽度。
# 这个值通常要和导出模型时约定一致。
REC_HEIGHT = 48

# OCR 识别模型输入最大宽度。
# 实际 ROI 会先按比例缩放到 REC_HEIGHT，再右侧补零到这个宽度。
# 如果牌子很长，过小的宽度会让字符被横向压缩得太厉害。
REC_WIDTH = 320

# OCR 裁图外扩比例预留参数。
# 早期版本里曾尝试直接按 YOLO 框裁牌子再识别，所以保留了这个入口。
# 当前主流程已经改成“整图 OCR det + rec + 中心点回匹配”，这里暂时没有直接接入。
# 如果后续重新启用“按牌子框裁图识别”的轻量路径，可以优先复用这个参数。
OCR_CROP_EXPAND_RATIO = 0.18

# OCR 小 ROI 的最小高度保护值预留参数。
# 和上面的 OCR_CROP_EXPAND_RATIO 一样，它主要服务于“直接裁牌子 ROI 做识别”的旧方案。
# 当前整图 OCR 路径里没有直接使用它，先保留作后续实验入口。

OCR_MIN_CROP_HEIGHT = 28

# 顶部 ROI 裁剪比例预留参数。
# 当前主流程没有直接用它裁图，但它保留下来是为了后续做轻量化时方便接入。
# 例如以后如果只想让某些模块看画面下半区，可以直接用这个比例。
ROI_TOP_CUT_RATIO = 0.3

# 分割蒙版显示时的叠加透明度。
# 只影响调试画面观感，不影响控制量计算。
MASK_ALPHA = 0.4


# ---------------------------------------------------------------------------
# 逆透视与物理尺度
# ---------------------------------------------------------------------------
# 逆透视输入四点，使用相对比例表达。
# 程序初始化时会乘以 SEG_SIZE，换成真实像素坐标。
# 这样做的好处是换分辨率时不需要重写一整套点位。
# 这些点决定“前视图的哪块区域”会被拉成鸟瞰图，对最终路径形状和转向控制量影响很大。
SRC_PTS = np.float32([
    [0.432, 0.546],
    [0.566, 0.547],
    [0.856, 0.967],
    [0.175, 0.960],
])

# 逆透视输出四点，仍然用相对比例表达。
# 这组点定义了赛道在鸟瞰图里被拉到什么位置、什么宽度。
# 如果这里不合适，路径看起来可能是歪的，控制量也会跟着偏。
DST_PTS = np.float32([
    [0.400, 0.600],
    [0.600, 0.600],
    [0.600, 1.000],
    [0.400, 1.000],
])

# ---------------------------------------------------------------------------
# 路径规划与避障相关参数
# ---------------------------------------------------------------------------
# 分叉路口宽度比例预留参数。
# 当前版本主路径选择还主要靠 mask 搜索 + turn_intent，下面这些参数更多是为后续
# “检测结果主动参与改写路径”保留的，不是当前核心控制闭环的主开关。
FORK_WIDTH_RATIO = 0.35

# 分叉路口间隙比例预留参数。
FORK_GAP_RATIO = 0.15

# 分叉口 mask 判定时，底部连通白区至少要有多宽，单位是分割平面像素。
# 宽度不够时，认为仍是普通单车道，不判成岔路口。
FORK_MIN_BOTTOM_WIDTH = 110.0

# 分叉口 mask 判定时，上方用于检查“左右是否断开”的带状区域中心高度比例。
# 例如 0.60 表示在图像高度 60% 附近检查左右分支是否已经分开。
FORK_UPPER_ROW_RATIO = 0.60

# 上方分支判定带的高度，单位是分割平面像素。
FORK_UPPER_BAND_HEIGHT = 12

# 上方分支里，一段白色区域至少要有多少像素宽，才算一个有效分支。
FORK_MASK_MIN_BRANCH_PIXELS = 6

# 上方分支判定时，两段白色区域之间最少要断开多少像素，才算“不连通”。
FORK_MASK_GAP_THRESH = 12

# 上方左右分支之间最小横向分离距离，单位是分割平面像素。
FORK_MIN_BRANCH_SEP = 18.0

# 底部宽度判定时使用的底部带高度，单位是分割平面像素。
FORK_BOTTOM_BAND_HEIGHT = 16

# 检测目标扰动路径时使用的高斯扩散 sigma 预留值。
# 值越大，未来如果启用目标影响路径，障碍物对路径的“排斥范围”会越宽。
GAUSSIAN_SIGMA = 35.0

# 安全边界预留值。
# 后续若把障碍物投影到鸟瞰图并对路径做侵蚀/扩张，这个值通常就是安全留边的像素尺度。
SAFETY_MARGIN = 25

# 路径平滑窗口预留值。
# 后续如果对基础路径点做滑动平均或曲线平滑，这个窗口通常决定平滑强度。
SMOOTH_WINDOW = 5

# 分叉口路径锁定持续帧数。
# 一旦系统在明显分叉里选定左/右某一支，会优先沿该分支继续保持这么多帧，
# 用来抑制岔路口左右横跳。
PATH_LOCK_HOLD_FRAMES = 6

# 判定“当前确实出现了明显左右分叉”的最小横向间距，单位是分割平面像素。
# 调大:
# - 更谨慎，不容易触发锁定
# 调小:
# - 更容易把宽车道或轻微岔开也当成分叉
PATH_LOCK_FORK_MIN_SEP = 28.0

# 候选路径与上一帧路径之间的连续性惩罚权重。
# 调大:
# - 更强调与上一帧保持一致，分叉口更稳
# - 但也更容易“恋旧”，切换到新分支会慢一点
# 调小:
# - 更容易跟着当前帧几何最优结果切换
PATH_CONTINUITY_WEIGHT = 2.0

# 计算“当前候选路径和上一帧路径有多接近”时的采样点数量。
# 点越多，连续性判断越细，但计算也会稍重一点。
PATH_CONTINUITY_SAMPLE_COUNT = 6

# 会被投影到分割/鸟瞰图平面里的“规划相关类别”。
# 当前这些目标暂时主要用于调试显示，不直接改写主路径。
PLANNING_CLASS_NAMES = (
    "car",
    "coin",
    "person",
    "stone",
    "door",
)

# 在鸟瞰图里额外画半径圈的类别。
# 这些类别通常体积感更强，画圈更容易判断它们对路径的潜在影响范围。
PLANNING_CIRCLE_CLASS_NAMES = (
    "car",
    "person",
)

# 鸟瞰图调试点的显示风格。
# 格式:
#   class_name: {"color": (B, G, R), "label": "text"}
# 只影响可视化，不影响控制逻辑。
PLANNING_MARKER_STYLES = {
    "car": {"color": (0, 0, 255), "label": "car"},
    "coin": {"color": (0, 255, 255), "label": "coin"},
    "person": {"color": (255, 80, 80), "label": "person"},
    "stone": {"color": (0, 165, 255), "label": "stone"},
    "door": {"color": (255, 180, 0), "label": "door"},
    "zebra_crossing": {"color": (255, 255, 255), "label": "zebra"},
}


# ---------------------------------------------------------------------------
# 串口与控制参数
# ---------------------------------------------------------------------------
# 下位机串口设备名。
# 如果板卡串口号变了，这里要跟着改。
SERIAL_PORT = '/dev/ttyS2'

# 串口波特率。
# 必须和下位机固件配置一致，否则会通信异常或乱码。
BAUD_RATE = 115200

# 舵机中心值。
# 这是“车身理论正前方”对应的 PWM。
# 如果车辆总是轻微向一侧跑，即使视觉误差正常，也可能需要先调这里。
SERVO_CENTER = 750

# 舵机安全最小/最大 PWM。
# 用于硬限制输出，避免控制算法在极端情况下打到危险位置。
SERVO_MIN, SERVO_MAX = 590, 910

# 电机停止 PWM 预留值。
# 当前主线程真正下发的是 target_speed 这种速度级命令，不直接使用这个值，
# 但它保留下来是为了后续如果改成更底层 PWM 控制时方便接。
MOTOR_STOP = 2000

# 电机最大速度预留值。
# 目前 serial_control_thread() 里仍主要用内部的 10~30 速度档逻辑，
# 这个值暂时更多是配置语义保留位。
MOTOR_MAX_SPEED = 2350

# 单一转向控制量换算到舵机 PWM 的增益。
# 当前控制量定义为：
# “路径上每个点到图像最底部中点连线的斜率 * 该点行号”的总和。
# 调大:
# - 舵机转向更积极
# - 但更容易抖或打满
# 调小:
# - 舵机更稳
# - 但可能转不过弯
STEER_SIGNAL_PWM_GAIN = 0.003

# 用单一转向控制量做动态降速时的增益。
# 控制量绝对值越大，说明当前横向偏差/路径趋势越强，目标速度会随之降低。
STEER_SIGNAL_SPEED_GAIN = 0.02

# 计算“点到底部中点连线斜率”时使用的最小纵向间距。
# 作用是防止路径底部附近的点因为 dy 过小，把控制量瞬间放得过大。
STEER_SIGNAL_MIN_DY = 8.0


# ---------------------------------------------------------------------------
# 主流程运行时参数
# ---------------------------------------------------------------------------
# 主控制状态默认值。
# main.py 启动时会深拷贝这里，生成一份真正运行态。
# 这组值本身不应该在运行过程中被直接修改；真正变化的是 main.py 里的
# global_control_data。
# 字段说明：
# - steer_signal: 分割线程输出的单一转向控制量
# - turn_intent: OCR 识别出的语义分叉意图，-1 表示左，1 表示右
# - turn_intent_fid: 上一次更新 turn_intent 时对应的帧号
# - speed_limit: 当前已经正式生效的速度上限，None 表示尚无限速
# - speed_limit_fid: 上一次更新 speed_limit 时对应的帧号
# - limit_sign_history: 限速牌历史累计池，按数字聚合 count / score_sum
# - limit_sign_last_detect_fid: 最近一次检测到 limit_sign 的帧号
# - zebra_stopline_y: 停止线在 TARGET_RES 坐标系中的 y 值
# - traffic_light_state: 当前交通灯状态，允许值 red / yellow / green / ""
# - traffic_stop_active: 当前是否已经进入“红黄灯强制停车”状态
# - actual_servo_pwm: 当前串口线程真正准备下发的舵机 PWM
# - target_speed: 当前串口线程真正准备下发的目标速度档位
DEFAULT_CONTROL_DATA = {
    "steer_signal": 0.0,
    "turn_intent": -1,
    "turn_intent_fid": -1,
    "speed_limit": None,
    "speed_limit_fid": -1,
    "limit_sign_history": {},
    "limit_sign_last_detect_fid": -1,
    "zebra_stopline_y": None,
    "traffic_light_state": "",
    "traffic_stop_active": False,
    "actual_servo_pwm": SERVO_CENTER,
    "target_speed": 10,
}

# FPS 统计默认值。
# - seg_frames / yolo_frames: 统计窗口内累计的帧数
# - seg_fps / yolo_fps: 最近一个统计窗口算出来的平均 FPS
DEFAULT_FPS_STATS = {
    "seg_frames": 0,
    "yolo_frames": 0,
    "seg_fps": 0.0,
    "yolo_fps": 0.0,
}

# 三条工作队列容量。
# 当前策略是“宁可丢旧帧，也不堆积延迟”，所以容量都很小：
# - SEG_QUEUE_MAXSIZE: 分割主控链路，只保留最新一帧
# - YOLO_QUEUE_MAXSIZE: 检测链路，也只保留最新一帧
# - OCR_QUEUE_MAXSIZE: OCR 相对更慢，允许保留极少量待处理任务
# 如果你发现整机“反应慢但 FPS 看起来还行”，先不要盲目把这些值调大。
# 队列变大通常只会让显示和控制更滞后。
SEG_QUEUE_MAXSIZE = 1
YOLO_QUEUE_MAXSIZE = 1
OCR_QUEUE_MAXSIZE = 2

# 帧率统计刷新周期，单位秒。
# 值越小，页面上的 FPS 数字更新越灵敏，但波动也会更明显；
# 值越大，显示更平滑，但更不容易立刻看出性能抖动。
FPS_STATS_UPDATE_INTERVAL = 1.0


# ---------------------------------------------------------------------------
# 主流程日志节流参数
# ---------------------------------------------------------------------------
# 所有日志节流时间统一使用“秒”作为单位。
# 这些值只影响终端打印频率，不改变核心控制逻辑本身。
# 一般来说：
# - 调小: 更容易看到细节，但终端会更吵
# - 调大: 更安静，适合长时间跑车
# 如果需要某一类问题重点观察，优先只调对应那一项，而不是全局一起改。

# 默认日志节流间隔。
# 当调用 throttled_log() 时没有显式传入 min_interval，就会回退到这里。
LOG_INTERVAL_DEFAULT = 1.5

# limit_sign 消失太久导致历史清空时的日志节流。
LOG_INTERVAL_LIMIT_HISTORY_RESET = 1.5

# 限速正式生效时的日志节流。
# 这里适当短一点，便于近距离连续观察牌子生效过程。
LOG_INTERVAL_SPEED_LIMIT_EFFECTIVE = 0.8

# “检测到了牌子，但因为面积/贴边等原因没进 OCR”的日志节流。
LOG_INTERVAL_OCR_SKIP = 2.0

# OCR 任务真正入队时的日志节流。
LOG_INTERVAL_OCR_ENTER = 1.0

# OCR 队列满了、主动丢旧任务时的日志节流。
LOG_INTERVAL_OCR_QUEUE_DROP = 2.0

# 整图 OCR 本帧完全没找到文字区域时的日志节流。
LOG_INTERVAL_OCR_FULLFRAME_EMPTY = 2.0

# OCR 找到了文本，但和某个检测框没匹配上时的日志节流。
LOG_INTERVAL_OCR_MATCH_EMPTY = 2.0

# OCR 成功识别出文本时的日志节流。
LOG_INTERVAL_OCR_RESULT = 1.0

# OCR 结果因为分数太低被过滤掉时的日志节流。
LOG_INTERVAL_OCR_LOW_SCORE = 1.2

# LEFT / RIGHT 语义路牌生效时的日志节流。
LOG_INTERVAL_TURN_INTENT = 0.8

# 限速牌 OCR 读出了非数字文本时的日志节流。
LOG_INTERVAL_SPEED_LIMIT_PARSE_EMPTY = 1.2

# 限速牌仍在累计历史、尚未正式生效时的日志节流。
LOG_INTERVAL_SPEED_LIMIT_PENDING = 1.0

# 串口线程整体控速状态摘要的日志节流。
LOG_INTERVAL_CONTROL_STATE = 1.2

# 红绿灯停车链路细节日志的节流。
LOG_INTERVAL_TRAFFIC_STOP_DETAIL = 1.2


# ---------------------------------------------------------------------------
# 场景类别与主流程判定参数
# ---------------------------------------------------------------------------
# 下面这些名字会在 main.py 里用来从 CLASS_NAMES 里反查类别 id。
# 这样做的好处是：只要模型类别名顺序保持一致，就不必把所有逻辑都写死成数字。
ZEBRA_CROSSING_CLASS_NAME = "zebra_crossing"
TRAFFIC_LIGHT_RED_CLASS_NAME = "traffic_light_red"
TRAFFIC_LIGHT_GREEN_CLASS_NAME = "traffic_light_green"
TRAFFIC_LIGHT_YELLOW_CLASS_NAME = "traffic_light_yellow"

# 当上面的类别名在 CLASS_NAMES 里找不到时，回退使用的固定类别 id。
# 正常情况下这些值不应该生效；它们更像是一层容错保护。
ZEBRA_CROSSING_CLASS_ID_FALLBACK = 5
TRAFFIC_LIGHT_RED_CLASS_ID_FALLBACK = 6
TRAFFIC_LIGHT_GREEN_CLASS_ID_FALLBACK = 7
TRAFFIC_LIGHT_YELLOW_CLASS_ID_FALLBACK = 8

# 限速牌真正生效时的速度偏移量。
# 例如 OCR 读到 20，若这里为 1，则实际下发上限是 19。
# 这么做相当于给车辆留一点保守余量。
LIMIT_SIGN_EFFECTIVE_SPEED_OFFSET = 1

# OCR 检测框与 OCR 文字框做“最近中心点匹配”时使用的初始最大距离。
# 这是一个纯内部比较起点，只要足够大即可，不需要现场频繁调整。
OCR_MATCH_INIT_DIST = 1e9


# ---------------------------------------------------------------------------
# 主预览图绘制参数
# ---------------------------------------------------------------------------
# 这一组参数只影响网页预览和调试画面的观感，不影响车辆控制逻辑。
# 建议只有在你确实想调整页面布局、叠字位置、颜色区分度时再动它们。

# YOLO 框颜色：
# - 默认类别: 红色
# - sign: 黄色
# - limit_sign: 蓝色
YOLO_DEFAULT_BOX_COLOR = (0, 0, 255)
YOLO_SIGN_BOX_COLOR = (0, 255, 255)
YOLO_LIMIT_SIGN_BOX_COLOR = (255, 0, 0)

# YOLO 检测框和文字标签样式。
# 如果现场画面分辨率或网页缩放方式变了，看不清框或字太小，可优先调这里。
YOLO_BOX_THICKNESS = 3
YOLO_LABEL_FONT_SCALE = 0.55
YOLO_LABEL_THICKNESS = 2

# 标签文字相对框顶部/底部的布局参数。
# 作用是尽量避免文字压住框内部目标，同时防止贴顶越界。
YOLO_LABEL_TOP_MARGIN = 20
YOLO_LABEL_TOP_OFFSET = 8
YOLO_LABEL_BOTTOM_OFFSET = 18

# YOLO 摘要栏里最多展示多少个检测目标。
# 值过大时，摘要文本容易在页面下方横向挤爆。
YOLO_SUMMARY_MAX_ITEMS = 4

# 斑马线停止线的绘制颜色与粗细。
ZEBRA_STOPLINE_COLOR = (0, 255, 255)
ZEBRA_STOPLINE_THICKNESS = 3

# 页面状态面板与文字颜色。
# 这些颜色主要为了快速区分：
# - 常规文字
# - 强调值（如目标速度）
# - 红绿灯状态
# - 强制停车提示
PREVIEW_PANEL_BG_COLOR = (0, 0, 0)
PREVIEW_PANEL_BORDER_COLOR = (0, 255, 255)
PREVIEW_TEXT_COLOR = (255, 255, 255)
PREVIEW_TEXT_ACCENT_COLOR = (0, 255, 255)
PREVIEW_TEXT_LIMIT_COLOR = (0, 200, 255)
PREVIEW_TEXT_STOP_COLOR = (0, 0, 255)
PREVIEW_LIGHT_RED_COLOR = (0, 0, 255)
PREVIEW_LIGHT_GREEN_COLOR = (0, 255, 0)
PREVIEW_LIGHT_YELLOW_COLOR = (0, 255, 255)
PREVIEW_TEXT_FONT_SCALE = 0.42
PREVIEW_TEXT_THICKNESS = 1

# 两块信息面板在 TARGET_RES 画面中的位置。
# 第一块显示 FPS / 控制 / 速度 / 灯色等状态；
# 第二块显示简短 YOLO 检测摘要。
PREVIEW_STATUS_PANEL_TOP_LEFT = (2, 102)
PREVIEW_STATUS_PANEL_BOTTOM_RIGHT = (230, 154)
PREVIEW_YOLO_PANEL_TOP_LEFT = (2, 156)
PREVIEW_YOLO_PANEL_BOTTOM_RIGHT = (520, 178)

# 每一行叠字的绘制坐标。
# 如果你后面改了面板尺寸或加减显示项，通常也要同步调整这些位置。
PREVIEW_TEXT_POS_FPS = (6, 116)
PREVIEW_TEXT_POS_CTRL = (6, 132)
PREVIEW_TEXT_POS_SPEED = (6, 148)
PREVIEW_TEXT_POS_LIMIT = (150, 148)
PREVIEW_TEXT_POS_LIGHT = (6, 164)
PREVIEW_TEXT_POS_STOP = (150, 164)
PREVIEW_TEXT_POS_YOLO_SUMMARY = (6, 172)


# ---------------------------------------------------------------------------
# 串口控制与线程节奏参数
# ---------------------------------------------------------------------------
# 串口超时时间，单位秒。
# 串口偶尔抖动时，这个值太大可能会拖慢控制循环；太小则更容易把一次短暂卡顿视为失败。
SERIAL_TIMEOUT = 0.1

# 串口线程最终允许输出的目标速度范围。
# 当前并不是直接发电机 PWM，而是发一个速度档位：
# - CONTROL_MIN_SPEED: 常规最低巡航速度
# - CONTROL_MAX_SPEED: 直道或轻弯时允许的最高速度
CONTROL_MIN_SPEED = 10
CONTROL_MAX_SPEED = 30

# 串口数据包头尾。
# 只有在你同时修改上下位机通信协议时才需要调整。
SERIAL_PACKET_HEADER = (0xAA, 0x55)
SERIAL_PACKET_TAIL = (0x0D, 0x0A)

# 各线程等待/轮询节奏，单位秒。
# 这些值主要影响 CPU 占用、实时性和页面刷新感受：
# - 太小: 更灵敏，但更吃 CPU
# - 太大: 更省资源，但会更“顿”
CONTROL_LOOP_SLEEP = 0.01
SHM_FRAME_POLL_SLEEP = 0.002
SHM_RETRY_SLEEP = 1.0
VIDEO_FEED_IDLE_SLEEP = 0.01
VIDEO_FEED_FRAME_SLEEP = 0.02
STARTUP_SHARED_THREAD_SLEEP = 0.1
STARTUP_SEG_THREAD_SLEEP = 0.2

# Flask 推流监听地址。
# `0.0.0.0` 表示允许局域网其它设备访问当前板子的网页预览。
FLASK_HOST = "0.0.0.0"


# ---------------------------------------------------------------------------
# OCR det + rec 运行参数
# ---------------------------------------------------------------------------
# OCR 检测模型的输入尺寸。
# 调大通常更容易找到小字，但 det 开销更高。
OCR_DET_INPUT_SIZE = 480

# OCR 检测概率图二值化阈值。
# 调低: 更容易保留弱文字区域，但误检轮廓也可能增多
# 调高: 更干净，但小字/弱字更容易被漏掉
OCR_DET_BINARY_THRESH = 0.30

# OCR 检测后处理时，最小轮廓面积阈值。
# 用于滤掉非常小的噪声连通域。
OCR_DET_MIN_CONTOUR_AREA = 100.0

# OCR 检测后处理的膨胀核大小与膨胀次数。
# 作用是把轻微断裂的文字区域连起来，近似一个轻量版 unclip。
# - 调大: 更容易把碎片连起来，也更可能把相邻噪声粘在一起
# - 调小: 轮廓更保守
OCR_DET_DILATE_KERNEL_SIZE = 3
OCR_DET_DILATE_ITERATIONS = 2


# ---------------------------------------------------------------------------
# YOLO 预处理与解析参数
# ---------------------------------------------------------------------------
# PaddleDetection 标准预处理均值与方差。
# 只有在你更换模型、且新模型导出时使用了不同归一化方式，才需要调整。
YOLO_PREPROCESS_MEAN = (0.485, 0.456, 0.406)
YOLO_PREPROCESS_STD = (0.229, 0.224, 0.225)

# 检测框的最小宽高阈值，单位是输出坐标系像素。
# 过小的框通常没有足够语义价值，也容易成为噪声。
YOLO_BOX_MIN_SIZE = 2

# 是否只在第一次推理时打印详细调试信息。
# True 适合临时排查输出格式；平时建议保持 False。
YOLO_DEBUG_ONCE = False

# 是否强制走“单 tensor 原始输出”解析路径。
# 只有当你明确知道当前模型不是 PP-YOLOE demo 多分支输出时再打开。
YOLO_EXPECT_RAW_SINGLE_OUTPUT = False

# 单 tensor 输出的框格式是否是 cxcywh。
# 当前项目默认假设导出结果已经是 xyxy，通常不需要改。
YOLO_SINGLE_TENSOR_IS_CXCYWH = False

# 当检测分数看起来像 logits 而不是 0~1 概率时，Sigmoid 前的截断范围。
# 作用是避免 exp 过程中数值过大。
YOLO_SINGLE_TENSOR_LOGIT_CLIP = 10.0

# 自动判断“当前分数还没过 Sigmoid”的上界阈值。
# 如果某帧最高分明显大于这个值，程序会认为拿到的是 logits。
YOLO_SCORE_SIGMOID_TRIGGER_MAX = 1.5

# 自动判断“当前框坐标是否仍是归一化坐标”的阈值。
# 如果框坐标最大值不超过这个值，程序会尝试按输入宽高还原到像素尺度。
YOLO_NORMALIZED_BOX_MAX = 2.0


# ---------------------------------------------------------------------------
# 分割线程运行与调试参数
# ---------------------------------------------------------------------------
# 分割路径拟合系数的 EMA 历史权重。
# 越接近 1.0 越稳，越接近 0.0 响应越快。
SEG_EMA_ALPHA = 0.6

# 估计石头更靠近左/右分支时，左右候选路径至少要拉开这么多像素才认为可比较。
# 当前代码里石头主要还用于调试显示，这个参数暂时不直接影响正式分支选择。
STONE_BRANCH_MIN_SEP = 12

# 自底向上路径搜索参数。
# 这些值决定了 mask 搜索的采样密度、连通判定和候选路径数量上限。
# 如果分叉口容易漏掉某一支，或直道上路径抖动明显，优先看这里。
SEG_PATH_SEARCH_STEP_Y = 5

# 同一层里，相邻白色像素之间如果断开超过这个阈值，就认为属于不同分支。
SEG_PATH_GAP_THRESH = 15

# 路径搜索前，对二值 mask 底部局部做轻微膨胀，优先修补起步区域的小断裂。
# 膨胀只作用于“搜索用 mask”，不会改动原始分割模型输出。
SEG_PATH_DILATE_KERNEL = 3
SEG_PATH_DILATE_ITER = 1
SEG_PATH_DILATE_BOTTOM_HEIGHT = 40

# 路径搜索底部起点相关参数。
# - BOTTOM_MARGIN: 底部预留，避免正贴边采样
# - BOTTOM_SLICE_HEIGHT: 底部起始白区统计高度
# - MIN_START_PIXELS: 底部白像素至少达到多少才开始建路径
SEG_PATH_BOTTOM_MARGIN = 5
SEG_PATH_BOTTOM_SLICE_HEIGHT = 15
SEG_PATH_MIN_START_PIXELS = 8

# 向上搜索的最高比例位置。
# 例如 0.1 表示最多搜到图像高度 10% 的地方就停止。
SEG_PATH_SCAN_TOP_RATIO = 0.1

# 某一层最少要有多少白像素，才认为这一层值得参与路径连接。
SEG_PATH_MIN_SLICE_PIXELS = 4

# 单个连通分支至少要有多少个像素点，才作为候选分支中心。
SEG_PATH_MIN_BRANCH_POINTS = 3

# 一对左右边界之间至少要拉开多宽，才认为是有效通道。
# 过小通常只是边缘毛刺或小洞，不适合拿来建路径节点。
SEG_PATH_MIN_PAIR_WIDTH = 6

# 相邻两层中心点横向差值小于该阈值时，认为它们可以连成同一路径。
SEG_PATH_CONNECT_X_THRESH = 50

# 相邻两层的左右边界即使没有真正重叠，只要只差这么多像素，也允许视作同一路径。
# 这个值主要用来给轻微断裂、轻微错位留一点连接余量。
SEG_PATH_CONNECT_OVERLAP_MARGIN = 10

# 同时保留的候选路径上限。
# 调大能保留更多分支假设，但计算量和抖动风险也会上升。
SEG_PATH_MAX_ACTIVE_PATHS = 15

# 候选路径至少要有多少个节点，才参与最终打分。
SEG_PATH_MIN_LENGTH = 3

# 候选路径打分权重：
# - LENGTH_SCORE_GAIN: 路径越长越加分
# - SMOOTH_SCORE_GAIN: 横向变化越不平滑，扣分越多
# - CENTER_PENALTY_GAIN: 偏离局部中心越大，扣分越多
SEG_PATH_LENGTH_SCORE_GAIN = 50.0
SEG_PATH_SMOOTH_SCORE_GAIN = 20.0
SEG_PATH_CENTER_PENALTY_GAIN = 3.5

# 只在“最高分附近”的候选里做最终左右支选择。
# 值越大，进入最终池子的路径越多。
SEG_PATH_TOP_TIER_SCORE_GAP = 150.0

# 最终拟合路径重新采样成多少个密集点，用来计算 steer_signal 和画线。
SEG_PATH_DENSE_SAMPLES = 30

# 鸟瞰图上规划目标点的调试样式参数。
SEG_DEBUG_PLANNING_DOT_RADIUS = 4
SEG_DEBUG_PLANNING_MIN_RADIUS = 4
SEG_DEBUG_PLANNING_TEXT_OFFSET_X = 6
SEG_DEBUG_PLANNING_TEXT_OFFSET_Y = -6
SEG_DEBUG_PLANNING_TEXT_MIN_Y = 12
SEG_DEBUG_PLANNING_TEXT_FONT_SCALE = 0.4
SEG_DEBUG_PLANNING_TEXT_THICKNESS = 1

# 主分割调试图与鸟瞰图小窗的绘制风格。
SEG_DEBUG_PATH_COLOR = (255, 0, 255)
SEG_DEBUG_PATH_THICKNESS = 2
SEG_DEBUG_LEFT_BOUNDARY_COLOR = (255, 255, 0)
SEG_DEBUG_RIGHT_BOUNDARY_COLOR = (0, 165, 255)
SEG_DEBUG_BOUNDARY_THICKNESS = 2
SEG_DEBUG_BIRD_PATH_COLOR = (0, 0, 255)
SEG_DEBUG_BIRD_PATH_THICKNESS = 2
SEG_DEBUG_BIRD_LEFT_BOUNDARY_COLOR = (255, 255, 0)
SEG_DEBUG_BIRD_RIGHT_BOUNDARY_COLOR = (0, 165, 255)
SEG_DEBUG_BIRD_BOUNDARY_THICKNESS = 2
SEG_DEBUG_BOTTOM_MID_COLOR = (255, 255, 0)
SEG_DEBUG_BOTTOM_MID_RADIUS = 4
SEG_DEBUG_PIP_DIVISOR = 3
SEG_DEBUG_PIP_BORDER_COLOR = (255, 255, 255)
SEG_DEBUG_PIP_BORDER_THICKNESS = 1

# 分割调试图左上角文字的字号、位置与颜色。
# 这些信息主要用于现场快速确认：
# - Seg / YOLO FPS
# - 当前 steer_signal 与估算舵机 PWM
# - 石头分支判断的调试文本
SEG_DEBUG_TEXT_FONT_SCALE = 0.8
SEG_DEBUG_TEXT_THICKNESS = 1
SEG_DEBUG_TEXT_POS_FPS = (5, 18)
SEG_DEBUG_TEXT_POS_CTRL = (5, 36)
SEG_DEBUG_TEXT_POS_STONE = (5, 54)
SEG_DEBUG_TEXT_POS_BRANCH = (5, 72)
SEG_DEBUG_TEXT_COLOR_FPS = (0, 255, 0)
SEG_DEBUG_TEXT_COLOR_CTRL = (0, 255, 255)
SEG_DEBUG_TEXT_COLOR_STONE = (0, 200, 255)
SEG_DEBUG_TEXT_COLOR_BRANCH = (255, 200, 0)

# 分割链路阶段耗时统计。
# 开启后会按时间窗口打印 infer / preprocess / search / fit / render / total 的平均耗时。
SEG_PROFILE_ENABLED = True
SEG_PROFILE_PRINT_INTERVAL = 1.0

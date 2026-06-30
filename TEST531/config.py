# config.py
"""项目全局配置.

这个文件只放“参数”和“静态常量”，尽量不放会改变主流程控制结构的业务逻辑。
目标是让调参工作集中在这里完成：

1. 模型路径和输入尺寸改动时，不需要翻业务线程代码
2. 板卡现场调参时，可以快速看懂每个参数的作用和风险
3. README 里描述的系统行为，可以直接映射到这里的配置项
"""

from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent


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
# 这里的模型默认输入尺寸是 SEG_SIZE，对应下面的 416x160。
SEG_MODEL = str(PROJECT_ROOT / "models/seg/segv3/pipi416x160_argmax_rk3588_int8.rknn")

# 目标检测模型路径。
# 当前使用的是 PP-YOLOE 的 RKNN 版本，输出后处理由 modules/detector.py 负责。
# 如果替换模型，除了改这里，通常还要同步检查:
# - YOLO_SIZE
# - CLASS_NAMES
# - detector.py 的输出解析逻辑
YOLO_MODEL = str(PROJECT_ROOT / "models/det/detv3/ppyoloe_crn_m_80e_custom_official_split_rk3588_int8_512x384.rknn")

# OCR 检测模型路径。
# 当前改回 det + rec 全流程时使用。
# 这个模型会在整张 OCR 输入图上先找文字区域，再把文字区域交给 rec 识别。
OCR_DET_MODEL_PATH = str(PROJECT_ROOT / "models/ocr/ppocrv4_det_int8.rknn")

# OCR 识别模型路径。
# det 找到文字框后，再由这个 rec 模型读出具体文本。
REC_MODEL_PATH = str(PROJECT_ROOT / "models/ocr/ppocrv4_rec_fp16.rknn")

# OCR 字典路径。
# 识别模型输出的是字符索引序列，最终要靠这份字典把索引翻译成字符。
# 如果更换 OCR 模型，这份字典往往也要一起换。
DICT_PATH = str(PROJECT_ROOT / "models/ocr/keys.txt")


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

# 临时关闭限速牌链路。
# False 时:
# - limit_sign 不触发 OCR
# - 不再从历史结果写入 speed_limit
# - 串口控速忽略已有 speed_limit
# 普通 sign 的 LEFT / RIGHT 语义路牌不受影响。
LIMIT_SIGN_ENABLED = False

# 普通语义路牌在进入 OCR 前，框的最小面积阈值，单位是 TARGET_RES 坐标系像素面积。
# 这个值比单独宽高更灵活，能更直接表达“这块牌子整体够不够大”。
# 调大:
# - 误判更少
# - 但远处牌子更不容易被识别
# 调小:
# - 更早开始识别
# - 但远距离误判风险会上升
# detv3 输入从 detv2 的 768x576 换到 512x384，面积类门槛按输入面积比例缩放。
OCR_AREA_TRIGGER_SCALE = (512 * 384) / (768 * 576)
OCR_MIN_SIGN_BOX_AREA = int(round(40 * 160 * OCR_AREA_TRIGGER_SCALE))

# 限速牌在进入 OCR 前，框的最小面积阈值。
# 限速牌通常数字更集中，所以可以比普通语义牌略微放宽一点。
OCR_MIN_LIMIT_SIGN_BOX_AREA = int(round(45 * 45 * OCR_AREA_TRIGGER_SCALE))

# 限速牌开始“正式生效判定”的面积阈值。
# 当 limit_sign 框面积大于这个值时，系统认为牌子已经足够近：
# - 不再继续对这个牌子做 OCR
# - 转而从前面累计到的历史 OCR 结果里挑选最可靠的候选来生效
# 这个值应明显大于 OCR_MIN_LIMIT_SIGN_BOX_AREA，给系统留出“远处多帧观察”的时间。
LIMIT_SIGN_APPLY_MIN_AREA = int(round(80 * 80 * OCR_AREA_TRIGGER_SCALE))

# sign / limit_sign 在进入 OCR 前，四周需要保留的最小边距比例。
# 例如 0.03 表示检测框四边都要距离画面边界至少 3% 的宽/高。
# 这样可以尽量避开“框看起来够大，但其实有一部分已经贴边截断”的情况。
OCR_SIGN_EDGE_MARGIN_RATIO = 0.03

# OCR 识别结果的最小平均置信度阈值。
# 低于这个值的 OCR 文本会在进入主逻辑前直接丢弃，避免低分脏结果参与
# sign / limit_sign 的匹配和后续状态更新。
OCR_MIN_SCORE = 0.50

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
ZEBRA_STOPLINE_TRIGGER_DIST = 400

# YOLO 默认置信度阈值。
# 当某个类别没有在 CLASS_MIN_SCORES 里单独指定时，就回退到这个值。
YOLO_CONF_THRES = 0.50

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
    "car": 0.50,
    "coin": 0.50,
    "person": 0.50,
    "door": 0.50,
    "stone": 0.50,
    "zebra_crossing": 0.50,
    "traffic_light_red": 0.50,
    "traffic_light_green": 0.50,
    "traffic_light_yellow": 0.50,
    "sign": 0.50,
    "limit_sign": 0.50,
    "start": 0.50,
    "stop": 0.50,
}

# 类别级最大面积比例过滤，比例基准是最终输出坐标系画面面积。
# 用于压掉小目标类别上异常大的误框，参数对齐当前 detv3 测试脚本。
YOLO_MAX_AREA_RATIO_BY_CLASS = {
    "coin": 0.08,
    "traffic_light_red": 0.08,
    "traffic_light_green": 0.08,
    "traffic_light_yellow": 0.08,
    "sign": 0.16,
    "limit_sign": 0.16,
    "start": 0.16,
    "stop": 0.16,
}

# 如果检测框同时贴近两条边，且面积占比超过这里，就认为更像异常大框。
YOLO_EDGE_MARGIN_RATIO = 0.02
YOLO_EDGE_TOUCH_MAX_AREA_RATIO = 0.30

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
YOLO_SIZE = (512, 384)

# 分割模型输入尺寸，格式为 (width, height)。
# 分割线程直接用这张小图做主控路径搜索。
# 这是实时控制链路的关键性能点之一。
SEG_SIZE = (416, 160)

# 分割模型输入裁剪比例。
# 0.5 表示先裁掉原图上半部分，只把下半部分 resize 到 SEG_SIZE。
# 这个值要和当前 segv3 测试脚本里的 crop_y = h // 2 保持一致。
SEG_INPUT_CROP_TOP_RATIO = 0.5

# OCR 识别模型输入高度。
# PaddleOCR 一类识别模型通常固定高度，再按宽高比自适应宽度。
# 这个值通常要和导出模型时约定一致。
REC_HEIGHT = 48

# OCR 识别模型输入最大宽度。
# 实际 ROI 会先按比例缩放到 REC_HEIGHT，再右侧补零到这个宽度。
# 如果牌子很长，过小的宽度会让字符被横向压缩得太厉害。
REC_WIDTH = 320

# 分割蒙版显示时的叠加透明度。
# 只影响调试画面观感，不影响控制量计算。
MASK_ALPHA = 0.4

# 是否在预览画面上给整块分割 mask 染色。
# False 时只画路径/边界/文字，不改变原图大面积颜色，也能减少一点渲染开销。
SEG_DEBUG_DRAW_MASK = False

# 裁剪分割模型预览合成阈值。
# 网页预览仍以完整原图为底，只把 Seg 渲染图里相对原图明显变化的像素叠上去。
# 值越小: mask/线更完整，但可能带入更多分割小图底色
# 值越大: 原图保留更干净，但弱透明 mask 可能变淡
SEG_PREVIEW_OVERLAY_DIFF_THRESH = 24


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
# 上方分支里，一段白色区域至少要有多少像素宽，才算一个有效分支。
FORK_MASK_MIN_BRANCH_PIXELS = 6

# 上方分支判定时，两段白色区域之间最少要断开多少像素，才算“不连通”。
FORK_MASK_GAP_THRESH = 16

# 上方左右分支之间最小横向分离距离，单位是分割平面像素。
FORK_MIN_BRANCH_SEP = 23.0

# 底部宽度判定时使用的底部带高度，单位是分割平面像素。
FORK_BOTTOM_BAND_HEIGHT = 16

# Y 岔路扫描范围，按分割平面 y 坐标配置。
# segv3 已经裁掉上半图，原 320 空间的 160~299 对应当前 0~139。
FORK_SCAN_Y_TOP = 0
FORK_SCAN_Y_BOTTOM = 139

# 分叉口“中间缺口双边张开”约束。
# 当前更关注的是：
# - 左支内边界向左张开
# - 右支内边界向右张开
# - 两者形成的中间缺口在一段连续区域里总体变大
#   其中局部允许短暂持平，甚至允许少量小回退
# - 但不能只靠极少数几步突然跳大，而是需要一段有效张开过程
# 这样比单看边界断裂更贴近真正的 Y 型岔路几何。
FORK_INNER_OPEN_MIN_ROWS = 5
FORK_INNER_OPEN_MIN_GAP_GROWTH = 14.0
FORK_INNER_OPEN_MIN_SIDE_GROWTH = 5.0
FORK_INNER_OPEN_MIN_STEP_GAIN = 1.0
FORK_INNER_OPEN_MIN_POSITIVE_GAP_ROWS = 3
FORK_INNER_OPEN_MIN_POSITIVE_SIDE_ROWS = 2
FORK_INNER_OPEN_MAX_STEP_REGRESSION = 3.0
FORK_INNER_OPEN_MAX_MISS_ROWS = 2

# 汇合场景扫描只看底部多少行。
# 当前 416x160 输入下设为 140 表示只扫描 y >= 20 的近处区域。
SEG_SCENE_SCAN_BOTTOM_HEIGHT = 140

# 汇合引导线参数。
# 先要求指定 y 范围内连续出现若干行“左/右边缘存在 mask 或最左白点到最右白点足够宽”，
# 再去搜索单侧汇合尖角。这里的宽度不要求中间联通，左右分支断开也会计入总宽。
# 如果尖角成立，就按“可信侧边界 +/- 当前行完整赛道宽度”补出缺失侧边界，
# 并按单路模式继续搜索，不再切成岔路。
MERGE_GUIDE_SCAN_Y_TOP = 0
MERGE_GUIDE_SCAN_Y_BOTTOM = 130
# 底部额外汇合扫描范围。
# 这段不需要满足“足够宽/贴边连续行”的前置触发条件，直接参与汇合角点特征搜索。
MERGE_GUIDE_FREE_SCAN_Y_TOP = 130
MERGE_GUIDE_FREE_SCAN_Y_BOTTOM = 160
MERGE_GUIDE_MIN_ROW_WIDTH = 286.0
MERGE_GUIDE_MIN_WIDE_ROWS = 4
MERGE_GUIDE_MIN_SIDE_DELTA = 13.0
# 汇合塌陷侧内边界需要有足够斜率/锐度，避免门或远处贴边造成的平缓漂移误判。
MERGE_GUIDE_MIN_INNER_ANGLE_DEG = 10.0
MERGE_GUIDE_MIN_INNER_SHARPNESS = 0.10
MERGE_GUIDE_REQUIRE_EDGE_ABOVE_INNER = True
MERGE_GUIDE_MIN_EDGE_ABOVE_ROWS = 2
MERGE_GUIDE_OPPOSITE_MAX_DRIFT = 16.0
# 塌陷侧成立后，对侧可信边界允许的逐行最大跳变。
# 例如左侧塌陷时，右侧边界必须连续稳定，不能中途突然横跳。
MERGE_GUIDE_OPPOSITE_MAX_STEP_JUMP = 10.0
MERGE_GUIDE_MAX_MISS_ROWS = 2
# 汇合补线命中后，沿 y 方向额外覆盖的行数。
# 这里不是斜率外推；每一行仍按“可信侧边界 +/- 当前行完整赛道宽度”单独计算。
MERGE_GUIDE_EXTEND_TOP_ROWS = 20
MERGE_GUIDE_EXTEND_BOTTOM_ROWS = 90
# 汇合补线最终允许保留的 y 范围。
# segv3 裁剪坐标系里 0~159 就是原底部半图。
MERGE_GUIDE_LINE_Y_MIN = 0
MERGE_GUIDE_LINE_Y_MAX = 160

# 汇合补线与可信对侧边界之间的横向保护间距。
# 补左线时，如果它离右边界太近，就限制到“右边界 - gap”左侧；
# 补右线时，如果它离左边界太近，就推到“左边界 + gap”右侧。
MERGE_GUIDE_LINE_MIN_GAP = 13.0
MERGE_GUIDE_LINE_THICKNESS = 2
# 汇合状态机：连续命中若干帧才进入补线；进入后等底部赛道宽度稳定恢复再退出。
MERGE_STATE_CONFIRM_FRAMES = 3
MERGE_STATE_EXIT_BOTTOM_ROWS = 5
MERGE_STATE_EXIT_WIDTH_THRESH = 340.0
MERGE_STATE_EXIT_CONFIRM_FRAMES = 2
MERGE_STATE_EXIT_NO_EDGE_Y_TOP = 40
MERGE_STATE_EXIT_NO_EDGE_Y_BOTTOM = 130

# 固定赛道宽度表的来源坐标系。
# 当前表仍然是旧 320x320 搜索平面里的样本；代码会按当前 SEG_SIZE 和裁剪比例映射到 416x160。
SEG_FIXED_WIDTH_SOURCE_SIZE = (320, 320)
SEG_FIXED_WIDTH_SOURCE_CROP_TOP_RATIO = 0.5

# 320x320 搜索平面里各 y 行对应的固定赛道宽度。
# 这组值来自现场采集的 Seg320Width 样本逐行取均值后固化，
# 汇合口修正时会按“完整赛道宽度”反推缺失侧边界。
SEG_FIXED_WIDTHS_320 = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 8, 14, 18, 22, 25,
    30, 75, 85, 96, 98, 86, 68, 63, 63, 65, 66, 67, 68, 69, 70, 70, 71, 72, 73, 73, 74, 75, 76, 78, 80, 80, 81, 82, 83, 83,
    84, 85, 86, 87, 88, 88, 90, 91, 92, 93, 94, 96, 96, 98, 99, 100, 101, 103, 104, 104, 106, 108, 109, 110, 111, 113, 114, 115, 117, 118, 119, 120,
    122, 123, 125, 126, 127, 129, 130, 132, 132, 134, 135, 136, 138, 139, 140, 142, 143, 145, 146, 148, 149, 150, 152, 153, 154, 156, 158, 159, 160, 162, 164, 165,
    167, 168, 169, 170, 172, 172, 174, 176, 178, 178, 180, 180, 182, 184, 184, 186, 187, 188, 190, 190, 192, 193, 194, 195, 196, 197, 201, 202, 204, 205, 206, 207,
    208, 209, 210, 211, 212, 214, 214, 215, 217, 218, 218, 220, 221, 222, 223, 224, 226, 227, 228, 230, 232, 233, 233, 234, 235, 236, 237, 239, 240, 241, 241, 241,
    241, 241,
]

# 平滑后的固定赛道宽度表。原始 `SEG_FIXED_WIDTHS_320` 保留用于对照和回退；
# 补线默认优先使用这张表，减少局部宽度跳变导致的补线抖动。
SEG_FIXED_WIDTHS_320_SMOOTH = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 15, 17, 24, 31, 41,
    51, 59, 65, 70, 74, 78, 77, 75, 72, 68, 67, 67, 68, 69, 70, 70,
    71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 83, 84, 85,
    86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 103,
    104, 105, 106, 108, 109, 110, 111, 113, 114, 115, 117, 118, 119, 121, 122, 123,
    125, 126, 127, 129, 130, 131, 133, 134, 135, 136, 138, 139, 140, 142, 143, 145,
    146, 148, 149, 150, 152, 153, 155, 156, 158, 159, 161, 162, 164, 165, 166, 168,
    169, 170, 172, 173, 174, 176, 177, 178, 180, 181, 182, 183, 185, 186, 187, 188,
    189, 191, 192, 193, 194, 196, 197, 199, 200, 201, 203, 204, 206, 207, 208, 209,
    210, 211, 212, 213, 214, 215, 217, 218, 219, 220, 221, 222, 223, 225, 226, 227,
    228, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 240, 241, 241, 241,
]

# 判定“当前确实出现了明显左右分叉”的最小横向间距，单位是分割平面像素。
# 调大:
# - 更谨慎，不容易触发锁定
# 调小:
# - 更容易把宽车道或轻微岔开也当成分叉
PATH_LOCK_FORK_MIN_SEP = 36.0

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

# 单一转向控制量换算到舵机 PWM 的增益。
# 当前控制量先由路径点斜率做“带远近权重的归一化平均”得到：
#   slope = (path_x - image_bottom_center_x) / max(image_bottom_y - path_y, STEER_SIGNAL_MIN_DY)
#   weight = path_y ** STEER_SIGNAL_ROW_WEIGHT_GAMMA
#   steer_signal = sum(slope * weight) / sum(weight)
# 再按普通巡线 / 金币 / 避障模式乘对应模式增益，最后乘本参数换算为 PWM。
# 调大:
# - 舵机转向更积极
# - 但更容易抖或打满
# 调小:
# - 舵机更稳
# - 但可能转不过弯
STEER_SIGNAL_PWM_GAIN = 0.015

# 用单一转向控制量做动态降速时的增益。
# 控制量绝对值越大，说明当前横向偏差/路径趋势越强，目标速度会随之降低。
STEER_SIGNAL_SPEED_GAIN = 0.002

# 计算“点到底部中点连线斜率”时使用的最小纵向间距。
# 作用是防止路径底部附近的点因为 dy 过小，把控制量瞬间放得过大。
STEER_SIGNAL_MIN_DY = 8.0
# 远近权重指数。越大，越强调靠近图像底部/车身近处的路径点；
# 越小，远处路径点占比越高。归一化不会抹掉远近信息，主要由这个指数保留远近差异。
STEER_SIGNAL_ROW_WEIGHT_GAMMA = 1.3
# 归一化控制量缩放。归一化后原始 steer_signal 常为个位数，
# 这里把它放大到更接近旧版累计控制量的显示和 PWM 调参量级。
STEER_SIGNAL_NORMALIZED_SCALE = 3000.0
# 避障车额外横向偏移控制项。
# car 避障路径会相对“当前选中道路原始基础循线路径”产生横向位移；
# 这里把每个控制行的横向偏移量单独累加进控制量，弥补只看斜率时横移响应偏弱的问题。
STEER_SIGNAL_CAR_LATERAL_OFFSET_GAIN = 1000.0
# 无金币/无车时，控制只看中下部这一段，底部最靠下 30 行不参与。
# 这两个值是分割图坐标里的 y 行号；在 416x160 输入下，60~130 表示只取中下部路径。
STEER_SIGNAL_NO_TARGET_ROW_MIN = 60.0
STEER_SIGNAL_NO_TARGET_ROW_MAX = 130.0
# 无目标控制增益：只在没有金币、没有避障车时乘到 steer_signal 上。
# 可用于补偿无目标控制行段变短、归一化后转向偏软等情况。
STEER_SIGNAL_NO_TARGET_GAIN = 1.0
# 金币控制增益：只在 coin 路径规划 active 时生效。
# 注意金币自身还可能有距离相关的 control_gain，这里是额外的全局模式增益。
STEER_SIGNAL_COIN_GAIN = 1.0
# 避障车控制增益：只在 car 避障 active 且没有 coin 控制接管时生效。
# 如果避障路径已经画得够左但舵机反应偏小，优先调大这个值。
STEER_SIGNAL_CAR_GAIN = 1.0


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
# - person_stop_active: 当前是否已经进入“行人强制停车”状态
# - person_*: 行人停车/放行判定用的最近状态
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
    "person_stop_active": False,
    "person_bottom_y": None,
    "person_bottom_center_x": None,
    "person_bottom_right_x": None,
    "person_dist_to_bottom": None,
    "person_left_boundary_x": None,
    "person_right_boundary_x": None,
    "person_clear_line_x": None,
    "person_stop_event": "",
    "person_clear_frames": 0,
    "person_miss_frames": 0,
    "person_last_frame_id": -1,
    "person_last_bottom_center_x": None,
    "person_released_outside_left": False,
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

# Seg 流水线模式。
# 开启后分割链路拆成:
# - NPU 推理线程: 从 seg_queue 取图，只负责输出 mask
# - 后处理线程: 取最新 mask，负责路径搜索 / 控制量 / 渲染
# 这样吞吐接近 max(推理耗时, 后处理耗时)，代价是控制结果通常会对应上一帧。
SEG_PIPELINE_ENABLED = True
SEG_PIPELINE_QUEUE_MAXSIZE = 1

# 帧率统计刷新周期，单位秒。
# 值越小，页面上的 FPS 数字更新越灵敏，但波动也会更明显；
# 值越大，显示更平滑，但更不容易立刻看出性能抖动。
FPS_STATS_UPDATE_INTERVAL = 1.0

# Seg 阶段耗时诊断日志。
# 开启后每隔一段时间打印 inference / search / fit / render / total，用来定位掉帧瓶颈。
SEG_PROFILE_LOG_ENABLED = False
SEG_PROFILE_LOG_INTERVAL = 2.0

# 主流程运行时耗时诊断日志。
# 开启后会额外打印采集预处理、Seg 推理线程等待、发布、MJPEG 编码等耗时。
# 用来判断页面 FPS 是卡在输入来帧、模型推理、后处理发布还是网页推流。
MAIN_PROFILE_LOG_ENABLED = False
MAIN_PROFILE_LOG_INTERVAL = 2.0


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
LOG_INTERVAL_DEFAULT = 2.0

# 限速正式生效时的日志节流。
# 只在正式达到限速生效条件时打印。
LOG_INTERVAL_SPEED_LIMIT_EFFECTIVE = 2.0

# 路牌达到 OCR 识别条件、任务真正入队时的日志节流。
LOG_INTERVAL_OCR_ENTER = 2.0

# LEFT / RIGHT 语义路牌生效时的日志节流。
LOG_INTERVAL_TURN_INTENT = 2.0

# 红绿灯停车条件满足、强制停车链路触发时的日志节流。
LOG_INTERVAL_TRAFFIC_STOP_DETAIL = 2.0

# 行人停车条件满足、强制停车链路触发时的日志节流。
LOG_INTERVAL_PERSON_STOP_DETAIL = 1.0

# 检测到行人但未必已经触发停车时的状态日志节流。
LOG_INTERVAL_PERSON_DETECT_DETAIL = 2.0

# 串口发送异常日志节流。
LOG_INTERVAL_SERIAL_ERROR = 2.0

# ---------------------------------------------------------------------------
# 场景类别与主流程判定参数
# ---------------------------------------------------------------------------
# 下面这些名字会在 main.py 里用来从 CLASS_NAMES 里反查类别 id。
# 这样做的好处是：只要模型类别名顺序保持一致，就不必把所有逻辑都写死成数字。
ZEBRA_CROSSING_CLASS_NAME = "zebra_crossing"
PERSON_CLASS_NAME = "person"
TRAFFIC_LIGHT_RED_CLASS_NAME = "traffic_light_red"
TRAFFIC_LIGHT_GREEN_CLASS_NAME = "traffic_light_green"
TRAFFIC_LIGHT_YELLOW_CLASS_NAME = "traffic_light_yellow"

# 当上面的类别名在 CLASS_NAMES 里找不到时，回退使用的固定类别 id。
# 正常情况下这些值不应该生效；它们更像是一层容错保护。
ZEBRA_CROSSING_CLASS_ID_FALLBACK = 5
PERSON_CLASS_ID_FALLBACK = 2
TRAFFIC_LIGHT_RED_CLASS_ID_FALLBACK = 6
TRAFFIC_LIGHT_GREEN_CLASS_ID_FALLBACK = 7
TRAFFIC_LIGHT_YELLOW_CLASS_ID_FALLBACK = 8

# 行人停车逻辑。
# 触发不做路径 ROI 过滤，和斑马线类似，只看 person 框底边是否足够靠近画面底部。
# 恢复前进需要同时满足:
# - 行人底部中心连续向左移动若干帧
# - person 框底部中心已经越过当前选中路的释放线
PERSON_STOP_TRIGGER_DIST = 300    #貌似200也不是很多（720）
PERSON_CLEAR_MOVE_FRAMES = 2
PERSON_CLEAR_MIN_LEFT_DX = 3.0
# 行人放行线位置。0.0 是当前道路左边界，0.5 是道路中线，1.0 是右边界。
# 当前行人从右往左穿行，越过中线后就允许恢复前进，避免等完全出界导致起步过晚。
PERSON_CLEAR_LANE_RATIO = 0.50
PERSON_STOP_MISS_RELEASE_FRAMES = 3

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
# 速度平滑。上升慢一点，避免金币/弯道控制量变化时突然加速；下降保留更快响应。
CONTROL_SPEED_SMOOTH_ENABLED = True
CONTROL_SPEED_MAX_STEP_UP = 1
CONTROL_SPEED_MAX_STEP_DOWN = 2

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

# RKNNLite 在加载静态模型时会打印大量原生 warning/info。
# 当前这些 warning 对运行无影响，默认在模型初始化阶段静默掉，让终端只保留业务日志。
SUPPRESS_RKNN_INIT_OUTPUT = True


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
# 当前 detv3 RKNN 已固化 mean/std，Python 侧只喂 0-255 RGB uint8。

# 检测框的最小宽高阈值，单位是输出坐标系像素。
# 过小的框通常没有足够语义价值，也容易成为噪声。
YOLO_BOX_MIN_SIZE = 3

# 每帧最多保留的检测框数量，按置信度从高到低截断。
YOLO_MAX_DETS = 50

# 进入 NMS 前，每个类别最多保留多少个候选框。
# detv3 低阈值会产生大量候选；如果全部做 Python NMS，会明显抢占 Seg 线程 CPU。
YOLO_PRE_NMS_TOPK_PER_CLASS = 80



# ---------------------------------------------------------------------------
# 分割线程运行与调试参数
# ---------------------------------------------------------------------------
# 分割路径拟合系数的 EMA 历史权重。
# 越接近 1.0 越稳，越接近 0.0 响应越快。
SEG_EMA_ALPHA = 0.6

# 相邻帧路径稳定约束。
# 这组参数工作在 SEG_SIZE 的 320x320 路径平面里，用来防止分割噪声或分叉候选切换
# 造成路径横向瞬间跳变。
# - MAX_FRAME_X_JUMP: 最终输出路径每帧允许横向移动的最大像素量，设为 0 可关闭限幅
# - TEMPORAL_SCORE_GAIN: 候选路径相对上一帧偏移越大，打分扣得越多
# - TEMPORAL_SOFT_MAX_JUMP: 超过这个单点跳变量后，候选会受到额外重罚
# - TEMPORAL_EXCESS_SCORE_GAIN: 超出软阈值部分的额外扣分权重
# - TEMPORAL_MIN_OVERLAP_POINTS: 候选与上一帧路径至少重叠多少个采样点才参与时域打分
# - HOLD_MISSING_FRAMES: 当前帧没搜到路径时，短暂沿用上一帧路径的最大帧数
SEG_PATH_STABILITY_ENABLED = True
SEG_PATH_MAX_FRAME_X_JUMP = 29.0
SEG_PATH_TEMPORAL_SCORE_GAIN = 5.0
SEG_PATH_TEMPORAL_SOFT_MAX_JUMP = 32.0
SEG_PATH_TEMPORAL_EXCESS_SCORE_GAIN = 18.0
SEG_PATH_TEMPORAL_MIN_OVERLAP_POINTS = 4
SEG_PATH_HOLD_MISSING_FRAMES = 2

# 估计石头更靠近左/右分支时，左右候选路径至少要拉开这么多像素才认为可比较。
# 当前代码里石头主要还用于调试显示，这个参数暂时不直接影响正式分支选择。
STONE_BRANCH_MIN_SEP = 12

# 自底向上路径搜索参数。
# 这些值决定了 mask 搜索的采样密度、连通判定和候选路径数量上限。
# 如果分叉口容易漏掉某一支，或直道上路径抖动明显，优先看这里。
SEG_PATH_SEARCH_STEP_Y = 14

# 快速中心线模式。
# 开启后不做多候选扩展，只在当前处理后的 mask 上逐行取八邻域边界左右中点。
# 汇合仍先补线，Y 岔路仍先切左右分支，再对选中的分支取中点。
SEG_CENTERLINE_ONLY_MODE = True
# 中心线模式下，先取最大连通白区，再按行直接取该区域左右边界中点。
SEG_CENTERLINE_LARGEST_COMPONENT_ONLY = True
# 中心线模式逐行采样步长。1 表示真正每一行都取中点；调大可进一步少点提速。
SEG_CENTERLINE_ROW_STEP = 1

# 同一层里，相邻白色像素之间如果断开超过这个阈值，就认为属于不同分支。
SEG_PATH_GAP_THRESH = 20

# 路径搜索前，对二值 mask 底部局部做轻微膨胀，优先修补起步区域的小断裂。
# 膨胀只作用于“搜索用 mask”，不会改动原始分割模型输出。
SEG_PATH_DILATE_KERNEL = 3
SEG_PATH_DILATE_ITER = 1
SEG_PATH_DILATE_BOTTOM_HEIGHT = 40

# 路径相关计算只保留底部多少行；窗口外视为无 mask。
# 设为 0 或负数可关闭这个限制。
SEG_PATH_ACTIVE_HEIGHT = 160

# 路径搜索前只保留触达底部的连通白区；如果没有触底区域，再回退到最大连通区。
# 这样能滤掉悬空碎片，同时避免把岔路里可走的较小触底支路误删。
SEG_KEEP_BOTTOM_COMPONENTS = True

# 路径搜索底部起点相关参数。
# - BOTTOM_MARGIN: 底部预留，避免正贴边采样
# - BOTTOM_TOUCH_HEIGHT: 底部多少行以内触达才允许起路径
SEG_PATH_BOTTOM_MARGIN = 5
SEG_PATH_BOTTOM_TOUCH_HEIGHT = 20

# 向上搜索的最高比例位置（兜底值）。
# 正常情况下会优先按当前帧 mask 的最高有效行动态截断；
# 这里只有在当前帧几乎没有有效 mask 时才会回退到这个比例上界。
SEG_PATH_SCAN_TOP_RATIO = 0.1

# 某一层最少要有多少白像素，才认为这一层值得参与路径连接。
SEG_PATH_MIN_SLICE_PIXELS = 30

# 单个连通分支至少要有多少个像素点，才作为候选分支中心。
SEG_PATH_MIN_BRANCH_POINTS = 3

# 一对左右边界之间至少要拉开多宽，才认为是有效通道。
# 过小通常只是边缘毛刺或小洞，不适合拿来建路径节点。
SEG_PATH_MIN_PAIR_WIDTH = 8

# 单行最多参与路径搜索的白区片段数量。
# 复杂 mask 下过多碎片会让候选路径组合膨胀，直接拖慢 search / fit。
SEG_PATH_MAX_ROW_SEGMENTS = 4

# 相邻两层中心点横向差值小于该阈值时，认为它们可以连成同一路径。
SEG_PATH_CONNECT_X_THRESH = 65

# 相邻两层的左右边界即使没有真正重叠，只要只差这么多像素，也允许视作同一路径。
# 这个值主要用来给轻微断裂、轻微错位留一点连接余量。
SEG_PATH_CONNECT_OVERLAP_MARGIN = 13

# 普通搜索同时保留的候选路径上限。
# 调大能保留更多分支假设，但计算量和抖动风险也会上升。
SEG_PATH_MAX_ACTIVE_PATHS = 4

# Y 型岔路已经切成左右区域后，每个区域内部只保留最优候选。
SEG_PATH_MAX_FORK_SIDE_ACTIVE_PATHS = 1

# 候选路径至少要有多少个节点，才参与最终打分。
SEG_PATH_MIN_LENGTH = 8

# 候选路径打分权重：
# - LENGTH_SCORE_GAIN: 路径越长越加分
# - SMOOTH_SCORE_GAIN: 横向变化越不平滑，扣分越多
# - CENTER_PENALTY_GAIN: 偏离局部中心越大，扣分越多
SEG_PATH_LENGTH_SCORE_GAIN = 50.0
SEG_PATH_SMOOTH_SCORE_GAIN = 40.0
SEG_PATH_CENTER_PENALTY_GAIN = 3.5

# 只在“最高分附近”的候选里做最终左右支选择。
# 值越大，进入最终池子的路径越多。
SEG_PATH_TOP_TIER_SCORE_GAP = 150.0

# 最终拟合路径重新采样成多少个密集点，用来计算 steer_signal 和画线。
SEG_PATH_DENSE_SAMPLES = 30

# 金币分段路径启用时，舵机控制最多只看车身底部往上这段距离。
# 金币比这个更远时，仍按普通底部窗口控制；金币进入窗口内后才按距离给比例增益。
COIN_PATH_CONTROL_FIRST_SEGMENT_ONLY = True
COIN_PATH_CONTROL_BAND_HEIGHT = 96.0
# 金币进入控制窗口后的参考距离。实际增益约为 BAND_HEIGHT / 金币距离。
# 例如 BAND=96，金币距底部 60 行，则增益约 1.6；超过 96 行则不加增益。
COIN_PATH_CONTROL_REFERENCE_ROWS = 96.0

# 终端打印当前赛道宽度的节流间隔，单位秒。
TRACK_WIDTH_LOG_INTERVAL = 1.5

# ---------------------------------------------------------------------------
# 车辆避障路径参数
# 这组参数工作在分割输入 `SEG_SIZE = 416x160` 的坐标系里。
# y 方向是 160 行高度，所有 `*_ROWS` 都是“离底部多少行”的意思。
# 当前策略不是“看不见车就立刻回正”，而是先走 `AVOIDING`，再等 `CLEAR`
# 条件满足后分两段回正，避免车尾擦到障碍物。
# ---------------------------------------------------------------------------
# 检测到 car 时，把车框左侧两个角点向左偏移后作为避障目标点。
# 这些目标点会像金币锚点一样插入路径，使车优先从障碍车左侧绕过。
CAR_AVOIDANCE_ENABLED = True
CAR_AVOIDANCE_TARGET_LEFT_OFFSET = 50.0
CAR_AVOIDANCE_TOP_LEFT_OFFSET = 50.0
# 当障碍车位于当前基础循线路径右侧时，避障目标点至少压到基础线左侧这么多像素。
CAR_AVOIDANCE_RIGHT_OBSTACLE_MIN_LEFT_OFFSET = 20.0
CAR_AVOIDANCE_TOP_ANCHOR_HEIGHT_RATIO = 1.0 / 3.0
CAR_AVOIDANCE_AREA_SCALE_ENABLED = False
CAR_AVOIDANCE_AREA_SCALE_MIN_AREA = 0.0
CAR_AVOIDANCE_AREA_SCALE_MAX_AREA = 14000.0
CAR_AVOIDANCE_AREA_SCALE_MIN = 1.2
CAR_AVOIDANCE_AREA_SCALE_MAX = 1.8
CAR_AVOIDANCE_EDGE_SKIP_BOTTOM_ANCHOR = True
CAR_AVOIDANCE_EDGE_MARGIN = 8.0
CAR_AVOIDANCE_BOTTOM_LEAD_ROWS = 12.0
CAR_AVOIDANCE_BOTTOM_NEAR_ROWS = 12.0
# 固定分段偏移：检测框底部离画面底部 150 行内开始左偏 60，
# 80 行内左偏 90。贴右下角且高度较矮时单独固定左偏 50。
CAR_AVOIDANCE_DYNAMIC_OFFSET_ENABLED = False
CAR_AVOIDANCE_START_OFFSET_ROWS = 150.0
CAR_AVOIDANCE_START_LEFT_OFFSET = 60.0
CAR_AVOIDANCE_NEAR_OFFSET_ROWS = 50.0
CAR_AVOIDANCE_NEAR_LEFT_OFFSET = 90.0
# car 跟踪锁定。锁定主要看车框底部中心点的连续性，面积只做异常框过滤。
# 连续命中后进入避障；退出时要等 `car` 丢失且路径稳定并越过障碍范围，
# 然后先轻微回正，再慢慢回到普通巡线。
CAR_AVOIDANCE_LOCK_HIT_FRAMES = 2
CAR_AVOIDANCE_SEARCH_RADIUS = 48.0
CAR_AVOIDANCE_SEARCH_RADIUS_MISS_GAIN = 16.0
CAR_AVOIDANCE_TRACK_EMA_ALPHA = 0.65
# car 框短暂变小、被遮挡或漏检时，继续沿用最近一次避障锚点的帧数。
# 调大可避免太早回正；过大会让已经绕过车后继续偏左太久。
CAR_AVOIDANCE_MISS_FRAMES = 5
CAR_AVOIDANCE_MASK_RADIUS = 10
CAR_AVOIDANCE_MIN_SCORE = 0.0
CAR_AVOIDANCE_MAX_AREA = 0.0

# 避障退出状态机。
# 车丢失后不立刻回正，而是先进入 CLEARING。
# CLEARING 里会先保留一段左偏，再慢慢回到正常巡线。
CAR_AVOIDANCE_CLEARING_MISS_FRAMES = 3
CAR_AVOIDANCE_CLEARING_DECAY_FRAMES = 12
CAR_AVOIDANCE_CLEARING_RESIDUAL_KEEP = 1.0
CAR_AVOIDANCE_CLEARING_DONE_RESIDUAL = 0.06
CAR_AVOIDANCE_CLEARING_LEFT_BIAS_MAX = 30.0
# 车框还在但已经很贴底、贴右且高度较小时，直接给固定左偏。
CAR_AVOIDANCE_FIXED_LEFT_BIAS_HEIGHT_THRESH = 60.0
CAR_AVOIDANCE_FIXED_LEFT_BIAS_RIGHT_MARGIN = 10.0
CAR_AVOIDANCE_FIXED_LEFT_BIAS_BOTTOM_MARGIN = 10.0
CAR_AVOIDANCE_FIXED_LEFT_BIAS_VALUE = 50.0
# 避障时允许金币的窗口。
# AVOIDING 期间只允许“底部第一段金币”；CLEARING 期间只允许更靠底且更安全的金币。
CAR_AVOIDANCE_COIN_ALLOW_BOTTOM_ROWS = 28.0
CAR_AVOIDANCE_CLEARING_COIN_SAFE_ROWS = 16.0

# ---------------------------------------------------------------------------
# 金币分段路径参数
# ---------------------------------------------------------------------------
# 只处理 coin:
# - 使用检测框底部两点中点作为金币锚点
# - 只保留落在当前路径纵向范围内、附近有 mask、且离当前中线不超过赛道半宽的金币点
# - 路径按“底部路径点 -> 金币点 -> 最远路径点”从近到远分段重采样
COIN_PATH_ENABLED = True
COIN_PATH_ROI_Y_MIN = 15
# coin 底部边缘严格区。靠近分割输入底边时，用“当前路径中线 +/- 赛道半宽”
# 再额外收窄，压掉车尾/路边被误识别成 coin 的情况。
COIN_PATH_ROI_BOTTOM_STRICT_ROWS = 80
COIN_PATH_BOTTOM_HALF_WIDTH_SCALE = 0.42
COIN_PATH_EDGE_REJECT_ENABLED = True
COIN_PATH_EDGE_REJECT_MARGIN = 8.0
COIN_PATH_MASK_RADIUS = 10
COIN_PATH_HALF_WIDTH_SCALE = 1.0
COIN_PATH_BYPASS_FRAME_JUMP = True
# 金币锁定/过滤参数。新目标不做连续帧确认，但锁定后允许短暂丢失沿用。
COIN_TRACK_MAX_MISS_FRAMES = 2
COIN_TRACK_SEARCH_RADIUS = 36.0
COIN_TRACK_MAX_AREA = 2400.0
COIN_TRACK_EAT_Y_MARGIN = 6.0
COIN_TRACK_BOTTOM_TOO_CLOSE_ROWS = 8.0

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
SEG_DEBUG_DRAW_CANDIDATE_PATHS = False
SEG_DEBUG_DRAW_BOUNDARIES = True
SEG_DEBUG_DRAW_MERGE_GUIDE = False
SEG_DEBUG_LEFT_PATH_COLOR = (255, 255, 0)
SEG_DEBUG_RIGHT_PATH_COLOR = (0, 200, 255)
SEG_DEBUG_CANDIDATE_PATH_THICKNESS = 1
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
SEG_DEBUG_FORK_DIVIDER_COLOR = (0, 255, 0)
SEG_DEBUG_FORK_DIVIDER_THICKNESS = 1
SEG_DEBUG_MERGE_GUIDE_COLOR = (255, 255, 255)
SEG_DEBUG_MERGE_GUIDE_THICKNESS = 2
SEG_DEBUG_COIN_PATH_ENABLED = True
SEG_DEBUG_COIN_PATH_COLOR = (0, 255, 255)
SEG_DEBUG_COIN_PATH_DOT_RADIUS = 4
SEG_DEBUG_COIN_BOTTOM_STRICT_LINE_ENABLED = True
SEG_DEBUG_COIN_BOTTOM_STRICT_LINE_COLOR = (0, 0, 255)
SEG_DEBUG_COIN_BOTTOM_STRICT_LINE_THICKNESS = 1
SEG_DEBUG_PIP_DIVISOR = 3
SEG_DEBUG_PIP_BORDER_COLOR = (255, 255, 255)
SEG_DEBUG_PIP_BORDER_THICKNESS = 1
SEG_DEBUG_DRAW_BIRD_VIEW = False
SEG_DEBUG_DRAW_PLANNING_POINTS = True

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

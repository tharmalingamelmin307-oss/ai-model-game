# config.py
"""项目全局配置.

这个文件只放“参数”和“静态常量”，尽量不放会改变主流程控制结构的业务逻辑。
目标是让调参工作集中在这里完成：

1. 模型路径和输入尺寸改动时，不需要翻业务线程代码
2. 板卡现场调参时，可以快速看懂每个参数的作用和风险
3. README 里描述的系统行为，可以直接映射到这里的配置项
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# 参数分区索引
# ---------------------------------------------------------------------------
# 1. 共享内存与网页推流
# 2. 模型路径
# 3. NPU 核心分配
# 4. 目标检测类别定义
# 5. 输入尺寸与基础预处理参数
# 6. 岔路口判断参数
# 7. 汇合判断参数
# 8. 固定赛道宽度表
# 9. 下位机串口、速度与转向控制参数
# 10. 主流程运行时参数
# 11. 场景停车、行人、交通灯与 OCR 参数
# 12. 路径搜索、稳定与调试参数
# 13. 车辆避障、金币规划与可视化参数


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
SEG_MODEL = str(PROJECT_ROOT / "models/seg/segv5/segv5_416x160_argmax_rk3588_int8.rknn")

# 目标检测模型路径。
# 当前使用的是 PP-YOLOE 的 RKNN 版本，输出后处理由 modules/detector.py 负责。
# 如果替换模型，除了改这里，通常还要同步检查:
# - YOLO_SIZE
# - CLASS_NAMES
# - detector.py 的输出解析逻辑
YOLO_MODEL = str(PROJECT_ROOT / "models/det/dev6/ppyoloe_merged_512x384_split_rk3588_int8.rknn")

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
# 分割线程占用的 NPU 核列表。路牌前只启用这组核心。
SEG_CORES = [0]

# 语义路牌结果出来后仍保持单核 Seg。实测双 Seg worker 会拖低帧率。
SEG_CORES_AFTER_SIGN = [0]

# YOLO 检测线程绑定的 NPU 核。
# 这里保持原先稳定可用的单核配置；部分板端 RKNNLite/模型组合不支持直接传多核 mask。
YOLO_CORE = 2

# YOLO 多 worker 配置。
# 不把单个 RKNN runtime 绑到多核，而是启动多个单核 YOLODetector 实例。
# 如果双 worker 不稳定，把这里改回 [YOLO_CORE] 即可回到单 worker。
YOLO_CORES = [1, 2]

# 语义路牌结果出来后仍保持 YOLO 双 worker。
YOLO_ACTIVE_WORKERS_AFTER_SIGN = 2

# OCR 识别线程绑定的 NPU 核。
# OCR 不再启动时常驻初始化，只有路牌面积达标、任务进队后才初始化。
# 触发 OCR 时会短时间和 YOLO 争用 Core1。
REC_CORE = 1

# 路牌 OCR 正在运行时，是否暂停 YOLO 检测。
# True: OCR 期间 YOLO 不跑推理，避免双核 YOLO 抢 NPU；OCR 结束后自动恢复。
# False: OCR 和 YOLO 并行跑，检测连续性更好，但 OCR 期间可能互相抢核。
YOLO_PAUSE_DURING_OCR = True

# YOLO 因 OCR 暂停的最长时间，单位秒。
# 超过这个时间还没收到 OCR 完成信号，就自动恢复 YOLO，避免检测 FPS 长时间为 0。
YOLO_PAUSE_OCR_TIMEOUT = 1.5


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

# 普通语义路牌在进入 OCR 前，框的最小面积阈值，单位是 TARGET_RES 坐标系像素面积。
# 这个值比单独宽高更灵活，能更直接表达“这块牌子整体够不够大”。
# 调大:
# - 误判更少
# - 但远处牌子更不容易被识别
# 调小:
# - 更早开始识别
# - 但远距离误判风险会上升
# detv3 输入从 detv2 的 768x576 换到 512x384，面积类门槛按输入面积比例缩放。
# 50 * 100 * (512 * 384) / (768 * 576) = 2222。
OCR_MIN_SIGN_BOX_AREA = 6200

# sign 在进入 OCR 前，四周需要保留的最小边距比例。
# 例如 0.03 表示检测框四边都要距离画面边界至少 3% 的宽/高。
# 这样可以尽量避开“框看起来够大，但其实有一部分已经贴边截断”的情况。
OCR_SIGN_EDGE_MARGIN_RATIO = 0.05

# 语义路牌停车触发比普通 OCR 更严格：框离边缘太近时不停车，避免牌子被截断还触发采样。
SIGN_LLM_TRIGGER_EDGE_MARGIN_RATIO = 0.015

# sign 停车采样时，把 YOLO 路牌框按宽高向外扩展的比例，用来收集同一块牌子上的多行 OCR 文本。
SIGN_OCR_MATCH_EXPAND_RATIO = 0.20

# OCR 识别结果的最小平均置信度阈值。
# 低于这个值的 OCR 文本会在进入主逻辑前直接丢弃，避免低分脏结果参与。
OCR_MIN_SCORE = 0.50

# 语义路牌大模型判定。
# sign 面积达到阈值且不贴边后先停车，停车期间连续收集若干次 OCR 结果，再一次性发给千帆。
# 是否启用“停车采样 OCR + 千帆综合判定”的语义路牌流程。
# False 时会回到更简单的 OCR 文本直接生效逻辑。
SIGN_LLM_ENABLED = True
# 触发语义路牌停车采样的 sign 框面积阈值，单位是 TARGET_RES 坐标系像素面积。
# 它会和 SIGN_LLM_TRIGGER_EDGE_MARGIN_RATIO 同时满足后才停车。
SIGN_LLM_TRIGGER_AREA = 13000
# 停车后希望采集的有效 OCR 样本数量。
# 收满后会提交给千帆；如果超时，也可能提前提交已有样本。
SIGN_LLM_OCR_SAMPLES = 5
# 最少有效样本数。当前主流程仍倾向收满 SIGN_LLM_OCR_SAMPLES；
# 这个值保留给采集失败/策略调整时作为下限参考。
SIGN_LLM_MIN_VALID_SAMPLES = 3
# 停车采集 OCR 的最长等待时间，单位秒。
# 到时后即使没收满样本，也会尝试 force 提交。
SIGN_LLM_COLLECT_TIMEOUT = 3.0
# 千帆 API 请求超时时间，单位秒。
# 太小可能导致正常网络波动被判失败；太大会让车辆等待更久。
SIGN_LLM_API_TIMEOUT = 10.0
# 千帆结果允许回写的最大帧龄。
# 如果结果返回时视觉帧已经过去太久，就丢弃，避免旧路牌影响新路口。
SIGN_LLM_RESULT_MAX_AGE_FRAMES = 1500
# 语义路线完成后，连续看到多少帧单路特征才释放 WAIT_SIGN_GONE/路线锁定。
SIGN_ROUTE_SINGLE_ROAD_EXIT_FRAMES = 20
# 按语义路牌选定方向后，进入岔路区域至少保持方向的时间，单位秒。
SIGN_ROUTE_MIN_FORK_HOLD_SECONDS = 5.0
# 按语义路牌选定方向后，最长保持该路线选择的时间，单位秒。
# 超过后即使单路释放条件没完全满足，也会避免永久锁住。
SIGN_ROUTE_MAX_DRIVE_HOLD_SECONDS = 10.0

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
# }
CLASS_MIN_SCORES = {
    "car": 0.50,
    "person": 0.50,
    "door": 0.50,
    "stone": 0.50,
    "sign": 0.50,
    "start": 0.50,
    "stop": 0.50,
}

# 类别级最大面积比例过滤，比例基准是最终输出坐标系画面面积。
# 用于压掉小目标类别上异常大的误框，参数对齐当前 detv3 测试脚本。
YOLO_MAX_AREA_RATIO_BY_CLASS = {
    "sign": 0.16,
    "start": 0.16,
    "stop": 0.16,
}

# 检测框贴边判定的边距比例。
# 用于异常大框过滤：靠边太近的大框通常更像截断/误检。
YOLO_EDGE_MARGIN_RATIO = 0.02
# 贴边框面积占比阈值。
# 如果检测框贴边且面积超过这个比例，就认为更像异常大框并过滤。
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
# 岔路口判断参数
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
# 中间缺口张开过程至少要持续多少行。
FORK_INNER_OPEN_MIN_ROWS = 5
# 张开段内左右内边界之间的 gap 总增长量下限。
FORK_INNER_OPEN_MIN_GAP_GROWTH = 14.0
# 张开段内每一侧内边界向外移动的最小总量。
FORK_INNER_OPEN_MIN_SIDE_GROWTH = 5.0
# 相邻采样行里被认为“有效张开”的最小单步增量。
FORK_INNER_OPEN_MIN_STEP_GAIN = 1.0
# gap 正向增长的行数下限，防止只靠一两次跳变误判。
FORK_INNER_OPEN_MIN_POSITIVE_GAP_ROWS = 3
# 左/右侧各自正向张开的行数下限。
FORK_INNER_OPEN_MIN_POSITIVE_SIDE_ROWS = 2
# 允许局部回退的最大单步量；超过则认为不是连续张开过程。
FORK_INNER_OPEN_MAX_STEP_REGRESSION = 3.0
# 张开段中允许跳过/缺失的最大行数。
FORK_INNER_OPEN_MAX_MISS_ROWS = 2
# Y 岔路分叉点以下必须有公共主干 mask 支撑。
# 从分叉点到底部中点拉线，沿线附近多数行应能命中 mask，否则认为这条分叉线是悬空误判。
FORK_TRUNK_SUPPORT_CHECK_ENABLED = True
# 检查公共主干支撑时，沿分界线左右各看的半径。
FORK_TRUNK_SUPPORT_RADIUS = 5
# 从分叉点到底部的支撑命中比例下限。
FORK_TRUNK_SUPPORT_MIN_RATIO = 0.55
# 公共主干连续缺失超过多少行就判为悬空误判。
FORK_TRUNK_SUPPORT_MAX_MISS_ROWS = 18
# 公共主干至少检查多少行才认为这个验证有意义。
FORK_TRUNK_SUPPORT_MIN_ROWS = 18


# ---------------------------------------------------------------------------
# 汇合判断参数
# ---------------------------------------------------------------------------
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
# 触发汇合检测的宽行阈值；一行最左白点到最右白点超过该宽度，认为可能有汇合/贴边宽带。
MERGE_GUIDE_MIN_ROW_WIDTH = 286.0
# 宽行/贴边行需要连续出现多少行，才允许进入汇合尖角搜索。
MERGE_GUIDE_MIN_WIDE_ROWS = 4
# 汇合塌陷侧内边界的最小横向收口量。
MERGE_GUIDE_MIN_SIDE_DELTA = 13.0
# 汇合塌陷侧内边界需要有足够斜率/锐度，避免门或远处贴边造成的平缓漂移误判。
MERGE_GUIDE_MIN_INNER_ANGLE_DEG = 10.0
# 汇合内边界最大单步变化占总塌陷量的比例下限，用来要求“尖锐度”。
MERGE_GUIDE_MIN_INNER_SHARPNESS = 0.10
# 是否要求塌陷侧上方存在贴边行，减少普通宽弯误判为汇合。
MERGE_GUIDE_REQUIRE_EDGE_ABOVE_INNER = True
# 开启 REQUIRE_EDGE_ABOVE 后，上方至少需要多少贴边行。
MERGE_GUIDE_MIN_EDGE_ABOVE_ROWS = 2
# 塌陷侧成立后，对侧边界从底到顶允许的总漂移量。
MERGE_GUIDE_OPPOSITE_MAX_DRIFT = 16.0
# 塌陷侧成立后，对侧可信边界允许的逐行最大跳变。
# 例如左侧塌陷时，右侧边界必须连续稳定，不能中途突然横跳。
MERGE_GUIDE_OPPOSITE_MAX_STEP_JUMP = 10.0
# 汇合尖角 run 中允许中断的最大行数。
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
# 汇合 guide line 画入搜索 mask 时使用的线宽。
MERGE_GUIDE_LINE_THICKNESS = 2
# 汇合状态机：连续命中若干帧才进入补线；进入后等底部赛道宽度稳定恢复再退出。
# 汇合特征连续命中多少帧才正式进入补线状态。
MERGE_STATE_CONFIRM_FRAMES = 3
# 汇合特征命中统计窗口，单位秒；窗口内累计命中 CONFIRM_FRAMES 次即确认。
MERGE_STATE_CONFIRM_WINDOW_SECONDS = 1
# 汇合确认过程中允许偶发漏检多少帧，避免 1/0/1/1 这种抖动把确认计数清零。
MERGE_STATE_MISS_TOLERANCE_FRAMES = 2
# 退出补线时检查底部多少行的总白区宽度。
MERGE_STATE_EXIT_BOTTOM_ROWS = 5
# 汇合补线进入后至少保持多久，单位秒；在这之前即使满足退出条件也不退出。
MERGE_STATE_MIN_HOLD_SECONDS = 2.0
# 底部总白区宽度低于该阈值时，认为更像恢复成单路。
MERGE_STATE_EXIT_WIDTH_THRESH = 340.0
# 退出条件连续满足多少帧才真正退出补线状态。
MERGE_STATE_EXIT_CONFIRM_FRAMES = 2
# 退出补线时检查“不再贴边”的 y 范围上界。
MERGE_STATE_EXIT_NO_EDGE_Y_TOP = 10
# 退出补线时检查“不再贴边”的 y 范围下界。
MERGE_STATE_EXIT_NO_EDGE_Y_BOTTOM = 140

# 贴边侧八邻域方向特征：作为汇合检测的额外 OR 条件。
# 从右侧底部种子所在的八连通边缘块中，按从上往下的连续八邻域生长方向记录方向码。
# 是否启用贴边侧八邻域方向模式检测。
MERGE_EDGE_TRACE_ENABLED = True
# 贴边方向检测的 y 搜索范围。
MERGE_EDGE_TRACE_SCAN_Y_TOP = 10
MERGE_EDGE_TRACE_SCAN_Y_BOTTOM = 130
# 判断右侧贴边的距离阈值：右边界距离画面右边小于该值时算贴边。
MERGE_EDGE_TRACE_TOUCH_DISTANCE = 10
# 至少多少行右侧贴边，才启用连续方向 walk。
MERGE_EDGE_TRACE_MIN_TOUCH_ROWS = 3
# 从最底部贴边行继续向下偏移多少行取起点，再沿边缘向上爬。
MERGE_EDGE_TRACE_START_BELOW_ROWS = 20
# 连续八邻域 walk 的安全上限；实际会在满足特征后提前停止。
MERGE_EDGE_TRACE_WALK_MAX_STEPS = 260
# 连续方向低频调试打印。0 表示关闭；开大一点避免终端 I/O 拖慢帧率。
MERGE_EDGE_TRACE_WALK_DEBUG_INTERVAL = 0
# 调试打印中最多输出多少个方向码。
MERGE_EDGE_TRACE_DEBUG_MAX_DIRS = 96
# 调试打印中最多输出多少段 run。
MERGE_EDGE_TRACE_DEBUG_MAX_RUNS = 32
# 汇合口方向特征：4 长段 -> 5/6 过渡 -> 6/7 延伸。
MERGE_EDGE_TRACE_MIN_LEFT_RUN = 12
MERGE_EDGE_TRACE_MIN_TURN_RUN = 4
MERGE_EDGE_TRACE_MIN_DOWN_RUN = 12
# 汇合口方向特征匹配时，每段内部允许少量杂向跳点。
MERGE_EDGE_TRACE_PATTERN_MAX_NOISE = 3
# walk 过程中每隔多少步检查一次是否已满足特征。
MERGE_EDGE_TRACE_MATCH_CHECK_STEP = 8


# ---------------------------------------------------------------------------
# 固定赛道宽度表
# ---------------------------------------------------------------------------
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

# 会被映射到分割平面里的“规划相关类别”。
# car 用于避车状态机，stone 用于分支选择，其它类别预留给场景逻辑。
PLANNING_CLASS_NAMES = (
    "car",
    "person",
    "stone",
    "door",
)

# ---------------------------------------------------------------------------
# 下位机串口、速度与转向控制参数
# ---------------------------------------------------------------------------
# 下位机串口设备名。
# 如果板卡串口号变了，这里要跟着改。
SERIAL_PORT = '/dev/ttyS2'

# 串口波特率。
# 必须和下位机固件配置一致，否则会通信异常或乱码。
BAUD_RATE = 115200

# 串口线程最终允许输出的目标速度范围。
# 当前并不是直接发电机 PWM，而是发一个速度档位：
# - CONTROL_MIN_SPEED: 常规最低巡航速度
# - CONTROL_MAX_SPEED: 直道或轻弯时允许的最高速度
CONTROL_MIN_SPEED = 60
CONTROL_MAX_SPEED = 60

# 用单一转向控制量做动态降速时的增益。
# 设为 0 表示关闭“打角越大就降速”的策略。
STEER_SIGNAL_SPEED_GAIN = 0.0

# 是否启用速度档位平滑。
CONTROL_SPEED_SMOOTH_ENABLED = True
# 速度上升时每个控制周期最多增加多少档，避免突然加速。
CONTROL_SPEED_MAX_STEP_UP = 1
# 速度下降时每个控制周期最多减少多少档，保留弯道/停车响应速度。
CONTROL_SPEED_MAX_STEP_DOWN = 2

# ---------------------------------------------------------------------------
# 转向公共参数: 舵机最终输出限制
# ---------------------------------------------------------------------------
# 这些参数不属于算法 A 或算法 B，而是两个算法最终都会经过的舵机输出层。

# 舵机中心值。
# 这是“车身理论正前方”对应的 PWM。
# 当前 750 已确认机械中直；除非重新装舵机/连杆，否则不把它当控制参数来调。
# SERVO_CENTER = 736
SERVO_CENTER = 770

# 舵机安全最小/最大 PWM。
# 用于硬限制输出，避免控制算法在极端情况下打到危险位置。
# SERVO_MIN, SERVO_MAX = 590, 910
# SERVO_MIN, SERVO_MAX = 596, 876
SERVO_MIN, SERVO_MAX = 630, 910

# 舵机输出低通滤波。作用在最终 servo_pwm 上，专门压车跑起来时的小幅高频抖动。
# - EMA_ALPHA 越大越稳，但响应越慢；0 表示不滤波，0.35~0.65 常用。
# - DEADBAND_PWM 表示新旧 PWM 差值小于该值时不更新，避免舵机追 1~2 个 PWM 的噪声。
# - MAX_STEP 表示每个串口周期最多变化多少 PWM，0 表示不限制。
SERVO_OUTPUT_FILTER_ENABLED = False
SERVO_OUTPUT_EMA_ALPHA = 0.0
SERVO_OUTPUT_DEADBAND_PWM = 0
SERVO_OUTPUT_MAX_STEP = 0

# 当前启用的转向控制器。
# - "weighted_slope": 算法 A，原始稳定算法。把路径点到底部中点的斜率做远近加权平均。
# - "stanley_band": 算法 B，按 STANLEY_* 前视行计算 e / 两点 psi / 两点前馈。
# - "control_c": 算法 C，线性 PD + 航向抑制: Kp*e + Kd*de - Kyaw*psi。
# STEER_CONTROL_MODE = "weighted_slope"
STEER_CONTROL_MODE = "stanley_band"
# STEER_CONTROL_MODE = "control_c"


# ---------------------------------------------------------------------------
# 转向算法 A: weighted_slope 参数
# ---------------------------------------------------------------------------
# 下面这组只服务算法 A。
# 当 STEER_CONTROL_MODE = "weighted_slope" 时，它们作为主转向算法使用。
#
# 算法公式：
#   slope = (path_x - image_bottom_center_x) / max(image_bottom_y - path_y, STEER_SIGNAL_MIN_DY)
#   weight = path_y ** STEER_SIGNAL_ROW_WEIGHT_GAMMA
#   p = sum(slope * weight) / sum(weight) * STEER_SIGNAL_NORMALIZED_SCALE
#   steer_signal = p + Kd * d(p_ema)
#
# A 算法 steer_signal 到舵机 PWM 的映射增益。
# 注意：B 算法正常工作时不用它；B 算法使用 STANLEY_PWM_GAIN。
# 调大:
# - 舵机转向更积极
# - 但更容易抖或打满
# 调小:
# - 舵机更稳
# --------------------------------------
# - 但可能转不过弯
STEER_SIGNAL_PWM_GAIN = 0.02

# 计算“点到底部中点连线斜率”时使用的最小纵向间距。
# 作用是防止路径底部附近的点因为 dy 过小，把控制量瞬间放得过大。
STEER_SIGNAL_MIN_DY = 8.0
# 远近权重指数。越大，越强调靠近图像底部/车身近处的路径点；
# 越小，远处路径点占比越高。归一化不会抹掉远近信息，主要由这个指数保留远近差异。
STEER_SIGNAL_ROW_WEIGHT_GAMMA = 1.2
# 归一化控制量缩放。归一化后原始 steer_signal 常为个位数，
# 这里把它放大到更接近旧版累计控制量的显示和 PWM 调参量级。
STEER_SIGNAL_NORMALIZED_SCALE = 2800.0
# A 算法输出端 D 系数，作用在 EMA 后 steer_signal 的帧间变化量上。
# 默认关闭；想试 A+PD 时先从 0.05 ~ 0.25 小步加。
STEER_SIGNAL_D_GAIN = 0
# D 项使用前先对 A 的 steer_signal 做 EMA 平滑。数值越大越稳，但 D 项反应越慢。
STEER_SIGNAL_D_EMA_ALPHA = 0.2
# A 算法航向角前馈。用路径远/近两行的 x 差估计路径朝向，提前给一点舵。
# 这项只做小前馈，不替代 P/D；太大会让直道受远处线噪声影响而左右飘。
STEER_SIGNAL_HEADING_FF_GAIN = 0.0
# 航向前馈自己的 EMA 平滑。越大越稳但更慢；0 表示不平滑。
STEER_SIGNAL_HEADING_FF_EMA_ALPHA = 0.5
# 航向前馈取样行，SEG_SIZE 坐标里 y 越小表示看得越远。
# far 看弯道趋势，near 看车前路径；两者 x 差 / y 差得到近似航向斜率。
STEER_SIGNAL_HEADING_FF_FAR_Y = 35.0
STEER_SIGNAL_HEADING_FF_NEAR_Y = 85.0
# A 算法普通巡线时只使用这段 y 行范围内的路径点。
# 这两个值是 SEG_SIZE 坐标里的 y 行号；y 越小表示看得越远。
# 如果中线最上端低于 SAMPLE_ROW_MIN，会额外补一个 SAMPLE_ROW_MIN 行的点，x 使用最上端点。
WEIGHTED_SLOPE_SAMPLE_ROW_MIN = 10.0
WEIGHTED_SLOPE_SAMPLE_ROW_MAX = 90.0


# ---------------------------------------------------------------------------
# 转向算法 B: 前视行 Stanley 参数
# ---------------------------------------------------------------------------
# B 算法按图中公式计算:
#   delta = atan(k * e / (v_s + k_soft)) + Kd * de + g_psi * psi_e + g_ff * psi_ff
# 这里不做逆透视，仍工作在 SEG_SIZE 图像坐标里:
# - e: 控制路径在 STANLEY_LOOKAHEAD_Y 这一行相对车身中线的横向误差，单位: pixel
# - de: e 经过 EMA 后的帧间变化量，单位: pixel/frame
# - psi_e: 控制路径两点之间的航向角，单位: rad
# - psi_ff: 更远两点之间的航向角前馈，替代原来的二次拟合曲率
# 因为速度暂时只有编码器档位，v_s 先用 STANLEY_SPEED_ESTIMATE 这个调参量。
# B 算法专用 PWM 映射增益。这里写成独立数值，不引用 A 的 STEER_SIGNAL_PWM_GAIN。
STANLEY_PWM_GAIN = 0.015

# B 算法旧的“边界中点路径”开关。
# 当前主流程不再使用，仅保留给外部脚本兼容读取。
STANLEY_USE_BOUNDARY_MIDPOINTS = False
# 横向误差前视行，SEG_SIZE 坐标系。图像 y 越小表示看得越远。
STANLEY_LOOKAHEAD_Y = 100.0
# 航向角只看近中距离区域，负责抑制当前车头附近的左右摆。
# TOP 是画面更上方/更远处，BOTTOM 是画面更下方/更近处。
STANLEY_HEADING_Y_TOP = 60.0
STANLEY_HEADING_Y_BOTTOM = 100.0
# 前馈角单独看更远区域，负责提前感知弯道趋势；不要和航向角共用同一段区域。
STANLEY_FF_Y_TOP = 10.0
STANLEY_FF_Y_BOTTOM = 30.0
# 旧参数名保留给外部脚本兼容，主逻辑优先读取上面的 TOP/BOTTOM。
STANLEY_HEADING_LOOKAHEAD_Y = 70.0
STANLEY_HEADING_FAR_Y = STANLEY_HEADING_Y_TOP
STANLEY_HEADING_NEAR_Y = STANLEY_HEADING_Y_BOTTOM
STANLEY_CURVATURE_LOOKAHEAD_Y = 30.0
STANLEY_FF_FAR_Y = STANLEY_FF_Y_TOP
STANLEY_FF_NEAR_Y = STANLEY_FF_Y_BOTTOM
# 横向误差优先使用拟合前中心点在前视行附近的平均值，减少拟合线底部失真影响。
STANLEY_LATERAL_AVG_HALF_WINDOW = 10.0
# 横向误差增益 k，控制 atan(k * e / (v_s + soft)) 的纠偏力度。
STANLEY_LATERAL_GAIN = 0.30
# 横向 D 系数 Kd，作用在 EMA 后横向误差的帧间变化量 de 上。
# 默认关闭；想试 B+d 时先从很小值开始，例如 0.040-0.045。
STANLEY_LATERAL_D_GAIN = 0.022
# D 项使用前先对 e 做 EMA 平滑。数值越大越稳，但 D 项反应越慢。
STANLEY_LATERAL_D_EMA_ALPHA = 0.1
# 航向误差增益 g_psi。
STANLEY_HEADING_GAIN = 0.25
# 航向误差 psi 的 EMA 平滑。只影响 STANLEY_HEADING_GAIN 非 0 时的航向项。
# 调大：航向项更稳、更不追拟合线小抖；过大则航向抑制反应变慢。
STANLEY_HEADING_EMA_ALPHA = 0.5
# 两点角度前馈增益 g_ff。替代原来的曲率前馈，减少曲线拟合不稳定造成的左右飘。
STANLEY_CURVATURE_FF_GAIN = 0.01
# 轴距 L，单位 m。保留兼容旧配置；当前两点角度前馈不再使用它。
STANLEY_WHEELBASE_M = 0.2
# 速度估计 v_s。当前仅算法 B 使用。
STANLEY_SPEED_ESTIMATE = CONTROL_MAX_SPEED
# 横向误差软化常数。当前仅算法 B 使用。
STANLEY_SOFT = 60.0
# Stanley 两项相加后的整体输出缩放。
# 它决定最终 steer_signal 的量级，再由 STANLEY_PWM_GAIN 映射成 PWM。
# 增大：整体舵机幅度变大；减小：整体舵机幅度变小。
STANLEY_SIGNAL_SCALE = 10000.0
# B 算法最终输出符号。若画面中路径在左但车辆实际右打，设为 -1.0。
STANLEY_OUTPUT_SIGN = 1.0

# 控制路径至少需要的点数；不足时当前算法输出 0，不切换到其它控制器。
STANLEY_MIN_FIT_POINTS = 3


# ---------------------------------------------------------------------------
# 转向算法 C: 线性 PD + 航向抑制参数
# ---------------------------------------------------------------------------
# B/C 航向角计算方式。True 时用最小二乘直线 x=a*y+b 的斜率 a 算航向；
# False 时用二次拟合曲线在 HEADING_LOOKAHEAD_Y 的局部切线算航向。
PATH_HEADING_LINEAR_FIT_ENABLED = True
# B/C 横向误差 e 是否使用最终滤波后的控制路径。
# True: e、航向和调试紫线使用同一条路径，岔路/汇合后不容易 raw 路径和滤波路径打架。
# False: e 优先使用拟合前原始中心点，反应更快但分叉后可能短暂取到不一致的路径。
PATH_LATERAL_USE_FILTERED_PATH = True

# C 算法: 小弯/大弯参数自动过渡控制器。
#
# 它不是硬切状态机，而是每帧算一个 curve_level:
# - curve_level = 0: 完全用“小弯参数”
# - curve_level = 1: 完全用“大弯参数”
# - curve_level = 0.5: 小弯/大弯参数各取一半
#
# 线性插值公式:
#   实际参数 = 小弯参数 + curve_level * (大弯参数 - 小弯参数)
#
# 控制输出:
#   steer_signal = 横向纠偏 + 抗摆 + 近处顺弯 + 远处提前顺弯
#
# 正负号说明:
# - e > 0 表示路径中心在图像右侧，控制量按当前符号约定增大。
# - psi / psi_ff 是路径相对图像竖直方向的航向角；符号和 Stanley B 保持一致。
#
# 调参优先级建议（当前 SERVO_CENTER=750 已确认机械中直，不把它当控制参数来调）:
# 1. 先只调 CONTROL_C_PWM_GAIN，让总体打角不过大也不过软。
# 2. 再调下面“小弯参数/大弯参数”里的横向纠偏力度。
# 3. 最后小幅调 HEADING/FF，负责弯里顺滑，不要让它们压过横向 e。

# C 算法专用 PWM 映射增益，只影响最终舵机幅度，不改变 control 内部比例。
# 调大：同样 steer_signal 下舵机打得更大；调小：舵机更温和。
CONTROL_C_PWM_GAIN = 1.0

# 横向误差 e 的取样行，SEG_SIZE 坐标系里 y 越小表示看得越远。
# 取小一点：提前看弯，反应更早，但可能更抖/更受远处误差影响。
# 取大一点：看近处，贴近车前实际位置，但高速和大弯可能反应慢。
CONTROL_C_LOOKAHEAD_Y = 105.0

# 航向误差 psi 的近中距离取样行。
# TOP 更远，BOTTOM 更近；两点角度用于判断车头附近路径朝向。
CONTROL_C_HEADING_Y_TOP = 55.0
CONTROL_C_HEADING_Y_BOTTOM = 105.0
# 前馈航向 psi_ff 的远处取样行，用来提前感知连续弯。
CONTROL_C_FF_Y_TOP = 10.0
CONTROL_C_FF_Y_BOTTOM = 35.0
# 旧参数名保留给外部脚本兼容读取。
CONTROL_C_HEADING_LOOKAHEAD_Y = CONTROL_C_HEADING_Y_TOP

# 横向误差 e 不是直接取拟合曲线，而是在 CONTROL_C_LOOKAHEAD_Y 附近取拟合前中心点平均。
# 这里是半窗口高度，5 表示取 y±5 行内的中心点平均。
# 调大：e 更稳，但会变钝；调小：e 更灵敏，但更容易抖。
CONTROL_C_LATERAL_AVG_HALF_WINDOW = 10.0

# 自动判断“小弯还是大弯”的灵敏度。
# 调小：更容易进入大弯参数。
# 调大：更不容易进入大弯参数。
CONTROL_C_CURVE_FULL_HEADING_RAD = 1.10
CONTROL_C_CURVE_FULL_DELTA_RAD = 0.85
# curve_level 的平滑。越大越丝滑，但进入/退出大弯参数会慢一点。
CONTROL_C_CURVE_LEVEL_EMA_ALPHA = 0.5

# ---------------- C 小弯参数 ----------------
# 小弯横向纠偏力度。车偏离中线时，负责拉回来。
CONTROL_C_LATERAL_GAIN_STRAIGHT = 0.32
# 小弯抗摆力度。主要压来回摆，太大会细碎抖。
CONTROL_C_LATERAL_D_GAIN_STRAIGHT = 0.10
# 小弯近处顺弯力度。看车前路径朝向，辅助转向。
CONTROL_C_HEADING_GAIN_STRAIGHT = 1.5
# 小弯远处提前顺弯力度。0 表示小弯时不提前抢方向。
CONTROL_C_FF_GAIN_STRAIGHT = 0.0

# ---------------- C 大弯参数 ----------------
# 大弯横向纠偏力度。大弯里通常要比小弯更积极。
CONTROL_C_LATERAL_GAIN_CURVE = 0.46
# 大弯抗摆力度。大弯出弯时用来压过冲。
CONTROL_C_LATERAL_D_GAIN_CURVE = 0.16
# 大弯近处顺弯力度。大弯里帮车头顺着路径方向走。
CONTROL_C_HEADING_GAIN_CURVE = 4.0
# 大弯远处提前顺弯力度。负责提前看弯，太大容易提前拐过头。
CONTROL_C_FF_GAIN_CURVE = 2.0

# 旧参数名保留给外部脚本兼容读取。
CONTROL_C_LATERAL_GAIN = CONTROL_C_LATERAL_GAIN_STRAIGHT
CONTROL_C_LATERAL_D_GAIN = CONTROL_C_LATERAL_D_GAIN_STRAIGHT
CONTROL_C_HEADING_GAIN = CONTROL_C_HEADING_GAIN_STRAIGHT
CONTROL_C_FF_GAIN = CONTROL_C_FF_GAIN_STRAIGHT

# D 项使用前先对 e 做 EMA 平滑。数值越大，越相信上一帧，de 越平滑。
# 调大：D 项更稳、更不抖，但反应更慢。
# 调小：D 项更灵敏，但更容易把图像/路径微小变化放成舵机抖动。
CONTROL_C_LATERAL_D_EMA_ALPHA = 0.55

# 航向误差 psi 的 EMA 平滑。只影响 CONTROL_C_HEADING_GAIN 非 0 时的航向项。
# 调大：航向项更稳、更不追拟合线小抖；过大则航向抑制反应变慢。
CONTROL_C_HEADING_EMA_ALPHA = 0.6

# C 算法最终输出符号。若路径在右侧但车辆实际左打，设为 -1.0。
CONTROL_C_OUTPUT_SIGN = 1.0

# 拟合线至少需要的路径点数；不足时 control_c 输出 0，不会自动切到其它控制器。
CONTROL_C_MIN_FIT_POINTS = 3

# C 调参打印。试车时打开，终端会周期性输出:
# pwm/steer/e/de/psi/psi_ff/curve_level/当前插值后的参数。
CONTROL_C_DEBUG_LOG_ENABLED = True
CONTROL_C_DEBUG_LOG_INTERVAL = 0.5


# ---------------------------------------------------------------------------
# 转向模式增益
# ---------------------------------------------------------------------------
# 无目标控制增益：只在没有金币、没有避障车时乘到 steer_signal 上。
# 可用于补偿无目标控制行段变短、归一化后转向偏软等情况。
STEER_SIGNAL_NO_TARGET_GAIN = 1.0
# 避障车控制增益：只在 car 避障 active 时生效。
# 如果中心偏移已经足够但舵机反应偏小，优先调大这个值。
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
# - sign_llm_*: 大面积语义路牌停车、多次 OCR、千帆判定状态
# - person_stop_active: 当前是否已经进入“行人强制停车”状态
# - person_*: 行人停车/放行判定用的最近状态
# - debug_keyboard_*: 终端键盘调试发车/停车状态
# - actual_servo_pwm: 当前串口线程真正准备下发的舵机 PWM
# - target_speed: 当前串口线程真正准备下发的目标速度档位

# 调试发车/停车总开关。默认先停车，避免程序一启动就直接给速度。
DEBUG_DRIVE_CONTROL_ENABLED = True
DEBUG_DRIVE_INITIAL_STOPPED = True

# 终端键盘监听开关。VSCode Remote 终端 + AI 插件在板端内存紧张时不稳定，
# 默认关闭终端 raw keyboard，优先使用网页预览页面的 B/E 按键控制。
DEBUG_KEYBOARD_DRIVE_ENABLED = False
DEBUG_KEYBOARD_DRIVE_INITIAL_STOPPED = True
DEBUG_KEYBOARD_DRIVE_START_KEY = "b"
DEBUG_KEYBOARD_DRIVE_STOP_KEY = "e"
DEBUG_KEYBOARD_DRIVE_POLL_INTERVAL = 0.05

DEFAULT_CONTROL_DATA = {
    "steer_signal": 0.0,
    "turn_intent": -1,
    "turn_intent_fid": -1,
    "sign_llm_stop_active": False,
    "sign_llm_collecting": False,
    "sign_llm_waiting_result": False,
    "sign_llm_completed_hold": False,
    "sign_llm_samples": [],
    "sign_llm_started_at": None,
    "sign_llm_frame_id": -1,
    "sign_llm_ocr_inflight": False,
    "sign_llm_ocr_inflight_started_at": None,
    "sign_llm_result": "",
    "sign_llm_error": "",
    "sign_route_state": "IDLE",
    "sign_route_choice": 0,
    "post_sign_phase": False,
    "sign_route_locked_rect": None,
    "sign_route_drive_started_at": None,
    "sign_route_fork_entered_at": None,
    "sign_route_single_road_frames": 0,
    "sign_route_api_submitted": False,
    "person_stop_active": False,
    "person_bottom_y": None,
    "person_bottom_center_x": None,
    "person_bottom_right_x": None,
    "person_bottom_area": None,
    "person_dist_to_bottom": None,
    "person_car_on_left": False,
    "person_left_boundary_x": None,
    "person_right_boundary_x": None,
    "person_road_center_x": None,
    "person_clear_line_x": None,
    "person_clear_line_side": "",
    "person_stop_cutoff_y": None,
    "person_stop_event": "",
    "person_move_direction": 0,
    "person_missing_started_at": None,
    "person_stop_started_at": None,
    "person_stop_max_released": False,
    "person_clear_frames": 0,
    "person_miss_frames": 0,
    "person_last_frame_id": -1,
    "person_last_bottom_center_x": None,
    "debug_keyboard_enabled": False,
    "debug_keyboard_stop_active": False,
    "debug_keyboard_message": "",
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
# 如果你发现整机“反应慢但 FPS 看起来还行”，先不要盲目把这些值调大。
# 队列变大通常只会让显示和控制更滞后。
# 分割主控链路队列容量，只保留最新帧，避免控制滞后。
SEG_QUEUE_MAXSIZE = 1
# YOLO 检测链路队列容量，只保留最新帧，避免检测结果过旧。
YOLO_QUEUE_MAXSIZE = 1

# YOLO 投帧降频。Seg 是主控链路，每帧都跑；YOLO 只需要更新目标状态，
# 不必和 Seg 抢每一帧的 NPU/CPU 时间。
# 1 表示每帧都投；10 表示每 10 帧投 1 帧。
YOLO_PRODUCER_FRAME_INTERVAL = 1
# OCR 队列容量。OCR 慢于检测，允许保留少量任务，但不应堆积太多旧帧。
OCR_QUEUE_MAXSIZE = 2

# Seg 流水线模式。
# 开启后分割链路拆成:
# - NPU 推理线程: 从 seg_queue 取图，只负责输出 mask
# - 后处理线程: 取最新 mask，负责路径搜索 / 控制量 / 渲染
# 这样吞吐接近 max(推理耗时, 后处理耗时)，代价是控制结果通常会对应上一帧。
SEG_PIPELINE_ENABLED = True
# Seg 流水线后处理队列容量，通常保持 1，只处理最新推理结果。
SEG_PIPELINE_QUEUE_MAXSIZE = 1

# 帧率统计刷新周期，单位秒。
# 值越小，页面上的 FPS 数字更新越灵敏，但波动也会更明显；
# 值越大，显示更平滑，但更不容易立刻看出性能抖动。
FPS_STATS_UPDATE_INTERVAL = 1.0

# Seg 阶段耗时诊断日志。
# 开启后每隔一段时间打印 inference / search / fit / render / total，用来定位掉帧瓶颈。
SEG_PROFILE_LOG_ENABLED = False
# Seg profile 日志节流间隔，单位秒。
SEG_PROFILE_LOG_INTERVAL = 2.0

# 主流程运行时耗时诊断日志。
# 开启后会额外打印采集预处理、Seg 推理线程等待、发布、MJPEG 编码等耗时。
# 用来判断页面 FPS 是卡在输入来帧、模型推理、后处理发布还是网页推流。
MAIN_PROFILE_LOG_ENABLED = False
# 主流程 profile 日志节流间隔，单位秒。
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

# 路牌达到 OCR 识别条件、任务真正入队时的日志节流。
LOG_INTERVAL_OCR_ENTER = 2.0

# OCR 原始结果调试日志节流。
# 现场排查“到底读到了什么 / 为什么没有读到文字”时可以调小到 0。
LOG_INTERVAL_OCR_RAW = 0.5

# LEFT / RIGHT 语义路牌生效时的日志节流。
LOG_INTERVAL_TURN_INTENT = 2.0

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
PERSON_CLASS_NAME = "person"

# 当上面的类别名在 CLASS_NAMES 里找不到时，回退使用的固定类别 id。
# 正常情况下这些值不应该生效；它们更像是一层容错保护。
PERSON_CLASS_ID_FALLBACK = 2

# 行人停车/放行逻辑。
# 触发不做路径 ROI 过滤；person 框底边足够靠近画面底部后停车观察。
# 当前策略: 先停车观察；确认行人沿某一方向稳定移动，并且底部中心跨过“行人放行线”后，再直接释放停车。
# 画面上先画“停车截至横线”，它直接对应 PERSON_STOP_TRIGGER_DIST。
# 竖向放行线后面再按调试需要打开。
# 行人框底边距离画面底部小于该值才触发停车，单位 TARGET_RES 像素。
PERSON_STOP_TRIGGER_DIST = 330
# 行人框面积至少达到该值，才允许触发行人停车，单位 TARGET_RES 像素面积。
PERSON_STOP_MIN_AREA = 7000
# 行人朝目标侧连续移动多少帧后，才允许从停车切到绕行。
PERSON_CLEAR_MOVE_FRAMES = 2
# 判定“行人横向移动”的最小底部中心 x 增量。
PERSON_CLEAR_MIN_MOVE_DX = 3.0
# 兼容旧参数名；如果外部脚本还在改旧名，也能继续生效。
PERSON_CLEAR_MIN_RIGHT_DX = PERSON_CLEAR_MIN_MOVE_DX
# 行人横向放行线相对画面中线的偏移量，单位 TARGET_RES 像素。
# 这条线先不默认绘制，留作后续调试用。
PERSON_CLEAR_LINE_OFFSET_X = 30.0
# 行人横向放行线在预览图上的颜色和粗细。
PERSON_CLEAR_LINE_COLOR = (0, 255, 255)
PERSON_CLEAR_LINE_THICKNESS = 2
# 行人停车截至横线在预览图上的颜色和粗细。
PERSON_STOP_CUTOFF_LINE_COLOR = (0, 255, 255)
PERSON_STOP_CUTOFF_LINE_THICKNESS = 2
# 竖向放行线是否默认绘制。
PERSON_DEBUG_DRAW_RELEASE_LINE = True
# 兼容旧参数名，保留给外部脚本读取；当前主逻辑不再依赖中心带状窗口。
PERSON_CLEAR_CENTER_WINDOW_X = 25.0
# 行人停车后连续漏检超过这个时间就放行，单位秒。
PERSON_STOP_MISSING_TIMEOUT_SECONDS = 2.0
# 兼容保留字段；当前普通行人停车不再按时间自动释放。
PERSON_STOP_MAX_SECONDS = 8.0
# 兼容保留字段；当前漏检放行按 PERSON_STOP_MISSING_TIMEOUT_SECONDS 计时。
PERSON_STOP_MISS_RELEASE_FRAMES = 3

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
YOLO_DEFAULT_BOX_COLOR = (0, 0, 255)
YOLO_SIGN_BOX_COLOR = (0, 255, 255)

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

# 页面状态面板与文字颜色。
# 这些颜色主要为了快速区分：
# - 常规文字
# - 强调值（如目标速度）
# - 强制停车提示
PREVIEW_PANEL_BG_COLOR = (0, 0, 0)
PREVIEW_PANEL_BORDER_COLOR = (0, 255, 255)
PREVIEW_TEXT_COLOR = (255, 255, 255)
PREVIEW_TEXT_ACCENT_COLOR = (0, 255, 255)
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
# 串口协议与线程节奏参数
# ---------------------------------------------------------------------------
# 串口超时时间，单位秒。
# 串口偶尔抖动时，这个值太大可能会拖慢控制循环；太小则更容易把一次短暂卡顿视为失败。
SERIAL_TIMEOUT = 0.1

# 串口数据包头尾。
# 只有在你同时修改上下位机通信协议时才需要调整。
SERIAL_PACKET_HEADER = (0xAA, 0x55)
SERIAL_PACKET_TAIL = (0x0D, 0x0A)

# 各线程等待/轮询节奏，单位秒。
# 这些值主要影响 CPU 占用、实时性和页面刷新感受：
# - 太小: 更灵敏，但更吃 CPU
# - 太大: 更省资源，但会更“顿”
# 串口控制线程循环间隔。0.003333 约等于 300Hz 下发频率。
CONTROL_LOOP_SLEEP = 1.0 / 300.0
# 共享内存无新帧时的轮询间隔。
SHM_FRAME_POLL_SLEEP = 0.002
# 共享内存连接失败后的重试间隔。
SHM_RETRY_SLEEP = 1.0
# 网页推流没有可用帧时的等待间隔。
VIDEO_FEED_IDLE_SLEEP = 0.01
# MJPEG 每帧推送后的主动 sleep，限制页面推流频率和编码压力。
VIDEO_FEED_FRAME_SLEEP = 0.02
# 启动共享内存/检测等普通线程之间的错峰等待。
STARTUP_SHARED_THREAD_SLEEP = 0.1
# 启动 Seg 线程之间的错峰等待，避免 RKNN 初始化同时抢资源。
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

# OCR 调试图保存。只在排查 rec 空文本时打开。
# 打开后会保存 OCR det 拉正后的 crop，不改变识别流程。
OCR_DEBUG_SAVE_EMPTY_CROPS = True
# OCR 调试图保存目录。
OCR_DEBUG_SAVE_DIR = str(PROJECT_ROOT / "debug_ocr")
# OCR 调试图最多保存多少张，避免长时间运行写爆磁盘。
OCR_DEBUG_SAVE_MAX_IMAGES = 30



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

# 相邻帧路径稳定约束总开关。
# 工作在 SEG_SIZE 路径平面里，用于抑制分割噪声或分叉候选切换导致的横跳。
SEG_PATH_STABILITY_ENABLED = True
# 最终输出路径每帧允许横向移动的最大像素量；设为 0 可关闭硬限幅。
SEG_PATH_MAX_FRAME_X_JUMP = 0.0
# 候选路径相对上一帧偏移越大，打分扣得越多。
SEG_PATH_TEMPORAL_SCORE_GAIN = 5.0
# 单点跳变量超过这个软阈值后，候选会受到额外重罚。
SEG_PATH_TEMPORAL_SOFT_MAX_JUMP = 32.0
# 超出软阈值部分的额外扣分权重。
SEG_PATH_TEMPORAL_EXCESS_SCORE_GAIN = 18.0
# 候选与上一帧路径至少重叠多少个采样点，才参与时域打分。
SEG_PATH_TEMPORAL_MIN_OVERLAP_POINTS = 4
# 当前帧没搜到路径时，短暂沿用上一帧路径的最大帧数。
SEG_PATH_HOLD_MISSING_FRAMES = 2

# 估计石头更靠近左/右分支时，左右候选路径至少要拉开这么多像素才认为可比较。
# 当前代码里石头主要还用于调试显示，这个参数暂时不直接影响正式分支选择。
STONE_BRANCH_MIN_SEP = 12

# 自底向上路径搜索参数。
# 这些值决定了 mask 搜索的采样密度、连通判定和候选路径数量上限。
# 如果分叉口容易漏掉某一支，或直道上路径抖动明显，优先看这里。
# 路径搜索沿 y 方向每次向上跳多少行。
# 调小更细致但更慢；调大更快但可能跳过窄分支。
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

# 终端打印当前赛道宽度的节流间隔，单位秒。
TRACK_WIDTH_LOG_INTERVAL = 1.5

# ---------------------------------------------------------------------------
# 车辆避障控制参数
# 这组参数工作在分割输入 `SEG_SIZE = 416x160` 的坐标系里。
# y 方向是 160 行高度，所有 `*_ROWS` 都是“离底部多少行”的意思。
# 当前策略用状态机锁定 car；进入避障后把控制基准线切到
# 左边界向中线内收的路径。看不见车后不立刻回中线，
# 而是先走 `CLEARING`，再把上一条绕车基准线逐步混回中线。
# ---------------------------------------------------------------------------
CAR_AVOIDANCE_ENABLED = True
# 锁定 car 后，普通避障实际循线基准: 左边界向中线方向内收多少像素。
CAR_AVOIDANCE_LEFT_BOUNDARY_INSET = 20.0
# car 底部中心距离画面底部超过这个行数时，不做内收，只保留跟踪锁定。
CAR_AVOIDANCE_START_BOUNDARY_ROWS = 110.0
# car 底部中心距离画面底部不超过这个行数时，切到近距离内收。
CAR_AVOIDANCE_NEAR_BOUNDARY_ROWS = 60.0
# 近距离内收量。
CAR_AVOIDANCE_NEAR_LEFT_BOUNDARY_INSET = 10.0
# car 跟踪锁定。锁定主要看车框底部中心点的连续性，面积只做异常框过滤。
# 连续命中后进入避障；短暂漏检会继续沿用锁定目标，超过允许帧数后进入 CLEARING。
# 新 car 目标需要连续命中多少帧才锁定。
CAR_AVOIDANCE_LOCK_HIT_FRAMES = 2
# 锁定目标和新检测框匹配的搜索半径。
CAR_AVOIDANCE_SEARCH_RADIUS = 48.0
# 目标漏检期间，搜索半径随漏检帧数增加的增益。
CAR_AVOIDANCE_SEARCH_RADIUS_MISS_GAIN = 16.0
# 锁定目标位置 EMA 平滑系数。
CAR_AVOIDANCE_TRACK_EMA_ALPHA = 0.65
# car 框短暂变小、被遮挡或漏检时，继续沿用最近一次锁定目标的帧数。
# 调大可避免太早回正；过大会让已经绕过车后继续偏左太久。
CAR_AVOIDANCE_MISS_FRAMES = 6
# car 检测最低置信度过滤；0 表示不额外过滤。
CAR_AVOIDANCE_MIN_SCORE = 0.0
# car 检测最大面积过滤；0 表示不额外过滤。
CAR_AVOIDANCE_MAX_AREA = 0.0
# 避障退出状态机。
# 车丢失后不立刻回正，而是先进入 CLEARING。
# CLEARING 里会先保留上一条左边界内收基准线，再慢慢回到正常巡线。
# 进入 CLEARING 前需要连续漏检多少帧。
CAR_AVOIDANCE_CLEARING_MISS_FRAMES = 3
# CLEARING 状态里绕车基准线衰减到结束需要多少帧。
CAR_AVOIDANCE_CLEARING_DECAY_FRAMES = 15
# CLEARING 初期保留原绕车基准线的比例。
CAR_AVOIDANCE_CLEARING_RESIDUAL_KEEP = 1.0
# 衰减残余低于该比例时认为回正完成。
CAR_AVOIDANCE_CLEARING_DONE_RESIDUAL = 0.06
# 车框还在但已经很贴底、贴右且高度较小时，直接进入近距离边界基准。
# 贴右下特殊规则的 car 框高度上限。
CAR_AVOIDANCE_FIXED_BOUNDARY_HEIGHT_THRESH = 60.0
# 贴右下特殊规则要求 car 离右边界的最大距离。
CAR_AVOIDANCE_FIXED_BOUNDARY_RIGHT_MARGIN = 10.0
# 贴右下特殊规则要求 car 离底边的最大距离。
CAR_AVOIDANCE_FIXED_BOUNDARY_BOTTOM_MARGIN = 10.0
# 主分割调试图绘制风格。
# 最终路径线颜色。
SEG_DEBUG_PATH_COLOR = (255, 0, 255)
# 最终路径线粗细。
SEG_DEBUG_PATH_THICKNESS = 2
# 是否绘制候选左右路径，用于排查分支选择。
SEG_DEBUG_DRAW_CANDIDATE_PATHS = False
# 是否绘制当前选中路径的左右边界。
SEG_DEBUG_DRAW_BOUNDARIES = True
# 是否绘制汇合补线引导线。
SEG_DEBUG_DRAW_MERGE_GUIDE = False
# 左候选路径颜色。
SEG_DEBUG_LEFT_PATH_COLOR = (255, 255, 0)
# 右候选路径颜色。
SEG_DEBUG_RIGHT_PATH_COLOR = (0, 200, 255)
# 候选路径线粗细。
SEG_DEBUG_CANDIDATE_PATH_THICKNESS = 1
# 左边界颜色。
SEG_DEBUG_LEFT_BOUNDARY_COLOR = (255, 255, 0)
# 右边界颜色。
SEG_DEBUG_RIGHT_BOUNDARY_COLOR = (0, 165, 255)
# 左右边界线粗细。
SEG_DEBUG_BOUNDARY_THICKNESS = 2
# 底部车身参考点颜色。
SEG_DEBUG_BOTTOM_MID_COLOR = (255, 255, 0)
# 底部车身参考点半径。
SEG_DEBUG_BOTTOM_MID_RADIUS = 4
# 兼容旧文件读取：coin 追踪/规划已删除，这些值只防止混版本运行时报缺配置。
COIN_PATH_ENABLED = False
COIN_PATH_ROI_BOTTOM_STRICT_ROWS = 0.0
SEG_DEBUG_COIN_PATH_ENABLED = False
SEG_DEBUG_COIN_PATH_COLOR = (0, 255, 255)
SEG_DEBUG_COIN_PATH_DOT_RADIUS = 4
SEG_DEBUG_COIN_BOTTOM_STRICT_LINE_ENABLED = False
SEG_DEBUG_COIN_BOTTOM_STRICT_LINE_COLOR = (0, 0, 255)
SEG_DEBUG_COIN_BOTTOM_STRICT_LINE_THICKNESS = 1
# Y 岔路分界线颜色。
SEG_DEBUG_FORK_DIVIDER_COLOR = (0, 255, 0)
# Y 岔路分界线粗细。
SEG_DEBUG_FORK_DIVIDER_THICKNESS = 1
# 汇合引导线颜色。
SEG_DEBUG_MERGE_GUIDE_COLOR = (255, 255, 255)
# 汇合引导线粗细。
SEG_DEBUG_MERGE_GUIDE_THICKNESS = 2
# 是否绘制 steer_signal 斜率累计实际使用的 y 区域，两条横线分别表示参与控制点的上下边界。
SEG_DEBUG_CONTROL_BAND_ENABLED = True
# steer_signal 控制区域横线颜色。
SEG_DEBUG_CONTROL_BAND_COLOR = (255, 0, 255)
# steer_signal 控制区域横线粗细。
SEG_DEBUG_CONTROL_BAND_THICKNESS = 2
# 分割调试图左上角文字的字号、位置与颜色。
# 这些信息主要用于现场快速确认：
# - Seg / YOLO FPS
# - 当前 steer_signal 与估算舵机 PWM
# - 石头分支判断的调试文本
# 调试文字字号。
SEG_DEBUG_TEXT_FONT_SCALE = 0.8
# 调试文字线宽。
SEG_DEBUG_TEXT_THICKNESS = 1
# Seg/Yolo FPS 文本位置。
SEG_DEBUG_TEXT_POS_FPS = (5, 18)
# steer_signal / PWM 文本位置。
SEG_DEBUG_TEXT_POS_CTRL = (5, 36)
# 石头分支判断文本位置。
SEG_DEBUG_TEXT_POS_STONE = (5, 54)
# 分叉/汇合调试文本位置。
SEG_DEBUG_TEXT_POS_BRANCH = (5, 72)
# 金币调试文本位置。
SEG_DEBUG_TEXT_POS_COIN = (5, 90)
# FPS 文本颜色。
SEG_DEBUG_TEXT_COLOR_FPS = (0, 255, 0)
# 控制量文本颜色。
SEG_DEBUG_TEXT_COLOR_CTRL = (0, 255, 255)
# 石头文本颜色。
SEG_DEBUG_TEXT_COLOR_STONE = (0, 200, 255)
# 分叉/汇合文本颜色。
SEG_DEBUG_TEXT_COLOR_BRANCH = (255, 200, 0)
# 金币文本颜色。
SEG_DEBUG_TEXT_COLOR_COIN = (0, 255, 255)

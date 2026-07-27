# Aero-Twin 5.25

基于 RK3588 的自动驾驶视觉主程序。当前版本优先追求的是板端实时性、链路稳定性和现场可调性，而不是离线条件下的最高精度。

这套系统的当前分工是：

- `Seg` 负责主控闭环，持续输出循迹控制量
- `YOLO` 负责环境目标检测和路牌框检测
- `OCR` 负责整图 `det + rec`；`sign` 只在停车采样状态下触发 OCR，`limit_sign` 链路保留但当前通过 `LIMIT_SIGN_ENABLED=False` 临时关闭
- 各线程都优先保“最新帧”，必要时主动丢掉旧任务，避免累积延迟

## 快速索引

- [config 参数分区索引](./config.py#L16)
- [下位机控制、速度与转向](./config.py#L547)
- [岔路口判断](./config.py#L316)
- [汇合判断](./config.py#L373)
- [固定宽度与规划类别](./config.py#L478)
- [路径搜索、稳定与调试](./config.py#L1007)
- [车辆避障与分割画面](./config.py#L1131)
- [README 参数说明](#参数调试索引)
- [README 常用调参入口](#常用调参入口)
- [项目结构](#项目结构)
- [当前系统总览](#当前系统总览)

## 项目结构

```text
.
├── config.py              # 核心参数、模型路径、调参说明
├── main.py                # 主入口，线程调度、共享内存拉流、串口控制、网页推流入口
├── models/
│   ├── det/               # 目标检测 RKNN 模型
│   ├── ocr/               # OCR det / rec 模型与字典
│   └── seg/               # 分割 RKNN 模型
├── modules/
│   ├── debug_tools.py     # 网页预览、终端日志、性能打印和调试画面叠加
│   ├── detector.py        # PP-YOLOE 检测封装与后处理
│   ├── ocr_system.py      # OCR det + rec 封装
│   ├── path_controller.py # 图像路径点到 steer_signal 的 A/B/C 控制器
│   └── segmentor.py       # 分割、路径搜索、避车状态机
└── utils/
    └── image_proc.py      # OCR 文字框透视拉正工具
```

## 当前系统总览

### 1. 输入与线程结构

主流程由 `main.py` 驱动，当前线程分工如下：

1. `ai_producer_thread()`
   从共享内存 `shm_ar_video` 读取最新图像
2. `yolo_worker()`
   执行目标检测，并筛出值得触发 OCR 的 `sign / limit_sign`
3. `ocr_worker()`
   对整张 `TARGET_RES` 图执行 OCR `det + rec`，再把结果匹配回检测框
4. `seg_worker()`
   默认拆成 Seg 推理流水线和后处理流水线：当前线程做 NPU 推理，内部后处理线程做路径搜索、目标规划和控制量估计
5. `serial_control_thread()`
   把视觉结果转成速度与舵机命令发给下位机
6. Flask 推流线程
   将最终调试画面编码成 MJPEG 供网页预览

当前默认配置下：

- `SEG_PIPELINE_ENABLED=True`
- `SEG_CORES=[0]`
- `app.run(..., threaded=True)`

固定业务线程大致如下：

| 线程 | 数量 | 作用 |
|---|---:|---|
| 主线程 / Flask | 1 | 跑 Flask 服务 |
| `ai_producer_thread` | 1 | 从共享内存取图，分发给 Seg / YOLO |
| `serial_control_thread` | 1 | 下发速度和舵机命令 |
| `seg_worker` | 1 | Seg NPU 推理 |
| `postprocess_loop` | 1 | Seg 后处理、路径搜索、渲染；仅流水线模式开启 |
| `yolo_worker` | 1 | 目标检测 |
| `ocr_worker` | 1 | OCR det + rec |

所以：

- 流水线开启且不打开网页：固定约 `7` 个线程
- 流水线开启并打开一个 `/video_feed`：通常约 `8` 个或更多线程
- `SEG_PIPELINE_ENABLED=False` 时少一个 `postprocess_loop`
- 串行模式不打开网页：固定约 `6` 个线程
- 串行模式打开一个 `/video_feed`：通常约 `7` 个或更多线程

`/video_feed` 会多占线程，是因为 Flask 使用 `threaded=True`。浏览器访问主页后，图片流是一个持续不断的 MJPEG 长连接，请求线程会一直负责向浏览器发送 JPEG 帧；刷新页面、多个浏览器窗口或断线重连时，短时间内可能看到更多请求线程。

设计原则：

- `seg_queue` 和 `yolo_queue` 都只保留最新帧
- `seg_worker()` 内部的 `mask_queue` 也只保留最新 mask
- `ocr_queue` 只保留很少量的新任务，旧任务会被主动丢掉
- 目标是避免“上一帧还没算完，下一帧已经来了”的排队迟滞

### 2. NPU 核分配

默认配置见 `config.py`：

- `Seg -> Core 0`
- `OCR -> Core 1`
- `YOLO -> Core 2`

这么做的目的很直接：

- 分割是主控闭环，优先保证稳定
- OCR 不会反过来拖慢 YOLO
- YOLO 和 OCR 都不会直接抢占 Seg 的主链路资源

### 3. 坐标系约定

当前代码里有三套常见坐标系：

- 共享内存原图坐标系
  上游采集帧本来的分辨率
- `TARGET_RES`
  系统内部统一显示/检测坐标系，检测框、OCR 回写文字、网页叠框都在这里对齐
- `SEG_SIZE`
  分割与路径规划使用的模型输入坐标系。当前默认是 `416x160`，输入前先裁掉原图上半部分，只把下半图送进分割模型；历史 `320x320` 配置保存在 `320*320/` 目录中

关键约定：

- `modules/detector.py` 输出的检测框统一映射到 `TARGET_RES`
- `ocr_worker()` 跑 OCR 时使用的是同帧 `TARGET_RES` 大图
- `modules/segmentor.py` 内部会把检测框从 `TARGET_RES` 映射到 `SEG_SIZE`，供 `car / stone` 等规划逻辑使用

## 模块与链路说明

### 1. 共享内存拉流

`ai_producer_thread()` 会反复读取共享内存头部：

- `frame_id`
- `width`
- `height`

如果 `frame_id` 没变化，说明还是旧帧，就继续等待。  
如果来了新帧：

- 分割支路得到 `SEG_SIZE` 的 `RGB` 小图
- 检测支路得到 `YOLO_SIZE` 的 `BGR` 小图
- 同时保留一份 `TARGET_RES` 的 `BGR` 大图给 OCR 和网页可视化使用

### 2. YOLO 检测逻辑

`modules/detector.py` 当前只按 dev4 split RKNN 模型解析：

- `boxes`: `[N, 4]`，xyxy
- `scores`: `[N, num_classes]`
- Python 侧只喂 `0-255 RGB uint8`，mean/std 已固化在 RKNN 内部

检测后处理的当前真实行为：

- 先拿每个候选位置的最高分类得分
- 再按“每个类别自己的阈值”过滤
- 然后做按类别 NMS
- 再按类别最大面积比例和贴边大框规则过滤异常框
- 最终统一输出：

```python
{
    "rect": [x, y, w, h],
    "class_id": int,
    "class_name": str,
    "score": float,
}
```

注意：

- 当前不再兼容旧的多分支 / 单 tensor 检测模型输出
- 类别阈值统一由 `YOLO_CONF_THRES + CLASS_MIN_SCORES` 控制
- 异常大框过滤由 `YOLO_MAX_AREA_RATIO_BY_CLASS / YOLO_EDGE_*` 控制

### 3. OCR 触发逻辑

`YOLO` 检到 `sign / limit_sign` 后，不会立刻对每个框单独裁图识别。  
当前真实流程是：

1. 先判断这个牌子框值不值得触发 OCR
2. 如果值得，就把同帧 `TARGET_RES` 整图送给 `ocr_worker()`
3. `ocr_worker()` 对整图做一次 OCR `det + rec`
4. 再把 OCR 结果按中心点最近原则匹配回原来的牌子框

当前 OCR 触发门槛在 `main.py -> should_enqueue_ocr_job()`：

- `sign` 需要通过 `OCR_MIN_SIGN_BOX_AREA`
- `limit_sign` 需要通过 `OCR_MIN_LIMIT_SIGN_BOX_AREA`
- 两类都必须满足 `OCR_SIGN_EDGE_MARGIN_RATIO`
  也就是四周要离画面边界留出足够安全距离

这样做的目的：

- 太远太小的牌子不进 OCR，减少误判
- 明显贴边、可能已经被截断的牌子不进 OCR，减少半截字误识别

### 4. OCR det + rec 逻辑

`modules/ocr_system.py` 当前是标准的整图 `det + rec` 链路：

1. `run_text_detection()`
   在整张图上做 OCR det，得到文字四点框
2. `get_rotate_crop_image()`
   把文字四点框透视拉正
3. `run_single_crop()`
   把拉正后的文本图送入 rec 模型识别
4. `_decode()`
   用 CTC 规则解码成文本和平均置信度

当前 OCR 模块本身并不区分 `sign` 和 `limit_sign`。  
它只负责输出：

```python
{
    "points": np.ndarray,   # 原图四点框
    "text": str,
    "score": float,
}
```

类别语义是在 `main.py` 里处理的：

- `sign`
  主要关心 `LEFT / RIGHT`
- `limit_sign`
  先用通用 OCR 识别，再从结果里提取数字字符

### 5. `sign` 语义路牌与固定序列逻辑

`SIGN_ROUTE_DECISION_MODE` 控制岔路策略：

- `llm_once`: 第一圈没有路牌，不触发岔路路牌逻辑；第一次看到岔路牌即第二圈，路牌达标后停车采样 OCR，并把样本提交千帆；千帆返回 `LEFT / RIGHT` 后锁定本次方向；第三圈再遇到路牌时不 OCR、不请求千帆，直接对第二圈方向取反。
- `fixed_sequence`: 不启动路牌 OCR/千帆线程；默认只在同一时刻看见 `sign` 和分割 Y 岔时推进固定序列，且 `sign` 不看面积；第二圈按 `SIGN_ROUTE_FIXED_FIRST_CHOICE` 走，第三圈自动取反。

方向编码保持一致：`LEFT` 表示左侧外圈，`RIGHT` 表示右侧内圈。`SIGN_ROUTE_FIXED_FIRST_CHOICE` 默认是 `LEFT`，即第二圈左外圈、第三圈右内圈；改成 `RIGHT` 后顺序互换。

两种模式都复用原有 `CHOICE_READY -> IN_FORK -> WAIT_SIGN_GONE/IDLE` 的路线锁定和补线状态机，避免同一岔路内左右支路反复跳变。

`turn_intent` 会被 `segmentor.py` 在分叉路径选择时使用。分叉选择优先级是：

- 检测到 `stone` 时，优先绕开石头更接近的那一支。
- 没有石头干预时，使用千帆返回的 `turn_intent`。
- 千帆没有有效结果时，`turn_intent` 回到默认左支。

### 6. `limit_sign` 限速牌逻辑

注意：当前 `config.py` 中 `LIMIT_SIGN_ENABLED=False`，所以 `limit_sign` 不触发 OCR，也不会写入 `speed_limit`。下面记录的是限速链路重新开启后的保留逻辑。

限速牌当前逻辑分成两个阶段：

1. 先把 OCR 文本中的数字字符提出来
2. 在牌子还比较远时，按“不同数字”做历史聚合统计
3. 当牌子面积达到 `LIMIT_SIGN_APPLY_MIN_AREA` 后，不再继续 OCR
4. 直接从历史聚合结果里选最优候选写入 `speed_limit`

关键点：

- 历史统计不是记录每一条明细，而是只按数字累计：
  `count` 和 `score_sum`
- 平均置信度用 `score_sum / count` 计算
- 候选选择时优先比较 `count`
- 如果 `count` 一样，再比较平均置信度
- `LIMIT_SIGN_CONFIRM_FRAMES` 现在表示“稳定候选的最低累计次数门槛”
- `LIMIT_SIGN_APPLY_MIN_AREA` 决定什么时候从“继续观察”切到“正式生效”
- `LIMIT_SIGN_HISTORY_MAX_MISS_FRAMES` 用来限制历史结果保留时长，避免旧牌子串到下一块新牌子

另外还有一个实现细节：

- 真正写入的上限是 `识别数字 - 1`

例如历史统计最终选中了 `20`，系统真正写入的 `speed_limit` 是 `19`。  
这相当于留了一点安全余量。

### 7. 分割与路径规划逻辑

`modules/segmentor.py` 是当前主控核心，主要流程如下：

1. `infer_mask()` 对 `SEG_SIZE` 小图执行分割，得到二值赛道 `mask`
2. `seg_worker()` 把最新 `mask` 连同当时的 YOLO 框、`turn_intent` 放入内部 `mask_queue`
3. `postprocess_mask()` 从最新 `mask` 开始做路径规划
4. 先只对搜索用 `mask` 的底部局部做轻微膨胀，修补近车处的小断裂
5. 先检查全图是否连续出现若干行“主白区宽度足够大”
6. 只有满足这条宽带条件时，才全图搜索“单侧向下扩张的汇合尖角”
7. 如果命中汇合尖角，就按固定赛道宽度反推缺失侧边界，补出一条汇合引导线，并按单路模式继续搜索
8. 如果没有命中汇合尖角，再去全图扫描“中间缺口双边张开”的连续区域，判断是否为真正的 Y 型岔路
9. 一旦确认 Y 型岔路，就用“分叉点 -> 底部中点”的分界线把路面切成左右两大区域
10. 在整图或左右分区内，自底向上分层搜索候选路径
11. 路径必须从图像真实底部 `SEG_PATH_BOTTOM_TOUCH_HEIGHT` 行内起步，中途新冒出来但没接到底部的悬空候选会被丢弃
12. 对候选路径按长度、平滑度、通道宽度、局部中心偏离做打分
13. 如果检测到 `stone`，优先绕开石头更接近的那一支
14. 如果没有明确石头干预，则使用语义路牌状态机写入的 `turn_intent`；没有有效结果时默认偏向左支
15. 对最终路径做多项式拟合
16. 对拟合系数做 EMA 平滑
17. 结合避车状态机选择控制路径或避车基准线
18. 调用 `modules/path_controller.py`，按 `STEER_CONTROL_MODE` 将路径转换成单一 `steer_signal`
19. 调用 `modules/debug_tools.py` 渲染调试画线、文字和网页预览相关输出

当前输出：

- `steer_signal`
  单一转向控制量，后续由串口线程映射成舵机 PWM
- `ai_view`
  调试渲染图，后续会被放大回 `TARGET_RES`

转向控制器目前有三种，三套控制器互相独立，不会自动切换或回退：

- `weighted_slope`
  算法 A。对控制路径点计算“路径点到底部中点连线斜率”，再按行号做远近加权平均。这个模式简单、抗噪，适合作为基线对照。
- `stanley_band`
  算法 B。按前视行 Stanley 公式计算：
  `atan(k * e / (v_s + soft)) + g_psi * psi + g_ff * psi_ff`。
  当前普通巡线优先用处理后的左右边界中点做控制路径，航向和前馈都用两点之间的角度，不再依赖二次拟合曲率。
- `control_c`
  算法 C。当前默认模式。线性 PD + 航向抑制：
  `control = Kp * e + Kd * (e - e_last) - Kyaw * psi`。
  横向误差优先使用拟合前中心点在前视行附近的平均值，航向来自拟合线切线。

补充说明：

- `SEG_PIPELINE_ENABLED=True` 时，Seg 推理和上一帧 mask 后处理会重叠执行
- `SEG_PIPELINE_ENABLED=False` 时，会回到旧串行模式：推理完当前帧后，立刻在同一线程里处理当前帧
- 页面 `Seg FPS` 表示后处理线程实际产出控制量/预览画面的频率
- 终端 `SegProfile total / est` 表示单帧从开始推理到渲染完成的端到端耗时估算
- 当前岔路主判据已经不是“外边界断裂”，而是“中间缺口双边张开 + 缺口总体变大”
- 当前还额外支持一种“汇合引导线”补线逻辑：只有先看到全图连续宽带，再检测到单侧下扩尖角时，才会按单路模式补线，而不是直接切成岔路
- 当前代码里 `stone` 已经会参与左右分支选择，但还没有进一步写成更复杂的代价场避障
- 真正直接参与主分支选择的，当前是“候选路径打分 + `stone` 左右关系 + `turn_intent` 左右偏向覆盖”

### 8. 车辆避障逻辑

当前车辆控制有三套互相独立的 B 控制参数：

1. 普通巡线使用 `STANLEY_*`
2. `AVOIDING / CLEARING` 使用 `CAR_AVOIDANCE_STANLEY_*` 和 `CAR_AVOIDANCE_TARGET_SPEED`
3. 绕完 `POST_CAR_CONTROL_AFTER_CYCLES` 辆车且回正完成后，使用 `POST_CAR_STANLEY_*` 和 `POST_CAR_TARGET_SPEED`

避车状态机会先锁定同一辆车，把左边界作为控制路径；丢失车辆后先保持避车路径，再在 `CLEARING` 窗口逐步混回正常路径。第三套参数只在第二辆车的 `CLEARING` 完成、状态回到 `FOLLOW_LANE` 后启用，不会提前影响第二次避车。

### 9. 红绿灯与斑马线逻辑

当前交通灯停车链路是：

1. YOLO 检测 `zebra_crossing`
2. 把斑马线框底边当作停止线
3. 取距离画面底部最近的那条停止线
4. 如果当前灯色是 `red` 或 `yellow`
5. 并且停止线到底部的距离小于 `ZEBRA_STOPLINE_TRIGGER_DIST`
6. 则触发强制停车
7. 如果灯色变成 `green`
   就解除这个停车状态
8. 如果已经停车后连续 5 帧检测不到灯色，则进入后退找灯
9. 后退到停止线到底部距离达到 150 后停止后退，重新等待灯色

### 10. 行人停车逻辑

当前行人停车不走路径 ROI 过滤，触发方式和斑马线接近：

1. YOLO 检测 `person`
2. 取最靠近车身的行人框，也就是底边 `y` 最大的那个框
3. 如果行人框底边到画面底部距离小于 `PERSON_STOP_TRIGGER_DIST`
4. 再检查行人框面积是否大于 `PERSON_STOP_MIN_AREA`
5. 两个条件都满足才进入 `person_stop_active`，串口速度强制置零
6. 画面上先画一条“停车截至横线”，它对应 `PERSON_STOP_TRIGGER_DIST`
7. 停车期间继续观察行人框底部中心点判断横向运动方向
8. 若行人持续朝同一方向移动，并且对应侧底角越过中线偏移 `PERSON_CLEAR_LINE_OFFSET_X` 的放行线，连续满足 `PERSON_CLEAR_MOVE_FRAMES` 帧，则解除停车
9. 如果停车后连续 `PERSON_STOP_MISSING_TIMEOUT_SECONDS` 秒看不到行人，也解除停车
10. 放行后直接恢复正常寻中线
11. 当前行人逻辑只保留停车、观察、过线/漏检放行这一套状态机

关键点：

- 触发停车只看“底部是否靠近”，不先要求行人在路径 ROI 内
- 放行是有锁的：一旦满足条件并解除停车，同一轮靠近过程中不会在下一帧又重新停住
- 短暂漏检仍保持停车，连续漏检满 2 秒才放行
- 预览图上会先画出这条停车截至横线，竖向放行线后面再按调试需要打开
- 同一帧 YOLO 结果不会被流水线重复计入“连续左移帧”
- 页面上会显示 `STOP_BY_PERSON`，终端会打印行人底边距离、底边右端、左边界和连续左移放行帧数

### 11. 串口控速逻辑

`serial_control_thread()` 当前的速度逻辑分三层：

1. 先根据弯道程度算基础速度 `dynamic_target_speed`
2. 如果 `speed_limit` 已生效，则把它当作速度上限
3. 如果红/黄灯停车或行人停车条件满足，则强制把最终速度置零

可以把它理解成：

```text
target_speed = dynamic_target_speed
target_speed = min(target_speed, speed_limit)   # 若有限速
target_speed = 0                                # 若红/黄灯停车或行人停车成立
```

这意味着：

- 限速牌只是速度上限，不会破坏弯道自动减速
- 红黄灯停车和行人停车优先级高于限速和弯道速度

### 12. 结果回写与旧帧保护

当前系统对 OCR / LLM 结果做了多层保护，避免旧结果污染新状态：

- OCR 任务会带着对应的 `frame_id`
- 回写 `global_yolo_boxes / turn_intent / speed_limit` 前会再次核对帧号
- 语义路牌提交千帆后停止继续 `sign` OCR
- 同一块语义路牌只跑一次停车采样和千帆判定流程

作用是：

- 旧 OCR 结果即使晚到，也不容易把新状态覆盖回去
- 千帆失败不会反复停车重试，而是按石头优先 / 默认左路兜底放行
- 页面上显示的 OCR 文本与当前检测框更一致

### 13. 终端日志

当前终端默认只保留低频、条件触发型日志：

- 停车采样状态下，路牌达到 OCR 识别条件并真正入队
- OCR 最终识别结果（语义路牌文本 / 限速牌数字）
- 千帆任务开始、返回、最终 `LEFT / RIGHT` 结果
- 千帆失败后的石头优先 / 默认左路兜底
- 限速牌正式生效
- 红绿灯停车条件正式触发

此外，启动失败、线程异常、共享内存拉流异常、YOLO 解析异常这类错误会打印，但同一类错误只首报一次，不会持续刷屏。

## 页面预览会显示什么

网页预览里当前会叠加这些信息：

- 分割结果
- 规划线
- YOLO 检测框
- OCR 文本
- 斑马线停止线
- Seg / YOLO FPS
- 当前目标速度
- 当前限速值
- 当前交通灯状态
- 是否处于红黄灯停车状态

这张图的主要用途是快速定位问题是在：

- 上游没来帧
- 模型没出结果
- OCR 匹配错了
- 路径规划偏了
- 还是控制参数不合适

## 当前使用的模型

默认配置见 `config.py`：

- 分割模型：`models/seg/segv4/pipi416x160_77_1600_argmax_rk3588_int8.rknn`
- 检测模型：`models/det/dev4/ppyoloe_crn_m_100e_custom_7_5_512x384_split_rk3588_int8.rknn`
- OCR det 模型：`models/ocr/ppocrv4_det_int8.rknn`
- OCR rec 模型：`models/ocr/ppocrv4_rec_fp16.rknn`
- OCR 字典：`models/ocr/keys.txt`

切换模型时，至少要同步检查：

1. `SEG_MODEL / YOLO_MODEL / OCR_DET_MODEL_PATH / REC_MODEL_PATH`
2. `SEG_SIZE / YOLO_SIZE / REC_HEIGHT / REC_WIDTH`
3. `CLASS_NAMES`
4. `modules/detector.py` 的输出解析逻辑
5. `DICT_PATH`

## 参数调试索引

`config.py` 是唯一集中调参入口。下面按功能分区列出当前参数用途；实际运行值以 `config.py` 为准。

快速定位：

- [基础运行与模型](#基础运行与模型)
- [类别、检测与 OCR 触发](#类别检测与-ocr-触发)
- [输入尺寸与预处理](#输入尺寸与预处理)
- [岔路口判断](#岔路口判断)
- [汇合判断](#汇合判断)
- [固定宽度与规划类别](#固定宽度与规划类别)
- [下位机控制、速度与转向](#下位机控制速度与转向)
- [运行态、队列与日志](#运行态队列与日志)
- [场景停车与 OCR 后处理](#场景停车与-ocr-后处理)
- [路径搜索、稳定与调试](#路径搜索稳定与调试)
- [车辆避障与分割画面](#车辆避障与分割画面)

### 基础运行与模型

| 参数 | 用途 |
|---|---|
| `PROJECT_ROOT` | 项目根目录，所有模型路径都基于它拼接 |
| `SHM_NAME` | 上游共享内存名称，必须和采集端一致 |
| `SHM_HEADER_SIZE` | 共享内存头部协议长度，当前固定 16 字节 |
| `STREAM_PORT` | Flask 网页预览端口 |
| `JPEG_QUALITY` | MJPEG 预览 JPEG 质量，影响清晰度、CPU 和带宽 |
| `SEG_MODEL` | 分割 RKNN 模型路径 |
| `YOLO_MODEL` | YOLO/PP-YOLOE 检测 RKNN 模型路径 |
| `OCR_DET_MODEL_PATH` | OCR 检测模型路径 |
| `REC_MODEL_PATH` | OCR 识别模型路径 |
| `DICT_PATH` | OCR 字典路径 |
| `SEG_CORES` | Seg 推理绑定的 NPU 核列表 |
| `YOLO_CORE` | YOLO 推理绑定的 NPU 核 |
| `REC_CORE` | OCR 推理绑定的 NPU 核 |

### 类别、检测与 OCR 触发

| 参数 | 用途 |
|---|---|
| `CLASS_NAMES` | 检测模型类别顺序，必须和模型导出顺序一致 |
| `SIGN_CLASS_ID` | 普通语义路牌类别 id |
| `LIMIT_SIGN_CLASS_ID` | 限速牌类别 id |
| `LIMIT_SIGN_ENABLED` | 是否启用限速牌 OCR 与限速生效链路 |
| `OCR_MIN_SIGN_BOX_AREA` | sign 进入 OCR 的最小框面积 |
| `OCR_MIN_LIMIT_SIGN_BOX_AREA` | limit_sign 进入 OCR 的最小框面积 |
| `LIMIT_SIGN_APPLY_MIN_AREA` | 限速牌从历史观察切到正式生效的面积门槛 |
| `OCR_SIGN_EDGE_MARGIN_RATIO` | 路牌框距离画面边缘的安全边距比例 |
| `OCR_MIN_SCORE` | OCR 文本进入主逻辑的最低平均置信度 |
| `SIGN_ROUTE_DECISION_MODE` | 岔路策略：`llm_once` 为第二圈 OCR/千帆、第三圈取反；`fixed_sequence` 为固定序列 |
| `SIGN_ROUTE_SKIP_FIRST_PASS` | 固定序列模式下是否忽略第一次分割岔路事件；路牌模式第一次看到路牌不跳过 |
| `SIGN_ROUTE_FIXED_FIRST_CHOICE` | 固定序列第二圈方向，默认 `LEFT` 表示左侧外圈 |
| `SIGN_ROUTE_FIXED_REQUIRE_SIGN` | 固定序列是否要求同一时刻看见 `sign` 和 Y 岔；开启时 sign 不看面积 |
| `SIGN_LLM_ENABLED` | `llm_once` 模式下是否启用语义路牌停车、多次 OCR、千帆综合判定 |
| `SIGN_LLM_FORK_POINT_TRIGGER_ROWS` | Y 岔特征点距分割平面底部多少行内提前停车 OCR，默认 `100`；仍要求 sign 不贴边 |
| `SIGN_LLM_TRIGGER_AREA` | sign 面积达到多少且不贴边后触发停车采样，当前默认 16000 |
| `SIGN_LLM_OCR_SAMPLES` | 停车后收集多少条有效 OCR 结果再发给千帆 |
| `SIGN_LLM_MIN_VALID_SAMPLES` | 保留参数；当前语义路牌流程要求收满 `SIGN_LLM_OCR_SAMPLES` 条有效样本 |
| `SIGN_LLM_COLLECT_TIMEOUT` / `SIGN_LLM_API_TIMEOUT` | OCR 采集超时保留参数 / 千帆 API 超时 |
| `LIMIT_SIGN_CONFIRM_FRAMES` | 限速候选至少累计多少次才优先生效 |
| `LIMIT_SIGN_HISTORY_MAX_MISS_FRAMES` | 限速历史在连续丢失后保留的最大帧数 |
| `ZEBRA_STOPLINE_EXTEND_RATIO` | 斑马线停止线左右延长比例 |
| `ZEBRA_STOPLINE_TRIGGER_DIST` | 停止线距离底部多近时触发红黄灯停车 |
| `TRAFFIC_LIGHT_RECOVER_MISS_FRAMES` | 红绿灯停车后连续丢灯多少检测帧才后退找灯 |
| `TRAFFIC_LIGHT_RECOVER_TARGET_DIST` | 后退找灯时停止线距离底部达到多少像素后停止后退 |
| `TRAFFIC_LIGHT_RECOVER_RELEASE_TIMEOUT` | 进入恢复链路后仍检测不到灯时，等待多少秒后直接放行 |
| `TRAFFIC_LIGHT_RECOVER_BACK_SPEED` | 后退找灯时下发的倒车速度档位 |
| `YOLO_CONF_THRES` | YOLO 全局默认置信度阈值 |
| `YOLO_NMS_THRES` | YOLO NMS 阈值 |
| `CLASS_MIN_SCORES` | 各类别单独置信度阈值 |
| `YOLO_MAX_AREA_RATIO_BY_CLASS` | 各类别异常大框面积比例上限 |
| `YOLO_EDGE_MARGIN_RATIO` | 判断检测框贴边的边距比例 |
| `YOLO_EDGE_TOUCH_MAX_AREA_RATIO` | 同时贴边且面积过大时的过滤阈值 |
| `YOLO_BOX_MIN_SIZE` | 检测框最小宽高 |
| `YOLO_MAX_DETS` | 单帧最多保留检测框数量 |
| `YOLO_PRE_NMS_TOPK_PER_CLASS` | NMS 前每类最多保留候选数 |

### 输入尺寸与预处理

| 参数 | 用途 |
|---|---|
| `TARGET_RES` | 系统统一显示/检测坐标系尺寸 |
| `YOLO_SIZE` | YOLO 模型输入尺寸 |
| `SEG_SIZE` | Seg 模型输入与路径规划坐标系尺寸 |
| `SEG_INPUT_CROP_TOP_RATIO` | 分割输入裁掉原图上半部分的比例 |
| `REC_HEIGHT` | OCR rec 输入高度 |
| `REC_WIDTH` | OCR rec 输入最大宽度 |
| `MASK_ALPHA` | 分割 mask 叠加透明度 |
| `SEG_DEBUG_DRAW_MASK` | 是否在预览上绘制整块 mask |
| `SEG_PREVIEW_OVERLAY_DIFF_THRESH` | Seg 渲染图叠回主预览时的差异阈值 |

### 岔路口判断

| 参数 | 用途 |
|---|---|
| `FORK_MASK_MIN_BRANCH_PIXELS` | 单个分支最少白像素宽度 |
| `FORK_MASK_GAP_THRESH` | 白区断开多宽才认为分成两支 |
| `FORK_MIN_BRANCH_SEP` | 左右分支最小横向距离 |
| `FORK_BOTTOM_BAND_HEIGHT` | 底部宽度判定使用的行带高度 |
| `FORK_SCAN_Y_TOP` / `FORK_SCAN_Y_BOTTOM` | Y 岔路扫描 y 范围 |
| `FORK_INNER_OPEN_MIN_ROWS` | 分叉缺口张开需要持续的最少行数 |
| `FORK_INNER_OPEN_MIN_GAP_GROWTH` | 中间缺口整体增长门槛 |
| `FORK_INNER_OPEN_MIN_SIDE_GROWTH` | 单侧内边界张开门槛 |
| `FORK_INNER_OPEN_MIN_STEP_GAIN` | 单步张开的最小有效增量 |
| `FORK_INNER_OPEN_MIN_POSITIVE_GAP_ROWS` | 缺口正增长的最少行数 |
| `FORK_INNER_OPEN_MIN_POSITIVE_SIDE_ROWS` | 单侧正增长的最少行数 |
| `FORK_INNER_OPEN_MAX_STEP_REGRESSION` | 允许的单步回退上限 |
| `FORK_INNER_OPEN_MAX_MISS_ROWS` | 张开过程中允许缺失的最大行数 |
| `FORK_TRUNK_SUPPORT_*` | Y 岔路分叉点到底部主干线附近的 mask 支撑约束 |
| `FORK_BOUNDARY_WIDTH_ENABLED` | 岔路区域是否按目标方向外侧边界补另一侧边界；当前只在无路牌路线任务时启用，默认左路会拟合左边界斜率后平移补右边 |
| `FORK_BOUNDARY_HOLD_SECONDS` | 无路牌岔路补线方向的保持时间；当前为 0.5 秒 |
| `BOUNDARY_PATCH_HALF_WIDTH_RATIO` | 岔路/汇合补线时控制中线距离可信边界的半宽比例；当前为 0.9 |

### 汇合判断

| 参数 | 用途 |
|---|---|
| `SEG_SCENE_SCAN_BOTTOM_HEIGHT` | 汇合/场景扫描只看底部多少行 |
| `MERGE_GUIDE_SCAN_Y_TOP` / `MERGE_GUIDE_SCAN_Y_BOTTOM` | 汇合宽带前置扫描范围 |
| `MERGE_GUIDE_FREE_SCAN_Y_TOP` / `MERGE_GUIDE_FREE_SCAN_Y_BOTTOM` | 底部自由汇合扫描范围 |
| `MERGE_GUIDE_MIN_ROW_WIDTH` | 认为场景足够宽的最小行宽 |
| `MERGE_GUIDE_MIN_WIDE_ROWS` | 宽行连续命中的最少行数 |
| `MERGE_GUIDE_MIN_SIDE_DELTA` | 汇合尖角侧边界变化门槛 |
| `MERGE_GUIDE_MIN_INNER_ANGLE_DEG` | 汇合内边界最小角度 |
| `MERGE_GUIDE_MIN_INNER_SHARPNESS` | 汇合内边界锐度门槛 |
| `MERGE_GUIDE_REQUIRE_EDGE_ABOVE_INNER` | 是否要求尖角上方仍有边缘支持 |
| `MERGE_GUIDE_MIN_EDGE_ABOVE_ROWS` | 上方边缘支持最少行数 |
| `MERGE_GUIDE_OPPOSITE_MAX_DRIFT` | 可信对侧边界最大漂移 |
| `MERGE_GUIDE_OPPOSITE_MAX_STEP_JUMP` | 可信对侧边界逐行最大跳变 |
| `MERGE_GUIDE_MAX_MISS_ROWS` | 汇合特征允许缺失的最大行数 |
| `MERGE_GUIDE_EXTEND_TOP_ROWS` / `MERGE_GUIDE_EXTEND_BOTTOM_ROWS` | 汇合补线向上/向下覆盖行数 |
| `MERGE_GUIDE_LINE_Y_MIN` / `MERGE_GUIDE_LINE_Y_MAX` | 汇合补线允许的 y 范围 |
| `MERGE_GUIDE_LINE_MIN_GAP` | 补线与对侧边界的最小保护间距 |
| `MERGE_GUIDE_LINE_THICKNESS` | 汇合补线绘制粗细 |
| `BOUNDARY_PATCH_HALF_WIDTH_RATIO` | 汇合 guide 与补线重算边界时使用的半宽比例；当前为 0.9 |
| `MERGE_STATE_CONFIRM_FRAMES` | 汇合连续命中多少帧才进入状态 |
| `MERGE_STATE_EXIT_BOTTOM_ROWS` | 汇合退出时检查底部多少行 |
| `MERGE_STATE_EXIT_WIDTH_THRESH` | 底部宽度恢复到多少才允许退出 |
| `MERGE_STATE_EXIT_CONFIRM_FRAMES` | 汇合退出条件连续满足帧数 |
| `MERGE_STATE_EXIT_NO_EDGE_Y_TOP` / `MERGE_STATE_EXIT_NO_EDGE_Y_BOTTOM` | 汇合补线侧原始边界无贴边退出检查范围；补左线只看左边，补右线只看右边 |
| `MERGE_EDGE_TRACE_*` | 贴边侧八邻域方向特征，用作汇合判断的额外 OR 条件 |

### 固定宽度与规划类别

| 参数 | 用途 |
|---|---|
| `SEG_FIXED_WIDTH_SOURCE_SIZE` | 固定宽度表来源坐标系 |
| `SEG_FIXED_WIDTH_SOURCE_CROP_TOP_RATIO` | 固定宽度表来源裁剪比例 |
| `SEG_FIXED_WIDTHS_320` | 原始固定赛道宽度表 |
| `SEG_FIXED_WIDTHS_320_SMOOTH` | 平滑后的固定赛道宽度表，优先用于补线 |
| `PATH_LOCK_FORK_MIN_SEP` | 判断左右候选已经明显分叉的最小横距 |
| `PLANNING_CLASS_NAMES` | 会映射进分割平面参与规划/场景判断的类别 |

### 下位机控制、速度与转向

| 参数 | 用途 |
|---|---|
| `SERIAL_PORT` | 下位机串口设备名 |
| `BAUD_RATE` | 串口波特率 |
| `CONTROL_MIN_SPEED` / `CONTROL_MAX_SPEED` | 串口线程目标速度范围 |
| `STEER_SIGNAL_SPEED_GAIN` | 根据转向幅度动态降速的增益，0 表示关闭 |
| `CONTROL_SPEED_SMOOTH_ENABLED` | 是否启用目标速度平滑 |
| `CONTROL_SPEED_MAX_STEP_UP` / `CONTROL_SPEED_MAX_STEP_DOWN` | 速度单帧最大上升/下降步长 |
| **公共舵机输出** |  |
| `SERVO_CENTER` | 舵机中位 PWM |
| `SERVO_MIN` / `SERVO_MAX` | 舵机 PWM 安全上下限 |
| `SERVO_OUTPUT_FILTER_ENABLED` | 是否启用最终舵机 PWM 输出滤波，缓和大幅打角冲击 |
| `SERVO_OUTPUT_EMA_ALPHA` | 舵机输出 EMA 平滑系数，0 表示不滤波 |
| `SERVO_OUTPUT_DEADBAND_PWM` | 舵机输出死区，小于该 PWM 差值时不更新 |
| `SERVO_OUTPUT_MAX_STEP` | 舵机每个控制周期最大 PWM 步长，0 表示不限制 |
| `STEER_CONTROL_MODE` | 转向控制器模式，`weighted_slope` 为算法 A，`stanley_band` 为算法 B，`control_c` 为算法 C |
| **算法 A: `weighted_slope`** | 原始稳定算法，把路径点到底部中点的斜率做远近加权平均 |
| `STEER_SIGNAL_PWM_GAIN` | `weighted_slope` 的 steer_signal 转舵机 PWM 增益 |
| `STEER_SIGNAL_MIN_DY` | 斜率计算最小纵向距离 |
| `STEER_SIGNAL_ROW_WEIGHT_GAMMA` | 路径点远近权重指数 |
| `STEER_SIGNAL_NORMALIZED_SCALE` | 归一化 steer_signal 的整体放大系数 |
| `WEIGHTED_SLOPE_SAMPLE_ROW_MIN` / `WEIGHTED_SLOPE_SAMPLE_ROW_MAX` | `weighted_slope` 普通巡线时使用的独立 y 行取样范围 |
| `PATH_HEADING_USE_TRUSTED_BOUNDARY` | B/C 的 `psi` 和 `psi_ff` 是否使用可信边界；默认左边界，右岔用右边界，等待路牌结果拉线时用中间拉线，汇合用真实对侧边界 |
| `PATH_LATERAL_USE_FILTERED_PATH` | B/C 的横向误差 `e` 是否使用拟合后路径；当前为 `False`，使用拟合前中点加权计算 |
| **算法 B: `stanley_band`** | 前视行 Stanley 公式：横向误差 + 两点航向角 + 两点角度前馈 |
| `STANLEY_PWM_GAIN` | `stanley_band` 专用 PWM 映射增益 |
| `STANLEY_USE_BOUNDARY_MIDPOINTS` | B 方案旧的边界中点开关，仅兼容保留，不再参与主流程 |
| `STANLEY_LOOKAHEAD_Y` | 算横向误差 `e` 的前视行，`SEG_SIZE` 坐标系里 y 越小看得越远 |
| `STANLEY_HEADING_Y_TOP` / `STANLEY_HEADING_Y_BOTTOM` | 算航向误差 `psi` 的近中距离区域 |
| `STANLEY_FF_Y_TOP` / `STANLEY_FF_Y_BOTTOM` | 算前馈角 `psi_ff` 的远处区域，应和航向区域分开 |
| `STANLEY_LATERAL_AVG_HALF_WINDOW` | 横向误差取拟合前中心点加权平均时的半窗口高度 |
| `STANLEY_LATERAL_GAIN` | Stanley 横向误差增益 |
| `STANLEY_HEADING_GAIN` | Stanley 航向误差增益 |
| `STANLEY_CURVATURE_FF_GAIN` | Stanley 两点角度前馈增益，保留旧名兼容配置 |
| `STANLEY_WHEELBASE_M` | 轴距，保留旧配置兼容；当前两点角度前馈不使用它 |
| `STANLEY_SPEED_ESTIMATE` / `STANLEY_SOFT` | Stanley 横向项分母里的速度估计和软化常数 |
| `STANLEY_SIGNAL_SCALE` | Stanley 输出整体缩放；越大舵机幅度越大 |
| `STANLEY_MIN_FIT_POINTS` | Stanley 控制路径最低点数；不足时当前模式输出 0，不切换到其它控制器 |
| **算法 C: `control_c`** | 线性 PD + 航向抑制：`Kp*e + Kd*de - Kyaw*psi` |
| `CONTROL_C_PWM_GAIN` | `control_c` 专用 PWM 映射增益；只影响最终舵机幅度，不改变 C 内部 P/D/航向比例 |
| `CONTROL_C_LOOKAHEAD_Y` | 横向误差 `e` 的取样行；y 越小看得越远，反应更早但可能更抖 |
| `CONTROL_C_HEADING_LOOKAHEAD_Y` | 航向误差 `psi` 的取样行；通常先和横向行一致，想提前抑制大弯可取更远 |
| `CONTROL_C_LATERAL_AVG_HALF_WINDOW` | 横向误差取拟合前中心点加权平均时的半窗口高度；越大越稳但越钝 |
| `CONTROL_C_LATERAL_GAIN` | 横向 P 系数 `Kp`；调大回中更快，过大容易左右摆 |
| `CONTROL_C_LATERAL_D_GAIN` | 横向 D 系数 `Kd`；调大压过冲/慢摆，过大容易细碎抖 |
| `CONTROL_C_LATERAL_D_EMA_ALPHA` | D 项前的横向误差 EMA 平滑；越大越稳但反应更慢 |
| `CONTROL_C_HEADING_GAIN` | 航向抑制系数 `Kyaw`，以 `-Kyaw*psi` 使用；只做阻尼，过大会和横向项打架 |
| `CONTROL_C_MIN_FIT_POINTS` | C 算法拟合最低点数；不足时当前模式输出 0，不切换到其它控制器 |
| **模式增益** | 普通巡线、car 等模式对最终控制量的额外修正 |
| `STEER_SIGNAL_NO_TARGET_GAIN` | 普通巡线模式控制增益 |
| `SERIAL_TIMEOUT` | 串口读写超时 |
| `SERIAL_PACKET_HEADER` / `SERIAL_PACKET_TAIL` | 串口协议包头包尾 |
| `CONTROL_LOOP_SLEEP` | 串口控制循环 sleep |
| `SHM_FRAME_POLL_SLEEP` | 共享内存轮询 sleep |
| `SHM_RETRY_SLEEP` | 共享内存重连 sleep |
| `VIDEO_FEED_IDLE_SLEEP` / `VIDEO_FEED_FRAME_SLEEP` | 网页推流空闲/帧间 sleep |
| `STARTUP_SHARED_THREAD_SLEEP` / `STARTUP_SEG_THREAD_SLEEP` | 启动时线程错峰 sleep |
| `FLASK_HOST` | Flask 监听地址 |
| `SUPPRESS_RKNN_INIT_OUTPUT` | 是否静默 RKNN 初始化日志 |

### 运行态、队列与日志

| 参数 | 用途 |
|---|---|
| `DEFAULT_CONTROL_DATA` | 主流程控制状态默认结构 |
| `DEFAULT_FPS_STATS` | FPS 统计默认结构 |
| `SEG_QUEUE_MAXSIZE` | Seg 输入队列容量 |
| `YOLO_QUEUE_MAXSIZE` | YOLO 输入队列容量 |
| `OCR_QUEUE_MAXSIZE` | OCR 任务队列容量 |
| `SEG_PIPELINE_ENABLED` | 是否启用 Seg 推理/后处理流水线 |
| `SEG_PIPELINE_QUEUE_MAXSIZE` | Seg 流水线 mask 队列容量 |
| `FPS_STATS_UPDATE_INTERVAL` | FPS 统计刷新周期 |
| `SEG_PROFILE_LOG_ENABLED` / `SEG_PROFILE_LOG_INTERVAL` | Seg 耗时诊断日志开关和间隔 |
| `MAIN_PROFILE_LOG_ENABLED` / `MAIN_PROFILE_LOG_INTERVAL` | 主流程耗时诊断日志开关和间隔 |
| `LOG_INTERVAL_DEFAULT` | 默认日志节流间隔 |
| `LOG_INTERVAL_SPEED_LIMIT_EFFECTIVE` | 限速生效日志间隔 |
| `LOG_INTERVAL_OCR_ENTER` | OCR 入队日志间隔 |
| `LOG_INTERVAL_TURN_INTENT` | LEFT/RIGHT 生效日志间隔 |
| `LOG_INTERVAL_TRAFFIC_STOP_DETAIL` | 红黄灯停车日志间隔 |
| `LOG_INTERVAL_PERSON_STOP_DETAIL` | 行人停车日志间隔 |
| `LOG_INTERVAL_PERSON_DETECT_DETAIL` | 行人检测状态日志间隔 |
| `LOG_INTERVAL_SERIAL_ERROR` | 串口异常日志间隔 |

### 场景停车与 OCR 后处理

| 参数 | 用途 |
|---|---|
| `ZEBRA_CROSSING_CLASS_NAME` / `PERSON_CLASS_NAME` | 斑马线/行人类别名 |
| `TRAFFIC_LIGHT_RED_CLASS_NAME` / `TRAFFIC_LIGHT_GREEN_CLASS_NAME` / `TRAFFIC_LIGHT_YELLOW_CLASS_NAME` | 交通灯类别名 |
| `ZEBRA_CROSSING_CLASS_ID_FALLBACK` / `PERSON_CLASS_ID_FALLBACK` | 类别名查找失败时的回退 id |
| `TRAFFIC_LIGHT_RED_CLASS_ID_FALLBACK` / `TRAFFIC_LIGHT_GREEN_CLASS_ID_FALLBACK` / `TRAFFIC_LIGHT_YELLOW_CLASS_ID_FALLBACK` | 交通灯回退 id |
| `PERSON_STOP_TRIGGER_DIST` | 行人底边距画面底部多近时触发停车 |
| `PERSON_STOP_MIN_AREA` | 行人框面积至少多大才允许触发停车 |
| `PERSON_STOP_MAX_SECONDS` | 兼容保留字段，当前行人停车不再按时间自动释放 |
| `PERSON_STOP_MISSING_TIMEOUT_SECONDS` | 行人停车后连续漏检多少秒才放行 |
| `PERSON_CLEAR_MOVE_FRAMES` | 行人朝目标侧连续移动多少帧才允许放行 |
| `PERSON_CLEAR_MIN_MOVE_DX` | 判定横向移动的最小像素量 |
| `PERSON_CLEAR_LINE_OFFSET_X` | 行人放行线相对当前车道中线的横向偏移 |
| `LIMIT_SIGN_EFFECTIVE_SPEED_OFFSET` | 限速牌识别值生效前扣掉的保守余量 |
| `OCR_MATCH_INIT_DIST` | OCR 文本框匹配检测框时的初始最大距离 |
| `OCR_DET_INPUT_SIZE` | OCR det 输入尺寸 |
| `OCR_DET_BINARY_THRESH` | OCR det 概率图二值化阈值 |
| `OCR_DET_MIN_CONTOUR_AREA` | OCR det 最小轮廓面积 |
| `OCR_DET_DILATE_KERNEL_SIZE` / `OCR_DET_DILATE_ITERATIONS` | OCR det 膨胀核大小和次数 |

### 路径搜索、稳定与调试

| 参数 | 用途 |
|---|---|
| `SEG_EMA_ALPHA` | 路径拟合系数 EMA 历史权重 |
| `SEG_PATH_STABILITY_ENABLED` | 是否启用相邻帧路径稳定约束 |
| `SEG_PATH_MAX_FRAME_X_JUMP` | 最终路径每帧最大横向移动，0 表示关闭 |
| `SEG_PATH_TEMPORAL_SCORE_GAIN` | 候选路径相对上一帧偏移扣分权重 |
| `SEG_PATH_TEMPORAL_SOFT_MAX_JUMP` | 软跳变阈值 |
| `SEG_PATH_TEMPORAL_EXCESS_SCORE_GAIN` | 超出软阈值部分的额外扣分 |
| `SEG_PATH_TEMPORAL_MIN_OVERLAP_POINTS` | 时域打分要求的最少重叠点数 |
| `SEG_PATH_HOLD_MISSING_FRAMES` | 当前帧无路径时沿用旧路径的最大帧数 |
| `STONE_BRANCH_MIN_SEP` | 石头左右分支判断要求的最小分支间距 |
| `SEG_PATH_SEARCH_STEP_Y` | 自底向上路径搜索 y 步长 |
| `SEG_CENTERLINE_ONLY_MODE` | 是否启用快速中心线模式 |
| `SEG_CENTERLINE_LARGEST_COMPONENT_ONLY` | 中心线模式是否只取最大连通白区 |
| `SEG_CENTERLINE_ROW_STEP` | 中心线逐行采样步长 |
| `SEG_PATH_GAP_THRESH` | 单行白区断开阈值 |
| `SEG_PATH_DILATE_KERNEL` / `SEG_PATH_DILATE_ITER` | 搜索 mask 底部膨胀核和次数 |
| `SEG_PATH_DILATE_BOTTOM_HEIGHT` | 底部膨胀影响高度 |
| `SEG_PATH_ACTIVE_HEIGHT` | 搜索只保留底部多少行 |
| `SEG_KEEP_BOTTOM_COMPONENTS` | 是否只保留触底连通白区 |
| `SEG_PATH_BOTTOM_MARGIN` / `SEG_PATH_BOTTOM_TOUCH_HEIGHT` | 路径底部起点预留和触底判定高度 |
| `SEG_PATH_SCAN_TOP_RATIO` | 搜索最高比例兜底值 |
| `SEG_PATH_MIN_SLICE_PIXELS` | 单层最少白像素数 |
| `SEG_PATH_MIN_BRANCH_POINTS` | 单个分支最少点数 |
| `SEG_PATH_MIN_PAIR_WIDTH` | 有效左右边界最小宽度 |
| `SEG_PATH_MAX_ROW_SEGMENTS` | 单行最多白区片段数 |
| `SEG_PATH_CONNECT_X_THRESH` | 相邻层中心点连接横向阈值 |
| `SEG_PATH_CONNECT_OVERLAP_MARGIN` | 相邻层边界重叠容忍距离 |
| `SEG_PATH_MAX_ACTIVE_PATHS` | 普通搜索保留候选路径上限 |
| `SEG_PATH_MAX_FORK_SIDE_ACTIVE_PATHS` | Y 岔路每侧候选路径上限 |
| `SEG_PATH_MIN_LENGTH` | 候选路径最少节点数 |
| `SEG_PATH_LENGTH_SCORE_GAIN` | 路径长度加分权重 |
| `SEG_PATH_SMOOTH_SCORE_GAIN` | 横向不平滑扣分权重 |
| `SEG_PATH_CENTER_PENALTY_GAIN` | 偏离局部中心扣分权重 |
| `SEG_PATH_TOP_TIER_SCORE_GAP` | 最终候选池分数差范围 |
| `SEG_PATH_DENSE_SAMPLES` | 最终路径重采样点数 |
| `TRACK_WIDTH_LOG_INTERVAL` | 赛道宽度日志节流间隔 |

### 车辆避障与分割画面

| 参数 | 用途 |
|---|---|
| `CAR_AVOIDANCE_ENABLED` | 是否启用 car 避障状态机 |
| `CAR_AVOIDANCE_SERVO_BIAS_ENABLED` | 是否启用旧版最终舵机 PWM 偏置，当前为启用 |
| `CAR_AVOIDANCE_SERVO_BIAS_MODE` | 旧版避障偏置模式，当前为 `distance_bias` |
| `CAR_AVOIDANCE_SERVO_BIAS_MIN_PWM` / `CAR_AVOIDANCE_SERVO_BIAS_MAX_PWM` | car 由远到近时额外叠加的 PWM 偏置范围，当前为 `20/60` |
| `CAR_AVOIDANCE_LEFT_BOUNDARY_TARGET_RATIO` | 左边界修正目标位置，当前为画面中线 `0.50` |
| `CAR_AVOIDANCE_LEFT_BOUNDARY_P_GAIN` / `CAR_AVOIDANCE_LEFT_BOUNDARY_P_MAX_PWM` | 左边界过中线时抵消避障偏置的 P 修正，当前为 `1.2/120` |
| `CAR_AVOIDANCE_LEFT_BOUNDARY_P_REVERSE_MAX_PWM` | 左边界 P 修正允许的最大反向拉回 PWM，当前为 `55` |
| `CAR_AVOIDANCE_LEFT_BOUNDARY_RANGE_LOW_ERROR` / `CAR_AVOIDANCE_LEFT_BOUNDARY_RANGE_HIGH_ERROR` | 左边界误差保持区间，当前为 `-20/35` |
| `CAR_AVOIDANCE_LEFT_BOUNDARY_RANGE_FULL_LOW_ERROR` | 左边界偏左时旧避障正向偏置释放到最低的阈值，当前为 `-60` |
| `CAR_AVOIDANCE_LEFT_BOUNDARY_ERROR_EMA_ALPHA` | 左边界误差滤波，避免单帧跳点触发大反打，当前为 `0.55` |
| `CAR_AVOIDANCE_LEFT_BOUNDARY_FORWARD_RECOVER_GAIN` / `CAR_AVOIDANCE_LEFT_BOUNDARY_FORWARD_RECOVER_MAX_PWM` | 左边界过度偏左时补回中间的正向托回 PWM，当前为 `0.6/25` |
| `CAR_AVOIDANCE_LEFT_BOUNDARY_REVERSE_RELEASE_DECAY` / `CAR_AVOIDANCE_LEFT_BOUNDARY_REVERSE_RELEASE_DEADBAND` | 回到保持区间后反向拉回 PWM 的释放速度，当前为 `0.2/4` |
| `CAR_AVOIDANCE_LEFT_BOUNDARY_NEUTRAL_RELEASE_DECAY` | 左边界在保持区间内时，基础避障偏置的释放速度，当前为 `0.85` |
| `CAR_AVOIDANCE_SERVO_BIAS_STEP_MAX_PWM` / `CAR_AVOIDANCE_SERVO_BIAS_CLEARING_STEP_MAX_PWM` | 避障 PWM 偏置每帧最大变化，当前为 `22/14` |
| `CAR_AVOIDANCE_CLEARING_REVERSE_MAX_PWM` | CLEARING 阶段最大反向拉回 PWM，当前为 `35` |
| `CAR_AVOIDANCE_LEFT_BOUNDARY_INSET` / `CAR_AVOIDANCE_NEAR_LEFT_BOUNDARY_INSET` | 关闭 servo_bias 后，边界路径模式的远/近左边界内收目标，当前为 `25/10` |
| `CAR_AVOIDANCE_START_BOUNDARY_ROWS` | 车辆开始接管的距离窗口，当前为 150 |
| `CAR_AVOIDANCE_NEAR_BOUNDARY_ROWS` | 近距离漏检判定窗口 |
| `CAR_AVOIDANCE_PASS_ROWS` | car 贴近底部多少行内主动进入 CLEARING，当前为 15 |
| `CAR_AVOIDANCE_LOCK_HIT_FRAMES` | car 连续命中多少帧才确认锁定，当前为 1 |
| `CAR_AVOIDANCE_SEARCH_RADIUS` | car 跟踪匹配搜索半径 |
| `CAR_AVOIDANCE_SEARCH_RADIUS_MISS_GAIN` | car 漏检时搜索半径扩大增益 |
| `CAR_AVOIDANCE_NEAR_MISS_FRAMES` | 近距离 car 遮挡/贴底时允许连续漏检帧数，当前与普通漏检一致 |
| `CAR_AVOIDANCE_TRACK_EMA_ALPHA` | car 底部中心跟踪 EMA 权重 |
| `CAR_AVOIDANCE_MISS_FRAMES` | car 锁定后允许连续漏检帧数，漏 1/2 帧继续正常避障 |
| `CAR_AVOIDANCE_MIN_SCORE` | car 避障最低置信度 |
| `CAR_AVOIDANCE_MAX_AREA` | car 最大面积过滤，0 表示关闭 |
| `CAR_AVOIDANCE_CLEARING_MISS_FRAMES` | CLEARING 起步额外保持帧数，当前为 0 |
| `CAR_AVOIDANCE_CLEARING_DECAY_FRAMES` | CLEARING 从左边界内收 10 推到中线的帧数，当前为 5 |
| `CAR_AVOIDANCE_CLEARING_MAX_FRAMES` | CLEARING 硬退出上限，避免丢车后长期挂在 PD 状态 |
| `SEG_DEBUG_PATH_COLOR` / `SEG_DEBUG_PATH_THICKNESS` | 最终路径颜色和粗细 |
| `SEG_DEBUG_DRAW_CANDIDATE_PATHS` | 是否绘制候选路径 |
| `SEG_DEBUG_DRAW_BOUNDARIES` | 是否绘制左右边界 |
| `SEG_DEBUG_DRAW_MERGE_GUIDE` | 是否绘制汇合引导线 |
| `SEG_DEBUG_LEFT_PATH_COLOR` / `SEG_DEBUG_RIGHT_PATH_COLOR` | 左右候选路径颜色 |
| `SEG_DEBUG_CANDIDATE_PATH_THICKNESS` | 候选路径粗细 |
| `SEG_DEBUG_LEFT_BOUNDARY_COLOR` / `SEG_DEBUG_RIGHT_BOUNDARY_COLOR` | 左右边界颜色 |
| `SEG_DEBUG_BOUNDARY_THICKNESS` | 边界线粗细 |
| `SEG_DEBUG_BOTTOM_MID_COLOR` / `SEG_DEBUG_BOTTOM_MID_RADIUS` | 底部参考点颜色和半径 |
| `SEG_DEBUG_FORK_DIVIDER_COLOR` / `SEG_DEBUG_FORK_DIVIDER_THICKNESS` | Y 岔路分界线颜色和粗细 |
| `SEG_DEBUG_MERGE_GUIDE_COLOR` / `SEG_DEBUG_MERGE_GUIDE_THICKNESS` | 汇合引导线颜色和粗细 |
| `SEG_DEBUG_TEXT_FONT_SCALE` / `SEG_DEBUG_TEXT_THICKNESS` | Seg 调试文字字号和粗细 |
| `SEG_DEBUG_TEXT_POS_FPS` / `SEG_DEBUG_TEXT_POS_CTRL` / `SEG_DEBUG_TEXT_POS_STONE` / `SEG_DEBUG_TEXT_POS_BRANCH` | Seg 调试文字位置 |
| `SEG_DEBUG_TEXT_COLOR_FPS` / `SEG_DEBUG_TEXT_COLOR_CTRL` / `SEG_DEBUG_TEXT_COLOR_STONE` / `SEG_DEBUG_TEXT_COLOR_BRANCH` | Seg 调试文字颜色 |

### 主预览图绘制

| 参数 | 用途 |
|---|---|
| `YOLO_DEFAULT_BOX_COLOR` / `YOLO_SIGN_BOX_COLOR` / `YOLO_LIMIT_SIGN_BOX_COLOR` | YOLO 默认/sign/limit_sign 框颜色 |
| `YOLO_BOX_THICKNESS` | YOLO 框线粗细 |
| `YOLO_LABEL_FONT_SCALE` / `YOLO_LABEL_THICKNESS` | YOLO 标签字号和粗细 |
| `YOLO_LABEL_TOP_MARGIN` / `YOLO_LABEL_TOP_OFFSET` / `YOLO_LABEL_BOTTOM_OFFSET` | YOLO 标签避让和偏移 |
| `YOLO_SUMMARY_MAX_ITEMS` | YOLO 摘要栏最多显示目标数 |
| `ZEBRA_STOPLINE_COLOR` / `ZEBRA_STOPLINE_THICKNESS` | 斑马线停止线颜色和粗细 |
| `PREVIEW_PANEL_BG_COLOR` / `PREVIEW_PANEL_BORDER_COLOR` | 预览信息面板背景和边框颜色 |
| `PREVIEW_TEXT_COLOR` / `PREVIEW_TEXT_ACCENT_COLOR` / `PREVIEW_TEXT_LIMIT_COLOR` / `PREVIEW_TEXT_STOP_COLOR` | 预览文字常规/强调/限速/停车颜色 |
| `PREVIEW_LIGHT_RED_COLOR` / `PREVIEW_LIGHT_GREEN_COLOR` / `PREVIEW_LIGHT_YELLOW_COLOR` | 红绿灯状态颜色 |
| `PREVIEW_TEXT_FONT_SCALE` / `PREVIEW_TEXT_THICKNESS` | 预览文字字号和粗细 |
| `PREVIEW_STATUS_PANEL_TOP_LEFT` / `PREVIEW_STATUS_PANEL_BOTTOM_RIGHT` | 状态面板矩形位置 |
| `PREVIEW_YOLO_PANEL_TOP_LEFT` / `PREVIEW_YOLO_PANEL_BOTTOM_RIGHT` | YOLO 摘要面板矩形位置 |
| `PREVIEW_TEXT_POS_FPS` / `PREVIEW_TEXT_POS_CTRL` / `PREVIEW_TEXT_POS_SPEED` / `PREVIEW_TEXT_POS_LIMIT` / `PREVIEW_TEXT_POS_LIGHT` / `PREVIEW_TEXT_POS_STOP` / `PREVIEW_TEXT_POS_YOLO_SUMMARY` | 主预览各行文字坐标 |

## 常用调参入口

### 1. YOLO 检测

优先看这些参数：

- `YOLO_CONF_THRES`
- `CLASS_MIN_SCORES`
- `YOLO_NMS_THRES`
- `YOLO_SIZE`

调参建议：

- 某个类别误检很多
  优先在 `CLASS_MIN_SCORES` 里单独抬高它
- 所有类别都太容易漏
  先看 `YOLO_CONF_THRES` 是否太高
- 远处小目标总看不见
  再考虑增大 `YOLO_SIZE`

### 2. OCR 触发与稳定性

优先看这些参数：

- `OCR_MIN_SIGN_BOX_AREA`
- `OCR_MIN_LIMIT_SIGN_BOX_AREA`
- `OCR_SIGN_EDGE_MARGIN_RATIO`
- `OCR_MIN_SCORE`
- `LIMIT_SIGN_APPLY_MIN_AREA`
- `LIMIT_SIGN_CONFIRM_FRAMES`
- `LIMIT_SIGN_HISTORY_MAX_MISS_FRAMES`

调参建议：

- 太远的小牌子经常误识别
  增大最小面积阈值
- 框已经够大，但经常只拍到半截字
  增大 `OCR_SIGN_EDGE_MARGIN_RATIO`
- OCR 经常读出低分脏文本
  增大 `OCR_MIN_SCORE`
- 想让系统观察更久、等牌子更近再生效
  增大 `LIMIT_SIGN_APPLY_MIN_AREA`
- 历史里错误数字太容易压过正确数字
  增大 `LIMIT_SIGN_CONFIRM_FRAMES`
- 相邻两块限速牌容易互相串历史
  减小 `LIMIT_SIGN_HISTORY_MAX_MISS_FRAMES`

### 3. 分割与路径规划

优先看这些参数：

- `SEG_EMA_ALPHA`
- `SEG_CENTERLINE_ONLY_MODE`
- `SEG_CENTERLINE_ROW_STEP`
- `SEG_PATH_SEARCH_STEP_Y`
- `SEG_PATH_GAP_THRESH`
- `SEG_PATH_CONNECT_X_THRESH`
- `SEG_PATH_TOP_TIER_SCORE_GAP`
- `PLANNING_CLASS_NAMES`
- `MERGE_GUIDE_*`
- `FORK_INNER_OPEN_*`

调参建议：

- 线看着没问题，但控制量明显不对
  优先检查 `SEG_INPUT_CROP_TOP_RATIO`、路径搜索步长和 `STEER_SIGNAL_PWM_GAIN`
- 分叉处容易来回横跳
  先看 OCR 的 `turn_intent` 是否稳定，再看 `SEG_EMA_ALPHA` 和候选路径筛选参数

### 4. 车辆控制

优先看这些参数：

- `SERVO_CENTER`
- `SERVO_MIN / SERVO_MAX`
- `STEER_CONTROL_MODE`
- `STEER_SIGNAL_PWM_GAIN`
- `STANLEY_PWM_GAIN`
- `CONTROL_C_PWM_GAIN`
- `STEER_SIGNAL_SPEED_GAIN`
- `CONTROL_C_LATERAL_GAIN`
- `CONTROL_C_LATERAL_D_GAIN`
- `CONTROL_C_HEADING_GAIN`
- `CONTROL_MIN_SPEED / CONTROL_MAX_SPEED`
- `ZEBRA_STOPLINE_TRIGGER_DIST`

调参建议：

- 想切换控制器
  直接改 `STEER_CONTROL_MODE` 为 `weighted_slope`、`stanley_band` 或 `control_c`。三套控制器互相独立，不会自动回退到其它模式。
- 车总是自然偏向一侧
  如果舵机中直还没确认，才检查 `SERVO_CENTER`；当前已确认 `750` 中直时，不要靠改中位修控制问题
- 舵机转向不够积极，明显拐不过弯
  先按当前模式增大对应 PWM 映射增益：A 用 `STEER_SIGNAL_PWM_GAIN`，B 用 `STANLEY_PWM_GAIN`，C 用 `CONTROL_C_PWM_GAIN`
- 转向一激烈就容易抖或打满
  先按当前模式减小对应 PWM 映射增益：A 用 `STEER_SIGNAL_PWM_GAIN`，B 用 `STANLEY_PWM_GAIN`，C 用 `CONTROL_C_PWM_GAIN`
- `control_c` 拉不回中线
  优先增大 `CONTROL_C_LATERAL_GAIN`
- `control_c` 慢慢过冲、左右摆
  优先增大 `CONTROL_C_LATERAL_D_GAIN`；如果变成细碎抖，再减小 D 或增大 `CONTROL_C_LATERAL_D_EMA_ALPHA`
- `control_c` 直线小幅打角抖动或出弯不丝滑
  小幅调 `CONTROL_C_HEADING_GAIN`，它以 `-Kyaw*psi` 形式抑制打角
- `control_c` 整体控制量太小或太大
  如果只是舵机幅度不合适，调 `CONTROL_C_PWM_GAIN`；如果页面 `Ctrl` 本身量级不合适，按现象调 `CONTROL_C_LATERAL_GAIN / CONTROL_C_LATERAL_D_GAIN / CONTROL_C_HEADING_GAIN`
- 弯道时车速降得不够
  增大 `STEER_SIGNAL_SPEED_GAIN`
- 整体跑得太慢或太快
  先看 `CONTROL_MIN_SPEED / CONTROL_MAX_SPEED`
- 红黄灯停车太早或太晚
  调 `ZEBRA_STOPLINE_TRIGGER_DIST`

### 5. 页面与运行节奏

优先看这些参数：

- `SEG_QUEUE_MAXSIZE / YOLO_QUEUE_MAXSIZE / OCR_QUEUE_MAXSIZE`
- `SEG_PIPELINE_ENABLED`
- `SEG_PIPELINE_QUEUE_MAXSIZE`
- `JPEG_QUALITY`
- `FPS_STATS_UPDATE_INTERVAL`
- `CONTROL_LOOP_SLEEP`
- `VIDEO_FEED_FRAME_SLEEP`

调参建议：

- 页面看起来卡，但主控制似乎还正常
  先降 `JPEG_QUALITY`
- 页面显示延迟越来越大
  不要盲目增大队列，反而应优先保持小队列
- 想让 FPS 数字更新更灵敏
  适当减小 `FPS_STATS_UPDATE_INTERVAL`

### 6. Seg 流水线与实测性能记录

当前 `SEG_PIPELINE_ENABLED=True` 时，Seg 链路拆成两段：

```text
seg_queue -> infer_mask() -> mask_queue -> postprocess_mask() -> steer_signal / preview
```

其中：

- `infer_mask()` 只做 RKNN 分割推理，输出二值 `mask`
- `mask_queue` 只保留最新一帧，旧 mask 会被丢掉
- `postprocess_mask()` 做路径搜索、拟合，调用 `PathController` 计算控制量并渲染调试图
- `SegProfile total` 是单帧端到端耗时，不等同于页面 `Seg FPS`

历史串行模式和流水线模式的现场对比如下：

| 项目 | 串行 | 现在流水线 |
|---|---:|---:|
| 执行方式 | 推理完再后处理 | 推理和上一帧后处理同时跑 |
| 页面 `Seg FPS` | 约 `16-25fps` | 约 `31fps` |
| 控制刷新周期 | `40-60ms/次` | `30-33ms/次` |
| 单帧端到端延迟 | 约 `40-60ms` | 约 `55-60ms` |
| 控制使用的画面 | 当前帧结果 | 通常落后一帧左右 |
| 舵机更新细腻度 | 一般 | 更连续 |
| 高速反应速度 | 取决于 `total`，偏慢 | 刷新更快，但画面时间戳更旧 |
| CPU / 线程调度压力 | 低一点 | 高一点 |
| 稳定性 | 更简单稳 | 多一个队列/线程，需实测 |
| 适合场景 | 看重低复杂度、低错位 | 看重控制更新频率 |

一次 `320x320 / segv2` 流水线模式下的现场 `SegProfile` 样本：

```text
SegProfile infer=23.4ms prep=0.3ms search=10.4ms fit=8.2ms render=5.4ms total=56.2ms est=17.8fps
SegProfile infer=22.8ms prep=0.4ms search=11.4ms fit=8.6ms render=5.7ms total=58.7ms est=17.0fps
SegProfile infer=21.2ms prep=0.4ms search=10.0ms fit=7.9ms render=4.2ms total=52.5ms est=19.0fps
SegProfile infer=21.8ms prep=0.4ms search=15.8ms fit=7.7ms render=3.1ms total=56.7ms est=17.6fps
SegProfile infer=25.0ms prep=0.4ms search=13.4ms fit=7.8ms render=2.5ms total=63.1ms est=15.8fps
SegProfile infer=18.5ms prep=0.6ms search=12.4ms fit=7.7ms render=5.2ms total=44.8ms est=22.3fps
SegProfile infer=20.2ms prep=0.3ms search=10.6ms fit=7.5ms render=3.3ms total=42.7ms est=23.4fps
SegProfile infer=19.9ms prep=0.1ms search=12.2ms fit=7.5ms render=3.1ms total=43.3ms est=23.1fps
SegProfile infer=17.7ms prep=0.0ms search=10.0ms fit=7.8ms render=3.0ms total=38.7ms est=25.9fps
SegProfile infer=18.9ms prep=0.0ms search=10.8ms fit=7.2ms render=3.9ms total=41.3ms est=24.2fps
SegProfile infer=23.3ms prep=0.0ms search=12.9ms fit=9.0ms render=4.7ms total=50.1ms est=19.9fps
```

这组数据的观察结论：

- 11 条样本平均约为：`infer=21.2ms prep=0.3ms search=11.8ms fit=7.9ms render=4.0ms total=49.8ms est=20.5fps`
- `infer` 大多在 `18-25ms`
- `search + fit` 大多在 `17-24ms`
- `render` 大多在 `3-6ms`
- `total` 大多在 `40-60ms`
- 终端 `est` 是 `1 / total`，所以大多显示 `16-25fps`
- 页面 `Seg FPS` 是实际产出控制量/预览图的频率，流水线下可以高于终端 `est`

当前 `416x160 / segv3` 裁下半图模型的现场样本如下。这个版本页面上曾观察到 `Seg FPS` 约 `37-38fps`，但终端 `SegProfile est` 仍要按单帧 `total` 单独看：

```text
SegProfile infer=21.4ms prep=0.0ms search=10.1ms fit=14.2ms render=3.2ms queue_wait=7.9ms total=49.0ms est=20.4fps
SegProfile infer=23.3ms prep=0.3ms search=11.1ms fit=11.0ms render=2.9ms queue_wait=4.5ms total=48.5ms est=20.6fps
SegProfile infer=23.1ms prep=0.3ms search=10.5ms fit=13.8ms render=1.3ms queue_wait=7.4ms total=49.0ms est=20.4fps
SegProfile infer=20.5ms prep=0.3ms search=13.0ms fit=7.1ms render=2.1ms queue_wait=6.8ms total=43.0ms est=23.3fps
SegProfile infer=23.7ms prep=0.3ms search=12.8ms fit=4.2ms render=2.4ms queue_wait=4.5ms total=43.4ms est=23.1fps
SegProfile infer=22.2ms prep=0.3ms search=9.3ms fit=5.7ms render=1.8ms queue_wait=2.4ms total=39.4ms est=25.4fps
SegProfile infer=19.2ms prep=0.4ms search=10.8ms fit=5.0ms render=1.6ms queue_wait=6.4ms total=37.0ms est=27.0fps
SegProfile infer=18.3ms prep=0.3ms search=9.6ms fit=5.3ms render=2.0ms queue_wait=2.7ms total=35.6ms est=28.1fps
SegProfile infer=21.1ms prep=0.3ms search=13.7ms fit=5.2ms render=1.2ms queue_wait=4.5ms total=41.5ms est=24.1fps
```

`320x320 / segv2` 与当前 `416x160 / segv3` 的对比：

| 项目 | `320x320 / segv2` 流水线记录 | 当前 `416x160 / segv3` 流水线记录 |
|---|---:|---:|
| 分割输入 | 全图 resize 到 `320x320` | 裁掉上半图，下半图 resize 到 `416x160` |
| 页面 `Seg FPS` | 流水线下可到 `30fps+` | 现场最高约 `37-38fps`，波动时约 `25fps` |
| 平均 `infer` | `21.2ms` | `21.4ms` |
| 平均 `search + fit` | `19.7ms` | `19.1ms` |
| 平均 `render` | `4.0ms` | `2.1ms` |
| 平均 `total` | `49.8ms` | `42.9ms` |
| 平均终端 `est` | `20.5fps` | `23.6fps` |
| 规划适配 | 方形输入，包含较多上半图远处信息 | 宽屏下半图，横向更细，更适合障碍物和近处路径规划 |

这次对比的结论：

- 当前 `416x160` 不是明显降低了 NPU 推理耗时；`infer` 基本仍在 `20ms` 左右
- 端到端 `total` 从历史约 `49.8ms` 降到当前约 `42.9ms`，主要收益来自渲染和部分后处理
- 页面 `Seg FPS` 能到 `37-38fps`，说明流水线吞吐更高；但车辆“看见到反应”的延迟仍主要看 `SegProfile total`
- 虽然速度提升没有输入面积变化看起来那么大，但 `416x160` 对障碍物绕行和近处 ROI 规划更合适

因此：

- 想让舵机更新更连续，主要看页面 `Seg FPS`
- 想让车辆“看见到反应”更快，主要看 `SegProfile total`
- 如果要继续降端到端延迟，优先压 `infer`、`search + fit` 和 `render`

## 当前实现里的几个容易误解的点

1. OCR 现在是整图 `det + rec`，不是直接裁 YOLO 框识别。
2. 当前 `LIMIT_SIGN_ENABLED=False`，所以 `limit_sign` 不触发 OCR，也不写入 `speed_limit`。
3. `sign` 语义路牌只有停车采样状态才触发 OCR，未停车前不会识别文字。
4. 一个语义路牌事件只跑一次停车采样和千帆判定；失败后直接按石头优先 / 默认左路放行。
5. 如果重新开启限速，`limit_sign` 不是“等靠近牌子再降速”，而是确认通过后立即生效。
6. 如果重新开启限速，限速不会自动超时恢复，只会被新的限速牌覆盖。
7. 如果重新开启限速，实际写入的限速是 `识别数字 - 1`。
8. 流水线模式下，页面 `Seg FPS` 和终端 `SegProfile est` 不相同是正常现象：前者是吞吐，后者是单帧端到端延迟换算。

## 启动方式

直接运行：

```bash
python3 main.py
```

网页预览地址：

```text
http://<板卡IP>:以 config.STREAM_PORT 为准
```

## 现场排障建议

如果系统表现不对，可以按这个顺序排查：

1. 先看网页预览有没有持续刷新
2. 再看 `Seg / YOLO FPS` 是否正常
3. 看 YOLO 框是否已经稳定出现
4. 看终端里有没有“路牌入队 / LEFT-RIGHT 生效 / 限速生效 / 红绿灯停车触发”这些条件日志
5. 看页面上的 `Limit / Light / STOP_BY_LIGHT`
6. 最后再动 `SERVO_CENTER / STEER_SIGNAL_PWM_GAIN / STEER_SIGNAL_SPEED_GAIN`

这样通常能更快判断问题是在：

- 输入链路
- 检测链路
- OCR 链路
- 路径规划
- 还是底层控制参数

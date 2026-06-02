# Aero-Twin 5.25

基于 RK3588 的自动驾驶视觉主程序。当前版本优先追求的是板端实时性、链路稳定性和现场可调性，而不是离线条件下的最高精度。

这套系统的当前分工是：

- `Seg` 负责主控闭环，持续输出循迹控制量
- `YOLO` 负责环境目标检测和路牌框检测
- `OCR` 负责整图 `det + rec`，再把结果回匹配到 `sign / limit_sign`
- 各线程都优先保“最新帧”，必要时主动丢掉旧任务，避免累积延迟

## 项目结构

```text
.
├── config.py              # 核心参数、模型路径、调参说明
├── main.py                # 主入口，线程调度、共享内存拉流、串口控制、网页推流
├── models/
│   ├── det/               # 目标检测 RKNN 模型
│   ├── ocr/               # OCR det / rec 模型与字典
│   └── seg/               # 分割 RKNN 模型
├── modules/
│   ├── detector.py        # PP-YOLOE 检测封装与后处理
│   ├── ocr_system.py      # OCR det + rec 封装
│   └── segmentor.py       # 分割、路径搜索、逆透视与控制量计算
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
   执行分割、路径搜索、逆透视和控制量估计
5. `serial_control_thread()`
   把视觉结果转成速度与舵机命令发给下位机
6. Flask 推流线程
   将最终调试画面编码成 MJPEG 供网页预览

设计原则：

- `seg_queue` 和 `yolo_queue` 都只保留最新帧
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
  分割与路径规划使用的 320x320 小图坐标系

关键约定：

- `modules/detector.py` 输出的检测框统一映射到 `TARGET_RES`
- `ocr_worker()` 跑 OCR 时使用的是同帧 `TARGET_RES` 大图
- `modules/segmentor.py` 内部会把检测框从 `TARGET_RES` 投影到 `SEG_SIZE` 与鸟瞰图空间

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

`modules/detector.py` 当前支持两类输出格式：

- PP-YOLOE 官方 demo 风格的多分支输出
- 单 tensor 的回退输出格式

检测后处理的当前真实行为：

- 先拿每个候选位置的最高分类得分
- 再按“每个类别自己的阈值”过滤
- 然后做按类别 NMS
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

- 现在已经不再使用面积或贴边几何规则过滤检测框
- 当前只保留“框尺寸合法 + 类别置信度阈值”两层过滤
- 类别阈值统一由 `YOLO_CONF_THRES + CLASS_MIN_SCORES` 控制

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

### 5. `sign` 语义路牌逻辑

当 OCR 结果匹配回 `sign` 框后：

- 如果识别到 `LEFT`
  就把 `turn_intent` 写成 `-1`
- 如果识别到 `RIGHT`
  就把 `turn_intent` 写成 `1`

`turn_intent` 会被 `segmentor.py` 在分叉路径选择时使用。  
当前策略是：

- 没有特殊干预时，默认偏向左支
- 如果 OCR 给出 `LEFT / RIGHT`，会覆盖默认偏向

### 6. `limit_sign` 限速牌逻辑

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

1. 对 `SEG_SIZE` 小图执行分割
2. 从分割输出中得到二值赛道 `mask`
3. 先只对搜索用 `mask` 的底部局部做轻微膨胀，修补近车处的小断裂
4. 如果检测到明显岔路候选，就在整张有效高度范围里全局扫描“中间缺口双边张开”的连续区域
5. 一旦确认 Y 型岔路，就用“分叉点 -> 底部中点”的分界线把路面切成左右两大区域
6. 在整图或左右分区内，自底向上分层搜索候选路径
7. 如果左右分区后有一侧路径只在底部附近出现一小段就结束，会回退成整图单路搜索，不再按岔路处理
8. 路径必须从图像真实底部 `SEG_PATH_BOTTOM_TOUCH_HEIGHT` 行内起步，中途新冒出来但没接到底部的悬空路径会被丢弃
9. 对候选路径按长度、平滑度、通道宽度、局部中心偏离做打分
10. 如果检测到 `stone`，优先绕开石头更接近的那一支
11. 如果没有明确石头干预，则默认偏向左支；如果 OCR 给出 `turn_intent`，再用 `LEFT / RIGHT` 覆盖默认偏向
12. 对最终路径做多项式拟合
13. 对拟合系数做 EMA 平滑
14. 将拟合路径转换成单一 `steer_signal`，并渲染调试图

当前输出：

- `steer_signal`
  由“路径点到底部中点的斜率 * 行号”聚合得到的单一转向控制量
- `ai_view`
  调试渲染图，后续会被放大回 `TARGET_RES`

补充说明：

- 当前岔路主判据已经不是“外边界断裂”，而是“中间缺口双边张开 + 缺口总体变大”
- 当前代码里 `stone` 已经会参与左右分支选择，但还没有进一步写成更复杂的代价场避障
- 真正直接参与主分支选择的，当前是“候选路径打分 + `stone` 左右关系 + `turn_intent` 左右偏向覆盖”

### 8. 红绿灯与斑马线逻辑

当前交通灯停车链路是：

1. YOLO 检测 `zebra_crossing`
2. 把斑马线框底边当作停止线
3. 取距离画面底部最近的那条停止线
4. 如果当前灯色是 `red` 或 `yellow`
5. 并且停止线到底部的距离小于 `ZEBRA_STOPLINE_TRIGGER_DIST`
6. 则触发强制停车
7. 如果灯色变成 `green`
   就解除这个停车状态

### 9. 串口控速逻辑

`serial_control_thread()` 当前的速度逻辑分三层：

1. 先根据弯道程度算基础速度 `dynamic_target_speed`
2. 如果 `speed_limit` 已生效，则把它当作速度上限
3. 如果红/黄灯停车条件满足，则强制把最终速度置零

可以把它理解成：

```text
target_speed = dynamic_target_speed
target_speed = min(target_speed, speed_limit)   # 若有限速
target_speed = 0                                # 若红/黄灯停车成立
```

这意味着：

- 限速牌只是速度上限，不会破坏弯道自动减速
- 红黄灯停车优先级高于限速和弯道速度

### 10. 结果回写与旧帧保护

当前系统对 OCR 结果做了两层保护，避免旧结果污染新状态：

- OCR 任务会带着对应的 `frame_id`
- 回写 `global_yolo_boxes / turn_intent / speed_limit` 前会再次核对帧号

作用是：

- 旧 OCR 结果即使晚到，也不容易把新状态覆盖回去
- 页面上显示的 OCR 文本与当前检测框更一致

### 11. 终端日志

当前终端默认只保留低频、条件触发型日志：

- 路牌达到 OCR 识别条件并真正入队
- OCR 最终识别结果（语义路牌文本 / 限速牌数字）
- `LEFT / RIGHT` 语义路牌正式生效
- 限速牌正式生效
- 红绿灯停车条件正式触发

此外，启动失败、线程异常、共享内存拉流异常、YOLO 解析异常这类错误会打印，但同一类错误只首报一次，不会持续刷屏。

## 页面预览会显示什么

网页预览里当前会叠加这些信息：

- 分割结果
- 规划线
- 鸟瞰图小窗
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

- 分割模型：`models/seg/ppliteseg_320_320_int8.rknn`
- 检测模型：`models/det/detv2/ppyoloe_crn_m_80e_custom_raw_rk3588_fp16.rknn`
- OCR det 模型：`models/ocr/ppocrv4_det_int8.rknn`
- OCR rec 模型：`models/ocr/ppocrv4_rec_fp16.rknn`
- OCR 字典：`models/ocr/keys.txt`

切换模型时，至少要同步检查：

1. `SEG_MODEL / YOLO_MODEL / OCR_DET_MODEL_PATH / REC_MODEL_PATH`
2. `SEG_SIZE / YOLO_SIZE / REC_HEIGHT / REC_WIDTH`
3. `CLASS_NAMES`
4. `modules/detector.py` 的输出解析逻辑
5. `DICT_PATH`

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

- `SRC_PTS`
- `DST_PTS`
- `SEG_EMA_ALPHA`
- `SEG_PATH_SEARCH_STEP_Y`
- `SEG_PATH_GAP_THRESH`
- `SEG_PATH_CONNECT_X_THRESH`
- `SEG_PATH_TOP_TIER_SCORE_GAP`
- `PLANNING_CLASS_NAMES`
- `PLANNING_MARKER_STYLES`

调参建议：

- 线看着没问题，但控制量明显不对
  优先检查逆透视点、路径搜索步长和 `STEER_SIGNAL_PWM_GAIN`
- 分叉处容易来回横跳
  先看 OCR 的 `turn_intent` 是否稳定，再看 `SEG_EMA_ALPHA` 和候选路径筛选参数

### 4. 车辆控制

优先看这些参数：

- `SERVO_CENTER`
- `SERVO_MIN / SERVO_MAX`
- `STEER_SIGNAL_PWM_GAIN`
- `STEER_SIGNAL_SPEED_GAIN`
- `CONTROL_MIN_SPEED / CONTROL_MAX_SPEED`
- `ZEBRA_STOPLINE_TRIGGER_DIST`

调参建议：

- 车总是自然偏向一侧
  优先校准 `SERVO_CENTER`
- 舵机转向不够积极，明显拐不过弯
  适当增大 `STEER_SIGNAL_PWM_GAIN`
- 转向一激烈就容易抖或打满
  适当减小 `STEER_SIGNAL_PWM_GAIN`
- 弯道时车速降得不够
  增大 `STEER_SIGNAL_SPEED_GAIN`
- 整体跑得太慢或太快
  先看 `CONTROL_MIN_SPEED / CONTROL_MAX_SPEED`
- 红黄灯停车太早或太晚
  调 `ZEBRA_STOPLINE_TRIGGER_DIST`

### 5. 页面与运行节奏

优先看这些参数：

- `SEG_QUEUE_MAXSIZE / YOLO_QUEUE_MAXSIZE / OCR_QUEUE_MAXSIZE`
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

## 当前实现里的几个容易误解的点

1. OCR 现在是整图 `det + rec`，不是直接裁 YOLO 框识别。
2. `limit_sign` 不是“等靠近牌子再降速”，而是确认通过后立即生效。
3. 当前限速不会自动超时恢复，只会被新的限速牌覆盖。
4. 实际写入的限速是 `识别数字 - 1`。
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

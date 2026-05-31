# Aero-Twin 5.11test

基于 RK3588 的自动驾驶视觉主程序。当前版本把视觉链路拆成三条独立工作流：

- `Seg`：赛道分割 + 路径规划
- `PPYOLO`：目标检测
- `OCR`：只识别 `sign` 框内文字

其中 `YOLO` 和 `OCR` 已经分开运行，不再共用同一个 NPU 核。

## 项目结构

```text
.
├── config.py              # 全局配置，模型路径、NPU 核分配、控制参数都在这里
├── main.py                # 主入口，线程调度、共享内存拉流、串口控制、网页推流
├── models/
│   ├── det/               # 检测模型
│   ├── ocr/               # OCR 模型和字典
│   └── seg/               # 分割模型
├── modules/
│   ├── detector.py        # PP-YOLOE RKNN 推理与后处理
│   ├── ocr_system.py      # OCR 识别封装，只负责单 ROI 识别
│   └── segmentor.py       # 分割、路径搜索、逆透视与控制量计算
└── utils/                 # 工具函数
```

## 当前架构

### 数据流

1. 上游采集进程把图像写入共享内存 `shm_ar_video`
2. `ai_producer_thread()` 从共享内存读取最新帧
3. 一帧图像被拆成两条支路
   - `seg_queue`：送给分割线程，输入尺寸 `320x320`
   - `yolo_queue`：送给检测线程，先统一到 `TARGET_RES`
4. `yolo_worker()` 执行检测
5. 如果检测到 `sign`，只把对应 ROI 任务投给 `ocr_worker()`
6. `ocr_worker()` 异步识别 `LEFT / RIGHT`，回写到检测框结果里
7. `seg_worker()` 结合当前检测框和 `turn_intent` 输出 `err_x / l_k`
8. `serial_control_thread()` 把控制量下发给下位机
9. Flask 持续推送 `global_preview_frame`

### NPU 核分配

当前配置在 [config.py](./config.py)：

- `SEG_CORES = [0]`
- `REC_CORE = 1`
- `YOLO_CORE = 2`

这样做的目的：

- 分割链路是主控链路，必须稳定，单独占一个核
- OCR 只在出现 `sign` 时运行，但一旦运行会比较拖，所以单独给它一核
- YOLO 需要尽量保持实时性，不再被 OCR 串行阻塞

## 关键文件说明

### `main.py`

系统调度中心，主要负责：

- 共享内存拉流
- 启动分割、检测、OCR 三个工作线程
- 启动串口控制线程
- 提供网页预览

主要线程：

- `ai_producer_thread`
  只保留最新帧，旧帧会被主动丢弃，优先保证实时性

- `yolo_worker`
  只做目标检测，不做 OCR

- `ocr_worker`
  只做 `sign` ROI 识别，并回写 `global_yolo_boxes`

- `seg_worker`
  主控制线程，负责生成路径、计算误差、渲染调试图

- `serial_control_thread`
  持续把 `err_x`、`l_k` 转成电机 / 舵机命令

### `modules/segmentor.py`

这是当前最核心的控制模块。

主要流程：

1. 对 `320x320 RGB` 图执行分割
2. 生成二值赛道 `mask`
3. 从底部向上搜索候选路径
4. 对候选路径打分
5. 根据 `turn_intent` 在分叉场景下偏向左路或右路
6. 将最终路径投影到鸟瞰图
7. 计算：
   - `err_x`：横向误差
   - `l_k`：预瞄斜率

它输出三样东西：

- `err_x`
- `l_k`
- `rendered_img`

### `modules/detector.py`

负责把 RKNN 检测模型统一封装成固定输出格式：

```python
{
    "rect": [x, y, w, h],
    "class_id": int,
    "class_name": str,
    "score": float,
}
```

当前后处理假设：

- 模型输出是单个 tensor
- 形状兼容 `[1, N, C]` / `[N, C]` / `[C, N]`
- 前 4 维是框坐标
- 后续维度是类别分数

如果后面替换别的 RKNN 检测模型，优先检查这里。

### `modules/ocr_system.py`

当前 OCR 只做识别，不做检测。

也就是说它只接受一张已经裁好的 ROI 图，然后输出：

- `text`
- `score`

目前主流程只对 `SIGN_CLASS_ID` 触发 OCR。

## 当前使用的模型

默认配置在 [config.py](./config.py)：

- 分割模型：`models/seg/ppliteseg_320_320_int8.rknn`
- 检测模型：`models/det/ppyoloe_crn_m_80e_custom_rk3588_fp16.rknn`
- OCR 模型：`models/ocr/ppocrv4_rec_fp16.rknn`

检测模型目录里还保留了这些版本：

- `ppyoloe_crn_m_80e_custom_raw.rknn`
- `ppyoloe_crn_m_80e_custom_raw_rk3588.rknn`
- `ppyoloe_crn_m_80e_custom_rk3588_int8.rknn`
- `yolov8_1.rknn`

如果切换模型，最少需要同步检查：

1. `config.py` 里的 `YOLO_MODEL`
2. `config.py` 里的 `YOLO_SIZE`
3. `modules/detector.py` 的输出解析逻辑

## 启动方式

直接运行：

```bash
python3 main.py
```

网页预览地址：

```text
http://<板卡IP>:5003
```

## 常用调参入口

### 1. 性能相关

看 [config.py](./config.py)：

- `JPEG_QUALITY`
  网页越卡，先降这个

- `YOLO_SIZE`
  越大检测越稳，但延迟越高

- `SEG_SIZE`
  越大分割细节越好，但主控链路会变慢

### 2. 控制相关

- `KP`
- `KD`
- `SERVO_CENTER`
- `SERVO_MIN / SERVO_MAX`
- `MOTOR_MAX_SPEED`

### 3. 鸟瞰图标定相关

- `SRC_PTS`
- `DST_PTS`
- `CM_PER_PIXEL_X`

这些参数直接影响 `err_x` 和路径视觉效果。

## 常见问题

### 1. 网页延迟很高

先看是不是这几类问题：

- `JPEG_QUALITY` 太高
- `YOLO_SIZE` 太大
- OCR 频繁触发
- 浏览器预览卡，不等于底层控制一定同样卡

### 2. 检测结果不稳，感觉不如 ONNX

优先检查：

1. 当前 RKNN 模型和代码里的后处理是否匹配
2. `YOLO_SIZE` 是否和导出时一致
3. 当前使用的是不是你想要的那份 `.rknn`

### 3. 有检测框，但 OCR 没字

常见原因：

- `sign` 框太小
- ROI 裁图越界后为空
- OCR 模型输入尺寸与当前牌子实际比例不匹配

### 4. 页面有分割图，但车控不正常

优先看：

- `err_x`
- `l_k`
- `SERVO_CENTER`
- `SRC_PTS / DST_PTS`

很多时候不是模型没跑，而是逆透视标定不对。

## 维护建议

如果后续继续优化性能，建议按这个顺序来：

1. 先优化 `YOLO` 输入链路
   当前还是“先放大到 `TARGET_RES`，再在 detector 里缩回 `YOLO_SIZE`”

2. 再优化网页推流
   这是最容易造成“看起来很卡”的部分

3. 最后再看模型替换
   比如在 `FP16 / INT8 / YOLOv8` 之间比较延迟和精度

## 备注

这份 `README` 按当前代码状态编写：

- `YOLO` 与 `OCR` 已拆线程
- `OCR` 独占 `Core 1`
- `YOLO` 独占 `Core 2`
- `Seg` 继续占 `Core 0`

如果后续主流程改了线程结构，请同步更新这里。

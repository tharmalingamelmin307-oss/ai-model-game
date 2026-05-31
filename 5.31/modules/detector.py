"""PP-YOLOE RKNN 检测封装.

职责:
1. 接收 BGR 图像并完成模型输入预处理。
2. 调用 RKNNLite 执行推理。
3. 自动判断输出格式，并使用匹配的后处理:
   - PP-YOLOE 官方 demo 风格多分支输出
   - 单 tensor xyxy + class score 回退路径
4. 将结果整理成统一检测框结构:
   {
       "rect": [x, y, w, h],
       "class_id": int,
       "class_name": str,
       "score": float,
   }

实现约定:
1. 所有输出框最终都统一映射到 `config.TARGET_RES` 坐标系。
2. 当前检测后处理只保留“框尺寸合法 + 类别置信度阈值”两层过滤，
   不再使用面积或贴边几何规则。
3. 类别级阈值由 `config.CLASS_MIN_SCORES` 控制，PP-YOLOE 多分支输出和
   单 tensor 回退路径都会走同一套阈值逻辑。
"""

import cv2
import numpy as np
from rknnlite.api import RKNNLite
import config


class YOLODetector:
    def __init__(self, core_id):
        """加载模型并绑定到指定 NPU 核."""
        self.rknn = RKNNLite()
        print(f"--> [Detector] 加载模型: {config.YOLO_MODEL}", flush=True)
        if self.rknn.load_rknn(config.YOLO_MODEL) != 0:
            raise RuntimeError("YOLO 加载失败")
        if self.rknn.init_runtime(core_mask=core_id) != 0:
            raise RuntimeError("YOLO 初始化失败")

        self.conf_thres = float(getattr(config, "YOLO_CONF_THRES", 0.25))
        self.nms_thres = float(getattr(config, "YOLO_NMS_THRES", 0.45))
        
        # [已修改] 关闭单次调试打印，保持终端整洁。如需排错可改为 True。
        self.debug_once = False 
        
        self.num_classes = len(config.CLASS_NAMES)
        
        # 允许程序自动判断 PP-YOLOE 的多分支输出格式
        self.expect_raw_single_output = False

    def _class_name_from_id(self, cls_id):
        cls_id = int(cls_id)
        if 0 <= cls_id < len(config.CLASS_NAMES):
            return config.CLASS_NAMES[cls_id]
        return str(cls_id)

    def _get_class_conf_thres(self, cls_id=None, cls_name=None):
        """返回某个类别当前应使用的置信度阈值。"""
        if cls_name is None and cls_id is not None:
            cls_name = self._class_name_from_id(cls_id)

        min_scores = getattr(config, "CLASS_MIN_SCORES", {})
        return float(min_scores.get(cls_name, self.conf_thres))

    def _build_score_keep_mask(self, class_ids, scores):
        """为一批候选框生成“按类别阈值保留”的布尔掩码。"""
        if len(class_ids) == 0 or len(scores) == 0:
            return np.array([], dtype=bool)

        score_thres = np.array(
            [self._get_class_conf_thres(cls_id=int(cls_id)) for cls_id in class_ids],
            dtype=np.float32,
        )
        return scores >= score_thres

    def _is_valid_detection(self, x, y, w, h, cls_name, score, orig_w, orig_h):
        """只按基础框尺寸和类别置信度过滤检测结果。"""
        if w < 2 or h < 2:
            return False

        if score < self._get_class_conf_thres(cls_name=cls_name):
            return False

        return True

    def _preprocess(self, frame):
        """将上游 BGR 图像整理成 RKNN 实际接收的 NHWC 输入."""
        input_w, input_h = config.YOLO_SIZE
        if frame.shape[1] != input_w or frame.shape[0] != input_h:
            img = cv2.resize(frame, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        else:
            img = frame
            
        # 1. 转换颜色空间 BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. 恢复归一化操作！转为 float32
        img = img.astype(np.float32) / 255.0
        
        # 3. PaddleDetection 的标准均值和方差
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # 返回 float32 类型的张量供 NPU 推理
        return np.expand_dims(img, axis=0)

    def _softmax(self, x, axis):
        x = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def _dfl(self, position):
        """用 numpy 实现 PP-YOLOE 的 DFL 解码，避免引入 torch 依赖。"""
        n, c, h, w = position.shape
        parts = 4
        bins = c // parts
        y = position.reshape(n, parts, bins, h, w)
        y = self._softmax(y, axis=2)
        acc = np.arange(bins, dtype=np.float32).reshape(1, 1, bins, 1, 1)
        return np.sum(y * acc, axis=2)

    def _box_process(self, position, input_w, input_h):
        """按 PP-YOLOE 的 DFL 方式把位置分支解码成 xyxy 像素框。"""
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(
            np.arange(grid_w, dtype=np.float32),
            np.arange(grid_h, dtype=np.float32)
        )
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array(
            [input_w / float(grid_w), input_h / float(grid_h)],
            dtype=np.float32
        ).reshape(1, 2, 1, 1)

        position = self._dfl(position.astype(np.float32))
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        return np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    def _sp_flatten(self, arr):
        ch = arr.shape[1]
        arr = arr.transpose(0, 2, 3, 1)
        return arr.reshape(-1, ch)

    def _nms_boxes_xyxy(self, boxes, scores):
        if len(boxes) == 0:
            return np.array([], dtype=np.int32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h
            union = areas[i] + areas[order[1:]] - inter
            iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

            inds = np.where(iou <= self.nms_thres)[0]
            order = order[inds + 1]

        return np.array(keep, dtype=np.int32)

    def _classwise_nms_xyxy(self, boxes, class_ids, scores):
        """按类别分别做 NMS，避免不同类别之间互相压框。"""
        if len(boxes) == 0:
            return boxes, class_ids, scores

        keep_boxes = []
        keep_classes = []
        keep_scores = []

        for cls_id in np.unique(class_ids):
            inds = np.where(class_ids == cls_id)[0]
            cls_boxes = boxes[inds]
            cls_scores = scores[inds]
            keep = self._nms_boxes_xyxy(cls_boxes, cls_scores)
            if len(keep) == 0:
                continue
            keep_boxes.append(cls_boxes[keep])
            keep_classes.append(np.full(len(keep), cls_id, dtype=np.int32))
            keep_scores.append(cls_scores[keep])

        if not keep_boxes:
            return None, None, None

        return (
            np.concatenate(keep_boxes, axis=0),
            np.concatenate(keep_classes, axis=0),
            np.concatenate(keep_scores, axis=0),
        )

    def _build_results(self, boxes_xyxy, class_ids, scores, output_w, output_h, input_w, input_h):
        """把内部 xyxy 结果转换成统一的 [x, y, w, h] 输出结构。"""
        results = []
        scale_x = output_w / float(input_w)
        scale_y = output_h / float(input_h)

        for i, box in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = box.astype(np.float32)
            x1 = float(np.clip(x1, 0, input_w - 1))
            y1 = float(np.clip(y1, 0, input_h - 1))
            x2 = float(np.clip(x2, 0, input_w - 1))
            y2 = float(np.clip(y2, 0, input_h - 1))

            w = x2 - x1
            h = y2 - y1
            if w < 2 or h < 2:
                continue

            x = x1 * scale_x
            y = y1 * scale_y
            w = w * scale_x
            h = h * scale_y

            cls_id = int(class_ids[i])
            cls_name = self._class_name_from_id(cls_id)
            score = float(scores[i])

            if not self._is_valid_detection(x, y, w, h, cls_name, score, output_w, output_h):
                continue

            results.append({
                "rect": [int(x), int(y), int(w), int(h)],
                "class_id": cls_id,
                "class_name": cls_name,
                "score": score,
            })

        return results

    def _looks_like_ppyoloe_outputs(self, outputs):
        if outputs is None or len(outputs) < 6 or len(outputs) % 3 != 0:
            return False

        pair_per_branch = len(outputs) // 3
        if pair_per_branch < 2:
            return False

        try:
            for i in range(3):
                pos = np.array(outputs[pair_per_branch * i])
                cls = np.array(outputs[pair_per_branch * i + 1])
                if pos.ndim != 4 or cls.ndim != 4:
                    return False
                if pos.shape[0] != 1 or cls.shape[0] != 1:
                    return False
                if pos.shape[2:] != cls.shape[2:]:
                    return False
                if pos.shape[1] % 4 != 0:
                    return False
                if cls.shape[1] != self.num_classes:
                    return False
        except Exception:
            return False

        return True

    def _decode_ppyoloe_outputs(self, outputs, output_w, output_h, input_w, input_h):
        """解析 PP-YOLOE 官方 demo 风格的多分支输出。"""
        pair_per_branch = len(outputs) // 3
        boxes = []
        class_confs = []

        for i in range(3):
            pos = np.array(outputs[pair_per_branch * i], dtype=np.float32)
            cls = np.array(outputs[pair_per_branch * i + 1], dtype=np.float32)
            boxes.append(self._box_process(pos, input_w, input_h))
            class_confs.append(cls)

        boxes = np.concatenate([self._sp_flatten(v) for v in boxes], axis=0)
        class_confs = np.concatenate([self._sp_flatten(v) for v in class_confs], axis=0)

        # 这里先选每个位置得分最高的类别，再按“该类别自己的阈值”过滤。
        class_ids = np.argmax(class_confs, axis=1).astype(np.int32)
        scores = class_confs[np.arange(class_confs.shape[0]), class_ids].astype(np.float32)
        keep = self._build_score_keep_mask(class_ids, scores)
        if not np.any(keep):
            return []

        boxes = boxes[keep]
        class_ids = class_ids[keep]
        scores = scores[keep]

        boxes, class_ids, scores = self._classwise_nms_xyxy(boxes, class_ids, scores)
        if boxes is None:
            return []
            
        if self.debug_once:
            self.debug_once = False

        return self._build_results(boxes, class_ids, scores, output_w, output_h, input_w, input_h)

    def _normalize_single_tensor_preds(self, outputs):
        """把单 tensor 输出整理成 [N, 4 + num_classes] 的统一形状。"""
        if outputs is None or len(outputs) == 0:
            return None

        preds = np.array(outputs[0])
        if preds.ndim == 3 and preds.shape[0] == 1:
            preds = preds[0]
        if preds.ndim != 2:
            raise RuntimeError(f"Unexpected pred shape: {preds.shape}")

        feat_dim = 4 + self.num_classes
        if preds.shape[1] == feat_dim:
            return preds.astype(np.float32)
        if preds.shape[0] == feat_dim:
            return preds.transpose(1, 0).astype(np.float32)

        raise RuntimeError(
            f"Unexpected pred shape after squeeze: {preds.shape}, expect [N, {feat_dim}]"
        )

    def _decode_single_tensor_outputs(self, outputs, output_w, output_h, input_w, input_h):
        """解析单 tensor 风格输出，兼容 logits / sigmoid 两种分数形态。"""
        preds = self._normalize_single_tensor_preds(outputs)
        if preds is None or len(preds) == 0:
            return []

        # 提取前4个值为框，后面13个值为类别分数
        boxes = preds[:, :4].astype(np.float32)
        scores = preds[:, 4:].astype(np.float32)

        # ====== 强制打印第一帧的数据统计 ======
        if self.debug_once:
            print("-" * 50, flush=True)
            print(f"--> [暴力查错] 框的最大值/最小值: Max={np.max(boxes):.2f}, Min={np.min(boxes):.2f}", flush=True)
            print(f"--> [暴力查错] 分数最大值/最小值: Max={np.max(scores):.4f}, Min={np.min(scores):.4f}", flush=True)
            print(f"--> [暴力查错] 第1个Anchor的框数据: {boxes[0]}", flush=True)
            print(f"--> [暴力查错] 第1个Anchor的分数数据: {scores[0]}", flush=True)
            print("-" * 50, flush=True)
        # ======================================

        # 自动判定是否需要 Sigmoid 激活
        if np.min(scores) < 0 or np.max(scores) > 1.5:
            if self.debug_once:
                print("--> [自动修复] 检测到原始 Logits，应用 Sigmoid 激活...", flush=True)
            scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -10, 10)))

        class_ids = np.argmax(scores, axis=1).astype(np.int32)
        max_scores = scores[np.arange(scores.shape[0]), class_ids].astype(np.float32)
        
        keep = self._build_score_keep_mask(class_ids, max_scores)

        if not np.any(keep):
            if self.debug_once:
                print(
                    f"--> [YOLO 拦截] 无框保留！当前全图最高分数为: {np.max(max_scores):.4f} "
                    f"(默认阈值 {self.conf_thres})",
                    flush=True,
                )
                self.debug_once = False # 查错完毕，关闭打印
            return []

        boxes = boxes[keep]
        class_ids = class_ids[keep]
        max_scores = max_scores[keep]

        # [已修复]：当前导出的单 Tensor 模型直接就是 xyxy 格式绝对坐标
        is_cxcywh = False
        if is_cxcywh:
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            boxes = np.stack([x1, y1, x2, y2], axis=1)

        # 归一化坐标自动还原
        if np.max(boxes) <= 2.0:
            if self.debug_once:
                print("--> [自动修复] 检测到归一化坐标，还原到像素尺度...", flush=True)
            boxes[:, [0, 2]] *= input_w
            boxes[:, [1, 3]] *= input_h

        if self.debug_once:
            print(f"--> [YOLO 成功] 过滤后得到 {len(boxes)} 个候选框。准备执行 NMS...", flush=True)
            self.debug_once = False # 查错完毕，关闭打印

        boxes, class_ids, max_scores = self._classwise_nms_xyxy(boxes, class_ids, max_scores)
        if boxes is None:
            return []

        return self._build_results(boxes, class_ids, max_scores, output_w, output_h, input_w, input_h)

    def run(self, frame_data, output_size=None):
        """执行一次完整检测流程."""
        try:
            infer_h, infer_w = frame_data.shape[:2]
            if output_size is None:
                output_w, output_h = infer_w, infer_h
            else:
                output_w, output_h = output_size
            input_w, input_h = config.YOLO_SIZE
            blob = self._preprocess(frame_data)

            outputs = self.rknn.inference(
                inputs=[blob],
                data_format=['nhwc']
            )
            if outputs is None or len(outputs) == 0:
                return []

            if self.debug_once:
                print(f"YOLO input shape: {blob.shape}, dtype: {blob.dtype}", flush=True)
                print("YOLO output num:", len(outputs), flush=True)
                for idx, out in enumerate(outputs):
                    arr = np.array(out)
                    print(f"output[{idx}] shape: {arr.shape}, dtype: {arr.dtype}", flush=True)

            if self.expect_raw_single_output:
                if self.debug_once:
                    print("YOLO decode path: raw_single_tensor_xyxy_scores", flush=True)
                return self._decode_single_tensor_outputs(outputs, output_w, output_h, input_w, input_h)

            if self._looks_like_ppyoloe_outputs(outputs):
                if self.debug_once:
                    print("YOLO decode path: ppyoloe_demo_postprocess", flush=True)
                return self._decode_ppyoloe_outputs(outputs, output_w, output_h, input_w, input_h)

            if self.debug_once:
                print("YOLO decode path: single_tensor_xyxy", flush=True)
            return self._decode_single_tensor_outputs(outputs, output_w, output_h, input_w, input_h)

        except Exception as e:
            print(f"YOLO 解析异常: {e}", flush=True)
            return []

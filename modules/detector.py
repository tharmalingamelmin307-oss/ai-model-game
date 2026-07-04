"""PP-YOLOE RKNN 检测封装.

职责:
1. 接收 BGR 图像并完成模型输入预处理。
2. 调用 RKNNLite 执行推理。
3. 按当前 detv3 official split 模型解析两个输出:
   - boxes: [N, 4] xyxy
   - scores: [N, num_classes]
4. 将结果整理成统一检测框结构:
   {
       "rect": [x, y, w, h],
       "class_id": int,
       "class_name": str,
       "score": float,
   }

实现约定:
1. 所有输出框最终都统一映射到 `config.TARGET_RES` 坐标系。
2. 当前检测后处理对齐 detv3 测试脚本：框尺寸、类别置信度、
   类别最大面积比例和贴边大框过滤。
3. 类别级阈值由 `config.CLASS_MIN_SCORES` 控制。
4. 当前 RKNN 已固化 mean/std，Python 侧只喂 0-255 RGB uint8。
"""

import cv2
import numpy as np
from rknnlite.api import RKNNLite
import config


class YOLODetector:
    def __init__(self, core_id):
        """加载模型并绑定到指定 NPU 核."""
        self.rknn = RKNNLite()
        if self.rknn.load_rknn(config.YOLO_MODEL) != 0:
            raise RuntimeError("YOLO 加载失败")
        if self.rknn.init_runtime(core_mask=core_id) != 0:
            raise RuntimeError("YOLO 初始化失败")

        self.conf_thres = float(config.YOLO_CONF_THRES)
        self.nms_thres = float(config.YOLO_NMS_THRES)
        
        self.runtime_error_logged = False

        self.num_classes = len(config.CLASS_NAMES)

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
        """按当前 detv3 测试脚本的规则过滤检测结果。"""
        if w < config.YOLO_BOX_MIN_SIZE or h < config.YOLO_BOX_MIN_SIZE:
            return False

        if score < self._get_class_conf_thres(cls_name=cls_name):
            return False

        frame_area = float(max(orig_w * orig_h, 1))
        area_ratio = float(w * h) / frame_area
        max_area_ratio = getattr(config, "YOLO_MAX_AREA_RATIO_BY_CLASS", {}).get(cls_name)
        if max_area_ratio is not None and area_ratio > float(max_area_ratio):
            return False

        edge_margin_x = float(orig_w) * float(config.YOLO_EDGE_MARGIN_RATIO)
        edge_margin_y = float(orig_h) * float(config.YOLO_EDGE_MARGIN_RATIO)
        edge_touch_count = 0
        if x <= edge_margin_x:
            edge_touch_count += 1
        if y <= edge_margin_y:
            edge_touch_count += 1
        if (x + w) >= float(orig_w) - edge_margin_x:
            edge_touch_count += 1
        if (y + h) >= float(orig_h) - edge_margin_y:
            edge_touch_count += 1
        if edge_touch_count >= 2 and area_ratio > float(config.YOLO_EDGE_TOUCH_MAX_AREA_RATIO):
            return False

        return True

    def _preprocess(self, frame):
        """将上游 BGR 图像整理成 RKNN 实际接收的 NHWC 输入."""
        input_w, input_h = config.YOLO_SIZE
        if frame.shape[1] != input_w or frame.shape[0] != input_h:
            img = cv2.resize(frame, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        else:
            img = frame
            
        # 当前 detv3 RKNN 内部已固化 mean/std，这里只喂 0-255 RGB。
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.expand_dims(img, axis=0)

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

    def _limit_pre_nms_candidates(self, boxes, class_ids, scores):
        """按类别保留 Top-K 候选，避免低阈值下 NMS 抢占过多 CPU。"""
        topk = int(getattr(config, "YOLO_PRE_NMS_TOPK_PER_CLASS", 0))
        if topk <= 0 or len(boxes) == 0:
            return boxes, class_ids, scores

        keep_indices = []
        for cls_id in np.unique(class_ids):
            inds = np.where(class_ids == cls_id)[0]
            if len(inds) > topk:
                order = scores[inds].argsort()[::-1][:topk]
                inds = inds[order]
            keep_indices.append(inds)

        if not keep_indices:
            return boxes[:0], class_ids[:0], scores[:0]

        keep_indices = np.concatenate(keep_indices, axis=0)
        return boxes[keep_indices], class_ids[keep_indices], scores[keep_indices]

    def _scale_boxes_to_output(self, boxes_xyxy, output_w, output_h, input_w, input_h):
        """把输入坐标系 xyxy 框缩放到统一输出坐标系。"""
        if len(boxes_xyxy) == 0:
            return boxes_xyxy.astype(np.float32)

        boxes = boxes_xyxy.astype(np.float32).copy()
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, input_w - 1) * output_w / float(input_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, input_h - 1) * output_h / float(input_h)
        return boxes

    def _build_results(self, boxes_xyxy, class_ids, scores, output_w, output_h):
        """把输出坐标系 xyxy 结果转换成统一的 [x, y, w, h] 输出结构。"""
        results = []

        for i, box in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = box.astype(np.float32)
            x1 = float(np.clip(x1, 0, output_w - 1))
            y1 = float(np.clip(y1, 0, output_h - 1))
            x2 = float(np.clip(x2, 0, output_w - 1))
            y2 = float(np.clip(y2, 0, output_h - 1))

            w = x2 - x1
            h = y2 - y1
            if w < config.YOLO_BOX_MIN_SIZE or h < config.YOLO_BOX_MIN_SIZE:
                continue

            cls_id = int(class_ids[i])
            cls_name = self._class_name_from_id(cls_id)
            score = float(scores[i])

            if not self._is_valid_detection(x1, y1, w, h, cls_name, score, output_w, output_h):
                continue

            results.append({
                "rect": [int(x1), int(y1), int(w), int(h)],
                "class_id": cls_id,
                "class_name": cls_name,
                "score": score,
            })

        return results

    def _normalize_split_outputs(self, outputs):
        """把 detv3 官方 split 输出整理成 boxes[N,4] 与 scores[N,num_classes]."""
        if outputs is None:
            raise RuntimeError("YOLO returned no outputs")
        if len(outputs) != 2:
            shapes = [np.array(out).shape for out in outputs]
            raise RuntimeError(f"Unexpected detv3 output count {len(outputs)}, shapes={shapes}")

        a = np.array(outputs[0], dtype=np.float32)
        b = np.array(outputs[1], dtype=np.float32)

        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        if b.ndim == 3 and b.shape[0] == 1:
            b = b[0]

        if a.ndim == 2 and a.shape[0] == 4:
            a = a.T
        if b.ndim == 2 and b.shape[0] == 4:
            b = b.T
        if a.ndim == 2 and a.shape[0] == self.num_classes:
            a = a.T
        if b.ndim == 2 and b.shape[0] == self.num_classes:
            b = b.T

        if a.ndim == 2 and a.shape[1] == 4:
            boxes, scores = a, b
        elif b.ndim == 2 and b.shape[1] == 4:
            boxes, scores = b, a
        else:
            raise RuntimeError(f"Unexpected split output shapes: {a.shape}, {b.shape}")

        if scores.ndim != 2 or scores.shape[1] != self.num_classes:
            raise RuntimeError(f"Unexpected split score shape: {scores.shape}")

        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        return boxes.astype(np.float32), scores.astype(np.float32)

    def _decode_split_outputs(self, outputs, output_w, output_h, input_w, input_h):
        """解析 detv3 官方 split 输出: boxes xyxy + class scores."""
        boxes, scores_all = self._normalize_split_outputs(outputs)
        if boxes is None or scores_all is None or len(boxes) == 0:
            return []

        class_ids = np.argmax(scores_all, axis=1).astype(np.int32)
        scores = scores_all[np.arange(scores_all.shape[0]), class_ids].astype(np.float32)

        keep = self._build_score_keep_mask(class_ids, scores)
        if not np.any(keep):
            return []

        boxes = boxes[keep]
        class_ids = class_ids[keep]
        scores = scores[keep]

        boxes, class_ids, scores = self._limit_pre_nms_candidates(boxes, class_ids, scores)
        boxes = self._scale_boxes_to_output(boxes, output_w, output_h, input_w, input_h)
        boxes, class_ids, scores = self._classwise_nms_xyxy(boxes, class_ids, scores)
        if boxes is None:
            return []

        results = self._build_results(boxes, class_ids, scores, output_w, output_h)
        max_dets = int(getattr(config, "YOLO_MAX_DETS", 0))
        if max_dets > 0 and len(results) > max_dets:
            results.sort(key=lambda item: item["score"], reverse=True)
            results = results[:max_dets]
        return results

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
            return self._decode_split_outputs(outputs, output_w, output_h, input_w, input_h)

        except Exception as e:
            if not self.runtime_error_logged:
                print(f"YOLO解析异常: {e}", flush=True)
                self.runtime_error_logged = True
            return []

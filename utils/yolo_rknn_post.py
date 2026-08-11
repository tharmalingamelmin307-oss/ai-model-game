"""YOLO RKNN deployment helpers for predfl / seg_predfl models."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxTransform:
    scale_x: float
    scale_y: float
    pad_x: int
    pad_y: int
    input_w: int
    input_h: int
    orig_w: int
    orig_h: int


def sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def letterbox_bgr_or_rgb(img, dst_size, pad_value=114, scaleup=False):
    """Resize with unchanged aspect ratio and pad to (width, height)."""
    dst_w, dst_h = int(dst_size[0]), int(dst_size[1])
    orig_h, orig_w = img.shape[:2]
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"invalid image shape: {img.shape}")

    scale = min(dst_w / float(orig_w), dst_h / float(orig_h))
    if not scaleup:
        scale = min(scale, 1.0)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((dst_h, dst_w, img.shape[2]), int(pad_value), dtype=img.dtype)
    pad_x = (dst_w - new_w) // 2
    pad_y = (dst_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    transform = LetterboxTransform(
        scale_x=float(new_w) / float(orig_w),
        scale_y=float(new_h) / float(orig_h),
        pad_x=int(pad_x),
        pad_y=int(pad_y),
        input_w=dst_w,
        input_h=dst_h,
        orig_w=int(orig_w),
        orig_h=int(orig_h),
    )
    return canvas, transform


def make_anchor_grid(input_size, strides):
    """Build anchor grid for rectangular YOLO inputs.

    Args:
        input_size: (width, height).
        strides: iterable of stride values.
    """
    input_w, input_h = int(input_size[0]), int(input_size[1])
    anchor_list = []
    stride_list = []
    for stride in strides:
        stride = int(stride)
        h = input_h // stride
        w = input_w // stride
        xs = np.arange(w, dtype=np.float32) + 0.5
        ys = np.arange(h, dtype=np.float32) + 0.5
        grid_x, grid_y = np.meshgrid(xs, ys)
        anchor_list.append(np.stack([grid_x.ravel(), grid_y.ravel()], axis=1))
        stride_list.extend([stride] * (h * w))
    return np.concatenate(anchor_list, axis=0).astype(np.float32), np.asarray(stride_list, dtype=np.float32)


def decode_dfl_boxes(raw_dfl, input_size, strides, reg_max=16, candidate_idx=None):
    """Decode YOLOv8 DFL logits to xyxy boxes in letterbox input coordinates."""
    raw = np.asarray(raw_dfl, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[0] != 1:
        raise ValueError(f"raw_dfl must be [1,C,N] or [1,N,C], got {raw.shape}")

    if raw.shape[1] == 4 * int(reg_max):
        raw = raw
    elif raw.shape[2] == 4 * int(reg_max):
        raw = raw.transpose(0, 2, 1)
    else:
        cands = [dim for dim in raw.shape[1:] if dim % 4 == 0 and dim > 4]
        if not cands:
            raise ValueError(f"cannot infer DFL channels from shape {raw.shape}")
        inferred_reg_max = int(cands[0] // 4)
        reg_max = inferred_reg_max
        if raw.shape[1] != 4 * reg_max:
            raw = raw.transpose(0, 2, 1)

    b, channels, anchors_n = raw.shape
    if channels % 4 != 0 or channels <= 4:
        raise ValueError(f"DFL channels must be 4*reg_max, got {channels}")
    reg_max = channels // 4

    if candidate_idx is None:
        selected = slice(None)
    else:
        selected = np.asarray(candidate_idx, dtype=np.int64)
        raw = raw[:, :, selected]

    x = raw.reshape(b, 4, reg_max, -1).transpose(0, 2, 1, 3)
    x = np.exp(x - x.max(axis=1, keepdims=True))
    x = x / np.maximum(x.sum(axis=1, keepdims=True), 1e-12)
    bins = np.arange(reg_max, dtype=np.float32)
    ltrb = (x * bins[None, :, None, None]).sum(axis=1)[0]

    anchors, stride_vals = make_anchor_grid(input_size, strides)
    if anchors.shape[0] != anchors_n:
        raise ValueError(f"anchor count mismatch: output={anchors_n}, grid={anchors.shape[0]}")
    anchors = anchors[selected]
    stride_vals = stride_vals[selected]

    l, t, r, btm = ltrb
    ax, ay = anchors[:, 0], anchors[:, 1]
    x1 = (ax - l) * stride_vals
    y1 = (ay - t) * stride_vals
    x2 = (ax + r) * stride_vals
    y2 = (ay + btm) * stride_vals
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


def nms_xyxy(boxes, scores, iou_threshold=0.7):
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if len(scores) == 0:
        return np.array([], dtype=np.int32)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= float(iou_threshold)]
    return np.asarray(keep, dtype=np.int32)


def class_aware_nms(boxes, scores, class_ids, max_det, iou_threshold=0.7, max_nms=30000):
    if len(scores) == 0:
        return np.array([], dtype=np.int32)

    order = np.argsort(-scores, kind="stable")
    if len(order) > int(max_nms):
        order = order[:int(max_nms)]

    if hasattr(cv2.dnn, "NMSBoxesBatched"):
        boxes_xywh = boxes[order].astype(np.float32).copy()
        boxes_xywh[:, 2] = np.maximum(0.0, boxes_xywh[:, 2] - boxes_xywh[:, 0])
        boxes_xywh[:, 3] = np.maximum(0.0, boxes_xywh[:, 3] - boxes_xywh[:, 1])
        try:
            keep_local = cv2.dnn.NMSBoxesBatched(
                boxes_xywh.tolist(),
                scores[order].astype(float).tolist(),
                class_ids[order].astype(int).tolist(),
                0.0,
                float(iou_threshold),
                top_k=int(max_det),
            )
            keep_local = np.asarray(keep_local, dtype=np.int32).reshape(-1)
            if keep_local.size > 0:
                return order[keep_local[:int(max_det)]].astype(np.int32)
        except Exception:
            pass

    max_wh = 7680.0
    boxes_for_nms = boxes[order] + class_ids[order].astype(np.float32)[:, None] * max_wh
    keep_local = nms_xyxy(boxes_for_nms, scores[order], iou_threshold)
    keep = order[keep_local]
    return keep[:int(max_det)].astype(np.int32)


def score_sum_candidates(score_sum, conf_thresh):
    values = np.asarray(score_sum, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.zeros((0,), dtype=np.int64)
    unique = np.unique(values)
    if unique.size <= 1 or float(values.max()) <= float(conf_thresh):
        return np.arange(values.size, dtype=np.int64)
    positive_steps = np.diff(unique)
    positive_steps = positive_steps[positive_steps > 0]
    margin = float(positive_steps.min()) if positive_steps.size else 0.0
    return np.flatnonzero(values >= float(conf_thresh) - margin).astype(np.int64)


def scale_boxes_from_letterbox(boxes_xyxy, transform, output_size):
    """Map letterbox input-space boxes to output_size coordinates."""
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).copy()
    if boxes.size == 0:
        return boxes.reshape(0, 4)

    output_w, output_h = int(output_size[0]), int(output_size[1])
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - float(transform.pad_x)) / max(float(transform.scale_x), 1e-12)
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - float(transform.pad_y)) / max(float(transform.scale_y), 1e-12)
    boxes[:, [0, 2]] *= output_w / float(transform.orig_w)
    boxes[:, [1, 3]] *= output_h / float(transform.orig_h)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, output_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, output_h - 1)
    return boxes


def assemble_seg_masks(seg_result, input_size, mask_threshold=0.5):
    """Reconstruct binary instance masks in model input coordinates."""
    coeffs = np.asarray(seg_result["coeffs"], dtype=np.float32)
    proto = np.asarray(seg_result["proto"], dtype=np.float32)
    input_w, input_h = int(input_size[0]), int(input_size[1])
    if coeffs.shape[0] == 0:
        return np.zeros((0, input_h, input_w), dtype=np.uint8)

    channels, proto_h, proto_w = proto.shape
    masks = coeffs @ proto.reshape(channels, -1)
    masks = sigmoid(masks).reshape(-1, proto_h, proto_w)

    masks_up = np.empty((masks.shape[0], input_h, input_w), dtype=np.float32)
    for i, mask in enumerate(masks):
        masks_up[i] = cv2.resize(mask, (input_w, input_h), interpolation=cv2.INTER_LINEAR)

    boxes = np.ceil(np.asarray(seg_result["boxes"], dtype=np.float32)).astype(np.int32)
    cropped = np.zeros_like(masks_up)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x1 = max(int(x1), 0)
        y1 = max(int(y1), 0)
        x2 = min(int(x2), input_w)
        y2 = min(int(y2), input_h)
        if x2 > x1 and y2 > y1:
            cropped[i, y1:y2, x1:x2] = masks_up[i, y1:y2, x1:x2]
    return (cropped > float(mask_threshold)).astype(np.uint8)

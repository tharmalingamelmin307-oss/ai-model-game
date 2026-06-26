# utils/image_proc.py
"""图像几何处理工具.

当前这里只保留 OCR 会直接使用的一项能力：
- 把 OCR det 输出的四点文字框透视拉正成规则矩形
"""

import cv2
import numpy as np

def get_rotate_crop_image(img, points):
    """把四点文字框透视拉正成规则文本图。

    这是 OCR det + rec 链路里的标准中间步骤：
    1. 根据四点框估计目标宽高
    2. 用透视变换把倾斜文本拉正
    3. 如果结果明显是“竖着的长条”，再旋转成更适合 rec 的横向文本
    """
    points = np.array(points, dtype=np.float32)
    # 计算目标矩形的宽高
    width = int(max(np.linalg.norm(points[0] - points[1]), 
                    np.linalg.norm(points[2] - points[3])))
    height = int(max(np.linalg.norm(points[0] - points[3]), 
                     np.linalg.norm(points[1] - points[2])))
    
    pts_std = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    # 矫正
    dst_img = cv2.warpPerspective(img, M, (width, height),
                                 borderMode=cv2.BORDER_REPLICATE, 
                                 flags=cv2.INTER_CUBIC)
    
    # 针对长窄文字的特殊处理：如果高度明显大于宽度，旋转90度
    if dst_img.shape[0] * 1.0 / dst_img.shape[1] >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img

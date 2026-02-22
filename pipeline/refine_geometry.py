# pipeline/refine_geometry.py
import cv2
import numpy as np
from utils.visualize import draw_mask, draw_polygon, img_show


def analyze_mask(mask, image_path=None, show_result=False):
    """
    分析 mask 的几何特征
    
    Args:
        mask: 二值 mask
        image_path: 图片路径（用于可视化）
        show_result: 是否弹窗显示结果（GUI 模式下应设为 False）
    """
    mask_u8 = (mask * 255).astype("uint8")

    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    # ---------- 原有几何 ----------
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h > 0 else 0 # 宽高比，可以用来判断物体的形态，是横着放还是竖着放

    # ----------轮廓点分布方差 ----------
    # 计算几何中心
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = 0, 0

    # 所有轮廓点到中心的距离
    points = cnt.squeeze()  # (N, 2)
    peri = cv2.arcLength(cnt, True)
    epsilon = 0.01 * peri
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    polygon = approx.squeeze().tolist()
    dists = np.linalg.norm(points - np.array([cx, cy]), axis=1)

    mean_dist = np.mean(dists)
    var_dist = np.var(dists)

    # 归一化分布方差（尺度无关）
    normalized_contour_variance = (
        float(var_dist / (mean_dist ** 2)) if mean_dist > 0 else 0
    )

    # ----------凸包面积比 ----------
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)

    convexity_ratio = (
        float(area / hull_area) if hull_area > 0 else 0
    )
    
    # 只有在非 GUI 模式下才弹窗显示
    if show_result and image_path:
        img = cv2.imread(image_path)
        if img is not None:
            img = draw_polygon(img, polygon, color=(255, 0, 0))
            img_show(img, win_name="Mask with Polygon")
    return {
        "mask_area": float(area),
        "refined_bbox_xywh": [int(x), int(y), int(w), int(h)],
        "aspect_ratio": float(aspect_ratio),
        # 边缘点的到中心点的方差归一化的值
        "normalized_contour_variance": normalized_contour_variance,
        # 图形原来的面积和其凸包的面积比
        "convexity_ratio": convexity_ratio,

        "polygon": polygon,
        "convex_hull": hull.squeeze().tolist()
    }

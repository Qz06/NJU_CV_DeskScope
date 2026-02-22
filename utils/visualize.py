import cv2
import numpy as np


def draw_yolo_boxes(image, detections, show_labels=True, font_scale=None, thickness=None):
    """
    绘制 YOLO 检测框和标签
    
    image: np.ndarray (BGR)
    detections: list of dict (JSON 结构)
    show_labels: 是否显示物品名称标签
    font_scale: 字体大小（None 则根据图片大小自动计算）
    thickness: 线条粗细（None 则根据图片大小自动计算）
    """
    img = image.copy()
    h, w = img.shape[:2]
    
    # 根据图片大小自动调整参数
    if font_scale is None:
        font_scale = max(0.5, min(w, h) / 1000)
    if thickness is None:
        thickness = max(1, int(min(w, h) / 500))
    
    # 颜色映射（不同类别不同颜色）
    color_map = {}
    colors = [
        (0, 255, 0),    # 绿色
        (255, 0, 0),    # 蓝色
        (0, 255, 255),  # 黄色
        (255, 0, 255),  # 紫色
        (0, 165, 255),  # 橙色
        (255, 255, 0),  # 青色
        (128, 0, 255),  # 粉色
        (0, 128, 255),  # 浅橙
    ]
    
    for obj in detections:
        x1, y1, x2, y2 = map(int, obj["bbox_xyxy"])
        cls = obj["class"]
        conf = obj.get("confidence", 0)
        
        # 为每个类别分配颜色
        if cls not in color_map:
            color_map[cls] = colors[len(color_map) % len(colors)]
        color = color_map[cls]
        
        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        if show_labels:
            # 准备标签文字
            label = f"{cls} {conf:.2f}"
            
            # 计算文字大小
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # 绘制标签背景
            label_y1 = max(y1 - text_h - 10, 0)
            label_y2 = y1
            cv2.rectangle(img, (x1, label_y1), (x1 + text_w + 6, label_y2), color, -1)
            
            # 绘制文字（白色）
            cv2.putText(
                img,
                label,
                (x1 + 3, label_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness
            )

    return img


def draw_detection_with_sam3(image, detections, show_labels=True):
    """
    绘制完整的检测结果，包括 YOLO 边界框和 SAM3 精细分割结果
    
    image: np.ndarray (BGR)
    detections: list of dict，包含 sam3 字段的检测结果
    """
    img = image.copy()
    h, w = img.shape[:2]
    
    # 根据图片大小自动调整参数
    font_scale = max(0.4, min(w, h) / 1200)
    thickness = max(1, int(min(w, h) / 600))
    
    # 颜色映射
    color_map = {}
    colors = [
        (0, 255, 0),    # 绿色
        (255, 0, 0),    # 蓝色
        (0, 255, 255),  # 黄色
        (255, 0, 255),  # 紫色
        (0, 165, 255),  # 橙色
        (255, 255, 0),  # 青色
        (128, 0, 255),  # 粉色
        (0, 128, 255),  # 浅橙
    ]
    
    for obj in detections:
        x1, y1, x2, y2 = map(int, obj["bbox_xyxy"])
        cls = obj["class"]
        conf = obj.get("confidence", 0)
        sam3_data = obj.get("sam3")
        
        # 为每个类别分配颜色
        if cls not in color_map:
            color_map[cls] = colors[len(color_map) % len(colors)]
        color = color_map[cls]
        
        # 如果有 SAM3 精细分割结果，绘制多边形
        if sam3_data and "polygon" in sam3_data and sam3_data["polygon"]:
            polygon = np.array(sam3_data["polygon"], dtype=np.int32)
            # 绘制半透明填充
            overlay = img.copy()
            cv2.fillPoly(overlay, [polygon], color)
            img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
            # 绘制多边形边界
            cv2.polylines(img, [polygon], True, color, thickness)
        else:
            # 没有 SAM3 结果，只绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        if show_labels:
            # 准备标签文字
            label = f"{cls} {conf:.2f}"
            
            # 计算文字大小
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # 绘制标签背景
            label_y1 = max(y1 - text_h - 10, 0)
            label_y2 = y1
            cv2.rectangle(img, (x1, label_y1), (x1 + text_w + 6, label_y2), color, -1)
            
            # 绘制文字（白色）
            cv2.putText(
                img,
                label,
                (x1 + 3, label_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness
            )

    return img

def draw_mask(image, mask, color=(0, 0, 255), alpha=0.4):
    """
    image: np.ndarray (BGR)
    mask: np.ndarray (H, W) 0/1 or bool
    """
    img = image.copy()

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    colored_mask = np.zeros_like(img)
    colored_mask[mask > 0] = color

    img = cv2.addWeighted(img, 1.0, colored_mask, alpha, 0)
    return img


def draw_polygon(image, polygon, color=(255, 0, 0)):
    img = image.copy()
    pts = np.array(polygon, dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
    return img

def img_show(img, win_name="image", wait=True, max_size=1200):
    import cv2

    if img is None:
        raise ValueError("img is None")

    h, w = img.shape[:2]
    scale = min(1.0, max_size / max(h, w))

    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    h, w = img.shape[:2]

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, w, h)
    cv2.imshow(win_name, img)

    if wait:
        cv2.waitKey(0)
        try:
            cv2.destroyWindow(win_name)
        except cv2.error:
            pass

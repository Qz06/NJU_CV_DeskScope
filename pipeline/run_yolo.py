# pipeline/run_yolo.py
from ultralytics import YOLO
import cv2
from utils.visualize import draw_yolo_boxes, img_show
print("yolo is ready")

SAM3_CLASSES = {
  "paper", "notebook", "folder", "book", "document",
  "stacked paper", "scrap paper", "waste paper", "old newspaper",
  "sticky notes", "tissue", "paper towel", "napkin",

  "snack bag", "chip bag", "bread bag",
  "plastic bag", "paper bag",
  "food wrapper", "trash packaging", "wrappers",
  "plastic wrap", "sugar wrapper", "snack leftovers", "debris",

  "cup", "plastic cup", "paper cup",
  "water bottle", "bottle", "drink bottle",
  "can", "soda can",

  "food container", "instant noodles",
  "carton", "box", "biscuit box", "cookie box",

  "cable", "charger", "data wire",   # 可选
  "headphones", "earphones"           # 可选
}



def run_yolo(image_path, model_path, show_result=True):
    """
    运行 YOLO 检测
    
    Args:
        image_path: 图片路径
        model_path: 模型路径
        show_result: 是否弹窗显示结果（GUI 模式下应设为 False）
    """
    model = YOLO(model_path)
    results = model(image_path)   #这里的results是一个列表，image_path可以是包含多个路径的list
    objects = []
    r = results[0]
    for i, box in enumerate(r.boxes):
        print("find class:", r.names[int(box.cls)])
        obj = {
            "id": i,
            "class": r.names[int(box.cls)],
            "confidence": float(box.conf),
            "bbox_xyxy": [
                float(box.xyxy[0][0]),
                float(box.xyxy[0][1]),
                float(box.xyxy[0][2]),
                float(box.xyxy[0][3]),
            ],
            "geometry_required": r.names[int(box.cls)] in SAM3_CLASSES, #需要判断几何结构的物品可以写在这里
            "sam3": None   #后期可拓展的东西
        }
        objects.append(obj)
    
    # 只有在非 GUI 模式下才弹窗显示
    if show_result:
        img = cv2.imread(image_path)
        img = draw_yolo_boxes(img, objects)
        img_show(img, win_name="YOLO Detections")
    
    return {
        "image": image_path,
        "objects": objects
    }

import numpy as np
from PIL import Image
import os

_yolo = None

YOLO_MODEL = "models/yolov8n.pt"


def load_yolo():
    global _yolo
    if _yolo is not None:
        return _yolo
    from ultralytics import YOLO
    path = YOLO_MODEL if os.path.exists(YOLO_MODEL) else "yolov8n.pt"
    _yolo = YOLO(path)
    return _yolo


def detect_objects(image: Image.Image) -> list[dict]:
    model = load_yolo()
    results = model(image, verbose=False)[0]
    boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = results.names[cls_id]
        boxes.append({
            "label": label,
            "confidence": round(conf, 4),
            "bbox": [round(x1), round(y1), round(x2), round(y2)],
        })
    return boxes

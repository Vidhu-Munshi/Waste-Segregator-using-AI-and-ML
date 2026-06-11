"""
Fine-tune YOLOv8 on a custom waste detection dataset.

Usage:
    python training/train_yolo.py --data datasets/waste_yolo/data.yaml --epochs 50

datasets/waste_yolo/ must be in YOLO format:
    images/train/  images/val/
    labels/train/  labels/val/
    data.yaml
"""
import argparse
from ultralytics import YOLO


def main(data_yaml: str, epochs: int, imgsz: int):
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project="models",
        name="waste_yolo",
        exist_ok=True,
    )
    print("Training complete:", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/waste_yolo/data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    main(args.data, args.epochs, args.imgsz)

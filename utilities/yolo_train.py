"""
Train a YOLO model on a custom OCT detection dataset using Ultralytics.

Usage (example):

  python -m utilities.yolo_train \
    --data /path/to/oct.yaml \
    --model yolov8n.pt \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --project runs/yolo \
    --name oct_yolov8n

The YAML should follow Ultralytics format:

names: ["roi"]                     # list of class names
path: /abs/or/rel/dataset/root      # optional; base path for relative splits
train: images/train                 # or an absolute path to train images
val: images/val                     # or an absolute path to val images

And corresponding YOLO labels under labels/train and labels/val (txt per image):

<class_id> <x_center> <y_center> <width> <height>   # all normalized to [0,1]

If you don’t have box annotations yet, start with a single class (e.g., "roi")
and create boxes for regions of interest. After training, pass the resulting
weights (.pt) to main.py via --yolo_weights to enable ROI-guided classification.
"""

from __future__ import annotations

import argparse
import os
from ultralytics import YOLO


def train_yolo(
    data_yaml: str,
    model: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = -1,  # auto
    device: str | int | None = None,
    project: str = "runs/yolo",
    name: str = "oct",
    pretrained: bool = True,
    workers: int = 8,
):
    assert os.path.exists(data_yaml), f"Data YAML not found: {data_yaml}"

    model_obj = YOLO(model)
    results = model_obj.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        pretrained=pretrained,
        workers=workers,
    )

    # Ultralytics returns a Results object; best weights are typically at:
    # {project}/{name}/weights/best.pt
    best = os.path.join(project, name, "weights", "best.pt")
    if os.path.exists(best):
        print(f"Training complete. Best weights: {best}")
    else:
        print("Training complete. Check run directory for weights.")

    return results


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO on OCT detection data")
    p.add_argument("--data", required=True, help="Path to Ultralytics data YAML")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO model or weights to start from")
    p.add_argument("--epochs", type=int, default=100, help="Training epochs")
    p.add_argument("--imgsz", type=int, default=640, help="Input image size")
    p.add_argument("--batch", type=int, default=-1, help="Batch size (-1 auto)")
    p.add_argument("--device", default=None, help="Device index or string (e.g., 0, 'cpu')")
    p.add_argument("--project", default="runs/yolo", help="Project directory for runs")
    p.add_argument("--name", default="oct", help="Run name")
    p.add_argument("--no-pretrained", action="store_true", help="Disable pretrained backbone")
    p.add_argument("--workers", type=int, default=8, help="Data loading workers")
    return p.parse_args()


def main():
    args = parse_args()
    train_yolo(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=not args.no_pretrained,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()


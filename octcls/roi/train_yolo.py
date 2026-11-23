from __future__ import annotations

import argparse
import os
from ultralytics import YOLO


def train_yolo(
    data_yaml: str,
    model: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = -1,
    device: str | int | None = None,
    project: str = "runs/yolo",
    name: str = "oct",
    pretrained: bool = True,
    workers: int = 8,
):
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
    best = os.path.join(project, name, "weights", "best.pt")
    if os.path.exists(best):
        print(f"Training complete. Best weights: {best}")
    else:
        print("Training complete. Check run directory for weights.")
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO on OCT detection data")
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=-1)
    p.add_argument("--device", default=None)
    p.add_argument("--project", default="runs/yolo")
    p.add_argument("--name", default="oct")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--workers", type=int, default=8)
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


from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from octcls.data.roi_dataset import ROIDataset
from octcls.data.transforms import TrimBorders
from octcls.cam.gradcam import make_cam

from octcls.models.VisionTransformer import ViTClassifier
from octcls.models.CNN import CNNClassifier
from octcls.models.VGG16 import VGG16Classifier
from octcls.models.ResNet50 import ResNet50Classifier
from octcls.models.CNN_to_VisionTransformer import CNN_ViT_Classifier


def build_model(model_type: str, num_classes: int):
    mt = model_type.lower()
    if mt == "vit":
        return ViTClassifier(num_classes=num_classes)
    if mt == "cnn":
        return CNNClassifier(num_classes=num_classes)
    if mt == "vgg16":
        return VGG16Classifier(num_classes=num_classes)
    if mt == "resnet50":
        return ResNet50Classifier(num_classes=num_classes)
    if mt == "cnnvit":
        return CNN_ViT_Classifier(num_classes=num_classes)
    raise ValueError(f"Unsupported model_type: {model_type}")


def _pick_target_layer(model: nn.Module):
    if hasattr(model, "cnn"):
        root = model.cnn
    elif hasattr(model, "features"):
        root = model.features
    else:
        root = model
    convs = [m for m in root.modules() if isinstance(m, nn.Conv2d)]
    for m in reversed(convs):
        if getattr(m, "kernel_size", None) != (1, 1):
            return m
    return convs[-1] if convs else None


def make_cam_extractor(model: nn.Module, model_type: str):
    return make_cam(model)


def extract_boxes_from_cam(cam: np.ndarray, percentile: float, top_k: int, min_area_frac: float) -> List[Tuple[int,int,int,int]]:
    H, W = cam.shape
    thr = float(np.percentile(cam, percentile))
    mask = (cam >= thr).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    comps: List[Tuple[float,Tuple[int,int,int,int]]] = []
    min_area = int(max(1, min_area_frac * H * W))
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if (w * h) < min_area:
            continue
        score = float(cam[y:y+h, x:x+w].mean())
        comps.append((score, (x, y, x + w, y + h)))
    comps.sort(key=lambda t: -t[0])
    return [b for _, b in comps[:top_k]]


def map_224_box_to_orig(xyxy_224, orig_w, orig_h, trim_top, trim_bottom):
    x1, y1, x2, y2 = xyxy_224
    x1o = x1 * (orig_w / 224.0)
    x2o = x2 * (orig_w / 224.0)
    effective_h = max(1, (orig_h - trim_top - trim_bottom))
    y1o = y1 * (effective_h / 224.0) + trim_top
    y2o = y2 * (effective_h / 224.0) + trim_top
    return (float(x1o), float(y1o), float(x2o), float(y2o))


def xyxy_to_yolo(x1, y1, x2, y2, W, H):
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    xc = x1 + w / 2.0
    yc = y1 + h / 2.0
    return (xc / W, yc / H, w / W, h / H)


def generate_cam_pseudo_labels(
    img_dir: str,
    csv_path: str,
    model_type: str,
    checkpoint: str,
    labels_out: str,
    top_k: int = 3,
    percentile: float = 92.0,
    min_area_frac: float = 0.005,
    trim_top: int = 12,
    trim_bottom: int = 12,
    batch_size: int = 32,
):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    transform = transforms.Compose([
        TrimBorders(top=trim_top, bottom=trim_bottom),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    ds = ROIDataset(csv_path, img_dir, transform, use_yolo=False)
    num_classes = len(ds.label_to_idx)

    model = build_model(model_type, num_classes)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()
    cam_extractor = make_cam_extractor(model, model_type)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out_root = Path(labels_out).expanduser().resolve()
    img_root = Path(img_dir).expanduser().resolve()

    for batch in loader:
        images, _, meta = batch
        B = images.size(0)
        for j in range(B):
            img = images[j:j+1].to(device)
            with torch.set_grad_enabled(True):
                heat = cam_extractor(img, class_idx=None)
            cam_np = heat.detach().cpu().numpy() if isinstance(heat, torch.Tensor) else heat

            boxes_224 = extract_boxes_from_cam(cam_np, percentile, top_k, min_area_frac)

            path = meta["path"][j] if isinstance(meta, dict) else meta[j]["path"]
            from PIL import Image as _PILImage
            with _PILImage.open(path) as im:
                W, H = im.size
            mapped = [map_224_box_to_orig(b, W, H, trim_top, trim_bottom) for b in boxes_224]

            rel = Path(path).resolve().relative_to(img_root)
            out_file = (out_root / rel).with_suffix('.txt')
            out_file.parent.mkdir(parents=True, exist_ok=True)
            lines: List[str] = []
            for (x1, y1, x2, y2) in mapped:
                xc, yc, w, h = xyxy_to_yolo(x1, y1, x2, y2, W, H)
                if w <= 0 or h <= 0:
                    continue
                lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            with open(out_file, 'w') as f:
                f.writelines(lines)

    try:
        cam_extractor.release()
    except Exception:
        pass

    print(f"Wrote YOLO pseudo-labels under: {out_root}")


def parse_args():
    from octcls.cli._common import add_dataset_args

    p = argparse.ArgumentParser(description="Generate YOLO pseudo-labels from classifier CAMs")
    add_dataset_args(p)
    p.add_argument("--model_type", required=True, choices=["vit", "cnn", "vgg16", "resnet50", "cnnvit"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--labels_out", required=True)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--percentile", type=float, default=85.0)
    p.add_argument("--min_area_frac", type=float, default=0.01)
    p.add_argument("--trim_top", type=int, default=12)
    p.add_argument("--trim_bottom", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    generate_cam_pseudo_labels(
        img_dir=args.img_dir,
        csv_path=args.csv_path,
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        labels_out=args.labels_out,
        top_k=args.top_k,
        percentile=args.percentile,
        min_area_frac=args.min_area_frac,
        trim_top=args.trim_top,
        trim_bottom=args.trim_bottom,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

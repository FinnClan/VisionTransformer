"""
Generate YOLO-format pseudo-labels from classifier CAM heatmaps.

Steps per image:
 1) Run the trained classifier and compute a CAM heatmap (224x224).
 2) Threshold the CAM (percentile) and extract up to top-K connected components.
 3) Convert each component to a bounding box, filter small boxes.
 4) Map 224x224 coords back to original image size, accounting for TrimBorders.
 5) Write YOLO .txt file with normalized xc,yc,w,h values.

Example:

  python -m utilities.cam_to_yolo \
    --img_dir /Users/jimmy/projects/octdl_unsplit \
    --csv_path /Users/jimmy/projects/octid_octdl_complete_extensions.csv \
    --model_type vit \
    --checkpoint vit_classifier.pth \
    --labels_out /Users/jimmy/projects/oct_cam_labels \
    --top_k 3 --percentile 85 --min_area_frac 0.01 --trim_top 12 --trim_bottom 12
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torchvision import transforms

from utilities.ROIDataset import ROIDataset
from utilities.ViTGradCAM import ViTGradCAM
from utilities.gradcam import GradCAM

# Models
from models.VisionTransformer import ViTClassifier
from models.CNN import CNNClassifier
from models.VGG16 import VGG16Classifier
from models.ResNet50 import ResNet50Classifier
from models.CNN_to_VisionTransformer import CNN_ViT_Classifier


class TrimBorders:
    def __init__(self, top=12, bottom=12):
        self.top, self.bottom = top, bottom
    def __call__(self, img):
        w, h = img.size
        return img.crop((0, self.top, w, h - self.bottom))


def build_model(model_type: str, num_classes: int):
    model_type = model_type.lower()
    if model_type == "vit":
        return ViTClassifier(num_classes=num_classes)
    elif model_type == "cnn":
        return CNNClassifier(num_classes=num_classes)
    elif model_type == "vgg16":
        return VGG16Classifier(num_classes=num_classes)
    elif model_type == "resnet50":
        return ResNet50Classifier(num_classes=num_classes)
    elif model_type == "cnnvit":
        return CNN_ViT_Classifier(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def make_cam_extractor(model, model_type: str):
    if model_type.lower() == "vit":
        return ViTGradCAM(model.vit, device=next(model.parameters()).device)
    # Fallback to last CNN conv
    from utilities.train import pick_target_layer
    tl = pick_target_layer(model)
    return GradCAM(model, tl)


def extract_boxes_from_cam(cam: np.ndarray, percentile: float, top_k: int, min_area_frac: float):
    H, W = cam.shape
    thr = np.percentile(cam, percentile)
    mask = (cam >= thr).astype(np.uint8) * 255

    # remove tiny noise by morphology (optional)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    comps = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < max(1, int(min_area_frac * H * W)):
            continue
        # score = sum of heatmap over the box
        box_cam = cam[y:y+h, x:x+w]
        score = float(box_cam.mean())
        comps.append((score, (x, y, x + w, y + h)))
    comps.sort(key=lambda t: -t[0])
    return [b for _, b in comps[:top_k]]


def map_224_box_to_orig(xyxy_224, orig_w, orig_h, trim_top, trim_bottom):
    x1, y1, x2, y2 = xyxy_224
    # Width mapping
    x1o = x1 * (orig_w / 224.0)
    x2o = x2 * (orig_w / 224.0)
    # Height mapping with TrimBorders
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


def parse_args():
    p = argparse.ArgumentParser(description="Generate YOLO pseudo-labels from classifier CAMs")
    p.add_argument("--img_dir", required=True)
    p.add_argument("--csv_path", required=True)
    p.add_argument("--model_type", required=True, choices=["vit","cnn","vgg16","resnet50","cnnvit"]) 
    p.add_argument("--checkpoint", required=True, help="Path to trained classifier weights (.pth)")
    p.add_argument("--labels_out", required=True, help="Output root for YOLO .txt labels")
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--percentile", type=float, default=85.0)
    p.add_argument("--min_area_frac", type=float, default=0.01)
    p.add_argument("--trim_top", type=int, default=12)
    p.add_argument("--trim_bottom", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    transform = transforms.Compose([
        TrimBorders(top=args.trim_top, bottom=args.trim_bottom),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    ds = ROIDataset(args.csv_path, args.img_dir, transform, use_yolo=False)
    num_classes = len(ds.label_to_idx)

    model = build_model(args.model_type, num_classes)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    cam_extractor = make_cam_extractor(model, args.model_type)

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    labels_out = Path(args.labels_out).expanduser().resolve()

    inv_mean = [0.5, 0.5, 0.5]
    inv_std = [0.5, 0.5, 0.5]

    for batch in loader:
        images, _, meta = batch
        B = images.size(0)
        for j in range(B):
            img = images[j:j+1].to(device)
            # Enable grads for CAM
            with torch.set_grad_enabled(True):
                heat = cam_extractor(img, class_idx=None)  # (224,224) numpy
            if isinstance(heat, torch.Tensor):
                cam_np = heat.detach().cpu().numpy()
            else:
                cam_np = heat

            # extract boxes
            boxes_224 = extract_boxes_from_cam(cam_np, args.percentile, args.top_k, args.min_area_frac)

            # map to original
            path = meta["path"][j] if isinstance(meta, dict) else meta[j]["path"]
            from PIL import Image as _PILImage
            with _PILImage.open(path) as im:
                W, H = im.size
            mapped = [map_224_box_to_orig(b, W, H, args.trim_top, args.trim_bottom) for b in boxes_224]

            # write YOLO .txt
            rel = Path(path).resolve().relative_to(Path(args.img_dir).expanduser().resolve())
            out_file = (labels_out / rel).with_suffix('.txt')
            out_file.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for (x1, y1, x2, y2) in mapped:
                xc, yc, w, h = xyxy_to_yolo(x1, y1, x2, y2, W, H)
                # single class 'roi' -> id 0
                if w <= 0 or h <= 0:
                    continue
                lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            if lines:
                with open(out_file, 'w') as f:
                    f.writelines(lines)
            else:
                # write empty to signal no object
                with open(out_file, 'w') as f:
                    f.write("")

    # Clean hooks
    try:
        cam_extractor.release()
    except Exception:
        pass

    print(f"Wrote YOLO pseudo-labels under: {labels_out}")


if __name__ == "__main__":
    main()

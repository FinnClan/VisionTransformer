from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from octcls.data import collate_with_meta, default_transform
from octcls.data.roi_dataset import ROIDataset
from octcls.cli._common import add_dataset_args, add_model_args, add_yolo_args, add_perf_args

from octcls.models.CNN import CNNClassifier
from octcls.models.CNN_to_VisionTransformer import CNN_ViT_Classifier
from octcls.models.ResNet50 import ResNet50Classifier
from octcls.models.VGG16 import VGG16Classifier
from octcls.models.VisionTransformer import ViTClassifier


def build_model(model_type: str, num_classes: int):
    mt = model_type.lower()
    if mt in ("vit", "yolovit"):
        return ViTClassifier(num_classes=num_classes)
    if mt in ("cnn", "yolocnn"):
        return CNNClassifier(num_classes=num_classes)
    if mt == "vgg16":
        return VGG16Classifier(num_classes=num_classes)
    if mt == "resnet50":
        return ResNet50Classifier(num_classes=num_classes)
    if mt == "cnnvit":
        return CNN_ViT_Classifier(num_classes=num_classes)
    raise ValueError(f"Unknown model type: {model_type}")


def parse_args():
    p = argparse.ArgumentParser(description="Batch inference; writes predictions to CSV")
    add_dataset_args(p)
    add_model_args(p)
    p.add_argument('--checkpoint', required=True)
    add_yolo_args(p)
    p.set_defaults(batch_size=64)
    add_perf_args(p)
    p.add_argument('--label_map', type=Path, help='Training label-to-index JSON; defaults to the checkpoint sidecar')
    p.add_argument('--no-include-global', dest='include_global', action='store_false', default=True,
                   help='Disable adding a global full-image crop alongside YOLO crops')
    p.add_argument('--output_csv', required=True)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    use_yolo = args.model_type.lower() in ("yolocnn", "yolovit")

    ds = ROIDataset(
        args.csv_path, args.img_dir, default_transform(),
        use_yolo=use_yolo, yolo_model_path=args.yolo_weights,
        yolo_topk=args.yolo_topk, yolo_conf=args.yolo_conf, yolo_iou=args.yolo_iou,
        yolo_max_det=args.yolo_max_det, yolo_min_conf=args.yolo_min_conf,
        yolo_device=args.yolo_device, yolo_cache_dir=args.yolo_cache_dir,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_with_meta, num_workers=args.num_workers, pin_memory=args.pin_memory)

    checkpoint_path = Path(args.checkpoint)
    sidecar_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_labels.json")
    if args.label_map is not None or sidecar_path.is_file():
        with (args.label_map or sidecar_path).open() as f:
            label_to_idx = json.load(f)
        inv_label = {int(idx): label for label, idx in label_to_idx.items()}
    else:
        inv_label = {idx: label for label, idx in ds.label_to_idx.items()}

    model = build_model(args.model_type, num_classes=len(inv_label))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()
    rows = []

    with torch.no_grad():
        for batch in loader:
            images, labels, meta = batch
            images = images.to(device)
            B = images.size(0)

            # Aggregate over ROIs as in training
            out_logits = []
            for j in range(B):
                img_j = images[j]
                bb_list = None; bb_224 = None
                if isinstance(meta, dict):
                    bb_list = meta.get("bboxes_224", [None] * B)[j]
                    bb_224 = meta.get("bbox_224", [None] * B)[j]
                elif isinstance(meta, (list, tuple)) and len(meta) > j and isinstance(meta[j], dict):
                    bb_list = meta[j].get("bboxes_224", None)
                    bb_224 = meta[j].get("bbox_224", None)
                if not bb_list:
                    bb_list = []
                    if bb_224 is not None and not (isinstance(bb_224, (list, tuple)) and len(bb_224) >= 4 and all(v == -1 for v in bb_224[:4])):
                        bb_list = [bb_224]
                if args.include_global:
                    bb_list = list(bb_list) + [None]
                # build crops
                from octcls.train.loop import _crop_and_resize_tensor
                crops = []
                for bb in bb_list:
                    if bb is None or (isinstance(bb, (list, tuple)) and len(bb) >= 4 and all(v == -1 for v in bb[:4])):
                        continue
                    crops.append(_crop_and_resize_tensor(images[j], bb, out_hw=(224, 224)))
                if not crops:
                    crops = [_crop_and_resize_tensor(images[j], None, out_hw=(224, 224))]
                crops_t = torch.stack(crops, dim=0).to(device)
                logits_k = model(crops_t)
                if args.roi_agg == 'max':
                    logits = logits_k.max(dim=0).values
                else:
                    logits = logits_k.mean(dim=0)
                out_logits.append(logits)
            outputs = torch.stack(out_logits, dim=0)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            # write rows
            paths = meta["path"] if isinstance(meta, dict) else [m.get("path") for m in meta]
            for p, pi, prob_vec in zip(paths, preds.cpu().tolist(), probs.cpu().tolist()):
                rows.append({
                    "path": p,
                    "pred_idx": int(pi),
                    "pred_label": inv_label.get(int(pi), str(pi)),
                    "pred_prob": float(max(prob_vec)),
                })

    # write CSV
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "pred_idx", "pred_label", "pred_prob"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote predictions: {args.output_csv}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from octcls.train import train_classifier
from octcls.utils import set_seed
from octcls.cli._common import (
    add_dataset_args, add_model_args, add_yolo_args, add_perf_args,
    add_heatmap_arg, add_bootstrap_args,
)

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
    p = argparse.ArgumentParser(description="Train classifier with optional YOLO ROI hints (top-K aggregation)")
    add_dataset_args(p)
    add_model_args(p)
    add_yolo_args(p)
    add_heatmap_arg(p)
    add_perf_args(p)
    p.add_argument('--max_overlay_box_frac', type=float, default=0.95,
                   help='Skip CAM overlays when a YOLO box covers >= this fraction of the image')
    p.add_argument('--save_best', action='store_true', help='Save best checkpoint (by val acc) to the output file')
    p.add_argument('--no-include-global', dest='include_global', action='store_false', default=True,
                   help='Disable adding a global full-image crop alongside YOLO crops')
    add_bootstrap_args(p)
    # Optional external test set (cross-dataset evaluation)
    p.add_argument('--test_img_dir', default=None, help='Optional: image root for evaluation after training')
    p.add_argument('--test_csv_path', default=None, help='Optional: CSV/XLSX for evaluation after training')
    args = p.parse_args()
    if args.epochs < 1:
        p.error("--epochs must be at least 1")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    # quick dataset peek to determine classes
    from octcls.data.roi_dataset import ROIDataset
    from octcls.data import default_transform
    ds_tmp = ROIDataset(args.csv_path, args.img_dir, default_transform(), use_yolo=False, verbose=False)
    num_classes = len(ds_tmp.label_to_idx)

    model = build_model(args.model_type, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    use_yolo = args.model_type.lower() in ("yolocnn", "yolovit")

    out_name = {
        "vit": "vit_classifier.pth",
        "yolovit": "vit_classifier.pth",
        "cnn": "cnn_classifier.pth",
        "yolocnn": "cnn_classifier.pth",
        "vgg16": "vgg16_classifier.pth",
        "resnet50": "resnet50_classifier.pth",
        "cnnvit": "cnn_vit_classifier.pth",
    }[args.model_type.lower()]

    checkpoint_path = Path(out_name)
    label_map_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_labels.json")
    with TemporaryDirectory(prefix=".octcls-checkpoint-", dir=checkpoint_path.parent) as output_dir:
        pending_checkpoint = Path(output_dir) / checkpoint_path.name
        pending_label_map = Path(output_dir) / label_map_path.name
        trained = train_classifier(
            args.img_dir,
            args.csv_path,
            model,
            device=device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            use_yolo=use_yolo,
            yolo_weights=args.yolo_weights,
            yolo_topk=args.yolo_topk,
            yolo_conf=args.yolo_conf,
            yolo_iou=args.yolo_iou,
            yolo_max_det=args.yolo_max_det,
            yolo_min_conf=args.yolo_min_conf,
            yolo_device=args.yolo_device,
            yolo_cache_dir=args.yolo_cache_dir,
            heatmap_every=args.heatmap_every,
            roi_agg=args.roi_agg,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            save_best_path=str(pending_checkpoint) if args.save_best else None,
            include_global=args.include_global,
            max_overlay_box_frac=args.max_overlay_box_frac,
        )
        if not args.save_best:
            torch.save(trained.state_dict(), pending_checkpoint)
        elif args.test_img_dir and args.test_csv_path:
            model.load_state_dict(torch.load(pending_checkpoint, map_location="cpu"))

        with pending_label_map.open("w") as f:
            json.dump({str(k): int(v) for k, v in ds_tmp.label_to_idx.items()}, f)
        pending_checkpoint.replace(checkpoint_path)
        pending_label_map.replace(label_map_path)
    from octcls.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info(f"Saved model to {out_name}")

    # Optional: evaluate on separate test set (cross-dataset)
    if args.test_img_dir and args.test_csv_path:
        logger.info("[eval] Evaluating trained model on external test set...")
        from octcls.data import default_transform, collate_with_meta
        from octcls.data.roi_dataset import ROIDataset
        use_yolo = args.model_type.lower() in ("yolocnn", "yolovit")
        test_ds = ROIDataset(
            args.test_csv_path, args.test_img_dir, default_transform(),
            use_yolo=use_yolo, yolo_model_path=args.yolo_weights,
            yolo_topk=args.yolo_topk, yolo_conf=args.yolo_conf, yolo_iou=args.yolo_iou,
            yolo_max_det=args.yolo_max_det, yolo_min_conf=args.yolo_min_conf,
            yolo_device=args.yolo_device, yolo_cache_dir=args.yolo_cache_dir,
            verbose=False,
        )
        # Check label compatibility
        if set(test_ds.label_to_idx.keys()) != set(ds_tmp.label_to_idx.keys()):
            logger.warning("[eval] Label sets differ between train and test; skipping evaluation.")
        else:
            test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                                     collate_fn=collate_with_meta,
                                                     num_workers=args.num_workers, pin_memory=args.pin_memory)
            device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
            model.to(device).eval()
            total, correct = 0, 0
            all_labels_list, all_preds_list = [], []
            all_probs_list = []
            import torch as _torch
            from octcls.train.loop import _crop_and_resize_tensor
            with _torch.no_grad():
                for batch in test_loader:
                    images, labels, meta = batch
                    images = images.to(device)
                    B = images.size(0)
                    total += B
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
                        crops = []
                        for bb in bb_list:
                            if bb is None or (isinstance(bb, (list, tuple)) and len(bb) >= 4 and all(v == -1 for v in bb[:4])):
                                continue
                            crops.append(_crop_and_resize_tensor(images[j], bb, out_hw=(224, 224)))
                        if not crops:
                            crops = [_crop_and_resize_tensor(images[j], None, out_hw=(224, 224))]
                        crops_t = _torch.stack(crops, dim=0).to(device)
                        logits_k = model(crops_t)
                        if args.roi_agg == 'max':
                            logits = logits_k.max(dim=0).values
                        else:
                            logits = logits_k.mean(dim=0)
                        out_logits.append(logits)
                    outputs = _torch.stack(out_logits, dim=0)
                    probs = _torch.softmax(outputs, dim=1).cpu()
                    preds = probs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    all_labels_list.append(labels.cpu())
                    all_preds_list.append(preds.cpu())
                    all_probs_list.append(probs)
            acc = correct / max(1, total)
            import numpy as _np
            from sklearn.metrics import f1_score as _f1, confusion_matrix as _cm, roc_auc_score as _auc
            y_true = _torch.cat(all_labels_list).numpy()
            y_pred = _torch.cat(all_preds_list).numpy()
            y_prob = _torch.cat(all_probs_list).numpy()
            f1_macro = _f1(y_true, y_pred, average='macro')
            f1_micro = _f1(y_true, y_pred, average='micro')
            # Confusion matrix with class order by training label map
            class_names = [k for k, v in sorted(ds_tmp.label_to_idx.items(), key=lambda kv: kv[1])]
            cm = _cm(y_true, y_pred, labels=list(range(len(class_names))))
            # AUC (may fail if a class is missing in test labels)
            try:
                auc_macro = _auc(y_true, y_prob, multi_class='ovr', average='macro')
                auc_micro = _auc(y_true, y_prob, multi_class='ovr', average='micro')
            except Exception as e:
                auc_macro = auc_micro = float('nan')
                logger.warning(f"[eval] AUC could not be computed: {e}")

            logger.info(f"[eval] External test: acc={acc:.4f} macroF1={f1_macro:.4f} microF1={f1_micro:.4f} macroAUC={auc_macro:.4f} microAUC={auc_micro:.4f} (n={total})")
            # Pretty-print confusion matrix
            cm_rows = cm.tolist()
            header = 'cm\\pred ' + ' '.join(f"{name:>8}" for name in class_names)
            logger.info(header)
            for name, row in zip(class_names, cm_rows):
                logger.info(f"{name:>7} " + ' '.join(f"{val:>8}" for val in row))

    # Optional: CAM -> YOLO bootstrap
    if args.bootstrap_yolo:
        from octcls.cam.gen_labels import generate_cam_pseudo_labels
        from octcls.data.prepare_split import SplitConfig, prepare_yolo_split
        from octcls.roi.train_yolo import train_yolo

        labels_out = Path(args.bootstrap_labels_out).expanduser().resolve()
        out_root = Path(args.bootstrap_out_root).expanduser().resolve()

        # 1) Generate pseudo-labels from CAMs
        logger.info("[bootstrap] Generating CAM -> YOLO labels...")
        generate_cam_pseudo_labels(
            img_dir=args.img_dir,
            csv_path=args.csv_path,
            model_type=args.model_type if args.model_type != 'yolovit' else 'vit',
            checkpoint=out_name,
            labels_out=str(labels_out),
            top_k=args.yolo_topk,
            percentile=85.0,
            min_area_frac=0.01,
            trim_top=12,
            trim_bottom=12,
            batch_size=max(16, args.batch_size),
        )

        # 2) Prepare YOLO dataset split (symlinks; non-destructive)
        logger.info("[bootstrap] Preparing YOLO split + YAML...")
        cfg = SplitConfig(
            images_root=Path(args.img_dir).expanduser().resolve(),
            out_root=out_root,
            labels_root=labels_out,
            val_size=0.2,
            seed=args.seed,
            names=("roi",),
            copy=False,
            clobber=False,
            allow_missing_labels=True,
        )
        yaml_path = prepare_yolo_split(cfg)

        # 3) Train YOLO and report best.pt
        logger.info("[bootstrap] Training YOLO on CAM labels...")
        run_name = args.bootstrap_name or f"bootstrap_{args.model_type.lower()}"
        train_yolo(
            data_yaml=str(yaml_path),
            model=args.bootstrap_yolo_model,
            epochs=args.bootstrap_epochs,
            imgsz=640,
            batch=-1,
            device=args.yolo_device,
            project=args.bootstrap_project,
            name=run_name,
            pretrained=True,
            workers=max(4, args.num_workers),
        )
        best = Path(args.bootstrap_project) / run_name / "weights" / "best.pt"
        logger.info(f"[bootstrap] YOLO best weights ready: {best}")


if __name__ == "__main__":
    main()

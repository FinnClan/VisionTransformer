from __future__ import annotations

from typing import Optional
import torch
from torch.utils.data import DataLoader

from octcls.data import collate_with_meta, default_transform
from octcls.data.roi_dataset import ROIDataset
from octcls.train.loop import train_model


def train_classifier(
    img_dir: str,
    csv_path: str,
    model,
    *,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
    epochs: int = 50,
    use_yolo: bool = False,
    yolo_weights: str = "yolov8n.pt",
    yolo_topk: int = 2,
    yolo_conf: float = 0.25,
    yolo_iou: float = 0.7,
    yolo_max_det: int = 10,
    yolo_min_conf: float = 0.10,
    heatmap_every: int = 0,
    roi_agg: str = "mean",
    num_workers: int = 0,
    pin_memory: bool = False,
    yolo_device: str | None = None,
    yolo_cache_dir: str | None = None,
    save_best_path: str | None = None,
    include_global: bool = True,
    max_overlay_box_frac: float = 0.95,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    transform = default_transform()
    ds = ROIDataset(
        csv_path, img_dir, transform,
        use_yolo=use_yolo, yolo_model_path=yolo_weights,
        yolo_topk=yolo_topk, yolo_conf=yolo_conf, yolo_iou=yolo_iou,
        yolo_max_det=yolo_max_det, yolo_min_conf=yolo_min_conf,
        yolo_device=yolo_device, yolo_cache_dir=yolo_cache_dir,
    )

    from sklearn.model_selection import train_test_split
    idx = list(range(len(ds)))
    labels = ds.data["label_idx"].tolist()
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=50, stratify=labels)

    train_loader = DataLoader(torch.utils.data.Subset(ds, train_idx), batch_size=batch_size, shuffle=True, collate_fn=collate_with_meta, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(torch.utils.data.Subset(ds, val_idx), batch_size=batch_size, collate_fn=collate_with_meta, num_workers=num_workers, pin_memory=pin_memory)

    model.to(device)
    return train_model(device, model, train_loader, epochs=epochs,
                       heatmap_every=heatmap_every, roi_agg=roi_agg,
                       val_loader=val_loader, save_best_path=save_best_path,
                       include_global=include_global,
                       max_overlay_box_frac=max_overlay_box_frac)

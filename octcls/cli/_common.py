from __future__ import annotations

import argparse


def add_dataset_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--img_dir', required=True, help='Root directory of images (unsplit)')
    p.add_argument('--csv_path', required=True, help='CSV (or .xlsx) with file_name,label columns')


def add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--model_type', required=True,
                   choices=["vit","cnn","vgg16","resnet50","cnnvit","yolocnn","yolovit"])
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--seed', type=int, default=50)


def add_heatmap_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument('--heatmap_every', type=int, default=0,
                   help='Generate Grad-CAM overlays every N batches (0 disables)')


def add_perf_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--pin_memory', action='store_true')


def add_yolo_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--yolo_weights', default='yolov8n.pt')
    p.add_argument('--yolo_topk', type=int, default=2)
    p.add_argument('--yolo_conf', type=float, default=0.25)
    p.add_argument('--yolo_iou', type=float, default=0.7)
    p.add_argument('--yolo_max_det', type=int, default=10)
    p.add_argument('--yolo_min_conf', type=float, default=0.10)
    p.add_argument('--yolo_device', default=None)
    p.add_argument('--yolo_cache_dir', default='.yolo_cache')
    p.add_argument('--roi_agg', choices=['mean','max'], default='mean')


def add_bootstrap_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--bootstrap_yolo', action='store_true', help='Train YOLO from CAM pseudo-labels after training')
    p.add_argument('--bootstrap_labels_out', default='./cam_labels')
    p.add_argument('--bootstrap_out_root', default='./oct_yolo')
    p.add_argument('--bootstrap_yolo_model', default='yolov8n.pt')
    p.add_argument('--bootstrap_epochs', type=int, default=100)
    p.add_argument('--bootstrap_project', default='runs/yolo')
    p.add_argument('--bootstrap_name', default=None)

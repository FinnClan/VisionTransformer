VisionTransformer + YOLO ROI Hints

Overview
- This project trains image classifiers (ViT, CNN, VGG16, ResNet50, CNN->ViT), originally on OCT scans, and optionally uses a YOLO detector to propose Regions of Interest (ROIs) as hints ("yolovit"/"yolocnn").
- You can also bootstrap a YOLO detector from classifier Grad‑CAM heatmaps (weak supervision), then use that detector to guide the classifier.

Key Capabilities
- Train classifiers on image‑level labels from a CSV.
- Use YOLO as an ROI proposer with top‑K box aggregation (mean or max).
- Generate YOLO labels from classifier CAMs (pseudo‑labels).
- Create a YOLO dataset split (symlinks + YAML) and train YOLO (Ultralytics).
- Run batch inference and export predictions to CSV.

Project Layout
- `octcls/` - library modules and CLIs
  - `octcls/data/` - dataset (`roi_dataset.py`), transforms, collate, split
  - `octcls/train/` - training loop and trainer wrapper
  - `octcls/cam/` - Grad‑CAM for CNNs and ViTs, CAM->labels
  - `octcls/roi/` - YOLO train wrapper
  - `octcls/cli/` - CLIs (train, infer, gen CAM labels, prepare split, train YOLO)
- `octcls/models/` - model definitions (CNN, VGG16, ResNet50, ViT, CNN->ViT)
- `pyproject.toml` - package metadata, dependencies, and console scripts
- `CLI_REFERENCE.md` - CLI quick reference and tips

Installation
From the repository root, run `python -m pip install -e .` in a Python 3.8+ environment.
This installs the package and its dependencies and makes the console scripts available.
Training can use CUDA or MPS when available.

Data Expectations
- Images live under a single root folder (unsplit).
- Labels CSV has at least two columns: `file_name`, `label`.
  - `file_name` can be a filename (with or without extension), a relative path from the image root, or a mix (the loader resolves robustly)

Quickstart

1) Train a classifier (no YOLO hints)
- CNN example (unified entrypoint):
  - `python -m octcls.cli train \
     --img_dir /path/to/images \
     --csv_path /path/to/labels.csv \
     --model_type cnn \
     --epochs 50 --batch_size 32 --heatmap_every 0 --save_best`
- Other model types: `vit`, `vgg16`, `resnet50`, `cnnvit`

2) Train with YOLO hints (yolovit / yolocnn)
- Use a pretrained YOLO (e.g., `yolov8n.pt`) as ROI proposer, aggregate top‑K crops:
  - `python -m octcls.cli train \
     --img_dir /path/to/images \
     --csv_path /path/to/labels.csv \
     --model_type yolovit \
     --yolo_weights yolov8n.pt \
     --yolo_topk 3 --yolo_conf 0.05 --yolo_max_det 20 \
     --epochs 50 --batch_size 32 --save_best`

3) One‑shot YOLO Bootstrap from CAMs (optional)
- After training a classifier, automatically:
  - generate CAM‑based labels -> prepare YOLO split -> train YOLO -> print best.pt
- Example:
  - `python -m octcls.cli train \
     --img_dir /path/to/images \
     --csv_path /path/to/labels.csv \
     --model_type vit --save_best \
     --bootstrap_yolo \
     --bootstrap_labels_out ./cam_labels \
     --bootstrap_out_root ./oct_yolo \
     --bootstrap_yolo_model yolov8n.pt \
     --bootstrap_epochs 100`


Standalone Workflows

- Generate CAM->YOLO labels (pseudo‑labels):
  - `python -m octcls.cli gen-cam-labels \
     --img_dir /path/to/images \
     --csv_path /path/to/labels.csv \
     --model_type vit \
     --checkpoint vit_classifier.pth \
     --labels_out ./cam_labels --top_k 3`

- Prepare YOLO split + YAML:
  - `python -m octcls.cli prepare-yolo-split \
     --images_root /path/to/images \
     --out_root ./oct_yolo \
     --labels_root ./cam_labels --val_size 0.2`

- Train YOLO (Ultralytics):
  - `python -m octcls.cli train-yolo \
     --data ./oct_yolo/dataset.yaml --model yolov8n.pt \
     --epochs 100 --project runs/yolo --name oct_cam_bootstrap`

- Inference to CSV:
  - `python -m octcls.cli infer \
     --img_dir /path/to/images \
     --csv_path /path/to/labels.csv \
     --model_type yolovit \
     --checkpoint vit_classifier.pth \
     --yolo_weights /path/to/yolo/best.pt \
     --yolo_topk 3 \
     --output_csv preds.csv`
     

Concepts and Options
- ROI aggregation: For yolovit/yolocnn, top‑K YOLO boxes produce K crops per image.
  - `--yolo_topk`: how many boxes to keep
  - `--roi_agg mean|max`: aggregate logits over K crops

- YOLO thresholds:
  - `--yolo_conf`, `--yolo_iou`, `--yolo_max_det`, `--yolo_min_conf`

- Devices:
  - Classifier uses CUDA -> MPS -> CPU automatically.
  - YOLO device can be forced with `--yolo_device cuda:0|mps|cpu`.

- Performance:
  - `--num_workers N` and `--pin_memory` speed up I/O.
  - ROI crops are flattened across the batch for a single forward pass.

- Checkpoints and labels
  - With `--save_best`, training keeps the weights from the epoch with the highest validation accuracy. Otherwise it saves the final epoch's weights.
  - Keep the checkpoint and its `<checkpoint-stem>_labels.json` file together. Inference uses the training label map to predict across all trained classes, even when the input CSV contains only some of them. Use `--label_map` to supply the map from another location.


Unified entrypoint and console scripts
- Unified entry (subcommands): `python -m octcls.cli <command> [options]`
  - Commands: `train`, `infer`, `gen-cam-labels`, `prepare-yolo-split`, `train-yolo`

- Or use the installed console scripts:
  - `oct-train-classifier`, `oct-infer`, `oct-gen-cam`, `oct-prepare-yolo`, `oct-train-yolo`


Troubleshooting
- YOLO weights not found:
  - Use an absolute path for `--yolo_weights`, or a model name (e.g., `yolov8n.pt`) that Ultralytics can auto‑download.

- No YOLO detections / only one box:
  - Lower `--yolo_conf`, increase `--yolo_max_det`, and ensure `--yolo_topk` > 1. You can also set `--yolo_device` to `cuda:0|mps|cpu` explicitly.

- MPS quirks on Apple Silicon:
  - Set `PYTORCH_ENABLE_MPS_FALLBACK=1` to allow CPU fallback for unsupported ops.

- CAM overlay errors during training:
  - Overlays are disabled with `--heatmap_every 0`. The code creates transient CAM hooks only during overlay generation; if you see in‑place op issues, reduce the frequency or disable overlays.

- .xlsx label files fail to load:
  - Ensure `openpyxl` is installed (included in the package dependencies).

- Out‑of‑memory (OOM):
  - Reduce `--batch_size` and/or `--yolo_topk`, or use a smaller model type.

- Reproducibility:
  - Use `--seed` to set Python/NumPy/Torch seeds.

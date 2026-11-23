CLI Reference

Purpose
- A concise, copy‑pasteable set of commands for the most common tasks: training, YOLO‑hinted training, CAM->labels, YOLO split/training, and inference.

Train Classifiers
- CNN (no YOLO hints):
  - `python -m octcls.cli train \
     --img_dir /path/to/images \
     --csv_path /path/to/labels.csv \
     --model_type cnn \
     --epochs 50 --batch_size 32 --heatmap_every 0 --save_best`
- Other model types: `vit`, `vgg16`, `resnet50`, `cnnvit`

Train with YOLO ROI Hints
- ViT with YOLO proposals (top‑K crops, aggregated):
  - `python -m octcls.cli train \
     --img_dir /path/to/images \
     --csv_path /path/to/labels.csv \
     --model_type yolovit \
     --yolo_weights yolov8n.pt \
     --yolo_topk 3 --yolo_conf 0.05 --yolo_max_det 20 \
     --epochs 50 --batch_size 32 --save_best`
- CNN with YOLO proposals: `--model_type yolocnn` (same flags)

One‑Shot YOLO Bootstrap From CAMs
- Train classifier -> CAM labels -> YOLO split -> train YOLO -> prints best.pt:
  - `python -m octcls.cli train \
     --img_dir /path/to/images \
     --csv_path /path/to/labels.csv \
     --model_type vit --save_best \
     --bootstrap_yolo \
     --bootstrap_labels_out ./cam_labels \
     --bootstrap_out_root ./oct_yolo \
     --bootstrap_yolo_model yolov8n.pt \
     --bootstrap_epochs 100`

Standalone CAM->YOLO Labels
- `python -m octcls.cli gen-cam-labels \
   --img_dir /path/to/images \
   --csv_path /path/to/labels.csv \
   --model_type vit \
   --checkpoint vit_classifier.pth \
   --labels_out ./cam_labels --top_k 3`

Prepare YOLO Dataset Split
- `python -m octcls.cli prepare-yolo-split \
   --images_root /path/to/images \
   --out_root ./oct_yolo \
   --labels_root ./cam_labels --val_size 0.2`

Train YOLO (Ultralytics)
- `python -m octcls.cli train-yolo \
   --data ./oct_yolo/dataset.yaml --model yolov8n.pt \
   --epochs 50 --project runs/yolo --name oct_cam_bootstrap`

Batch Inference to CSV
- `python -m octcls.cli infer \
   --img_dir /path/to/images \
   --csv_path /path/to/labels.csv \
   --model_type yolovit \
   --checkpoint vit_classifier.pth \
   --yolo_weights /path/to/yolo/best.pt \
   --yolo_topk 3 \
   --output_csv preds.csv`

Useful Options
- Performance: `--num_workers N`, `--pin_memory`
- YOLO device: `--yolo_device cuda:0|mps|cpu`
- Persist YOLO detections: `--yolo_cache_dir .yolo_cache`
- ROI aggregation: `--yolo_topk K`, `--roi_agg mean|max`

import argparse
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
import logging
from ultralytics.utils import LOGGER
import torch
from models.CNN import CNNClassifier
from models.CNN_to_VisionTransformer import CNN_ViT_Classifier
from models.ResNet50 import ResNet50Classifier
from models.VGG16 import VGG16Classifier
from models.VisionTransformer import ViTClassifier
from utilities.ROIDataset import ROIDataset
from utilities.train import train_model
from utilities.collate import collate_with_meta
LOGGER.setLevel(logging.WARNING)

# -----------------------
# Main Execution
# -----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a classifier on medical image data.')
    parser.add_argument('--img_dir', type=str, required=True, help='Path to the directory containing images.')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file with image labels.')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=["vit", "cnn", "vgg16", "resnet50", "cnnvit", "yolocnn", "yolovit"],
                        help='Architecture to use.')
    parser.add_argument('--yolo_weights', type=str, default='yolov8n.pt',
                        help='Path to YOLO weights to use for ROI detection when using yolocnn/yolovit.')
    parser.add_argument('--yolo_topk', type=int, default=3,
                        help='Number of top YOLO boxes to use per image as ROI hints (default: 3).')
    parser.add_argument('--yolo_agg', type=str, default='mean', choices=['mean', 'max'],
                        help='How to aggregate logits over ROI crops (default: mean).')
    parser.add_argument('--yolo_conf', type=float, default=0.15,
                        help='YOLO confidence threshold for detections (passed to Ultralytics). Default: 0.15')
    parser.add_argument('--yolo_iou', type=float, default=0.7,
                        help='YOLO NMS IoU threshold. Default: 0.7')
    parser.add_argument('--yolo_max_det', type=int, default=10,
                        help='Max detections per image to keep from YOLO. Default: 10')
    parser.add_argument('--yolo_min_conf', type=float, default=0.10,
                        help='Post-filter minimum confidence per box after YOLO (dataset-level). Default: 0.10')
    parser.add_argument('--heatmap_every', type=int, default=10,
                        help='Generate Grad-CAM overlays every N batches (0 disables). Default: 10')
    # Dataset sampling/splitting controls
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='Validation fraction for split (default: 0.2)')
    parser.add_argument('--limit_total', type=int, default=None,
                        help='If set, randomly sample this many images from the dataset before splitting.')
    parser.add_argument('--seed', type=int, default=50, help='Random seed for sampling/splitting.')

    args = parser.parse_args()

    image_dir = args.img_dir
    csv_path = args.csv_path
    model_type = args.model_type.lower()

    # Determine if YOLO cropping is needed for this model
    yolo_models = ["yolocnn", "yolovit"]
    use_yolo = model_type in yolo_models

    # Transform
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    # ])

    class TrimBorders:
        def __init__(self, top=8, bottom=8):
            self.top, self.bottom = top, bottom

        def __call__(self, img):  # img is PIL.Image
            w, h = img.size
            return img.crop((0, self.top, w, h - self.bottom))


    # In your pipeline, before Resize:
    transform = transforms.Compose([
        TrimBorders(top=12, bottom=12),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    # Dataset
    full_dataset = ROIDataset(
        csv_path, image_dir, transform,
        use_yolo=use_yolo, yolo_model_path=args.yolo_weights,
        yolo_topk=args.yolo_topk,
        yolo_conf=args.yolo_conf, yolo_iou=args.yolo_iou, yolo_max_det=args.yolo_max_det,
        yolo_min_conf=args.yolo_min_conf,
    )

    # Optional: sample a subset before splitting (stratified by label when possible)
    all_indices = list(range(len(full_dataset)))
    # labels for stratification
    try:
        labels_all = full_dataset.data["label_idx"].tolist()
    except Exception:
        labels_all = None

    if args.limit_total is not None and args.limit_total > 0 and args.limit_total < len(all_indices):
        subset_indices, _ = train_test_split(
            all_indices,
            train_size=args.limit_total,
            random_state=args.seed,
            stratify=[labels_all[i] for i in all_indices] if labels_all is not None else None,
        )
    else:
        subset_indices = all_indices

    # Train/val split (stratified when possible)
    train_idx, val_idx = train_test_split(
        subset_indices,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=[labels_all[i] for i in subset_indices] if labels_all is not None else None,
    )
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_with_meta)
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_with_meta)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Select Model
    if model_type == "vit" or model_type == "yolovit":
        model = ViTClassifier(num_classes=len(full_dataset.label_to_idx))
        output_name = "vit_classifier.pth"
    elif model_type == "cnn" or model_type == "yolocnn":
        model = CNNClassifier(num_classes=len(full_dataset.label_to_idx))
        output_name = "cnn_classifier.pth"
    elif model_type == "vgg16":
        model = VGG16Classifier(num_classes=len(full_dataset.label_to_idx))
        output_name = "vgg16_classifier.pth"
    elif model_type == "resnet50":
        model = ResNet50Classifier(num_classes=len(full_dataset.label_to_idx))
        output_name = "resnet50_classifier.pth"
    elif model_type == "cnnvit":
        model = CNN_ViT_Classifier(num_classes=len(full_dataset.label_to_idx))
        output_name = "cnn_vit_classifier.pth"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train
    model.to(device)
    trained_model = train_model(device, model, train_loader, epochs=50,
                                heatmap_every=args.heatmap_every, roi_agg=args.yolo_agg)

    # Save
    torch.save(trained_model.state_dict(), output_name)
    print(f"Training complete. Model saved to {output_name}.")

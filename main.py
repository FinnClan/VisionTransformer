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

    args = parser.parse_args()

    image_dir = args.img_dir
    csv_path = args.csv_path
    model_type = args.model_type.lower()

    # Determine if YOLO cropping is needed for this model
    yolo_models = ["yolocnn", "yolovit"]
    use_yolo = model_type in yolo_models

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    # Dataset
    full_dataset = ROIDataset(csv_path, image_dir, transform, use_yolo=use_yolo)
    #train 80%, test 20%
    train_idx, val_idx = train_test_split(range(len(full_dataset)), test_size=0.2, random_state=50)
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

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
    trained_model = train_model(device, model, train_loader, epochs=50)

    # Save
    torch.save(trained_model.state_dict(), output_name)
    print(f"Training complete. Model saved to {output_name}.")

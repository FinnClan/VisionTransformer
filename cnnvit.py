import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from buildcnn import BuildCnn
from visiontransformer import VisionTransformer


class MedicalImageDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Load labels from CSV file
        df = pd.read_csv(labels_file)
        all_filenames = set(os.listdir(image_dir))  # Get list of available images in the directory

        # Convert categorical labels to integers
        df.iloc[:, 1] = df.iloc[:, 1].astype(str).str.strip()  # Remove extra spaces
        unique_labels = sorted(df.iloc[:, 1].unique())  # Get unique labels in sorted order
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}  # Map labels to indices
        df.iloc[:, 1] = df.iloc[:, 1].map(self.label_mapping)  # Convert labels to numbers

        # Ensure filenames have .jpg or .jpeg extension and filter out missing files
        self.labels_dict = {f"{filename}.{ext}": label for filename, label in zip(df.iloc[:, 0], df.iloc[:, 1])
                            for ext in ["jpg", "jpeg"] if f"{filename}.{ext}" in all_filenames}
        self.image_filenames = list(self.labels_dict.keys())

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("L")  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(self.labels_dict[img_name], dtype=torch.long)  # Use mapped integer directly


# Custom Transformation to Duplicate Channels
class DuplicateChannel:
    def __call__(self, x):
        return torch.cat([x, x], dim=0)  # Duplicate grayscale channel to create a 2-channel image


# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input
    transforms.ToTensor(),
    DuplicateChannel(),  # Convert grayscale to 2-channel format
    transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])  # Normalize both channels
])

# Directory with raw images
image_dir = "/Users/jimmy/projects/combined_unsplit"
# CSV with image file names in column 1 and labels in column 2
labels_file = "/Users/jimmy/projects/octid_octdl_complete_extensions.csv"

# Create dataset
dataset = MedicalImageDataset(image_dir, labels_file, transform=transform)

# Split dataset into train (70%), validation (15%), and test (15%)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
dataloader_train = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
dataloader_val = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
dataloader_test = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)


# CNN_ViT Model Definition
class CNN_ViT(nn.Module):
    def __init__(self, cnn, vit, flatten_dim=196, vit_dim=128, pooled_size=14):
        super(CNN_ViT, self).__init__()
        self.cnn = cnn  # CNN model for feature extraction
        self.vit = vit  # Vision Transformer model for classification

        # Extract different feature levels from CNN
        self.low_level_cnn = nn.Sequential(*list(cnn.children())[:1])  # First convolutional layer
        self.high_level_cnn = nn.Sequential(*list(cnn.children())[1:])  # Remaining layers

        # Adaptive Pooling to match spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((pooled_size, pooled_size))

        # Reduce channel dimensions before feeding into ViT
        self.channel_projection = nn.Conv2d(in_channels=96, out_channels=vit_dim, kernel_size=1)

        # Positional embedding for ViT input
        num_patches = (pooled_size * pooled_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, vit_dim))

        self.norm = nn.LayerNorm(vit_dim)

    def forward(self, x):
        x_low = self.low_level_cnn(x)  # Extract low-level features
        x_high = self.high_level_cnn(x_low)  # Extract high-level features

        # Ensure spatial dimensions match before concatenation
        if x_low.shape[-1] != x_high.shape[-1]:
            x_low = nn.functional.interpolate(x_low, size=x_high.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x_low, x_high], dim=1)  # Concatenate low- and high-level features
        x = self.adaptive_pool(x)  # Reduce spatial dimensions
        x = self.channel_projection(x)  # Reduce channel dimensions

        # Reshape for Vision Transformer input
        batch_size, channels, height, width = x.shape
        num_patches = height * width
        x = x.view(batch_size, num_patches, -1)
        x = self.norm(x + self.pos_embedding)  # Apply positional encoding

        return self.vit(x)  # Pass through Vision Transformer


# Training Function
def train_model(model, train_loader, epochs, criterion, optimizer, device):
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} running...")
        running_loss = 0.0
        all_preds, all_labels = [], []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')  # Weighted F1-score
        recall = recall_score(all_labels, all_preds, average='weighted')
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}")
    return model


# Initialize and Combine CNN and Vision Transformer
cnn = BuildCnn(input_channels=2, conv_layers=[(32, 3, 1), (64, 3, 1)], use_pooling=True)
vit = VisionTransformer(image_size=224, patch_size=16, num_classes=7, dim=128, depth=4, heads=4, mlp_dim=256, input_channels=32)
cnn_vit_model = CNN_ViT(cnn, vit, flatten_dim=196, vit_dim=128, pooled_size=14)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    cnn_vit_model.to(device)
    criterion = nn.CrossEntropyLoss()

    #AdamW chosen for its effectiveness with Vision Transformer models
    optimizer = torch.optim.AdamW(cnn_vit_model.parameters(), lr=0.0008)
    trained_model = train_model(cnn_vit_model, dataloader_train, dataloader_val, epochs=70, criterion=criterion, optimizer=optimizer, device=device)
    torch.save(trained_model.state_dict(), "cnn_vit_checkpoint.pth")
    print("Training complete. Model saved.")

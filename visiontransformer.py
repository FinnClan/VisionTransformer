import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self,
                 image_size=224,   # Input image size (assumes square images)
                 patch_size=16,    # Size of each patch extracted from the image
                 num_classes=7,    # Number of output classes for classification
                 dim=128,          # Dimension of patch embeddings
                 depth=4,          # Number of transformer encoder layers
                 heads=4,          # Number of attention heads per encoder layer
                 mlp_dim=256,      # Hidden dimension for MLP layers in the transformer
                 dropout=0.1,      # Dropout rate for regularization
                 input_channels=3):# Number of input channels (e.g., 3 for RGB, 1 for grayscale)

        super().__init__()

        self.dim = dim  # Store embedding dimension

        # Convert image patches into embeddings using a convolutional layer
        self.patch_embedding = nn.Conv2d(
            in_channels=input_channels,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size  # Ensures non-overlapping patches
        )

        # Learnable classification token (CLS token) added to the sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # Shape: (1, 1, dim)

        # Positional encoding to provide spatial information to the transformer
        self.pos_embedding = nn.Parameter(
            torch.randn(1, (image_size // patch_size) ** 2 + 1, dim)  # Shape: (1, num_patches + 1, dim)
        )

        # Transformer encoder consisting of multiple self-attention layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,           # Must match embedding dimension
                nhead=heads,           # Number of self-attention heads
                dim_feedforward=mlp_dim,  # Size of feedforward layers in each encoder block
                dropout=dropout        # Apply dropout for regularization
            ),
            num_layers=depth  # Stack multiple encoder layers
        )

        # Final classification head: Normalization + Fully Connected Layer
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),  # Normalize before feeding into classifier
            nn.Linear(dim, num_classes)  # Map final embedding to class logits
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # Flatten input to (batch_size, num_patches, embedding_dim)
        x = x.view(batch_size, -1, self.dim)  # Ensures correct shape for transformer input
        assert len(x.shape) == 3, f"Expected 3D input to Transformer, but got shape {x.shape}"

        # Apply transformer encoder (self-attention + feedforward layers)
        x = self.transformer(x)

        # Extract the CLS token representation for final classification
        x = x[:, 0, :]  # Shape: (batch_size, dim)

        # Apply classification head
        x = self.mlp_head(x)  # Shape: (batch_size, num_classes)

        return x

# Example usage:
# vit = VisionTransformer(image_size=224, patch_size=16, num_classes=7, dim=128, depth=4, heads=4, mlp_dim=256)
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16

# -----------------------
# Vision Transformer
# -----------------------
class ViTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit = vit_b_16(pretrained=True)
        in_features = self.vit.heads[-1].in_features
        self.vit.heads[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vit(x)
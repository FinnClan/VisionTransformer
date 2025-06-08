import torch.nn as nn
from torchvision.models import vgg16

# -----------------------
# VGG16 Classifier
# -----------------------
class VGG16Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vgg = vgg16(pretrained=True)
        in_features = self.vgg.classifier[-1].in_features

        self.vgg.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vgg(x)
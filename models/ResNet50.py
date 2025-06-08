import torch.nn as nn
from torchvision.models import resnet50

# -----------------------
# ResNet50
# -----------------------
class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        in_features = self.resnet.fc.in_features

        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
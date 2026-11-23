import torch.nn as nn
from torchvision.models import vgg16


class VGG16Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vgg = vgg16(pretrained=True)
        in_features = self.vgg.classifier[-1].in_features
        self.vgg.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vgg(x)

    def get_cam_target_layer(self):
        convs = [m for m in self.vgg.features.modules() if isinstance(m, nn.Conv2d)]
        if not convs:
            return None
        for m in reversed(convs):
            if getattr(m, "kernel_size", None) != (1, 1):
                return m
        return convs[-1]


import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16


class CNN_ViT_Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.project = nn.Conv2d(512, 768, kernel_size=1)
        self.vit = vit_b_16(pretrained=True)
        self.vit.conv_proj = nn.Identity()
        self.vit._process_input = lambda x: x
        self.vit.heads = nn.Linear(self.vit.heads.head.in_features, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.project(x)
        x = x.flatten(2).transpose(1, 2)
        return self.vit(x)

    def get_cam_target_layer(self):
        convs = [m for m in self.cnn.modules() if isinstance(m, nn.Conv2d)]
        if not convs:
            return None
        for m in reversed(convs):
            if getattr(m, "kernel_size", None) != (1, 1):
                return m
        return convs[-1]


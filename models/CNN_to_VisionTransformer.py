import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16

# -----------------------
# CNN to ViT
# -----------------------
class CNN_ViT_Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # CNN feature extractor (down to 14x14)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Match ViT hidden dimension
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
from torchvision import transforms


class TrimBorders:
    def __init__(self, top: int = 12, bottom: int = 12):
        self.top = int(top)
        self.bottom = int(bottom)

    def __call__(self, img):
        w, h = img.size
        return img.crop((0, self.top, w, h - self.bottom))


def default_transform(img_size=(224, 224), mean=None, std=None, trim_top=12, trim_bottom=12):
    if mean is None:
        mean = [0.5, 0.5, 0.5]
    if std is None:
        std = [0.5, 0.5, 0.5]
    return transforms.Compose([
        TrimBorders(top=trim_top, bottom=trim_bottom),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


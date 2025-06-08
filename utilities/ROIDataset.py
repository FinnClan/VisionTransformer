import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class ROIDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, use_yolo=False, yolo_model_path='yolov8n.pt'):
        self.data = pd.read_excel(csv_file) if csv_file.endswith('.xlsx') else pd.read_csv(csv_file)

        # Clean column names
        self.data.columns = self.data.columns.str.strip().str.replace('"', '').str.replace("'", "").str.lower()
        print("Cleaned column names:", list(self.data.columns))

        self.image_dir = image_dir
        self.transform = transform
        self.use_yolo = use_yolo

        # Load YOLO only if requested
        if self.use_yolo:
            assert YOLO is not None, "Ultralytics YOLO is not installed."
            self.yolo_model = YOLO(yolo_model_path)

        # Build image map from directory
        all_images = []
        for root, _, files in os.walk(image_dir):
            for f in files:
                full_path = os.path.join(root, f)
                all_images.append((os.path.basename(f).lower(), full_path))

        image_map = {os.path.splitext(filename)[0].lower(): full_path for filename, full_path in all_images}

        # Keep only valid entries
        valid_rows = []
        for _, row in self.data.iterrows():
            filename = os.path.splitext(str(row["file_name"]).strip().lower())[0]
            if filename in image_map:
                row["file_name"] = image_map[filename]
                valid_rows.append(row)

        self.data = pd.DataFrame(valid_rows)
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(self.data["label"].unique()))}
        self.data["label_idx"] = self.data["label"].map(self.label_to_idx)

        print("Label mapping:", self.label_to_idx)
        print(f"Filtered dataset size: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row["file_name"]
        image = Image.open(image_path).convert("RGB")

        if self.use_yolo:
            results = self.yolo_model(image)
            boxes = results[0].boxes.xyxy
            if len(boxes) > 0:
                x1, y1, x2, y2 = boxes[0]
                image = image.crop((int(x1), int(y1), int(x2), int(y2)))
            # else, nothing detected and use original image

        if self.transform:
            image = self.transform(image)

        label = row["label_idx"]
        return image, label

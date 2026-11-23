from __future__ import annotations

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def _scale_bbox(bbox, src_size, dst_size):
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    sw, sh = src_size
    dw, dh = dst_size
    sx = dw / float(sw)
    sy = dh / float(sh)
    return int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)


class ROIDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, use_yolo=False,
                 yolo_model_path='yolov8n.pt', return_path=True, yolo_topk: int = 1,
                 yolo_conf: float = 0.25, yolo_iou: float = 0.7, yolo_max_det: int = 10,
                 yolo_min_conf: float = 0.10, yolo_device: str | None = None,
                 yolo_cache_dir: str | None = None,
                 verbose: bool = True):
        self.data = pd.read_excel(csv_file) if csv_file.endswith('.xlsx') else pd.read_csv(csv_file)
        self.data.columns = (self.data.columns
                             .str.strip().str.replace('"', '').str.replace("'", "").str.lower())
        self.verbose = verbose
        if self.verbose:
            print("Cleaned column names:", list(self.data.columns))

        self.image_dir = image_dir
        self.transform = transform
        self.use_yolo = use_yolo
        self.return_path = return_path
        self.yolo_topk = max(1, int(yolo_topk)) if use_yolo else 0
        self.yolo_conf = float(yolo_conf)
        self.yolo_iou = float(yolo_iou)
        self.yolo_max_det = int(yolo_max_det)
        self.yolo_min_conf = float(yolo_min_conf)
        self._yolo_cache = {}
        self.yolo_device = yolo_device
        self.yolo_cache_dir = yolo_cache_dir

        if self.use_yolo:
            assert YOLO is not None, "Ultralytics YOLO is not installed."
            self.yolo_model = YOLO(yolo_model_path)
            try:
                self.yolo_model.fuse()
            except Exception:
                if self.verbose:
                    print("[WARN] YOLO fuse() failed; continuing with unfused model")
            if self.yolo_device is not None:
                try:
                    self.yolo_model.to(self.yolo_device)
                except Exception as exc:
                    if self.verbose:
                        print(f"[WARN] Failed to move YOLO model to {self.yolo_device}: {exc}")

        # Build robust mapping to images: prefer relative path (no extension), else unique stem
        img_dir_abs = os.path.abspath(image_dir)
        rel_map = {}
        stem_map = {}
        for root, _, files in os.walk(image_dir):
            for f in files:
                full_path = os.path.join(root, f)
                rel_key = os.path.splitext(os.path.relpath(full_path, img_dir_abs))[0].replace('\\', '/').lower()
                rel_map[rel_key] = full_path
                stem = os.path.splitext(os.path.basename(f))[0].lower()
                stem_map.setdefault(stem, []).append(full_path)

        def _map_name(name: str):
            s = str(name).strip().lower()
            noext = os.path.splitext(s)[0]
            if '/' in noext or '\\' in noext:
                key = noext.replace('\\', '/')
                return rel_map.get(key)
            paths = stem_map.get(noext, [])
            if len(paths) == 1:
                return paths[0]
            elif len(paths) > 1:
                print(f"[WARN] Ambiguous stem '{noext}' maps to multiple files; using first: {paths[0]}")
                return paths[0]
            return None

        self.data["_mapped_path"] = self.data["file_name"].apply(_map_name)
        self.data = self.data[~self.data["_mapped_path"].isna()].copy()
        self.data["file_name"] = self.data["_mapped_path"].astype(str)
        self.data.drop(columns=["_mapped_path"], inplace=True)

        if self.verbose:
            print("Final columns before mapping:", self.data.columns.tolist())

        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(self.data["label"].unique()))}
        self.data["label_idx"] = self.data["label"].map(self.label_to_idx)

        if self.verbose:
            print("Label mapping:", self.label_to_idx)
            print(f"Filtered dataset size: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_path = row["file_name"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to open {image_path}: {e}")

        W, H = image.size

        # --- YOLO detection (optional) ---
        bbox = None
        bboxes = []
        if self.use_yolo:
            cache_key = (image_path, self.yolo_conf, self.yolo_iou, self.yolo_max_det, self.yolo_min_conf, self.yolo_topk)
            if cache_key in self._yolo_cache:
                bboxes = list(self._yolo_cache[cache_key])
                bbox = bboxes[0] if bboxes else None
            else:
                # try on-disk cache if configured
                relkey = os.path.splitext(os.path.relpath(image_path, os.path.abspath(self.image_dir)))[0].replace('\\','/').lower()
                cache_hit = False
                if self.yolo_cache_dir:
                    import hashlib, json
                    h = hashlib.sha256((str(self.yolo_conf)+"|"+str(self.yolo_iou)+"|"+str(self.yolo_max_det)+"|"+str(self.yolo_min_conf)+"|"+str(self.yolo_topk)).encode()).hexdigest()[:12]
                    cache_dir = os.path.join(self.yolo_cache_dir, h)
                    os.makedirs(cache_dir, exist_ok=True)
                    cache_file = os.path.join(cache_dir, relkey + '.json')
                    if os.path.exists(cache_file):
                        try:
                            with open(cache_file, 'r') as f:
                                data = json.load(f)
                            if isinstance(data, list):
                                for item in data:
                                    if isinstance(item, (list, tuple)) and len(item) >= 4:
                                        x1,y1,x2,y2 = map(int, item[:4])
                                        bboxes.append((x1,y1,x2,y2))
                            cache_hit = True
                        except Exception:
                            cache_hit = False

                if not cache_hit:
                    self.yolo_model.eval()
                    results = self.yolo_model(image, conf=self.yolo_conf, iou=self.yolo_iou,
                                              max_det=self.yolo_max_det, device=self.yolo_device, verbose=False)  # list of length 1
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        import numpy as _np
                        xyxy = boxes.xyxy.detach().cpu().numpy()
                        conf = boxes.conf.detach().cpu().numpy() if getattr(boxes, "conf", None) is not None else _np.ones((len(xyxy),), dtype=_np.float32)

                        # sort by confidence desc
                        order = _np.argsort(-conf)
                        kept = 0
                        for idx in order:
                            x1, y1, x2, y2 = map(int, xyxy[idx])
                            w, h = max(0, x2 - x1), max(0, y2 - y1)
                            if w <= 2 or h <= 2:
                                continue
                            area_frac = (w * h) / float(max(1, W * H))
                            min_margin = min(x1, y1, W - 1 - x2, H - 1 - y2)
                            conf_ok = conf[idx] >= self.yolo_min_conf
                            if area_frac > 0.98 or min_margin < 1 or not conf_ok:
                                continue
                            bboxes.append((x1, y1, x2, y2))
                            kept += 1
                            if kept >= self.yolo_topk:
                                break
                    # write to disk cache
                    if self.yolo_cache_dir:
                        try:
                            import json
                            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                            with open(cache_file, 'w') as f:
                                json.dump(bboxes, f)
                        except Exception:
                            pass
                if bboxes:
                    bbox = bboxes[0]
                self._yolo_cache[cache_key] = list(bboxes)

        # --- Transform to model input ---
        if self.transform is None:
            raise ValueError("transform must not be None; ensure it resizes and returns a Tensor.")
        img_tensor = self.transform(image)

        # --- Scale bbox to model input size (sentinel if missing) ---
        bbox_224 = _scale_bbox(bbox, (W, H), (224, 224))
        if bbox_224 is None:
            bbox_224 = (-1, -1, -1, -1)
        bboxes_224 = []
        for bb in bboxes:
            bb_224 = _scale_bbox(bb, (W, H), (224, 224))
            if bb_224 is not None:
                bboxes_224.append(bb_224)

        label = int(row["label_idx"])
        meta = {
            "path": image_path,
            "bbox_224": bbox_224,
            "bboxes_224": bboxes_224,
            "orig_size": (W, H),
            "has_yolo": int(bbox is not None),
        }
        return img_tensor, label, meta

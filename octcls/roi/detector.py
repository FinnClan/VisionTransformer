from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore


@dataclass
class DetectorConfig:
    weights: str = "yolov8n.pt"
    conf: float = 0.15
    iou: float = 0.7
    max_det: int = 10
    device: Optional[str] = None


class UltralyticsDetector:
    def __init__(self, cfg: DetectorConfig):
        assert YOLO is not None, "Ultralytics is not installed."
        self.cfg = cfg
        self.model = YOLO(cfg.weights)

    def detect(self, image) -> List[Tuple[int, int, int, int, float]]:
        """Return list of (x1,y1,x2,y2,conf)."""
        res = self.model(
            image,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            max_det=self.cfg.max_det,
            device=self.cfg.device,
            verbose=False,
        )
        boxes = getattr(res[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        import numpy as np
        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy() if getattr(boxes, "conf", None) is not None else np.ones((len(xyxy),), dtype=np.float32)
        return [(int(a), int(b), int(c), int(d), float(s)) for (a, b, c, d), s in zip(xyxy, conf)]


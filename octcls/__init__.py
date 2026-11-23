"""octcls: Organized training, ROI, CAM, and CLI modules.

Provides importable libraries and focused command‑line tools for:
- Classifier training (with optional YOLO ROI aggregation)
- Grad‑CAM extraction and CAM→YOLO pseudo‑labels
- YOLO dataset preparation and Ultralytics training
"""

__all__ = [
    "data",
    "cam",
    "roi",
    "train",
    "viz",
    "utils",
]

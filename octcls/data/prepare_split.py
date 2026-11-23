from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import random


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_images(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files


def rel_to(base: Path, p: Path) -> Path:
    try:
        return p.relative_to(base)
    except ValueError:
        return Path(p.name)


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_link_or_copy(src: Path, dst: Path, link: bool, clobber: bool):
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        if clobber:
            try:
                if dst.is_symlink() or dst.is_file():
                    dst.unlink()
                elif dst.is_dir():
                    shutil.rmtree(dst)
            except FileNotFoundError:
                pass
        else:
            return
    if link:
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def write_yaml(out_root: Path, names: List[str]) -> Path:
    yaml_path = out_root / "dataset.yaml"
    content = (
        f"names: {names}\n"
        f"path: {out_root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
    )
    yaml_path.write_text(content)
    return yaml_path


@dataclass
class SplitConfig:
    images_root: Path
    out_root: Path
    labels_root: Optional[Path] = None
    val_size: float = 0.2
    seed: int = 50
    names: Tuple[str, ...] = ("roi",)
    copy: bool = False
    clobber: bool = False
    allow_missing_labels: bool = True


def prepare_yolo_split(cfg: SplitConfig) -> Path:
    rng = random.Random(cfg.seed)
    images = find_images(cfg.images_root)
    if not images:
        raise SystemExit(f"No images found under {cfg.images_root}")

    rng.shuffle(images)
    n_total = len(images)
    n_val = int(round(n_total * float(cfg.val_size)))
    val_set = set(images[:n_val])

    img_train = cfg.out_root / "images" / "train"
    img_val = cfg.out_root / "images" / "val"
    lbl_train = cfg.out_root / "labels" / "train"
    lbl_val = cfg.out_root / "labels" / "val"
    for d in (img_train, img_val, lbl_train, lbl_val):
        d.mkdir(parents=True, exist_ok=True)

    missing = 0
    for img in images:
        is_val = img in val_set
        img_dst = (img_val if is_val else img_train) / rel_to(cfg.images_root, img)

        if cfg.labels_root is not None:
            rel = rel_to(cfg.images_root, img)
            lbl_src = (cfg.labels_root / rel).with_suffix(".txt")
        else:
            lbl_src = img.with_suffix(".txt")
        lbl_dst = (lbl_val if is_val else lbl_train) / rel_to(cfg.images_root, img).with_suffix(".txt")

        safe_link_or_copy(img, img_dst, link=(not cfg.copy), clobber=cfg.clobber)

        if lbl_src.exists():
            safe_link_or_copy(lbl_src, lbl_dst, link=(not cfg.copy), clobber=cfg.clobber)
        else:
            if cfg.allow_missing_labels:
                ensure_parent(lbl_dst)
                lbl_dst.write_text("")
            missing += 1

    yaml_path = write_yaml(cfg.out_root, list(cfg.names))
    return yaml_path


"""
Create a YOLO-style dataset split (images/train, images/val, labels/train, labels/val)
from an unsplit images directory. By default, creates symlinks so it does not
duplicate data or interfere with your current setup. Optionally copies files.

Example:

  python -m utilities.prepare_yolo_split \
    --images_root /Users/jimmy/projects/octdl_unsplit \
    --out_root    /Users/jimmy/projects/oct_yolo \
    --val_size 0.2 --seed 50 --names roi \
    --labels_root /path/to/yolo_labels_unsplit --allow_missing_labels

This produces:
  /Users/jimmy/projects/oct_yolo/
    images/train  images/val
    labels/train  labels/val
    dataset.yaml

Then train YOLO with:
  python -m utilities.yolo_train --data /Users/jimmy/projects/oct_yolo/dataset.yaml --model yolov8n.pt --epochs 100
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Optional
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
        # if not under base, fall back to filename
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


def write_yaml(out_root: Path, names: List[str]):
    yaml_path = out_root / "dataset.yaml"
    content = (
        f"names: {names}\n"
        f"path: {out_root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
    )
    yaml_path.write_text(content)
    return yaml_path


def parse_args():
    p = argparse.ArgumentParser(description="Prepare YOLO dataset split from unsplit images directory")
    p.add_argument("--images_root", required=True, help="Unsplit images root directory")
    p.add_argument("--out_root", required=True, help="Output YOLO dataset root")
    p.add_argument("--labels_root", default=None, help="Optional separate root where YOLO label .txt files live")
    p.add_argument("--val_size", type=float, default=0.2, help="Validation fraction (default 0.2)")
    p.add_argument("--seed", type=int, default=50, help="Random seed")
    p.add_argument("--names", nargs="+", default=["roi"], help="Class names list for YAML (default: roi)")
    p.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks")
    p.add_argument("--clobber", action="store_true", help="Overwrite existing files in out_root if present")
    p.add_argument("--allow_missing_labels", action="store_true", help="Create empty label files when missing")
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    images_root = Path(args.images_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    labels_root: Optional[Path] = (
        Path(args.labels_root).expanduser().resolve() if args.labels_root else None
    )

    images = find_images(images_root)
    if not images:
        raise SystemExit(f"No images found under {images_root}")

    # Shuffle and split
    rng.shuffle(images)
    n_total = len(images)
    n_val = int(round(n_total * float(args.val_size)))
    val_set = set(images[:n_val])

    # Prepare output dirs
    img_train = out_root / "images" / "train"
    img_val = out_root / "images" / "val"
    lbl_train = out_root / "labels" / "train"
    lbl_val = out_root / "labels" / "val"
    for d in (img_train, img_val, lbl_train, lbl_val):
        d.mkdir(parents=True, exist_ok=True)

    n_linked = 0
    n_missing_lbl = 0

    for img in images:
        is_val = img in val_set
        img_dst = (img_val if is_val else img_train) / rel_to(images_root, img)

        # Determine label source path
        if labels_root is not None:
            rel = rel_to(images_root, img)
            lbl_src = (labels_root / rel).with_suffix(".txt")
        else:
            lbl_src = img.with_suffix(".txt")

        lbl_dst = (lbl_val if is_val else lbl_train) / rel_to(images_root, img).with_suffix(".txt")

        # Create links/copies
        safe_link_or_copy(img, img_dst, link=(not args.copy), clobber=args.clobber)

        if lbl_src.exists():
            safe_link_or_copy(lbl_src, lbl_dst, link=(not args.copy), clobber=args.clobber)
        else:
            if args.allow_missing_labels:
                ensure_parent(lbl_dst)
                lbl_dst.write_text("")
                n_missing_lbl += 1
            else:
                # If label is missing and not allowed, skip creating label, but keep image placed.
                n_missing_lbl += 1

        n_linked += 1

    yaml_path = write_yaml(out_root, args.names)
    print(f"Prepared YOLO dataset at: {out_root}")
    print(f"  images: train={len(images)-n_val}, val={n_val}")
    print(f"  labels: missing={n_missing_lbl} (empty files created: {args.allow_missing_labels})")
    print(f"  yaml:   {yaml_path}")
    print(f"Links used: {not args.copy} (set --copy to duplicate files)")


if __name__ == "__main__":
    main()


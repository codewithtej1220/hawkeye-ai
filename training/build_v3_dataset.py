"""Build the v3 dataset on top of the v2 dataset (hard links, so unchanged files cost no disk).

Additions over v2:
  1. INFERNO copy of every thermal training/validation image (v2 had 30%), so the INFERNO
     palette - the preferred thermal view - is a first-class input rather than an augmentation.
  2. Mini Human Pose (COCO 2017 keypoint images, one fully-labelled person per image): people
     sitting, lying, crouching, falling, playing sport. Greyscale (30%) and INFERNO (20%) copies.
     No weapon labels: pseudo-labelling with v2 was tried and every one of its ~110 "weapons"
     was a phone, remote, skateboard, kite etc., so these photos serve as hard negatives that
     teach the model those everyday objects are not weapons.
  3. 90-degree rotated copies of 15% of thermal training images (random palette): no real thermal
     data of lying people exists, and for top-down drone views any orientation is physical.

    python build_v3_dataset.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "HawkEye_Training" / "dataset"
DST = ROOT / "HawkEye_Training" / "dataset_v3"
POSE = Path(__file__).resolve().parent / "datasets" / "raw" / "pose" / "data"
SPLITS = {"train": "train", "validation": "val", "test": "test"}


def frac(key: str) -> float:
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def write(image: np.ndarray, lines: list[str], split: str, name: str, stats: Counter, kind: str) -> None:
    cv2.imwrite(str(DST / "images" / split / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (DST / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
    stats[f"{split}:{kind}"] += 1


def inferno(image: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_INFERNO)


def grey3(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def rotate90(image: np.ndarray, lines: list[str]) -> tuple[np.ndarray, list[str]]:
    """Rotate clockwise; YOLO box (cx, cy, w, h) -> (1 - cy, cx, h, w)."""
    out = []
    for line in lines:
        c, cx, cy, w, h = line.split()[:5]
        out.append(f"{c} {1 - float(cy):.6f} {float(cx):.6f} {float(h):.6f} {float(w):.6f}")
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), out


def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    stats: Counter = Counter()

    # 0. v2 dataset, hard-linked
    for f in SRC.rglob("*"):
        if f.is_file() and f.suffix in (".jpg", ".txt"):
            target = DST / f.relative_to(SRC)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(f, target)
            stats["v2_files"] += 1
    print("linked v2 files:", stats["v2_files"], flush=True)

    # 1 + 3. INFERNO copies and rotated copies of thermal images
    for split in ("train", "val"):
        existing = {p.stem for p in (DST / "images" / split).glob("thermal__*_inferno.jpg")}
        for path in sorted((DST / "images" / split).glob("thermal__*.jpg")):
            stem = path.stem
            if stem.endswith(("_inferno", "_grey", "_rot90")):
                continue
            lines = [l for l in (DST / "labels" / split / f"{stem}.txt").read_text().splitlines() if l.strip()]
            image = None
            if f"{stem}_inferno" not in existing:
                image = cv2.imread(str(path))
                write(inferno(image), lines, split, f"{stem}_inferno", stats, "thermal_inferno_copy")
            if split == "train" and lines and frac("rot" + stem) < 0.15:
                image = cv2.imread(str(path)) if image is None else image
                rotated, rot_lines = rotate90(image, lines)
                if frac("rotpal" + stem) < 0.5:
                    rotated = inferno(rotated)
                write(rotated, rot_lines, split, f"{stem}_rot90", stats, "thermal_rot90_copy")
    print("thermal copies:", {k: v for k, v in stats.items() if "thermal" in k}, flush=True)

    # 2. Mini Human Pose (COCO single-person keypoint images)
    for coco_split, split in SPLITS.items():
        data = json.loads((POSE / f"{coco_split}_annotation.json").read_text())
        boxes: dict[int, list] = {}
        for ann in data["annotations"]:
            if not ann.get("iscrowd"):
                boxes.setdefault(ann["image_id"], []).append(ann["bbox"])
        for info in data["images"]:
            image = cv2.imread(str(POSE / coco_split / info["file_name"]))
            if image is None:
                continue
            h, w = image.shape[:2]
            lines = []
            for x, y, bw, bh in boxes.get(info["id"], []):
                if bw > 2 and bh > 2:
                    lines.append(f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
            name = f"rgb__pose__{Path(info['file_name']).stem}"
            write(image, lines, split, name, stats, "pose")
            if split == "train":
                if frac("grey" + name) < 0.3:
                    write(grey3(image), lines, split, f"{name}_grey", stats, "pose_grey_copy")
                elif frac("inf" + name) < 0.2 / 0.7:
                    write(inferno(image), lines, split, f"{name}_inferno", stats, "pose_inferno_copy")
    print("pose:", {k: v for k, v in stats.items() if "pose" in k}, flush=True)

    (DST / "manifest_v3.json").write_text(json.dumps(stats, indent=2))
    for split in ("train", "val", "test"):
        print(split, len(list((DST / "images" / split).glob("*.jpg"))), "images")


if __name__ == "__main__":
    main()

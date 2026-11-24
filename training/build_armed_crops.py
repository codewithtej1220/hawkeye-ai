"""Person-crop dataset for the "armed person" classifier (distant and concealed weapons).

When a weapon is only a few pixels wide a detector cannot box it reliably, but the person as a
whole still carries the evidence: a protrusion, the pose of holding something, or (in thermal) the
cold patch of metal under clothing. This builds crops of people labelled armed / unarmed from the
v3 dataset and degrades them to look distant:

  armed     a person with a labelled weapon in hand or on the body: rifles, handguns/knives,
            dangerous objects (RGB) and TCWD concealed weapons (thermal)
  unarmed   people from scenes with no weapons at all: LLVIP street (thermal + RGB), HIT-UAV drone
            thermal, COCO everyday postures (phones, bags, sports gear = hard negatives)

Each crop is shrunk so the person is 14-90 px tall (or kept as is), enlarged back to the classifier
size, given sensor-like noise and a random palette (colour / greyscale / INFERNO). The AirSim
frames are not used, so they remain an independent test.

    python build_armed_crops.py
"""

from __future__ import annotations

import hashlib
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "HawkEye_Training" / "dataset_v3"
DST = ROOT / "HawkEye_Training" / "armed_crops"
SIZE = 160
ARMED_SOURCES = ("rgb__rifles__", "rgb__weapons_rgb__", "rgb__dangerous_objects__", "thermal__tcwd__")
UNARMED_SOURCES = ("thermal__llvip_ir__", "rgb__llvip_rgb__", "thermal__hituav__", "rgb__pose__")
HEIGHTS = (None, 90, 56, 36, 24, 16)  # None = keep resolution (up to SIZE)
rng = random.Random(0)


def frac(key: str) -> float:
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def read_boxes(label: Path, w: int, h: int):
    out = []
    for line in label.read_text().splitlines() if label.exists() else []:
        if line.strip():
            c, cx, cy, bw, bh = map(float, line.split()[:5])
            out.append((int(c), ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return out


def crop_box(person, w, h):
    """Person plus margin (same geometry the backend uses for its second stage), made square."""
    x1, y1, x2, y2 = person
    pw, ph = x2 - x1, y2 - y1
    side = max(pw + 0.7 * max(pw, ph), ph * 1.4)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return (int(max(0, cx - side / 2)), int(max(0, cy - side / 2)), int(min(w, cx + side / 2)), int(min(h, cy + side / 2)))


def held_weapons(person, weapons):
    x1, y1, x2, y2 = person
    pw, ph = x2 - x1, y2 - y1
    return [wb for wb in weapons
            if x1 - 0.5 * pw <= (wb[0] + wb[2]) / 2 <= x2 + 0.5 * pw and y1 - 0.3 * ph <= (wb[1] + wb[3]) / 2 <= y2 + 0.3 * ph]


def degrade(crop: np.ndarray, person_h: float, target_h, thermal: bool, key: str) -> np.ndarray:
    ch, cw = crop.shape[:2]
    if target_h is not None and person_h > target_h:
        s = target_h / person_h
        crop = cv2.resize(crop, (max(1, round(cw * s)), max(1, round(ch * s))), interpolation=cv2.INTER_AREA)
    out = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC if crop.shape[0] < SIZE else cv2.INTER_AREA)
    noise = np.random.default_rng(int(frac("n" + key) * 1e9)).normal(0, 2.5, out.shape)
    out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    grey = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    p = frac("pal" + key)
    if thermal or p < 0.25:  # thermal always gets a thermal palette; some RGB goes greyscale/INFERNO
        if p < 0.55:
            return cv2.applyColorMap(grey, cv2.COLORMAP_INFERNO)
        return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    return out


def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    stats: Counter = Counter()
    for split_src, split_dst in (("train", "train"), ("val", "val")):
        for cls in ("armed", "unarmed"):
            (DST / split_dst / cls).mkdir(parents=True)
        files = sorted((SRC / "images" / split_src).glob("*.jpg"))
        armed, unarmed = [], []
        for f in files:
            name = f.name
            if "_inferno" in name or "_grey" in name or "_dup" in name:
                continue  # palettes are re-randomised here; avoid near-duplicates
            src_armed = name.startswith(ARMED_SOURCES)
            src_unarmed = name.startswith(UNARMED_SOURCES)
            if not (src_armed or src_unarmed):
                continue
            armed_label = (SRC / "labels" / split_src / (f.stem + ".txt"))
            text = armed_label.read_text() if armed_label.exists() else ""
            has_weapon = any(l.startswith("1 ") for l in text.splitlines())
            has_person = any(l.startswith("0 ") for l in text.splitlines())
            if not has_person or (src_unarmed and has_weapon) or (src_armed and not has_weapon):
                continue
            (armed if src_armed else unarmed).append(f)

        def emit(paths, cls, per_image_cap, variants):
            for f in paths:
                img = cv2.imread(str(f))
                if img is None:
                    continue
                h, w = img.shape[:2]
                gt = read_boxes(SRC / "labels" / f.parent.name / (f.stem + ".txt"), w, h)
                people = [b for c, b in gt if c == 0 and b[3] - b[1] >= 12]
                weapons = [b for c, b in gt if c == 1]
                if cls == "armed":
                    people = [p for p in people if held_weapons(p, weapons)]
                rng.shuffle(people)
                thermal = f.name.startswith("thermal__")
                for k, person in enumerate(people[:per_image_cap]):
                    x0, y0, x1, y1 = crop_box(person, w, h)
                    crop = img[y0:y1, x0:x1]
                    if crop.size == 0:
                        continue
                    heights = rng.sample(HEIGHTS, variants)
                    for target in heights:
                        key = f"{f.stem}_{k}_{target}"
                        out = degrade(crop, person[3] - person[1], target, thermal, key)
                        cv2.imwrite(str(DST / split_dst / cls / f"{key}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        stats[f"{split_dst}:{cls}"] += 1
                        stats[f"{split_dst}:{cls}:{f.name.split('__')[1]}"] += 1

        # armed people are rarer: more distance variants per person; cap crowded unarmed scenes
        emit(armed, "armed", per_image_cap=3, variants=4)
        emit(unarmed, "unarmed", per_image_cap=2, variants=2)
        print(split_dst, {k: v for k, v in stats.items() if k.startswith(split_dst)}, flush=True)


if __name__ == "__main__":
    main()

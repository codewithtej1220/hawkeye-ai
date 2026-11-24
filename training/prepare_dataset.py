"""Merge the downloaded real datasets into one 2-class YOLO dataset.

What this does for every source in sources.py:
  * finds its images/<split> folders and the class names in its dataset YAML
  * remaps its classes onto HawkEye's  0=person, 1=weapon  (anything else is dropped)
  * if the source has no person labels, pseudo-labels people with COCO-pretrained
    YOLO11m so those people are not learned as "background"
  * creates val/test splits for sources that ship without them
  * thermal palette augmentation: adds INFERNO-coloured copies of training images,
    because the AirSim feed is INFERNO-coloured while real cameras are mostly white-hot

Output: datasets/hawkeye_real/{images,labels}/{train,val,test} + hawkeye_real.yaml
File names are prefixed "<modality>__<source>__" so evaluate.py can report per domain.
"""

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from sources import CLASS_KEYWORDS, CLASS_NAMES, MERGED_DIR, RAW_DIR, SOURCES

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")
SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val",
                 "validation": "val", "test": "test", "testing": "test"}
PERSON, WEAPON = CLASS_NAMES.index("person"), CLASS_NAMES.index("weapon")


def stable_fraction(key: str) -> float:
    """Deterministic pseudo-random number in [0, 1) derived from a string."""
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def map_class_name(name: str, overrides: dict) -> int | None:
    if name in overrides:
        target = overrides[name]
        return None if target is None else CLASS_NAMES.index(target)
    lowered = name.lower().replace("_", " ").replace("-", " ")
    words = set(lowered.split())
    for target, keywords in CLASS_KEYWORDS.items():
        if any(k in words or (len(k) > 3 and k in lowered) for k in keywords):
            return CLASS_NAMES.index(target)
    return None


def find_class_names(root: Path) -> list[str]:
    for path in sorted(root.rglob("*.yaml"), key=lambda p: len(p.parts)):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore"))
        except yaml.YAMLError:
            continue
        if isinstance(data, dict) and data.get("names"):
            names = data["names"]
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names)]
            print(f"  classes from {path.relative_to(root)}: {names}")
            return [str(n) for n in names]
    raise SystemExit(f"[error] no dataset YAML with 'names' found under {root}")


def find_split_dirs(root: Path) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = defaultdict(list)
    for images_dir in root.rglob("images"):
        if not images_dir.is_dir():
            continue
        # Layout A: images/<split>/   Layout B (Roboflow, LLVIP): <split>/images/
        parent_split = SPLIT_ALIASES.get(images_dir.parent.name.lower())
        if parent_split:
            found[parent_split].append(images_dir)
            continue
        for child in images_dir.iterdir():
            split = SPLIT_ALIASES.get(child.name.lower())
            if child.is_dir() and split:
                found[split].append(child)
    return found


def label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    label = Path(*parts).with_suffix(".txt")
    # Dangerous Objects' test split pairs image_0001.jpg with label_0001.txt
    if not label.exists() and label.stem.startswith("image_"):
        label = label.with_name("label_" + label.name[len("image_"):])
    return label


def read_labels(path: Path, class_lut: dict[int, int | None]) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        target = class_lut.get(int(float(values[0])))
        if target is None:
            continue
        coords = [float(v) for v in values[1:]]
        if len(coords) == 4:
            cx, cy, w, h = coords
        else:  # segmentation polygon -> bounding box
            xs, ys = coords[0::2], coords[1::2]
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        if w > 0.002 and h > 0.002:
            boxes.append((target, cx, cy, w, h))
    return boxes


def pseudo_label_people(image_paths: list[Path], device: str, conf: float) -> dict[Path, list]:
    from ultralytics import YOLO

    print(f"  pseudo-labelling people in {len(image_paths)} images with COCO yolo11m (conf>={conf})")
    model = YOLO("yolo11m.pt")
    results: dict[Path, list] = {}
    # A list source is inferred as a single batch, so feed it in small chunks.
    for start in range(0, len(image_paths), 8):
        chunk = image_paths[start:start + 8]
        for path, result in zip(chunk, model.predict([str(p) for p in chunk], classes=[0], conf=conf,
                                                     device=device, verbose=False)):
            results[path] = [(PERSON, *box) for box in result.boxes.xywhn.tolist()]
    return results


def to_inferno(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def write_sample(image: np.ndarray, boxes: list, split: str, name: str) -> None:
    cv2.imwrite(str(MERGED_DIR / "images" / split / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in boxes]
    (MERGED_DIR / "labels" / split / f"{name}.txt").write_text("\n".join(lines))


def process_source(source: dict, args, stats: dict) -> None:
    slug, modality = source["slug"], source["modality"]
    download_root = RAW_DIR / source.get("raw", slug)
    root = download_root / source.get("subdir", "")
    print(f"\n[{slug}] ({modality})")
    if not (download_root / ".complete").exists():
        raise SystemExit(f"[error] {slug} not downloaded; run download_datasets.py first")

    names = find_class_names(download_root)
    overrides = source.get("class_map", {})
    class_lut = {i: map_class_name(n, overrides) for i, n in enumerate(names)}
    print("  class mapping:", {names[i]: (CLASS_NAMES[t] if t is not None else "-") for i, t in class_lut.items()})
    has_person_labels = PERSON in class_lut.values()

    # Collect (image, split), deriving val/test from train when the source lacks them.
    split_dirs = find_split_dirs(root)
    items: list[tuple[Path, str]] = []
    for split, dirs in split_dirs.items():
        for directory in dirs:
            items += [(p, split) for p in sorted(directory.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    if not items:
        raise SystemExit(f"[error] no images/<split> folders found under {root}")
    def derived_split(path: Path, split: str) -> str:
        if split != "train":
            return split
        f = stable_fraction(path.name)
        if "val" not in split_dirs and f < 0.10:
            return "val"
        if "test" not in split_dirs and 0.10 <= f < 0.20:
            return "test"
        return split

    items = [(p, derived_split(p, s)) for p, s in items]
    cap = source.get("max_train") or args.max_per_source
    if cap:
        train = [it for it in items if it[1] == "train"]
        if len(train) > cap:
            train.sort(key=lambda it: stable_fraction(it[0].name))
            drop = {p for p, _ in train[cap:]}
            items = [it for it in items if it[0] not in drop]

    people = {}
    if not has_person_labels and not args.no_pseudo_labels:
        conf = args.pseudo_conf_thermal if modality == "thermal" else args.pseudo_conf
        people = pseudo_label_people([p for p, _ in items], args.device, conf)

    palette_p = args.thermal_palette_p if modality == "thermal" else args.rgb_palette_p
    for path, split in items:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        scale = args.max_side / max(image.shape[:2])
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        boxes = read_labels(label_path_for(path), class_lut) + people.get(path, [])
        name = f"{modality}__{slug}__{path.stem}"
        write_sample(image, boxes, split, name)
        stats[slug][split] += 1
        for c, *_ in boxes:
            stats[slug][f"{split}_{CLASS_NAMES[c]}"] += 1

        if split == "train" and stable_fraction("palette" + name) < palette_p:
            write_sample(to_inferno(image), boxes, split, name + "_inferno")
            stats[slug]["train_inferno_copies"] += 1
        # Greyscale (monochrome CCTV / night cameras) copies of colour images
        elif split == "train" and modality == "rgb" and stable_fraction("grey" + name) < args.rgb_grey_p:
            grey = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            write_sample(grey, boxes, split, name + "_grey")
            stats[slug]["train_grey_copies"] += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="0", help="device for pseudo-labelling (0 = first GPU, cpu)")
    parser.add_argument("--pseudo-conf", type=float, default=0.45, help="person pseudo-label confidence (RGB)")
    # COCO weights are less confident on thermal; TCWD frames always contain a person, so a
    # lower threshold recovers ~2x more of them without adding noticeable false positives.
    parser.add_argument("--pseudo-conf-thermal", type=float, default=0.20)
    parser.add_argument("--no-pseudo-labels", action="store_true")
    parser.add_argument("--rgb-grey-p", type=float, default=0.25,
                        help="fraction of RGB train images that also get a greyscale copy")
    parser.add_argument("--thermal-palette-p", type=float, default=0.3,
                        help="fraction of thermal train images that also get an INFERNO copy")
    parser.add_argument("--rgb-palette-p", type=float, default=0.1,
                        help="fraction of RGB train images that also get a grey->INFERNO copy")
    parser.add_argument("--max-side", type=int, default=1280, help="downscale larger images to save disk")
    parser.add_argument("--max-per-source", type=int, default=0, help="cap train images per source (0 = no cap)")
    parser.add_argument("--delete-raw", action="store_true", help="delete raw downloads afterwards to free disk")
    args = parser.parse_args()

    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)
    for kind in ("images", "labels"):
        for split in SPLITS:
            (MERGED_DIR / kind / split).mkdir(parents=True)

    stats: dict[str, Counter] = defaultdict(Counter)
    for source in SOURCES:
        process_source(source, args, stats)

    data_yaml = MERGED_DIR / "hawkeye_real.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(MERGED_DIR),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(CLASS_NAMES)),
    }, sort_keys=False))
    (MERGED_DIR / "manifest.json").write_text(json.dumps(stats, indent=2))

    print("\n=== merged dataset ===")
    for slug, counter in stats.items():
        print(f"  {slug:12s} " + ", ".join(f"{k}={v}" for k, v in sorted(counter.items())))
    print(f"\nWrote {data_yaml}")

    if args.delete_raw:
        shutil.rmtree(RAW_DIR)
        print(f"Deleted {RAW_DIR}")


if __name__ == "__main__":
    main()

"""Before/after check for detector-level changes, on real test images.

Runs the deployed detector (auto method, deployed thresholds) twice on the same images: once with
the previous behaviour ("old") and once as it is now ("new"), and counts correct detections,
misses and false alarms per class (IoU >= 0.5 against the labels).

    python regression_check.py                # 100 images per source
    python regression_check.py --per-source 200

Old behaviour = contrast measured without max-pooling, uncapped auto-gain, no
frame-filling-person filter."""
import argparse
import random
import sys
from pathlib import Path

import albumentations  # noqa: F401  (must load before torch on this machine)
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project" / "backend"))
from app import detector as detector_module  # noqa: E402
from app import modalities  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.detector import ThreatDetector  # noqa: E402

TEST = ROOT / "HawkEye_Training" / "dataset_v3" / "images" / "test"
SOURCES = ["thermal__llvip_ir", "thermal__hituav", "thermal__tcwd", "rgb__llvip_rgb", "rgb__pose", "rgb__rifles",
           "rgb__dangerous_objects"]


def crossover(grey, rng):
    # hot day: people and background at similar temperatures (same model as evaluate_modalities.py)
    g = grey.astype(np.float32)
    g = g.mean() + (g - g.mean()) * 0.18 + rng.normal(0, 3, g.shape)
    return np.clip(g, 0, 255).astype(np.uint8)


def labels_of(image_path, w, h):
    path = Path(str(image_path).replace("images", "labels")).with_suffix(".txt")
    boxes = []
    if path.exists():
        for line in path.read_text().split("\n"):
            if line.strip():
                c, cx, cy, bw, bh = line.split()[:5]
                cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                boxes.append((int(c), (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)))
    return boxes


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union > 0 else 0.0


def score(detections, truth, w, h, counts):
    for cls, kind in ((0, "person"), (1, "weapon")):
        found = [(d.bbox.x / 100 * w, d.bbox.y / 100 * h, (d.bbox.x + d.bbox.width) / 100 * w,
                  (d.bbox.y + d.bbox.height) / 100 * h) for d in detections if d.kind == kind and d.evidence != "classifier"]
        wanted = [b for c, b in truth if c == cls]
        used = set()
        for box in found:
            best = max(((iou(box, t), j) for j, t in enumerate(wanted) if j not in used), default=(0, -1))
            if best[0] >= 0.5:
                used.add(best[1])
                counts[kind][0] += 1
            else:
                counts[kind][1] += 1
        counts[kind][2] += len(wanted) - len(used)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-source", type=int, default=100)
    parser.add_argument("--sources", nargs="+", default=SOURCES)
    parser.add_argument("--old-models", nargs=2, metavar=("MODEL", "ENSEMBLE"), default=None,
                        help="compare models instead of code: 'old' = these two weight files (paths relative "
                             "to project/backend, e.g. ../models/hawkeye_real_yolo11m.pt ../models/hawkeye_v3.pt), "
                             "'new' = the models in config.py; the code is the current one for both")
    args = parser.parse_args()
    det = ThreatDetector(get_settings())
    det.load()
    detectors = {"old": det, "new": det}
    if args.old_models:
        detectors["old"] = ThreatDetector(get_settings().model_copy(update={
            "model_path": args.old_models[0], "ensemble_model_path": args.old_models[1]}))
        detectors["old"].load()
        print("old:", detectors["old"].model_name, "| new:", det.model_name)
    new = {"fill": detector_module._drop_frame_filling_people, "gain": modalities.AGC_MAX_GAIN}

    def configure(old):
        if args.old_models:
            return
        detector_module._drop_frame_filling_people = (lambda d, shape, labels: d) if old else new["fill"]
        modalities.AGC_MAX_GAIN = 1e9 if old else new["gain"]
        modalities.POOL_HOT_TARGETS = not old

    rnd = random.Random(3)
    suites = []
    for source in args.sources:
        files = sorted(TEST.glob(source + "__*"))
        suites.append((source, rnd.sample(files, min(args.per_source, len(files))), False))
        if source == "thermal__llvip_ir":
            suites.append((source + " (hot day, low contrast)", suites[-1][1], True))

    print(f"{'suite':44s} {'':5s} {'person ok/false/missed':>24s}   {'weapon ok/false/missed':>24s}")
    totals = {"old": {"person": [0, 0, 0], "weapon": [0, 0, 0]}, "new": {"person": [0, 0, 0], "weapon": [0, 0, 0]}}
    for name, files, low_contrast in suites:
        for label in ("old", "new"):
            configure(label == "old")
            counts = {"person": [0, 0, 0], "weapon": [0, 0, 0]}
            rng = np.random.default_rng(5)
            for f in files:
                image = cv2.imread(str(f))
                if low_contrast:
                    image = cv2.cvtColor(crossover(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), rng), cv2.COLOR_GRAY2BGR)
                h, w = image.shape[:2]
                payload, _ = detectors[label].analyze(image, None, 0, "check")
                score(payload.detections, labels_of(f, w, h), w, h, counts)
            for kind in counts:
                for i in range(3):
                    totals[label][kind][i] += counts[kind][i]
            print(f"{name:44s} {label:5s} {'/'.join(map(str, counts['person'])):>24s}   {'/'.join(map(str, counts['weapon'])):>24s}", flush=True)
    for label in ("old", "new"):
        t = totals[label]
        print(f"{'TOTAL':44s} {label:5s} {'/'.join(map(str, t['person'])):>24s}   {'/'.join(map(str, t['weapon'])):>24s}")


if __name__ == "__main__":
    main()

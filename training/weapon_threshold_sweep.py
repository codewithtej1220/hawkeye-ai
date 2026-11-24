"""Choose the confidence threshold for the thermal-rifle specialists (config.py:
weapon_specialist_confidence).

The specialists only speak when the main ensemble found no weapon on a person in a thermal view.
For every image this script records (a) whether the main pipeline alone reports a weapon and
(b) the strongest specialist score, then prints for each threshold how many weapon-free real
images would be flagged (false alarms) and how many weapon images would be found.

    python weapon_threshold_sweep.py [--per-set 150]

Weapon-free: real thermal street and drone footage. With weapons: real thermal concealed weapons
(TCWD) and held-out simulator frames from AirSimDemos/range_test_results (never used in training)."""
import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import albumentations  # noqa: F401
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project" / "backend"))
from app.config import get_settings  # noqa: E402
from app.detector import ThreatDetector  # noqa: E402

TEST = ROOT / "HawkEye_Training" / "dataset_v3" / "images" / "test"
THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)


def has_weapon(image_path):
    label = Path(str(image_path).replace("images", "labels")).with_suffix(".txt")
    return label.exists() and any(line.startswith("1 ") for line in label.read_text().splitlines())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-set", type=int, default=150)
    parser.add_argument("--sim-frames", default=str(ROOT / "AirSimDemos" / "range_test_results"))
    args = parser.parse_args()

    det = ThreatDetector(get_settings().model_copy(update={"weapon_specialist_confidence": 0.05}))
    det.load()
    specialists = det._specialists
    rnd = random.Random(1)

    def weapons(image, use_specialists):
        det._specialists = specialists if use_specialists else []
        payload, _ = det.analyze(image, None, 0, "sweep")
        return [d.confidence for d in payload.detections if d.kind == "weapon" and d.evidence != "classifier"]

    def record(image):
        base = weapons(image, False)
        if base:
            return True, 0.0
        return False, max(weapons(image, True), default=0.0)

    sets = {}
    for name in ("thermal__llvip_ir", "thermal__hituav"):
        files = [f for f in sorted(TEST.glob(name + "__*")) if not has_weapon(f)]
        sets[("weapon-free", name)] = rnd.sample(files, min(args.per_set, len(files)))
    files = [f for f in sorted(TEST.glob("thermal__tcwd__*")) if has_weapon(f)]
    sets[("weapons", "thermal__tcwd (real)")] = rnd.sample(files, min(args.per_set, len(files)))
    sim = []
    for results in glob.glob(os.path.join(args.sim_frames, "*", "results.json")):
        folder = os.path.dirname(results)
        for row in json.load(open(results))["rows"]:
            for fr in row["frames"]:
                if fr.get("weapon_px", 0) >= 12 and fr.get("person_px", 0) >= 20:
                    sim.append(os.path.join(folder, fr["tag"] + "_thermal.png"))
    sets[("weapons", "simulator rifle (held out)")] = sim

    results = {}
    for key, files in sets.items():
        results[key] = [record(cv2.imread(str(f))) for f in files]
        print(f"scored {key[0]:11s} {key[1]}: {len(files)} images", flush=True)

    print("\nimages reported with a weapon (main pipeline alone, then with the specialists at each threshold)")
    print(f"{'set':42s} {'main only':>10s} " + " ".join(f"{t:>8.2f}" for t in THRESHOLDS))
    for key, values in results.items():
        n = len(values)
        base = sum(1 for b, _ in values if b)
        cells = [sum(1 for b, score in values if b or score >= t) for t in THRESHOLDS]
        print(f"{key[0] + ': ' + key[1]:42s} {base:5d}/{n:<4d} " + " ".join(f"{c:4d}/{n:<3d}" for c in cells))


if __name__ == "__main__":
    main()

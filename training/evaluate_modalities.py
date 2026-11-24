"""Evaluate a HawkEye model with each sensing method under different environments.

Methods: rgb, thermal_grey, thermal_inferno, rgbt (decision-level fusion, as in the backend).

  * Paired RGB+thermal: the aligned LLVIP test pairs, under simulated environments
      normal | dark (night, no street light) | fog/smoke | glare | thermal crossover (hot day,
      no temperature contrast)
  * Thermal only (grey vs INFERNO palette): TCWD concealed weapons, HIT-UAV drone views
  * Visible only (colour vs monochrome camera): weapon datasets
  * AirSim pairs (RGB + thermal with masks-derived ground truth), if captured frames exist

Reports AP50 per class (threshold-free model quality). Used to (1) choose which method fits
which environment and (2) gate a new model: it must not be worse than the current one.

    python evaluate_modalities.py --model ..\\project\\models\\hawkeye_real_yolo11m.pt --out v2.json
    python evaluate_modalities.py --model new.pt --compare v2.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import albumentations  # noqa: F401  (Windows: load before torch/CUDA to avoid a DLL deadlock)
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project" / "backend"))
from app.modalities import (assess_environment, fuse_detections, grey_to_inferno, select_method,  # noqa: E402
                            thermal_agc, to_grey3)

# dataset_v3 test split = the v2 test split plus the COCO posture test photos
DATASET = ROOT / "HawkEye_Training" / ("dataset_v3" if (ROOT / "HawkEye_Training" / "dataset_v3").exists() else "dataset")
CLASSES = ("person", "weapon")
rng = np.random.default_rng(0)


# ----------------------------------------------------------------------------------------
# Environments (applied to aligned RGB / thermal-grey pairs)
# ----------------------------------------------------------------------------------------

def env_normal(rgb, grey):
    return rgb, grey


def env_dark(rgb, grey):
    # No street lighting: ~1/8 exposure plus sensor noise. Thermal is unaffected by light.
    dark = rgb.astype(np.float32) * 0.12 + rng.normal(0, 4, rgb.shape)
    return np.clip(dark, 0, 255).astype(np.uint8), grey


def env_fog(rgb, grey):
    # Dense fog/smoke scatters visible light strongly; long-wave IR far less (mild loss + blur).
    foggy = rgb.astype(np.float32) * 0.3 + 230 * 0.7
    foggy = cv2.GaussianBlur(np.clip(foggy, 0, 255).astype(np.uint8), (0, 0), 1.5)
    g = grey.astype(np.float32)
    g = g.mean() + (g - g.mean()) * 0.8
    return foggy, cv2.GaussianBlur(np.clip(g, 0, 255).astype(np.uint8), (0, 0), 1.0)


def env_glare(rgb, grey):
    # Low sun / headlights straight into the lens: overexposure washes out the visible image.
    return np.clip(rgb.astype(np.float32) * 2.6 + 70, 0, 255).astype(np.uint8), grey


def env_crossover(rgb, grey):
    # Hot day: people and background at similar temperatures -> almost no thermal contrast.
    g = grey.astype(np.float32)
    g = g.mean() + (g - g.mean()) * 0.18 + rng.normal(0, 3, g.shape)
    return rgb, np.clip(g, 0, 255).astype(np.uint8)


ENVIRONMENTS = {"normal": env_normal, "dark": env_dark, "fog/smoke": env_fog, "glare": env_glare,
                "thermal crossover": env_crossover}


# ----------------------------------------------------------------------------------------
# Detection + AP
# ----------------------------------------------------------------------------------------

def load_labels(image_path: Path, width: int, height: int) -> list[tuple[int, list[float]]]:
    label = DATASET / "labels" / image_path.parent.name / (image_path.stem + ".txt")
    boxes = []
    if label.exists():
        for line in label.read_text().splitlines():
            if not line.strip():
                continue
            c, cx, cy, w, h = map(float, line.split()[:5])
            boxes.append((int(c), [(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height]))
    return boxes


def predict(model, image) -> list[tuple[str, float, tuple]]:
    r = model.predict(image, conf=0.05, verbose=False)[0]
    return [(CLASSES[int(k)], float(c), tuple(b)) for b, k, c in
            zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def average_precision(samples, cls: str) -> float | None:
    """samples: list of (detections, ground_truth). COCO-style 101-point AP at IoU 0.5."""
    cls_id = CLASSES.index(cls)
    total_gt = sum(sum(1 for c, _ in gt if c == cls_id) for _, gt in samples)
    if total_gt == 0:
        return None
    scored = []
    for i, (dets, gt) in enumerate(samples):
        for label, conf, box in dets:
            if label == cls:
                scored.append((conf, i, box))
    scored.sort(key=lambda s: -s[0])
    matched = [set() for _ in samples]
    tp = np.zeros(len(scored))
    for k, (conf, i, box) in enumerate(scored):
        best, best_iou = None, 0.5
        for j, (c, g) in enumerate(samples[i][1]):
            if c != cls_id or j in matched[i]:
                continue
            o = iou(box, g)
            if o >= best_iou:
                best, best_iou = j, o
        if best is not None:
            matched[i].add(best)
            tp[k] = 1
    if not len(scored):
        return 0.0
    ctp = np.cumsum(tp)
    recall = ctp / total_gt
    precision = ctp / np.arange(1, len(scored) + 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        above = precision[recall >= t]
        ap += (above.max() if len(above) else 0.0) / 101
    return float(ap)


def run_methods(model, rgb, grey) -> dict[str, list]:
    """Detections for every method on one aligned pair (either image may be None)."""
    out = {}
    env = assess_environment(rgb, grey)
    if rgb is not None:
        out["rgb"] = predict(model, rgb)
    if grey is not None:
        # same preprocessing as the backend: auto-gain only at low thermal contrast
        grey3 = thermal_agc(grey) if env.thermal_agc else to_grey3(grey)
        out["thermal_grey"] = predict(model, grey3)
        out["thermal_inferno"] = predict(model, grey_to_inferno(grey3))
    out["_env"] = env
    if rgb is not None and grey is not None:
        fused = fuse_detections(out["rgb"], out["thermal_inferno"], env.rgb_reliability, env.thermal_reliability)
        out["rgbt"] = [(label, conf, box) for label, conf, box, _ in fused]
    return out


# ----------------------------------------------------------------------------------------
# Suites
# ----------------------------------------------------------------------------------------

def paired_suite(model, limit: int) -> dict:
    thermal = sorted((DATASET / "images" / "test").glob("thermal__llvip_ir__*.jpg"))[:limit]
    results = {}
    for env_name, env_fn in ENVIRONMENTS.items():
        per_method: dict[str, list] = {}
        chosen_hits = []
        for t_path in thermal:
            rgb_path = t_path.with_name(t_path.name.replace("thermal__llvip_ir__", "rgb__llvip_rgb__"))
            rgb = cv2.imread(str(rgb_path))
            grey = cv2.cvtColor(cv2.imread(str(t_path)), cv2.COLOR_BGR2GRAY)
            rgb_e, grey_e = env_fn(rgb, grey)
            gt = load_labels(t_path, grey.shape[1], grey.shape[0])
            dets = run_methods(model, rgb_e, grey_e)
            for method, d in dets.items():
                if not method.startswith("_"):
                    per_method.setdefault(method, []).append((d, gt))
            chosen, _ = select_method(True, True, dets["_env"])
            chosen_hits.append((dets[chosen], gt, chosen))
        res = {m: average_precision(s, "person") for m, s in per_method.items()}
        res["auto"] = average_precision([(d, g) for d, g, _ in chosen_hits], "person")
        picks = {}
        for *_, c in chosen_hits:
            picks[c] = picks.get(c, 0) + 1
        res["auto_picks"] = picks
        results[env_name] = res
        print(f"[paired] {env_name:18s} " + "  ".join(f"{m}={v:.3f}" for m, v in res.items() if isinstance(v, float))
              + f"  picks={picks}", flush=True)
    return results


def single_suite(model, pattern: str, kind: str, limit: int, step: int = 1) -> dict:
    files = sorted((DATASET / "images" / "test").glob(pattern))[::step][:limit]
    per_method: dict[str, list] = {}
    for f in files:
        img = cv2.imread(str(f))
        gt = load_labels(f, img.shape[1], img.shape[0])
        if kind == "thermal":
            dets = run_methods(model, None, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        else:
            dets = {"rgb": predict(model, img), "greyscale_camera": predict(model, to_grey3(img))}
        for method, d in dets.items():
            if not method.startswith("_"):
                per_method.setdefault(method, []).append((d, gt))
    res = {f"{m}/{c}": average_precision(s, c) for m, s in per_method.items() for c in CLASSES}
    res = {k: v for k, v in res.items() if v is not None}
    print(f"[single] {pattern:34s} " + "  ".join(f"{k}={v:.3f}" for k, v in res.items()), flush=True)
    return res


def false_alarm_suite(model, pattern: str, limit: int, threshold: float = 0.35) -> dict:
    """Everyday photos with people and objects (phones, remotes, sports gear) but no weapons:
    share of images in which the model reports a weapon at the backend's weapon threshold."""
    files = sorted((DATASET / "images" / "test").glob(pattern))[:limit]
    flagged = 0
    for f in files:
        dets = predict(model, cv2.imread(str(f)))
        flagged += any(label == "weapon" and conf >= threshold for label, conf, _ in dets)
    res = {"images": len(files), "false_weapon_images": flagged, "false_weapon_rate": flagged / max(len(files), 1)}
    print(f"[false alarms] {pattern}: {flagged}/{len(files)} images with a weapon >= {threshold}", flush=True)
    return res


def sim_suite(model) -> dict:
    """AirSim RGB + thermal pairs; ground truth from the thermal frames' body / weapon masks."""
    base = ROOT / "AirSimDemos" / "HawkEye_Manual_Dataset"
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_INFERNO)[:, 0].astype(np.int32)
    per_method: dict[str, list] = {}
    for t_path in sorted(glob.glob(str(base / "thermal" / "*" / "*.png"))):
        rgb_path = Path(t_path.replace("\\thermal\\", "\\rgb\\"))
        if not rgb_path.exists():
            continue
        thermal = cv2.resize(cv2.imread(t_path), (640, 480))
        rgb = cv2.resize(cv2.imread(str(rgb_path)), (640, 480))
        blue = (thermal[..., 0] > 200) & (thermal[..., 1] < 40) & (thermal[..., 2] < 40)
        d = ((thermal.reshape(-1, 1, 3).astype(np.int32) - lut[None]) ** 2).sum(-1)
        g = d.argmin(1).astype(np.uint8).reshape(thermal.shape[:2])
        body = ((g >= 175) & ~blue).astype(np.uint8)
        gt = []
        for mask, cls in ((body, 0), (blue.astype(np.uint8), 1)):
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
            if n > 1:
                k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                x, y, w, h = stats[k, :4]
                if stats[k, cv2.CC_STAT_AREA] >= 6:
                    gt.append((cls, [x, y, x + w, y + h]))
        # render the thermal frame the way the current simulator does (white-hot, cold weapon)
        grey = g.copy()
        grey[blue] = 45
        grey[body > 0] = np.maximum(grey[body > 0], 200)
        dets = run_methods(model, rgb, grey)
        for method, dd in dets.items():
            if not method.startswith("_"):
                per_method.setdefault(method, []).append((dd, gt))
    res = {f"{m}/{c}": average_precision(s, c) for m, s in per_method.items() for c in CLASSES}
    res = {k: v for k, v in res.items() if v is not None}
    print(f"[sim] {len(next(iter(per_method.values()), []))} pairs  " + "  ".join(f"{k}={v:.3f}" for k, v in res.items()), flush=True)
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "project" / "models" / "hawkeye_real_yolo11m.pt"))
    parser.add_argument("--limit", type=int, default=840, help="max images per suite")
    parser.add_argument("--out", help="write results JSON")
    parser.add_argument("--compare", help="baseline results JSON: report differences")
    parser.add_argument("--quick", action="store_true", help="paired suite only")
    parser.add_argument("--suites", default="all", help="comma list: paired,thermal,rgb,pose,sim (default all)")
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)

    suites = {"paired"} if args.quick else (
        {"paired", "thermal", "rgb", "pose", "sim"} if args.suites == "all" else set(args.suites.split(",")))
    results = {"model": Path(args.model).name}
    if "paired" in suites:
        results["paired_llvip"] = paired_suite(model, args.limit)
    if "thermal" in suites:
        results["thermal_tcwd"] = single_suite(model, "thermal__tcwd__*.jpg", "thermal", args.limit)
        results["thermal_hituav"] = single_suite(model, "thermal__hituav__*.jpg", "thermal", args.limit)
    if "rgb" in suites:
        results["rgb_dangerous"] = single_suite(model, "rgb__dangerous_objects__*.jpg", "rgb", args.limit)
        results["rgb_rifles"] = single_suite(model, "rgb__rifles__*.jpg", "rgb", args.limit)
        results["rgb_weapons"] = single_suite(model, "rgb__weapons_rgb__*.jpg", "rgb", args.limit)
    if "pose" in suites:
        # People in any posture (COCO), in colour, greyscale and INFERNO renderings
        results["pose_people"] = single_suite(model, "rgb__pose__*.jpg", "rgb", args.limit)
        results["pose_false_alarms"] = false_alarm_suite(model, "rgb__pose__*.jpg", args.limit)
    if "sim" in suites:
        results["sim"] = sim_suite(model)

    if args.out:
        out = Path(args.out)
        if args.suites != "all" and out.exists():  # partial run: merge into existing results
            merged = json.loads(out.read_text())
            merged.update(results)
            results = merged
        out.write_text(json.dumps(results, indent=2, default=str))
    if args.compare:
        base = json.loads(Path(args.compare).read_text())
        print("\n=== change vs", base.get("model"), "===")
        worst = 0.0
        for suite, vals in results.items():
            if not isinstance(vals, dict):
                continue
            for key, value in vals.items():
                pairs = value.items() if isinstance(value, dict) else [(None, value)]
                for sub, v in pairs:
                    b = base.get(suite, {}).get(key)
                    b = b.get(sub) if isinstance(b, dict) and sub is not None else b
                    if isinstance(v, float) and isinstance(b, float):
                        delta = v - b
                        worst = min(worst, delta)
                        flag = "  <-- WORSE" if delta < -0.02 else ""
                        print(f"  {suite:15s} {key:20s} {sub or '':16s} {b:.3f} -> {v:.3f} ({delta:+.3f}){flag}")
        print(f"\nlargest drop: {worst:+.3f}")


if __name__ == "__main__":
    main()

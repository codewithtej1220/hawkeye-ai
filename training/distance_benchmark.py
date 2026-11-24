"""Range benchmark: how well is a weapon found as the armed person gets smaller (farther away)?

Held-out test images with an armed person are shrunk so the person is N pixels tall and placed in
a 640x480 frame (over a heavily blurred copy of the scene, so the surroundings stay plausible).
The same is done with unarmed people to count false alarms. Detection runs through the deployed
backend pipeline (ThreatDetector), so any second-stage method is included.

Sets:
  thermal concealed   TCWD: real thermal footage of weapons hidden under clothing (INFERNO view)
  thermal rifle (sim) AirSim rifle carriers in the white-hot render (already distant; native size only)
  rgb rifles / rgb handguns   real photos of people holding rifles / pistols and knives
  negatives           unarmed people: LLVIP thermal + RGB street, HIT-UAV drone thermal, COCO posture

    python distance_benchmark.py                     # deployed settings
    python distance_benchmark.py armed_classifier=0  # backend setting overrides
"""

from __future__ import annotations

import ast
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import albumentations  # noqa: F401  (Windows DLL deadlock workaround)
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project" / "backend"))
from app.config import Settings  # noqa: E402
from app.detector import ThreatDetector  # noqa: E402

TEST = ROOT / "HawkEye_Training" / "dataset_v3" / "images" / "test"
LABELS = ROOT / "HawkEye_Training" / "dataset_v3" / "labels" / "test"
HEIGHTS = (None, 120, 80, 50, 32, 20)  # target person height in px; None = native size
CANVAS_W, CANVAS_H = 640, 480
PER_SET = 50


def boxes(path: Path, w: int, h: int):
    out = []
    lbl = LABELS / (path.stem + ".txt")
    for line in lbl.read_text().splitlines() if lbl.exists() else []:
        if line.strip():
            c, cx, cy, bw, bh = map(float, line.split()[:5])
            out.append((int(c), np.array([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])))
    return out


def armed_people(gt):
    """(person box, weapon boxes held by that person) pairs."""
    people = [b for c, b in gt if c == 0]
    weapons = [b for c, b in gt if c == 1]
    pairs = []
    for p in people:
        ph, pw = p[3] - p[1], p[2] - p[0]
        held = [w for w in weapons
                if p[0] - 0.5 * pw <= (w[0] + w[2]) / 2 <= p[2] + 0.5 * pw and p[1] - 0.3 * ph <= (w[1] + w[3]) / 2 <= p[3] + 0.3 * ph]
        if held:
            pairs.append((p, held))
    return pairs


_BACKGROUNDS: dict[str, list] = {}


def backgrounds(kind: str) -> list:
    """Real scenes to place people into: night-street thermal or RGB (LLVIP test split)."""
    if kind not in _BACKGROUNDS:
        pattern = "thermal__llvip_ir__*.jpg" if kind == "thermal" else "rgb__llvip_rgb__*.jpg"
        files = sorted(TEST.glob(pattern))[1::25][:12]
        _BACKGROUNDS[kind] = [cv2.resize(cv2.imread(str(f)), (CANVAS_W, CANVAS_H)) for f in files]
    return _BACKGROUNDS[kind]


def place(image, target_box, height, kind="rgb", index=0):
    """Return (640x480 frame, transform(box) -> box) with the person `height` px tall.

    height None: the whole image, scaled to fit. Otherwise the person (plus a margin) is cut out,
    shrunk, and blended with soft edges into a real background of the same camera type, at the
    frame centre. Returns None if that would mean enlarging the person."""
    h, w = image.shape[:2]
    if height is None:
        s = min(1.0, CANVAS_W / w, CANVAS_H / h)
        frame = cv2.resize(image, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
        canvas[:frame.shape[0], :frame.shape[1]] = frame
        return canvas, (lambda b: b * s)
    ph = max(target_box[3] - target_box[1], 1)
    s = height / ph
    if s >= 1:
        return None
    pw = target_box[2] - target_box[0]
    mx, my = 0.35 * max(pw, ph), 0.2 * ph
    x0, y0 = int(max(0, target_box[0] - mx)), int(max(0, target_box[1] - my))
    x1, y1 = int(min(w, target_box[2] + mx)), int(min(h, target_box[3] + my))
    crop = image[y0:y1, x0:x1]
    cw, ch = max(1, round((x1 - x0) * s)), max(1, round((y1 - y0) * s))
    small = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_AREA).astype(np.float32)
    bg_pool = backgrounds(kind)
    canvas = bg_pool[index % len(bg_pool)].copy().astype(np.float32)
    ox, oy = CANVAS_W // 2 - cw // 2, CANVAS_H // 2 - ch // 2
    # soft-edged alpha so the cut-out has no hard rectangle border
    alpha = np.zeros((ch, cw), np.float32)
    bx, by = max(1, cw // 6), max(1, ch // 8)
    alpha[by:ch - by, bx:cw - bx] = 1
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(0.6, min(cw, ch) / 10))[..., None]
    region = canvas[oy:oy + ch, ox:ox + cw]
    canvas[oy:oy + ch, ox:ox + cw] = alpha * small + (1 - alpha) * region
    return canvas.astype(np.uint8), (lambda b: np.array([(b[0] - x0) * s + ox, (b[1] - y0) * s + oy,
                                                          (b[2] - x0) * s + ox, (b[3] - y0) * s + oy]))


def hit(detections, weapon_boxes, person_box):
    """A weapon is found if a weapon detection lands on a true weapon (generous for few-pixel
    objects), or the person is classified as armed."""
    for d in detections:
        if d.kind == "weapon":
            x = (d.bbox.x + d.bbox.width / 2) / 100 * CANVAS_W
            y = (d.bbox.y + d.bbox.height / 2) / 100 * CANVAS_H
            for wb in weapon_boxes:
                pad = max(6.0, 0.5 * max(wb[2] - wb[0], wb[3] - wb[1]))
                if wb[0] - pad <= x <= wb[2] + pad and wb[1] - pad <= y <= wb[3] + pad:
                    return True
        if d.kind == "person" and d.armed:
            x = (d.bbox.x + d.bbox.width / 2) / 100 * CANVAS_W
            y = (d.bbox.y + d.bbox.height / 2) / 100 * CANVAS_H
            if person_box[0] <= x <= person_box[2] and person_box[1] <= y <= person_box[3]:
                return True
    return False


def sim_frames():
    """AirSim rifle carriers rendered like the live simulator (white-hot), with mask-derived boxes."""
    tree = ast.parse((ROOT / "AirSimDemos" / "thermal.py").read_text(encoding="utf-8"))
    keep = [n for n in tree.body if (isinstance(n, ast.Assign) and any(getattr(t, "id", "").startswith(("WHITEHOT", "_noise")) for t in n.targets))
            or (isinstance(n, ast.FunctionDef) and n.name == "render_whitehot")]
    ns = {"np": np, "cv2": cv2}
    exec(compile(ast.Module(keep, []), "thermal.py", "exec"), ns)
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_INFERNO)[:, 0].astype(np.int32)
    out = []
    for f in sorted(glob.glob(str(ROOT / "AirSimDemos" / "HawkEye_Manual_Dataset" / "thermal" / "*" / "*.png"))):
        im = cv2.resize(cv2.imread(f), (CANVAS_W, CANVAS_H))
        blue = (im[..., 0] > 200) & (im[..., 1] < 40) & (im[..., 2] < 40)
        g = ((im.reshape(-1, 1, 3).astype(np.int32) - lut[None]) ** 2).sum(-1).argmin(1).astype(np.uint8).reshape(im.shape[:2])
        body = (((g >= 175) & ~blue).astype(np.uint8)) * 255
        body = cv2.morphologyEx(body, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        if not body.any() or not blue.any():
            continue
        scene = np.clip((g.astype(np.float32) - 25) / 0.25, 0, 255).astype(np.uint8)
        scene[body > 0] = 150
        frame = ns["render_whitehot"](scene, body, blue.astype(np.uint8) * 255)
        ys, xs = np.nonzero(body)
        wy, wx = np.nonzero(blue)
        out.append((frame, np.array([xs.min(), ys.min(), xs.max(), ys.max()], float),
                    [np.array([wx.min(), wy.min(), wx.max(), wy.max()], float)]))
    return out


def main() -> None:
    overrides = {}
    for arg in sys.argv[1:]:
        k, v = arg.split("=", 1)
        try:
            overrides[k] = float(v)
        except ValueError:
            overrides[k] = v
    det = ThreatDetector(Settings(**overrides))
    det.load()

    positives = {
        "thermal concealed (TCWD)": ("thermal__tcwd__*.jpg", "thermal"),
        "rgb rifles": ("rgb__rifles__*.jpg", "rgb"),
        "rgb handguns/knives": ("rgb__weapons_rgb__*.jpg", "rgb"),
    }
    negatives = {
        "thermal street (LLVIP)": ("thermal__llvip_ir__*.jpg", "thermal"),
        "thermal drone (HIT-UAV)": ("thermal__hituav__*.jpg", "thermal"),
        "rgb street (LLVIP)": ("rgb__llvip_rgb__*.jpg", "rgb"),
        "rgb everyday (COCO)": ("rgb__pose__*.jpg", "rgb"),
    }

    def analyse(frame, kind):
        if kind == "thermal":
            return det.process_views(None, frame, 0, "benchmark").detections
        return det.process_views(frame, None, 0, "benchmark").detections

    results = {"settings": overrides, "positives": {}, "negatives": {}}
    t0 = time.time()
    for name, (pattern, kind) in positives.items():
        rows = {}
        samples = []
        for f in sorted(TEST.glob(pattern)):
            img = cv2.imread(str(f))
            pairs = armed_people(boxes(f, img.shape[1], img.shape[0]))
            if pairs:
                samples.append((img, *max(pairs, key=lambda p: (p[0][3] - p[0][1]))))
            if len(samples) >= PER_SET:
                break
        for height in HEIGHTS:
            found = total = 0
            for i, (img, person, weapons) in enumerate(samples):
                placed = place(img, person, height, kind, i)
                if placed is None:
                    continue
                frame, tf = placed
                total += 1
                found += hit(analyse(frame, kind), [tf(w) for w in weapons], tf(person))
            rows[str(height or "native")] = [found, total]
        results["positives"][name] = rows
        print(f"[armed] {name:26s} " + "  ".join(f"{h}px={v[0]}/{v[1]}" for h, v in rows.items()), flush=True)

    sim = sim_frames()
    found = sum(hit(det.process_views(None, fr, 0, "benchmark").detections, wb, pb) for fr, pb, wb in sim)
    results["positives"]["thermal rifle (AirSim)"] = {"native": [found, len(sim)]}
    print(f"[armed] {'thermal rifle (AirSim)':26s} native={found}/{len(sim)}", flush=True)

    for name, (pattern, kind) in negatives.items():
        rows = {}
        samples = []
        for f in sorted(TEST.glob(pattern))[::3]:
            img = cv2.imread(str(f))
            gt = boxes(f, img.shape[1], img.shape[0])
            people = [b for c, b in gt if c == 0]
            if people and not any(c == 1 for c, _ in gt):
                samples.append((img, max(people, key=lambda b: b[3] - b[1])))
            if len(samples) >= PER_SET:
                break
        for height in HEIGHTS:
            flagged = total = 0
            for i, (img, person) in enumerate(samples):
                placed = place(img, person, height, kind, i)
                if placed is None:
                    continue
                total += 1
                dets = analyse(placed[0], kind)
                flagged += any(d.kind == "weapon" or (d.kind == "person" and d.armed) for d in dets)
            rows[str(height or "native")] = [flagged, total]
        results["negatives"][name] = rows
        print(f"[false] {name:26s} " + "  ".join(f"{h}px={v[0]}/{v[1]}" for h, v in rows.items()), flush=True)

    found_all = sum(v[0] for r in results["positives"].values() for v in r.values())
    total_all = sum(v[1] for r in results["positives"].values() for v in r.values())
    fp_all = sum(v[0] for r in results["negatives"].values() for v in r.values())
    neg_all = sum(v[1] for r in results["negatives"].values() for v in r.values())
    print(f"\nTOTAL armed found {found_all}/{total_all} ({found_all / max(total_all, 1):.1%}) | "
          f"false alarms {fp_all}/{neg_all} ({fp_all / max(neg_all, 1):.1%}) | {time.time() - t0:.0f}s")
    out = Path(__file__).parent / "eval" / f"distance_{'_'.join(f'{k}-{v}' for k, v in overrides.items()) or 'deployed'}.json"
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

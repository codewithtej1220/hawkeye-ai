"""Small-person search: does tiling (split the frame into overlapping tiles, enlarge each) find the
distant people the full-frame pass misses? Person recall per distance and time per frame."""
import time

import cv2
import numpy as np

import distance_benchmark as db
from ultralytics import YOLO

models = [YOLO(str(db.ROOT / "project" / "models" / m)) for m in ("hawkeye_real_yolo11m.pt", "hawkeye_v3.pt")]


def people(img, conf=0.25):
    out = []
    for m in models:
        r = m.predict(img, conf=conf, verbose=False)[0]
        out += [tuple(b) + (c,) for b, k, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()) if int(k) == 0]
    return out


def tiled(img, grid):
    h, w = img.shape[:2]
    th, tw = int(h / grid * 1.25), int(w / grid * 1.25)  # 25% overlap
    out = []
    for gy in range(grid):
        for gx in range(grid):
            y0 = min(h - th, int(gy * (h - th) / max(grid - 1, 1)))
            x0 = min(w - tw, int(gx * (w - tw) / max(grid - 1, 1)))
            tile = cv2.resize(img[y0:y0 + th, x0:x0 + tw], (640, int(640 * th / tw)), interpolation=cv2.INTER_CUBIC)
            s = tw / 640
            out += [(x0 + a * s, y0 + b * s, x0 + c * s, y0 + d * s, p) for a, b, c, d, p in people(tile)]
    return out


def found(dets, box):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(d[0] <= cx <= d[2] and d[1] <= cy <= d[3] for d in dets)


for name, (pattern, kind) in {"TCWD concealed": ("thermal__tcwd__*.jpg", "thermal"), "rgb handguns": ("rgb__weapons_rgb__*.jpg", "rgb"),
                              "rgb rifles": ("rgb__rifles__*.jpg", "rgb"), "thermal street": ("thermal__llvip_ir__*.jpg", "thermal")}.items():
    samples = []
    for f in sorted(db.TEST.glob(pattern)):
        img = cv2.imread(str(f))
        gt = db.boxes(f, img.shape[1], img.shape[0])
        pairs = db.armed_people(gt)
        ppl = [p for p, _ in pairs] if pairs else [b for c, b in gt if c == 0]
        if ppl:
            samples.append((img, max(ppl, key=lambda b: b[3] - b[1])))
        if len(samples) >= 30:
            break
    for height in (80, 50, 32, 20):
        counts = {"full": 0, "2x2": 0, "3x3": 0}
        times = {"full": 0.0, "2x2": 0.0, "3x3": 0.0}
        n = 0
        for i, (img, person) in enumerate(samples):
            placed = db.place(img, person, height, kind, i)
            if placed is None:
                continue
            frame, tf = placed
            view = cv2.applyColorMap(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_INFERNO) if kind == "thermal" else frame
            pb = tf(person)
            n += 1
            for mode, fn in (("full", lambda v: people(v)), ("2x2", lambda v: tiled(v, 2)), ("3x3", lambda v: tiled(v, 3))):
                t = time.time()
                d = fn(view)
                times[mode] += time.time() - t
                counts[mode] += found(d, pb)
        print(f"TILE {name:16s} {height:>3}px  person found: full {counts['full']}/{n}  2x2 {counts['2x2']}/{n}  3x3 {counts['3x3']}/{n}"
              f"  | ms/frame full {times['full'] / n * 1000:.0f}  2x2 {times['2x2'] / n * 1000:.0f}  3x3 {times['3x3'] / n * 1000:.0f}", flush=True)

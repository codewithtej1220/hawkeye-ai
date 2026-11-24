"""Why does the distant stage miss? Per distance: is the person detected at all, and what does the
armed classifier say on the detected box vs the true (oracle) box."""
import sys

import cv2
import numpy as np

import distance_benchmark as db

sys.path.insert(0, str(db.ROOT / "project" / "backend"))
from app.config import Settings  # noqa: E402
from app.detector import ThreatDetector  # noqa: E402
from ultralytics import YOLO  # noqa: E402

det = ThreatDetector(Settings())
det.load()
clf = YOLO(str(db.ROOT / "project" / "models" / "armed_classifier.pt"))
armed_idx = list(clf.names.values()).index("armed")


def crop(frame, box):
    x1, y1, x2, y2 = box
    pw, ph = x2 - x1, y2 - y1
    side = max(pw + 0.7 * max(pw, ph), ph * 1.4)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    h, w = frame.shape[:2]
    c = frame[int(max(0, cy - side / 2)):int(min(h, cy + side / 2)), int(max(0, cx - side / 2)):int(min(w, cx + side / 2))]
    return cv2.resize(c, (160, 160), interpolation=cv2.INTER_CUBIC)


def p_armed(img):
    return float(clf.predict(img, imgsz=160, verbose=False)[0].probs.data[armed_idx])


sets = {"TCWD concealed": ("thermal__tcwd__*.jpg", "thermal", True), "rgb handguns": ("rgb__weapons_rgb__*.jpg", "rgb", True),
        "rgb rifles": ("rgb__rifles__*.jpg", "rgb", True), "thermal street (neg)": ("thermal__llvip_ir__*.jpg", "thermal", False)}
for name, (pattern, kind, armed) in sets.items():
    samples = []
    for f in sorted(db.TEST.glob(pattern))[:: (1 if armed else 3)]:
        img = cv2.imread(str(f))
        gt = db.boxes(f, img.shape[1], img.shape[0])
        if armed:
            pairs = db.armed_people(gt)
            if pairs:
                samples.append((img, max(pairs, key=lambda p: p[0][3] - p[0][1])[0]))
        else:
            people = [b for c, b in gt if c == 0]
            if people and not any(c == 1 for c, _ in gt):
                samples.append((img, max(people, key=lambda b: b[3] - b[1])))
        if len(samples) >= 30:
            break
    for height in (120, 80, 50, 32, 20):
        detected = 0
        oracle, onbox = [], []
        for i, (img, person) in enumerate(samples):
            placed = db.place(img, person, height, kind, i)
            if placed is None:
                continue
            frame, tf = placed
            pb = tf(person)
            view = cv2.applyColorMap(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_INFERNO) if kind == "thermal" else frame
            oracle.append(p_armed(crop(view, pb)))
            det._acquire(True)
            try:
                raw = det._predict(view)
            finally:
                det._lock.release()
            cx, cy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
            hits = [d for d in raw if d.label == "person" and d.xyxy[0] <= cx <= d.xyxy[2] and d.xyxy[1] <= cy <= d.xyxy[3]]
            if hits:
                detected += 1
                onbox.append(p_armed(crop(view, max(hits, key=lambda d: d.confidence).xyxy)))
        n = len(oracle)
        print(f"DIAG {name:22s} {height:>3}px  person detected {detected}/{n}  "
              f"p_armed oracle-box mean {np.mean(oracle):.2f} (>=0.8: {np.mean(np.array(oracle) >= 0.8):.0%})  "
              f"detected-box mean {np.mean(onbox) if onbox else float('nan'):.2f}", flush=True)

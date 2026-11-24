"""Video-style range test: per person and distance, a 6-frame clip with small frame-to-frame changes
(position jitter, slight scale change, sensor noise - a hovering drone), run through the backend and
the per-person tracker exactly as the live stream / video analysis does. Counts people confirmed
armed (armed in >= 2 of 6 frames) - armed clips should be confirmed, unarmed clips should not.

    python temporal_benchmark.py
"""
import sys

import cv2
import numpy as np

import distance_benchmark as db

sys.path.insert(0, str(db.ROOT / "project" / "backend"))
from app.config import Settings  # noqa: E402
from app.detector import ThreatDetector  # noqa: E402
from app.tracking import ArmedTracker  # noqa: E402

FRAMES = 6
HEIGHTS = (120, 80, 50, 32)
PER_SET = 25
overrides = {}
for arg in sys.argv[1:]:
    k, v = arg.split("=", 1)
    overrides[k] = {"True": True, "False": False}.get(v, v)
    try:
        overrides[k] = float(v)
    except ValueError:
        pass
det = ThreatDetector(Settings(**overrides))
det.load()
print("settings", overrides, flush=True)


def clip(img, person, height, kind, index):
    """FRAMES frames of the same scene with small variations."""
    rng = np.random.default_rng(index)
    frames = []
    for k in range(FRAMES):
        h = height * (1 + rng.uniform(-0.06, 0.06))
        placed = db.place(img, person, h, kind, index)
        if placed is None:
            return None, None
        frame, tf = placed
        dx, dy = rng.integers(-6, 7, 2)
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        frame = cv2.warpAffine(frame, m, (db.CANVAS_W, db.CANVAS_H), borderMode=cv2.BORDER_REFLECT)
        frame = np.clip(frame.astype(np.float32) + rng.normal(0, 2.0, frame.shape), 0, 255).astype(np.uint8)
        pb = tf(person) + np.array([dx, dy, dx, dy])
        frames.append((frame, pb))
    return frames, None


def run_clip(frames, kind):
    """Returns (raised to High, flagged at all) for the clip's person: High = an alert; flagged = High
    or an amber "possible weapon (distant)" on that person."""
    tracker = ArmedTracker(det.settings.track_window, det.settings.track_confirm_hits)
    high = flagged = False
    for frame, pb in frames:
        p = (det.process_views(None, frame, 0, "clip") if kind == "thermal" else det.process_views(frame, None, 0, "clip"))
        dets = tracker.update(p.detections)
        summary = det._summarize(dets, p.distant_mode)
        cx, cy = (pb[0] + pb[2]) / 2 / db.CANVAS_W * 100, (pb[1] + pb[3]) / 2 / db.CANVAS_H * 100
        on_person = [d for d in dets if d.kind == "person" and d.bbox.x <= cx <= d.bbox.x + d.bbox.width
                     and d.bbox.y <= cy <= d.bbox.y + d.bbox.height]
        high = high or summary.level.value in ("High", "Critical")
        flagged = flagged or high or any(d.armed for d in on_person)
    return high, flagged


def load(pattern, armed):
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
        if len(samples) >= PER_SET:
            break
    return samples


totals = {"armed": [0, 0, 0], "false": [0, 0, 0]}
for name, pattern, kind, armed in (("thermal concealed", "thermal__tcwd__*.jpg", "thermal", True),
                                    ("rgb rifles", "rgb__rifles__*.jpg", "rgb", True),
                                    ("rgb handguns", "rgb__weapons_rgb__*.jpg", "rgb", True),
                                    ("thermal street", "thermal__llvip_ir__*.jpg", "thermal", False),
                                    ("rgb everyday", "rgb__pose__*.jpg", "rgb", False)):
    samples = load(pattern, armed)
    cells = []
    for height in HEIGHTS:
        n = hi = fl = 0
        for i, (img, person) in enumerate(samples):
            frames, _ = clip(img, person, height, kind, i)
            if frames is None:
                continue
            high, flagged = run_clip(frames, kind)
            n += 1
            hi += high
            fl += flagged
        cells.append(f"{height}px High {hi}/{n} flagged {fl}/{n}")
        t = totals["armed" if armed else "false"]
        t[0] += hi
        t[1] += fl
        t[2] += n
    print(f"[{'armed' if armed else 'false'}] {name:18s} " + " | ".join(cells), flush=True)
a, f = totals["armed"], totals["false"]
print(f"\nCLIPS armed: High {a[0]}/{a[2]} ({a[0] / max(a[2], 1):.1%}), flagged incl. possible {a[1]}/{a[2]} ({a[1] / max(a[2], 1):.1%})"
      f" | unarmed: High {f[0]}/{f[2]} ({f[0] / max(f[2], 1):.1%}), flagged {f[1]}/{f[2]} ({f[1] / max(f[2], 1):.1%})")

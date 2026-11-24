"""Quick screen of candidate models on the tests most sensitive to fine-tuning (same samples for all).

    python screen_models.py model_a.pt model_b.pt ...
"""
import sys
from pathlib import Path

import cv2

import evaluate_modalities as ev

PAIRS = 300


def screen(path_or_model) -> dict:
    from ultralytics import YOLO
    model = YOLO(path_or_model) if isinstance(path_or_model, str) else path_or_model
    out = {}
    thermal = sorted((ev.DATASET / "images" / "test").glob("thermal__llvip_ir__*.jpg"))[::2][:PAIRS]
    for env_name in ("normal", "thermal crossover"):
        per = {}
        for t_path in thermal:
            rgb = cv2.imread(str(t_path).replace("thermal__llvip_ir__", "rgb__llvip_rgb__"))
            grey = cv2.cvtColor(cv2.imread(str(t_path)), cv2.COLOR_BGR2GRAY)
            rgb_e, grey_e = ev.ENVIRONMENTS[env_name](rgb, grey)
            gt = ev.load_labels(t_path, grey.shape[1], grey.shape[0])
            for method, dets in ev.run_methods(model, rgb_e, grey_e).items():
                if not method.startswith("_"):
                    per.setdefault(method, []).append((dets, gt))
        for method in ("rgb", "thermal_inferno", "rgbt"):
            out[f"{env_name}/{method}"] = ev.average_precision(per[method], "person")
    for name, pattern, cls, step in (("dangerous/person", "rgb__dangerous_objects__*.jpg", "person", 3),
                                     ("weapons_rgb/weapon", "rgb__weapons_rgb__*.jpg", "weapon", 1),
                                     ("rifles/weapon", "rgb__rifles__*.jpg", "weapon", 1),
                                     ("tcwd/weapon", "thermal__tcwd__*.jpg", "weapon", 1),
                                     ("pose/person", "rgb__pose__*.jpg", "person", 1)):
        samples = []
        for f in sorted((ev.DATASET / "images" / "test").glob(pattern))[::step][:PAIRS]:
            img = cv2.imread(str(f))
            if pattern.startswith("thermal__"):
                img = cv2.applyColorMap(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_INFERNO)
            samples.append((ev.predict(model, img), ev.load_labels(f, img.shape[1], img.shape[0])))
        out[name] = ev.average_precision(samples, cls)
    out["pose/false_alarm_rate"] = ev.false_alarm_suite(model, "rgb__pose__*.jpg", 545)["false_weapon_rate"]
    return out


if __name__ == "__main__":
    rows = {Path(p).stem: screen(p) for p in sys.argv[1:]}
    keys = list(next(iter(rows.values())))
    print("\nSCREEN " + "metric".ljust(34) + "".join(name[:16].rjust(17) for name in rows))
    for k in keys:
        print("SCREEN " + k.ljust(34) + "".join(f"{rows[n][k]:17.3f}" for n in rows))

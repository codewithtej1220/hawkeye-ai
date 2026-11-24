"""Screen a confidence-averaging ensemble of two models against the single models.

Both models run on the same image; detections of the same class that overlap are merged with the
mean confidence, and a detection found by only one model keeps half its confidence. Agreement is
kept, a single model's false alarm is damped.

    python screen_ensemble.py model_a.pt model_b.pt
"""
import sys

import evaluate_modalities as ev
import screen_models
from ultralytics import YOLO

_original_predict = ev.predict


def ensemble_predict(models, image, match_iou=0.55):
    a, b = (_original_predict(m, image) for m in models)
    merged, used = [], set()
    for label, conf_a, box_a in a:
        best, best_iou = None, match_iou
        for j, (label_b, conf_b, box_b) in enumerate(b):
            if j in used or label_b != label:
                continue
            overlap = ev.iou(box_a, box_b)
            if overlap >= best_iou:
                best, best_iou = j, overlap
        if best is None:
            merged.append((label, conf_a / 2, box_a))
        else:
            used.add(best)
            _, conf_b, box_b = b[best]
            w = conf_a + conf_b
            box = tuple((conf_a * x + conf_b * y) / w for x, y in zip(box_a, box_b))
            merged.append((label, (conf_a + conf_b) / 2, box))
    merged += [(label, conf / 2, box) for j, (label, conf, box) in enumerate(b) if j not in used]
    return merged


class Ensemble:
    def __init__(self, paths):
        self.models = [YOLO(p) for p in paths]


if __name__ == "__main__":
    ens = Ensemble(sys.argv[1:3])
    ev.predict = lambda model, image: ensemble_predict(ens.models, image) if model is ens else _original_predict(model, image)
    result = screen_models.screen(ens)
    print("\nSCREEN " + "metric".ljust(34) + "ensemble".rjust(17))
    for k, v in result.items():
        print("SCREEN " + k.ljust(34) + f"{v:17.3f}")

"""Evaluate the trained model per domain, and visually check it on AirSim frames.

    python evaluate.py                                   # thermal vs RGB test metrics
    python evaluate.py --sim-dir ..\\AirSimDemos\\HawkEye_Manual_Dataset\\thermal

The --sim-dir run saves annotated predictions to runs/sim_check/ so you can see
how the real-data model transfers to the simulator's rendering.
"""

import argparse
import tempfile
from pathlib import Path

import yaml

from sources import MERGED_DIR, OUTPUT_MODEL, RUNS_DIR


def evaluate_domain(model, domain: str, args) -> None:
    images = sorted((MERGED_DIR / "images" / "test").glob(f"{domain}__*.jpg"))
    if not images:
        print(f"[{domain}] no test images")
        return
    with tempfile.TemporaryDirectory() as tmp:
        listing = Path(tmp) / "test.txt"
        listing.write_text("\n".join(str(p) for p in images))
        data = Path(tmp) / "data.yaml"
        data.write_text(yaml.safe_dump({"path": str(MERGED_DIR), "train": str(listing), "val": str(listing),
                                        "names": model.names}))
        metrics = model.val(data=str(data), split="val", imgsz=args.imgsz, batch=8, device=args.device,
                            plots=False, verbose=False, project=str(RUNS_DIR), name=f"eval_{domain}", exist_ok=True)
    print(f"\n[{domain}] {len(images)} test images")
    for idx, class_id in enumerate(metrics.box.ap_class_index):
        print(f"  {model.names[int(class_id)]:7s} P={metrics.box.p[idx]:.3f} R={metrics.box.r[idx]:.3f} "
              f"AP50={metrics.box.ap50[idx]:.3f} AP50-95={metrics.box.ap[idx]:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(OUTPUT_MODEL))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--sim-dir", help="folder of AirSim thermal frames to annotate")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    for domain in ("thermal", "rgb"):
        evaluate_domain(model, domain, args)

    if args.sim_dir:
        out = RUNS_DIR / "sim_check"
        model.predict(source=args.sim_dir, conf=args.conf, imgsz=args.imgsz, device=args.device,
                      save=True, project=str(RUNS_DIR), name="sim_check", exist_ok=True, verbose=False)
        print(f"\nAnnotated simulator frames saved to {out}")


if __name__ == "__main__":
    main()

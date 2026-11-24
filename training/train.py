"""Fine-tune YOLO11m on the merged real thermal + RGB dataset.

Defaults are sized for a 6 GB laptop GPU (RTX 4050). On completion the best
checkpoint is copied to project/models/hawkeye_real_yolo11m.pt, which the
backend loads by default.

    python train.py                     # full run
    python train.py --smoke             # quick VRAM/speed check, no export
    python train.py --resume            # continue an interrupted run
"""

import argparse
import shutil

# Ultralytics lazily imports albumentations -> sklearn -> pandas -> pyarrow mid-training.
# On this Windows setup, loading pyarrow's DLL that late (after CUDA init) deadlocks,
# so pull the chain in up front.
try:
    import albumentations  # noqa: F401
except ImportError:
    pass

from sources import MERGED_DIR, OUTPUT_MODEL, RUNS_DIR

RUN_NAME = "hawkeye_real_yolo11m"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11m.pt", help="starting weights (COCO-pretrained YOLO11m)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8, help="8 fits in 6 GB VRAM at 640px with AMP")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="1 epoch on 10%% of the data to check VRAM/speed; does not export the model")
    args = parser.parse_args()
    run_name = "smoke" if args.smoke else RUN_NAME

    from ultralytics import YOLO

    data_yaml = MERGED_DIR / "hawkeye_real.yaml"
    if not data_yaml.exists():
        raise SystemExit("Merged dataset not found; run prepare_dataset.py first")

    if args.resume:
        model = YOLO(str(RUNS_DIR / RUN_NAME / "weights" / "last.pt"))
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(
            data=str(data_yaml),
            epochs=1 if args.smoke else args.epochs,
            fraction=0.1 if args.smoke else 1.0,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            patience=args.patience,
            project=str(RUNS_DIR),
            name=run_name,
            exist_ok=True,
            optimizer="AdamW",
            lr0=0.001,
            cos_lr=True,
            close_mosaic=10,
            amp=True,
            # Thermal frames are near-monochrome palettes: keep hue jitter tiny so
            # the INFERNO/white-hot mapping is not scrambled.
            hsv_h=0.005,
            hsv_s=0.4,
            hsv_v=0.4,
            degrees=5.0,
            flipud=0.1,  # drone views are often top-down
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.05,
            plots=True,
        )

    if args.smoke:
        print("\nSmoke test finished; model not exported.")
        return

    best = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, OUTPUT_MODEL)
    print(f"\nCopied {best} -> {OUTPUT_MODEL}")

    print("\nEvaluating on the held-out test split...")
    metrics = YOLO(str(OUTPUT_MODEL)).val(data=str(data_yaml), split="test", imgsz=args.imgsz,
                                          batch=args.batch, device=args.device,
                                          project=str(RUNS_DIR), name=RUN_NAME + "_test", exist_ok=True)
    for i, name in metrics.names.items():
        print(f"  {name:7s} AP50={metrics.box.ap50[i]:.3f}  AP50-95={metrics.box.ap[i]:.3f}")
    print(f"  all     mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}")


if __name__ == "__main__":
    main()

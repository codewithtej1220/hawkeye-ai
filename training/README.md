# HawkEye real-data training

Trains a single **YOLO11m** detector with two classes, `person` and `weapon`, on **real**
thermal and RGB imagery. It replaces the original models, which were trained on simulator
frames where weapons were painted solid blue.

## Datasets

| Source | Modality | What it contributes |
|---|---|---|
| [HIT-UAV](https://www.kaggle.com/datasets/pandrii000/hituav-a-highaltitude-infrared-thermal-dataset) | Thermal, drone-mounted | People seen from 60–130 m altitude, which matches HawkEye's viewpoint |
| [TCWD: Thermal Concealed Weapon Detection](https://www.kaggle.com/datasets/noormuhannad/thermal-concealed-weapon-detection-dataset) | Thermal | Real weapons as they appear on a thermal camera |
| [Weapon Detection Dataset for YOLOv5](https://www.kaggle.com/datasets/raghavnanjappan/weapon-dataset-for-yolov5) | RGB | 4k real photos of handguns, rifles and knives |

Edit `sources.py` to add more. Any YOLO-format Kaggle dataset works, and classes are mapped
by name.

## How the data is merged (`prepare_dataset.py`)

- **Class remapping.** Every source's classes are mapped onto `person` / `weapon`
  (for example `Person`, `pistol` and `knife` are kept, while `Car` and `Bicycle` are
  dropped).
- **Person pseudo-labels.** Some sources label only weapons. In those, the people in the
  frame would otherwise be learned as "background", so COCO-pretrained YOLO11m labels them
  first.
- **Palette augmentation.** Real thermal cameras output white-hot greyscale, but the
  AirSim feed uses the INFERNO colormap. Half the thermal training images (and a quarter
  of the RGB ones) get an extra INFERNO copy, so the model handles both palettes.
- **Splits.** Each source keeps its own train/val/test split. Sources that ship without one
  get a deterministic 10% split.

## Run

Prerequisite: a Kaggle API token at `%USERPROFILE%\.kaggle\kaggle.json`.

```bat
pip install -r requirements.txt
RUN_TRAINING.bat
```

The same pipeline, step by step:

```bat
python download_datasets.py           :: ~0.9 GB download
python prepare_dataset.py --delete-raw
python train.py                       :: --smoke for a quick check, --resume to continue
python evaluate.py --sim-dir ..\AirSimDemos\HawkEye_Manual_Dataset\thermal
```

The defaults (`batch=8`, `imgsz=640`, AMP) fit a 6 GB laptop GPU.

## Integration

- `train.py` copies the best weights to `project/models/hawkeye_real_yolo11m.pt`.
- The backend (`project/backend/app/config.py`) loads that model by default. It falls back
  to the legacy `weapon_model.pt` if the file is missing.
- `AirSimDemos/thermal.py` switches from blue weapons to **cold-metal** weapons (dark
  against the warm body, as on a real thermal camera) when the new model exists. To force
  a style, set `HAWKEYE_WEAPON_STYLE=realistic` or `HAWKEYE_WEAPON_STYLE=blue`.

## Evaluation and model selection (v3)

| Script | Purpose |
|---|---|
| `evaluate_modalities.py` | AP50 for RGB / thermal grey / thermal INFERNO / RGBT fusion / auto, under 5 simulated environments (normal, dark, fog/smoke, glare, thermal crossover), plus palette, posture, everyday false-alarm and AirSim suites. `--compare old.json` reports every drop. |
| `build_v3_dataset.py` | v3 dataset: INFERNO thermal copies, COCO posture photos as hard negatives, rotated thermal people |
| `interpolate_weights.py` | weight-space interpolation between two checkpoints (WiSE-FT) |
| `screen_models.py` / `screen_ensemble.py` | quick side-by-side screen of candidate models / of an averaging ensemble |

Results are in `eval/`.

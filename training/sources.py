"""Real-world datasets used to train the HawkEye unified person/weapon detector.

Every source is a public Kaggle dataset in YOLO format. Class names found in each
source's own dataset YAML are mapped onto the two HawkEye classes by keyword
(see CLASS_KEYWORDS); `class_map` overrides that per source when needed.
"""

from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent
DATASETS_DIR = TRAINING_DIR / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
MERGED_DIR = DATASETS_DIR / "hawkeye_real"
RUNS_DIR = TRAINING_DIR / "runs"
MODELS_DIR = PROJECT_ROOT / "project" / "models"
OUTPUT_MODEL = MODELS_DIR / "hawkeye_real_yolo11m.pt"

# Final model classes. Both names are already in the backend's person/weapon label sets.
CLASS_NAMES = ["person", "weapon"]

CLASS_KEYWORDS = {
    "person": ["person", "people", "human", "pedestrian", "man", "woman"],
    "weapon": ["weapon", "gun", "pistol", "handgun", "rifle", "knife", "firearm",
               "shotgun", "revolver", "concealed", "grenade", "sword"],
}

SOURCES = [
    {
        # Drone-mounted thermal camera, 60-130 m altitude. Person/Car/Bicycle/OtherVehicle.
        "slug": "hituav",
        "kaggle": "pandrii000/hituav-a-highaltitude-infrared-thermal-dataset",
        "modality": "thermal",
        "approx_mb": 200,
        "prune": [],
    },
    {
        # Real thermal-camera footage of people carrying (concealed) weapons.
        "slug": "tcwd",
        "kaggle": "noormuhannad/thermal-concealed-weapon-detection-dataset",
        "modality": "thermal",
        "approx_mb": 454,
        # Raw videos and segmentation masks are not needed for detection training.
        "prune": ["TCWD/raw_videos", "TCWD/masks"],
    },
    {
        # Real RGB photos of handguns and knives.
        "slug": "weapons_rgb",
        "kaggle": "raghavnanjappan/weapon-dataset-for-yolov5",
        "modality": "rgb",
        "approx_mb": 227,
        "prune": [],
    },
    # ---- v2 additions: humans in any pose/camera, and every kind of weapon ----
    {
        # LLVIP: 4,200 infrared + 4,200 aligned visible-light images of pedestrians from
        # street surveillance cameras (mostly night). One download, two sources.
        "slug": "llvip_ir",
        "raw": "llvip",
        "subdir": "LLVIP_converted/infrared",
        "kaggle": "penjin/llvip-converted",
        "modality": "thermal",
        "approx_mb": 1110,
        "max_train": 2500,  # consecutive video frames; more adds little
        "prune": [],
    },
    {
        "slug": "llvip_rgb",
        "raw": "llvip",
        "subdir": "LLVIP_converted/visible",
        "kaggle": "penjin/llvip-converted",
        "modality": "rgb",
        "approx_mb": 0,
        "max_train": 1500,
        "prune": [],
    },
    {
        # 17k RGB images: firearm, pistol, knife, grenade, rocket (launcher), swords, fire.
        "slug": "dangerous_objects",
        "kaggle": "alinadilawaiz/dangerous-objects-dataset",
        "modality": "rgb",
        "approx_mb": 917,
        "max_train": 5000,
        "class_map": {"fire": None, "rocket": "weapon"},
        "prune": ["vid1.mp4", "vid2.mp4"],
    },
    {
        # 2.4k RGB images of rifles / assault rifles.
        "slug": "rifles",
        "kaggle": "trnphcnguyn/rifle-wepon-detection-2-v1i-yolov11",
        "modality": "rgb",
        "approx_mb": 109,
        "prune": [],
    },
]

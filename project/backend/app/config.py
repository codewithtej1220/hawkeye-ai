from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASE_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "HawkEye AI Backend"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )
    # The dashboard may land on another local port (Vite picks 5174+ when 5173 is busy)
    cors_origin_regex: str = r"https?://(localhost|127\.0\.0\.1)(:\d+)?"

    # Diagnostics: save every Nth received live frame (thermal + RGB + detections) to this folder
    live_record_dir: str = ""
    live_record_every: int = 3

    # Event log (officials' review): SQLite database + annotated snapshots
    event_db_path: str = "data/hawkeye_events.db"
    event_snapshot_dir: str = "data/snapshots"
    event_max: int = 20000              # oldest events (and their snapshots) are pruned beyond this
    event_threat_update_s: float = 15.0   # while High/Critical persists, log an update this often
    event_presence_update_s: float = 60.0  # while only unarmed people are in view
    live_view_width: int = 720          # width of the per-camera live views sent to the dashboard
    # Drone stream only: a person or weapon box wider than this share of the frame is not real. When
    # the thermal picture has nothing warm in it, the detector sometimes outlines most of the frame as
    # a person or a weapon (15 of 80 empty-forest frames). The soldier at 10 m with the zoom lens fills
    # ~15% of the width. Uploaded close-up photos are not filtered.
    live_max_person_width_pct: float = 40.0

    video_source: str = "airsim"
    frame_interval_ms: int = 250
    jpeg_quality: int = 82

    # Unified person+weapon model trained on real thermal/RGB data (training/train.py).
    # Falls back to the legacy synthetic-data model until that has been trained.
    model_path: str = "../models/hawkeye_real_yolo11m.pt"
    fallback_model_path: str = "../models/weapon_model.pt"
    # Second model for a confidence-averaging ensemble (empty string disables). v3 adds postures,
    # INFERNO thermal and everyday-object hard negatives; v2 is stronger on night-street RGB. The
    # pair beat or matched v2 on every evaluation suite (training/eval) and cut false weapons on
    # everyday photos from 17.6% to ~6%, at roughly twice the inference time.
    ensemble_model_path: str = "../models/hawkeye_v3.pt"
    model_type: str = "ultralytics"
    device: str = "cuda:0"
    # Minimum confidence for any detection (people). 0.15 rather than 0.20 because the ensemble halves
    # the confidence of a person only one model sees; at 0.15 it finds at least as many people as v2
    # alone did at 0.20 on every validation set.
    confidence_threshold: float = 0.15
    weapon_confidence_threshold: float = 0.35  # weapons need more evidence than people

    # SAHI tiling was measured to cause false weapons on ordinary scenes (62/80 weapon-free street
    # images flagged); whole-frame detection gave 0/240. Off by default.
    use_sahi: bool = False
    slice_height: int = 512
    slice_width: int = 512
    overlap_height_ratio: float = 0.3  # Increased overlap for better small object detection
    overlap_width_ratio: float = 0.3

    # Stand-off weapon identification ("detect then zoom"): at surveillance range a weapon is
    # only a few pixels wide, so each detected person is cropped, upscaled and re-examined for
    # weapons - a software equivalent of slewing a zoom camera onto the target.
    zoom_weapon_pass: bool = True
    zoom_pad: float = 2.5           # crop side = person box long side x this (context reduces false alarms)
    zoom_min_crop: int = 64         # px; smallest crop taken around very distant people
    zoom_size: int = 640            # crops are upscaled to this before re-detection
    zoom_confidence: float = 0.6    # tuned: 2% false-alarm rate on weapon-free footage
    zoom_max_people: int = 8        # caps the extra latency per frame
    zoom_min_person_confidence: float = 0.20  # don't zoom on faint (possibly false) person detections

    # Distant / concealed weapons - second stage. After normal detection (full frame + zoom), each
    # small (distant) person with no weapon found is judged by an "armed person" classifier that
    # looks at the whole low-resolution person: protrusions, the pose of holding something, or in
    # thermal the cold patch of metal under clothing. Weapons too small to box are still caught.
    # Small-person search (part of the distant stage): people under ~80 px are mostly missed by the
    # full-frame pass, so the frame is also split into overlapping 2x2 and 3x3 tiles, each enlarged,
    # and searched for PEOPLE only (weapons are then checked on those people by zoom + classifier).
    # Measured: thermal people at 32 px found 6/30 -> 26/30. Costs ~0.6 s per frame, so the live
    # stream runs it every Nth frame and the tracker carries results in between.
    # Distant-weapon stage switch:
    #   "off"   - full frame + zoom only (no extra false alarms; distant weapons mostly missed)
    #   "flag" / "alert" (default "flag", both behave the same now) - small-person search + armed-person
    #             classifier + thermal-rifle specialist. Their findings are never shown on a single
    #             frame; a person counts as armed (High) on them only once tracking confirms it over
    #             frames (detector.finalize). Weapons the detector or zoom pass sees count at once.
    # Measured on 6-frame clips at 32-120 px (training/temporal_benchmark.py): armed people found 5% (off)
    # vs 31% (flag/alert); unarmed clips raised to High 0% (off) vs 11% (alert).
    distant_mode: str = "flag"
    distant_search_grids: str = "2,3"
    distant_search_live_every: int = 3
    armed_classifier_path: str = "../models/armed_classifier.pt"
    armed_classifier_max_height: int = 160  # px; people taller than this are left to the detector
    armed_classifier_threshold: float = 0.80  # tuned on the distance benchmark (training/eval)

    # Thermal-rifle specialist. The main ensemble was trained on real footage, where an openly carried
    # rifle seen by a thermal camera is rare. hawkeye_thermal_rifle.pt is v3 fine-tuned on auto-labelled
    # simulator frames of an armed and an unarmed soldier mixed with real data
    # (HawkEye_Training/TRAIN_V4.bat). Replacing the main models with fine-tuned copies lost accuracy on
    # real data (training/regression_check.py), so it is only asked for a second opinion in the zoom
    # pass, on thermal views, about a person on whom the main ensemble found no weapon.
    # Comma-separated list; empty string disables.
    weapon_specialist_paths: str = "../models/hawkeye_thermal_rifle.pt"
    # Measured (training/eval/specialist_sweep.txt, threshold 0.50): simulator rifles found 6/52 -> 18/52;
    # real weapon-free thermal images given a "possible weapon" note 8/300 -> 33/300. Because of that
    # second number its findings are never shown on a single frame: a person counts as armed on its
    # evidence only once tracking confirms it in track_confirm_hits of the last track_window frames.
    weapon_specialist_confidence: float = 0.50

    # Live stream / video: a person armed in >= track_confirm_hits of their last track_window
    # frames is confirmed armed and stays marked when a frame misses the weapon
    track_window: int = 6
    track_confirm_hits: int = 2

    # User-supplied media (dashboard "Analyze media" section)
    max_upload_mb: int = 500
    video_sample_fps: float = 2.0   # frames analysed per second of video
    video_max_frames: int = 120     # longer videos are sampled more sparsely
    video_key_frames: int = 6       # annotated moments returned for review

    person_labels: str = "person,human,thermal"
    weapon_labels: str = "gun,pistol,rifle,knife,weapon,firearm"

    @property
    def resolved_event_db_path(self) -> Path:
        path = Path(self.event_db_path)
        return path if path.is_absolute() else BASE_DIR / path

    @property
    def resolved_event_snapshot_dir(self) -> Path:
        path = Path(self.event_snapshot_dir)
        return path if path.is_absolute() else BASE_DIR / path

    @property
    def resolved_armed_classifier_path(self) -> str | None:
        if not self.armed_classifier_path:
            return None
        path = Path(self.armed_classifier_path)
        path = path if path.is_absolute() else BASE_DIR / path
        return str(path) if path.exists() else None

    @property
    def resolved_specialist_paths(self) -> list[str]:
        paths = []
        for item in self.weapon_specialist_paths.split(","):
            if item.strip():
                path = Path(item.strip())
                path = path if path.is_absolute() else BASE_DIR / path
                if path.exists():
                    paths.append(str(path))
        return paths

    @property
    def resolved_ensemble_path(self) -> str | None:
        if not self.ensemble_model_path:
            return None
        path = Path(self.ensemble_model_path)
        path = path if path.is_absolute() else BASE_DIR / path
        return str(path) if path.exists() else None

    @property
    def resolved_model_path(self) -> str:
        for candidate in (self.model_path, self.fallback_model_path):
            path = Path(candidate) if Path(candidate).is_absolute() else BASE_DIR / candidate
            if path.exists():
                return str(path)
        return str(BASE_DIR / self.model_path)

    @property
    def person_label_set(self) -> set[str]:
        return {label.strip().lower() for label in self.person_labels.split(",") if label.strip()}

    @property
    def weapon_label_set(self) -> set[str]:
        return {label.strip().lower() for label in self.weapon_labels.split(",") if label.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()

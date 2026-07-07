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

    video_source: str = "airsim"
    frame_interval_ms: int = 250
    jpeg_quality: int = 82

    model_path: str = "../models/weapon_model.pt"
    model_type: str = "ultralytics"
    device: str = "cuda:0"
    confidence_threshold: float = 0.20  # Lowered for better weapon detection in thermal images

    use_sahi: bool = True
    slice_height: int = 512
    slice_width: int = 512
    overlap_height_ratio: float = 0.3  # Increased overlap for better small object detection
    overlap_width_ratio: float = 0.3

    person_labels: str = "person,human,thermal"
    weapon_labels: str = "gun,pistol,rifle,knife,weapon,firearm"

    @property
    def resolved_model_path(self) -> str:
        model_path = Path(self.model_path)
        return str(model_path if model_path.is_absolute() else BASE_DIR / model_path)

    @property
    def person_label_set(self) -> set[str]:
        return {label.strip().lower() for label in self.person_labels.split(",") if label.strip()}

    @property
    def weapon_label_set(self) -> set[str]:
        return {label.strip().lower() for label in self.weapon_labels.split(",") if label.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()

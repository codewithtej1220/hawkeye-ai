from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ThreatLevel(str, Enum):
    clear = "Clear"
    elevated = "Elevated"
    high = "High"
    critical = "Critical"


class BoundingBox(BaseModel):
    x: float = Field(ge=0, le=100)
    y: float = Field(ge=0, le=100)
    width: float = Field(ge=0, le=100)
    height: float = Field(ge=0, le=100)


class Detection(BaseModel):
    id: str
    label: str
    kind: Literal["person", "weapon", "other"]
    confidence: float = Field(ge=0, le=1)
    bbox: BoundingBox
    armed: bool = False
    weapon_count: int = 0
    associated_person_id: str | None = None


class ThreatSummary(BaseModel):
    level: ThreatLevel
    total_people: int
    armed_people: int
    weapon_count: int
    risk_score: int
    average_confidence: int


class FramePayload(BaseModel):
    frame_id: int
    timestamp: str
    source: str
    image: str
    annotated_image: str | None = None  # Image with bounding boxes drawn
    detections: list[Detection]
    summary: ThreatSummary


class HealthResponse(BaseModel):
    status: Literal["ok"]
    detector_ready: bool
    video_source: str
    use_sahi: bool

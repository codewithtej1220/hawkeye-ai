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
    sensor: str | None = None  # which sensor saw it: "rgb", "thermal" or "rgb+thermal"
    # weapons: how it was found - "detector" (full frame), "zoom" (magnified look at a person) or
    # "classifier" (distant person judged armed; weapon too small to box)
    evidence: str | None = None
    concealed: bool = False  # weapon on the body and not visible to the colour camera / cold patch
    concealment: str | None = None  # explanation when concealed
    # people
    armed_evidence: str | None = None  # "detector" | "zoom" | "specialist" | "classifier" | "tracked"
    track_id: int | None = None  # live stream / video: same person across frames
    armed_frames: str | None = None  # e.g. "3/6": armed in 3 of this person's last 6 frames
    armed_confirmed: bool = False  # armed in enough recent frames to be confirmed
    armed_possible: bool = False  # distant classifier finding that does not raise the threat level (verify)
    # estimated slant range from the camera (live stream with drone telemetry only): from the
    # person's apparent size, the camera's field of view and its look-down angle
    range_m: float | None = None


class ThreatSummary(BaseModel):
    level: ThreatLevel
    total_people: int
    armed_people: int
    weapon_count: int
    risk_score: int
    average_confidence: int
    possible_armed: int = 0  # distant people judged armed in one frame only (verify)


class FramePayload(BaseModel):
    frame_id: int
    timestamp: str
    source: str
    image: str
    annotated_image: str | None = None  # Image with bounding boxes drawn
    detections: list[Detection]
    summary: ThreatSummary
    method: str | None = None  # rgb | thermal_grey | thermal_inferno | rgbt
    distant_mode: str | None = None  # off | flag | alert
    method_label: str | None = None
    method_reasons: list[str] = []
    sensors: list[str] = []  # sensors that supplied an image
    environment: dict | None = None  # measured conditions and sensor reliabilities
    # Live stream: the same frame as each camera mode shows it, with the detections drawn on:
    # rgb | thermal_grey | thermal_inferno | rgbt (data URLs). Only modes whose sensor is present.
    views: dict[str, str] = {}
    telemetry: dict | None = None  # drone state sent by the capture system (altitude, heading, ...)
    processing_ms: int | None = None
    event_id: int | None = None  # set when this frame was written to the event log


class Acknowledgement(BaseModel):
    by: str = Field(default="operator", max_length=80)
    note: str = Field(default="", max_length=1000)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    detector_ready: bool
    video_source: str
    use_sahi: bool
    model: str | None = None
    detail: str | None = None
    stream: dict | None = None  # live stream state: frames received, last frame age, rate
    log_events: int | None = None


class MediaFrame(BaseModel):
    """One analysed image, or one sampled moment of a video."""
    time_s: float | None = None
    frame_index: int
    summary: ThreatSummary
    detections: list[Detection]
    annotated_image: str
    method: str | None = None
    method_label: str | None = None
    method_reasons: list[str] = []
    environment: dict | None = None


class TimelineEntry(BaseModel):
    time_s: float | None = None
    frame_index: int
    level: ThreatLevel
    total_people: int
    armed_people: int
    weapon_count: int
    confirmed: bool = True  # False: weapon seen in this sample only (possible false alarm)


class MediaAnalysis(BaseModel):
    """Result of analysing an uploaded or linked image/video."""
    kind: Literal["image", "video"]
    filename: str
    width: int
    height: int
    summary: ThreatSummary  # the most threatening moment
    frames_analyzed: int
    duration_s: float | None = None
    processing_ms: int
    key_frames: list[MediaFrame]
    timeline: list[TimelineEntry]
    unconfirmed_weapon_frames: int = 0
    sensors: list[str] = []  # "rgb" and/or "thermal"
    methods_used: dict[str, int] = {}  # method -> number of frames analysed with it


class UrlAnalysisRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2048)
    method: Literal["auto", "rgb", "thermal_grey", "thermal_inferno", "rgbt"] = "auto"

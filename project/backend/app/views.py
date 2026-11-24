"""Live camera views and range estimates for the dashboard.

One frame from the drone is shown as every camera mode sees it (RGB, thermal greyscale, thermal
INFERNO, RGBT fusion), each with the detections drawn on, and each person gets an estimated range."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .media import annotate_frame
from .modalities import grey_to_inferno
from .schemas import Detection
from .video import encode_frame

PERSON_HEIGHT_M = 1.75
PERSON_DEPTH_M = 0.5  # shoulder width / body depth: what a steep look-down view sees of a person


def fusion_image(rgb: np.ndarray, thermal_grey: np.ndarray) -> np.ndarray:
    """RGBT picture: the colour image with the thermal heat map laid over it, warm objects fully
    in thermal colours and the cooler scene mostly visible-light."""
    grey = thermal_grey if thermal_grey.ndim == 2 else cv2.cvtColor(thermal_grey, cv2.COLOR_BGR2GRAY)
    inferno = grey_to_inferno(grey).astype(np.float32)
    background = float(np.median(grey))
    heat = np.clip((grey.astype(np.float32) - background - 12) / 45.0, 0, 1)
    heat = cv2.GaussianBlur(heat, (0, 0), 1.2)[..., None]
    alpha = 0.3 + 0.7 * heat
    return (rgb.astype(np.float32) * (1 - alpha) + inferno * alpha).astype(np.uint8)


def build_views(rgb: np.ndarray | None, thermal_grey: np.ndarray | None, detections: list[Detection],
                width: int = 720, quality: int = 80) -> dict[str, str]:
    """{mode: JPEG data URL} for every camera mode the available sensors allow."""
    images: dict[str, np.ndarray] = {}
    if rgb is not None:
        images["rgb"] = rgb
    if thermal_grey is not None:
        images["thermal_grey"] = thermal_grey
        images["thermal_inferno"] = grey_to_inferno(thermal_grey)
    if rgb is not None and thermal_grey is not None:
        images["rgbt"] = fusion_image(rgb, thermal_grey)
    views = {}
    for name, image in images.items():
        height, image_width = image.shape[:2]
        if image_width > width:
            image = cv2.resize(image, (width, int(height * width / image_width)), interpolation=cv2.INTER_AREA)
        views[name] = encode_frame(annotate_frame(image, detections), quality=quality)
    return views


def estimate_ranges(detections: list[Detection], telemetry: dict | None) -> None:
    """Set range_m on each person: slant range estimated from apparent size.

    A person of height H seen at look-down angle a spans about H*cos(a) + D*sin(a) metres across
    the line of sight (D = body depth), so range = focal_px * that / pixel_height. Needs the
    camera's field of view and pitch from the drone telemetry; an estimate, good to roughly 20%
    for a standing adult."""
    if not telemetry or not telemetry.get("camera_fov_deg"):
        return
    width, height = telemetry.get("image_width"), telemetry.get("image_height")
    if not width or not height:
        return
    focal = (width / 2) / math.tan(math.radians(float(telemetry["camera_fov_deg"])) / 2)
    pitch = math.radians(-float(telemetry.get("camera_pitch_deg") or 0.0))  # look-down angle of the optical axis
    for d in detections:
        if d.kind != "person":
            continue
        pixel_h = d.bbox.height / 100 * height
        pixel_w = d.bbox.width / 100 * width
        if pixel_h < 4:
            continue
        centre_y = (d.bbox.y + d.bbox.height / 2) / 100 * height
        angle = min(max(pitch + math.atan((centre_y - height / 2) / focal), 0.0), math.pi / 2)
        extent = PERSON_HEIGHT_M * math.cos(angle) + PERSON_DEPTH_M * math.sin(angle)
        # a crouching or partly hidden person looks shorter: use the larger box side near-overhead
        pixels = max(pixel_h, pixel_w) if angle > math.radians(60) else pixel_h
        d.range_m = round(focal * extent / pixels, 1)

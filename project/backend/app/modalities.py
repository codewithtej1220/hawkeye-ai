"""Sensing methods (RGB, thermal greyscale, thermal INFERNO, RGBT fusion) and the environment-based
choice between them.

All four methods share one detector (person/weapon YOLO). They differ in which sensor image is fed
to it and how results are combined:

  rgb              visible-light camera image
  thermal_grey     thermal image, white-hot greyscale palette
  thermal_inferno  the same thermal data rendered with the INFERNO palette (preferred thermal view)
  rgbt             decision-level sensor fusion: detections from the RGB and the thermal view are
                   matched and merged, each weighted by how reliable that sensor is in the current
                   conditions (e.g. RGB counts for little at night, thermal for little when the scene
                   has no temperature contrast)

Decision-level fusion is used rather than a 4-channel network because no paired RGB+thermal data
with weapons exists to train one; fusing decisions keeps all weapon knowledge of the shared model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

METHODS = ("rgb", "thermal_grey", "thermal_inferno", "rgbt")
METHOD_LABELS = {
    "rgb": "RGB (visible light)",
    "thermal_grey": "Thermal greyscale",
    "thermal_inferno": "Thermal INFERNO",
    "rgbt": "RGBT sensor fusion",
}

# ----------------------------------------------------------------------------------------
# Palettes
# ----------------------------------------------------------------------------------------

_INFERNO_LUT = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_INFERNO)[:, 0]  # BGR
_Q = 32  # colour quantisation for the inverse lookup table


def _build_inverse_lut() -> tuple[np.ndarray, np.ndarray]:
    """For every quantised BGR colour: nearest INFERNO grey level and its distance."""
    levels = (np.arange(_Q) * (256 / _Q) + 256 / _Q / 2).astype(np.float32)
    b, g, r = np.meshgrid(levels, levels, levels, indexing="ij")
    colours = np.stack([b, g, r], axis=-1).reshape(-1, 1, 3)
    lut = _INFERNO_LUT.astype(np.float32).reshape(1, -1, 3)
    grey = np.empty(len(colours), np.uint8)
    dist = np.empty(len(colours), np.float32)
    for start in range(0, len(colours), 4096):  # chunked to bound memory
        d = ((colours[start:start + 4096] - lut) ** 2).sum(-1)
        grey[start:start + 4096] = d.argmin(1)
        dist[start:start + 4096] = np.sqrt(d.min(1))
    return grey.reshape(_Q, _Q, _Q), dist.reshape(_Q, _Q, _Q)


_INV_GREY, _INV_DIST = _build_inverse_lut()


def _quantised_index(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = (image >> 3).astype(np.intp)  # 256/32 = 8
    return q[..., 0], q[..., 1], q[..., 2]


def grey_to_inferno(image: np.ndarray) -> np.ndarray:
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(grey, cv2.COLORMAP_INFERNO)


def inferno_to_grey(image: np.ndarray) -> np.ndarray:
    """Recover the thermal intensity from an INFERNO-palette image (3-channel grey output)."""
    b, g, r = _quantised_index(image)
    grey = _INV_GREY[b, g, r]
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def to_grey3(image: np.ndarray) -> np.ndarray:
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def classify_image(image: np.ndarray) -> str:
    """'grey' (white-hot thermal or monochrome camera), 'inferno' (INFERNO-palette thermal) or 'rgb'."""
    small = cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA) if image.shape[1] > 160 else image
    pixels = small.reshape(-1, 3).astype(np.int16)
    chroma = (pixels.max(1) - pixels.min(1))
    if np.mean(chroma < 12) > 0.97:
        return "grey"
    b, g, r = _quantised_index(small)
    on_palette = _INV_DIST[b, g, r] < 28
    # Coloured pixels must also mostly lie on the INFERNO curve (dark purple -> orange -> yellow)
    if on_palette.mean() > 0.9:
        return "inferno"
    return "rgb"


# ----------------------------------------------------------------------------------------
# Environment assessment
# ----------------------------------------------------------------------------------------

@dataclass
class Environment:
    """Measured conditions for the available sensors. Values are rounded for display."""
    rgb_brightness: float | None = None  # mean luminance 0-255
    rgb_contrast: float | None = None  # luminance standard deviation
    rgb_haze: float | None = None  # 0-1, dark-channel haze estimate (fog/smoke)
    rgb_glare: float | None = None  # fraction of saturated pixels
    thermal_contrast: float | None = None  # 0.1-99.9 percentile spread of thermal intensity
    thermal_agc: bool = False  # low contrast: thermal image auto-gain stretched before detection
    rgb_reliability: float | None = None  # 0-1
    thermal_reliability: float | None = None  # 0-1
    conditions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


# Thermal contrast (0.1-99.9 percentile spread) below which automatic gain control is applied
AGC_BELOW = 60


# Measure thermal contrast so that small hot targets count (training/regression_check.py turns this
# off to compare with the previous behaviour)
POOL_HOT_TARGETS = True

# Largest gain automatic gain control may apply. Without a limit a flat scene (nothing warm in view)
# is stretched until sensor noise fills the picture, and the detector finds "people" in the noise.
AGC_MAX_GAIN = 4.0


def thermal_agc(grey: np.ndarray, low: float = 0.1, high: float = 99.9) -> np.ndarray:
    """Percentile contrast stretch of a thermal image (3-channel grey in, 3-channel grey out),
    limited to AGC_MAX_GAIN and centred on the scene's mid level."""
    g = grey if grey.ndim == 2 else cv2.cvtColor(grey, cv2.COLOR_BGR2GRAY)
    a, b = np.percentile(g, (low, high))
    if b - a < 1:
        return to_grey3(g)
    gain = min(255.0 / (b - a), AGC_MAX_GAIN)
    mid = (a + b) / 2
    return to_grey3(np.clip((g.astype(np.float32) - mid) * gain + 127.5, 0, 255).astype(np.uint8))


def _ramp(value: float, low: float, high: float) -> float:
    """0 at/below low, 1 at/above high, linear between."""
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def assess_environment(rgb: np.ndarray | None = None, thermal_grey: np.ndarray | None = None) -> Environment:
    env = Environment()
    if rgb is not None:
        small = cv2.resize(rgb, (320, int(320 * rgb.shape[0] / rgb.shape[1])), interpolation=cv2.INTER_AREA)
        lum = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        env.rgb_brightness = float(lum.mean())
        env.rgb_contrast = float(lum.std())
        dark_channel = cv2.erode(small.min(axis=2), np.ones((9, 9), np.uint8))
        env.rgb_haze = float(np.percentile(dark_channel, 50) / 255)
        env.rgb_glare = float((lum >= 250).mean())

        light = _ramp(env.rgb_brightness, 25, 70)
        detail = _ramp(env.rgb_contrast, 12, 35)
        clarity = 1 - _ramp(env.rgb_haze, 0.45, 0.75)
        # glare: blown-out highlights, or the whole image washed out (low sun, headlights into the lens)
        exposure = (1 - _ramp(env.rgb_glare, 0.15, 0.5)) * (1 - 0.6 * _ramp(env.rgb_brightness, 185, 235))
        env.rgb_reliability = round(light * detail * clarity * exposure, 3)
        if env.rgb_brightness < 40:
            env.conditions.append("low light / night")
        if env.rgb_haze > 0.5:
            env.conditions.append("fog or smoke")
        if env.rgb_glare > 0.15 or env.rgb_brightness > 185:
            env.conditions.append("glare / overexposure")

    if thermal_grey is not None:
        grey = thermal_grey if thermal_grey.ndim == 2 else cv2.cvtColor(thermal_grey, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(grey, (320, int(320 * grey.shape[0] / grey.shape[1])), interpolation=cv2.INTER_AREA)
        # 0.1-99.9 percentiles: small hot targets (people seen from a drone over uniform ground)
        # count as contrast. A 2-98 spread ignored them and misjudged such scenes as low-contrast.
        low = np.percentile(small, 0.1)
        # The hot end is measured on a max-pooled copy, so a distant person (a few hot pixels, which
        # plain down-scaling averages away) still counts as temperature contrast
        k = max(1, grey.shape[1] // 320)
        if POOL_HOT_TARGETS:
            hot = cv2.dilate(grey, np.ones((k, k), np.uint8))[::k, ::k] if k > 1 else grey
            high = np.percentile(hot, 99.95)
        else:
            high = np.percentile(small, 99.9)
        env.thermal_contrast = float(high - low)
        env.thermal_reliability = round(_ramp(env.thermal_contrast, 25, 80), 3)
        if env.thermal_contrast < AGC_BELOW:
            # Like a real thermal camera's automatic gain control: stretch the little contrast
            # there is. Measured on hot-day (crossover) scenes: +1-2 AP; it slightly hurts at
            # normal contrast, so it is only applied here. The stretched image regains part of
            # the lost reliability.
            env.thermal_agc = True
            env.thermal_reliability = round(max(env.thermal_reliability, 0.5), 3)
        if env.thermal_contrast < 55:
            env.conditions.append("low thermal contrast (thermal crossover)")

    if not env.conditions:
        env.conditions.append("normal")
    return env


# ----------------------------------------------------------------------------------------
# Method selection
# ----------------------------------------------------------------------------------------

# Below this reliability a sensor adds (almost) nothing to fusion, so it is skipped to halve the work.
# Chosen from evaluation on aligned RGB/thermal pairs (training/evaluate_modalities.py): with both
# sensors present, RGBT fusion was best or within 0.005 AP of the best in every environment tested
# (normal, dark, fog/smoke, glare, thermal crossover).
NEGLIGIBLE = 0.1


def select_method(has_rgb: bool, has_thermal: bool, env: Environment, preferred_thermal: str = "thermal_inferno",
                  ambiguous_grey: bool = False) -> tuple[str, list[str]]:
    """Choose the sensing method for these conditions. Returns (method, reasons).

    ambiguous_grey: the only image is greyscale of unknown origin (white-hot thermal or a monochrome
    visible camera); it is analysed as-is rather than re-coloured as thermal.
    """
    reasons: list[str] = []
    issues = ", ".join(c for c in env.conditions if c != "normal")
    if ambiguous_grey and not has_rgb:
        reasons.append("greyscale image (thermal white-hot or monochrome camera): analysed as-is")
        return "thermal_grey", reasons

    if has_rgb and has_thermal:
        r_rgb, r_t = env.rgb_reliability, env.thermal_reliability
        if r_rgb < NEGLIGIBLE <= r_t:
            reasons.append(f"visible image unusable ({issues or 'too dark'}; RGB reliability {r_rgb:.2f}): thermal only")
            return _thermal_method(env, preferred_thermal, reasons)
        if r_t < NEGLIGIBLE <= r_rgb:
            reasons.append(f"no usable temperature contrast (thermal reliability {r_t:.2f}): RGB only")
            return "rgb", reasons
        reasons.append(f"fusing both sensors, weighted by reliability (RGB {r_rgb:.2f}, thermal {r_t:.2f})"
                       + (f"; conditions: {issues}" if issues else ""))
        if env.thermal_agc:
            reasons.append("thermal auto-gain applied (low temperature contrast)")
        return "rgbt", reasons
    if has_thermal:
        reasons.append("only a thermal image is available")
        return _thermal_method(env, preferred_thermal, reasons)
    reasons.append("only a visible-light image is available")
    return "rgb", reasons


def _thermal_method(env: Environment, preferred: str, reasons: list[str]) -> tuple[str, list[str]]:
    if env.thermal_agc:
        # At low contrast the INFERNO palette squeezes the remaining detail into a narrow colour band;
        # measured: greyscale+AGC 0.844 vs INFERNO+AGC 0.827 AP on hot-day scenes
        reasons.append("low temperature contrast: auto-gain + greyscale palette (keeps more detail than INFERNO here)")
        return "thermal_grey", reasons
    return preferred, reasons


# ----------------------------------------------------------------------------------------
# Decision-level fusion
# ----------------------------------------------------------------------------------------

def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def fuse_detections(rgb_dets: list, thermal_dets: list, r_rgb: float, r_thermal: float,
                    match_iou: float = 0.4, with_meta: bool = False) -> list:
    """Merge detections from two aligned views.

    Inputs are (label, confidence, xyxy) tuples. A detection confirmed by both sensors gets
    1 - (1 - w1*p1)(1 - w2*p2), so agreement raises confidence. A detection seen by one sensor
    keeps its confidence scaled by that sensor's relative reliability - e.g. a person seen only
    in thermal at night keeps (almost) full confidence, because RGB is known to be blind there.
    Returns (label, confidence, xyxy, sources) with sources in {"rgb", "thermal", "rgb+thermal"}.
    with_meta: inputs carry a 4th element (e.g. how the detection was found), passed through as a
    5th output element (the thermal one when both sensors matched).
    """
    top = max(r_rgb, r_thermal, 1e-6)
    w_rgb, w_t = r_rgb / top, r_thermal / top  # the more reliable sensor has weight 1
    fused = []
    used = set()
    meta = (lambda d: d[3]) if with_meta else (lambda d: None)
    for det_t in sorted(thermal_dets, key=lambda d: -d[1]):
        label, p_t, box_t = det_t[:3]
        best, best_iou = None, match_iou
        for j, det_r in enumerate(rgb_dets):
            label_r, p_r, box_r = det_r[:3]
            if j in used or label_r != label:
                continue
            overlap = _iou(box_t, box_r)
            if overlap >= best_iou:
                best, best_iou = j, overlap
        if best is None:
            fused.append((label, p_t * w_t, box_t, "thermal", meta(det_t)))
            continue
        used.add(best)
        _, p_r, box_r = rgb_dets[best][:3]
        a, b = w_t * p_t, w_rgb * p_r
        box = tuple((a * bt + b * br) / (a + b) for bt, br in zip(box_t, box_r))
        fused.append((label, 1 - (1 - a) * (1 - b), box, "rgb+thermal", meta(det_t)))
    for j, det_r in enumerate(rgb_dets):
        if j not in used:
            label, p_r, box_r = det_r[:3]
            fused.append((label, p_r * w_rgb, box_r, "rgb", meta(det_r)))
    return fused if with_meta else [f[:4] for f in fused]

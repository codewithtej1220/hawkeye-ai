"""Analysis of user-supplied media (uploaded or fetched from a URL) and shared annotation."""

from __future__ import annotations

import ipaddress
import socket
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

from .config import Settings
from .detector import ThreatDetector
from .schemas import Detection, MediaAnalysis, MediaFrame, ThreatLevel, ThreatSummary
from .tracking import ArmedTracker
from .video import encode_frame

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv", ".mpg", ".mpeg"}
# Max movement (% of frame) of a weapon between consecutive samples for it to count as the same one
MAX_WEAPON_SHIFT = 15.0
LEVEL_RANK = {ThreatLevel.clear: 0, ThreatLevel.elevated: 1, ThreatLevel.high: 2, ThreatLevel.critical: 3}


class MediaError(ValueError):
    """Problem with the supplied media; reported to the user as a 400."""


# ----------------------------------------------------------------------------------------
# Annotation (shared with the live AirSim stream)
# ----------------------------------------------------------------------------------------

def annotate_frame(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw detections without hiding them: corner brackets instead of full boxes, a small padding
    so the outline sits beside the object rather than on it, and one small semi-transparent label
    per person outside the box (a weapon carried by a person is listed in that person's label).
    Colours: green = person, red = armed person / weapon."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    scale = max(width, height) / 960
    thickness = max(1, round(1.5 * scale))
    font_scale = max(0.38, 0.45 * scale)
    font_thickness = max(1, round(scale))
    labels: list[tuple[int, int, int, int, str, tuple[int, int, int]]] = []
    people = {d.id: d for d in detections if d.kind == "person"}

    for detection in detections:
        if detection.kind == "weapon" and detection.evidence == "classifier":
            continue  # judged from the whole person: shown on the person's own label
        box = detection.bbox
        x1, y1 = box.x * width / 100, box.y * height / 100
        x2, y2 = (box.x + box.width) * width / 100, (box.y + box.height) * height / 100
        pad = max(2.0, 0.03 * max(x2 - x1, y2 - y1))  # a slim margin: the outline sits just off the object
        x1, y1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
        x2, y2 = int(min(width - 1, x2 + pad)), int(min(height - 1, y2 + pad))

        if detection.kind == "person":
            color = (40, 40, 235) if detection.armed else (60, 200, 60)  # BGR: red / green
            if not detection.armed:
                text = "PERSON"
            elif detection.armed_evidence == "classifier":
                text = "ARMED (DISTANT)"
            else:
                text = "ARMED"
            weapons = [w for w in detections if w.kind == "weapon" and w.associated_person_id == detection.id
                       and w.evidence != "classifier"]
            text = f"{text} {round(detection.confidence * 100)}%"
            if weapons:
                best = max(weapons, key=lambda w: w.confidence)
                text += f" | {'CONCEALED ' if best.concealed else ''}WEAPON {round(best.confidence * 100)}%"
            if detection.range_m:
                text += f" | {detection.range_m:.0f} m"
        elif detection.kind == "weapon":
            color = (40, 40, 235)
            text = "" if detection.associated_person_id in people else \
                f"{'CONCEALED ' if detection.concealed else ''}WEAPON {round(detection.confidence * 100)}%"
        else:
            color, text = (0, 220, 220), f"{detection.label.upper()} {round(detection.confidence * 100)}%"

        # corner brackets: each side shows only its outer quarter, so the middle stays clear
        lx, ly = max(4, (x2 - x1) // 4), max(4, (y2 - y1) // 4)
        for (ax, ay), (dx, dy) in (((x1, y1), (1, 1)), ((x2, y1), (-1, 1)), ((x1, y2), (1, -1)), ((x2, y2), (-1, -1))):
            cv2.line(annotated, (ax, ay), (ax + dx * lx, ay), color, thickness, cv2.LINE_AA)
            cv2.line(annotated, (ax, ay), (ax, ay + dy * ly), color, thickness, cv2.LINE_AA)
        if text:
            labels.append((x1, y1, x2, y2, text, color))

    # labels last, on a translucent band above the box (below it when there is no room above)
    overlay = annotated.copy()
    placed = []
    for x1, y1, x2, y2, text, color in labels:
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        tx = min(max(0, x1), max(0, width - tw - 6))
        ty = y1 - 4 if y1 - th - base - 6 > 0 else min(height - base - 2, y2 + th + 6)
        cv2.rectangle(overlay, (tx, ty - th - 3), (tx + tw + 6, ty + base), (20, 20, 20), -1)
        cv2.rectangle(overlay, (tx, ty - th - 3), (tx + 2, ty + base), color, -1)
        placed.append((tx, ty, text))
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
    for tx, ty, text in placed:
        cv2.putText(annotated, text, (tx + 5, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                    font_thickness, cv2.LINE_AA)
    return annotated


# ----------------------------------------------------------------------------------------
# Loading media
# ----------------------------------------------------------------------------------------

def media_kind(filename: str, content_type: str | None) -> str:
    suffix = Path(filename).suffix.lower()
    content_type = (content_type or "").lower()
    if suffix in IMAGE_EXTS or content_type.startswith("image/"):
        return "image"
    if suffix in VIDEO_EXTS or content_type.startswith("video/"):
        return "video"
    raise MediaError(f"Unsupported file type '{suffix or content_type}'. Use an image "
                     f"({', '.join(sorted(IMAGE_EXTS))}) or a video ({', '.join(sorted(VIDEO_EXTS))}).")


def decode_image(data: bytes) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise MediaError("Could not read the image; the file may be corrupt or in an unsupported format.")
    return frame


def fetch_url(url: str, settings: Settings) -> tuple[str, str, Path]:
    """Download a direct image/video link to a temp file. Returns (kind, filename, path)."""
    import requests

    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise MediaError("Only http:// or https:// links to an image or video file are supported.")
    _reject_private_host(parsed.hostname)

    try:
        response = requests.get(url, stream=True, timeout=(10, 30), allow_redirects=True,
                                headers={"User-Agent": "HawkEye-AI/1.0"})
    except requests.RequestException as exc:
        raise MediaError(f"Could not download the link: {exc}") from exc
    if response.status_code != 200:
        raise MediaError(f"The link returned HTTP {response.status_code}.")
    _reject_private_host(urlparse(response.url).hostname or "")

    filename = Path(urlparse(response.url).path).name or "download"
    content_type = response.headers.get("content-type", "").split(";")[0]
    if content_type.startswith("text/html"):
        raise MediaError("That link is a web page, not a media file. Use a direct link to an image or video "
                         "(right-click the image -> 'Copy image address').")
    kind = media_kind(filename, content_type)
    path = _save_stream(response.iter_content(1 << 20), Path(filename).suffix or _ext_for(content_type),
                        settings.max_upload_mb)
    return kind, filename, path


def save_upload(chunks, suffix: str, settings: Settings) -> Path:
    return _save_stream(chunks, suffix, settings.max_upload_mb)


def _save_stream(chunks, suffix: str, max_mb: int) -> Path:
    limit = max_mb * 1024 * 1024
    total = 0
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".bin")
    try:
        with tmp:
            for chunk in chunks:
                total += len(chunk)
                if total > limit:
                    raise MediaError(f"File is larger than the {max_mb} MB limit.")
                tmp.write(chunk)
    except BaseException:
        Path(tmp.name).unlink(missing_ok=True)
        raise
    if total == 0:
        Path(tmp.name).unlink(missing_ok=True)
        raise MediaError("The file is empty.")
    return Path(tmp.name)


def _ext_for(content_type: str) -> str:
    return {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/bmp": ".bmp",
            "video/mp4": ".mp4", "video/webm": ".webm", "video/quicktime": ".mov",
            "video/x-msvideo": ".avi"}.get(content_type, "")


def _reject_private_host(hostname: str) -> None:
    """Block links into the local machine / private network (the backend would fetch them)."""
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(hostname, None)}
    except socket.gaierror as exc:
        raise MediaError(f"Could not resolve host '{hostname}'.") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise MediaError("Links to local or private network addresses are not allowed.")


# ----------------------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------------------

def analyze_image(frame: np.ndarray, filename: str, detector: ThreatDetector, method: str = "auto",
                  thermal: np.ndarray | None = None, distant_mode: str | None = None) -> MediaAnalysis:
    """frame: the image (or, when `thermal` is given, the visible-light image of an aligned pair)."""
    started = time.perf_counter()
    payload, display = _run(detector, frame, thermal, 0, method, distant_mode)
    # a single image has no history: weak evidence cannot be confirmed, so only seen weapons count
    payload.detections, payload.summary = detector.finalize(payload.detections, payload.distant_mode)
    key_frame = _media_frame(display, payload, time_s=None, frame_index=0)
    return MediaAnalysis(
        kind="image", filename=filename, width=display.shape[1], height=display.shape[0],
        summary=payload.summary, frames_analyzed=1, duration_s=None,
        processing_ms=round((time.perf_counter() - started) * 1000),
        key_frames=[key_frame], timeline=[_timeline_entry(key_frame)],
        sensors=payload.sensors, methods_used={payload.method: 1},
    )


def _run(detector: ThreatDetector, frame: np.ndarray, thermal: np.ndarray | None, index: int, method: str,
         distant_mode: str | None = None):
    try:
        return detector.analyze(frame, thermal, index, "upload", method=method, distant_mode=distant_mode)
    except ValueError as exc:  # e.g. RGBT requested for a single image
        raise MediaError(str(exc)) from exc


def analyze_video(path: Path, filename: str, detector: ThreatDetector, settings: Settings, method: str = "auto",
                  thermal_path: Path | None = None, distant_mode: str | None = None) -> MediaAnalysis:
    """path: the video (or, with `thermal_path`, the visible-light video of a synchronised pair)."""
    started = time.perf_counter()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise MediaError("Could not open the video; the file may be corrupt or use an unsupported codec.")
    thermal_capture = None
    if thermal_path is not None:
        thermal_capture = cv2.VideoCapture(str(thermal_path))
        if not thermal_capture.isOpened():
            capture.release()
            raise MediaError("Could not open the thermal video.")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0
        fps = fps if 0 < fps < 240 else 25.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total / fps if total > 0 else None

        # Sample at video_sample_fps, but never more than video_max_frames over the whole clip
        step = max(1, round(fps / settings.video_sample_fps))
        if total > 0:
            step = max(step, int(np.ceil(total / settings.video_max_frames)))

        frames: list[MediaFrame] = []
        sensors: list[str] = []
        tracker = ArmedTracker(settings.track_window, settings.track_confirm_hits)
        width = height = 0
        index = 0
        while len(frames) < settings.video_max_frames:
            if index % step == 0:
                ok, frame = capture.read()
                thermal = None
                if thermal_capture is not None:
                    ok_t, thermal = thermal_capture.read()
                    ok = ok and ok_t
                if not ok:
                    break
                payload, display = _run(detector, frame, thermal, index, method, distant_mode)
                payload.detections, payload.summary = detector.finalize(tracker.update(payload.detections),
                                                                        payload.distant_mode)
                height, width = display.shape[:2]
                sensors = payload.sensors
                frames.append(_media_frame(display, payload, time_s=round(index / fps, 2), frame_index=index))
            else:
                ok = capture.grab()
                if thermal_capture is not None:
                    ok = thermal_capture.grab() and ok
                if not ok:
                    break
            index += 1
    finally:
        capture.release()
        if thermal_capture is not None:
            thermal_capture.release()

    if not frames:
        raise MediaError("No frames could be read from the video.")

    # A weapon must persist: a neighbouring sample (~0.5 s either side) needs a weapon at nearly the
    # same place. One-frame detections are shown on the timeline as unconfirmed but don't set the
    # video's verdict - real weapons persist, most false alarms flicker.
    def weapon_centres(frame: MediaFrame) -> list[tuple[float, float]]:
        return [(d.bbox.x + d.bbox.width / 2, d.bbox.y + d.bbox.height / 2)
                for d in frame.detections if d.kind == "weapon"]

    def confirmed(i: int) -> bool:
        here = weapon_centres(frames[i])
        if not here:
            return True
        for j in (i - 1, i + 1):
            if 0 <= j < len(frames):
                for hx, hy in here:
                    if any(abs(hx - nx) <= MAX_WEAPON_SHIFT and abs(hy - ny) <= MAX_WEAPON_SHIFT
                           for nx, ny in weapon_centres(frames[j])):
                        return True
        return False

    flags = [confirmed(i) for i in range(len(frames))]
    trusted = [f for f, ok in zip(frames, flags) if ok] or frames
    # The verdict level must also persist: the highest level reached in at least two consecutive
    # samples, so a single flickering frame (e.g. a spurious second person) can't define the video
    rank = [LEVEL_RANK[f.summary.level] if ok else 0 for f, ok in zip(frames, flags)]
    sustained = max((min(rank[i], rank[i + 1]) for i in range(len(rank) - 1)), default=max(rank))
    worst = max((f for f, r in zip(frames, rank) if r == sustained and f in trusted), key=_threat_key,
                default=max(trusted, key=_threat_key))
    timeline = [_timeline_entry(f, ok) for f, ok in zip(frames, flags)]
    return MediaAnalysis(
        kind="video", filename=filename, width=width, height=height,
        summary=worst.summary, frames_analyzed=len(frames),
        duration_s=round(duration, 2) if duration else round(frames[-1].time_s or 0, 2),
        processing_ms=round((time.perf_counter() - started) * 1000),
        key_frames=_pick_key_frames(trusted, settings.video_key_frames),
        timeline=timeline,
        unconfirmed_weapon_frames=flags.count(False),
        sensors=sensors,
        methods_used={m: sum(f.method == m for f in frames) for m in {f.method for f in frames}},
    )


def _media_frame(frame: np.ndarray, payload, time_s: float | None, frame_index: int) -> MediaFrame:
    annotated = annotate_frame(frame, payload.detections)
    # Keep the response small: key frames are shown at dashboard size
    max_side = 1280
    scale = max_side / max(annotated.shape[:2])
    if scale < 1:
        annotated = cv2.resize(annotated, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return MediaFrame(time_s=time_s, frame_index=frame_index, summary=payload.summary,
                      detections=payload.detections, annotated_image=encode_frame(annotated, quality=82),
                      method=payload.method, method_label=payload.method_label,
                      method_reasons=payload.method_reasons, environment=payload.environment)


def _threat_key(frame: MediaFrame):
    s = frame.summary
    return (LEVEL_RANK[s.level], s.armed_people, s.weapon_count, s.total_people, s.average_confidence)


def _pick_key_frames(frames: list[MediaFrame], limit: int) -> list[MediaFrame]:
    """The most threatening moments, kept in time order; evenly spaced frames if nothing was found."""
    if all(f.summary.level == ThreatLevel.clear for f in frames):
        picks = [frames[round(i * (len(frames) - 1) / max(limit - 1, 1))] for i in range(min(limit, len(frames)))]
    else:
        picks = sorted(frames, key=_threat_key, reverse=True)[:limit]
    unique = {f.frame_index: f for f in picks}
    return sorted(unique.values(), key=lambda f: f.frame_index)


def _timeline_entry(frame: MediaFrame, confirmed: bool = True) -> dict:
    s: ThreatSummary = frame.summary
    return {"time_s": frame.time_s, "frame_index": frame.frame_index, "level": s.level,
            "total_people": s.total_people, "armed_people": s.armed_people, "weapon_count": s.weapon_count,
            "confirmed": confirmed}

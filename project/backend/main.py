import asyncio
import base64
import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.detector import DetectorBusy, DetectorNotReady, ThreatDetector
from app.eventlog import EventLog, LiveEventPolicy
from app.eventlog import describe as describe_summary
from app.media import (MediaError, analyze_image, analyze_video, annotate_frame, decode_image, fetch_url,
                       media_kind, save_upload)
from app.schemas import Acknowledgement, HealthResponse, MediaAnalysis, UrlAnalysisRequest
from app.tracking import ArmedTracker
from app.video import VideoStream, encode_frame
from app.views import build_views, estimate_ranges

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("hawkeye_backend")

settings = get_settings()
app = FastAPI(title=settings.app_name, version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=settings.cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = ThreatDetector(settings)
video_stream = VideoStream(settings)
# Live drone stream: accumulate "armed" evidence per person across frames
live_tracker = ArmedTracker(settings.track_window, settings.track_confirm_hits)
# Permanent record for later review: level changes, periodic updates while a threat persists, uploads
event_log = EventLog(settings.resolved_event_db_path, settings.resolved_event_snapshot_dir, settings.event_max)
live_policy = LiveEventPolicy(threat_update_s=settings.event_threat_update_s,
                              presence_update_s=settings.event_presence_update_s)


class StreamStats:
    """Live stream health for the dashboard: frames received, rate, age of the last frame."""

    def __init__(self) -> None:
        self.frames = 0
        self.skipped = 0
        self.last_at: float | None = None
        self._times: deque[float] = deque(maxlen=20)

    def frame(self) -> None:
        self.frames += 1
        self.last_at = time.time()
        self._times.append(time.monotonic())

    def as_dict(self) -> dict:
        age = None if self.last_at is None else round(time.time() - self.last_at, 1)
        live = age is not None and age < 5
        fps = 0.0
        if live and len(self._times) >= 2 and self._times[-1] > self._times[0]:
            fps = round((len(self._times) - 1) / (self._times[-1] - self._times[0]), 1)
        return {"frames": self.frames, "skipped": self.skipped, "fps": fps, "last_frame_age_s": age, "live": live}


stream_stats = StreamStats()

class StreamManager:
    def __init__(self) -> None:
        self.connections: set[WebSocket] = set()
        self.task: asyncio.Task | None = None
        self.last_message: dict[str, Any] | None = None
        self.stop_event = asyncio.Event()

    async def connect(self, websocket: WebSocket) -> None:
        self.connections.add(websocket)
        if self.last_message is not None:
            await websocket.send_json(self.last_message)

        if self.task is None or self.task.done():
            self.stop_event.clear()
            self.task = asyncio.create_task(self._run_stream())

    def disconnect(self, websocket: WebSocket) -> None:
        self.connections.discard(websocket)
        if not self.connections:
            self.stop_event.set()

    async def broadcast(self, message: dict[str, Any]) -> None:
        self.last_message = message
        if not self.connections:
            return

        disconnected: list[WebSocket] = []
        for websocket in list(self.connections):
            try:
                await websocket.send_json(message)
            except Exception as exc:
                logger.warning("Connection dropped while sending message: %s", exc)
                disconnected.append(websocket)

        for websocket in disconnected:
            self.connections.discard(websocket)

    async def broadcast_event(self, message: dict[str, Any]) -> None:
        """Send a non-frame message (e.g. a new log entry) without replacing the last frame."""
        last = self.last_message
        await self.broadcast(message)
        self.last_message = last

    async def _run_stream(self) -> None:
        # Don't try to read from video_stream when using airsim source
        # Images come via POST /detect endpoint instead
        while not self.stop_event.is_set():
            await asyncio.sleep(1)  # Just keep connection alive

stream_manager = StreamManager()

@app.on_event("startup")
async def startup_event() -> None:
    detector.load()
    if settings.video_source.lower() == "demo":
        logger.info("Using demo video source: no model load required.")
    elif not detector.ready:
        logger.warning("Detector failed to initialize: %s", detector.status_message)

@app.on_event("shutdown")
async def shutdown_event() -> None:
    video_stream.close()
    stream_manager.stop_event.set()
    if stream_manager.task is not None:
        await stream_manager.task

@app.get("/", include_in_schema=False)
def root():
    """Opening the backend's address in a browser shows its interactive API page, not a 404."""
    return RedirectResponse("/docs")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        detector_ready=detector.ready or settings.video_source.lower() == "demo",
        video_source=settings.video_source,
        use_sahi=settings.use_sahi,
        model=detector.model_name,
        detail=detector.status_message,
        stream=stream_stats.as_dict(),
        log_events=event_log.stats()["total"],
    )

@app.post("/detect")
async def detect_frame(data: dict):
    """Receive a frame from the AirSim capture system, run detection, broadcast to the dashboard.

    Body: {"image": <base64 thermal frame>, "rgb": <optional base64 visible-light frame of the same
    view>, "method": "auto" | "rgb" | "thermal_grey" | "thermal_inferno" | "rgbt", "frame_id": int,
    "telemetry": <optional drone state: altitude_m, heading_deg, camera_pitch_deg, camera_fov_deg, ...>}
    """
    frame_id = int(data.get("frame_id", 0) or 0)
    method = str(data.get("method") or "auto")
    telemetry = data.get("telemetry") if isinstance(data.get("telemetry"), dict) else None
    started = time.perf_counter()
    try:
        thermal = decode_image(base64.b64decode(data["image"]))
        rgb = decode_image(base64.b64decode(data["rgb"])) if data.get("rgb") else None
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid frame: {exc}") from exc

    # The small-person search is slow (~0.6 s), so the live stream runs it on every Nth frame;
    # the tracker keeps people found by it between those frames
    distant_search = frame_id % max(1, settings.distant_search_live_every) == 0
    try:
        # Inference takes ~0.2 s on the GPU; run it off the event loop so the WebSocket stays responsive.
        # The capture system's "image" is always its thermal camera.
        payload, display, sources = await run_in_threadpool(lambda: detector.process_views(
            rgb, thermal, frame_id, "airsim", wait=False, method=method, distant_search=distant_search,
            return_sources=True))
        payload.detections, payload.summary = detector.finalize(live_tracker.update(payload.detections),
                                                                payload.distant_mode, drone=True)
        if telemetry is not None:
            telemetry = {**telemetry, "image_width": display.shape[1], "image_height": display.shape[0]}
            estimate_ranges(payload.detections, telemetry)
            payload.telemetry = telemetry
        # Each analysed frame is shown with its own camera views (RGB / thermal grey / INFERNO / RGBT);
        # the dashboard draws this frame's boxes on them, so boxes and picture always match
        payload.views = await run_in_threadpool(build_views, sources["rgb"], sources["thermal_grey"],
                                                [], settings.live_view_width)
    except DetectorBusy:
        # Live video: drop this frame rather than queue it behind an upload analysis
        stream_stats.skipped += 1
        return {"status": "skipped", "frame_id": frame_id, "detections": 0}
    except DetectorNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error processing frame from AirSim")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    stream_stats.frame()
    payload.processing_ms = int((time.perf_counter() - started) * 1000)
    event_type, previous_level = live_policy.decide(payload.summary.level.value)
    event = None
    if event_type is not None:
        annotated = await run_in_threadpool(annotate_frame, display, payload.detections)
        event = await run_in_threadpool(lambda: event_log.add(
            source="live", event_type=event_type, summary=payload.summary, detections=payload.detections,
            snapshot=annotated, previous_level=previous_level, method=payload.method,
            method_label=payload.method_label, conditions=(payload.environment or {}).get("conditions"),
            telemetry=telemetry))
        payload.event_id = event["id"]

    message = payload.model_dump(mode="json")
    message["stream"] = stream_stats.as_dict()
    await stream_manager.broadcast({"type": "frame", "payload": message})
    if event is not None:
        await stream_manager.broadcast_event({"type": "log", "payload": event})

    if settings.live_record_dir and frame_id % max(1, settings.live_record_every) == 0:
        record = Path(settings.live_record_dir)
        record.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(record / f"{frame_id:06d}_thermal.png"), thermal)
        if rgb is not None:
            cv2.imwrite(str(record / f"{frame_id:06d}_rgb.png"), rgb)
        (record / f"{frame_id:06d}.json").write_text(json.dumps(
            [{"label": d.label, "confidence": d.confidence, "kind": d.kind} for d in payload.detections]))

    detection_info = ", ".join(f"{d.kind}:{d.label}" for d in payload.detections)
    logger.info("Frame %s [%s]: %s detections (%s), threat=%s, people=%s, weapons=%s", frame_id, payload.method,
                len(payload.detections), detection_info, payload.summary.level.value,
                payload.summary.total_people, payload.summary.weapon_count)

    # people (boxes in % of the frame) let an autonomous drone turn toward and approach a sighting
    people = [{"x": d.bbox.x, "y": d.bbox.y, "width": d.bbox.width, "height": d.bbox.height,
               "confidence": d.confidence, "range_m": d.range_m, "armed": bool(d.armed)}
              for d in payload.detections if d.kind == "person"]
    return {"status": "ok", "frame_id": frame_id, "detections": len(payload.detections),
            "threat": payload.summary.level.value, "people": people}

@app.post("/analyze/upload", response_model=MediaAnalysis)
async def analyze_upload(file: UploadFile = File(...), thermal: UploadFile | None = File(None),
                         method: str = Form("auto"), distant: str | None = Form(None)) -> MediaAnalysis:
    """Analyse an uploaded image or video for people, weapons and threat level.

    file: the image/video (thermal or visible light - recognised automatically).
    thermal: optional thermal image/video of the same scene; `file` is then the visible-light one,
             enabling RGBT fusion. method: auto | rgb | thermal_grey | thermal_inferno | rgbt.
    """
    filename = Path(file.filename or "upload").name
    paths: list[Path] = []
    try:
        kind = media_kind(filename, file.content_type)
        # Stream in 1 MB chunks so the size limit applies before the whole file is held in memory
        chunks = iter(lambda: file.file.read(1 << 20), b"")
        paths.append(await run_in_threadpool(save_upload, chunks, Path(filename).suffix, settings))
        if thermal is not None and thermal.filename:
            thermal_name = Path(thermal.filename).name
            if media_kind(thermal_name, thermal.content_type) != kind:
                raise MediaError("The visible-light and thermal files must both be images or both be videos.")
            chunks = iter(lambda: thermal.file.read(1 << 20), b"")
            paths.append(await run_in_threadpool(save_upload, chunks, Path(thermal_name).suffix, settings))
            filename = f"{filename} + {thermal_name}"
    except MediaError as exc:
        for p in paths:
            p.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return await _analyze_file(kind, filename, paths[0], method, paths[1] if len(paths) > 1 else None, distant)

@app.post("/analyze/url", response_model=MediaAnalysis)
async def analyze_url(request: UrlAnalysisRequest) -> MediaAnalysis:
    """Download a direct image/video link and analyse it."""
    try:
        kind, filename, path = await run_in_threadpool(fetch_url, request.url, settings)
    except MediaError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return await _analyze_file(kind, filename, path, request.method)

async def _analyze_file(kind: str, filename: str, path: Path, method: str = "auto",
                        thermal_path: Path | None = None, distant: str | None = None) -> MediaAnalysis:
    if not detector.ready:
        path.unlink(missing_ok=True)
        if thermal_path:
            thermal_path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail=f"Detector not ready: {detector.status_message}")
    try:
        if kind == "image":
            frame = await run_in_threadpool(decode_image, path.read_bytes())
            thermal = await run_in_threadpool(decode_image, thermal_path.read_bytes()) if thermal_path else None
            result = await run_in_threadpool(analyze_image, frame, filename, detector, method, thermal, distant)
        else:
            result = await run_in_threadpool(analyze_video, path, filename, detector, settings, method, thermal_path,
                                             distant)
    except MediaError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error analysing %s", filename)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
    finally:
        path.unlink(missing_ok=True)
        if thermal_path:
            thermal_path.unlink(missing_ok=True)

    try:
        best = result.key_frames[0] if result.key_frames else None
        snapshot = decode_image(base64.b64decode(best.annotated_image.split(",", 1)[-1])) if best else None
        event = event_log.add(source="upload", event_type="analysis", summary=result.summary,
                              detections=best.detections if best else [], snapshot=snapshot,
                              method=best.method if best else None, method_label=best.method_label if best else None,
                              conditions=(best.environment or {}).get("conditions") if best else None,
                              description=f'{kind.capitalize()} "{filename}": {describe_summary(result.summary)}')
        await stream_manager.broadcast_event({"type": "log", "payload": event})
    except Exception:
        logger.exception("Could not write the analysis of %s to the event log", filename)

    logger.info("Analysed %s %s %s: %s frame(s), threat=%s, people=%s, armed=%s, weapons=%s, %s ms",
                kind, filename, result.methods_used, result.frames_analyzed, result.summary.level.value,
                result.summary.total_people, result.summary.armed_people, result.summary.weapon_count,
                result.processing_ms)
    return result

@app.get("/logs")
def list_logs(level: str | None = None, min_level: str | None = None, source: str | None = None,
              since: str | None = None, until: str | None = None, search: str | None = None,
              unacknowledged: bool = False, limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    """Event log, newest first. since/until are ISO timestamps (UTC)."""
    return event_log.query(level=level, min_level=min_level, source=source, since=since, until=until,
                           search=search, unacknowledged=unacknowledged, limit=limit, offset=offset)


@app.get("/logs/stats")
def log_stats():
    return event_log.stats()


@app.get("/logs/export.csv")
def export_logs(level: str | None = None, min_level: str | None = None, source: str | None = None,
                since: str | None = None, until: str | None = None, search: str | None = None,
                unacknowledged: bool = False):
    content = event_log.export_csv(level=level, min_level=min_level, source=source, since=since, until=until,
                                   search=search, unacknowledged=unacknowledged)
    name = f"hawkeye_events_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(content, media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/logs/{event_id}")
def get_log(event_id: int):
    event = event_log.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="No such event.")
    return event


@app.get("/logs/{event_id}/snapshot")
def get_log_snapshot(event_id: int):
    path = event_log.snapshot_path(event_id)
    if path is None:
        raise HTTPException(status_code=404, detail="No snapshot for this event.")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/logs/{event_id}/acknowledge")
async def acknowledge_log(event_id: int, body: Acknowledgement):
    """Record that a reviewer has seen an event (who + note). The event itself is never changed."""
    event = event_log.acknowledge(event_id, body.by, body.note)
    if event is None:
        raise HTTPException(status_code=404, detail="No such event.")
    await stream_manager.broadcast_event({"type": "log_update", "payload": event})
    return event


@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    await stream_manager.connect(websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        stream_manager.disconnect(websocket)
    except Exception:
        stream_manager.disconnect(websocket)
        raise

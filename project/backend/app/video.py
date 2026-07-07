from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import cv2
import numpy as np

from .config import BASE_DIR, Settings


class VideoStream:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.source = settings.video_source
        self._capture: cv2.VideoCapture | None = None

    async def read(self, frame_id: int) -> np.ndarray:
        if self.source.lower() == "demo":
            await asyncio.sleep(self.settings.frame_interval_ms / 1000)
            return _demo_frame(frame_id)

        if self._capture is None or not self._capture.isOpened():
            self._open_capture()

        ok, frame = self._capture.read()
        if not ok:
            self.close()
            await asyncio.sleep(1)
            self._open_capture()
            ok, frame = self._capture.read()

        if not ok:
            raise RuntimeError(f"Unable to read frame from video source: {self.source}")

        await asyncio.sleep(self.settings.frame_interval_ms / 1000)
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _open_capture(self) -> None:
        source = _resolve_source(self.source)
        self._capture = cv2.VideoCapture(source)
        if not self._capture.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.source}")


def encode_frame(frame: np.ndarray, quality: int = 82) -> str:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Unable to encode frame.")

    return f"data:image/jpeg;base64,{base64.b64encode(encoded).decode('ascii')}"


def _resolve_source(source: str) -> int | str:
    if source.isdigit():
        return int(source)

    source_path = Path(source)
    if source_path.exists() or source.startswith(("rtsp://", "rtmp://", "http://", "https://")):
        return source

    return str(BASE_DIR / source)


def _demo_frame(frame_id: int) -> np.ndarray:
    height, width = 720, 960
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    gradient = np.linspace(30, 145, width, dtype=np.uint8)
    frame[:, :, 0] = gradient
    frame[:, :, 1] = np.flip(gradient)
    frame[:, :, 2] = 60

    cv2.putText(frame, "Demo thermal feed", (32, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2)
    cv2.putText(frame, f"Frame {frame_id}", (32, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210, 210, 210), 2)

    scenario = frame_id % 3
    _draw_person(frame, (475, 405), armed=False)

    if scenario >= 1:
        _draw_person(frame, (365, 350), armed=True)
        cv2.line(frame, (374, 318), (430, 342), (20, 20, 20), 8)

    if scenario == 2:
        _draw_person(frame, (575, 382), armed=True)
        _draw_person(frame, (202, 255), armed=True, scale=0.65)
        cv2.line(frame, (575, 330), (655, 360), (20, 20, 20), 8)
        cv2.line(frame, (212, 240), (250, 260), (20, 20, 20), 6)
        cv2.line(frame, (595, 375), (630, 395), (20, 20, 20), 5)

    return frame


def _draw_person(frame: np.ndarray, center: tuple[int, int], armed: bool, scale: float = 1) -> None:
    x, y = center
    color = (245, 245, 245) if not armed else (235, 235, 235)
    radius = int(24 * scale)
    body_width = int(44 * scale)
    body_height = int(120 * scale)

    cv2.circle(frame, (x, y - body_height // 2), radius, color, -1)
    cv2.ellipse(frame, (x, y + 8), (body_width, body_height // 2), 0, 0, 360, color, -1)
    cv2.line(frame, (x - body_width, y + 12), (x - body_width - int(22 * scale), y + int(70 * scale)), color, 10)
    cv2.line(frame, (x + body_width, y + 12), (x + body_width + int(22 * scale), y + int(70 * scale)), color, 10)

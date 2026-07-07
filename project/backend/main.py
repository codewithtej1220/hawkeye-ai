import asyncio
import logging
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.detector import ThreatDetector
from app.schemas import FramePayload, HealthResponse
from app.video import VideoStream

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("hawkeye_backend")

settings = get_settings()
app = FastAPI(title=settings.app_name, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = ThreatDetector(settings)
video_stream = VideoStream(settings)

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
        for websocket in self.connections:
            try:
                await websocket.send_json(message)
            except Exception as exc:
                logger.warning("Connection dropped while sending message: %s", exc)
                disconnected.append(websocket)

        for websocket in disconnected:
            self.connections.discard(websocket)

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

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        detector_ready=detector.ready or settings.video_source.lower() == "demo",
        video_source=settings.video_source,
        use_sahi=settings.use_sahi,
    )

@app.post("/detect")
async def detect_frame(data: dict):
    """Receive thermal image from AirSim capture system and run AI detection"""
    try:
        import base64
        import numpy as np
        import cv2
        
        # Decode base64 image
        img_data = base64.b64decode(data["image"])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process frame with AI detector
        frame_id = data.get("frame_id", 0)
        payload = detector.process_frame(frame, frame_id, "airsim")
        
        # Keep original image for left panel
        _, buffer_original = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        original_base64 = base64.b64encode(buffer_original).decode('utf-8')
        payload.image = f"data:image/jpeg;base64,{original_base64}"
        
        # Draw bounding boxes on a copy for right panel (matching model's style)
        annotated_frame = frame.copy()
        for detection in payload.detections:
            # Calculate pixel coordinates from percentage
            h, w = annotated_frame.shape[:2]
            x1 = int(detection.bbox.x * w / 100)
            y1 = int(detection.bbox.y * h / 100)
            x2 = int((detection.bbox.x + detection.bbox.width) * w / 100)
            y2 = int((detection.bbox.y + detection.bbox.height) * h / 100)
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            
            # Use model's actual style from reference images
            if detection.kind == "person":
                box_color = (0, 255, 0)    # Green (BGR)
                text_color = (255, 255, 255)  # White text for better visibility
                bg_color = (0, 255, 0)     # Green background
                label = "HUMAN"
            elif detection.kind == "weapon":
                box_color = (0, 0, 255)    # Red (BGR)
                text_color = (255, 255, 255)  # White text
                bg_color = (0, 0, 255)     # Red background
                label = "WEAPON"
            else:
                box_color = (0, 255, 255)  # Yellow
                text_color = (255, 255, 255)
                bg_color = (0, 255, 255)
                label = detection.label.upper()
            
            # Draw rectangle with 2px thickness
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw label with background for better visibility
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Position text at top of box
            text_x = x1 + 3
            text_y = y1 - 5 if y1 > 25 else y1 + text_h + 5
            
            # Draw background rectangle for text
            cv2.rectangle(annotated_frame, 
                         (text_x - 2, text_y - text_h - 2), 
                         (text_x + text_w + 2, text_y + baseline + 2), 
                         bg_color, -1)
            
            # Draw text
            cv2.putText(annotated_frame, label, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
        
        # Encode annotated image for right panel
        _, buffer_annotated = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_base64 = base64.b64encode(buffer_annotated).decode('utf-8')
        payload.annotated_image = f"data:image/jpeg;base64,{annotated_base64}"
        
        # Broadcast to all connected WebSocket clients
        await stream_manager.broadcast({"type": "frame", "payload": payload.dict()})
        
        # Detailed logging
        detection_info = ", ".join([f"{d.kind}:{d.label}" for d in payload.detections])
        logger.info(f"Frame {frame_id}: {len(payload.detections)} detections ({detection_info}), threat={payload.summary.level}, people={payload.summary.total_people}, weapons={payload.summary.weapon_count}")
        
        return {"status": "ok", "frame_id": frame_id, "detections": len(payload.detections)}
    except Exception as e:
        logger.exception("Error processing frame from AirSim")
        return {"status": "error", "message": str(e)}

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

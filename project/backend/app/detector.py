from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from .config import Settings
from .modalities import (METHOD_LABELS, METHODS, assess_environment, classify_image, fuse_detections,
                         grey_to_inferno, inferno_to_grey, select_method, thermal_agc, to_grey3)
from .schemas import BoundingBox, Detection, FramePayload, ThreatLevel, ThreatSummary
from .video import encode_frame

# Overlap (IoU) at which the two ensemble models' boxes count as the same detection. Weapons get
# their own value: a rifle's box is long and thin, so two correct boxes overlap far less than two
# person boxes do (chosen with training/weapon_threshold_sweep.py).
# Weaker weapon evidence (distant armed-person classifier, thermal-rifle specialist): never shown on
# its own; it counts only once the person is confirmed armed over several frames (see finalize)
UNCONFIRMED_EVIDENCE = ("classifier", "specialist")

PERSON_MATCH_IOU = 0.55
WEAPON_MATCH_IOU = 0.55


@dataclass
class RawDetection:
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    sensor: str | None = None  # "rgb", "thermal" or "rgb+thermal"
    evidence: str = "detector"  # "detector" | "zoom" | "specialist" | "classifier"


class DetectorNotReady(RuntimeError):
    pass


class DetectorBusy(RuntimeError):
    """Raised for live frames when the model is busy (e.g. analysing an upload)."""


class ThreatDetector:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ready = False
        self.status_message = "Detector has not been loaded."
        self._detection_model: Any | None = None
        self._yolo_model: Any | None = None
        self._models: list[Any] = []  # [primary] or [primary, ensemble partner]
        self._armed_model: Any | None = None  # distant/concealed weapon classifier
        self._specialists: list[Any] = []  # thermal-rifle specialists (zoom pass, second opinion)
        self._zoom_model: Any | None = None
        self.model_name: str | None = None
        # The live stream and dashboard uploads share one GPU model; run one frame at a time.
        # Live frames take priority: uploads pause between their frames while a live frame waits.
        self._lock = threading.Lock()
        self._live_waiting = threading.Event()

    def load(self) -> None:
        if self.settings.video_source.lower() == "demo":
            self.status_message = "Demo source enabled; using deterministic demo detections."
            return

        model_path = Path(self.settings.resolved_model_path)
        if not model_path.exists():
            self.status_message = f"Model file not found: {model_path}"
            return

        try:
            from ultralytics import YOLO

            if self.settings.use_sahi:
                from sahi import AutoDetectionModel

                self._detection_model = AutoDetectionModel.from_pretrained(
                    model_type=self.settings.model_type,
                    model_path=str(model_path),
                    confidence_threshold=self.settings.confidence_threshold,
                    device=self.settings.device,
                )
            else:
                self._yolo_model = YOLO(str(model_path))
            self._models = [self._yolo_model or YOLO(str(model_path))]
            partner = self.settings.resolved_ensemble_path
            if partner and Path(partner).resolve() != model_path.resolve():
                self._models.append(YOLO(partner))

            if self.settings.zoom_weapon_pass:
                self._zoom_model = self._models[0]
            armed_path = self.settings.resolved_armed_classifier_path
            if armed_path:
                self._armed_model = YOLO(armed_path)
            self._specialists = [YOLO(path) for path in self.settings.resolved_specialist_paths]

            self.ready = True
            self.model_name = " + ".join([model_path.name] + ([Path(partner).name] if len(self._models) > 1 else [])
                                         + ([Path(armed_path).name] if self._armed_model is not None else []))
            self.status_message = f"Detector ready ({self.model_name})."
            print(f"[DETECTOR] Loaded {self.model_name}")
        except Exception as exc:
            self.ready = False
            self.status_message = f"Detector failed to load: {exc}"

    def process_frame(self, frame: np.ndarray, frame_id: int, source: str, wait: bool = True,
                      method: str = "auto") -> FramePayload:
        """Detect people/weapons in one image. Thermal images (greyscale or INFERNO palette) are
        recognised automatically; anything else is treated as visible light."""
        if source == "demo":
            detections = self._classify(self._demo_detections(frame, frame_id), frame.shape[1], frame.shape[0])
            return FramePayload(frame_id=frame_id, timestamp=datetime.now(timezone.utc).isoformat(), source=source,
                                image=encode_frame(frame, quality=self.settings.jpeg_quality),
                                detections=detections, summary=self._summarize(detections))
        return self.analyze(frame, None, frame_id, source, wait, method)[0]

    def analyze(self, primary: np.ndarray, thermal: np.ndarray | None, frame_id: int, source: str,
                wait: bool = True, method: str = "auto", distant_mode: str | None = None):
        """Entry point for uploads. With a separate thermal image, `primary` is the visible-light
        image; otherwise `primary` may be either kind and is recognised automatically.
        Returns (payload, analysed image)."""
        if thermal is not None:
            return self.process_views(primary, thermal, frame_id, source, wait, method, return_display=True,
                                      distant_mode=distant_mode)
        kind = classify_image(primary)
        if kind == "inferno":
            return self.process_views(None, primary, frame_id, source, wait, method, return_display=True,
                                      distant_mode=distant_mode)
        if kind == "grey":
            # White-hot thermal or a monochrome visible camera: can't tell, so don't re-colour it
            return self.process_views(None, primary, frame_id, source, wait, method, return_display=True,
                                      thermal_known=False, distant_mode=distant_mode)
        return self.process_views(primary, None, frame_id, source, wait, method, return_display=True,
                                  distant_mode=distant_mode)

    def process_views(self, rgb: np.ndarray | None, thermal: np.ndarray | None, frame_id: int, source: str,
                      wait: bool = True, method: str = "auto", return_display: bool = False,
                      thermal_known: bool = True, distant_search: bool | None = None,
                      distant_mode: str | None = None, return_sources: bool = False):
        """Detect with the best sensing method for the conditions (or the one requested).

        rgb: visible-light BGR image. thermal: thermal image in greyscale or INFERNO palette,
        aligned with rgb when both are given. thermal_known=False marks a lone greyscale image that
        may not be thermal at all. With wait=False a busy model raises DetectorBusy instead of
        queueing, so a live stream drops stale frames rather than falling behind.
        """
        if rgb is None and thermal is None:
            raise ValueError("No image supplied.")
        distant_mode = distant_mode or self.settings.distant_mode
        if distant_mode not in ("off", "flag", "alert"):
            raise ValueError("distant_mode must be off, flag or alert.")
        thermal_grey = None
        if thermal is not None:
            thermal_grey = inferno_to_grey(thermal) if classify_image(thermal) == "inferno" else to_grey3(thermal)
            if rgb is not None and rgb.shape[:2] != thermal_grey.shape[:2]:
                # Different camera resolutions: assume the same field of view and align sizes
                rgb = cv2.resize(rgb, (thermal_grey.shape[1], thermal_grey.shape[0]), interpolation=cv2.INTER_AREA)

        env = assess_environment(rgb, thermal_grey)
        if env.thermal_agc:
            thermal_grey = thermal_agc(thermal_grey)
        if method == "auto":
            method, reasons = select_method(rgb is not None, thermal_grey is not None, env,
                                            ambiguous_grey=not thermal_known)
        else:
            if method not in METHODS:
                raise ValueError(f"Unknown method '{method}'. Use one of: auto, {', '.join(METHODS)}.")
            if method in ("rgb", "rgbt") and rgb is None:
                raise ValueError(f"{METHOD_LABELS[method]} needs a visible-light (RGB) image.")
            if method != "rgb" and thermal_grey is None:
                raise ValueError(f"{METHOD_LABELS[method]} needs a thermal image.")
            reasons = [f"{METHOD_LABELS[method]} requested"]

        views: dict[str, np.ndarray] = {}
        if method in ("rgb", "rgbt"):
            views["rgb"] = rgb
        if method == "thermal_grey":
            views["thermal"] = thermal_grey
        elif method in ("thermal_inferno", "rgbt"):
            views["thermal"] = grey_to_inferno(thermal_grey)

        self._acquire(wait)
        try:
            # distant_search=False lets the live stream skip the slow person search on some frames
            distant = distant_mode != "off"
            search = distant and (distant_search is None or distant_search)
            searched = set(views) if search else set()
            if search and not wait and len(views) > 1:
                # Live RGBT: the tile search (~0.8 s per view) runs on the more reliable sensor only,
                # so a live frame stays near 1 s; uploads (wait=True) search both views
                searched = {"rgb" if (env.rgb_reliability or 0) > (env.thermal_reliability or 0) else "thermal"}
            try:
                per_view = {name: self._detect_view(image, name in searched, distant, name == "thermal")
                            for name, image in views.items()}
            except RuntimeError as exc:
                # The GPU is shared with Unreal: a momentary CUDA out-of-memory is retried once after
                # freeing the cache, then the frame is skipped instead of failing the stream
                if "out of memory" not in str(exc).lower():
                    raise
                import torch
                torch.cuda.empty_cache()
                try:
                    per_view = {name: self._detect_view(image, name in searched, distant, name == "thermal")
                                for name, image in views.items()}
                except RuntimeError as retry_exc:
                    if "out of memory" not in str(retry_exc).lower():
                        raise
                    torch.cuda.empty_cache()
                    raise DetectorBusy("GPU out of memory; frame skipped.") from retry_exc
        finally:
            self._lock.release()

        if method == "rgbt":
            fused = fuse_detections(
                [(d.label, d.confidence, d.xyxy, d.evidence) for d in per_view["rgb"]],
                [(d.label, d.confidence, d.xyxy, d.evidence) for d in per_view["thermal"]],
                env.rgb_reliability, env.thermal_reliability, with_meta=True)
            weapon_labels = self.settings.weapon_label_set
            raw_detections = [
                RawDetection(label, conf, box, sensor, evidence)
                for label, conf, box, sensor, evidence in fused
                if conf >= (self.settings.weapon_confidence_threshold if label.lower() in weapon_labels
                            else self.settings.confidence_threshold)
            ]
        else:
            sensor = "rgb" if method == "rgb" else "thermal"
            raw_detections = [RawDetection(d.label, d.confidence, d.xyxy, sensor, d.evidence)
                              for d in next(iter(per_view.values()))]

        # Shown to the operator: the thermal view (INFERNO when that was used) or the RGB image
        display = views.get("thermal", views.get("rgb"))
        detections = self._classify(raw_detections, display.shape[1], display.shape[0])
        _mark_concealed(detections, display, method, env.rgb_reliability)
        payload = FramePayload(
            frame_id=frame_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=source,
            image=encode_frame(display, quality=self.settings.jpeg_quality),
            detections=detections,
            summary=self._summarize(detections, distant_mode),
            method=method,
            distant_mode=distant_mode,
            method_label=METHOD_LABELS[method],
            method_reasons=reasons,
            sensors=[name for name, image in (("rgb", rgb), ("thermal", thermal_grey)) if image is not None],
            environment=env.as_dict(),
        )
        if return_sources:
            # the aligned sensor images, for the per-camera live views
            return payload, display, {"rgb": rgb, "thermal_grey": thermal_grey}
        return (payload, display) if return_display else payload

    def _acquire(self, wait: bool) -> None:
        if wait:
            deadline = time.monotonic() + 1.0
            while self._live_waiting.is_set() and time.monotonic() < deadline:
                time.sleep(0.005)
            self._lock.acquire()
            return
        self._live_waiting.set()
        try:
            acquired = self._lock.acquire(timeout=0.3)
        finally:
            self._live_waiting.clear()
        if not acquired:
            raise DetectorBusy("Detector busy; frame skipped.")

    def _detect_view(self, frame: np.ndarray, distant_search: bool = False,
                     distant_classifier: bool = True, thermal_view: bool = False) -> list[RawDetection]:
        """Full single-image pipeline (caller holds the lock).

        1. full-frame detection   2. small-person search (tiles; optional)   3. zoom on each person for
        weapons too small for the full frame   4. armed-person classifier for distant people with no
        weapon found yet."""
        raw_detections = self._predict(frame)
        # Weapons need more evidence than people: a false "armed" alert is costly
        weapon_labels = self.settings.weapon_label_set
        raw_detections = [d for d in raw_detections if d.label.lower() not in weapon_labels
                          or d.confidence >= self.settings.weapon_confidence_threshold]
        # Drop body-part fragments (e.g. a "person" box on just the legs) inside a stronger box
        raw_detections = _suppress_contained(raw_detections)
        raw_detections = _drop_frame_filling_people(raw_detections, frame.shape, self.settings.person_label_set)
        if distant_search:
            raw_detections += self._find_small_people(frame, raw_detections)
        if self._zoom_model is not None:
            raw_detections += self._zoom_weapons(frame, raw_detections, thermal_view)
        if self._armed_model is not None and distant_classifier:
            raw_detections += self._classify_distant_people(frame, raw_detections)
        return raw_detections

    def _find_small_people(self, frame: np.ndarray, raw_detections: list[RawDetection]) -> list[RawDetection]:
        """Search overlapping, enlarged tiles for distant people the full-frame pass missed. Only
        person detections are taken from tiles (tiles produced false weapons on ordinary scenes)."""
        s = self.settings
        h, w = frame.shape[:2]
        known = [d.xyxy for d in raw_detections if d.label.lower() in s.person_label_set]
        found: list[RawDetection] = []
        for grid in (int(g) for g in s.distant_search_grids.split(",") if g.strip()):
            th, tw = int(h / grid * 1.25), int(w / grid * 1.25)  # 25% overlap
            for gy in range(grid):
                for gx in range(grid):
                    y0 = min(h - th, int(gy * (h - th) / max(grid - 1, 1)))
                    x0 = min(w - tw, int(gx * (w - tw) / max(grid - 1, 1)))
                    scale = tw / 640
                    tile = cv2.resize(frame[y0:y0 + th, x0:x0 + tw], (640, max(1, int(th / scale))),
                                      interpolation=cv2.INTER_CUBIC)
                    for d in self._run_models(tile):
                        if d.label.lower() not in s.person_label_set or d.confidence < max(s.confidence_threshold, 0.25):
                            continue
                        a, b, c, e = d.xyxy
                        box = (x0 + a * scale, y0 + b * scale, x0 + c * scale, y0 + e * scale)
                        if box[3] - box[1] > s.armed_classifier_max_height:
                            continue  # large people are already handled by the full-frame pass
                        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                        if any(k[0] <= cx <= k[2] and k[1] <= cy <= k[3] for k in known):
                            continue
                        duplicate = next((i for i, f in enumerate(found) if _iou(f.xyxy, box) > 0.4), None)
                        if duplicate is not None:
                            if d.confidence > found[duplicate].confidence:
                                found[duplicate] = RawDetection(d.label, d.confidence, box)
                            continue
                        found.append(RawDetection(d.label, d.confidence, box))
        if found:
            print(f"[DISTANT] small-person search found {len(found)} additional people")
        return found

    def _classify_distant_people(self, frame: np.ndarray, raw_detections: list[RawDetection]) -> list[RawDetection]:
        """Second stage for distant people: judge each small person with no weapon found yet as
        armed / unarmed from the whole low-resolution crop. Returns inferred weapon detections
        (box = the person's body), evidence "classifier"."""
        s = self.settings
        height, width = frame.shape[:2]
        people = [d for d in raw_detections if d.label.lower() in s.person_label_set
                  and d.confidence >= s.zoom_min_person_confidence
                  and d.xyxy[3] - d.xyxy[1] <= s.armed_classifier_max_height]
        weapons = [d.xyxy for d in raw_detections if d.label.lower() in s.weapon_label_set]
        candidates, crops = [], []
        for person in people[: s.zoom_max_people]:
            x1, y1, x2, y2 = person.xyxy
            pw, ph = x2 - x1, y2 - y1
            if any(x1 - 0.5 * pw <= (w[0] + w[2]) / 2 <= x2 + 0.5 * pw and y1 - 0.3 * ph <= (w[1] + w[3]) / 2 <= y2 + 0.3 * ph
                   for w in weapons):
                continue  # already armed by the detector
            # same crop geometry the classifier was trained on (training/build_armed_crops.py)
            side = max(pw + 0.7 * max(pw, ph), ph * 1.4)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            box = (int(max(0, cx - side / 2)), int(max(0, cy - side / 2)), int(min(width, cx + side / 2)), int(min(height, cy + side / 2)))
            crop = frame[box[1]:box[3], box[0]:box[2]]
            if crop.size == 0:
                continue
            candidates.append(person)
            crops.append(cv2.resize(crop, (160, 160), interpolation=cv2.INTER_CUBIC))
        found = []
        if crops:
            results = self._armed_model.predict(crops, imgsz=160, device=s.device, verbose=False)
            for person, result in zip(candidates, results):
                p_armed = float(result.probs.data[list(result.names.values()).index("armed")])
                if p_armed >= s.armed_classifier_threshold:
                    x1, y1, x2, y2 = person.xyxy
                    found.append(RawDetection("weapon", p_armed, (x1, y1, x2, y2), evidence="classifier"))
                    print(f"[DISTANT] person ({y2 - y1:.0f}px tall) judged armed (p={p_armed:.2f})")
        return found

    def _predict(self, frame: np.ndarray) -> list[RawDetection]:
        if not self.ready:
            raise DetectorNotReady(self.status_message)

        rgb_frame = frame[:, :, ::-1]

        if self._detection_model is not None:
            from sahi.predict import get_sliced_prediction

            result = get_sliced_prediction(
                rgb_frame,
                self._detection_model,
                slice_height=self.settings.slice_height,
                slice_width=self.settings.slice_width,
                overlap_height_ratio=self.settings.overlap_height_ratio,
                overlap_width_ratio=self.settings.overlap_width_ratio,
                postprocess_type="NMS",  # Non-maximum suppression for overlapping boxes
                postprocess_match_metric="IOS",  # Intersection over smaller area
                postprocess_match_threshold=0.5,  # Merge threshold
                postprocess_class_agnostic=False,  # Keep class-specific NMS
                verbose=0,
            )

            raw_detections = [
                RawDetection(
                    label=prediction.category.name,
                    confidence=float(prediction.score.value),
                    xyxy=tuple(float(value) for value in prediction.bbox.to_xyxy()),
                )
                for prediction in result.object_prediction_list
            ]
            
            # Log all detections for debugging
            weapon_labels = self.settings.weapon_label_set
            for rd in raw_detections:
                label_lower = rd.label.lower()
                is_weapon = label_lower in weapon_labels
                print(f"[DETECTION] {rd.label} (conf={rd.confidence:.2f}) - {'WEAPON' if is_weapon else 'OTHER'}")
            
            return raw_detections

        if self._models:
            return [d for d in self._run_models(frame) if d.confidence >= self.settings.confidence_threshold]

        raise DetectorNotReady(self.status_message)

    def _run_models(self, image: np.ndarray, models: list[Any] | None = None) -> list[RawDetection]:
        """All model outputs for one image (BGR). With an ensemble, overlapping same-class boxes are
        merged with the mean confidence and a box found by only one model keeps half its
        confidence: agreement is kept, a single model's false alarm is damped."""
        outputs = []
        for model in (models if models is not None else self._models):
            result = model.predict(image, conf=0.05, device=self.settings.device, verbose=False)[0]
            outputs.append([
                RawDetection(str(result.names[int(c)]), float(p), tuple(float(v) for v in b))
                for b, c, p in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist())
            ])
        if len(outputs) == 1:
            return outputs[0]
        first, second = outputs
        merged, used = [], set()
        for a in first:
            best, best_iou = None, (WEAPON_MATCH_IOU if a.label.lower() in self.settings.weapon_label_set
                                    else PERSON_MATCH_IOU)
            for j, b in enumerate(second):
                if j in used or b.label != a.label:
                    continue
                overlap = _iou(a.xyxy, b.xyxy)
                if overlap >= best_iou:
                    best, best_iou = j, overlap
            if best is None:
                merged.append(RawDetection(a.label, a.confidence / 2, a.xyxy))
                continue
            used.add(best)
            b = second[best]
            w = a.confidence + b.confidence
            box = tuple((a.confidence * x + b.confidence * y) / w for x, y in zip(a.xyxy, b.xyxy))
            merged.append(RawDetection(a.label, (a.confidence + b.confidence) / 2, box))
        merged += [RawDetection(b.label, b.confidence / 2, b.xyxy) for j, b in enumerate(second) if j not in used]
        return merged

    def _zoom_weapons(self, frame: np.ndarray, raw_detections: list[RawDetection],
                      thermal_view: bool = False) -> list[RawDetection]:
        """Re-examine each detected person at higher magnification for weapons too small to see
        in the full frame. Returns only new weapon detections, in full-frame coordinates."""
        import cv2

        s = self.settings
        person_labels, weapon_labels = s.person_label_set, s.weapon_label_set
        # Only zoom on people the detector is reasonably sure of: a faint false "person" (e.g. part of a
        # car) would otherwise send a crop of an ordinary object to the weapon search
        people = sorted((d for d in raw_detections if d.label.lower() in person_labels
                         and d.confidence >= s.zoom_min_person_confidence),
                        key=lambda d: d.confidence, reverse=True)[: s.zoom_max_people]
        known_weapons = [d.xyxy for d in raw_detections if d.label.lower() in weapon_labels]
        height, width = frame.shape[:2]
        found: list[RawDetection] = []

        for person in people:
            x1, y1, x2, y2 = person.xyxy
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            side = max(max(x2 - x1, y2 - y1) * s.zoom_pad, s.zoom_min_crop)
            left, top = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
            right, bottom = int(min(width, cx + side / 2)), int(min(height, cy + side / 2))
            if right - left < 8 or bottom - top < 8:
                continue

            crop = cv2.resize(frame[top:bottom, left:right], (s.zoom_size, s.zoom_size),
                              interpolation=cv2.INTER_CUBIC)
            scale_x, scale_y = (right - left) / s.zoom_size, (bottom - top) / s.zoom_size
            reach = 0.75 * max(x2 - x1, y2 - y1)

            def near_person(box) -> bool:
                wx, wy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                return x1 - reach <= wx <= x2 + reach and y1 - reach <= wy <= y2 + reach

            def accept(candidates: list[RawDetection], threshold: float, source: str, evidence: str) -> int:
                accepted = 0
                for candidate in candidates:
                    label, conf = candidate.label, candidate.confidence
                    if label.lower() not in weapon_labels or conf < threshold:
                        continue
                    u1, v1, u2, v2 = candidate.xyxy
                    xyxy = (left + u1 * scale_x, top + v1 * scale_y, left + u2 * scale_x, top + v2 * scale_y)
                    # A held weapon is smaller than the person holding it and close to them
                    if (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]) > (x2 - x1) * (y2 - y1) or not near_person(xyxy):
                        continue
                    if any(_iou(xyxy, other) > 0.3 for other in known_weapons):
                        continue
                    known_weapons.append(xyxy)
                    found.append(RawDetection(label=label, confidence=float(conf), xyxy=xyxy, evidence=evidence))
                    accepted += 1
                    print(f"[{source}] {label} (conf={conf:.2f}) found near person (conf={person.confidence:.2f})")
                return accepted

            already_armed = any(near_person(box) for box in known_weapons)
            hits = accept(self._run_models(crop), s.zoom_confidence, "ZOOM", "zoom")
            # Thermal view and still no weapon on this person: second opinion from the thermal-rifle
            # specialists (see config.py), with their own threshold
            if thermal_view and self._specialists and not hits and not already_armed:
                accept(self._run_models(crop, self._specialists), s.weapon_specialist_confidence, "ZOOM-SPECIALIST",
                       "specialist")

        return found

    def _classify(self, raw_detections: list[RawDetection], frame_width: int, frame_height: int) -> list[Detection]:
        person_labels = self.settings.person_label_set
        weapon_labels = self.settings.weapon_label_set
        people: list[Detection] = []
        weapons: list[Detection] = []
        other: list[Detection] = []

        for raw_detection in raw_detections:
            normalized_label = raw_detection.label.lower()
            detection = Detection(
                id=str(uuid4()),
                label=raw_detection.label,
                kind="person" if normalized_label in person_labels else "weapon" if normalized_label in weapon_labels else "other",
                confidence=max(0, min(raw_detection.confidence, 1)),
                bbox=_normalize_box(raw_detection.xyxy, frame_width, frame_height),
                sensor=raw_detection.sensor,
                evidence=raw_detection.evidence if normalized_label in weapon_labels else None,
            )

            if detection.kind == "person":
                people.append(detection)
            elif detection.kind == "weapon":
                weapons.append(detection)
            else:
                other.append(detection)

        assignments = {person.id: 0 for person in people}
        evidence: dict[str, str] = {}
        rank = {"detector": 4, "zoom": 3, "specialist": 2, "classifier": 1}
        for weapon in weapons:
            person = _nearest_person(weapon, people)
            if person is not None:
                weapon.associated_person_id = person.id
                assignments[person.id] += 1
                current = evidence.get(person.id)
                if current is None or rank.get(weapon.evidence or "", 0) > rank.get(current, 0):
                    evidence[person.id] = weapon.evidence or "detector"

        for person in people:
            person.weapon_count = assignments[person.id]
            person.armed = person.weapon_count > 0
            person.armed_evidence = evidence.get(person.id)

        return [*people, *weapons, *other]

    def finalize(self, detections: list[Detection], distant_mode: str | None = None, drone: bool = False
                 ) -> tuple[list[Detection], ThreatSummary]:
        """Decide what is reported. A weapon the detector or the zoom pass saw is a weapon. Weaker
        evidence (distant classifier, thermal-rifle specialist) is dropped unless the tracker has
        confirmed the person armed over recent frames (live stream and video); then it is a weapon
        too. So the operator sees either a weapon (High) or a person (Elevated), never a "maybe".
        drone=True (live drone stream) also drops implausibly large "people" (see config).
        Call after ArmedTracker.update; returns (detections to show, summary)."""
        mode = distant_mode or self.settings.distant_mode
        if drone:
            # people or weapons outlined across most of the frame (no real target is that close)
            ghosts = {d.id for d in detections
                      if d.kind in ("person", "weapon") and d.evidence != "classifier"
                      and d.bbox.width > self.settings.live_max_person_width_pct}
            detections = [d for d in detections if d.id not in ghosts and d.associated_person_id not in ghosts]
        people = {d.id: d for d in detections if d.kind == "person"}
        kept: list[Detection] = []
        for d in detections:
            if d.kind == "weapon" and d.evidence in UNCONFIRMED_EVIDENCE:
                person = people.get(d.associated_person_id or "")
                if mode == "off" or person is None or not person.armed_confirmed:
                    if person is not None:
                        person.weapon_count = max(0, person.weapon_count - 1)
                    continue
            kept.append(d)
        for person in people.values():
            if person.armed and person.armed_evidence in UNCONFIRMED_EVIDENCE and not person.armed_confirmed:
                person.armed, person.armed_evidence = False, None
            person.armed_possible = False
        return kept, self._summarize(kept, mode)

    def _summarize(self, detections: list[Detection], distant_mode: str | None = None) -> ThreatSummary:
        people = [detection for detection in detections if detection.kind == "person"]
        # A person armed only on weak evidence (distant classifier, thermal-rifle specialist) that is
        # not yet confirmed over frames does not count. finalize() removes such findings before the
        # summary, so this only matters for callers that skip it; once confirmed they count in full.
        possible = [d for d in people if d.armed and d.armed_evidence in UNCONFIRMED_EVIDENCE
                    and not d.armed_confirmed]
        possible_ids = {d.id for d in possible}
        for d in people:
            d.armed_possible = d.id in possible_ids
        armed_people = [detection for detection in people if detection.armed and detection.id not in possible_ids]
        weapon_count = len([d for d in detections if d.kind == "weapon"
                            and not (d.evidence in UNCONFIRMED_EVIDENCE and d.associated_person_id in possible_ids)])
        
        # New threat logic:
        # - Any human detected = Elevated
        # - Human + weapon = High
        # - Multiple armed people OR multiple weapons = Critical
        if len(armed_people) >= 2 or (len(armed_people) >= 1 and weapon_count >= 2):
            level = ThreatLevel.critical
            risk_score = 10
        elif len(armed_people) >= 1 or weapon_count >= 1:
            level = ThreatLevel.high
            risk_score = 5
        elif len(people) >= 1:
            level = ThreatLevel.elevated
            risk_score = 2
        else:
            level = ThreatLevel.clear
            risk_score = 0
        
        # Mean model confidence over all detections, as reported by the model (0-100)
        if detections:
            average_confidence = round(100 * sum(d.confidence for d in detections) / len(detections))
        else:
            average_confidence = 0

        return ThreatSummary(
            level=level,
            total_people=len(people),
            armed_people=len(armed_people),
            weapon_count=weapon_count,
            risk_score=risk_score,
            average_confidence=average_confidence,
            possible_armed=len(possible),
        )

    def _demo_detections(self, frame: np.ndarray, frame_id: int) -> list[RawDetection]:
        scenario = frame_id % 3
        if scenario == 0:
            return [RawDetection(label="person", confidence=0.92, xyxy=(430, 300, 520, 520))]
        if scenario == 1:
            return [
                RawDetection(label="person", confidence=0.89, xyxy=(330, 240, 400, 460)),
                RawDetection(label="person", confidence=0.94, xyxy=(620, 310, 700, 530)),
                RawDetection(label="rifle", confidence=0.81, xyxy=(374, 318, 430, 342)),
            ]

        return [
            RawDetection(label="person", confidence=0.96, xyxy=(510, 230, 635, 535)),
            RawDetection(label="person", confidence=0.86, xyxy=(170, 170, 235, 340)),
            RawDetection(label="rifle", confidence=0.87, xyxy=(575, 330, 655, 360)),
            RawDetection(label="pistol", confidence=0.78, xyxy=(212, 240, 250, 260)),
            RawDetection(label="knife", confidence=0.72, xyxy=(595, 375, 630, 395)),
        ]


def _normalize_box(xyxy: tuple[float, float, float, float], frame_width: int, frame_height: int) -> BoundingBox:
    x1, y1, x2, y2 = xyxy
    x1 = _clamp(x1, 0, frame_width)
    x2 = _clamp(x2, 0, frame_width)
    y1 = _clamp(y1, 0, frame_height)
    y2 = _clamp(y2, 0, frame_height)

    return BoundingBox(
        x=(x1 / frame_width) * 100,
        y=(y1 / frame_height) * 100,
        width=max(0, (x2 - x1) / frame_width * 100),
        height=max(0, (y2 - y1) / frame_height * 100),
    )


def _nearest_person(weapon: Detection, people: list[Detection]) -> Detection | None:
    if not people:
        return None

    weapon_center = _center(weapon.bbox)
    best_person: Detection | None = None
    best_distance = float("inf")

    for person in people:
        # A held weapon can extend about half a body-length beyond the person's box. The margin
        # scales with the person so distant (small) people don't claim weapons far away.
        expanded_box = _expand(person.bbox, max(person.bbox.width, person.bbox.height) + 2)
        if _contains(expanded_box, weapon_center):
            distance = _distance(weapon_center, _center(person.bbox))
            if distance < best_distance:
                best_distance = distance
                best_person = person

    return best_person


CONCEALED_MIN_PERSON_PX = 150


def _mark_concealed(detections: list[Detection], view: np.ndarray, method: str,
                    rgb_reliability: float | None = None) -> None:
    """Flag weapons that are likely hidden under clothing.

    RGBT fusion: a weapon seen only by thermal, on a person's body, with no counterpart in the colour
    image - the colour camera cannot see it, so it is concealed. Thermal only: a weapon lying inside
    the body outline (torso/waist) that is colder than the surrounding body - the cold patch metal
    leaves on warm clothing. A rifle held away from the body is neither."""
    if method == "rgb":
        return
    people = {d.id: d for d in detections if d.kind == "person"}
    grey = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY) if view.ndim == 3 else view
    if method in ("thermal_inferno", "rgbt") and view.ndim == 3:
        grey = cv2.cvtColor(inferno_to_grey(view), cv2.COLOR_BGR2GRAY)
    h, w = grey.shape[:2]
    for weapon in (d for d in detections if d.kind == "weapon"):
        person = people.get(weapon.associated_person_id or "")
        if person is None or weapon.evidence in UNCONFIRMED_EVIDENCE:
            continue
        wb, pb = weapon.bbox, person.bbox
        ix = max(0.0, min(wb.x + wb.width, pb.x + pb.width) - max(wb.x, pb.x))
        iy = max(0.0, min(wb.y + wb.height, pb.y + pb.height) - max(wb.y, pb.y))
        inside = ix * iy / max(wb.width * wb.height, 1e-6)
        cy = wb.y + wb.height / 2
        on_torso = pb.y + 0.15 * pb.height <= cy <= pb.y + 0.85 * pb.height
        if inside < 0.8 or not on_torso:
            continue  # extends beyond the body: carried openly
        # Hidden vs held in front of the body cannot be told apart on a distant, small person
        # (the cold patch is a few pixels): only judged at close range
        if pb.height / 100 * h < CONCEALED_MIN_PERSON_PX:
            continue
        # "Invisible to the colour camera" only means something when that camera can see well
        # (at night it misses things simply because it is dark)
        if method == "rgbt" and weapon.sensor == "thermal" and (rgb_reliability or 0) >= 0.35:
            weapon.concealed = True
            weapon.concealment = "seen only by the thermal camera: on the body, invisible to the colour camera"
            continue

        def mean(b):
            x0, y0 = int(b.x / 100 * w), int(b.y / 100 * h)
            x1, y1 = max(x0 + 1, int((b.x + b.width) / 100 * w)), max(y0 + 1, int((b.y + b.height) / 100 * h))
            return float(grey[y0:y1, x0:x1].mean())

        if mean(wb) < mean(pb) - 8:
            weapon.concealed = True
            weapon.concealment = "cold patch on the body in thermal: likely carried under clothing"


def _drop_frame_filling_people(detections: list[RawDetection], shape: tuple, person_labels: set[str]) -> list[RawDetection]:
    """A faint "person" covering nearly the whole picture (seen on dark, featureless frames) is not
    a person standing in front of the lens."""
    area = shape[0] * shape[1]
    return [d for d in detections if not (d.label.lower() in person_labels and d.confidence < 0.5
                                          and (d.xyxy[2] - d.xyxy[0]) * (d.xyxy[3] - d.xyxy[1]) > 0.9 * area)]


def _suppress_contained(detections: list[RawDetection], threshold: float = 0.7) -> list[RawDetection]:
    """Per class, drop a box when most of it (intersection over its own area) lies inside a
    higher-confidence box. Standard NMS (IoU) misses small fragments inside a large box."""
    kept: list[RawDetection] = []
    for detection in sorted(detections, key=lambda d: d.confidence, reverse=True):
        x1, y1, x2, y2 = detection.xyxy
        area = max(1e-6, (x2 - x1) * (y2 - y1))
        contained = False
        for other in kept:
            if other.label != detection.label:
                continue
            ix = max(0.0, min(x2, other.xyxy[2]) - max(x1, other.xyxy[0]))
            iy = max(0.0, min(y2, other.xyxy[3]) - max(y1, other.xyxy[1]))
            if ix * iy / area > threshold:
                contained = True
                break
            # The reverse case for people: a much larger, weaker box wrapped around a confident person
            # (common at close range) is the same person, not a second one
            other_area = max(1e-6, (other.xyxy[2] - other.xyxy[0]) * (other.xyxy[3] - other.xyxy[1]))
            if (detection.label.lower() in ("person", "human") and ix * iy / other_area > 0.9
                    and area > 2.5 * other_area):
                contained = True
                break
        if not contained:
            kept.append(detection)
    return kept


def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(first[0], second[0]), max(first[1], second[1])
    ix2, iy2 = min(first[2], second[2]), min(first[3], second[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = ((first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1]) - inter)
    return inter / union if union > 0 else 0.0


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _center(box: BoundingBox) -> tuple[float, float]:
    return box.x + box.width / 2, box.y + box.height / 2


def _contains(box: BoundingBox, point: tuple[float, float]) -> bool:
    x, y = point
    return box.x <= x <= box.x + box.width and box.y <= y <= box.y + box.height


def _expand(box: BoundingBox, amount: float) -> BoundingBox:
    return BoundingBox(
        x=_clamp(box.x - amount / 2, 0, 100),
        y=_clamp(box.y - amount / 2, 0, 100),
        width=_clamp(box.width + amount, 0, 100),
        height=_clamp(box.height + amount, 0, 100),
    )


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5

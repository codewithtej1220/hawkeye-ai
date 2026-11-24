"""Per-person evidence over time, for the live stream and video analysis.

A weapon a few pixels wide is found in only some frames. Tracking each person and accumulating
"armed" evidence across frames turns that weak per-frame signal into a reliable one: a person seen
armed in at least `confirm_hits` of their last `window` frames is confirmed armed and stays marked
even in frames where the weapon is missed. A false alarm in a single frame never reaches
confirmation, so it does not stick.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .schemas import Detection


@dataclass
class _Track:
    track_id: int
    box: tuple[float, float, float, float]  # x, y, w, h in % of the frame
    hits: deque = field(default_factory=deque)  # per frame: how the person was armed, or None
    missed: int = 0


class ArmedTracker:
    def __init__(self, window: int = 6, confirm_hits: int = 2, max_missed: int = 4):
        self.window = window
        self.confirm_hits = confirm_hits
        self.max_missed = max_missed
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Detection]:
        """Match people to tracks, record whether each is armed in this frame, and mark people
        whose track is confirmed armed. Returns the (updated) detections."""
        people = [d for d in detections if d.kind == "person"]
        unmatched = set(range(len(self._tracks)))
        for person in sorted(people, key=lambda d: -d.confidence):
            box = (person.bbox.x, person.bbox.y, person.bbox.width, person.bbox.height)
            best, best_score = None, 0.0
            for i in unmatched:
                score = _match_score(self._tracks[i].box, box)
                if score > best_score:
                    best, best_score = i, score
            if best is None or best_score < 0.25:
                track = _Track(self._next_id, box, deque(maxlen=self.window))
                self._next_id += 1
                self._tracks.append(track)
            else:
                unmatched.discard(best)
                track = self._tracks[best]
                track.box, track.missed = box, 0
            track.hits.append((person.armed_evidence or "detector") if person.armed else None)
            person.track_id = track.track_id
            armed_hits = sum(h is not None for h in track.hits)
            # weapon actually seen (detector / zoom), as opposed to unconfirmed evidence
            # (distant classifier, thermal-rifle specialist)
            seen_hits = sum(h not in (None, "classifier", "specialist") for h in track.hits)
            person.armed_frames = f"{armed_hits}/{len(track.hits)}"
            if armed_hits >= self.confirm_hits:
                person.armed_confirmed = True
                if not person.armed:
                    # Weapon missed in this frame, but this person was armed in recent frames. Keep the
                    # kind of evidence: unconfirmed-only history stays an unconfirmed finding.
                    person.armed = True
                    unconfirmed = [h for h in track.hits if h in ("classifier", "specialist")]
                    person.armed_evidence = ("tracked" if seen_hits >= self.confirm_hits
                                             else unconfirmed[-1] if unconfirmed else "classifier")
        for i in unmatched:
            self._tracks[i].missed += 1
        self._tracks = [t for t in self._tracks if t.missed <= self.max_missed]
        return detections


def _match_score(a, b) -> float:
    """IoU, falling back to centre distance relative to size for small, fast-moving people."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    iou = inter / union if union > 0 else 0.0
    size = max(aw, ah, bw, bh, 1e-6)
    dist = ((ax + aw / 2 - bx - bw / 2) ** 2 + (ay + ah / 2 - by - bh / 2) ** 2) ** 0.5
    proximity = max(0.0, 1 - dist / (1.5 * size))
    return max(iou, 0.6 * proximity)

"""Persistent event log for officials to review later.

Every threat-level change on the live stream, a periodic update while a threat persists, and every
uploaded/linked media analysis is stored in SQLite with an annotated snapshot on disk. Events are
never edited; a reviewer can only add an acknowledgement (who, when, note)."""
from __future__ import annotations

import csv
import io
import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .schemas import Detection, ThreatSummary

LEVEL_RANK = {"Clear": 0, "Elevated": 1, "High": 2, "Critical": 3}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    source TEXT NOT NULL,
    event_type TEXT NOT NULL,
    level TEXT NOT NULL,
    previous_level TEXT,
    people INTEGER NOT NULL DEFAULT 0,
    armed_people INTEGER NOT NULL DEFAULT 0,
    weapons INTEGER NOT NULL DEFAULT 0,
    possible_armed INTEGER NOT NULL DEFAULT 0,
    average_confidence INTEGER NOT NULL DEFAULT 0,
    method TEXT,
    method_label TEXT,
    conditions TEXT,
    altitude_m REAL,
    range_m REAL,
    description TEXT,
    telemetry TEXT,
    detections TEXT,
    snapshot TEXT,
    acknowledged INTEGER NOT NULL DEFAULT 0,
    ack_by TEXT,
    ack_note TEXT,
    ack_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events (ts);
CREATE INDEX IF NOT EXISTS idx_events_level ON events (level);
"""

CSV_COLUMNS = ["id", "ts", "source", "event_type", "level", "previous_level", "people", "armed_people", "weapons",
               "possible_armed", "average_confidence", "method_label", "conditions", "altitude_m", "range_m",
               "description", "acknowledged", "ack_by", "ack_note", "ack_at"]


def describe(summary: ThreatSummary) -> str:
    """One plain sentence for the log."""
    people, armed, weapons = summary.total_people, summary.armed_people, summary.weapon_count
    possible = summary.possible_armed
    unarmed = max(0, people - armed - possible)
    parts = []
    if armed:
        who = f"{armed} armed {'person' if armed == 1 else 'people'}"
        # armed people with no weapon in this frame were seen armed in recent frames (tracking)
        parts.append(f"{who} with {weapons} weapon{'' if weapons == 1 else 's'}" if weapons
                     else f"{who} (weapon seen in recent frames)")
    elif weapons:
        parts.append(f"{weapons} weapon{'' if weapons == 1 else 's'} detected with no person attached")
    if possible:
        parts.append(f"{possible} {'person' if possible == 1 else 'people'} possibly armed (verify)")
    if unarmed:
        parts.append(f"{unarmed} unarmed {'person' if unarmed == 1 else 'people'}")
    return (", ".join(parts) + ".").capitalize() if parts else "No people or weapons in view."


class EventLog:
    def __init__(self, db_path: Path, snapshot_dir: Path, max_events: int = 20000):
        self.db_path, self.snapshot_dir, self.max_events = Path(db_path), Path(snapshot_dir), max_events
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._db.commit()

    # ------------------------------------------------------------------ writing
    def add(self, *, source: str, event_type: str, summary: ThreatSummary, detections: list[Detection],
            snapshot: np.ndarray | None = None, previous_level: str | None = None, method: str | None = None,
            method_label: str | None = None, conditions: list[str] | None = None,
            telemetry: dict | None = None, description: str | None = None) -> dict:
        now = datetime.now(timezone.utc)
        ranges = [d.range_m for d in detections if d.kind == "person" and d.range_m]
        snapshot_name = None
        if snapshot is not None:
            day = self.snapshot_dir / now.strftime("%Y%m%d")
            day.mkdir(parents=True, exist_ok=True)
            snapshot_name = f"{now.strftime('%Y%m%d')}/{now.strftime('%H%M%S_%f')}.jpg"
            height, width = snapshot.shape[:2]
            if width > 1280:  # keep the archive compact
                snapshot = cv2.resize(snapshot, (1280, int(height * 1280 / width)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(self.snapshot_dir / snapshot_name), snapshot, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        row = {
            "ts": now.isoformat(timespec="milliseconds"),
            "source": source,
            "event_type": event_type,
            "level": summary.level.value,
            "previous_level": previous_level,
            "people": summary.total_people,
            "armed_people": summary.armed_people,
            "weapons": summary.weapon_count,
            "possible_armed": summary.possible_armed,
            "average_confidence": summary.average_confidence,
            "method": method,
            "method_label": method_label,
            "conditions": ", ".join(conditions or []),
            "altitude_m": (telemetry or {}).get("altitude_m"),
            "range_m": round(min(ranges), 1) if ranges else None,
            "description": description or describe(summary),
            "telemetry": json.dumps(telemetry) if telemetry else None,
            "detections": json.dumps([_compact(d) for d in detections]),
            "snapshot": snapshot_name,
        }
        with self._lock:
            cursor = self._db.execute(
                f"INSERT INTO events ({', '.join(row)}) VALUES ({', '.join('?' for _ in row)})", list(row.values()))
            self._db.commit()
            row["id"] = cursor.lastrowid
            self._prune()
        return _public(row)

    def _prune(self) -> None:
        count = self._db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        if count <= self.max_events:
            return
        old = self._db.execute("SELECT id, snapshot FROM events ORDER BY id LIMIT ?", (count - self.max_events,)).fetchall()
        for row in old:
            if row["snapshot"]:
                (self.snapshot_dir / row["snapshot"]).unlink(missing_ok=True)
        self._db.execute("DELETE FROM events WHERE id <= ?", (old[-1]["id"],))
        self._db.commit()

    def acknowledge(self, event_id: int, by: str, note: str) -> dict | None:
        with self._lock:
            self._db.execute("UPDATE events SET acknowledged = 1, ack_by = ?, ack_note = ?, ack_at = ? WHERE id = ?",
                             (by.strip()[:80] or "operator", note.strip()[:1000],
                              datetime.now(timezone.utc).isoformat(timespec="seconds"), event_id))
            self._db.commit()
        return self.get(event_id)

    # ------------------------------------------------------------------ reading
    def _where(self, level: str | None, source: str | None, since: str | None, until: str | None,
               search: str | None, unacknowledged: bool, min_level: str | None) -> tuple[str, list[Any]]:
        clauses, params = [], []
        if level:
            clauses.append("level = ?"); params.append(level)
        if min_level and min_level in LEVEL_RANK:
            wanted = [name for name, rank in LEVEL_RANK.items() if rank >= LEVEL_RANK[min_level]]
            clauses.append(f"level IN ({', '.join('?' for _ in wanted)})"); params += wanted
        if source:
            clauses.append("source = ?"); params.append(source)
        if since:
            clauses.append("ts >= ?"); params.append(since)
        if until:
            clauses.append("ts <= ?"); params.append(until)
        if search:
            clauses.append("(description LIKE ? OR ack_note LIKE ? OR ack_by LIKE ? OR method_label LIKE ?)")
            params += [f"%{search}%"] * 4
        if unacknowledged:
            clauses.append("acknowledged = 0")
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def query(self, *, level: str | None = None, source: str | None = None, since: str | None = None,
              until: str | None = None, search: str | None = None, unacknowledged: bool = False,
              min_level: str | None = None, limit: int = 50, offset: int = 0) -> dict:
        where, params = self._where(level, source, since, until, search, unacknowledged, min_level)
        with self._lock:
            total = self._db.execute(f"SELECT COUNT(*) FROM events{where}", params).fetchone()[0]
            rows = self._db.execute(f"SELECT * FROM events{where} ORDER BY id DESC LIMIT ? OFFSET ?",
                                    [*params, max(1, min(limit, 500)), max(0, offset)]).fetchall()
        return {"total": total, "events": [_public(dict(r), with_detections=False) for r in rows]}

    def get(self, event_id: int) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        return _public(dict(row)) if row else None

    def snapshot_path(self, event_id: int) -> Path | None:
        with self._lock:
            row = self._db.execute("SELECT snapshot FROM events WHERE id = ?", (event_id,)).fetchone()
        if not row or not row["snapshot"]:
            return None
        path = self.snapshot_dir / row["snapshot"]
        return path if path.exists() else None

    def stats(self) -> dict:
        day_ago = datetime.fromtimestamp(time.time() - 86400, timezone.utc).isoformat(timespec="milliseconds")
        with self._lock:
            by_level = {r["level"]: r["n"] for r in self._db.execute(
                "SELECT level, COUNT(*) AS n FROM events GROUP BY level")}
            last_24h = {r["level"]: r["n"] for r in self._db.execute(
                "SELECT level, COUNT(*) AS n FROM events WHERE ts >= ? GROUP BY level", (day_ago,))}
            total, unack, first, last = self._db.execute(
                "SELECT COUNT(*), SUM(CASE WHEN acknowledged = 0 AND level IN ('High', 'Critical') THEN 1 ELSE 0 END), "
                "MIN(ts), MAX(ts) FROM events").fetchone()
        return {"total": total, "by_level": {k: by_level.get(k, 0) for k in LEVEL_RANK},
                "last_24h": {k: last_24h.get(k, 0) for k in LEVEL_RANK},
                "unacknowledged_threats": unack or 0, "first_event": first, "last_event": last}

    def export_csv(self, **filters) -> str:
        where, params = self._where(filters.get("level"), filters.get("source"), filters.get("since"),
                                    filters.get("until"), filters.get("search"), filters.get("unacknowledged", False),
                                    filters.get("min_level"))
        with self._lock:
            rows = self._db.execute(f"SELECT {', '.join(CSV_COLUMNS)} FROM events{where} ORDER BY id DESC", params).fetchall()
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(CSV_COLUMNS)
        writer.writerows([list(r) for r in rows])
        return buffer.getvalue()


def _compact(d: Detection) -> dict:
    out = {"kind": d.kind, "label": d.label, "confidence": round(d.confidence, 3),
           "bbox": [round(v, 2) for v in (d.bbox.x, d.bbox.y, d.bbox.width, d.bbox.height)]}
    for key in ("armed", "concealed", "armed_possible", "armed_confirmed"):
        if getattr(d, key):
            out[key] = True
    for key in ("sensor", "evidence", "armed_evidence", "range_m", "track_id"):
        if getattr(d, key) is not None:
            out[key] = getattr(d, key)
    return out


def _public(row: dict, with_detections: bool = True) -> dict:
    row = dict(row)
    row["acknowledged"] = bool(row.get("acknowledged"))
    row["has_snapshot"] = bool(row.pop("snapshot", None))
    row["telemetry"] = json.loads(row["telemetry"]) if row.get("telemetry") else None
    detections = row.pop("detections", None)
    if with_detections:
        row["detections"] = json.loads(detections) if detections else []
    return row


class LiveEventPolicy:
    """Decides which live frames become log entries: a threat-level change that holds for
    `confirm_frames` frames, then a periodic update while a threat persists."""

    def __init__(self, confirm_frames: int = 2, threat_update_s: float = 15.0, presence_update_s: float = 60.0,
                 stale_s: float = 10.0):
        self.confirm_frames, self.threat_update_s = confirm_frames, threat_update_s
        self.presence_update_s, self.stale_s = presence_update_s, stale_s
        self.level = "Clear"          # last logged level
        self._candidate, self._candidate_count = "Clear", 0
        self._last_logged, self._last_frame = 0.0, 0.0

    def decide(self, level: str) -> tuple[str | None, str]:
        """Returns (event_type or None, previous_level) for a frame at `level`."""
        now = time.monotonic()
        if now - self._last_frame > self.stale_s and self.level != "Clear":
            self.level = "Clear"      # the stream stopped: the next frame starts a new observation
        self._last_frame = now
        if level != self.level:
            if level == self._candidate:
                self._candidate_count += 1
            else:
                self._candidate, self._candidate_count = level, 1
            # an escalation to a weapon is logged on the first confirmation; flicker is ignored
            if self._candidate_count >= self.confirm_frames:
                previous, self.level = self.level, level
                self._last_logged = now
                return "level_change", previous
            return None, self.level
        self._candidate_count = 0
        period = self.threat_update_s if LEVEL_RANK[level] >= 2 else self.presence_update_s if level == "Elevated" else None
        if period and now - self._last_logged >= period:
            self._last_logged = now
            return "update", self.level
        return None, self.level

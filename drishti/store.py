"""SQLite event store and watchlist persistence (WAL mode, safe across threads)."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from .types import PlateEvent
from .watchlist import WatchEntry, normalize_plate

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    camera TEXT,
    track_id INTEGER,
    plate TEXT,
    confidence REAL,
    format TEXT,
    first_ts REAL,
    last_ts REAL,
    frames INTEGER,
    reads INTEGER,
    vehicle_type TEXT,
    vehicle_color TEXT,
    direction TEXT,
    low_confidence INTEGER,
    watch_hit TEXT,
    crop_path TEXT,
    payload TEXT,
    created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_events_plate ON events(plate);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(first_ts);
CREATE TABLE IF NOT EXISTS watchlist (
    plate TEXT PRIMARY KEY,
    reason TEXT,
    level TEXT,
    created_at REAL
);
"""


class EventStore:
    def __init__(self, db_path: str | Path):
        self.path = str(db_path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def add_event(self, ev: PlateEvent, plate_value: str | None = None) -> None:
        d = ev.to_dict()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ev.event_id, ev.camera, ev.track_id, plate_value if plate_value is not None else ev.text,
                    ev.confidence, ev.fmt, ev.first_ts, ev.last_ts, ev.n_frames, ev.n_reads, ev.vehicle_type,
                    ev.vehicle_color, ev.direction, int(ev.low_confidence),
                    json.dumps(ev.watchlist_hit) if ev.watchlist_hit else None, ev.crop_path, json.dumps(d), time.time(),
                ),
            )
            self._conn.commit()

    def recent_similar(self, plate: str, window_s: float, exclude_camera: str | None = None,
                       cameras: list[str] | None = None, max_distance: float = 1.0) -> str | None:
        """event_id of the same plate stored by another camera within the last `window_s` seconds."""
        from .watchlist import weighted_distance

        sql = "SELECT event_id, camera, plate FROM events WHERE created_at >= ? AND plate IS NOT NULL"
        args: list = [time.time() - window_s]
        if exclude_camera is not None:
            sql += " AND camera != ?"
            args.append(exclude_camera)
        if cameras:
            sql += f" AND camera IN ({','.join('?' * len(cameras))})"
            args += list(cameras)
        with self._lock:
            rows = self._conn.execute(sql + " ORDER BY created_at DESC", args).fetchall()
        for event_id, _cam, p in rows:
            if p == plate or (max_distance > 0 and weighted_distance(p, plate) <= max_distance):
                return event_id
        return None

    def query(self, plate: str | None = None, since: float | None = None, until: float | None = None,
              camera: str | None = None, limit: int = 100) -> list[dict]:
        sql, args = "SELECT payload, plate FROM events WHERE 1=1", []
        if plate:
            if "*" in plate or "%" in plate:
                sql += " AND plate LIKE ?"
                args.append(plate.replace("*", "%").upper())
            else:
                sql += " AND plate = ?"
                args.append(normalize_plate(plate))
        if since is not None:
            sql += " AND last_ts >= ?"
            args.append(since)
        if until is not None:
            sql += " AND first_ts <= ?"
            args.append(until)
        if camera:
            sql += " AND camera = ?"
            args.append(camera)
        sql += " ORDER BY first_ts DESC LIMIT ?"
        args.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        out = []
        for payload, stored in rows:
            d = json.loads(payload)
            d["plate"] = stored
            out.append(d)
        return out

    def purge_older_than(self, days: float) -> int:
        cutoff = time.time() - days * 86400
        with self._lock:
            cur = self._conn.execute("DELETE FROM events WHERE created_at < ?", (cutoff,))
            self._conn.commit()
            return cur.rowcount

    # watchlist -------------------------------------------------------------
    def watch_add(self, entry: WatchEntry) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO watchlist VALUES (?,?,?,?)",
                               (normalize_plate(entry.plate), entry.reason, entry.level, time.time()))
            self._conn.commit()

    def watch_remove(self, plate: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM watchlist WHERE plate = ?", (normalize_plate(plate),))
            self._conn.commit()
            return cur.rowcount > 0

    def watch_list(self) -> list[WatchEntry]:
        with self._lock:
            rows = self._conn.execute("SELECT plate, reason, level FROM watchlist ORDER BY plate").fetchall()
        return [WatchEntry(p, r or "", l or "alert") for p, r, l in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

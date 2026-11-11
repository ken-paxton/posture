"""Local history in SQLite: per-minute aggregates and events only. Never images or landmarks."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import defaultdict
from pathlib import Path

from .config import data_dir

SCHEMA = """
CREATE TABLE IF NOT EXISTS minutes (
    ts INTEGER PRIMARY KEY,        -- unix time of the minute start
    profile TEXT,
    present_s REAL NOT NULL,
    good_s REAL NOT NULL,
    bad_s REAL NOT NULL,
    score_sum REAL NOT NULL,
    score_n INTEGER NOT NULL,
    issues TEXT NOT NULL           -- JSON {rule: seconds}
);
CREATE TABLE IF NOT EXISTS events (
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    issues TEXT,
    data TEXT
);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
"""


class MinuteAccumulator:
    def __init__(self):
        self.reset(None)

    def reset(self, minute: int | None):
        self.minute = minute
        self.present_s = self.good_s = self.bad_s = self.score_sum = 0.0
        self.score_n = 0
        self.issues: dict[str, float] = defaultdict(float)


class History:
    def __init__(self, path: Path | None = None):
        self.path = path or (data_dir() / "history.sqlite3")
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._acc = MinuteAccumulator()
        self.profile: str | None = None

    def close(self):
        self.flush()
        with self._lock:
            self._db.close()

    # -- writes ---------------------------------------------------------------
    def add_sample(self, wall_t: float, dt: float, present: bool, score: float | None, bad: bool, issues: list[str]):
        minute = int(wall_t // 60) * 60
        if self._acc.minute is not None and minute != self._acc.minute:
            self.flush()
        if self._acc.minute is None:
            self._acc.reset(minute)
        a = self._acc
        if not present or score is None:
            return
        a.present_s += dt
        if bad:
            a.bad_s += dt
            for k in issues:
                a.issues[k] += dt
        else:
            a.good_s += dt
        a.score_sum += score
        a.score_n += 1

    def flush(self):
        a = self._acc
        if a.minute is None:
            return
        if a.present_s > 0:
            with self._lock:
                self._db.execute(
                    "INSERT OR REPLACE INTO minutes VALUES (?,?,?,?,?,?,?,?)",
                    (a.minute, self.profile, a.present_s, a.good_s, a.bad_s, a.score_sum, a.score_n,
                     json.dumps(dict(a.issues))),
                )
                self._db.commit()
        a.reset(None)

    def add_event(self, wall_t: float, kind: str, issues: list[str], data: dict):
        with self._lock:
            self._db.execute("INSERT INTO events VALUES (?,?,?,?)", (wall_t, kind, json.dumps(issues), json.dumps(data)))
            self._db.commit()

    # -- reads ----------------------------------------------------------------
    def good_share(self, days: float = 7.0) -> tuple[float, float]:
        """(good fraction, monitored hours) over the last ``days``."""
        since = time.time() - days * 86400
        with self._lock:
            row = self._db.execute(
                "SELECT COALESCE(SUM(good_s),0), COALESCE(SUM(present_s),0) FROM minutes WHERE ts >= ?", (since,)
            ).fetchone()
        good, present = row
        return (good / present if present else 0.0), present / 3600.0

    def daily(self, days: int = 30) -> list[dict]:
        since = time.time() - days * 86400
        with self._lock:
            rows = self._db.execute(
                "SELECT date(ts,'unixepoch','localtime') d, SUM(present_s), SUM(good_s), SUM(bad_s), "
                "SUM(score_sum), SUM(score_n) FROM minutes WHERE ts >= ? GROUP BY d ORDER BY d",
                (since,),
            ).fetchall()
            ev = self._db.execute(
                "SELECT date(ts,'unixepoch','localtime') d, kind, COUNT(*) FROM events WHERE ts >= ? GROUP BY d, kind",
                (since,),
            ).fetchall()
            issue_rows = self._db.execute(
                "SELECT date(ts,'unixepoch','localtime') d, issues FROM minutes WHERE ts >= ?", (since,)
            ).fetchall()
        events: dict[str, dict[str, int]] = defaultdict(dict)
        for d, kind, n in ev:
            events[d][kind] = n
        issues: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for d, js in issue_rows:
            for k, v in json.loads(js).items():
                issues[d][k] += v
        out = []
        for d, present, good, bad, ssum, sn in rows:
            out.append(
                {
                    "date": d,
                    "monitored_min": round(present / 60, 1),
                    "good_pct": round(100 * good / present, 1) if present else None,
                    "avg_score": round(ssum / sn, 1) if sn else None,
                    "alerts": events[d].get("posture_alert", 0),
                    "corrected": events[d].get("corrected", 0),
                    "movement_reminders": events[d].get("movement", 0) + events[d].get("break", 0),
                    "issues_min": {k: round(v / 60, 1) for k, v in sorted(issues[d].items(), key=lambda kv: -kv[1])},
                }
            )
        return out

    def today_minutes(self) -> list[dict]:
        start = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
        with self._lock:
            rows = self._db.execute(
                "SELECT ts, present_s, good_s, score_sum, score_n FROM minutes WHERE ts >= ? ORDER BY ts", (start,)
            ).fetchall()
        return [
            {"ts": ts, "good_pct": round(100 * g / p, 1) if p else None, "score": round(s / n, 1) if n else None}
            for ts, p, g, s, n in rows
        ]

    def recent_events(self, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT ts, kind, issues, data FROM events ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [{"ts": ts, "kind": k, "issues": json.loads(i or "[]"), "data": json.loads(d or "{}")} for ts, k, i, d in rows]

    def clear(self):
        with self._lock:
            self._db.execute("DELETE FROM minutes")
            self._db.execute("DELETE FROM events")
            self._db.commit()

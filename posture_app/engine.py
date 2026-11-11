"""The processing loop: frames -> landmarks -> features -> smoothing -> evaluation -> feedback."""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import asdict

import cv2
import numpy as np

from .calibration import Baseline, BaselineStore, CalibrationError, Calibrator
from .config import Settings
from .evaluator import Assessment, Evaluator, Event
from .filters import FeatureSmoother
from .landmarks import L_EYE, L_SHOULDER, R_EYE, R_SHOULDER, LandmarkEngine, Observation
from .metrics import compute_features
from . import notifier

log = logging.getLogger(__name__)

PROFILE_SWITCH_HOLD_S = 10.0
SKILL_REFRESH_S = 600.0


def skill_from_history(good_share: float, hours: float) -> float:
    """Map recent good-posture share to 0..1 for faded feedback (needs >= 2 h of data)."""
    if hours < 2.0:
        return 0.0
    return max(0.0, min(1.0, (good_share - 0.70) / 0.25))


class Engine:
    def __init__(self, settings: Settings, source, store: BaselineStore | None = None, history=None,
                 notify: bool = True, on_event=None):
        self.settings = settings
        self.source = source
        self.store = store or BaselineStore()
        self.history = history
        self.notify_enabled = notify
        self.on_event = on_event
        self.landmarks = LandmarkEngine(settings.pose_model, settings.max_width)
        self.smoother = FeatureSmoother()
        self.evaluator = Evaluator(settings, self.store.profiles.get(settings.active_profile))
        self.calibrator: Calibrator | None = None
        self.last_calibration: dict | None = None
        self.paused = False
        self.events: list[dict] = []
        self.timeline: list[dict] = []  # filled when record_timeline=True
        self.record_timeline = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._snapshot: dict = {"state": "starting"}
        self._preview_jpeg: bytes | None = None
        self._preview_wanted_until = 0.0
        self._profile_candidate: tuple[str, float] | None = None
        self._skill_t = -1e9
        self._proc_ms = 0.0
        self._fps = 0.0
        self._last_t: float | None = None
        self._last_wall: float | None = None

    # -- lifecycle -----------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self.run, daemon=True, name="engine")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self.source.close()
        self.landmarks.close()
        if self.history:
            self.history.flush()

    def run(self):
        last_seq, next_t = -1, 0.0
        interval = 1.0 / max(0.5, self.settings.analysis_fps)
        while not self._stop.is_set():
            if self.paused:
                with self._lock:
                    self._snapshot = {**self._snapshot, "state": "paused"}
                time.sleep(0.2)
                continue
            if self.source.is_file:
                got = self.source.read(last_seq, skip_to=next_t)
            else:
                got = self.source.read(last_seq)
            if got is None:
                if self.source.is_file and getattr(self.source, "finished", False):
                    break
                with self._lock:
                    self._snapshot = {**self._snapshot, "state": "no_camera"}
                continue
            last_seq, frame, t = got
            if not self.source.is_file and t < next_t:
                time.sleep(min(0.02, next_t - t))
                continue
            next_t = t + interval
            self.step(frame, t)
        with self._lock:
            self._snapshot = {**self._snapshot, "state": "stopped"}

    # -- commands --------------------------------------------------------------
    def start_calibration(self, profile: str, countdown: float = 3.0, duration: float = 5.0):
        profile = (profile or "sitting").strip()[:32] or "sitting"
        with self._lock:
            # start_t is set from the next frame's timestamp
            self.calibrator = Calibrator(profile, None, countdown, duration)
            self.last_calibration = None

    def cancel_calibration(self):
        with self._lock:
            self.calibrator = None

    def select_profile(self, profile: str):
        if profile in self.store.profiles:
            self.settings.active_profile = profile
            self.evaluator.set_baseline(self.store.profiles[profile])
            if self.history:
                self.history.profile = profile

    def delete_profile(self, profile: str):
        self.store.delete(profile)
        if self.settings.active_profile == profile:
            remaining = list(self.store.profiles)
            if remaining:
                self.select_profile(remaining[0])
            else:
                self.evaluator.set_baseline(None)

    def apply_settings(self, settings: Settings):
        self.settings = settings
        self.evaluator.settings = settings

    def want_preview(self):
        self._preview_wanted_until = time.monotonic() + 3.0

    def preview_jpeg(self) -> bytes | None:
        self.want_preview()
        return self._preview_jpeg

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._snapshot)

    # -- per-frame processing ----------------------------------------------------
    def step(self, frame: np.ndarray, t: float) -> Assessment:
        t0 = time.perf_counter()
        obs = self.landmarks.process(frame, t)
        raw = compute_features(obs)
        feats = self.smoother(raw, t)
        dt = 0.0 if self._last_t is None else max(0.0, min(2.0, t - self._last_t))
        self._last_t = t

        calib_status = self._calibration_step(feats, t, obs)
        self._auto_profile(feats, t)

        if self.history and not self.source.is_file and time.monotonic() - self._skill_t > SKILL_REFRESH_S:
            self._skill_t = time.monotonic()
            self.evaluator.skill = skill_from_history(*self.history.good_share(7.0))

        a = self.evaluator.update(feats if obs.has_person else None, t)
        if self.calibrator is not None:
            a.events = []  # no nagging mid-calibration
        for ev in a.events:
            self._handle_event(ev)

        wall = time.time()
        if self.history and not self.source.is_file:
            self.history.add_sample(wall, dt, a.present, a.score, a.bad, a.issues)

        self._proc_ms = 0.9 * self._proc_ms + 0.1 * (time.perf_counter() - t0) * 1000 if self._proc_ms else (time.perf_counter() - t0) * 1000
        if self._last_wall is not None:
            inst = 1.0 / max(1e-3, time.monotonic() - self._last_wall)
            self._fps = 0.9 * self._fps + 0.1 * inst if self._fps else inst
        self._last_wall = time.monotonic()

        snap = self._build_snapshot(a, feats, calib_status)
        if self.record_timeline:
            self.timeline.append({"t": round(t, 3), "score": a.score, "bad": a.bad, "issues": a.issues,
                                  "features": {k: v for k, v in feats.items() if v is not None},
                                  "rules": {r.key: r.deviation for r in a.rules}})
        if time.monotonic() < self._preview_wanted_until:
            self._preview_jpeg = self._render_preview(frame, obs, a)
        with self._lock:
            self._snapshot = snap
        return a

    def _calibration_step(self, feats, t, obs) -> dict | None:
        cal = self.calibrator
        if cal is None:
            return self.last_calibration
        if cal.start_t is None:
            cal.start_t = t
        phase = cal.phase(t)
        if phase == "capturing" and obs.has_person:
            cal.add(feats, t)
        elif phase == "capturing":
            cal.frames_seen += 1
        phase_remaining = cal.start_t + cal.countdown - t if phase == "countdown" else cal.remaining(t)
        status = {"phase": phase, "profile": cal.profile, "progress": round(cal.progress(t), 2),
                  "remaining": round(cal.remaining(t), 1), "phase_remaining": round(max(0.0, phase_remaining), 1)}
        if phase == "done":
            try:
                b = cal.finish()
                self.store.set(b)
                self.select_profile(b.profile)
                status = {"phase": "ok", "profile": b.profile, "warnings": b.warnings,
                          "rules": b.available_rules}
            except CalibrationError as e:
                status = {"phase": "failed", "profile": cal.profile, "error": str(e)}
            with self._lock:
                self.calibrator = None
                self.last_calibration = status
        return status

    def _auto_profile(self, feats, t):
        if not self.settings.auto_profile or len(self.store.profiles) < 2 or self.calibrator is not None:
            return
        current = self.store.profiles.get(self.settings.active_profile)
        best, best_d = self.store.nearest(feats)
        if best is None or current is None or best == self.settings.active_profile or not math.isfinite(best_d):
            self._profile_candidate = None
            return
        from .calibration import profile_distance

        cur_d = profile_distance(feats, current)
        if best_d < 0.5 * cur_d and cur_d > 0.15:
            if self._profile_candidate is None or self._profile_candidate[0] != best:
                self._profile_candidate = (best, t)
            elif t - self._profile_candidate[1] >= PROFILE_SWITCH_HOLD_S:
                log.info("auto-switching profile to %s", best)
                self.select_profile(best)
                self._profile_candidate = None
        else:
            self._profile_candidate = None

    def _handle_event(self, ev: Event):
        d = {"kind": ev.kind, "t": ev.t, "wall": time.time(), "title": ev.title, "message": ev.message,
             "issues": ev.issues, "data": ev.data}
        self.events.append(d)
        del self.events[:-50]
        if self.history and not self.source.is_file:
            self.history.add_event(d["wall"], ev.kind, ev.issues, ev.data)
        if self.notify_enabled and self.settings.desktop_notifications and ev.kind != "corrected":
            notifier.notify(ev.title, ev.message, urgent=ev.kind == "posture_alert")
        if self.on_event:
            self.on_event(d)

    def _build_snapshot(self, a: Assessment, feats, calib_status) -> dict:
        b = self.evaluator.baseline
        return {
            "state": "running",
            "t": round(a.t, 2),
            "present": a.present,
            "calibrated": a.calibrated,
            "profile": self.settings.active_profile,
            "profiles": {k: {"created": v.created, "rules": v.available_rules, "warnings": v.warnings}
                         for k, v in self.store.profiles.items()},
            "score": None if a.score is None else round(a.score),
            "bad": a.bad,
            "bad_for": round(a.bad_for, 1),
            "alert_delay": round(a.alert_delay, 1),
            "issues": a.issues,
            "rules": [asdict(r) for r in a.rules],
            "still_for": round(a.still_for),
            "present_for": round(a.present_for),
            "calibration": calib_status,
            "features": {k: (None if v is None else round(v, 4)) for k, v in feats.items()},
            "baseline": None if b is None else {k: round(v, 4) for k, v in b.values.items()},
            "events": self.events[-10:],
            "perf": {"proc_ms": round(self._proc_ms, 1), "fps": round(self._fps, 1)},
            "skill": round(self.evaluator.skill, 2),
            "source": "video" if self.source.is_file else "camera",
        }

    def _render_preview(self, frame, obs: Observation, a: Assessment) -> bytes | None:
        img = self.landmarks.prepare(frame).copy()
        color = (80, 80, 80) if not a.present else ((60, 60, 230) if a.bad else (90, 200, 90))
        p = obs.pose

        def pt(i):
            q = p.get(i)
            return None if q is None or q.visibility < 0.5 else (int(q.x), int(q.y))

        ls, rs, le, re_ = pt(L_SHOULDER), pt(R_SHOULDER), pt(L_EYE), pt(R_EYE)
        if ls and rs:
            cv2.line(img, ls, rs, color, 3, cv2.LINE_AA)
            sm = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
            if le and re_:
                em = ((le[0] + re_[0]) // 2, (le[1] + re_[1]) // 2)
                cv2.line(img, sm, em, color, 2, cv2.LINE_AA)
                cv2.circle(img, em, 5, color, -1, cv2.LINE_AA)
            for q in (ls, rs, sm):
                cv2.circle(img, q, 6, color, -1, cv2.LINE_AA)
        if obs.face:
            xs = [q.x for q in obs.face.values()]
            ys = [q.y for q in obs.face.values()]
            cv2.rectangle(img, (int(min(xs)) - 10, int(min(ys)) - 10), (int(max(xs)) + 10, int(max(ys)) + 10),
                          color, 1, cv2.LINE_AA)
        if self.settings.mirror_preview and not self.source.is_file:
            img = cv2.flip(img, 1)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes() if ok else None


def analyze_video(path: str, settings: Settings | None = None, calibrate: tuple[float, float] | None = (0.5, 5.0),
                  baseline: Baseline | None = None, fps: float | None = None) -> dict:
    """Run the full pipeline over a video file as fast as possible (for tests/evaluation).

    The baseline comes from the ``calibrate`` window (start, end seconds) of the
    same video, or from ``baseline`` if given. No history or notifications.
    """
    from .sources import VideoSource
    import tempfile
    from pathlib import Path

    settings = settings or Settings()
    if fps:
        settings.analysis_fps = fps
    tmp = Path(tempfile.mkdtemp(prefix="posture-analyze-"))
    store = BaselineStore(tmp / "baselines.json")
    if baseline is not None:
        store.profiles[baseline.profile] = baseline
        settings.active_profile = baseline.profile
    src = VideoSource(path, realtime=False)
    eng = Engine(settings, src, store=store, history=None, notify=False)
    eng.record_timeline = True
    if baseline is None and calibrate is not None:
        start, end = calibrate
        eng.calibrator = Calibrator("video", start, countdown=0.0, duration=end - start)
    eng.run()
    eng.landmarks.close()
    src.close()
    return {
        "path": path,
        "calibration": eng.last_calibration,
        "baseline": store.profiles.get(eng.settings.active_profile),
        "timeline": eng.timeline,
        "events": eng.events,
        "perf_ms": eng._proc_ms,
    }

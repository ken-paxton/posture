"""Frame sources: a live camera or a pre-recorded video file."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np


class CameraSource:
    """Reads frames on a background thread and keeps only the latest one (no lag build-up)."""

    is_file = False

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._frame: np.ndarray | None = None
        self._t = 0.0
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="camera")
        self._thread.start()

    def _run(self):
        fails = 0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                fails += 1
                time.sleep(0.05 if fails < 40 else 0.5)
                continue
            fails = 0
            with self._lock:
                self._frame, self._t, self._seq = frame, time.monotonic(), self._seq + 1

    def read(self, last_seq: int = -1, timeout: float = 2.0):
        """Return (seq, frame, t) for a frame newer than ``last_seq``, or None on timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self._stop.is_set():
            with self._lock:
                if self._frame is not None and self._seq != last_seq:
                    return self._seq, self._frame, self._t
            time.sleep(0.005)
        return None

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.cap.release()


class VideoSource:
    """Plays a video file. ``realtime`` paces playback to the file's frame rate.

    Timestamps come from the frame index, so analysis is deterministic.
    """

    is_file = True

    def __init__(self, path: str, realtime: bool = True, loop: bool = False):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video {path}.")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.realtime = realtime
        self.loop = loop
        self._idx = 0
        self._loops = 0
        self._start_wall: float | None = None
        self.finished = False

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0

    def read(self, last_seq: int = -1, timeout: float = 2.0, skip_to: float | None = None):
        """Next frame (optionally skipping ahead to timestamp ``skip_to``)."""
        if self._start_wall is None:
            self._start_wall = time.monotonic()
        if skip_to is not None:
            target = int(skip_to * self.fps) - self._loops * self.frame_count
            while self._idx < target:
                if not self.cap.grab():
                    break
                self._idx += 1
        ok, frame = self.cap.read()
        if not ok:
            if self.loop and self._idx > 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._loops += 1
                self._idx = 0
                ok, frame = self.cap.read()
            if not ok:
                self.finished = True
                return None
        t = (self._loops * self.frame_count + self._idx) / self.fps
        self._idx += 1
        if self.realtime:
            delay = self._start_wall + t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        return self._loops * self.frame_count + self._idx, frame, t

    def close(self):
        self.cap.release()

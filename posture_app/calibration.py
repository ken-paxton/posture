"""Baseline ("your good posture") capture.

The user settles into a relaxed, supported, neutral posture (see GUIDANCE) and
holds it for a few seconds while raw features are recorded. The baseline is
each feature's median, which is robust to blinks, detection glitches and small
fidgets. The per-rule measurement noise is the robust spread of each sample's
deviation from that median. Alert thresholds are widened by this noise, so a
noisy camera or lighting setup doesn't produce false alarms.

The baseline is only changed when the user explicitly recalibrates. It is
never adapted automatically from live data, because that would let a slowly
worsening posture become the new normal.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .config import data_dir
from .metrics import RAW_FEATURES
from .rules import RULES

# Shown to the user before and during capture.
GUIDANCE = [
    "Sit all the way back so the chair supports your lower back; feet flat on the floor.",
    "Let your shoulders relax down. Don't pull them back or puff your chest out.",
    "Head balanced over your shoulders, chin level, eyes looking at the middle of the screen.",
    "Screen at arm's length, with its top line at or slightly below eye level.",
    "Breathe normally. It should feel easy to hold, not effortful.",
]

MIN_COVERAGE = 0.6  # fraction of frames in which a feature must be measurable
NOISE_K = 3.0  # thresholds are at least NOISE_K x the calibration noise

# If a calibration sample moves more than this, the user wasn't holding still.
MAX_SPREAD = {"neck_ratio": 0.05, "head_pitch": 5.0, "head_roll": 4.0, "head_distance": 0.06, "shoulder_tilt": 3.0}


@dataclass
class Baseline:
    profile: str
    created: float
    values: dict[str, float]
    noise: dict[str, float]  # per rule, in deviation units
    available_rules: list[str]
    warnings: list[str] = field(default_factory=list)
    samples: int = 0

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Baseline":
        return cls(**d)


class CalibrationError(Exception):
    pass


def build_baseline(profile: str, samples: list[dict[str, float | None]]) -> Baseline:
    if len(samples) < 10:
        raise CalibrationError("Not enough frames captured. Make sure you're in view of the camera.")

    values: dict[str, float] = {}
    for k in RAW_FEATURES:
        v = np.array([s[k] for s in samples if s.get(k) is not None], float)
        if len(v) >= MIN_COVERAGE * len(samples):
            values[k] = float(np.median(v))

    if "neck_ratio" not in values and "head_pitch" not in values:
        raise CalibrationError(
            "Couldn't see your face or shoulders clearly. Face the camera with your head and both "
            "shoulders in view, in even lighting."
        )

    # Stillness check
    moved = []
    for k, limit in MAX_SPREAD.items():
        if k not in values:
            continue
        v = np.array([s[k] for s in samples if s.get(k) is not None], float)
        spread = 1.4826 * float(np.median(np.abs(v - values[k])))
        if k == "head_distance":
            spread /= values[k]
        if spread > limit:
            moved.append(k)
    if len(moved) >= 2:
        raise CalibrationError("You moved quite a bit during capture. Settle in and hold still for a few seconds.")

    noise: dict[str, float] = {}
    available: list[str] = []
    for rule in RULES:
        if not all(k in values for k in rule.needs):
            continue
        devs = []
        for s in samples:
            if all(s.get(k) is not None for k in rule.needs):
                try:
                    devs.append(rule.deviation(s, values))
                except ZeroDivisionError:
                    pass
        if len(devs) < MIN_COVERAGE * len(samples):
            continue
        d = np.array(devs, float)
        noise[rule.key] = float(1.4826 * np.median(np.abs(d - np.median(d))))
        available.append(rule.key)

    warnings = []
    if "neck_ratio" not in values:
        warnings.append(
            "Your shoulders aren't fully in view, so slump and forward-head checks are off. "
            "Tilt the screen or camera down slightly, or sit a little further back."
        )
    if abs(values.get("head_yaw", 0.0)) > 25:
        warnings.append("The camera is well off to one side. Centred, head-on cameras give the most reliable results.")
    if abs(values.get("head_roll", 0.0)) > 8:
        warnings.append("Your head looked tilted during calibration. Recalibrate with your head level if that wasn't intended.")
    if abs(values.get("shoulder_tilt", 0.0)) > 6:
        warnings.append("Your shoulders looked uneven. This can be the camera angle, but check that you're sitting evenly.")
    if values.get("head_pitch", 0.0) > 20:
        warnings.append(
            "Your head was tipped well down relative to the camera. If you're looking down at a laptop, "
            "consider raising the screen (with a separate keyboard) so its top is near eye level."
        )
    if "head_distance" in values and values["head_distance"] < 35:
        warnings.append("You're quite close to the camera. Arm's length (50-100 cm) from the screen is recommended.")
    if moved:
        warnings.append("There was some movement during capture; consider recalibrating while holding still.")

    return Baseline(
        profile=profile,
        created=time.time(),
        values=values,
        noise=noise,
        available_rules=available,
        warnings=warnings,
        samples=len(samples),
    )


class Calibrator:
    """Collects features for ``duration`` seconds after a ``countdown``."""

    def __init__(self, profile: str, start_t: float | None, countdown: float = 3.0, duration: float = 5.0):
        self.profile = profile
        self.start_t = start_t
        self.countdown = countdown
        self.duration = duration
        self.samples: list[dict[str, float | None]] = []
        self.frames_seen = 0

    def phase(self, t: float) -> str:
        if t < self.start_t + self.countdown:
            return "countdown"
        if t < self.start_t + self.countdown + self.duration:
            return "capturing"
        return "done"

    def progress(self, t: float) -> float:
        return min(1.0, max(0.0, (t - self.start_t - self.countdown) / self.duration))

    def remaining(self, t: float) -> float:
        return max(0.0, self.start_t + self.countdown + self.duration - t)

    def add(self, features: dict[str, float | None], t: float) -> None:
        if self.phase(t) == "capturing":
            self.frames_seen += 1
            self.samples.append(features)

    def finish(self) -> Baseline:
        b = build_baseline(self.profile, self.samples)
        if self.frames_seen and len(self.samples) / self.frames_seen < 0.5:
            b.warnings.append("You were only partly visible during capture.")
        return b


class BaselineStore:
    """Profiles (e.g. sitting / standing) persisted as JSON in the data directory."""

    def __init__(self, path=None):
        self.path = path or (data_dir() / "baselines.json")
        self.profiles: dict[str, Baseline] = {}
        try:
            raw = json.loads(self.path.read_text())
            self.profiles = {k: Baseline.from_json(v) for k, v in raw.items()}
        except (OSError, ValueError, TypeError):
            pass

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: v.to_json() for k, v in self.profiles.items()}, indent=2))
        tmp.replace(self.path)

    def set(self, baseline: Baseline) -> None:
        self.profiles[baseline.profile] = baseline
        self.save()

    def delete(self, profile: str) -> None:
        self.profiles.pop(profile, None)
        self.save()

    def nearest(self, features: dict[str, float | None]) -> tuple[str | None, float]:
        """Profile whose framing (eye height, apparent shoulder width) best matches."""
        best, best_d = None, math.inf
        for name, b in self.profiles.items():
            d = profile_distance(features, b)
            if d < best_d:
                best, best_d = name, d
        return best, best_d


def profile_distance(features: dict[str, float | None], b: Baseline) -> float:
    terms = []
    if features.get("eye_y") is not None and "eye_y" in b.values:
        terms.append(abs(features["eye_y"] - b.values["eye_y"]) * 3.0)
    if features.get("shoulder_width") is not None and "shoulder_width" in b.values:
        terms.append(abs(math.log(features["shoulder_width"] / b.values["shoulder_width"])))
    elif features.get("head_distance") is not None and "head_distance" in b.values:
        terms.append(abs(math.log(features["head_distance"] / b.values["head_distance"])))
    return sum(terms) / len(terms) if terms else math.inf

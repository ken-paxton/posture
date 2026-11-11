"""Posture rules: how each live feature is compared with the calibrated baseline.

Design principles, drawn from the research summary in RESEARCH.md:

* Compare against the user's own relaxed, supported, neutral posture, not a
  population "ideal". There is no single correct posture, and spinal curves
  vary naturally between people (Slater et al., JOSPT 2019; Korakakis et al. 2019).
* Only flag deviations in the harmful direction. For example, sitting taller
  than baseline never triggers a slump alert. This avoids nudging people toward
  rigid "military" upright sitting, which raises thoracic muscle activity
  (Caneiro et al., Manual Therapy 2010).
* Thresholds sit where ergonomic standards start to treat a deviation as
  non-neutral: head flexion more than ~15-20 deg (RULA neck score 10-20 deg;
  ISO 11226 neutral range 0-25 deg). They are also widened by the measurement
  noise seen during calibration, so jitter doesn't cause false alarms.
* Only sustained deviations count (see Evaluator). Brief excursions are normal
  and healthy; ISO 11226 treats postures as "static" only after 4 s.
* Messages coach toward relaxed, supported positions and setup fixes rather
  than "pull your shoulders back" style over-correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

Features = dict[str, "float | None"]


@dataclass(frozen=True)
class Rule:
    key: str
    label: str
    unit: str
    threshold: float  # deviation at which the posture counts as out of range
    needs: tuple[str, ...]  # raw features required (live and in baseline)
    deviation: Callable[[Features, Features], float]  # (live, baseline) -> signed deviation, + = worse
    message: str
    face_metric: bool = False  # unreliable when the head is turned far from baseline
    alert_default: bool = True


def _rel_drop(key: str):
    return lambda live, base: (base[key] - live[key]) / base[key]


RULES: tuple[Rule, ...] = (
    Rule(
        key="slump",
        label="Head dropped / slumping",
        unit="%",
        # Frontal pose models partly infer shoulder position from the head (a body
        # prior), so they under-report head drop by roughly half. 10% measured
        # corresponds to roughly a 3-5 cm real drop at typical desk distances.
        threshold=0.10,
        needs=("neck_ratio",),
        deviation=_rel_drop("neck_ratio"),
        message=(
            "Your head has sunk toward your shoulders. Sit back so the chair supports "
            "your lower back, let your shoulders relax down, and grow tall through the crown of your head."
        ),
    ),
    Rule(
        key="forward_head",
        label="Head forward of shoulders",
        unit="%",
        threshold=0.10,
        needs=("head_distance", "shoulder_width"),
        # Face magnification relative to shoulder magnification. If the whole
        # trunk moves toward the camera both scale together (ratio ~1); if only
        # the head juts forward, the face grows more than the shoulders.
        deviation=lambda live, base: (base["head_distance"] / live["head_distance"])
        / (live["shoulder_width"] / base["shoulder_width"])
        - 1.0,
        face_metric=True,
        message=(
            "Your head has drifted forward of your shoulders. Gently glide your chin straight back "
            "(no need to tuck it down) and let your upper back rest against the chair."
        ),
    ),
    Rule(
        key="lean_in",
        label="Leaning toward screen",
        unit="%",
        threshold=0.15,
        needs=("head_distance",),
        deviation=lambda live, base: base["head_distance"] / live["head_distance"] - 1.0,
        face_metric=True,
        message=(
            "You've moved much closer to the screen. Sit back into your chair. If you're leaning in "
            "to read, enlarge the text or bring the screen closer instead."
        ),
    ),
    Rule(
        key="head_flexion",
        label="Looking down",
        unit="°",
        threshold=15.0,
        needs=("head_pitch",),
        deviation=lambda live, base: live["head_pitch"] - base["head_pitch"],
        face_metric=True,
        message=(
            "Your head has been tipped down for a while. Raise the screen so its top line is at or "
            "slightly below eye level, or lift your gaze and let your chin come level."
        ),
    ),
    Rule(
        key="head_extension",
        label="Chin up / poking",
        unit="°",
        threshold=15.0,
        needs=("head_pitch",),
        deviation=lambda live, base: base["head_pitch"] - live["head_pitch"],
        face_metric=True,
        message=(
            "Your chin is tilted up. Check that the screen isn't too high (or that you aren't "
            "reading through the bottom of bifocals), then let your chin settle level."
        ),
    ),
    Rule(
        key="head_tilt",
        label="Head tilted sideways",
        unit="°",
        threshold=10.0,
        needs=("head_roll",),
        deviation=lambda live, base: abs(live["head_roll"] - base["head_roll"]),
        face_metric=True,
        message="Your head is tilted to one side. Level your head. Is something (a phone, a second screen) pulling it over?",
    ),
    Rule(
        key="shoulder_tilt",
        label="Leaning to one side",
        unit="°",
        threshold=7.0,
        needs=("shoulder_tilt",),
        deviation=lambda live, base: abs(live["shoulder_tilt"] - base["shoulder_tilt"]),
        message=(
            "You're leaning to one side. Spread your weight evenly across both hips and check "
            "that armrests and your mouse aren't pulling one shoulder up or out."
        ),
    ),
    Rule(
        key="trunk_lean",
        label="Trunk leaning sideways",
        unit="°",
        threshold=10.0,
        needs=("trunk_lean",),
        deviation=lambda live, base: abs(live["trunk_lean"] - base["trunk_lean"]),
        message="Your trunk is leaning sideways. Re-centre over both hips.",
    ),
    Rule(
        key="ear_depth",
        label="Head forward (depth, experimental)",
        unit="cm",
        threshold=0.04,
        needs=("ear_depth",),
        deviation=lambda live, base: base["ear_depth"] - live["ear_depth"],
        alert_default=False,
        message="Your head appears to have moved forward of your shoulders.",
    ),
)

RULES_BY_KEY = {r.key: r for r in RULES}

# Deviation is reported in these display units.
DISPLAY_SCALE = {"%": 100.0, "°": 1.0, "cm": 100.0}

# Messages for the movement-variety reminders.
MOVEMENT_MESSAGE = (
    "You've held much the same position for {minutes} minutes. Change it up: shift position, "
    "roll your shoulders, or stand and stretch for 30 seconds. Variety matters more than any single 'perfect' posture."
)
BREAK_MESSAGE = (
    "You've been at the screen for {minutes} minutes. A short break (stand, walk, look into "
    "the distance) helps reduce discomfort and eye strain."
)

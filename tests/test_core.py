"""Fast unit tests with synthetic landmarks / feature streams (no models needed)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from posture_app.calibration import Baseline, CalibrationError, build_baseline
from posture_app.config import Settings
from posture_app.evaluator import Evaluator
from posture_app.filters import OneEuroFilter
from posture_app.landmarks import (
    L_EAR, L_EYE, L_SHOULDER, R_EAR, R_EYE, R_SHOULDER, HeadPose, Observation, Point, head_pose_from_matrix,
)
from posture_app.metrics import compute_features


# -- helpers --------------------------------------------------------------------

def make_obs(eye_drop=0.0, roll_deg=0.0, mirror=False, scale=1.0, head=None):
    """Frontal person: shoulders 200px wide at y=400, eyes 130px above (scaled about image centre)."""
    pts = {
        R_SHOULDER: (220, 400), L_SHOULDER: (420, 400),
        R_EYE: (295, 270 + eye_drop), L_EYE: (345, 270 + eye_drop),
        R_EAR: (270, 280 + eye_drop), L_EAR: (370, 280 + eye_drop),
    }
    c = np.array([320.0, 240.0])
    r = math.radians(roll_deg)
    rot = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    obs = Observation(timestamp=0.0, width=640, height=480)
    for i, (x, y) in pts.items():
        p = c + scale * rot @ (np.array([x, y]) - c)
        if mirror:
            p[0] = 640 - p[0]
        obs.pose[i] = Point(float(p[0]), float(p[1]), 0.0, 0.99)
    obs.head = head
    return obs


BASE_FEATS = {
    "neck_ratio": 0.65, "shoulder_width": 200.0, "head_distance": 60.0, "head_pitch": 5.0, "head_roll": 0.0,
    "head_yaw": 0.0, "shoulder_tilt": 0.0, "trunk_lean": None, "ear_depth": -0.07, "eye_y": 0.55,
}


def baseline(**noise):
    rules = ["slump", "forward_head", "lean_in", "head_flexion", "head_extension", "head_tilt", "shoulder_tilt"]
    return Baseline("sitting", 0.0, {k: v for k, v in BASE_FEATS.items() if v is not None},
                    {r: noise.get(r, 0.0) for r in rules}, rules)


def feats(**over):
    f = dict(BASE_FEATS)
    f.update(over)
    return f


def run(ev, stream, t0=0.0, dt=0.1):
    """Feed (duration_s, features) segments; return all events and final assessment."""
    t, events, a = t0, [], None
    for dur, f in stream:
        for _ in range(int(round(dur / dt))):
            a = ev.update(f, t)
            events += a.events
            t += dt
    return events, a, t


# -- filters ------------------------------------------------------------------------

def test_one_euro_reduces_jitter_and_tracks_steps():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=0.4, beta=0.02)
    xs = 10 + rng.normal(0, 1.0, 300)
    ys = [f(x, i / 10) for i, x in enumerate(xs)]
    assert np.std(ys[50:]) < 0.5 * np.std(xs[50:])
    # A sustained step is followed within a few seconds.
    for i in range(300, 400):
        y = f(20.0, i / 10)
    assert abs(y - 20.0) < 0.5


# -- metrics ------------------------------------------------------------------------

def test_neck_ratio_falls_when_head_drops():
    up = compute_features(make_obs())
    down = compute_features(make_obs(eye_drop=30))
    assert up["neck_ratio"] == pytest.approx(130 / 200, rel=1e-3)
    assert down["neck_ratio"] < up["neck_ratio"]


@pytest.mark.parametrize("kw", [dict(roll_deg=8), dict(mirror=True), dict(scale=0.7)])
def test_neck_ratio_invariant_to_camera_roll_mirror_and_distance(kw):
    ref = compute_features(make_obs())["neck_ratio"]
    assert compute_features(make_obs(**kw))["neck_ratio"] == pytest.approx(ref, rel=1e-3)


def test_shoulder_tilt_measures_roll():
    assert compute_features(make_obs(roll_deg=6))["shoulder_tilt"] == pytest.approx(6, abs=0.01)


def test_low_visibility_shoulders_are_ignored():
    obs = make_obs()
    obs.pose[L_SHOULDER].visibility = 0.2
    f = compute_features(obs)
    assert f["neck_ratio"] is None and f["shoulder_width"] is None


def test_head_pose_decomposition():
    def rx(a):
        a = math.radians(a)
        return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])

    m = np.eye(4)
    m[:3, :3] = rx(20)
    m[2, 3] = -55.0
    hp = head_pose_from_matrix(m)
    assert hp.pitch == pytest.approx(20, abs=1e-6)
    assert hp.distance == pytest.approx(55.0)
    assert abs(hp.yaw) < 1e-6 and abs(hp.roll) < 1e-6


# -- calibration --------------------------------------------------------------------

def test_calibration_median_and_noise():
    rng = np.random.default_rng(1)
    samples = [feats(neck_ratio=0.65 + rng.normal(0, 0.005), head_pitch=5 + rng.normal(0, 0.5)) for _ in range(50)]
    b = build_baseline("sitting", samples)
    assert b.values["neck_ratio"] == pytest.approx(0.65, abs=0.005)
    assert 0 < b.noise["slump"] < 0.02
    assert "slump" in b.available_rules and "trunk_lean" not in b.available_rules


def test_calibration_rejects_absent_and_moving_user():
    with pytest.raises(CalibrationError):
        build_baseline("sitting", [feats()] * 5)
    with pytest.raises(CalibrationError):
        build_baseline("sitting", [{k: None for k in BASE_FEATS}] * 40)
    rng = np.random.default_rng(2)
    moving = [feats(neck_ratio=0.65 + rng.normal(0, 0.1), head_pitch=5 + rng.normal(0, 15)) for _ in range(50)]
    with pytest.raises(CalibrationError):
        build_baseline("sitting", moving)


def test_calibration_warns_when_shoulders_out_of_view():
    b = build_baseline("sitting", [feats(neck_ratio=None, shoulder_width=None, shoulder_tilt=None)] * 40)
    assert any("shoulders" in w for w in b.warnings)
    assert "slump" not in b.available_rules and "head_flexion" in b.available_rules


# -- evaluator ----------------------------------------------------------------------

def settings(**kw):
    s = Settings(alert_delay_s=30, alert_cooldown_s=180, fade_feedback=False, movement_reminder_min=0,
                 break_reminder_min=0)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_good_posture_never_alerts():
    events, a, _ = run(Evaluator(settings(), baseline()), [(120, feats())])
    assert not events and a.score == 100 and not a.bad


def test_sustained_slump_alerts_after_delay_only():
    ev = Evaluator(settings(), baseline())
    slump = feats(neck_ratio=0.65 * 0.8)  # 20% drop
    events, a, _ = run(ev, [(5, feats()), (25, slump)])
    assert not events and a.bad and "slump" in a.issues
    events, _, _ = run(ev, [(6, slump)], t0=30)
    assert [e.kind for e in events] == ["posture_alert"] and events[0].issues[0] == "slump"


def test_brief_deviations_do_not_alert():
    stream = [(10, feats()), (8, feats(neck_ratio=0.5)), (10, feats())] * 6
    events, _, _ = run(Evaluator(settings(), baseline()), stream)
    assert not events


def test_grace_period_keeps_timer_running():
    slump = feats(neck_ratio=0.5)
    stream = [(20, slump), (2, feats()), (12, slump)]
    events, _, _ = run(Evaluator(settings(grace_s=3), baseline()), stream)
    assert any(e.kind == "posture_alert" for e in events)


def test_cooldown_limits_repeat_alerts_and_correction_event():
    slump = feats(neck_ratio=0.5)
    events, _, _ = run(Evaluator(settings(), baseline()), [(300, slump), (10, feats())])
    kinds = [e.kind for e in events]
    assert kinds.count("posture_alert") == 2  # at ~30 s and ~210 s
    assert kinds[-1] == "corrected"


def test_sitting_taller_or_leaning_back_is_not_penalised():
    tall = feats(neck_ratio=0.75, head_distance=70.0)
    events, a, _ = run(Evaluator(settings(), baseline()), [(120, tall)])
    assert not events and a.score == 100


def test_forward_head_vs_whole_body_lean():
    # Whole trunk moves 15% closer: face and shoulders scale together -> lean_in only.
    lean = feats(head_distance=60 / 1.2, shoulder_width=200 * 1.2)
    _, a, _ = run(Evaluator(settings(), baseline()), [(5, lean)])
    assert "lean_in" in a.issues and "forward_head" not in a.issues
    # Head alone juts forward: face grows, shoulders don't -> forward_head.
    fh = feats(head_distance=60 / 1.15)
    _, a, _ = run(Evaluator(settings(), baseline()), [(5, fh)])
    assert "forward_head" in a.issues


def test_noise_widens_thresholds():
    ev = Evaluator(settings(), baseline(head_flexion=7.0))
    assert ev.thresholds()["head_flexion"] == pytest.approx(21.0)
    _, a, _ = run(ev, [(5, feats(head_pitch=5 + 18))])
    assert "head_flexion" not in a.issues


def test_sensitivity_presets_scale_thresholds():
    gentle = Evaluator(settings(sensitivity="gentle"), baseline()).thresholds()
    strict = Evaluator(settings(sensitivity="strict"), baseline()).thresholds()
    assert gentle["head_flexion"] > 15 > strict["head_flexion"]


def test_face_rules_skipped_when_head_turned():
    looking_away = feats(head_yaw=50.0, head_pitch=40.0)
    _, a, _ = run(Evaluator(settings(), baseline()), [(5, looking_away)])
    assert "head_flexion" not in a.issues


def test_absence_resets_bad_timer():
    slump = feats(neck_ratio=0.5)
    events, a, _ = run(Evaluator(settings(), baseline()), [(20, slump), (10, None), (20, slump)])
    assert not [e for e in events if e.kind == "posture_alert"]


def test_movement_reminder_after_long_stillness_and_reset_by_movement():
    s = settings(movement_reminder_min=10)
    events, _, _ = run(Evaluator(s, baseline()), [(11 * 60, feats())], dt=1.0)
    assert [e.kind for e in events] == ["movement"]
    moving = [(4 * 60, feats()), (5, feats(neck_ratio=0.75)), (4 * 60, feats(neck_ratio=0.75)), (5, feats())]
    events, _, _ = run(Evaluator(s, baseline()), moving * 2, dt=1.0)
    assert not events


def test_break_reminder_and_away_resets_it():
    s = settings(break_reminder_min=20, away_reset_s=120)
    events, _, _ = run(Evaluator(s, baseline()), [(21 * 60, feats())], dt=1.0)
    assert [e.kind for e in events] == ["break"]
    events, _, _ = run(Evaluator(s, baseline()), [(15 * 60, feats()), (180, None), (15 * 60, feats())], dt=1.0)
    assert not events


def test_faded_feedback_extends_delay():
    ev = Evaluator(settings(fade_feedback=True, fade_max_factor=2.0), baseline())
    ev.skill = 1.0
    assert ev.alert_delay() == pytest.approx(60.0)
    events, _, _ = run(ev, [(45, feats(neck_ratio=0.5))])
    assert not events


def test_settings_roundtrip_ignores_unknown_and_coerces():
    s = Settings().updated({"alert_delay_s": "45", "bogus": 1, "sensitivity": "nope", "sound": 1})
    assert s.alert_delay_s == 45.0 and s.sensitivity == "standard" and s.sound is True

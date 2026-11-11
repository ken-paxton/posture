"""End-to-end tests on pre-recorded videos (real models, real decoding).

* Real clips: the pipeline must detect the person, calibrate, and run fast enough.
* Synthetic variants (tests/tools/make_variants.py): calibrate on the untouched
  first seconds, then check that the warped posture triggers the expected alert
  and the untouched control does not.

Alert delay is shortened to 1.5 s because the clips are only 7-15 s long.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from posture_app.config import Settings
from posture_app.engine import analyze_video

ROOT = Path(__file__).resolve().parent
VIDEOS = ROOT / "videos"
SYN = VIDEOS / "synthetic"
CLIPS = sorted(VIDEOS.glob("*.mp4"))

pytestmark = pytest.mark.skipif(not CLIPS, reason="no test videos in tests/videos")


@pytest.fixture(scope="session", autouse=True)
def variants():
    if CLIPS and not list(SYN.glob("*__control.mp4")):
        subprocess.run([sys.executable, str(ROOT / "tools" / "make_variants.py")], check=True)


def fast_settings():
    return Settings(alert_delay_s=1.5, alert_cooldown_s=100, fade_feedback=False, movement_reminder_min=0,
                    break_reminder_min=0)


def alerts(result):
    return [set(e["issues"]) for e in result["events"] if e["kind"] == "posture_alert"]


ANIMATED = {"mixkit-10441"}


@pytest.mark.parametrize("clip", CLIPS, ids=lambda p: p.stem)
def test_real_clip_detects_and_calibrates(clip):
    r = analyze_video(str(clip), settings=fast_settings(), calibrate=(0.3, 3.0))
    tl = r["timeline"]
    present = [x for x in tl if x["features"]]
    assert len(present) / len(tl) > 0.9, "person not detected in most frames"
    if clip.stem in ANIMATED:
        # Talking and gesturing with the head swinging ~15 deg: the stillness check must
        # refuse to treat this as a baseline.
        assert r["calibration"]["phase"] == "failed" and "moved" in r["calibration"]["error"]
    else:
        assert r["calibration"]["phase"] == "ok", r["calibration"]
    assert r["perf_ms"] < 150, "pipeline too slow for real-time use"


EXPECT = {
    "zoom": {"lean_in"},
    "head_forward": {"forward_head"},
    "head_drop": {"slump"},
    "tilt": {"head_tilt", "shoulder_tilt"},
}

# Known limitation: in this clip the shoulders are a dark jacket partly hidden by the
# laptop, so both BlazePose and RTMPose re-infer the shoulders lower when the head
# drops (body prior). Neck ratio barely changes and shoulder confidence falls below
# the gate. See RESEARCH.md, "Frontal-view limitations".
XFAIL = {("pexels-6324532", "head_drop")}


def variant_cases():
    for stem in ("pexels-6324532", "mixkit-10457", "mixkit-10448"):
        for kind in EXPECT:
            marks = [pytest.mark.xfail(strict=True, reason="shoulder prior")] if (stem, kind) in XFAIL else []
            yield pytest.param(stem, kind, marks=marks, id=f"{stem}-{kind}")


@pytest.mark.parametrize("stem,kind", list(variant_cases()))
def test_variant_triggers_expected_alert(stem, kind):
    r = analyze_video(str(SYN / f"{stem}__{kind}.mp4"), settings=fast_settings(), calibrate=(0.5, 3.5))
    fired = set().union(*alerts(r)) if alerts(r) else set()
    assert fired & EXPECT[kind], f"expected {EXPECT[kind]}, got {fired}"


def test_still_control_has_no_alerts():
    r = analyze_video(str(SYN / "pexels-6324532__control.mp4"), settings=fast_settings(), calibrate=(0.5, 3.5))
    assert not alerts(r)
    scored = [x for x in r["timeline"] if x["score"] is not None]
    # The speaker genuinely dips his head in the final ~0.3 s (slump ~10.5%), which is
    # flagged as out of range but, correctly, too brief to alert.
    assert sum(x["bad"] for x in scored) / len(scored) < 0.10

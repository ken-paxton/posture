"""Generate synthetic posture-change videos from real clips, for end-to-end tests.

Each variant keeps the first ``HOLD`` seconds untouched (used for calibration),
then ramps a geometric warp in over ``RAMP`` seconds and holds it, simulating:

  zoom          whole upper body moves toward the camera   -> lean_in
  head_forward  head magnified relative to shoulders       -> forward_head (+ lean_in)
  head_drop     head shifted down toward the shoulders     -> slump
  tilt          whole frame rotated                        -> head_tilt + shoulder_tilt
  control       unchanged (re-encoded)                     -> no alerts

These are geometric approximations of real movements, not real recordings; they
check that each feature responds in the right direction and that the alert
logic fires. Use `python -m posture_app record` to capture real labelled clips.

Usage: python tests/tools/make_variants.py [clip ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from posture_app.landmarks import L_EYE, L_SHOULDER, R_EYE, R_SHOULDER, LandmarkEngine  # noqa: E402

HOLD, RAMP = 4.0, 1.0
OUT = ROOT / "tests" / "videos" / "synthetic"
DEFAULT_CLIPS = ["pexels-6324532.mp4", "mixkit-10457.mp4", "mixkit-10448.mp4"]

AMOUNT = {"zoom": 1.30, "head_forward": 1.25, "head_drop": 0.25, "tilt": 12.0}


def anchors(path: Path) -> dict:
    """Face centre/size and shoulder line from frames around HOLD seconds (full-res pixels)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    eng = LandmarkEngine()
    pts = []
    for i in range(int(HOLD * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        if i % 3:
            continue
        obs = eng.process(frame, i / fps)
        s = frame.shape[1] / obs.width
        if obs.face and L_SHOULDER in obs.pose:
            fx = [q.x * s for q in obs.face.values()]
            fy = [q.y * s for q in obs.face.values()]
            p = obs.pose
            pts.append(
                [
                    (min(fx) + max(fx)) / 2,
                    (min(fy) + max(fy)) / 2,
                    max(max(fx) - min(fx), max(fy) - min(fy)),
                    (p[L_SHOULDER].y + p[R_SHOULDER].y) / 2 * s,
                    abs(p[L_SHOULDER].x - p[R_SHOULDER].x) * s,
                    (p[L_EYE].y + p[R_EYE].y) / 2 * s,
                ]
            )
    eng.close()
    cap.release()
    cx, cy, size, sh_y, sw, eye_y = np.median(np.array(pts), axis=0)
    return dict(cx=cx, cy=cy, size=size, sh_y=sh_y, sw=sw, eye_y=eye_y)


def warp(frame: np.ndarray, kind: str, k: float, a: dict, grid) -> np.ndarray:
    """Apply ``kind`` at strength k (0..1)."""
    if kind == "control" or k <= 0:
        return frame
    h, w = frame.shape[:2]
    xs, ys = grid
    if kind == "zoom":
        s = 1 + (AMOUNT["zoom"] - 1) * k
        m = cv2.getRotationMatrix2D((a["cx"], a["cy"]), 0, s)
        return cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if kind == "tilt":
        m = cv2.getRotationMatrix2D((w / 2, a["sh_y"]), AMOUNT["tilt"] * k, 1.0)
        return cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if kind == "head_forward":
        s = 1 + (AMOUNT["head_forward"] - 1) * k
        # Magnify the head only; fade out before reaching the shoulders.
        r_out = min(a["size"] * 1.0, 0.9 * (a["sh_y"] - a["cy"]))
        r_in = 0.6 * r_out
        dx, dy = xs - a["cx"], ys - a["cy"]
        r = np.sqrt(dx * dx + dy * dy)
        t = np.clip((r - r_in) / (r_out - r_in), 0, 1)
        t = t * t * (3 - 2 * t)
        scale = s + (1 - s) * t  # s inside, 1 outside
        mapx = (a["cx"] + dx / scale).astype(np.float32)
        mapy = (a["cy"] + dy / scale).astype(np.float32)
        return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if kind == "head_drop":
        full = AMOUNT["head_drop"] * (a["sh_y"] - a["eye_y"])
        shift = full * k
        # Whole face (down to the chin) moves rigidly; only the neck is compressed.
        # The compressed band must be wider than the shift or the image folds over.
        bottom = a["sh_y"] - 0.03 * a["sw"]
        top = min(a["cy"] + 0.5 * a["size"], bottom - 1.6 * full)
        wgt = np.clip((bottom - ys) / max(1.0, bottom - top), 0, 1)
        wgt = wgt * wgt * (3 - 2 * wgt)
        mapy = (ys - shift * wgt).astype(np.float32)
        return cv2.remap(frame, xs.astype(np.float32), mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    raise ValueError(kind)


def make(path: Path, kinds=("control", "zoom", "head_forward", "head_drop", "tilt")) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    a = anchors(path)
    outs = []
    for kind in kinds:
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(3)), int(cap.get(4))
        grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        out = OUT / f"{path.stem}__{kind}.mp4"
        vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            k = min(1.0, max(0.0, (i / fps - HOLD) / RAMP))
            vw.write(warp(frame, kind, k, a, grid))
            i += 1
        vw.release()
        cap.release()
        outs.append(out)
        print("wrote", out.relative_to(ROOT))
    return outs


if __name__ == "__main__":
    clips = sys.argv[1:] or DEFAULT_CLIPS
    for c in clips:
        p = Path(c)
        make(p if p.exists() else ROOT / "tests" / "videos" / c)

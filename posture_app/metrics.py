"""Posture features computed from one frame's landmarks.

All features are chosen to be measurable from a head-on (frontal) webcam, where
the classic sagittal craniovertebral angle cannot be seen directly. Each one is
compared against the user's own calibrated baseline, never against a population
"ideal", because camera height, lens and body proportions differ per setup.

Raw features (per frame, ``None`` when not measurable):

``neck_ratio``      Perpendicular distance from the eye midpoint to the shoulder
                    line, divided by shoulder width. Falls when the head drops
                    or juts forward and down (slumping, forward head, looking
                    down at a laptop) and when shoulders shrug up toward the ears.
                    Dimensionless, so it doesn't change with distance to camera.
``shoulder_width``  Shoulder width in pixels (scale reference for the torso).
``head_distance``   Head-to-camera distance from the face transformation matrix.
                    Compensated for head rotation, unlike apparent face size.
``head_pitch``      Head flexion (+, chin down) / extension (-) in degrees.
``head_roll``       Lateral head tilt in degrees.
``head_yaw``        Head rotation in degrees (used to gate other metrics).
``shoulder_tilt``   Angle of the shoulder line from horizontal, in degrees.
``trunk_lean``      Lateral trunk lean from vertical (needs hips in view).
``ear_depth``       BlazePose world-space depth of the ears relative to the
                    shoulders, in metres (more negative = head further forward).
                    Learned from 2-D cues, so it is shown but not used for alerts.
``eye_y``           Eye height as a fraction of frame height (used only to tell
                    sitting and standing profiles apart).
"""

from __future__ import annotations

import math

from .landmarks import (
    L_EAR,
    L_EYE,
    L_HIP,
    L_SHOULDER,
    R_EAR,
    R_EYE,
    R_HIP,
    R_SHOULDER,
    Observation,
)

MIN_VIS = 0.5

RAW_FEATURES = (
    "neck_ratio",
    "shoulder_width",
    "head_distance",
    "head_pitch",
    "head_roll",
    "head_yaw",
    "shoulder_tilt",
    "trunk_lean",
    "ear_depth",
    "eye_y",
)


def _visible(obs: Observation, *idx: int, min_vis: float = MIN_VIS) -> bool:
    for i in idx:
        p = obs.pose.get(i)
        if p is None or p.visibility < min_vis:
            return False
        # Points extrapolated outside the frame are guesses; reject them.
        if not (-0.02 * obs.width <= p.x <= 1.02 * obs.width and -0.02 * obs.height <= p.y <= 1.02 * obs.height):
            return False
    return True


def _angle_from_horizontal(ax: float, ay: float, bx: float, by: float) -> float:
    """Angle of the line a->b from horizontal, in degrees, in (-90, 90]."""
    ang = math.degrees(math.atan2(by - ay, bx - ax))
    if ang > 90:
        ang -= 180
    elif ang <= -90:
        ang += 180
    return ang


def compute_features(obs: Observation) -> dict[str, float | None]:
    f: dict[str, float | None] = {k: None for k in RAW_FEATURES}
    p = obs.pose

    shoulders = _visible(obs, L_SHOULDER, R_SHOULDER)
    if shoulders:
        ls, rs = p[L_SHOULDER], p[R_SHOULDER]
        sw = math.hypot(ls.x - rs.x, ls.y - rs.y)
        if sw > 0.05 * obs.width:
            f["shoulder_width"] = sw
            # The subject's left shoulder appears on the image right in a non-mirrored frame.
            f["shoulder_tilt"] = _angle_from_horizontal(rs.x, rs.y, ls.x, ls.y)

            if _visible(obs, L_EYE, R_EYE):
                ex = (p[L_EYE].x + p[R_EYE].x) / 2
                ey = (p[L_EYE].y + p[R_EYE].y) / 2
                f["eye_y"] = ey / obs.height
                # Signed perpendicular distance from the eye midpoint to the shoulder
                # line (positive when the eyes are above it); robust to camera roll.
                dx, dy = ls.x - rs.x, ls.y - rs.y
                cross = dx * (ey - rs.y) - dy * (ex - rs.x)
                dist = -cross / sw if dx > 0 else cross / sw
                f["neck_ratio"] = dist / sw

            if _visible(obs, L_HIP, R_HIP, min_vis=0.6):
                mx_s, my_s = (ls.x + rs.x) / 2, (ls.y + rs.y) / 2
                mx_h, my_h = (p[L_HIP].x + p[R_HIP].x) / 2, (p[L_HIP].y + p[R_HIP].y) / 2
                f["trunk_lean"] = math.degrees(math.atan2(mx_s - mx_h, my_h - my_s))

    w = obs.pose_world
    if shoulders and w and _visible(obs, L_EAR, R_EAR, min_vis=0.6):
        ear_z = (w[L_EAR].z + w[R_EAR].z) / 2
        sh_z = (w[L_SHOULDER].z + w[R_SHOULDER].z) / 2
        f["ear_depth"] = ear_z - sh_z

    if obs.head is not None:
        f["head_distance"] = obs.head.distance
        f["head_pitch"] = obs.head.pitch
        f["head_roll"] = obs.head.roll
        f["head_yaw"] = obs.head.yaw

    return f

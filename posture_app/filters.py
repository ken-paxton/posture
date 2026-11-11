"""One Euro filter (Casiez, Roussel & Vogel, CHI 2012).

An adaptive low-pass filter: heavy smoothing when the signal is still (removing
landmark jitter), light smoothing when it moves fast (keeping lag low). It is
the standard choice for smoothing pose/landmark streams.
"""

from __future__ import annotations

import math


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 0.5, beta: float = 0.05, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: float | None = None
        self._dx = 0.0
        self._t: float | None = None

    def __call__(self, x: float, t: float) -> float:
        if self._x is None or self._t is None or t <= self._t:
            self._x, self._t, self._dx = x, t, 0.0
            return x
        dt = t - self._t
        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = _alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        self._t = t
        return self._x


class FeatureSmoother:
    """One filter per feature; parameters scaled to each feature's units."""

    # (min_cutoff Hz, beta). Posture changes over seconds, so cut-offs are low.
    PARAMS = {
        "neck_ratio": (0.4, 2.0),
        "shoulder_width": (0.4, 0.01),
        "head_distance": (0.4, 0.05),
        "head_pitch": (0.4, 0.02),
        "head_roll": (0.4, 0.02),
        "head_yaw": (0.6, 0.02),
        "shoulder_tilt": (0.4, 0.02),
        "trunk_lean": (0.4, 0.02),
        "ear_depth": (0.3, 1.0),
        "eye_y": (0.4, 1.0),
    }

    def __init__(self):
        self._filters = {k: OneEuroFilter(*v) for k, v in self.PARAMS.items()}
        self._last_seen: dict[str, float] = {}

    def reset(self) -> None:
        for f in self._filters.values():
            f.reset()
        self._last_seen.clear()

    def __call__(self, features: dict[str, float | None], t: float, max_gap: float = 2.0) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for k, v in features.items():
            f = self._filters.get(k)
            if v is None or f is None:
                out[k] = v
                continue
            # After a long dropout, restart instead of gliding from a stale value.
            if t - self._last_seen.get(k, -1e9) > max_gap:
                f.reset()
            self._last_seen[k] = t
            out[k] = f(v, t)
        return out

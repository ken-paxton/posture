"""Turns smoothed features + baseline into posture state, alerts and reminders.

Timing uses the frame timestamps passed in (not the wall clock), so recorded
videos replay deterministically in tests.

Alert logic:
  1. For each enabled rule, deviation / threshold gives a normalised severity
     (1.0 = at threshold). Thresholds are scaled by the sensitivity preset and
     never set below NOISE_K x the noise measured during calibration.
  2. Hysteresis: posture becomes "out of range" at severity >= 1.0 and returns
     to "ok" only below 0.8, so a borderline posture doesn't flicker.
  3. An alert fires only after the deviation has persisted for alert_delay
     seconds (short returns of up to grace_s don't reset the timer). Alerts are
     then rate-limited by a cooldown. Brief deviations are normal and healthy.
  4. Faded feedback: as the user's recent good-posture share improves, the
     delay grows (up to fade_max_factor x), so the user learns to rely on their
     own body awareness rather than the app (guidance hypothesis, motor learning).

Separately, movement reminders fire when the user has stayed in much the same
position for a long time (posture variety), and break reminders after long
continuous presence (micro-breaks).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .calibration import NOISE_K, Baseline
from .config import SENSITIVITY, Settings
from .rules import BREAK_MESSAGE, DISPLAY_SCALE, MOVEMENT_MESSAGE, RULES

ENTER, EXIT = 1.0, 0.8
YAW_GATE_DEG = 35.0  # beyond this head turn (vs baseline), face-derived rules are skipped
ABSENT_AFTER_S = 2.0
MIN_GAP_BETWEEN_NOTIFICATIONS_S = 60.0

# Change in these features since the last "anchor" counts as changing position.
MOVE_DELTAS = {"neck_ratio": 0.08, "head_pitch": 12.0, "head_roll": 10.0, "shoulder_tilt": 6.0, "eye_y": 0.06}
MOVE_REL_DELTAS = {"head_distance": 0.12, "shoulder_width": 0.12}


@dataclass
class Event:
    kind: str  # "posture_alert" | "movement" | "break" | "corrected"
    t: float
    title: str
    message: str
    issues: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)


@dataclass
class RuleState:
    key: str
    label: str
    unit: str
    deviation: float | None  # display units
    threshold: float  # display units
    severity: float | None  # deviation / threshold


@dataclass
class Assessment:
    t: float
    present: bool
    calibrated: bool
    score: float | None
    bad: bool
    bad_for: float
    issues: list[str]
    rules: list[RuleState]
    events: list[Event]
    alert_delay: float
    still_for: float
    present_for: float


class Evaluator:
    def __init__(self, settings: Settings, baseline: Baseline | None):
        self.settings = settings
        self.baseline = baseline
        self.skill = 0.0  # 0..1, recent good-posture share mapped by the engine
        self._last_present_t: float | None = None
        self._away_since: float | None = None
        self._present_since: float | None = None
        self._bad = False
        self._bad_since: float | None = None
        self._good_since: float | None = None
        self._last_alert_t = -1e9
        self._last_notify_t = -1e9
        self._alert_pending_correction: Event | None = None
        self._still_since: float | None = None
        self._anchor: dict[str, float] = {}
        self._last_break_t: float | None = None

    # -- configuration -----------------------------------------------------
    def set_baseline(self, baseline: Baseline | None) -> None:
        self.baseline = baseline
        self._bad = False
        self._bad_since = None

    def thresholds(self) -> dict[str, float]:
        s = self.settings
        mult = SENSITIVITY.get(s.sensitivity, 1.0)
        out = {}
        for r in RULES:
            thr = s.threshold_overrides.get(r.key, r.threshold) * mult
            if self.baseline and r.key in self.baseline.noise:
                thr = max(thr, NOISE_K * self.baseline.noise[r.key])
            out[r.key] = thr
        return out

    def alert_delay(self) -> float:
        s = self.settings
        if not s.fade_feedback:
            return s.alert_delay_s
        return s.alert_delay_s * (1.0 + (s.fade_max_factor - 1.0) * max(0.0, min(1.0, self.skill)))

    # -- main step -----------------------------------------------------------
    def update(self, feats: dict[str, float | None] | None, t: float) -> Assessment:
        events: list[Event] = []
        present = bool(feats) and any(
            feats.get(k) is not None for k in ("neck_ratio", "head_pitch", "shoulder_width", "head_distance")
        )

        if present:
            if self._away_since is not None:
                away = t - self._away_since
                if away >= self.settings.away_reset_s:
                    self._present_since = t
                    self._still_since = None
                    self._last_break_t = None
                self._away_since = None
            if self._present_since is None:
                self._present_since = t
            self._last_present_t = t
        else:
            if self._last_present_t is None or t - self._last_present_t > ABSENT_AFTER_S:
                if self._away_since is None:
                    self._away_since = self._last_present_t if self._last_present_t is not None else t
                self._bad = False
                self._bad_since = None
            return self._assessment(t, present=False, score=None, rules=[], issues=[], events=events)

        rules_state, worst, issues = self._score_rules(feats)
        score = None
        if worst is not None:
            score = 100.0 if worst <= 0.5 else max(0.0, 100.0 - (worst - 0.5) / 1.5 * 100.0)

        # Hysteresis + sustained timing
        if worst is not None:
            if not self._bad and worst >= ENTER:
                self._bad = True
                if self._bad_since is None:
                    self._bad_since = t
                self._good_since = None
            elif self._bad and worst < EXIT:
                self._bad = False
                self._good_since = t
            if not self._bad and self._bad_since is not None and self._good_since is not None:
                if t - self._good_since > self.settings.grace_s:
                    self._bad_since = None
                    if self._alert_pending_correction is not None:
                        a = self._alert_pending_correction
                        events.append(
                            Event("corrected", t, "Nice correction", "Posture back in range.", a.issues,
                                  {"latency_s": round(t - a.t, 1)})
                        )
                        self._alert_pending_correction = None

        delay = self.alert_delay()
        bad_for = (t - self._bad_since) if (self._bad and self._bad_since is not None) else 0.0
        if (
            self._bad
            and bad_for >= delay
            and t - self._last_alert_t >= self.settings.alert_cooldown_s
            and t - self._last_notify_t >= MIN_GAP_BETWEEN_NOTIFICATIONS_S / 2
            and issues
        ):
            top = issues[0]
            rule = next(r for r in RULES if r.key == top)
            ev = Event("posture_alert", t, f"Posture check: {rule.label.lower()}", rule.message, list(issues),
                       {"bad_for_s": round(bad_for, 1)})
            events.append(ev)
            self._last_alert_t = self._last_notify_t = t
            self._alert_pending_correction = ev

        events.extend(self._movement(feats, t))
        return self._assessment(t, present=True, score=score, rules=rules_state, issues=issues, events=events,
                                bad_for=bad_for)

    # -- helpers -------------------------------------------------------------
    def _score_rules(self, feats):
        b = self.baseline
        if b is None:
            return [], None, []
        thr = self.thresholds()
        enabled = set(self.settings.enabled_rules)
        yaw_ok = True
        if feats.get("head_yaw") is not None and "head_yaw" in b.values:
            yaw_ok = abs(feats["head_yaw"] - b.values["head_yaw"]) <= YAW_GATE_DEG

        states, worst, sev_by_rule = [], None, {}
        for r in RULES:
            if r.key not in b.available_rules:
                continue
            scale = DISPLAY_SCALE.get(r.unit, 1.0)
            dev = sev = None
            if all(feats.get(k) is not None for k in r.needs) and (yaw_ok or not r.face_metric):
                try:
                    dev = r.deviation(feats, b.values)
                except ZeroDivisionError:
                    dev = None
                if dev is not None:
                    sev = dev / thr[r.key]
            states.append(RuleState(r.key, r.label, r.unit, None if dev is None else dev * scale,
                                    thr[r.key] * scale, sev))
            if sev is not None and r.key in enabled:
                sev_by_rule[r.key] = sev
                worst = sev if worst is None else max(worst, sev)
        # Issues: rules currently past the entry threshold, most severe first. While
        # in the bad state, keep reporting rules above the exit threshold too.
        cut = EXIT if self._bad else ENTER
        issues = [k for k, v in sorted(sev_by_rule.items(), key=lambda kv: -kv[1]) if v >= cut]
        return states, worst, issues

    def _movement(self, feats, t) -> list[Event]:
        s = self.settings
        out: list[Event] = []
        current = {k: feats[k] for k in list(MOVE_DELTAS) + list(MOVE_REL_DELTAS) if feats.get(k) is not None}
        if self._still_since is None or self._moved(current):
            self._still_since = t
            self._anchor = current

        if s.movement_reminder_min > 0 and t - self._still_since >= s.movement_reminder_min * 60:
            if t - self._last_notify_t >= MIN_GAP_BETWEEN_NOTIFICATIONS_S:
                minutes = int(round((t - self._still_since) / 60))
                out.append(Event("movement", t, "Time to move", MOVEMENT_MESSAGE.format(minutes=minutes)))
                self._last_notify_t = t
                self._still_since = t
                self._anchor = current

        if s.break_reminder_min > 0 and self._present_since is not None:
            since = max(self._present_since, self._last_break_t or self._present_since)
            if t - since >= s.break_reminder_min * 60 and t - self._last_notify_t >= MIN_GAP_BETWEEN_NOTIFICATIONS_S:
                minutes = int(round((t - self._present_since) / 60))
                out.append(Event("break", t, "Break time", BREAK_MESSAGE.format(minutes=minutes)))
                self._last_notify_t = t
                self._last_break_t = t
        return out

    def _moved(self, current: dict[str, float]) -> bool:
        for k, d in MOVE_DELTAS.items():
            if k in current and k in self._anchor and abs(current[k] - self._anchor[k]) > d:
                return True
        for k, d in MOVE_REL_DELTAS.items():
            if k in current and k in self._anchor and self._anchor[k]:
                if abs(current[k] / self._anchor[k] - 1.0) > d:
                    return True
        return False

    def _assessment(self, t, present, score, rules, issues, events, bad_for=0.0) -> Assessment:
        return Assessment(
            t=t,
            present=present,
            calibrated=self.baseline is not None,
            score=score,
            bad=self._bad,
            bad_for=bad_for,
            issues=issues,
            rules=rules,
            events=events,
            alert_delay=self.alert_delay(),
            still_for=(t - self._still_since) if (present and self._still_since is not None) else 0.0,
            present_for=(t - self._present_since) if (present and self._present_since is not None) else 0.0,
        )

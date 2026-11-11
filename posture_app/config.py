"""User settings and on-disk locations. Everything is stored locally."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

APP_NAME = "posture-coach"


def data_dir() -> Path:
    override = os.environ.get("POSTURE_DATA_DIR")
    if override:
        d = Path(override)
    elif sys.platform == "darwin":
        d = Path.home() / "Library" / "Application Support" / APP_NAME
    elif os.name == "nt":
        d = Path(os.environ.get("APPDATA", Path.home())) / APP_NAME
    else:
        d = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


SENSITIVITY = {"gentle": 1.3, "standard": 1.0, "strict": 0.8}


@dataclass
class Settings:
    # Capture / inference
    camera: int = 0
    pose_model: str = "pose_full"  # pose_lite | pose_full | pose_heavy
    analysis_fps: float = 10.0  # frames analysed per second; posture changes slowly
    max_width: int = 640
    mirror_preview: bool = True

    # Alerting
    sensitivity: str = "standard"  # gentle | standard | strict
    alert_delay_s: float = 30.0  # how long a deviation must persist before alerting
    alert_cooldown_s: float = 180.0  # minimum gap between posture alerts
    grace_s: float = 3.0  # brief returns to good posture that don't reset the timer
    enabled_rules: list[str] = field(
        default_factory=lambda: [
            "slump",
            "forward_head",
            "lean_in",
            "head_flexion",
            "head_extension",
            "head_tilt",
            "shoulder_tilt",
            "trunk_lean",
        ]
    )
    threshold_overrides: dict[str, float] = field(default_factory=dict)
    desktop_notifications: bool = True
    sound: bool = False

    # Adaptive ("faded") feedback: as the user's good-posture share rises, alerts wait longer
    fade_feedback: bool = True
    fade_max_factor: float = 2.0

    # Movement / break reminders
    movement_reminder_min: float = 30.0  # 0 disables
    break_reminder_min: float = 50.0  # 0 disables
    away_reset_s: float = 120.0  # time out of frame that counts as a break

    # Profiles
    active_profile: str = "sitting"
    auto_profile: bool = True

    @classmethod
    def path(cls) -> Path:
        return data_dir() / "settings.json"

    @classmethod
    def load(cls) -> "Settings":
        s = cls()
        try:
            raw = json.loads(cls.path().read_text())
        except (OSError, ValueError):
            return s
        return s.updated(raw)

    def updated(self, raw: dict) -> "Settings":
        """Return a copy with known keys from ``raw`` applied (unknown keys ignored)."""
        names = {f.name: f for f in fields(self)}
        data = asdict(self)
        for k, v in raw.items():
            if k in names:
                default = data[k]
                if isinstance(default, bool):
                    v = bool(v)
                elif isinstance(default, (int, float)) and not isinstance(default, bool):
                    v = type(default)(v)
                data[k] = v
        if data["sensitivity"] not in SENSITIVITY:
            data["sensitivity"] = "standard"
        return Settings(**data)

    def save(self) -> None:
        tmp = self.path().with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.replace(self.path())

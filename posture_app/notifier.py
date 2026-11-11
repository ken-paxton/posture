"""Native desktop notifications via OS-provided local tools (no network, no extra deps)."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys

log = logging.getLogger(__name__)


def _ps_escape(s: str) -> str:
    return s.replace("'", "''")


def notify(title: str, message: str, urgent: bool = False) -> bool:
    try:
        if sys.platform.startswith("linux") and shutil.which("notify-send"):
            cmd = ["notify-send", "--app-name=Posture Coach", "--icon=preferences-desktop-accessibility",
                   f"--urgency={'normal' if urgent else 'low'}", "--expire-time=8000", title, message]
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        if sys.platform == "darwin":
            script = f"display notification {json_str(message)} with title {json_str(title)}"
            subprocess.Popen(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        if os.name == "nt":
            ps = (
                "[void][System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms');"
                "$n=New-Object System.Windows.Forms.NotifyIcon;$n.Icon=[System.Drawing.SystemIcons]::Information;"
                f"$n.BalloonTipTitle='{_ps_escape(title)}';$n.BalloonTipText='{_ps_escape(message)}';"
                "$n.Visible=$true;$n.ShowBalloonTip(8000);Start-Sleep -s 9;$n.Dispose()"
            )
            subprocess.Popen(["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return True
    except OSError as e:
        log.warning("notification failed: %s", e)
    log.info("NOTIFY %s: %s", title, message)
    return False


def json_str(s: str) -> str:
    """AppleScript string literal."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'

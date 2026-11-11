"""Local dashboard server (standard library only).

Privacy and security:
* Binds to 127.0.0.1 only, so it is unreachable from the network.
* Rejects requests whose Host header isn't localhost (blocks DNS-rebinding attacks
  that would otherwise let a web page read the camera stream).
* Every API call and the MJPEG stream require a random per-launch token, which is
  only embedded in the dashboard page itself. Other websites can't read or
  drive the app even though it listens on localhost.
* The page loads nothing from the internet (no CDNs, fonts or analytics).
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import SENSITIVITY, Settings
from .rules import RULES

log = logging.getLogger(__name__)
WEB = Path(__file__).resolve().parent / "web"
STATIC = {"/app.js": "application/javascript", "/style.css": "text/css"}


class App:
    """Glue between HTTP handlers and the engine/history."""

    def __init__(self, engine, history, settings: Settings):
        self.engine = engine
        self.history = history
        self.settings = settings
        self.token = secrets.token_urlsafe(24)

    def status(self) -> dict:
        snap = self.engine.snapshot()
        snap["paused"] = self.engine.paused
        snap["settings"] = self.settings.__dict__
        return snap

    def history_payload(self) -> dict:
        if not self.history:
            return {"daily": [], "today": [], "events": [], "good_share_7d": None}
        share, hours = self.history.good_share(7)
        return {
            "daily": self.history.daily(30),
            "today": self.history.today_minutes(),
            "events": self.history.recent_events(25),
            "good_share_7d": round(share * 100, 1) if hours > 0 else None,
            "hours_7d": round(hours, 1),
        }

    def meta(self) -> dict:
        return {
            "rules": [{"key": r.key, "label": r.label, "unit": r.unit, "default_threshold": r.threshold,
                       "alert_default": r.alert_default, "message": r.message} for r in RULES],
            "sensitivity": list(SENSITIVITY),
        }

    def post(self, path: str, body: dict) -> dict:
        e = self.engine
        if path == "/api/calibrate":
            e.start_calibration(str(body.get("profile") or "sitting"),
                                countdown=float(body.get("countdown", 3)), duration=float(body.get("duration", 5)))
        elif path == "/api/calibrate/cancel":
            e.cancel_calibration()
        elif path == "/api/pause":
            e.paused = bool(body.get("paused", not e.paused))
        elif path == "/api/profile":
            e.select_profile(str(body.get("profile", "")))
            self.settings.save()
        elif path == "/api/profile/delete":
            e.delete_profile(str(body.get("profile", "")))
            self.settings.save()
        elif path == "/api/settings":
            new = self.settings.updated(body)
            new.enabled_rules = [k for k in new.enabled_rules if any(r.key == k for r in RULES)]
            for k, v in self.settings.__dict__.items():
                setattr(self.settings, k, getattr(new, k))
            self.settings.save()
            e.apply_settings(self.settings)
        elif path == "/api/history/clear":
            if self.history:
                self.history.clear()
        elif path == "/api/test-notification":
            from .notifier import notify

            notify("Posture Coach", "Notifications are working.")
        else:
            raise KeyError(path)
        return {"ok": True}


def make_handler(app: App, port: int):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "PostureCoach"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            log.debug("http: " + fmt, *args)

        # -- guards --------------------------------------------------------------
        def _host_ok(self) -> bool:
            if self.headers.get("Host", "") not in allowed_hosts:
                self._send(HTTPStatus.FORBIDDEN, b"forbidden host", "text/plain")
                return False
            return True

        def _token_ok(self, query: dict) -> bool:
            tok = self.headers.get("X-Token") or (query.get("token") or [""])[0]
            if not secrets.compare_digest(tok, app.token):
                self._send(HTTPStatus.FORBIDDEN, b"bad token", "text/plain")
                return False
            return True

        def _send(self, status, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; "
                "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
            )
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status=HTTPStatus.OK):
            self._send(status, json.dumps(obj).encode(), "application/json")

        # -- verbs ---------------------------------------------------------------
        def do_GET(self):
            if not self._host_ok():
                return
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path in ("/", "/index.html"):
                html = (WEB / "index.html").read_text().replace("__TOKEN__", app.token)
                return self._send(HTTPStatus.OK, html.encode(), "text/html; charset=utf-8")
            if u.path in STATIC:
                return self._send(HTTPStatus.OK, (WEB / u.path.lstrip("/")).read_bytes(), STATIC[u.path])
            if not self._token_ok(q):
                return
            if u.path == "/api/status":
                return self._json(app.status())
            if u.path == "/api/history":
                return self._json(app.history_payload())
            if u.path == "/api/meta":
                return self._json(app.meta())
            if u.path == "/stream.mjpg":
                return self._stream()
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

        def do_POST(self):
            if not self._host_ok():
                return
            u = urlparse(self.path)
            if not self._token_ok({}):
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self._send(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, b"json only", "text/plain")
            length = min(int(self.headers.get("Content-Length") or 0), 65536)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError
                self._json(app.post(u.path, body))
            except KeyError:
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            except (ValueError, TypeError) as e:
                self._json({"ok": False, "error": str(e)}, HTTPStatus.BAD_REQUEST)

        def _stream(self):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            last = None
            try:
                while True:
                    jpg = app.engine.preview_jpeg()
                    if jpg is not None and jpg is not last:
                        last = jpg
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                        self.wfile.flush()
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    return Handler


def serve(app: App, port: int = 8765, tries: int = 20) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Serve on 127.0.0.1, using the next free port if ``port`` is taken."""
    for p in range(port, port + tries):
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", p), make_handler(app, p))
            break
        except OSError:
            continue
    else:
        raise OSError(f"no free port in {port}-{port + tries - 1}")
    httpd.daemon_threads = True
    th = threading.Thread(target=httpd.serve_forever, daemon=True, name="http")
    th.start()
    return httpd, th

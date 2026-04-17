#!/usr/bin/env python3
"""
Fire Alert System — browser-only portal (login + register).

Run:
  python fire_alert_browser.py
  python fire_alert_browser.py --host 127.0.0.1 --port 8766

Uses SQLite next to this script: fire_alert_portal.db

Camera RTSP page: uses **best-kiase.pt** and **best_fire.pt** (place next to this script).
Install: pip install ultralytics opencv-python
Optional env: FIRE_ALERT_CONF, FIRE_ALERT_DETECT_EVERY, FIRE_ALERT_IMGSZ, FIRE_ALERT_LOG_COOLDOWN
"""

from __future__ import annotations

import hashlib
import html
import os
import re
import secrets
import sqlite3
import sys
import threading
import time
import urllib.parse
import webbrowser
import cgi
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

try:
    import cv2
except Exception:
    cv2 = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# DB file beside this script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DB_PATH = os.path.join(_SCRIPT_DIR, "fire_alert_portal.db")
_LOG_DIR = os.path.join(_SCRIPT_DIR, "logs_browser")
os.makedirs(_LOG_DIR, exist_ok=True)
_REPORT_IMAGE_DIR = os.path.join(_LOG_DIR, "reports")
os.makedirs(_REPORT_IMAGE_DIR, exist_ok=True)

# Prefer TCP for RTSP stability (same as fire_alert_desktop1 helpers)
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

# Fire detection: dual YOLO models (class 0 = fire, same as desktop app)
MODEL_FILES = (
    ("best-kiase.pt", "kiase"),
    ("best_fire.pt", "fire"),
)
CONF_THRESHOLD = float(os.environ.get("FIRE_ALERT_CONF", "0.85"))
DETECT_EVERY_N = int(os.environ.get("FIRE_ALERT_DETECT_EVERY", "3"))
YOLO_IMGSZ = int(os.environ.get("FIRE_ALERT_IMGSZ", "640"))
LOG_FIRE_COOLDOWN_SEC = float(os.environ.get("FIRE_ALERT_LOG_COOLDOWN", "20"))

_SESSION_COOKIE = "fire_alert_portal_session"
_SESSION_TTL_SEC = 86400 * 7
_MJPEG_BOUNDARY = b"frame"


def _model_path(filename: str) -> str:
    return os.path.join(_SCRIPT_DIR, filename)


def load_fire_models() -> dict[str, object]:
    """Load best-kiase.pt and best_fire.pt; skip missing files."""
    out: dict[str, object] = {}
    if YOLO is None:
        print("YOLO: ultralytics not installed. Run: pip install ultralytics", file=sys.stderr)
        return out
    for fname, short in MODEL_FILES:
        p = _model_path(fname)
        if not os.path.isfile(p):
            print("YOLO: model not found (skipped): %s" % p, file=sys.stderr)
            continue
        try:
            out[short] = YOLO(p)
            print("YOLO: loaded %s as '%s'" % (fname, short))
        except Exception as e:
            print("YOLO: failed to load %s: %s" % (p, e), file=sys.stderr)
    return out


def _draw_fire_boxes(frame, boxes: list) -> None:
    """Draw boxes in-place. boxes: list of (label, x1, y1, x2, y2, conf)."""
    colors = [(0, 0, 255), (0, 165, 255)]
    for i, (name, x1, y1, x2, y2, cf) in enumerate(boxes):
        c = colors[i % len(colors)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        cv2.putText(
            frame,
            "%s %.2f" % (name, cf),
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            c,
            1,
        )


def _run_dual_fire_detect(models: list[tuple[object, str]], frame, conf: float, imgsz: int) -> tuple[list, bool, float]:
    """Run both models; return (boxes, any_fire, max_conf). Class 0 = fire."""
    boxes = []
    max_cf = 0.0
    for model, short in models:
        try:
            results = model(frame, conf=conf, verbose=False, imgsz=imgsz)
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    if int(box.cls[0]) != 0:
                        continue
                    cf = float(box.conf[0])
                    max_cf = max(max_cf, cf)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes.append((short, x1, y1, x2, y2, cf))
        except Exception:
            pass
    return boxes, bool(boxes), max_cf


class RtspBridge:
    """RTSP reader + YOLO fire detection using capture/detect split threads."""

    def __init__(self, state: dict | None = None, models_map: dict[str, object] | None = None) -> None:
        self._lock = threading.Lock()
        self._cap = None
        self._url = None
        self._latest_jpeg = None
        self._capture_thread = None
        self._detect_thread = None
        self._stop = threading.Event()
        self._state = state
        self._models_map = models_map or {}
        self._active_models: list[tuple[object, str]] = []
        self._active_model_names: list[str] = []
        self._conf_threshold = CONF_THRESHOLD
        self._frame_count = 0
        self._last_detected_frame_count = 0
        self._latest_frame = None
        self._cached_boxes: list = []
        self._last_log_time = 0.0

    def ensure_stream(self, rtsp_url: str, selected_models: list[str] | None = None, conf_threshold: float | None = None) -> None:
        picked = selected_models or list(self._models_map.keys())
        active = [(self._models_map[n], n) for n in picked if n in self._models_map]
        if not active:
            active = [(m, n) for n, m in self._models_map.items()]
        conf = conf_threshold if conf_threshold is not None else CONF_THRESHOLD
        key = (rtsp_url, round(float(conf), 4), ",".join(sorted([n for _, n in active])))
        with self._lock:
            same_threads_running = (
                self._capture_thread
                and self._capture_thread.is_alive()
                and self._detect_thread
                and self._detect_thread.is_alive()
            )
            if getattr(self, "_stream_key", None) == key and same_threads_running:
                return
            self._stop_unlocked()
            self._url = rtsp_url
            self._stream_key = key
            self._active_models = active
            self._active_model_names = [n for _, n in active]
            self._conf_threshold = max(0.01, min(0.99, float(conf)))
            self._frame_count = 0
            self._last_detected_frame_count = 0
            self._latest_frame = None
            self._cached_boxes = []
            self._stop.clear()
            self._capture_thread = threading.Thread(target=self._run_capture, daemon=True)
            self._detect_thread = threading.Thread(target=self._run_detect, daemon=True)
            self._capture_thread.start()
            self._detect_thread.start()

    def _save_snapshot(self, frame) -> str | None:
        if frame is None or cv2 is None:
            return None
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = "detect_%s_%d.jpg" % (ts, int(time.time() * 1000) % 1000)
            path = os.path.join(_LOG_DIR, name)
            ok = cv2.imwrite(path, frame)
            return path if ok else None
        except Exception:
            return None

    def _maybe_log_fire(self, rtsp_url: str, max_conf: float, snap_path: str | None = None) -> None:
        if not self._state or max_conf <= 0:
            return
        now = time.time()
        if now - self._last_log_time < LOG_FIRE_COOLDOWN_SEC:
            return
        self._last_log_time = now
        try:
            cam = rtsp_url[:180] if len(rtsp_url) > 180 else rtsp_url
            model_text = ",".join(self._active_model_names) if self._active_model_names else "none"
            details = "YOLO %s | threshold %.2f | max conf %.0f%%" % (model_text, self._conf_threshold, max_conf * 100.0)
            self._state["cursor"].execute(
                "INSERT INTO camera_detections (created_at, camera_name, status, severity, details, image_path) VALUES (?,?,?,?,?,?)",
                (
                    datetime.now().isoformat(timespec="seconds"),
                    cam,
                    "Fire detected",
                    "High",
                    details,
                    snap_path,
                ),
            )
            self._state["conn"].commit()
        except Exception:
            pass

    def _run_capture(self) -> None:
        url = self._url
        if not url or cv2 is None:
            return
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self._cap = cap
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.08)
                continue
            with self._lock:
                self._frame_count += 1
                self._latest_frame = frame
                boxes = list(self._cached_boxes)
            display = frame
            if boxes:
                display = frame.copy()
                _draw_fire_boxes(display, boxes)
            ok, buf = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                with self._lock:
                    self._latest_jpeg = buf.tobytes()
            time.sleep(1.0 / 20.0)
        try:
            cap.release()
        except Exception:
            pass
        with self._lock:
            self._cap = None

    def _run_detect(self) -> None:
        """AI thread: reads latest frame snapshot and updates cached boxes."""
        url = self._url
        if not url or cv2 is None:
            return
        while not self._stop.is_set():
            with self._lock:
                current_count = self._frame_count
                frame = None if self._latest_frame is None else self._latest_frame.copy()
                models = list(self._active_models)
                conf = self._conf_threshold
            if frame is None or not models:
                time.sleep(0.02)
                continue

            # Skip frames for detection, but keep display using latest cached boxes.
            if current_count % DETECT_EVERY_N != 0 or current_count == self._last_detected_frame_count:
                time.sleep(0.01)
                continue

            self._last_detected_frame_count = current_count
            boxes, fire, max_cf = _run_dual_fire_detect(models, frame, conf, YOLO_IMGSZ)
            with self._lock:
                self._cached_boxes = boxes
            if fire:
                snap = frame.copy()
                _draw_fire_boxes(snap, boxes)
                snap_path = self._save_snapshot(snap)
                self._maybe_log_fire(url, max_cf, snap_path=snap_path)
            time.sleep(0.005)

    def get_jpeg(self):
        with self._lock:
            return self._latest_jpeg

    def stop(self) -> None:
        with self._lock:
            self._stop_unlocked()

    def _stop_unlocked(self) -> None:
        self._stop.set()
        cap_t = self._capture_thread
        det_t = self._detect_thread
        self._capture_thread = None
        self._detect_thread = None
        self._url = None
        self._stream_key = None
        self._latest_jpeg = None
        self._latest_frame = None
        self._cached_boxes = []
        cap = self._cap
        self._cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if cap_t and cap_t.is_alive():
            cap_t.join(timeout=1.5)
        if det_t and det_t.is_alive():
            det_t.join(timeout=1.5)


def _pbkdf2_hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("ascii"), 200000)
    return salt + ":" + dk.hex()


def _pbkdf2_verify_password(password: str, stored: str) -> bool:
    try:
        salt, hexd = stored.split(":", 1)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("ascii"), 200000)
        return dk.hex() == hexd
    except Exception:
        return False


def _parse_cookies(header_val: str) -> dict:
    out = {}
    if not header_val:
        return out
    for part in header_val.split(";"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = urllib.parse.unquote(v.strip())
    return out


def _init_db(conn: sqlite3.Connection) -> None:
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS fire_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            created_by TEXT NOT NULL,
            address TEXT NOT NULL,
            severity TEXT NOT NULL,
            description TEXT NOT NULL,
            image_path TEXT
        )
        """
    )
    try:
        c.execute("ALTER TABLE fire_reports ADD COLUMN image_path TEXT")
    except Exception:
        pass
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS camera_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            camera_name TEXT NOT NULL,
            status TEXT NOT NULL,
            severity TEXT NOT NULL,
            details TEXT,
            image_path TEXT
        )
        """
    )
    try:
        c.execute("ALTER TABLE camera_detections ADD COLUMN image_path TEXT")
    except Exception:
        pass
    conn.commit()


def auth_page_html(active: str, error: str = "", notice: str = "") -> str:
    """Login / Register — dark card UI matching Fire Alert System."""
    active = "register" if active == "register" else "login"
    err_html = ('<p class="err">%s</p>' % html.escape(error)) if error else ""
    ok_html = ('<p class="ok">%s</p>' % html.escape(notice)) if notice else ""
    login_cls = "tab active" if active == "login" else "tab"
    reg_cls = "tab active" if active == "register" else "tab"
    login_form_display = "flex" if active == "login" else "none"
    reg_form_display = "flex" if active == "register" else "none"
    return """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fire Alert System</title>
<style>
:root {
  --bg: #121316;
  --card: #23252a;
  --input: #15171b;
  --muted: #8f96a3;
  --text: #f3f4f6;
  --accent: #ff4b52;
  --accent-dim: #f23f46;
  --line: #30343b;
}
* { box-sizing: border-box; }
body {
  margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center;
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  background:
    radial-gradient(1200px 700px at 50%% -180px, rgba(255, 75, 82, 0.08), transparent 60%%),
    #121316;
  color: var(--text);
  padding: 24px;
}
.card {
  width: 100%%; max-width: 420px;
  background: var(--card);
  border-radius: 14px;
  padding: 28px 28px 32px;
  box-shadow: 0 20px 50px rgba(0,0,0,.44), 0 0 0 1px rgba(255,255,255,.04);
}
.brand { display: flex; align-items: flex-start; gap: 12px; margin-bottom: 8px; }
.flame {
  font-size: 1.75rem; line-height: 1;
  filter: drop-shadow(0 2px 8px rgba(255, 75, 82, .45));
}
.brand h1 {
  margin: 0; font-size: 1.35rem; font-weight: 700;
  background: linear-gradient(135deg, #ff6a57, #ff3b47);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.tagline { margin: 6px 0 0; font-size: 0.88rem; font-weight: 400; color: var(--muted); line-height: 1.45; }
.tabs {
  display: flex; margin-top: 22px; border-bottom: 1px solid var(--line);
}
.tab {
  flex: 1; text-align: center; padding: 12px 8px; font-size: 0.95rem; font-weight: 600;
  color: var(--muted); text-decoration: none; border-bottom: 3px solid transparent;
  margin-bottom: -1px; transition: color .15s, border-color .15s;
}
.tab:hover { color: var(--text); }
.tab.active { color: var(--accent); border-bottom-color: var(--accent); }
label { display: block; font-size: 0.8rem; font-weight: 500; color: var(--text); margin-bottom: 6px; }
input[type="text"], input[type="email"], input[type="password"] {
  width: 100%%; padding: 12px 14px; border-radius: 8px; border: 1px solid var(--line);
  background: var(--input); color: var(--text); font-size: 0.95rem; outline: none;
}
input:focus { border-color: rgba(255, 75, 82, .45); box-shadow: 0 0 0 3px rgba(255, 75, 82, .12); }
.btn {
  width: 100%%; margin-top: 6px; padding: 13px; border: none; border-radius: 8px;
  font-size: 1rem; font-weight: 700; color: #fff; cursor: pointer;
  background: linear-gradient(180deg, var(--accent), var(--accent-dim));
  box-shadow: 0 4px 14px rgba(255, 75, 82, .28);
}
.btn:hover { filter: brightness(1.05); }
.btn:active { transform: translateY(1px); }
.err { color: #fca5a5; font-size: 0.85rem; margin: 0 0 8px 0; }
.ok { color: #86efac; font-size: 0.85rem; margin: 0 0 8px 0; }
</style></head><body>
<div class="card">
  <div class="brand">
    <span class="flame" aria-hidden="true">&#128293;</span>
    <div>
      <h1>Fire Alert System</h1>
      <p class="tagline">Report and track fire emergencies in real-time</p>
    </div>
  </div>
  <div class="tabs">
    <a class="%s" href="/">Login</a>
    <a class="%s" href="/?view=register">Register</a>
  </div>
  %s%s
  <form method="post" action="/login" style="display:%s;flex-direction:column;gap:16px;margin-top:22px;">
    <div>
      <label for="login_user">Username or Email</label>
      <input id="login_user" name="username" type="text" autocomplete="username" required>
    </div>
    <div>
      <label for="login_pass">Password</label>
      <input id="login_pass" name="password" type="password" autocomplete="current-password" required>
    </div>
    <button type="submit" class="btn">Login</button>
  </form>
  <form method="post" action="/register" style="display:%s;flex-direction:column;gap:16px;margin-top:22px;">
    <div>
      <label for="reg_user">Username</label>
      <input id="reg_user" name="username" type="text" autocomplete="username" required minlength="2" maxlength="64">
    </div>
    <div>
      <label for="reg_email">Email</label>
      <input id="reg_email" name="email" type="email" autocomplete="email" required>
    </div>
    <div>
      <label for="reg_pass">Password</label>
      <input id="reg_pass" name="password" type="password" autocomplete="new-password" required minlength="6">
    </div>
    <div>
      <label for="reg_pass2">Confirm password</label>
      <input id="reg_pass2" name="confirm" type="password" autocomplete="new-password" required>
    </div>
    <button type="submit" class="btn">Register</button>
  </form>
</div>
</body></html>""" % (
        login_cls,
        reg_cls,
        err_html,
        ok_html,
        login_form_display,
        reg_form_display,
    )


def _dashboard_layout_html(username: str, active: str, title: str, content_html: str, show_filters: bool = False) -> str:
    u = html.escape(username)
    nav = [
        ("dashboard", "Dashboard", "/home"),
        ("report-fire", "Report Fire", "/home/report-fire"),
        ("all-alerts", "All Alerts", "/home/all-alerts"),
        ("camera-detection", "Camera Detection", "/home/camera-detection"),
    ]
    nav_html = []
    for key, label, href in nav:
        cls = "nav-link active" if key == active else "nav-link"
        nav_html.append('<a class="%s" href="%s">%s</a>' % (cls, href, html.escape(label)))

    filters_html = ""
    if show_filters:
        filters_html = """
<div class="toolbar">
  <button class="refresh" onclick="window.location.reload()">Refresh</button>
  <select><option>All Status</option><option>Open</option><option>Resolved</option></select>
  <select><option>All Severity</option><option>Low</option><option>Medium</option><option>High</option><option>Critical</option></select>
</div>
"""

    return """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fire Alert Dashboard</title>
<style>
* { box-sizing: border-box; }
body { margin:0; min-height:100vh; font-family: system-ui, sans-serif; background:#111317; color:#f3f4f6; }
.app { display:grid; grid-template-columns: 220px 1fr; min-height:100vh; }
.sidebar { background:#202329; border-right:1px solid #2f343c; display:flex; flex-direction:column; }
.brand { padding:14px 16px; border-bottom:1px solid #2f343c; }
.brand h1 { margin:0; font-size:1.85rem; color:#ff4b52; font-weight:800; line-height:1; letter-spacing:.2px; }
.brand small { display:block; margin-top:8px; color:#9ca3af; font-size:.78rem; }
.logout-btn { margin-top:10px; background:#0f1115; color:#fff; border:1px solid #3a414c; border-radius:6px; padding:6px 10px; font-size:.78rem; cursor:pointer; }
.nav { padding:14px 0; display:flex; flex-direction:column; gap:4px; }
.nav-link { color:#c7cbd3; text-decoration:none; padding:10px 16px; font-size:.94rem; border-left:3px solid transparent; }
.nav-link:hover { background:#1a1d22; color:#fff; }
.nav-link.active { background:#13161b; border-left-color:#ff4b52; color:#ff4b52; }
.main { display:flex; flex-direction:column; }
.topbar { height:56px; border-bottom:1px solid #2f343c; display:flex; align-items:center; justify-content:space-between; padding:0 20px; background:#23262d; }
.topbar h2 { margin:0; font-size:2rem; font-weight:700; color:#fff; }
.toolbar { display:flex; gap:10px; align-items:center; }
.refresh { background:#1677ff; color:#fff; border:none; border-radius:6px; padding:8px 16px; font-weight:600; cursor:pointer; }
select { background:#0f1115; color:#d6dae1; border:1px solid #343a44; border-radius:6px; padding:7px 10px; min-width:120px; }
.content { padding:20px; }
.panel { background:#23262d; border:1px solid #353a44; border-radius:8px; padding:14px; }
.panel h3 { margin:0 0 12px 0; font-size:1.05rem; }
.empty { min-height:120px; display:flex; align-items:center; justify-content:center; color:#9ca3af; }
.report-form { display:grid; gap:12px; max-width:720px; }
label { color:#d6dae1; font-size:.92rem; }
input, textarea, select.form { width:100%%; background:#101318; color:#f3f4f6; border:1px solid #373e49; border-radius:8px; padding:10px 12px; }
textarea { min-height:110px; resize:vertical; }
.submit { width:max-content; background:#ff4b52; color:#fff; border:none; border-radius:8px; padding:10px 16px; font-weight:700; cursor:pointer; }
table { width:100%%; border-collapse:collapse; font-size:.9rem; }
th, td { text-align:left; padding:9px 10px; border-bottom:1px solid #343a44; }
th { color:#d6dae1; font-weight:700; background:#1d2026; }
.muted { color:#9ca3af; }
.notice { margin-bottom:10px; color:#86efac; }
</style>
</head><body>
<div class="app">
  <aside class="sidebar">
    <div class="brand">
      <h1>🔥 Fire Alert</h1>
      <small>%s</small>
      <form method="post" action="/logout"><button class="logout-btn" type="submit">Logout</button></form>
    </div>
    <nav class="nav">%s</nav>
  </aside>
  <main class="main">
    <header class="topbar">
      <h2>%s</h2>
      %s
    </header>
    <section class="content">%s</section>
  </main>
</div>
</body></html>""" % (
        u,
        "".join(nav_html),
        html.escape(title),
        filters_html,
        content_html,
    )


def dashboard_page_html(username: str, recent_rows: list[tuple]) -> str:
    if recent_rows:
        items = []
        for ts, source, severity, details, image_path in recent_rows:
            img_html = '<span class="muted">-</span>'
            if image_path:
                img_html = '<img src="/home/detection-image?path=%s" style="width:88px;height:50px;object-fit:cover;border:1px solid #343a44;border-radius:6px;background:#0f1115;">' % urllib.parse.quote(
                    image_path, safe=""
                )
            items.append(
                "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                % (
                    html.escape(str(ts)),
                    html.escape(source),
                    html.escape(severity),
                    html.escape(details),
                    img_html,
                )
            )
        content = """
<div class="panel">
  <h3>Recent Alerts</h3>
  <table>
    <thead><tr><th>Time</th><th>Source</th><th>Severity</th><th>Details</th><th>Image</th></tr></thead>
    <tbody>%s</tbody>
  </table>
</div>""" % "".join(items)
    else:
        content = """
<div class="panel">
  <h3>Recent Alerts</h3>
  <div class="empty">No alerts found</div>
</div>"""
    return _dashboard_layout_html(username, "dashboard", "Fire Alert Dashboard", content, show_filters=True)


def report_fire_page_html(username: str, notice: str = "") -> str:
    notice_html = '<div class="notice">%s</div>' % html.escape(notice) if notice else ""
    content = """
<div class="panel">
  <h3>Report Fire</h3>
  %s
  <form class="report-form" method="post" action="/home/report-fire" enctype="multipart/form-data">
    <div><label>Address</label><input name="address" type="text" required></div>
    <div><label>Severity</label>
      <select class="form" name="severity">
        <option>Low</option><option selected>Medium</option><option>High</option><option>Critical</option>
      </select>
    </div>
    <div><label>Description</label><textarea name="description" required></textarea></div>
    <div><label>Upload Image (optional)</label><input type="file" name="image" accept=".jpg,.jpeg,.png,.webp,image/*"></div>
    <button class="submit" type="submit">Submit Report</button>
  </form>
</div>""" % notice_html
    return _dashboard_layout_html(username, "report-fire", "Report Fire", content)


def all_alerts_page_html(
    username: str,
    rows: list[tuple],
    report_filter_mode: str = "all",
    report_day: str = "",
    report_month: str = "",
    notice: str = "",
) -> str:
    notice_html = '<div class="notice">%s</div>' % html.escape(notice) if notice else ""
    day_val = html.escape(report_day or "")
    month_val = html.escape(report_month or "")
    mode = report_filter_mode if report_filter_mode in ("all", "day", "month") else "all"
    select_all = " selected" if mode == "all" else ""
    select_day = " selected" if mode == "day" else ""
    select_month = " selected" if mode == "month" else ""

    filter_html = """
<div class="panel" style="margin-bottom:12px;">
  <h3>Report Filter</h3>
  <form method="get" action="/home/all-alerts" style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
    <label style="margin:0;">Mode</label>
    <select class="form" name="report_mode" style="width:140px;">
      <option value="all"%s>All</option>
      <option value="day"%s>Day</option>
      <option value="month"%s>Month</option>
    </select>
    <label style="margin:0;">Day</label>
    <input type="date" name="report_day" value="%s" style="width:180px;">
    <label style="margin:0;">Month</label>
    <input type="month" name="report_month" value="%s" style="width:160px;">
    <button class="submit" type="submit">Apply</button>
  </form>
</div>""" % (select_all, select_day, select_month, day_val, month_val)

    if rows:
        body = []
        for ts, source, severity, details, status, image_path, record_type, record_id in rows:
            img_html = '<span class="muted">-</span>'
            if image_path:
                img_html = '<img src="/home/report-image?path=%s" style="width:88px;height:50px;object-fit:cover;border:1px solid #343a44;border-radius:6px;background:#0f1115;">' % urllib.parse.quote(
                    image_path, safe=""
                )
            action_html = '<span class="muted">-</span>'
            if record_type == "report":
                action_html = (
                    '<form method="post" action="/home/report-delete" style="margin:0;">'
                    '<input type="hidden" name="report_id" value="%s">'
                    '<input type="hidden" name="report_mode" value="%s">'
                    '<input type="hidden" name="report_day" value="%s">'
                    '<input type="hidden" name="report_month" value="%s">'
                    '<button class="submit" type="submit" style="padding:6px 10px;font-size:.82rem;">Delete</button>'
                    "</form>"
                    % (
                        html.escape(str(record_id)),
                        html.escape(mode),
                        day_val,
                        month_val,
                    )
                )
            body.append(
                "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                % (
                    html.escape(str(ts)),
                    html.escape(source),
                    html.escape(severity),
                    html.escape(details),
                    html.escape(status),
                    img_html,
                    action_html,
                )
            )
        table = """
<div class="panel">
  <h3>All Alerts</h3>
  <table>
    <thead><tr><th>Time</th><th>Source</th><th>Severity</th><th>Details</th><th>Status</th><th>Image</th><th>Action</th></tr></thead>
    <tbody>%s</tbody>
  </table>
</div>""" % "".join(body)
    else:
        table = '<div class="panel"><h3>All Alerts</h3><div class="empty">No alerts found</div></div>'
    return _dashboard_layout_html(username, "all-alerts", "All Alerts", notice_html + filter_html + table)


def camera_detection_page_html(
    username: str,
    rows: list[tuple],
    rtsp_url: str = "",
    stream_error: str = "",
    yolo_status: str = "",
    conf_value: float = CONF_THRESHOLD,
    selected_models: list[str] | None = None,
    available_models: list[str] | None = None,
) -> str:
    rtsp_url = (rtsp_url or "").strip()
    selected_models = selected_models or []
    available_models = available_models or []
    conf_value = max(0.01, min(0.99, float(conf_value)))
    escaped_rtsp = html.escape(rtsp_url)
    stream_box = ""
    if stream_error:
        stream_box = '<div class="panel" style="margin-bottom:12px;"><div class="muted">%s</div></div>' % html.escape(stream_error)
    elif rtsp_url:
        q = [("url", rtsp_url), ("conf", ("%.2f" % conf_value))]
        for m in selected_models:
            q.append(("models", m))
        stream_src = "/home/camera-detection/stream?" + urllib.parse.urlencode(q)
        stream_box = (
            '<div class="panel" style="margin-bottom:12px;">'
            '<div style="display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:8px;">'
            '<h3 style="margin:0;">Live RTSP Preview</h3>'
            '<button class="submit" type="button" onclick="toggleStreamFullScreen()">Full Screen</button>'
            "</div>"
            '<div id="streamWrap" style="width:100%%;height:min(78vh,900px);background:#0b0d10;border:1px solid #343a44;border-radius:6px;overflow:hidden;">'
            '<img id="rtspStream" src="%s" alt="RTSP stream" style="width:100%%;height:100%%;object-fit:contain;background:#0b0d10;">'
            "</div>"
            '<script>'
            "function toggleStreamFullScreen(){"
            "const el=document.getElementById('streamWrap');"
            "if(!el)return;"
            "if(!document.fullscreenElement){el.requestFullscreen&&el.requestFullscreen();}"
            "else{document.exitFullscreen&&document.exitFullscreen();}"
            "}"
            "</script>"
            "</div>"
        ) % stream_src

    if rows:
        body = []
        for ts, camera, status, severity, details in rows:
            body.append(
                "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                % (
                    html.escape(str(ts)),
                    html.escape(camera),
                    html.escape(status),
                    html.escape(severity),
                    html.escape(details),
                )
            )
        table_html = """
<div class="panel">
  <h3>Detection Log</h3>
  <table>
    <thead><tr><th>Time</th><th>Camera</th><th>Status</th><th>Severity</th><th>Details</th></tr></thead>
    <tbody>%s</tbody>
  </table>
</div>""" % "".join(body)
    else:
        table_html = '<div class="panel"><h3>Detection Log</h3><div class="empty">No camera detections found</div></div>'

    yolo_html = ""
    if yolo_status:
        yolo_html = '<div class="panel muted" style="margin-bottom:12px;font-size:0.88rem;">%s</div>' % html.escape(
            yolo_status
        )

    checks = []
    for m in available_models:
        checked = " checked" if m in selected_models else ""
        checks.append(
            '<label style="display:inline-flex;align-items:center;gap:6px;margin-right:12px;">'
            '<input type="checkbox" name="models" value="%s"%s style="width:auto;"> %s'
            "</label>" % (html.escape(m), checked, html.escape(m))
        )

    form_html = """
<div class="panel" style="margin-bottom:12px;">
  <h3>RTSP Camera URL</h3>
  <form method="get" action="/home/camera-detection" style="display:grid;gap:10px;">
    <input type="text" name="url" placeholder="rtsp://username:password@ip:554/Streaming/Channels/101" value="%s" style="flex:1;min-width:320px;">
    <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
      <label style="margin:0;">Confidence</label>
      <input type="number" name="conf" min="0.01" max="0.99" step="0.01" value="%s" style="width:100px;">
      <div class="muted" style="font-size:0.85rem;">0.01 - 0.99</div>
    </div>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
      <div class="muted" style="font-size:0.9rem;">Models:</div>
      %s
    </div>
    <button class="submit" type="submit">Connect</button>
  </form>
</div>""" % (escaped_rtsp, ("%.2f" % conf_value), "".join(checks))

    content = yolo_html + form_html + stream_box + table_html
    return _dashboard_layout_html(username, "camera-detection", "Camera Detection", content)


def main() -> None:
    host = "127.0.0.1"
    port = 8766
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == "--port" and i + 1 < len(argv):
            port = int(argv[i + 1])
            i += 2
            continue
        if argv[i] == "--host" and i + 1 < len(argv):
            host = argv[i + 1]
            i += 2
            continue
        if argv[i] in ("-h", "--help"):
            print(__doc__.strip())
            sys.exit(0)
        print("Unknown argument:", argv[i], file=sys.stderr)
        sys.exit(2)

    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    _init_db(conn)
    cursor = conn.cursor()

    yolo_models = load_fire_models()
    yolo_model_names = list(yolo_models.keys())
    if yolo_models:
        yolo_status_text = (
            "Fire detection: best-kiase.pt + best_fire.pt active (conf≥%.0f%%, class 0). "
            "Requires: pip install ultralytics opencv-python"
            % (CONF_THRESHOLD * 100)
        )
    else:
        yolo_status_text = (
            "Fire detection models not loaded. Place best-kiase.pt and best_fire.pt next to this script "
            "and install: pip install ultralytics opencv-python"
        )

    sessions: dict[str, dict] = {}
    sess_lock = threading.Lock()

    def prune_sessions() -> None:
        now = time.time()
        with sess_lock:
            for t in [x for x, s in sessions.items() if s["until"] < now]:
                del sessions[t]

    def session_user(handler: BaseHTTPRequestHandler) -> str | None:
        prune_sessions()
        tok = _parse_cookies(handler.headers.get("Cookie", "")).get(_SESSION_COOKIE)
        if not tok:
            return None
        with sess_lock:
            s = sessions.get(tok)
            if not s or s["until"] < time.time():
                return None
            return s["username"]

    state = {
        "conn": conn,
        "cursor": cursor,
        "sessions": sessions,
        "sess_lock": sess_lock,
        "yolo_models": yolo_models,
        "yolo_model_names": yolo_model_names,
        "yolo_status_text": yolo_status_text,
        "rtsp_bridge": RtspBridge(state={"conn": conn, "cursor": cursor}, models_map=yolo_models),
    }

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _send_html(self, html_str: str, status: int = 200) -> None:
            body = html_str.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _redirect(self, location: str, clear_cookie: bool = False) -> None:
            self.send_response(302)
            self.send_header("Location", location)
            if clear_cookie:
                self.send_header(
                    "Set-Cookie",
                    "%s=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0" % _SESSION_COOKIE,
                )
            self.end_headers()

        def _read_form(self) -> dict | None:
            n = int(self.headers.get("Content-Length", 0))
            if n > 2_000_000:
                return None
            raw = self.rfile.read(n).decode("utf-8", errors="replace")
            return urllib.parse.parse_qs(raw, keep_blank_values=True)

        def _one(self, fields: dict, key: str) -> str:
            v = fields.get(key, [""])
            return v[0] if v else ""

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)

            if parsed.path == "/health":
                b = b'{"ok":true,"service":"fire-alert-browser-portal"}\n'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
                return

            if parsed.path == "/logout":
                tok = _parse_cookies(self.headers.get("Cookie", "")).get(_SESSION_COOKIE)
                if tok:
                    with state["sess_lock"]:
                        state["sessions"].pop(tok, None)
                self._redirect("/", clear_cookie=True)
                return

            if parsed.path == "/home":
                u = session_user(self)
                if not u:
                    self._redirect("/")
                    return
                state["cursor"].execute(
                    """
                    SELECT created_at, source, severity, details, image_path
                    FROM (
                        SELECT created_at, 'Report Fire' AS source, severity, description AS details, NULL AS image_path, id
                        FROM fire_reports
                        UNION ALL
                        SELECT created_at, camera_name AS source, severity, details, image_path, id
                        FROM camera_detections
                    )
                    ORDER BY created_at DESC
                    LIMIT 10
                    """
                )
                rows = state["cursor"].fetchall()
                self._send_html(dashboard_page_html(u, rows))
                return

            if parsed.path == "/home/detection-image":
                u = session_user(self)
                if not u:
                    self.send_error(401, "Login required")
                    return
                req_path = (qs.get("path") or [""])[0].strip()
                if not req_path:
                    self.send_error(404, "Not found")
                    return
                abs_path = os.path.abspath(req_path)
                if not abs_path.startswith(os.path.abspath(_LOG_DIR) + os.sep) or not os.path.isfile(abs_path):
                    self.send_error(404, "Not found")
                    return
                ext = os.path.splitext(abs_path)[1].lower()
                ctype = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png" if ext == ".png" else "image/webp" if ext == ".webp" else "application/octet-stream"
                with open(abs_path, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "private, max-age=120")
                self.end_headers()
                self.wfile.write(data)
                return

            if parsed.path == "/home/report-fire":
                u = session_user(self)
                if not u:
                    self._redirect("/")
                    return
                self._send_html(report_fire_page_html(u))
                return

            if parsed.path == "/home/all-alerts":
                u = session_user(self)
                if not u:
                    self._redirect("/")
                    return
                report_mode = (qs.get("report_mode") or ["all"])[0].strip().lower()
                report_day = (qs.get("report_day") or [""])[0].strip()
                report_month = (qs.get("report_month") or [""])[0].strip()
                if report_mode not in ("all", "day", "month"):
                    report_mode = "all"
                rep_where = ""
                rep_params: list = []
                if report_mode == "day" and re.match(r"^\d{4}-\d{2}-\d{2}$", report_day):
                    rep_where = " WHERE substr(created_at,1,10)=? "
                    rep_params.append(report_day)
                elif report_mode == "month" and re.match(r"^\d{4}-\d{2}$", report_month):
                    rep_where = " WHERE substr(created_at,1,7)=? "
                    rep_params.append(report_month)
                state["cursor"].execute(
                    """
                    SELECT created_at, source, severity, details, status, image_path, record_type, record_id
                    FROM (
                        SELECT created_at, 'Report Fire' AS source, severity, description AS details, 'Open' AS status, image_path, 'report' AS record_type, id AS record_id
                        FROM fire_reports
                        %s
                        UNION ALL
                        SELECT created_at, camera_name AS source, severity, COALESCE(details, '') AS details, status, image_path, 'detection' AS record_type, id AS record_id
                        FROM camera_detections
                    )
                    ORDER BY created_at DESC
                    LIMIT 300
                    """
                    % rep_where,
                    rep_params,
                )
                rows = state["cursor"].fetchall()
                self._send_html(
                    all_alerts_page_html(
                        u,
                        rows,
                        report_filter_mode=report_mode,
                        report_day=report_day,
                        report_month=report_month,
                    )
                )
                return

            if parsed.path == "/home/report-image":
                u = session_user(self)
                if not u:
                    self.send_error(401, "Login required")
                    return
                req_path = (qs.get("path") or [""])[0].strip()
                if not req_path:
                    self.send_error(404, "Not found")
                    return
                abs_path = os.path.abspath(req_path)
                if not abs_path.startswith(os.path.abspath(_REPORT_IMAGE_DIR) + os.sep) or not os.path.isfile(abs_path):
                    self.send_error(404, "Not found")
                    return
                ext = os.path.splitext(abs_path)[1].lower()
                ctype = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png" if ext == ".png" else "image/webp" if ext == ".webp" else "application/octet-stream"
                with open(abs_path, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "private, max-age=120")
                self.end_headers()
                self.wfile.write(data)
                return

            if parsed.path == "/home/camera-detection":
                u = session_user(self)
                if not u:
                    self._redirect("/")
                    return
                rtsp_url = (qs.get("url") or [""])[0].strip()
                conf_raw = (qs.get("conf") or [str(CONF_THRESHOLD)])[0].strip()
                try:
                    conf_value = max(0.01, min(0.99, float(conf_raw)))
                except Exception:
                    conf_value = CONF_THRESHOLD
                selected_models = [m for m in (qs.get("models") or []) if m in state["yolo_model_names"]]
                if not selected_models:
                    selected_models = list(state["yolo_model_names"])
                stream_error = ""
                if rtsp_url:
                    if not rtsp_url.lower().startswith("rtsp://"):
                        stream_error = "Invalid RTSP URL. It must start with rtsp://"
                    elif cv2 is None:
                        stream_error = "OpenCV is not available. Install with: pip install opencv-python"
                state["cursor"].execute(
                    "SELECT created_at, camera_name, status, severity, COALESCE(details, '') FROM camera_detections ORDER BY created_at DESC LIMIT 300"
                )
                rows = state["cursor"].fetchall()
                self._send_html(
                    camera_detection_page_html(
                        u,
                        rows,
                        rtsp_url=rtsp_url,
                        stream_error=stream_error,
                        yolo_status=state["yolo_status_text"],
                        conf_value=conf_value,
                        selected_models=selected_models,
                        available_models=state["yolo_model_names"],
                    )
                )
                return

            if parsed.path == "/home/camera-detection/stream":
                u = session_user(self)
                if not u:
                    self.send_error(401, "Login required")
                    return
                if cv2 is None:
                    self.send_error(503, "OpenCV not installed")
                    return
                rtsp_url = (qs.get("url") or [""])[0].strip()
                if not rtsp_url.lower().startswith("rtsp://"):
                    self.send_error(400, "url must start with rtsp://")
                    return
                conf_raw = (qs.get("conf") or [str(CONF_THRESHOLD)])[0].strip()
                try:
                    conf_value = max(0.01, min(0.99, float(conf_raw)))
                except Exception:
                    conf_value = CONF_THRESHOLD
                selected_models = [m for m in (qs.get("models") or []) if m in state["yolo_model_names"]]
                if not selected_models:
                    selected_models = list(state["yolo_model_names"])
                state["rtsp_bridge"].ensure_stream(rtsp_url, selected_models=selected_models, conf_threshold=conf_value)

                deadline = time.time() + 10.0
                while time.time() < deadline and state["rtsp_bridge"].get_jpeg() is None:
                    time.sleep(0.05)
                if state["rtsp_bridge"].get_jpeg() is None:
                    self.send_error(504, "RTSP stream not ready")
                    return

                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=" + _MJPEG_BOUNDARY.decode(),
                )
                self.send_header("Cache-Control", "no-cache, no-store")
                self.end_headers()
                try:
                    while True:
                        chunk = state["rtsp_bridge"].get_jpeg()
                        if chunk:
                            self.wfile.write(b"--" + _MJPEG_BOUNDARY + b"\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                            self.wfile.write(chunk)
                            self.wfile.write(b"\r\n")
                        time.sleep(1.0 / 20.0)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
                return

            if parsed.path == "/":
                if session_user(self):
                    self._redirect("/home")
                    return
                view = (qs.get("view") or [""])[0].lower()
                active = "register" if view == "register" else "login"
                self._send_html(auth_page_html(active))
                return

            self.send_error(404)

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)

            if parsed.path == "/home/report-fire":
                u = session_user(self)
                if not u:
                    self._redirect("/")
                    return
                ctype = (self.headers.get("Content-Type", "") or "").lower()
                address = ""
                severity = "Medium"
                description = ""
                image_path = None

                if "multipart/form-data" in ctype:
                    fs = cgi.FieldStorage(
                        fp=self.rfile,
                        headers=self.headers,
                        environ={
                            "REQUEST_METHOD": "POST",
                            "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                        },
                        keep_blank_values=True,
                    )
                    address = (fs.getvalue("address") or "").strip()
                    severity = (fs.getvalue("severity") or "Medium").strip()
                    description = (fs.getvalue("description") or "").strip()
                    if "image" in fs:
                        img = fs["image"]
                        if getattr(img, "filename", "") and getattr(img, "file", None):
                            raw = img.file.read()
                            if raw and len(raw) <= 5 * 1024 * 1024:
                                ext = os.path.splitext(img.filename.lower())[1]
                                if ext not in (".jpg", ".jpeg", ".png", ".webp"):
                                    ext = ".jpg"
                                safe = re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(img.filename)[0])[:50] or "report"
                                fname = "report_%s_%s%s" % (datetime.now().strftime("%Y%m%d_%H%M%S"), safe, ext)
                                abs_path = os.path.join(_REPORT_IMAGE_DIR, fname)
                                with open(abs_path, "wb") as f:
                                    f.write(raw)
                                image_path = abs_path
                else:
                    fields = self._read_form()
                    if fields is None:
                        self.send_error(413)
                        return
                    address = self._one(fields, "address").strip()
                    severity = self._one(fields, "severity").strip() or "Medium"
                    description = self._one(fields, "description").strip()

                if not address or not description:
                    self._send_html(report_fire_page_html(u, notice="Address and description are required."))
                    return
                if severity not in ("Low", "Medium", "High", "Critical"):
                    severity = "Medium"
                state["cursor"].execute(
                    "INSERT INTO fire_reports (created_at, created_by, address, severity, description, image_path) VALUES (?,?,?,?,?,?)",
                    (
                        datetime.now().isoformat(timespec="seconds"),
                        u,
                        address,
                        severity,
                        description,
                        image_path,
                    ),
                )
                state["conn"].commit()
                self._send_html(report_fire_page_html(u, notice="Report submitted successfully."))
                return

            if parsed.path == "/home/report-delete":
                u = session_user(self)
                if not u:
                    self._redirect("/")
                    return
                report_id_raw = self._one(fields, "report_id").strip()
                report_mode = self._one(fields, "report_mode").strip() or "all"
                report_day = self._one(fields, "report_day").strip()
                report_month = self._one(fields, "report_month").strip()
                try:
                    report_id = int(report_id_raw)
                except Exception:
                    self._redirect("/home/all-alerts")
                    return
                state["cursor"].execute("SELECT image_path FROM fire_reports WHERE id = ?", (report_id,))
                row = state["cursor"].fetchone()
                if row:
                    image_path = row[0]
                    state["cursor"].execute("DELETE FROM fire_reports WHERE id = ?", (report_id,))
                    state["conn"].commit()
                    try:
                        if image_path:
                            abs_path = os.path.abspath(image_path)
                            if abs_path.startswith(os.path.abspath(_REPORT_IMAGE_DIR) + os.sep) and os.path.isfile(abs_path):
                                os.remove(abs_path)
                    except Exception:
                        pass
                q = urllib.parse.urlencode(
                    {
                        "report_mode": report_mode,
                        "report_day": report_day,
                        "report_month": report_month,
                    }
                )
                self._redirect("/home/all-alerts?" + q)
                return

            fields = self._read_form()
            if fields is None:
                self.send_error(413)
                return

            if parsed.path == "/login":
                ident = self._one(fields, "username").strip()
                password = self._one(fields, "password")
                row = None
                if ident:
                    state["cursor"].execute(
                        "SELECT username, password_hash FROM users WHERE username = ? OR lower(email) = lower(?)",
                        (ident, ident),
                    )
                    row = state["cursor"].fetchone()
                if not row or not _pbkdf2_verify_password(password, row[1]):
                    self._send_html(auth_page_html("login", error="Invalid username/email or password."))
                    return
                token = secrets.token_urlsafe(32)
                with state["sess_lock"]:
                    state["sessions"][token] = {"username": row[0], "until": time.time() + _SESSION_TTL_SEC}
                self.send_response(302)
                self.send_header("Location", "/home")
                self.send_header(
                    "Set-Cookie",
                    "%s=%s; Path=/; HttpOnly; SameSite=Lax; Max-Age=%d"
                    % (_SESSION_COOKIE, token, _SESSION_TTL_SEC),
                )
                self.end_headers()
                return

            if parsed.path == "/register":
                username = self._one(fields, "username").strip()
                email = self._one(fields, "email").strip()
                password = self._one(fields, "password")
                confirm = self._one(fields, "confirm")
                if len(username) < 2 or len(username) > 64:
                    self._send_html(auth_page_html("register", error="Username must be 2–64 characters."))
                    return
                if not re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", email):
                    self._send_html(auth_page_html("register", error="Enter a valid email address."))
                    return
                if len(password) < 6:
                    self._send_html(auth_page_html("register", error="Password must be at least 6 characters."))
                    return
                if password != confirm:
                    self._send_html(auth_page_html("register", error="Passwords do not match."))
                    return
                try:
                    state["cursor"].execute(
                        "INSERT INTO users (username, email, password_hash, created_at) VALUES (?,?,?,?)",
                        (
                            username,
                            email.lower(),
                            _pbkdf2_hash_password(password),
                            datetime.now().isoformat(timespec="seconds"),
                        ),
                    )
                    state["conn"].commit()
                except sqlite3.IntegrityError:
                    self._send_html(auth_page_html("register", error="That username or email is already registered."))
                    return
                self._send_html(auth_page_html("login", notice="Account created. You can sign in now."))
                return

            if parsed.path == "/logout":
                tok = _parse_cookies(self.headers.get("Cookie", "")).get(_SESSION_COOKIE)
                if tok:
                    with state["sess_lock"]:
                        state["sessions"].pop(tok, None)
                self._redirect("/", clear_cookie=True)
                return

            self.send_error(404)

    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = ThreadingHTTPServer((host, port), Handler)
    url = "http://%s:%s/" % (host, port)
    print("Fire Alert System (browser): %s" % url)
    print("Database: %s" % _DB_PATH)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            state["rtsp_bridge"].stop()
        except Exception:
            pass
        server.server_close()
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

# Bootstrap: set crash handler and exe working dir FIRST (so "Run as administrator" works)
import os
import sys
import traceback

def _crash_handler(etype, value, tb):
    msg = "".join(traceback.format_exception(etype, value, tb))
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        log_path = os.path.join(exe_dir, "fire_alert_error.log")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(msg)
        except Exception:
            pass
        try:
            import tkinter as _tk
            from tkinter import messagebox as _mb
            root = _tk.Tk()
            root.withdraw()
            _mb.showerror("Fire Alert Error", "An error occurred. See fire_alert_error.log next to the exe.\n\n" + str(value))
            root.destroy()
        except Exception:
            pass
    else:
        print(msg, file=sys.stderr)
        try:
            input("Press Enter to exit...")
        except Exception:
            pass
    sys.exit(1)

sys.excepthook = _crash_handler

# When running as exe (e.g. "Run as administrator"), start in exe folder so config/logs work
if getattr(sys, "frozen", False):
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    try:
        os.chdir(exe_dir)
    except Exception:
        pass

import time
import sqlite3
import threading
import hashlib
import secrets
import re
import html
import cgi
from collections import deque
import numpy as np
import cv2
from datetime import datetime
from ultralytics import YOLO
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import urllib.parse
import webbrowser

# ------------------------------
# STOP WINDOW JUMP
# ------------------------------
os.environ['SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS'] = '0'

# ------------------------------
# PATH HANDLING (PyInstaller)
# ------------------------------
def resource_path(relative):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative)

# ------------------------------
# CONFIG
# ------------------------------
RTSP_CAM1 = "rtsp://admin:12qw!@QW@192.168.0.65:554/Streaming/Channels/101"
RTSP_CAM2 = "rtsp://admin:admin123456@192.168.0.105:554/ch=1&subtype=1"
DEFAULT_RTSP = RTSP_CAM1  # fallback single cam when no URLs given
DEFAULT_RTSP_URLS = [RTSP_CAM1, RTSP_CAM2]  # default list for dialog (both cams)
MODEL_OPTIONS = ["best-kiase.pt","best1.pt", "best.pt", "best2.pt","best3.pt"]  # select in dialog
MODEL_PATH = resource_path(MODEL_OPTIONS[0])  # overwritten by dialog selection

CELL_W, CELL_H = 480, 270   # each camera cell size (grid)
WIDTH, HEIGHT = CELL_W, CELL_H
MAX_CAMERAS = 16            # 1–16 CCTV streams
FPS = 20
CONF_THRESHOLD = 0.85 # confidence threshold for fire detection (higher = fewer false positives)
ALERT_COOLDOWN = 15  # seconds between recording alerts (screenshot, DB, sound) per camera
MESSAGE_COOLDOWN = 60  # seconds between sending WhatsApp messages (global; avoids flooding)
DETECT_EVERY_N = 3   # run YOLO every N frames (3 = responsive tracking of moving flames)

RECONNECT_DELAY = 3  # seconds before retry
RECONNECT_MAX = 20  # max reconnect attempts
READ_RETRIES = 20   # retry failed reads before declaring stream lost (fewer false "Reconnecting")

# Low-confidence: treat as review-only (log + save image, no WhatsApp)
LOW_CONFIDENCE_THRESHOLD = 0.6   # any detection with conf < this is low-confidence (no WhatsApp)

# WhatsApp: load numbers from whatsapp_phone_number.txt (one per line or comma-separated), else from env
def _whatsapp_phone_file_path():
    if getattr(sys, "frozen", False):
        return os.path.join(os.path.dirname(sys.executable), "whatsapp_phone_number.txt")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "whatsapp_phone_number.txt")

def load_whatsapp_phones_from_file():
    path = _whatsapp_phone_file_path()
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            return [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip() and p.strip().startswith("+")]
    except Exception:
        pass
    return []

def save_whatsapp_phones_to_file(phones: list):
    """Save WhatsApp phone numbers to file (one per line)."""
    path = _whatsapp_phone_file_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(phones))
        return True
    except Exception:
        return False

_wa = (os.environ.get("FIRE_ALERT_WHATSAPP_PHONE", "") or "").strip()
_wa_env = [p.strip() for p in _wa.replace("\n", ",").split(",") if p.strip() and p.strip().startswith("+")]
WHATSAPP_PHONES = load_whatsapp_phones_from_file() or _wa_env
WHATSAPP_ENABLED = bool(WHATSAPP_PHONES)

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Set by main() when run as application (or CLI/dialog)
RTSP_URLS = []
FULLSCREEN = True
HIGH_DEFINITION = True  # default on high definition

# ------------------------------
# DATABASE (SQLite)
# ------------------------------
db = sqlite3.connect("fire_alert.db")
cursor = db.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS fire_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    confidence REAL,
    image_path TEXT,
    camera_index INTEGER DEFAULT 0
)
""")
try:
    cursor.execute("ALTER TABLE fire_events ADD COLUMN camera_index INTEGER DEFAULT 0")
except Exception:
    pass
try:
    cursor.execute("ALTER TABLE fire_events ADD COLUMN low_confidence INTEGER DEFAULT 0")
except Exception:
    pass
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TEXT
)
""")
db.commit()


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


def ensure_users_table(conn):
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT
    )
    """)
    conn.commit()


def ensure_reports_table(conn):
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS fire_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            created_by TEXT NOT NULL,
            address TEXT NOT NULL,
            severity TEXT NOT NULL,
            description TEXT NOT NULL,
            photo_path TEXT
        )
        """
    )
    conn.commit()


_SESSION_COOKIE = "fire_alert_session"
_SESSION_TTL_SEC = 86400 * 7


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


def web_auth_html(active: str, error: str = "", notice: str = "") -> str:
    """Login / Register page — dark card UI matching Fire Alert System branding."""
    active = "register" if active == "register" else "login"
    err_html = ('<p class="err">%s</p>' % error) if error else ""
    ok_html = ('<p class="ok">%s</p>' % notice) if notice else ""
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
  --bg: #0c0c0e;
  --card: #1e1e22;
  --input: #141416;
  --muted: #9ca3af;
  --text: #f3f4f6;
  --accent: #f97316;
  --accent-dim: #ea580c;
  --line: #2a2a30;
}
* { box-sizing: border-box; }
body {
  margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center;
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  background: var(--bg); color: var(--text);
  padding: 24px;
}
.card {
  width: 100%%; max-width: 420px;
  background: var(--card);
  border-radius: 14px;
  padding: 28px 28px 32px;
  box-shadow: 0 20px 50px rgba(0,0,0,.45), 0 0 0 1px rgba(255,255,255,.04);
}
.brand { display: flex; align-items: flex-start; gap: 12px; margin-bottom: 8px; }
.flame {
  font-size: 1.75rem; line-height: 1;
  filter: drop-shadow(0 2px 8px rgba(249,115,22,.5));
}
.brand h1 {
  margin: 0; font-size: 1.35rem; font-weight: 700;
  background: linear-gradient(135deg, #fb923c, #ef4444);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.tagline { margin: 6px 0 0; font-size: 0.88rem; color: var(--muted); line-height: 1.4; }
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
label { font-size: 0.8rem; font-weight: 500; color: var(--text); }
input[type="text"], input[type="email"], input[type="password"] {
  width: 100%%; padding: 12px 14px; border-radius: 8px; border: 1px solid var(--line);
  background: var(--input); color: var(--text); font-size: 0.95rem; outline: none;
}
input:focus { border-color: rgba(249,115,22,.45); box-shadow: 0 0 0 3px rgba(249,115,22,.12); }
.btn {
  width: 100%%; margin-top: 6px; padding: 13px; border: none; border-radius: 8px;
  font-size: 1rem; font-weight: 700; color: #fff; cursor: pointer;
  background: linear-gradient(180deg, var(--accent), var(--accent-dim));
  box-shadow: 0 4px 14px rgba(249,115,22,.25);
}
.btn:hover { filter: brightness(1.05); }
.btn:active { transform: translateY(1px); }
.err { color: #fca5a5; font-size: 0.85rem; margin: 0 0 4px; }
.ok { color: #86efac; font-size: 0.85rem; margin: 0 0 4px; }
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
  <form class="panel login" method="post" action="/login" style="display:%s;flex-direction:column;gap:16px;margin-top:22px;">
    <div>
      <label for="login_user">Username or Email</label>
      <input id="login_user" name="username" type="text" autocomplete="username" required placeholder=" ">
    </div>
    <div>
      <label for="login_pass">Password</label>
      <input id="login_pass" name="password" type="password" autocomplete="current-password" required>
    </div>
    <button type="submit" class="btn">Login</button>
  </form>
  <form class="panel reg" method="post" action="/register" style="display:%s;flex-direction:column;gap:16px;margin-top:22px;">
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


def web_dashboard_html(num_streams: int) -> str:
    rows = []
    for j in range(num_streams):
        rows.append(
            '<div class="cam"><h3>Camera %d</h3><img src="/stream/%d" alt="Cam %d"></div>'
            % (j + 1, j, j + 1)
        )
    return """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fire Alert – Live</title>
<style>
body { font-family: system-ui, sans-serif; background: #1a1a1a; color: #eee; margin: 0; padding: 16px; }
.top { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-bottom: 12px; }
.top h1 { font-size: 1.25rem; margin: 0; }
.top a { color: #f97316; text-decoration: none; font-size: 0.9rem; }
.top a:hover { text-decoration: underline; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }
.cam { background: #252525; border-radius: 8px; padding: 8px; }
.cam h3 { margin: 0 0 8px 0; font-size: 0.95rem; }
.cam img { width: 100%%; height: auto; min-height: 200px; object-fit: contain; background: #111; border-radius: 4px; }
</style></head><body>
<div class="top">
  <h1>Fire Alert (live)</h1>
  <div style="display:flex;gap:12px;align-items:center;">
    <a href="/alerts">All Alerts</a>
    <a href="/report">Report Fire Alert</a>
    <form method="post" action="/logout" style="margin:0;"><button type="submit" class="linkbtn" style="background:none;border:none;color:#f97316;cursor:pointer;font:inherit;">Log out</button></form>
  </div>
</div>
<p style="opacity:0.8;font-size:0.9rem;">MJPEG streams. Press Ctrl+C in the terminal to stop the server.</p>
<div class="grid">
%s
</div>
</body></html>""" % "\n".join(rows)


def web_report_html(error: str = "", notice: str = "", values: dict | None = None) -> str:
    values = values or {}
    address = html.escape(values.get("address", ""))
    severity = values.get("severity", "Medium")
    description = html.escape(values.get("description", ""))
    err_html = ('<p class="err">%s</p>' % html.escape(error)) if error else ""
    ok_html = ('<p class="ok">%s</p>' % html.escape(notice)) if notice else ""
    severity_options = []
    for s in ["Low", "Medium", "High", "Critical"]:
        sel = " selected" if severity == s else ""
        severity_options.append('<option value="%s"%s>%s</option>' % (s, sel, s))
    return """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Report Fire Alert</title>
<style>
body { font-family: system-ui, sans-serif; background: #0c0c0e; color: #eee; margin: 0; padding: 24px; }
.wrap { max-width: 760px; margin: 0 auto; }
.top { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 12px; }
.top h1 { margin: 0; font-size: 1.25rem; }
.top a { color: #f97316; text-decoration: none; font-size: 0.92rem; }
.card { background: #1e1e22; border: 1px solid #2a2a30; border-radius: 12px; padding: 18px; }
.group { margin-bottom: 16px; }
.group h2 { margin: 0 0 8px 0; font-size: 1rem; color: #f3f4f6; }
label { display: block; margin-bottom: 6px; font-size: 0.88rem; color: #d1d5db; }
input[type="text"], select, textarea, input[type="file"] {
  width: 100%%; padding: 11px 12px; border-radius: 8px; border: 1px solid #2a2a30;
  background: #141416; color: #f3f4f6; font-size: 0.95rem;
}
textarea { min-height: 120px; resize: vertical; }
.btn {
  background: linear-gradient(180deg, #f97316, #ea580c); color: #fff; border: none;
  border-radius: 8px; padding: 12px 16px; font-weight: 700; cursor: pointer;
}
.err { color: #fca5a5; margin: 0 0 10px 0; font-size: 0.9rem; }
.ok { color: #86efac; margin: 0 0 10px 0; font-size: 0.9rem; }
</style></head><body>
<div class="wrap">
  <div class="top">
    <h1>Report Fire Alert</h1>
    <div style="display:flex;gap:12px;align-items:center;">
      <a href="/dashboard">Dashboard</a>
      <a href="/alerts">All Alerts</a>
    </div>
  </div>
  <form class="card" method="post" action="/report" enctype="multipart/form-data">
    %s%s
    <div class="group">
      <h2>Location</h2>
      <label for="address">Address</label>
      <input id="address" name="address" type="text" required maxlength="255" value="%s" placeholder="Enter address...">
    </div>
    <div class="group">
      <h2>Alert Details</h2>
      <label for="severity">Severity</label>
      <select id="severity" name="severity">%s</select>
      <label for="description" style="margin-top:10px;">Description</label>
      <textarea id="description" name="description" required maxlength="2000" placeholder="Describe the incident...">%s</textarea>
      <label for="photo" style="margin-top:10px;">Upload Photo (optional)</label>
      <input id="photo" name="photo" type="file" accept=".jpg,.jpeg,.png,.webp,image/*">
    </div>
    <button type="submit" class="btn">Submit Report</button>
  </form>
</div>
</body></html>""" % (err_html, ok_html, address, "".join(severity_options), description)


def _resolved_log_path(stored: str) -> str | None:
    """Ensure stored path points to a file under LOG_DIR (no path traversal)."""
    if not stored or not isinstance(stored, str):
        return None
    p = stored.strip().replace("/", os.sep)
    if os.path.isabs(p):
        abs_p = os.path.normpath(p)
    else:
        abs_p = os.path.normpath(os.path.join(os.getcwd(), p))
    root = os.path.abspath(LOG_DIR)
    if not abs_p.startswith(root + os.sep) and abs_p != root:
        return None
    if not os.path.isfile(abs_p):
        return None
    return abs_p


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def web_alerts_html(detection_rows: list, report_rows: list) -> str:
    """List CCTV detection alerts and user-submitted fire reports."""
    det_cells = []
    for rid, ts, conf, cam_idx, low_conf, img_path in detection_rows:
        low_lbl = "Yes" if low_conf else "No"
        img_html = (
            '<img class="thumb" src="/image/event/%d" alt="">' % rid
            if img_path
            else '<span class="muted">—</span>'
        )
        det_cells.append(
            "<tr><td>%s</td><td>%s</td><td>%s</td><td>%.0f%%</td><td>%s</td><td>%s</td></tr>"
            % (
                html.escape(str(rid)),
                html.escape(str(ts)),
                html.escape("Camera %d" % (int(cam_idx) + 1)),
                float(conf or 0) * 100.0,
                html.escape(low_lbl),
                img_html,
            )
        )
    if not det_cells:
        det_cells.append('<tr><td colspan="6" class="muted">No detection alerts yet.</td></tr>')

    rep_cells = []
    for rid, created, user, addr, sev, desc, photo_path in report_rows:
        thumb = (
            '<img class="thumb" src="/image/report/%d" alt="">' % rid
            if photo_path
            else '<span class="muted">—</span>'
        )
        rep_cells.append(
            "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td><span class=\"sev\">%s</span></td><td>%s</td><td>%s</td></tr>"
            % (
                html.escape(str(rid)),
                html.escape(str(created)),
                html.escape(str(user)),
                html.escape(str(addr)),
                html.escape(str(sev)),
                html.escape(_truncate(str(desc), 240)),
                thumb,
            )
        )
    if not rep_cells:
        rep_cells.append('<tr><td colspan="7" class="muted">No submitted reports yet.</td></tr>')

    return """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>All Alerts</title>
<style>
body { font-family: system-ui, sans-serif; background: #0c0c0e; color: #eee; margin: 0; padding: 24px; }
.wrap { max-width: 1200px; margin: 0 auto; }
.top { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
.top h1 { margin: 0; font-size: 1.25rem; }
.nav { display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }
.nav a { color: #f97316; text-decoration: none; font-size: 0.92rem; }
.nav a:hover { text-decoration: underline; }
section { margin-bottom: 28px; }
section h2 { font-size: 1.05rem; margin: 0 0 10px 0; color: #f3f4f6; }
table { width: 100%%; border-collapse: collapse; background: #1e1e22; border: 1px solid #2a2a30; border-radius: 10px; overflow: hidden; font-size: 0.88rem; }
th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #2a2a30; vertical-align: top; }
th { background: #18181b; color: #d4d4d8; font-weight: 600; }
tr:last-child td { border-bottom: none; }
.muted { color: #9ca3af; }
.thumb { max-width: 120px; max-height: 72px; object-fit: cover; border-radius: 6px; background: #111; }
.sev { font-weight: 600; color: #fb923c; }
</style></head><body>
<div class="wrap">
  <div class="top">
    <h1>All Alerts</h1>
    <div class="nav">
      <a href="/dashboard">Dashboard</a>
      <a href="/report">Report Fire Alert</a>
      <form method="post" action="/logout" style="margin:0;"><button type="submit" style="background:none;border:none;color:#f97316;cursor:pointer;font:inherit;">Log out</button></form>
    </div>
  </div>
  <section>
    <h2>CCTV detection alerts</h2>
    <table>
      <thead><tr><th>ID</th><th>Time</th><th>Camera</th><th>Confidence</th><th>Low conf.</th><th>Image</th></tr></thead>
      <tbody>%s</tbody>
    </table>
  </section>
  <section>
    <h2>Submitted reports</h2>
    <table>
      <thead><tr><th>ID</th><th>Time</th><th>Reported by</th><th>Address</th><th>Severity</th><th>Description</th><th>Photo</th></tr></thead>
      <tbody>%s</tbody>
    </table>
  </section>
</div>
</body></html>""" % (
        "\n".join(det_cells),
        "\n".join(rep_cells),
    )


def draw_screenshot_overlays(frame, camera_index: int):
    """Draw date/time (top-left) and Camera XX (bottom-right) on frame for saved screenshot."""
    h, w = frame.shape[:2]
    time_str = datetime.now().strftime("%d-%m-%Y %a %H:%M:%S")
    cam_label = f"Camera {camera_index + 1:02d}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)
    shadow = (0, 0, 0)
    cv2.putText(frame, time_str, (8, 22), font, font_scale, shadow, thickness + 1)
    cv2.putText(frame, time_str, (6, 20), font, font_scale, color, thickness)
    (tw, th), _ = cv2.getTextSize(cam_label, font, font_scale, thickness)
    cx, cy = w - tw - 12, h - 10
    cv2.putText(frame, cam_label, (cx + 2, cy + 2), font, font_scale, shadow, thickness + 1)
    cv2.putText(frame, cam_label, (cx, cy), font, font_scale, color, thickness)


def is_local_camera(source: str) -> bool:
    """True if source is local webcam: 0, 1, localhost, webcam, local."""
    if not source or not isinstance(source, str):
        return False
    s = source.strip().lower()
    if s in ("0", "1", "localhost", "webcam", "local"):
        return True
    if s.isdigit() and int(s) >= 0:
        return True
    return False


def get_camera_ip_from_rtsp(rtsp_url: str):
    """Extract host IP from RTSP URL; return 'Local' for localhost/webcam."""
    if is_local_camera(rtsp_url):
        return "Local"
    try:
        from urllib.parse import urlparse
        parsed = urlparse(rtsp_url)
        return parsed.hostname or "N/A"
    except Exception:
        return "N/A"


def put_image_in_clipboard(image_path: str) -> bool:
    """Put image from logs (e.g. fire_cam0_*.jpg) into clipboard so it can be pasted into WhatsApp."""
    if not image_path or not os.path.isfile(image_path):
        return False
    try:
        from io import BytesIO
        from PIL import Image
        import win32clipboard
        img = Image.open(image_path).convert("RGB")
        output = BytesIO()
        img.save(output, "BMP")
        data = output.getvalue()[14:]
        output.close()
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
        win32clipboard.CloseClipboard()
        return True
    except Exception:
        return False


def send_whatsapp_single(phone: str, msg: str, image_path: str = None):
    """Send WhatsApp message to a single phone number."""
    if not phone or not phone.startswith("+"):
        return False
    number = phone.strip()
    if image_path and os.path.isfile(image_path):
        put_image_in_clipboard(image_path)
    from urllib.parse import quote
    num_clean = number.replace("+", "").replace(" ", "").replace("-", "")
    if not num_clean.isdigit():
        print(f"  [WhatsApp] Invalid phone number: {phone}")
        return False
    encoded_text = quote(msg)
    try:
        import webbrowser
        import pyautogui
        app_url = f"whatsapp://send?phone={num_clean}&text={encoded_text}"
        webbrowser.open(app_url)
        time.sleep(4)
        pyautogui.press("enter")
        if image_path and os.path.isfile(image_path):
            time.sleep(1)
            put_image_in_clipboard(image_path)
            pyautogui.hotkey("ctrl", "v")
            time.sleep(0.8)
            pyautogui.press("enter")
        print(f"  [WhatsApp] Sent to {phone}")
        return True
    except Exception as e:
        print(f"  [WhatsApp] Failed to send to {phone}: {e}")
        return False

def send_whatsapp_fire_alert_all(phones: list, camera_index: int, conf: float, image_path: str = None, camera_ip: str = "N/A"):
    """Send fire alert via WhatsApp to ALL phone numbers in sequence."""
    if not phones:
        return
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"🔥 FIRE ALERT!\nCamera {camera_index + 1} (IP: {camera_ip})\nConfidence: {conf:.0%}\nTime: {time_str}"
    
    print(f"[WhatsApp] Sending fire alert to {len(phones)} number(s)...")
    sent_count = 0
    for i, phone in enumerate(phones):
        print(f"  [{i+1}/{len(phones)}] Sending to {phone}...")
        if send_whatsapp_single(phone, msg, image_path):
            sent_count += 1
        if i < len(phones) - 1:
            time.sleep(3)  # Wait between messages to avoid overlap
    print(f"[WhatsApp] Done! Sent to {sent_count}/{len(phones)} numbers.")


# ------------------------------
# UI: USER ENTER MULTIPLE RTSP URLS (one per line)
# ------------------------------
def parse_whatsapp_numbers(s: str):
    """Parse comma or newline separated phone numbers (with +)."""
    return [p.strip() for p in s.replace("\n", ",").split(",") if p.strip() and p.strip().startswith("+")]


def show_rtsp_dialog():
    import tkinter as tk
    from tkinter import ttk
    from tkinter import messagebox
    result_urls = [None]
    result_whatsapp = [WHATSAPP_PHONES.copy()]
    result_model = [MODEL_OPTIONS[0]]
    result_conf = [CONF_THRESHOLD]
    result_cooldown = [ALERT_COOLDOWN]
    result_message_cooldown = [MESSAGE_COOLDOWN]
    result_detect_n = [DETECT_EVERY_N]
    result_fullscreen = [True]
    result_hd = [True]

    root = tk.Tk()
    root.title("Fire Alert – Enter RTSP URLs (1–16 CCTV)")
    root.geometry("640x680")
    root.resizable(True, True)

    # Button bar at bottom (pack first with BOTTOM so it stays visible)
    btn_frame = ttk.Frame(root)
    btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

    # Display mode: window or full screen (no window), high definition
    display_frame = ttk.LabelFrame(root, text="Display – window or full screen")
    display_frame.pack(fill=tk.X, padx=10, pady=(10, 4))
    fullscreen_var = tk.BooleanVar(value=True)
    ttk.Radiobutton(display_frame, text="Full screen (no window)", variable=fullscreen_var, value=True).pack(anchor=tk.W, padx=8, pady=2)
    ttk.Radiobutton(display_frame, text="Window (resizable)", variable=fullscreen_var, value=False).pack(anchor=tk.W, padx=8, pady=2)
    hd_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(display_frame, text="High definition (720p/1080p – sharper video, higher GPU use)", variable=hd_var).pack(anchor=tk.W, padx=8, pady=2)

    ttk.Label(root, text="Fire detection model:").pack(anchor=tk.W, padx=10, pady=(10, 2))
    model_var = tk.StringVar(value=MODEL_OPTIONS[0])
    model_combo = ttk.Combobox(root, textvariable=model_var, values=MODEL_OPTIONS, state="readonly", width=20)
    model_combo.pack(anchor=tk.W, padx=10, pady=2)

    # Custom config: confidence, cooldown, detect every N
    conf_frame = ttk.LabelFrame(root, text="Detection settings")
    conf_frame.pack(fill=tk.X, padx=10, pady=(8, 4))
    row1 = ttk.Frame(conf_frame)
    row1.pack(fill=tk.X, padx=8, pady=4)
    ttk.Label(row1, text="Confidence threshold (0.01–0.99):").pack(side=tk.LEFT, padx=(0, 8))
    conf_var = tk.StringVar(value=str(CONF_THRESHOLD))
    conf_entry = ttk.Entry(row1, textvariable=conf_var, width=8)
    conf_entry.pack(side=tk.LEFT, padx=(0, 20))
    ttk.Label(row1, text="Alert cooldown (sec):").pack(side=tk.LEFT, padx=(0, 8))
    cooldown_var = tk.StringVar(value=str(ALERT_COOLDOWN))
    ttk.Entry(row1, textvariable=cooldown_var, width=6).pack(side=tk.LEFT, padx=(0, 20))
    ttk.Label(row1, text="Message cooldown (sec):").pack(side=tk.LEFT, padx=(0, 8))
    message_cooldown_var = tk.StringVar(value=str(MESSAGE_COOLDOWN))
    ttk.Entry(row1, textvariable=message_cooldown_var, width=6).pack(side=tk.LEFT)
    row2 = ttk.Frame(conf_frame)
    row2.pack(fill=tk.X, padx=8, pady=2)
    ttk.Label(row2, text="Detect every N frames:").pack(side=tk.LEFT, padx=(0, 8))
    detect_n_var = tk.StringVar(value=str(DETECT_EVERY_N))
    ttk.Entry(row2, textvariable=detect_n_var, width=6).pack(side=tk.LEFT)

    ttk.Label(root, text="Enter 1 to 16 RTSP URLs or use local camera (0, localhost, webcam – one per line):").pack(anchor=tk.W, padx=10, pady=(8, 2))
    text = tk.Text(root, height=8, width=72, font=("Consolas", 9))
    text.pack(fill=tk.X, padx=10, pady=4)
    text.insert("1.0", "\n".join(DEFAULT_RTSP_URLS))

    # WhatsApp number management
    wa_frame = ttk.LabelFrame(root, text="WhatsApp numbers for fire alerts")
    wa_frame.pack(fill=tk.X, padx=10, pady=(8, 4))
    
    wa_list_frame = ttk.Frame(wa_frame)
    wa_list_frame.pack(fill=tk.X, padx=8, pady=4)
    
    wa_listbox = tk.Listbox(wa_list_frame, height=4, width=40, font=("Consolas", 10))
    wa_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
    wa_scrollbar = ttk.Scrollbar(wa_list_frame, orient=tk.VERTICAL, command=wa_listbox.yview)
    wa_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    wa_listbox.config(yscrollcommand=wa_scrollbar.set)
    
    # Load existing numbers into listbox
    for phone in WHATSAPP_PHONES:
        wa_listbox.insert(tk.END, phone)
    
    wa_entry_frame = ttk.Frame(wa_frame)
    wa_entry_frame.pack(fill=tk.X, padx=8, pady=4)
    ttk.Label(wa_entry_frame, text="Phone (+xxx):").pack(side=tk.LEFT, padx=(0, 4))
    wa_entry_var = tk.StringVar()
    wa_entry = ttk.Entry(wa_entry_frame, textvariable=wa_entry_var, width=20)
    wa_entry.pack(side=tk.LEFT, padx=(0, 8))
    
    def wa_add():
        phone = wa_entry_var.get().strip()
        if phone and phone.startswith("+") and phone not in wa_listbox.get(0, tk.END):
            wa_listbox.insert(tk.END, phone)
            wa_entry_var.set("")
    
    def wa_edit():
        sel = wa_listbox.curselection()
        if sel:
            idx = sel[0]
            new_phone = wa_entry_var.get().strip()
            if new_phone and new_phone.startswith("+"):
                wa_listbox.delete(idx)
                wa_listbox.insert(idx, new_phone)
                wa_entry_var.set("")
    
    def wa_delete():
        sel = wa_listbox.curselection()
        if sel:
            wa_listbox.delete(sel[0])
    
    def wa_save():
        phones = list(wa_listbox.get(0, tk.END))
        if save_whatsapp_phones_to_file(phones):
            messagebox.showinfo("Saved", f"Saved {len(phones)} WhatsApp number(s) to file.")
        else:
            messagebox.showerror("Error", "Failed to save WhatsApp numbers.")
    
    def wa_on_select(event):
        sel = wa_listbox.curselection()
        if sel:
            wa_entry_var.set(wa_listbox.get(sel[0]))
    
    wa_listbox.bind("<<ListboxSelect>>", wa_on_select)
    
    wa_btn_frame = ttk.Frame(wa_frame)
    wa_btn_frame.pack(fill=tk.X, padx=8, pady=(0, 4))
    ttk.Button(wa_btn_frame, text="Add", command=wa_add, width=8).pack(side=tk.LEFT, padx=(0, 4))
    ttk.Button(wa_btn_frame, text="Edit", command=wa_edit, width=8).pack(side=tk.LEFT, padx=(0, 4))
    ttk.Button(wa_btn_frame, text="Delete", command=wa_delete, width=8).pack(side=tk.LEFT, padx=(0, 4))
    ttk.Button(wa_btn_frame, text="Save to File", command=wa_save, width=12).pack(side=tk.LEFT, padx=(0, 4))

    def parse_float(s, default, low, high):
        try:
            v = float(s.strip().replace(",", "."))
            return max(low, min(high, v))
        except (ValueError, TypeError):
            return default

    def parse_int(s, default, low=1, high=999):
        try:
            v = int(s.strip())
            return max(low, min(high, v))
        except (ValueError, TypeError):
            return default

    def on_connect():
        lines = text.get("1.0", tk.END).strip().splitlines()
        urls = [u.strip() for u in lines if u.strip()][:MAX_CAMERAS]
        result_urls[0] = urls if urls else list(DEFAULT_RTSP_URLS)
        result_whatsapp[0] = list(wa_listbox.get(0, tk.END))
        m = model_var.get().strip()
        result_model[0] = m if m in MODEL_OPTIONS else MODEL_OPTIONS[0]
        result_conf[0] = parse_float(conf_var.get(), CONF_THRESHOLD, 0.01, 0.99)
        result_cooldown[0] = parse_int(cooldown_var.get(), ALERT_COOLDOWN, 1, 300)
        result_message_cooldown[0] = parse_int(message_cooldown_var.get(), MESSAGE_COOLDOWN, 1, 3600)
        result_detect_n[0] = parse_int(detect_n_var.get(), DETECT_EVERY_N, 1, 60)
        result_fullscreen[0] = fullscreen_var.get()
        result_hd[0] = hd_var.get()
        root.quit()
        root.destroy()

    def on_cancel():
        root.quit()
        root.destroy()

    # Add Connect/Cancel to the bottom bar (already packed at BOTTOM)
    tk.Button(btn_frame, text="Connect", command=on_connect).pack(side=tk.LEFT, padx=(0, 8))
    tk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT)

    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.mainloop()
    return result_urls[0], result_whatsapp[0] or [], result_model[0], result_conf[0], result_cooldown[0], result_message_cooldown[0], result_detect_n[0], result_fullscreen[0], result_hd[0]

def main():
    """Fire Alert application entry point."""
    import pygame
    global RTSP_URLS, FULLSCREEN, HIGH_DEFINITION, MODEL_PATH, WHATSAPP_PHONES, WHATSAPP_ENABLED
    global CELL_W, CELL_H, WIDTH, HEIGHT, CONF_THRESHOLD, ALERT_COOLDOWN, MESSAGE_COOLDOWN, DETECT_EVERY_N

    whatsapp_from_dialog = ""
    if len(sys.argv) > 1:
        RTSP_URLS = [u.strip() for u in sys.argv[1].replace("|", "\n").splitlines() if u.strip()][:MAX_CAMERAS]
        if not RTSP_URLS:
            RTSP_URLS = list(DEFAULT_RTSP_URLS)
        HIGH_DEFINITION = True  # default on when using CLI
        # Optional argv[2]: "window" or "fullscreen" (default fullscreen)
        if len(sys.argv) > 2:
            mode = sys.argv[2].strip().lower()
            FULLSCREEN = mode not in ("window", "0", "no", "w")
    else:
        dialog_result = show_rtsp_dialog()
        if dialog_result is None or dialog_result[0] is None:
            sys.exit(0)
        RTSP_URLS, whatsapp_from_dialog, selected_model, CONF_THRESHOLD, ALERT_COOLDOWN, MESSAGE_COOLDOWN, DETECT_EVERY_N, FULLSCREEN, HIGH_DEFINITION = dialog_result
        if not RTSP_URLS:
            RTSP_URLS = list(DEFAULT_RTSP_URLS)
        MODEL_PATH = resource_path(selected_model if selected_model in MODEL_OPTIONS else MODEL_OPTIONS[0])
        if whatsapp_from_dialog:
            WHATSAPP_PHONES[:] = whatsapp_from_dialog
            WHATSAPP_ENABLED = bool(WHATSAPP_PHONES)
        if HIGH_DEFINITION:
            CELL_W, CELL_H = 960, 540
            WIDTH, HEIGHT = CELL_W, CELL_H
    RTSP_URLS = RTSP_URLS[:MAX_CAMERAS]
    if len(sys.argv) > 1 and len(sys.argv) <= 2:
        FULLSCREEN = True  # CLI with only URLs: default full screen
    YOLO_IMGSZ = 640 if HIGH_DEFINITION else 320
    print("Using", len(RTSP_URLS), "camera(s) (1–16):", RTSP_URLS)
    print("Model:", MODEL_PATH)
    print("Confidence:", CONF_THRESHOLD, "| Alert cooldown:", ALERT_COOLDOWN, "s | Message cooldown:", MESSAGE_COOLDOWN, "s | Detect every", DETECT_EVERY_N, "frames", "| HD:", HIGH_DEFINITION, "| Display:", "full screen" if FULLSCREEN else "window")
    print("Fire detection: YOLO only (no color/glow check)")
    print("Low confidence: conf <", LOW_CONFIDENCE_THRESHOLD, "→ log only (no WhatsApp).")
    if WHATSAPP_ENABLED and WHATSAPP_PHONES:
        print("WhatsApp fire alerts enabled for:", WHATSAPP_PHONES)

    # Load YOLO model
    model = YOLO(MODEL_PATH)

    # Multi-camera grid layout
    num_streams = len(RTSP_URLS)
    if num_streams == 1:
        cols, rows = 1, 1
        if FULLSCREEN:
            screen_w, screen_h = (1920, 1080) if HIGH_DEFINITION else (960, 540)  # actual size set by pygame fullscreen
        else:
            # Window mode: use smaller size so it's clearly a window, not full screen
            screen_w, screen_h = (1280, 720) if HIGH_DEFINITION else (960, 540)
        display_w, display_h = screen_w, screen_h
    else:
        cols = 4 if num_streams > 9 else (3 if num_streams > 4 else 2)
        rows = (num_streams + cols - 1) // cols
        screen_w = CELL_W * cols
        screen_h = CELL_H * rows
        display_w, display_h = CELL_W, CELL_H

    caps = [None] * num_streams
    reader_threads_list = [None] * num_streams
    frame_queues = [deque(maxlen=2) for _ in range(num_streams)]
    stream_lost_flags = [[False] for _ in range(num_streams)]
    reconnect_counts = [0] * num_streams
    last_frames = [None] * num_streams
    frame_counts = [0] * num_streams
    last_detections = [[] for _ in range(num_streams)]
    last_alert_times = [0.0] * num_streams
    last_message_sent_time = 0.0  # global cooldown for WhatsApp (message sending cooldown)

    pygame.init()
    if FULLSCREEN:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        screen_w, screen_h = screen.get_width(), screen.get_height()
    else:
        screen = pygame.display.set_mode((screen_w, screen_h), pygame.RESIZABLE)
    if num_streams == 1:
        display_w, display_h = screen_w, screen_h
    else:
        display_w = screen_w // cols
        display_h = screen_h // rows
    pygame.display.set_caption("🔥 Fire Alert – " + str(num_streams) + " camera(s)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)
    font_small = pygame.font.SysFont("Arial", 14)
    running = True

    while running:
        for i in range(num_streams):
            if caps[i] is None:
                caps[i] = start_opencv_capture(RTSP_URLS[i])
                if caps[i] is None:
                    reconnect_counts[i] += 1
                    if reconnect_counts[i] > RECONNECT_MAX:
                        continue
                    time.sleep(0.5)
                    continue
                stream_lost_flags[i][0] = False
                frame_queues[i].clear()
                reader_threads_list[i] = threading.Thread(
                    target=frame_reader_thread,
                    args=(caps[i], frame_queues[i], stream_lost_flags[i]),
                    daemon=True
                )
                reader_threads_list[i].start()
                reconnect_counts[i] = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F11:
                    # Toggle full screen on/off
                    FULLSCREEN = not FULLSCREEN
                    if FULLSCREEN:
                        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        screen_w, screen_h = screen.get_width(), screen.get_height()
                    else:
                        screen = pygame.display.set_mode((screen_w, screen_h), pygame.RESIZABLE)
                    if num_streams == 1:
                        display_w, display_h = screen_w, screen_h
                    else:
                        display_w = screen_w // cols
                        display_h = screen_h // rows
            elif event.type == pygame.VIDEORESIZE and not FULLSCREEN:
                screen_w, screen_h = event.w, event.h
                if num_streams == 1:
                    display_w, display_h = screen_w, screen_h
                else:
                    display_w = screen_w // cols
                    display_h = screen_h // rows
                screen = pygame.display.set_mode((screen_w, screen_h), pygame.RESIZABLE)

        if not running:
            break

        any_reconnecting = False
        for i in range(num_streams):
            if stream_lost_flags[i][0]:
                try:
                    if caps[i] is not None:
                        caps[i].release()
                except Exception:
                    pass
                caps[i] = None
                reader_threads_list[i] = None
                stream_lost_flags[i][0] = False
                reconnect_counts[i] += 1
                if reconnect_counts[i] <= RECONNECT_MAX:
                    any_reconnecting = True

        if any_reconnecting:
            show_reconnecting(screen, font, f"Reconnecting camera(s)... in {RECONNECT_DELAY}s")
            time.sleep(RECONNECT_DELAY)
            continue

        screen.fill((20, 20, 20))
        for i in range(num_streams):
            if num_streams == 1:
                px, py = 0, 0
            else:
                col, row = i % cols, i // cols
                px, py = col * display_w, row * display_h

            if frame_queues[i]:
                frame = frame_queues[i][-1]
                last_frames[i] = frame
            elif last_frames[i] is not None:
                frame = last_frames[i]
            else:
                cw, ch = display_w, display_h
                cell_surf = pygame.Surface((cw, ch))
                cell_surf.fill((40, 40, 40))
                txt = font.render(f"Camera {i+1} – Connecting...", True, (180, 180, 180))
                tr = txt.get_rect(center=(cw // 2, ch // 2))
                cell_surf.blit(txt, tr)
                screen.blit(cell_surf, (px, py))
                continue

            display_frame = frame.copy()
            fire_detected = False
            fire_conf = 0.0
            frame_counts[i] += 1

            if frame_counts[i] % DETECT_EVERY_N == 0:
                try:
                    results = model(frame, conf=CONF_THRESHOLD, verbose=False, imgsz=YOLO_IMGSZ)
                    last_detections[i] = []
                    for r in results:
                        if r.boxes is None:
                            continue
                        for box in r.boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            if cls == 0:
                                fire_detected = True
                                fire_conf = max(fire_conf, conf)
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                last_detections[i].append((x1, y1, x2, y2, conf))
                                print(f"[Cam {i+1}] Fire detected: conf={conf:.2f} box=({x1},{y1})-({x2},{y2})")
                except Exception as e:
                    print(f"[Cam {i+1}] Detection error: {e}")
            else:
                if last_detections[i]:
                    fire_detected = True
                    fire_conf = max(d[4] for d in last_detections[i])

            # Draw red bounding boxes for fire detections
            for (bx1, by1, bx2, by2, conf) in last_detections[i]:
                cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                cv2.putText(display_frame, f"FIRE {conf:.2f}", (bx1, max(0, by1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Low-confidence: show "LOW CONF" indicator
            is_low_conf = fire_detected and fire_conf < LOW_CONFIDENCE_THRESHOLD
            if is_low_conf:
                cv2.putText(display_frame, "LOW CONF (review)", (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            surface = pygame.transform.scale(surface, (display_w, display_h))
            screen.blit(surface, (px, py))
            label = font_small.render(f"Cam {i+1}", True, (255, 255, 255))
            screen.blit(label, (px + 4, py + 4))
            if fire_detected:
                alert_text = font_small.render("FIRE!", True, (255, 0, 0))
                screen.blit(alert_text, (px + 4, py + 24))

            now = time.time()
            if fire_detected and now - last_alert_times[i] > ALERT_COOLDOWN:
                last_alert_times[i] = now
                try:
                    low_conf = fire_conf < LOW_CONFIDENCE_THRESHOLD
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = f"{LOG_DIR}/fire_cam{i}_{timestamp}.jpg"
                    saved_frame = display_frame.copy()
                    draw_screenshot_overlays(saved_frame, i)
                    cv2.imwrite(image_path, saved_frame)
                    cursor.execute(
                        "INSERT INTO fire_events (timestamp, confidence, image_path, camera_index, low_confidence) VALUES (?, ?, ?, ?, ?)",
                        (timestamp, fire_conf, image_path, i, 1 if low_conf else 0)
                    )
                    db.commit()
                    camera_ip = get_camera_ip_from_rtsp(RTSP_URLS[i])
                    # Full alert (WhatsApp) only when not low-confidence and message sending cooldown elapsed
                    if WHATSAPP_ENABLED and WHATSAPP_PHONES and not low_conf:
                        if now - last_message_sent_time >= MESSAGE_COOLDOWN:
                            last_message_sent_time = now
                            # Send to all WhatsApp numbers in one background thread
                            threading.Thread(
                                target=send_whatsapp_fire_alert_all,
                                args=(WHATSAPP_PHONES.copy(), i, fire_conf, image_path, camera_ip),
                                daemon=True
                            ).start()
                except Exception:
                    pass

        pygame.display.flip()
        clock.tick(FPS)

    for i in range(num_streams):
        if caps[i] is not None:
            try:
                caps[i].release()
            except Exception:
                pass
    db.close()
    pygame.quit()
    sys.exit(0)


def start_opencv_capture(url):
    """OpenCV VideoCapture for RTSP (CAP_FFMPEG) or local camera (index 0, 1, etc.)."""
    try:
        if is_local_camera(url):
            idx = 0
            s = str(url).strip().lower()
            if s.isdigit():
                idx = int(s)
            elif s in ("localhost", "webcam", "local"):
                idx = 0
            cap = cv2.VideoCapture(idx)
        else:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            return cap
        cap.release()
    except Exception:
        pass
    return None


def frame_reader_thread(cap, frame_queue, stream_lost_flag):
    """Background: read frames with OpenCV; retry many times before declaring stream lost."""
    retries = 0
    while True:
        try:
            if cap is None or not cap.isOpened():
                stream_lost_flag[0] = True
                break
            ret, frame = cap.read()
            if ret and frame is not None:
                retries = 0
                if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
                    frame = cv2.resize(frame, (WIDTH, HEIGHT))
                frame_queue.append(frame.copy())
            else:
                retries += 1
                if retries >= READ_RETRIES:
                    stream_lost_flag[0] = True
                    break
                time.sleep(0.1)
        except Exception:
            retries += 1
            if retries >= READ_RETRIES:
                stream_lost_flag[0] = True
                break
            time.sleep(0.1)
    stream_lost_flag[0] = True

def show_reconnecting(screen, font, msg="Reconnecting..."):
    import pygame
    screen.fill((30, 30, 30))
    text = font.render(msg, True, (255, 200, 0))
    r = text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
    screen.blit(text, r)
    pygame.display.flip()


MJPEG_BOUNDARY = b"frame"


def web_main():
    """Run fire detection and serve a multi-camera MJPEG dashboard in the browser (no pygame window)."""
    global RTSP_URLS, MODEL_PATH, WHATSAPP_PHONES, WHATSAPP_ENABLED
    global CELL_W, CELL_H, WIDTH, HEIGHT

    argv = sys.argv[1:]
    host = os.environ.get("FIRE_ALERT_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("FIRE_ALERT_WEB_PORT", "8766"))
    i = 0
    filtered = []
    while i < len(argv):
        if argv[i] == "--port" and i + 1 < len(argv):
            port = int(argv[i + 1])
            i += 2
            continue
        if argv[i] == "--host" and i + 1 < len(argv):
            host = argv[i + 1]
            i += 2
            continue
        filtered.append(argv[i])
        i += 1

    if filtered:
        RTSP_URLS = [u.strip() for u in filtered[0].replace("|", "\n").splitlines() if u.strip()][:MAX_CAMERAS]
    else:
        RTSP_URLS = list(DEFAULT_RTSP_URLS)
    if not RTSP_URLS:
        RTSP_URLS = list(DEFAULT_RTSP_URLS)
    RTSP_URLS = RTSP_URLS[:MAX_CAMERAS]

    MODEL_PATH = resource_path(MODEL_OPTIONS[0])
    HIGH_DEFINITION = True
    if HIGH_DEFINITION:
        CELL_W, CELL_H = 960, 540
        WIDTH, HEIGHT = CELL_W, CELL_H
    YOLO_IMGSZ = 640 if HIGH_DEFINITION else 320

    num_streams = len(RTSP_URLS)
    print("Web mode: open http://%s:%s/ in your browser (Register / Login, then live dashboard)" % (host, port))
    print("Using", num_streams, "camera(s):", RTSP_URLS)
    print("Model:", MODEL_PATH)

    model = YOLO(MODEL_PATH)
    web_db = sqlite3.connect("fire_alert.db", check_same_thread=False)
    web_cursor = web_db.cursor()
    ensure_users_table(web_db)
    ensure_reports_table(web_db)
    report_upload_dir = os.path.join(LOG_DIR, "reports")
    os.makedirs(report_upload_dir, exist_ok=True)

    caps = [None] * num_streams
    reader_threads_list = [None] * num_streams
    frame_queues = [deque(maxlen=2) for _ in range(num_streams)]
    stream_lost_flags = [[False] for _ in range(num_streams)]
    reconnect_counts = [0] * num_streams
    last_frames = [None] * num_streams
    frame_counts = [0] * num_streams
    last_detections = [[] for _ in range(num_streams)]
    last_alert_times = [0.0] * num_streams
    last_message_sent_time = 0.0

    latest_jpeg = [None] * num_streams
    jpeg_lock = threading.Lock()
    stop_event = threading.Event()

    def encode_cell_jpeg(bgr_frame, dw, dh):
        if dw <= 0 or dh <= 0:
            return None
        try:
            if bgr_frame.shape[1] != dw or bgr_frame.shape[0] != dh:
                bgr_frame = cv2.resize(bgr_frame, (dw, dh))
            ok, buf = cv2.imencode(".jpg", bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            return buf.tobytes() if ok else None
        except Exception:
            return None

    def processing_loop():
        nonlocal caps, reader_threads_list, last_message_sent_time
        while not stop_event.is_set():
            for i in range(num_streams):
                if caps[i] is None:
                    caps[i] = start_opencv_capture(RTSP_URLS[i])
                    if caps[i] is None:
                        reconnect_counts[i] += 1
                        if reconnect_counts[i] > RECONNECT_MAX:
                            continue
                        time.sleep(0.5)
                        continue
                    stream_lost_flags[i][0] = False
                    frame_queues[i].clear()
                    reader_threads_list[i] = threading.Thread(
                        target=frame_reader_thread,
                        args=(caps[i], frame_queues[i], stream_lost_flags[i]),
                        daemon=True,
                    )
                    reader_threads_list[i].start()
                    reconnect_counts[i] = 0

            any_reconnecting = False
            for i in range(num_streams):
                if stream_lost_flags[i][0]:
                    try:
                        if caps[i] is not None:
                            caps[i].release()
                    except Exception:
                        pass
                    caps[i] = None
                    reader_threads_list[i] = None
                    stream_lost_flags[i][0] = False
                    reconnect_counts[i] += 1
                    if reconnect_counts[i] <= RECONNECT_MAX:
                        any_reconnecting = True

            if any_reconnecting:
                time.sleep(RECONNECT_DELAY)
                continue

            for i in range(num_streams):
                if frame_queues[i]:
                    frame = frame_queues[i][-1]
                    last_frames[i] = frame
                elif last_frames[i] is not None:
                    frame = last_frames[i]
                else:
                    blank = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                    blank[:] = (40, 40, 40)
                    cv2.putText(
                        blank,
                        "Camera %d – Connecting..." % (i + 1),
                        (20, HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (180, 180, 180),
                        2,
                    )
                    with jpeg_lock:
                        latest_jpeg[i] = encode_cell_jpeg(blank, WIDTH, HEIGHT)
                    continue

                display_frame = frame.copy()
                fire_detected = False
                fire_conf = 0.0
                frame_counts[i] += 1

                if frame_counts[i] % DETECT_EVERY_N == 0:
                    try:
                        results = model(frame, conf=CONF_THRESHOLD, verbose=False, imgsz=YOLO_IMGSZ)
                        last_detections[i] = []
                        for r in results:
                            if r.boxes is None:
                                continue
                            for box in r.boxes:
                                cls = int(box.cls[0])
                                conf = float(box.conf[0])
                                if cls == 0:
                                    fire_detected = True
                                    fire_conf = max(fire_conf, conf)
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    last_detections[i].append((x1, y1, x2, y2, conf))
                                    print("[Cam %d] Fire detected: conf=%.2f" % (i + 1, conf))
                    except Exception as e:
                        print("[Cam %d] Detection error: %s" % (i + 1, e))
                else:
                    if last_detections[i]:
                        fire_detected = True
                        fire_conf = max(d[4] for d in last_detections[i])

                for (bx1, by1, bx2, by2, conf) in last_detections[i]:
                    cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                    cv2.putText(
                        display_frame,
                        "FIRE %.2f" % conf,
                        (bx1, max(0, by1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )

                is_low_conf = fire_detected and fire_conf < LOW_CONFIDENCE_THRESHOLD
                if is_low_conf:
                    cv2.putText(
                        display_frame,
                        "LOW CONF (review)",
                        (8, 48),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 165, 255),
                        2,
                    )

                cv2.putText(
                    display_frame,
                    "Cam %d" % (i + 1),
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                if fire_detected:
                    cv2.putText(
                        display_frame,
                        "FIRE!",
                        (8, 48),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                with jpeg_lock:
                    latest_jpeg[i] = encode_cell_jpeg(display_frame, WIDTH, HEIGHT)

                now = time.time()
                if fire_detected and now - last_alert_times[i] > ALERT_COOLDOWN:
                    last_alert_times[i] = now
                    try:
                        low_conf = fire_conf < LOW_CONFIDENCE_THRESHOLD
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_path = "%s/fire_cam%d_%s.jpg" % (LOG_DIR, i, timestamp)
                        saved_frame = display_frame.copy()
                        draw_screenshot_overlays(saved_frame, i)
                        cv2.imwrite(image_path, saved_frame)
                        web_cursor.execute(
                            "INSERT INTO fire_events (timestamp, confidence, image_path, camera_index, low_confidence) VALUES (?, ?, ?, ?, ?)",
                            (timestamp, fire_conf, image_path, i, 1 if low_conf else 0),
                        )
                        web_db.commit()
                        camera_ip = get_camera_ip_from_rtsp(RTSP_URLS[i])
                        if WHATSAPP_ENABLED and WHATSAPP_PHONES and not low_conf:
                            if now - last_message_sent_time >= MESSAGE_COOLDOWN:
                                last_message_sent_time = now
                                threading.Thread(
                                    target=send_whatsapp_fire_alert_all,
                                    args=(WHATSAPP_PHONES.copy(), i, fire_conf, image_path, camera_ip),
                                    daemon=True,
                                ).start()
                    except Exception:
                        pass

            time.sleep(1.0 / max(1, min(FPS, 30)))

        for i in range(num_streams):
            if caps[i] is not None:
                try:
                    caps[i].release()
                except Exception:
                    pass
        try:
            web_db.close()
        except Exception:
            pass

    threading.Thread(target=processing_loop, daemon=True).start()

    state = {
        "num_streams": num_streams,
        "latest_jpeg": latest_jpeg,
        "lock": jpeg_lock,
        "stop": stop_event,
        "web_db": web_db,
        "web_cursor": web_cursor,
        "report_upload_dir": report_upload_dir,
        "sessions": {},
        "sess_lock": threading.Lock(),
    }

    def prune_sessions():
        now = time.time()
        with state["sess_lock"]:
            for t in [x for x, s in state["sessions"].items() if s["until"] < now]:
                del state["sessions"][t]

    def session_username(handler):
        prune_sessions()
        tok = _parse_cookies(handler.headers.get("Cookie", "")).get(_SESSION_COOKIE)
        if not tok:
            return None
        with state["sess_lock"]:
            s = state["sessions"].get(tok)
            if not s or s["until"] < time.time():
                return None
            return s["username"]

    class FireAlertWebHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            print("%s - %s" % (self.address_string(), fmt % args))

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")

        def _send_html(self, html: str, status: int = 200):
            body = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _redirect(self, location: str, clear_cookie: bool = False):
            self.send_response(302)
            self.send_header("Location", location)
            if clear_cookie:
                self.send_header(
                    "Set-Cookie",
                    "%s=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0" % _SESSION_COOKIE,
                )
            self.end_headers()

        def _read_form(self):
            n = int(self.headers.get("Content-Length", 0))
            if n > 2_000_000:
                return None
            raw = self.rfile.read(n).decode("utf-8", errors="replace")
            return urllib.parse.parse_qs(raw, keep_blank_values=True)

        def _form_one(self, fields, key):
            if not fields:
                return ""
            v = fields.get(key, [""])
            return v[0] if v else ""

        def _read_multipart(self):
            try:
                fs = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                    },
                    keep_blank_values=True,
                )
                return fs
            except Exception:
                return None

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            if parsed.path == "/health":
                body = b'{"ok":true,"service":"fire-alert-web"}\n'
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self._cors()
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/logout":
                tok = _parse_cookies(self.headers.get("Cookie", "")).get(_SESSION_COOKIE)
                if tok:
                    with state["sess_lock"]:
                        state["sessions"].pop(tok, None)
                self._redirect("/", clear_cookie=True)
                return
            if parsed.path == "/":
                if session_username(self):
                    self._redirect("/dashboard")
                    return
                view = (qs.get("view") or [""])[0].lower()
                active = "register" if view == "register" else "login"
                self._send_html(web_auth_html(active))
                return
            if parsed.path == "/dashboard":
                if not session_username(self):
                    self._redirect("/")
                    return
                self._send_html(web_dashboard_html(state["num_streams"]))
                return
            if parsed.path == "/report":
                if not session_username(self):
                    self._redirect("/")
                    return
                self._send_html(web_report_html())
                return
            if parsed.path == "/alerts":
                if not session_username(self):
                    self._redirect("/")
                    return
                state["web_cursor"].execute(
                    "SELECT id, timestamp, confidence, camera_index, low_confidence, image_path FROM fire_events ORDER BY id DESC LIMIT 500"
                )
                detection_rows = state["web_cursor"].fetchall()
                state["web_cursor"].execute(
                    "SELECT id, created_at, created_by, address, severity, description, photo_path FROM fire_reports ORDER BY id DESC LIMIT 500"
                )
                report_rows = state["web_cursor"].fetchall()
                self._send_html(web_alerts_html(list(detection_rows), list(report_rows)))
                return
            if parsed.path.startswith("/image/event/"):
                if not session_username(self):
                    self.send_error(401, "Login required")
                    return
                try:
                    eid = int(parsed.path.rstrip("/").split("/")[-1])
                except ValueError:
                    self.send_error(400, "Bad id")
                    return
                state["web_cursor"].execute("SELECT image_path FROM fire_events WHERE id = ?", (eid,))
                row = state["web_cursor"].fetchone()
                if not row or not row[0]:
                    self.send_error(404, "Not found")
                    return
                path = _resolved_log_path(row[0])
                if not path:
                    self.send_error(404, "Not found")
                    return
                ext = os.path.splitext(path)[1].lower()
                ctype = (
                    "image/jpeg"
                    if ext in (".jpg", ".jpeg")
                    else "image/png"
                    if ext == ".png"
                    else "image/webp"
                    if ext == ".webp"
                    else "application/octet-stream"
                )
                with open(path, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "private, max-age=300")
                self._cors()
                self.end_headers()
                self.wfile.write(data)
                return
            if parsed.path.startswith("/image/report/"):
                if not session_username(self):
                    self.send_error(401, "Login required")
                    return
                try:
                    rid = int(parsed.path.rstrip("/").split("/")[-1])
                except ValueError:
                    self.send_error(400, "Bad id")
                    return
                state["web_cursor"].execute("SELECT photo_path FROM fire_reports WHERE id = ?", (rid,))
                row = state["web_cursor"].fetchone()
                if not row or not row[0]:
                    self.send_error(404, "Not found")
                    return
                path = _resolved_log_path(row[0])
                if not path:
                    self.send_error(404, "Not found")
                    return
                ext = os.path.splitext(path)[1].lower()
                ctype = (
                    "image/jpeg"
                    if ext in (".jpg", ".jpeg")
                    else "image/png"
                    if ext == ".png"
                    else "image/webp"
                    if ext == ".webp"
                    else "application/octet-stream"
                )
                with open(path, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "private, max-age=300")
                self._cors()
                self.end_headers()
                self.wfile.write(data)
                return
            if parsed.path.startswith("/stream/"):
                if not session_username(self):
                    self.send_error(401, "Login required")
                    return
                part = parsed.path.replace("/stream/", "", 1).strip("/")
                try:
                    idx = int(part)
                except ValueError:
                    self.send_error(400, "Bad camera index")
                    return
                if idx < 0 or idx >= state["num_streams"]:
                    self.send_error(404, "Camera not found")
                    return
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=" + MJPEG_BOUNDARY.decode(),
                )
                self.send_header("Cache-Control", "no-cache, no-store")
                self._cors()
                self.end_headers()
                try:
                    while not state["stop"].is_set():
                        with state["lock"]:
                            chunk = state["latest_jpeg"][idx]
                        if chunk:
                            self.wfile.write(b"--" + MJPEG_BOUNDARY + b"\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                            self.wfile.write(chunk)
                            self.wfile.write(b"\r\n")
                        time.sleep(1.0 / 20.0)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
                return
            self.send_error(404, "Not found")

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)

            if parsed.path == "/report":
                username = session_username(self)
                if not username:
                    self._redirect("/")
                    return

                content_type = (self.headers.get("Content-Type", "") or "").lower()
                address = ""
                severity = "Medium"
                description = ""
                photo_bytes = None
                photo_name = ""

                if "multipart/form-data" in content_type:
                    fs = self._read_multipart()
                    if fs is None:
                        self._send_html(web_report_html(error="Failed to read submitted form."))
                        return
                    address = (fs.getvalue("address") or "").strip()
                    severity = (fs.getvalue("severity") or "Medium").strip()
                    description = (fs.getvalue("description") or "").strip()
                    if "photo" in fs:
                        photo_item = fs["photo"]
                        if getattr(photo_item, "filename", "") and getattr(photo_item, "file", None):
                            photo_name = os.path.basename(photo_item.filename)
                            photo_bytes = photo_item.file.read()
                else:
                    fields = self._read_form()
                    if fields is None:
                        self.send_error(413, "Payload too large")
                        return
                    address = self._form_one(fields, "address").strip()
                    severity = self._form_one(fields, "severity").strip() or "Medium"
                    description = self._form_one(fields, "description").strip()

                values = {"address": address, "severity": severity, "description": description}
                if not address:
                    self._send_html(web_report_html(error="Address is required.", values=values))
                    return
                if severity not in ("Low", "Medium", "High", "Critical"):
                    self._send_html(web_report_html(error="Severity is invalid.", values=values))
                    return
                if not description:
                    self._send_html(web_report_html(error="Description is required.", values=values))
                    return

                photo_path = None
                if photo_bytes:
                    if len(photo_bytes) > 5 * 1024 * 1024:
                        self._send_html(web_report_html(error="Photo is too large (max 5MB).", values=values))
                        return
                    ext = os.path.splitext(photo_name.lower())[1]
                    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
                        ext = ".jpg"
                    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(photo_name)[0])[:50] or "report"
                    saved_name = "report_%s_%s%s" % (
                        datetime.now().strftime("%Y%m%d_%H%M%S"),
                        safe_stem,
                        ext,
                    )
                    abs_path = os.path.join(state["report_upload_dir"], saved_name)
                    with open(abs_path, "wb") as f:
                        f.write(photo_bytes)
                    photo_path = abs_path.replace("\\", "/")

                state["web_cursor"].execute(
                    "INSERT INTO fire_reports (created_at, created_by, address, severity, description, photo_path) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        datetime.now().isoformat(timespec="seconds"),
                        username,
                        address,
                        severity,
                        description,
                        photo_path,
                    ),
                )
                state["web_db"].commit()
                self._send_html(web_report_html(notice="Fire alert report submitted successfully."))
                return

            fields = self._read_form()
            if fields is None:
                self.send_error(413, "Payload too large")
                return

            if parsed.path == "/login":
                ident = self._form_one(fields, "username").strip()
                password = self._form_one(fields, "password")
                row = None
                if ident:
                    state["web_cursor"].execute(
                        "SELECT username, password_hash FROM users WHERE username = ? OR lower(email) = lower(?)",
                        (ident, ident),
                    )
                    row = state["web_cursor"].fetchone()
                if not row or not _pbkdf2_verify_password(password, row[1]):
                    self._send_html(web_auth_html("login", error="Invalid username/email or password."))
                    return
                token = secrets.token_urlsafe(32)
                with state["sess_lock"]:
                    state["sessions"][token] = {"username": row[0], "until": time.time() + _SESSION_TTL_SEC}
                self.send_response(302)
                self.send_header("Location", "/dashboard")
                self.send_header(
                    "Set-Cookie",
                    "%s=%s; Path=/; HttpOnly; SameSite=Lax; Max-Age=%d"
                    % (_SESSION_COOKIE, token, _SESSION_TTL_SEC),
                )
                self.end_headers()
                return

            if parsed.path == "/register":
                username = self._form_one(fields, "username").strip()
                email = self._form_one(fields, "email").strip()
                password = self._form_one(fields, "password")
                confirm = self._form_one(fields, "confirm")
                if len(username) < 2 or len(username) > 64:
                    self._send_html(web_auth_html("register", error="Username must be 2–64 characters."))
                    return
                if not re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", email):
                    self._send_html(web_auth_html("register", error="Enter a valid email address."))
                    return
                if len(password) < 6:
                    self._send_html(web_auth_html("register", error="Password must be at least 6 characters."))
                    return
                if password != confirm:
                    self._send_html(web_auth_html("register", error="Passwords do not match."))
                    return
                try:
                    state["web_cursor"].execute(
                        "INSERT INTO users (username, email, password_hash, created_at) VALUES (?,?,?,?)",
                        (username, email.lower(), _pbkdf2_hash_password(password), datetime.now().isoformat(timespec="seconds")),
                    )
                    state["web_db"].commit()
                except sqlite3.IntegrityError:
                    self._send_html(web_auth_html("register", error="That username or email is already registered."))
                    return
                self._send_html(web_auth_html("login", notice="Account created. You can sign in now."))
                return

            if parsed.path == "/logout":
                tok = _parse_cookies(self.headers.get("Cookie", "")).get(_SESSION_COOKIE)
                if tok:
                    with state["sess_lock"]:
                        state["sessions"].pop(tok, None)
                self._redirect("/", clear_cookie=True)
                return

            self.send_error(404, "Not found")

    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = ThreadingHTTPServer((host, port), FireAlertWebHandler)
    try:
        webbrowser.open("http://%s:%s/" % (host, port))
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        time.sleep(0.3)
        server.server_close()
    sys.exit(0)


if __name__ == "__main__":
    if "--web" in sys.argv:
        sys.argv.remove("--web")
        web_main()
    else:
        main()

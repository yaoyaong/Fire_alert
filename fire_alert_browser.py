#!/usr/bin/env python3
"""
Fire Alert System — browser-only portal (login + register).

Run:
  python fire_alert_browser.py
  python fire_alert_browser.py --host 127.0.0.1 --port 8766

Delete old alerts (DB rows + linked images under logs_browser/; keeps user accounts):
  python fire_alert_browser.py --purge-days 30
  python fire_alert_browser.py --purge-days 0
  (0 = remove all fire_reports and camera_detections older than "now", i.e. effectively all alert data)

Uses SQLite next to this script: fire_alert_portal.db

Camera RTSP page: uses **best-kiase.pt** and **best_fire.pt** (place next to this script).
Install: pip install ultralytics opencv-python
WhatsApp auto-send (press Send + paste snapshot from logs_browser/): pip install pyautogui
  Windows image paste into WhatsAppDesktop: also pip install Pillow pywin32
Optional env: FIRE_ALERT_CONF, FIRE_ALERT_DETECT_EVERY, FIRE_ALERT_IMGSZ,
  FIRE_ALERT_LOG_COOLDOWN — seconds between DB rows/snapshots (default 20)
  FIRE_ALERT_DEBOUNCE_CHECKS — consecutive no-fire inference passes before a new snapshot episode (default 4; set 0 to disable)
  FIRE_ALERT_WA_COOLDOWN — WhatsApp send cooldown (seconds between alerts; default 60)
  FIRE_ALERT_WA_LAUNCH_PAUSE — seconds after opening WhatsApp chat before auto Send (default 8)
  FIRE_ALERT_WA_CTRL_ENTER_SEND=1 — use Ctrl+Enter instead of Enter if WhatsApp is set that way
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import secrets
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
import cgi
from datetime import datetime, timedelta
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
CONF_THRESHOLD = float(os.environ.get("FIRE_ALERT_CONF", "0.7"))
DETECT_EVERY_N = int(os.environ.get("FIRE_ALERT_DETECT_EVERY", "3"))
YOLO_IMGSZ = int(os.environ.get("FIRE_ALERT_IMGSZ", "640"))
LOG_FIRE_COOLDOWN_SEC = float(os.environ.get("FIRE_ALERT_LOG_COOLDOWN", "20"))
# Consecutive inference passes without fire before allowing another logged snapshot (threshold-flicker guard). 0 = off (still only snapshots when cooldown accepts log).
FIRE_ALERT_DEBOUNCE_CHECKS = max(0, int(os.environ.get("FIRE_ALERT_DEBOUNCE_CHECKS", "4")))
# Minimum seconds between WhatsApp sends (global; separate from DB log cooldown)
WHATSAPP_MESSAGE_COOLDOWN_SEC = float(os.environ.get("FIRE_ALERT_WA_COOLDOWN", "60"))
# Seconds to wait after opening whatsapp:// before sending keys (focus + WhatsApp Desktop startup)
_WHATSAPP_LAUNCH_PAUSE_SEC = float(os.environ.get("FIRE_ALERT_WA_LAUNCH_PAUSE", "12"))

_SESSION_COOKIE = "fire_alert_portal_session"
_SESSION_TTL_SEC = 86400 * 7
_MJPEG_BOUNDARY = b"frame"


def _is_local_camera_source(s: str) -> bool:
    """True when the field is an OpenCV webcam index or common aliases (same idea as desktop app)."""
    if not s or not isinstance(s, str):
        return False
    t = s.strip().lower()
    if t in ("localhost", "webcam", "local"):
        return True
    if t.isdigit():
        return int(t) >= 0
    return False


def _local_camera_index(s: str) -> int:
    """Resolve webcam index from user input (0 & 1 for typical multi-camera setups)."""
    t = s.strip().lower()
    if t in ("localhost", "webcam", "local"):
        return 0
    if t.isdigit():
        return int(t)
    return 0


def _camera_display_name(source: str) -> str:
    """Readable label for DB rows / UI when source is RTSP vs local."""
    if not source:
        return source
    if _is_local_camera_source(source):
        return "Local camera %d" % _local_camera_index(source)
    return source[:180] if len(source) > 180 else source


def _open_video_capture(source: str):
    """OpenCV capture: local index (0, 1, …) or RTSP URL with FFMPEG backend."""
    if cv2 is None or not source:
        return None
    try:
        if _is_local_camera_source(source):
            cap = cv2.VideoCapture(_local_camera_index(source))
        else:
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if cap is not None:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            cap.release()
    except Exception:
        pass
    return None


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
        self._debounce_cleared_streak = 0
        self._episode_open_for_next_log = True

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
            self._debounce_cleared_streak = 0
            self._episode_open_for_next_log = True
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

    def _maybe_log_fire(self, rtsp_url: str, max_conf: float, annotated_frame=None) -> bool:
        """Persist one row + JPG only when FIRE_ALERT_LOG_COOLDOWN allows (no imwrite spam while fire persists)."""
        if not self._state or max_conf <= 0:
            return False
        now = time.time()
        if now - self._last_log_time < LOG_FIRE_COOLDOWN_SEC:
            return False
        snap_path = None
        try:
            if annotated_frame is not None:
                snap_path = self._save_snapshot(annotated_frame)
            cam = _camera_display_name(rtsp_url)
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
            self._last_log_time = now
            self._post_log_whatsapp(cam, details, max_conf, snap_path)
            return True
        except Exception:
            return False

    def _post_log_whatsapp(self, cam_label: str, details: str, max_conf: float, snap_path: str | None) -> None:
        """Optional: send WhatsApp links on detection (same machine as server)."""
        wt = None
        if self._state:
            wt = self._state.get("wa_times")
        if not wt or self._state is None:
            return
        try:
            cur = self._state["cursor"]
            if not load_whatsapp_auto_send(cur):
                return
            phones = load_whatsapp_phones_list(cur)
            if not phones:
                return
            wa_lock = self._state.get("wa_lock")
            if wa_lock is None:
                wa_lock = threading.Lock()
                self._state["wa_lock"] = wa_lock
            now_ts = time.time()
            with wa_lock:
                if now_ts - float(wt.get("last_msg", 0.0)) < WHATSAPP_MESSAGE_COOLDOWN_SEC:
                    return
                wt["last_msg"] = now_ts
            phones_snapshot = list(phones)
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = (
                "【火警偵測 / Fire alert】\n鏡頭: %s\n最高置信度: %.0f%%\n詳情: %s\n時間: %s\n（請開啟系統查看截圖）"
                % (cam_label, max_conf * 100, details[:400], time_str)
            )
            threading.Thread(
                target=_portal_whatsapp_send_worker,
                args=(phones_snapshot, msg, snap_path),
                daemon=True,
            ).start()
        except Exception:
            pass

    def _run_capture(self) -> None:
        url = self._url
        if not url or cv2 is None:
            return
        cap = _open_video_capture(url)
        if cap is None:
            return
        self._cap = cap
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

            eligible = False
            if FIRE_ALERT_DEBOUNCE_CHECKS <= 0:
                eligible = bool(fire)
            else:
                if fire:
                    self._debounce_cleared_streak = 0
                    eligible = self._episode_open_for_next_log
                else:
                    self._debounce_cleared_streak += 1
                    if self._debounce_cleared_streak >= FIRE_ALERT_DEBOUNCE_CHECKS:
                        self._episode_open_for_next_log = True

            if eligible and fire:
                snap = frame.copy()
                _draw_fire_boxes(snap, boxes)
                if self._maybe_log_fire(url, max_cf, annotated_frame=snap) and FIRE_ALERT_DEBOUNCE_CHECKS > 0:
                    self._episode_open_for_next_log = False
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
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS portal_settings (
            key TEXT PRIMARY KEY NOT NULL,
            value TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _portal_setting_get(cursor: sqlite3.Cursor, key: str, default: str | None = None) -> str | None:
    cursor.execute("SELECT value FROM portal_settings WHERE key = ?", (key,))
    row = cursor.fetchone()
    return row[0] if row else default


def _portal_setting_set(conn: sqlite3.Connection, cursor: sqlite3.Cursor, key: str, value: str) -> None:
    cursor.execute("INSERT OR REPLACE INTO portal_settings (key, value) VALUES (?, ?)", (key, value))
    conn.commit()


def _normalize_whatsapp_phone_line(s: str) -> str | None:
    t = (s or "").strip()
    if not t.startswith("+"):
        return None
    digits = "".join(c for c in t if c.isdigit())
    if len(digits) < 8:
        return None
    return t


def parse_whatsapp_phones_blob(blob: str) -> list[str]:
    out: list[str] = []
    for line in (blob or "").replace(",", "\n").splitlines():
        n = _normalize_whatsapp_phone_line(line.strip())
        if n and n not in out:
            out.append(n)
    return out


def load_whatsapp_phones_list(cursor: sqlite3.Cursor) -> list[str]:
    raw = _portal_setting_get(cursor, "whatsapp_phones_json", "[]") or "[]"
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            phones: list[str] = []
            for x in data:
                n = _normalize_whatsapp_phone_line(str(x).strip())
                if n and n not in phones:
                    phones.append(n)
            return phones
    except Exception:
        pass
    return []


def load_whatsapp_auto_send(cursor: sqlite3.Cursor) -> bool:
    return (_portal_setting_get(cursor, "whatsapp_auto_send", "0") or "0") == "1"


def save_whatsapp_preferences(conn: sqlite3.Connection, cursor: sqlite3.Cursor, phones: list[str], auto_send: bool) -> None:
    _portal_setting_set(conn, cursor, "whatsapp_phones_json", json.dumps(phones))
    _portal_setting_set(conn, cursor, "whatsapp_auto_send", "1" if auto_send else "0")


def _safe_snap_path_under_logs(path: str | None) -> str | None:
    """Only allow attaching images under logs_browser/ (detection jpgs)."""
    if not path or not isinstance(path, str):
        return None
    p = os.path.abspath(path)
    root = os.path.abspath(_LOG_DIR) + os.sep
    if not p.startswith(root) or not os.path.isfile(p):
        return None
    return p


def put_logs_browser_image_in_clipboard(image_path: str) -> bool:
    """Put JPG/PNG under logs_browser into Windows clipboard as BMP (paste into WhatsApp). Same strategy as desktop app."""
    if _safe_snap_path_under_logs(image_path) is None:
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


def put_text_in_clipboard(text: str) -> bool:
    """Put plain Unicode text to clipboard (Windows first, fallback to pyperclip)."""
    if text is None:
        return False
    s = str(text)
    try:
        if sys.platform == "win32":
            import win32clipboard

            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardText(s, win32clipboard.CF_UNICODETEXT)
            win32clipboard.CloseClipboard()
            return True
    except Exception:
        pass
    try:
        import pyperclip

        pyperclip.copy(s)
        return True
    except Exception:
        return False


_wa_logged_no_pyautogui = False
_wa_logged_clipboard_help = False


def _wa_prefers_ctrl_enter_send() -> bool:
    """True when user set WhatsApp to send with Ctrl+Enter (otherwise Enter inserts newline)."""
    v = (os.environ.get("FIRE_ALERT_WA_CTRL_ENTER_SEND") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _win_activate_whatsapp_window() -> bool:
    """Bring WhatsApp (Desktop) chat to the foreground so pyautogui hits the composer, not Chrome/browser."""
    if sys.platform != "win32":
        return False
    try:
        import win32con
        import win32gui
    except ImportError:
        return False

    strict_matches: list[int] = []

    def cb_strict(hwnd, _ctx) -> bool:
        try:
            if not win32gui.IsWindowVisible(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd)
            if not title:
                return True
            tl = title.lower()
            if "whatsapp" not in tl:
                return True
            if any(
                x in tl
                for x in (
                    " - google chrome",
                    "mozilla firefox",
                    "microsoft edge",
                )
            ):
                return True
            strict_matches.append(hwnd)
        except Exception:
            pass
        return True

    try:
        win32gui.EnumWindows(cb_strict, None)
    except Exception:
        return False

    pool = strict_matches
    if not pool:
        broad: list[int] = []

        def cb_broad(hwnd, _ctx) -> bool:
            try:
                if win32gui.IsWindowVisible(hwnd):
                    tl = win32gui.GetWindowText(hwnd).lower()
                    if "whatsapp" in tl:
                        broad.append(hwnd)
            except Exception:
                pass
            return True

        try:
            win32gui.EnumWindows(cb_broad, None)
        except Exception:
            return False
        pool = broad

    if not pool:
        return False
    try:
        hwnd = pool[0]
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        return True
    except Exception:
        return False


def _portal_whatsapp_press_send(pyautogui) -> None:
    """Try both send shortcuts; WhatsApp send key setting varies by machine."""
    if _wa_prefers_ctrl_enter_send():
        pyautogui.hotkey("ctrl", "enter")
        time.sleep(0.28)
        pyautogui.press("enter")
    else:
        pyautogui.press("enter")
        time.sleep(0.28)
        pyautogui.hotkey("ctrl", "enter")


def _wait_whatsapp_foreground(pyautogui, timeout_sec: float = 10.0) -> bool:
    """Wait until WhatsApp window is foreground; optionally click center to force focus."""
    deadline = time.time() + max(1.0, timeout_sec)
    focused_once = False
    while time.time() < deadline:
        focused_once = _win_activate_whatsapp_window() or focused_once
        if focused_once:
            time.sleep(0.18)
            return True
        time.sleep(0.3)
    # Fallback: one center click can dismiss browser protocol prompt and give focus to app
    try:
        w, h = pyautogui.size()
        pyautogui.click(w // 2, h // 2)
        time.sleep(0.25)
    except Exception:
        pass
    return _win_activate_whatsapp_window()


def _open_whatsapp_chat_prefer_desktop(wa_url: str, web_fallback_url: str) -> bool:
    """Open WhatsApp chat, preferring desktop app executables on Windows."""
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", "")
        candidates = [
            os.path.join(local, "WhatsApp", "WhatsApp.exe"),
            os.path.join(local, "Programs", "WhatsApp", "WhatsApp.exe"),
        ]
        for exe in candidates:
            if os.path.isfile(exe):
                try:
                    # Step 1: launch desktop client
                    subprocess.Popen([exe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    # Step 2: invoke registered protocol handler (avoid explorer.exe side effects)
                    time.sleep(0.7)
                    webbrowser.open(wa_url)
                    return True
                except Exception:
                    pass
    try:
        webbrowser.open(wa_url)
        return True
    except Exception:
        pass
    try:
        webbrowser.open(web_fallback_url)
        return True
    except Exception:
        return False


def _open_whatsapp_chat_for_number(digits: str, q_text: str) -> bool:
    """Open deep-link for an exact target number."""
    wa_url = "whatsapp://send?phone=%s&text=%s" % (digits, q_text)
    wa_web_url = "https://wa.me/%s?text=%s" % (digits, q_text)
    return _open_whatsapp_chat_prefer_desktop(wa_url, wa_web_url)


def _portal_whatsapp_send_worker(phones: list[str], message: str, snapshot_path: str | None = None) -> None:
    """Desktop-style send flow (same idea as fire_alert_desktop1.send_whatsapp_single)."""
    from urllib.parse import quote

    if not phones or not message:
        return

    snap = _safe_snap_path_under_logs(snapshot_path)
    clipboard_supported = snap is not None and sys.platform == "win32"

    pyautogui = None
    try:
        import pyautogui as _pa

        pyautogui = _pa
        pyautogui.PAUSE = 0.12
        pyautogui.FAILSAFE = False
    except Exception:
        pass

    global _wa_logged_no_pyautogui
    global _wa_logged_clipboard_help
    if not pyautogui and not _wa_logged_no_pyautogui:
        _wa_logged_no_pyautogui = True
        print("[WhatsApp] pip install pyautogui — then alerts will press Send and optionally paste snapshots.", file=sys.stderr)
        return

    for i, ph in enumerate(phones):
        digits = "".join(c for c in ph if c.isdigit())
        if len(digits) < 8:
            continue
        q = quote(message[:3500])

        if snap and clipboard_supported:
            put_logs_browser_image_in_clipboard(snap)

        try:
            # Open target chat, then re-open same deep-link once to reduce wrong-chat focus.
            if not _open_whatsapp_chat_for_number(digits, q):
                continue
            time.sleep(max(4.0, _WHATSAPP_LAUNCH_PAUSE_SEC))
            _wait_whatsapp_foreground(pyautogui, timeout_sec=8.0)
            _open_whatsapp_chat_for_number(digits, q)
            time.sleep(1.0)
            _wait_whatsapp_foreground(pyautogui, timeout_sec=3.0)
            # Force message into composer to avoid cases where deep-link text is visible
            # but not actually focused/sent.
            if put_text_in_clipboard(message[:3500]):
                try:
                    pyautogui.hotkey("ctrl", "a")
                    time.sleep(0.12)
                    pyautogui.press("backspace")
                    time.sleep(0.12)
                    pyautogui.hotkey("ctrl", "v")
                    time.sleep(0.25)
                except Exception:
                    pass
            _portal_whatsapp_press_send(pyautogui)
            if snap and clipboard_supported:
                time.sleep(1.0)
                if put_logs_browser_image_in_clipboard(snap):
                    pyautogui.hotkey("ctrl", "v")
                    time.sleep(0.8)
                    _wait_whatsapp_foreground(pyautogui, timeout_sec=3.0)
                    _portal_whatsapp_press_send(pyautogui)
                elif not _wa_logged_clipboard_help:
                    _wa_logged_clipboard_help = True
                    print("[WhatsApp] pip install Pillow pywin32 to paste JPG snapshots on Windows.", file=sys.stderr)
        except Exception as ex:
            print("[WhatsApp] automation failed:", ex, file=sys.stderr)

        if i < len(phones) - 1:
            time.sleep(3.0)


def purge_old_portal_data(conn: sqlite3.Connection, days: int) -> tuple[int, int, int]:
    """Remove fire_reports and camera_detections with created_at before (now - days). Deletes linked files under logs_browser only. Returns (reports_deleted, detections_deleted, files_removed)."""
    if days < 0:
        raise ValueError("days must be >= 0")
    cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    c = conn.cursor()
    c.execute("SELECT image_path FROM fire_reports WHERE created_at < ?", (cutoff,))
    paths_fr = [r[0] for r in c.fetchall()]
    n_fr = len(paths_fr)
    c.execute("SELECT image_path FROM camera_detections WHERE created_at < ?", (cutoff,))
    paths_cd = [r[0] for r in c.fetchall()]
    n_cd = len(paths_cd)
    log_abs = os.path.abspath(_LOG_DIR) + os.sep
    rep_abs = os.path.abspath(_REPORT_IMAGE_DIR) + os.sep
    files_removed = 0

    def try_remove(path: str | None) -> None:
        nonlocal files_removed
        if not path:
            return
        p = os.path.abspath(path)
        if not (p.startswith(log_abs) or p.startswith(rep_abs)):
            return
        try:
            if os.path.isfile(p):
                os.remove(p)
                files_removed += 1
        except OSError:
            pass

    for p in paths_fr:
        try_remove(p)
    for p in paths_cd:
        try_remove(p)
    c.execute("DELETE FROM fire_reports WHERE created_at < ?", (cutoff,))
    c.execute("DELETE FROM camera_detections WHERE created_at < ?", (cutoff,))
    conn.commit()
    return (n_fr, n_cd, files_removed)


def clear_all_alert_records(conn: sqlite3.Connection) -> tuple[int, int, int]:
    """Delete every row in fire_reports and camera_detections; remove linked images under logs_browser. Returns (reports_deleted, detections_deleted, files_removed)."""
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM fire_reports")
    n_fr = int(c.fetchone()[0])
    c.execute("SELECT COUNT(*) FROM camera_detections")
    n_cd = int(c.fetchone()[0])
    c.execute("SELECT image_path FROM fire_reports")
    paths_fr = [r[0] for r in c.fetchall()]
    c.execute("SELECT image_path FROM camera_detections")
    paths_cd = [r[0] for r in c.fetchall()]
    log_abs = os.path.abspath(_LOG_DIR) + os.sep
    rep_abs = os.path.abspath(_REPORT_IMAGE_DIR) + os.sep
    files_removed = 0

    def try_remove(path: str | None) -> None:
        nonlocal files_removed
        if not path:
            return
        p = os.path.abspath(path)
        if not (p.startswith(log_abs) or p.startswith(rep_abs)):
            return
        try:
            if os.path.isfile(p):
                os.remove(p)
                files_removed += 1
        except OSError:
            pass

    for p in paths_fr:
        try_remove(p)
    for p in paths_cd:
        try_remove(p)
    c.execute("DELETE FROM fire_reports")
    c.execute("DELETE FROM camera_detections")
    conn.commit()
    return (n_fr, n_cd, files_removed)


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
.alert-thumb { cursor: zoom-in; vertical-align: middle; }
#imgLightbox { display: none; position: fixed; inset: 0; z-index: 10000; background: rgba(0,0,0,.88); align-items: center; justify-content: center; padding: 16px; cursor: zoom-out; }
#imgLightbox img { max-width: 95vw; max-height: 95vh; object-fit: contain; border-radius: 8px; pointer-events: none; box-shadow: 0 8px 40px rgba(0,0,0,.45); }
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
<div id="imgLightbox" role="dialog" aria-label="Full image">
  <img id="imgLightboxImg" alt="">
</div>
<script>
(function(){
  var box=document.getElementById('imgLightbox');
  var big=document.getElementById('imgLightboxImg');
  if(!box||!big)return;
  function close(){ box.style.display='none'; big.removeAttribute('src'); }
  document.addEventListener('dblclick',function(e){
    var el=e.target;
    if(!el||el.tagName!=='IMG'||!el.classList.contains('alert-thumb'))return;
    e.preventDefault();
    big.src=el.currentSrc||el.src;
    box.style.display='flex';
  });
  box.addEventListener('click',close);
  document.addEventListener('keydown',function(e){ if(e.key==='Escape'&&box.style.display==='flex')close(); });
})();
</script>
</body></html>""" % (
        u,
        "".join(nav_html),
        html.escape(title),
        filters_html,
        content_html,
    )


def _is_report_record(record_type: object) -> bool:
    return str(record_type or "").strip().lower() == "report"


def _alert_thumb_img(url: str) -> str:
    """Small table thumbnail; double-click opens full size via #imgLightbox in dashboard layout."""
    return (
        '<img class="alert-thumb" src="%s" title="Double-click to enlarge" '
        'style="width:88px;height:50px;object-fit:cover;border:1px solid #343a44;border-radius:6px;background:#0f1115;">'
        % html.escape(url, quote=True)
    )


def _alerts_context_menu_html(report_mode: str, report_day: str, report_month: str) -> str:
    """Right-click delete for Report Fire (report-row) and camera detection (detection-row) rows."""
    mode = html.escape(report_mode if report_mode in ("all", "day", "month") else "all")
    day = html.escape(report_day or "")
    month = html.escape(report_month or "")
    return """
<style>.report-row,.detection-row{{cursor:context-menu}}tr.report-row.ctx-selected,tr.detection-row.ctx-selected{{background:rgba(248,113,113,.14)!important;outline:1px solid rgba(248,113,113,.35);}}</style>
<div id="ctxParams" data-mode="{mode}" data-day="{day}" data-month="{month}" style="display:none"></div>
<div id="ctxMenu" style="display:none;position:fixed;z-index:9999;background:#23262d;border:1px solid #353a44;border-radius:8px;padding:6px 0;min-width:188px;box-shadow:0 8px 24px rgba(0,0,0,.45);">
<button type="button" id="ctxDelete" style="width:100%;text-align:left;padding:8px 12px;background:transparent;border:none;color:#f87171;cursor:pointer;font:inherit;">Delete</button>
</div>
<script>
(function(){{
var menu=document.getElementById('ctxMenu');
var btn=document.getElementById('ctxDelete');
var params=document.getElementById('ctxParams');
if(!menu||!btn||!params)return;
var kind=null;
var rowId=null;
var selected=null;
function hideMenu(){{
menu.style.display='none';
if(selected){{selected.classList.remove('ctx-selected');selected=null;}}
kind=null;rowId=null;
}}
document.addEventListener('contextmenu',function(e){{
var t=e.target;
var tr=t&&t.closest?t.closest('tr.report-row, tr.detection-row'):null;
if(!tr)return;
e.preventDefault();
e.stopPropagation();
if(tr.classList.contains('report-row')){{
kind='report';rowId=tr.getAttribute('data-report-id');
btn.textContent='Delete report';
}}else if(tr.classList.contains('detection-row')){{
kind='detection';rowId=tr.getAttribute('data-detection-id');
btn.textContent='Delete camera detection';
}}else{{return;}}
if(!rowId)return;
if(selected)selected.classList.remove('ctx-selected');
selected=tr;
tr.classList.add('ctx-selected');
menu.style.display='block';
var x=e.clientX,y=e.clientY;
menu.style.left=x+'px';
menu.style.top=y+'px';
var mw=menu.offsetWidth,mh=menu.offsetHeight;
if(x+mw>window.innerWidth-4)menu.style.left=Math.max(4,x-mw)+'px';
if(y+mh>window.innerHeight-4)menu.style.top=Math.max(4,y-mh)+'px';
}},true);
document.addEventListener('click',function(e){{
if(menu.style.display!=='none'&&!menu.contains(e.target))hideMenu();
}});
document.addEventListener('keydown',function(e){{if(e.key==='Escape')hideMenu();}});
btn.addEventListener('click',function(){{
if(!rowId||!kind)return;
var msg=kind==='report'?'Delete this report?':'Delete this camera detection record?';
if(!confirm(msg)){{hideMenu();return;}}
var f=document.createElement('form');
f.method='POST';
if(kind==='report'){{
f.action='/home/report-delete';
[['report_id',rowId],['report_mode',params.dataset.mode],['report_day',params.dataset.day||''],['report_month',params.dataset.month||'']].forEach(function(x){{
var i=document.createElement('input');i.type='hidden';i.name=x[0];i.value=x[1];f.appendChild(i);
}});
}}else{{
f.action='/home/detection-delete';
[['detection_id',rowId],['report_mode',params.dataset.mode],['report_day',params.dataset.day||''],['report_month',params.dataset.month||'']].forEach(function(x){{
var i=document.createElement('input');i.type='hidden';i.name=x[0];i.value=x[1];f.appendChild(i);
}});
}}
document.body.appendChild(f);
f.submit();
}});
}})();
</script>
""".format(
        mode=mode,
        day=day,
        month=month,
    )


def dashboard_page_html(username: str, recent_rows: list[tuple], notice: str = "") -> str:
    notice_html = '<div class="notice">%s</div>' % html.escape(notice) if notice else ""
    clear_form = """
<div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:10px;margin-bottom:10px;">
  <h3 style="margin:0;">Recent Alerts</h3>
  <form method="post" action="/home/clear-all-alerts" style="margin:0;" onsubmit="return confirm('Clear all fire reports and camera detection records? This cannot be undone.');">
    <button class="submit" type="submit" style="padding:8px 14px;font-size:.85rem;background:#374151;">Clear all records</button>
  </form>
</div>"""
    if recent_rows:
        items = []
        for ts, source, severity, details, image_path, record_type, record_id in recent_rows:
            img_html = '<span class="muted">-</span>'
            if image_path:
                q = urllib.parse.quote(image_path, safe="")
                if _is_report_record(record_type):
                    img_html = _alert_thumb_img("/home/report-image?path=%s" % q)
                else:
                    img_html = _alert_thumb_img("/home/detection-image?path=%s" % q)
            if _is_report_record(record_type):
                tr_open = '<tr class="alert-row report-row" data-report-id="%s">' % html.escape(str(record_id))
            else:
                tr_open = '<tr class="alert-row detection-row" data-detection-id="%s">' % html.escape(str(record_id))
            items.append(
                "%s<td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                % (
                    tr_open,
                    html.escape(str(ts)),
                    html.escape(source),
                    html.escape(severity),
                    html.escape(details),
                    img_html,
                )
            )
        content = (
            """
<div class="panel">
  %s
  %s
  <p class="muted" style="font-size:0.82rem;margin:0 0 10px 0;line-height:1.35;">Right-click a row to delete: <strong>Report Fire</strong> or <strong>camera detection</strong> (same as the Action column on All Alerts).</p>
  <table>
    <thead><tr><th>Time</th><th>Source</th><th>Severity</th><th>Details</th><th>Image</th></tr></thead>
    <tbody>%s</tbody>
  </table>
</div>%s"""
            % (clear_form, notice_html, "".join(items), _alerts_context_menu_html("all", "", ""))
        )
    else:
        content = """
<div class="panel">
  %s
  %s
  <div class="empty">No alerts found</div>
</div>""" % (
            clear_form,
            notice_html,
        )
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
                q = urllib.parse.quote(image_path, safe="")
                if _is_report_record(record_type):
                    img_html = _alert_thumb_img("/home/report-image?path=%s" % q)
                else:
                    img_html = _alert_thumb_img("/home/detection-image?path=%s" % q)
            action_html = '<span class="muted">-</span>'
            if _is_report_record(record_type):
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
                tr_open = '<tr class="alert-row report-row" data-report-id="%s">' % html.escape(
                    str(record_id)
                )
            else:
                action_html = (
                    '<form method="post" action="/home/detection-delete" style="margin:0;">'
                    '<input type="hidden" name="detection_id" value="%s">'
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
                tr_open = '<tr class="alert-row detection-row" data-detection-id="%s">' % html.escape(
                    str(record_id)
                )
            body.append(
                "%s<td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                % (
                    tr_open,
                    html.escape(str(ts)),
                    html.escape(source),
                    html.escape(severity),
                    html.escape(details),
                    html.escape(status),
                    img_html,
                    action_html,
                )
            )
        ctx_block = _alerts_context_menu_html(mode, report_day or "", report_month or "")
        table = """
<div class="panel">
  <h3>All Alerts</h3>
  <p class="muted" style="font-size:0.82rem;margin:0 0 10px 0;line-height:1.35;">Right-click a row to delete <strong>Report Fire</strong> or <strong>camera detection</strong>, or use the Delete button in the Action column.</p>
  <table>
    <thead><tr><th>Time</th><th>Source</th><th>Severity</th><th>Details</th><th>Status</th><th>Image</th><th>Action</th></tr></thead>
    <tbody>%s</tbody>
  </table>
</div>%s""" % ("".join(body), ctx_block)
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
    poll_since_id: int = 0,
    wa_phones_lines: str = "",
    wa_auto_send: bool = False,
    camera_notice: str = "",
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
            '<h3 style="margin:0;">Live camera preview</h3>'
            '<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">'
            '<form method="post" action="/home/camera-detection/stop" style="margin:0;">'
            '<button type="submit" class="submit" style="margin:0;background:#475569;">Stop detection</button>'
            "</form>"
            '<button class="submit" type="button" style="margin:0;" onclick="toggleStreamFullScreen()">Full Screen</button>'
            "</div>"
            "</div>"
            '<div id="streamWrap" style="width:100%%;height:min(78vh,900px);background:#0b0d10;border:1px solid #343a44;border-radius:6px;overflow:hidden;">'
            '<img id="rtspStream" src="%s" alt="Camera stream" style="width:100%%;height:100%%;object-fit:contain;background:#0b0d10;">'
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

    local_preset = [("conf", "%.2f" % conf_value)] + [("models", m) for m in selected_models]
    local0_href = "/home/camera-detection?" + urllib.parse.urlencode([("url", "0")] + local_preset)
    local1_href = "/home/camera-detection?" + urllib.parse.urlencode([("url", "1")] + local_preset)

    boot = {
        "sinceId": int(poll_since_id),
        "waAuto": bool(wa_auto_send),
        "pollMs": 1500,
    }
    boot_json = json.dumps(boot, ensure_ascii=False).replace("<", "\\u003c")

    notify_block = """<style>
#detectNotifyBar{position:sticky;top:0;z-index:100;display:none;align-items:flex-start;gap:12px;flex-wrap:wrap;
  margin:-8px -8px 14px;padding:12px 14px;border-radius:8px;background:linear-gradient(90deg,rgba(220,38,38,.95),rgba(251,146,60,.88));
  border:1px solid rgba(254,243,199,.35);color:#fff;box-shadow:0 10px 30px rgba(0,0,0,.35);}
#detectNotifyBar.show{display:flex;}
#detectNotifyBar .msg{flex:1;min-width:220px;font-size:0.95rem;line-height:1.45;margin:0;}
#detectNotifyBar button{background:#fff;color:#991b1b;border:none;padding:8px 12px;border-radius:6px;font-weight:700;cursor:pointer;}
#detectNotifyBar .sub{border-left:1px solid rgba(255,255,255,.35);padding-left:12px;display:flex;align-items:center;gap:8px;}
#waBackdrop{position:fixed;inset:0;background:rgba(0,0,0,.55);display:none;z-index:200;align-items:center;justify-content:center;padding:16px;}
#waBackdrop.show{display:flex;}
#waModal{background:#23262d;border:1px solid #3d4450;border-radius:12px;max-width:480px;width:100%%;padding:18px 20px;box-shadow:0 24px 50px rgba(0,0,0,.5);}
#waModal h3{margin:0 0 6px;font-size:1.1rem;}
#waModal .sub{margin:0 0 14px;font-size:0.85rem;color:#9ca3af;line-height:1.4;}
#waPhones{width:100%%;min-height:100px;font-family:inherit;font-size:0.92rem;background:#15181e;color:#f3f4f6;border:1px solid #3d4450;border-radius:8px;padding:10px;}
.wa-row{display:flex;justify-content:space-between;align-items:center;gap:12px;margin:12px 0;flex-wrap:wrap;}
.cam-toolbar{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:12px;}
</style>
<div class="cam-toolbar">
  <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
    <button class="submit" type="button" id="btnOpenWaModal" style="margin:0;">Manage WhatsApp recipients</button>
    <span class="muted" style="font-size:0.85rem;line-height:1.35;">After a detection, WhatsApp send links open on the PC running this server (not in the browser).</span>
  </div>
</div>
<div id="detectNotifyBar" role="alert" aria-live="polite">
  <p class="msg" id="detectNotifyText"></p>
  <div class="sub">
    <button type="button" id="btnNotifyWa">Manage numbers</button>
    <button type="button" id="btnNotifyDismiss">Dismiss</button>
  </div>
</div>
<div id="waBackdrop"><div id="waModal">
  <h3>WhatsApp recipients</h3>
  <p class="sub">One number per line, international format starting with <code style="background:#15181e;padding:2px 6px;border-radius:4px;">+</code>.<br/>
  Turn on &quot;Auto-send&quot; below to message on detection; saving applies to new events immediately.</p>
  <label for="waPhones" class="muted" style="font-size:0.86rem;display:block;margin-bottom:6px;">Phone numbers</label>
  <textarea id="waPhones" placeholder="+886912345678\n+85291234567">%s</textarea>
  <div class="wa-row">
    <span class="muted" style="font-size:0.9rem;">Auto-send WhatsApp after detection</span>
    <label style="display:inline-flex;align-items:center;gap:8px;cursor:pointer;">
      <input type="checkbox" id="waAutoToggle"%s style="width:44px;height:22px;">
      <span id="waAutoLabel">%s</span>
    </label>
  </div>
  <p id="waSaveMsg" class="muted" style="min-height:1.25em;margin:8px 0 0;font-size:0.86rem;"></p>
  <div style="display:flex;justify-content:flex-end;gap:10px;margin-top:14px;">
    <button type="button" class="submit" style="background:#4b5563;margin:0;" id="btnWaCancel">Cancel</button>
    <button type="button" class="submit" id="btnWaSave" style="margin:0;">Save</button>
  </div>
</div></div>
<script type="application/json" id="cam-det-boot-json">%s</script>
<script>
(function(){
var bootEl=document.getElementById('cam-det-boot-json');
var boot={sinceId:0,pollMs:1500,waAuto:false};
try{if(bootEl&&bootEl.textContent)boot=JSON.parse(bootEl.textContent);}catch(e){}
var sinceId=typeof boot.sinceId==='number'?boot.sinceId:0;
var pollMs=typeof boot.pollMs==='number'?boot.pollMs:1500;

var bar=document.getElementById('detectNotifyBar');
var txt=document.getElementById('detectNotifyText');
var bd=document.getElementById('waBackdrop');
function openWa(){if(bd)bd.classList.add('show');fetch('/home/camera-detection/api/settings',{credentials:'same-origin'})
  .then(function(r){return r.json();})
  .then(function(j){var ta=document.getElementById('waPhones');var tg=document.getElementById('waAutoToggle');var lb=document.getElementById('waAutoLabel');
    if(j&&j.phones&&ta)ta.value=j.phones.join(String.fromCharCode(10));if(j&&typeof j.auto_send==='boolean'&&tg){tg.checked=j.auto_send;lb.textContent=j.auto_send?'On':'Off';}}).catch(function(){});}
function closeWa(){if(bd)bd.classList.remove('show');}
function showNotify(summary){
  if(!txt||!bar)return;
  txt.textContent=summary||'';bar.classList.add('show');
}
var ob=document.getElementById('btnOpenWaModal');if(ob)ob.addEventListener('click',function(e){e.preventDefault();openWa();});
var bn=document.getElementById('btnNotifyWa');if(bn)bn.addEventListener('click',function(e){e.preventDefault();openWa();});
var bnd=document.getElementById('btnNotifyDismiss');if(bnd)bnd.addEventListener('click',function(){if(bar)bar.classList.remove('show');});
var bwc=document.getElementById('btnWaCancel');if(bwc)bwc.addEventListener('click',closeWa);
if(bd)bd.addEventListener('click',function(e){if(e.target===bd)closeWa();});

var waT=document.getElementById('waAutoToggle');
var waL=document.getElementById('waAutoLabel');
if(waT&&waL){waT.addEventListener('change',function(){waL.textContent=waT.checked?'On':'Off';});}

var bws=document.getElementById('btnWaSave');if(bws)bws.addEventListener('click',function(){
  var phonesRaw=(document.getElementById('waPhones')&&document.getElementById('waPhones').value)||'';
  var wtg=document.getElementById('waAutoToggle');var auto=wtg&&wtg.checked;
  var sm=document.getElementById('waSaveMsg');
  fetch('/home/camera-detection/api/settings',{method:'POST',credentials:'same-origin',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({phones_text:phonesRaw,auto_send:!!auto})}).then(function(r){return r.json().then(function(j){return {ok:r.ok,j:j};});})
  .then(function(o){if(sm)sm.style.color=o.ok?'#86efac':'#fca5a5';if(sm)sm.textContent=o.ok?(o.j.saved||'Saved'):(o.j.error||'Save failed');}).catch(function(){if(sm){sm.style.color='#fca5a5';sm.textContent='Network error';}});
});

function poll(){
  fetch('/home/camera-detection/poll-detections?since_id='+encodeURIComponent(sinceId),{credentials:'same-origin'})
    .then(function(r){return r.json();}).then(function(data){
       if(!data||!data.rows||!data.rows.length)return;
       sinceId=data.max_id!=null?data.max_id:sinceId;
       var last=data.rows[data.rows.length-1];
       var summary='Detection: '+ (last.camera_name||'') +' — '+ (last.created_at||'') +' — '+ (last.status||'');
       showNotify(summary);
    }).catch(function(){});
}
poll();
setInterval(poll,pollMs);
document.addEventListener('keydown',function(e){if(e.key==='Escape'&&bd&&bd.classList.contains('show'))closeWa();});
})();
</script>""" % (
        html.escape(wa_phones_lines),
        " checked" if wa_auto_send else "",
        html.escape("On" if wa_auto_send else "Off"),
        boot_json,
    )

    form_html = """
<div class="panel" style="margin-bottom:12px;">
  <h3>Camera source</h3>
  <p class="muted" style="font-size:0.88rem;margin:0 0 10px 0;line-height:1.4;">Enter an <strong>RTSP</strong> URL, or a <strong>local webcam index</strong> (<code>0</code> = first camera, <code>1</code> = second). Aliases <code>webcam</code> / <code>local</code> use index 0.</p>
  <form method="get" action="/home/camera-detection" style="display:grid;gap:10px;">
    <input type="text" name="url" placeholder="rtsp://… or 0 / 1 for local cameras" value="%s" style="flex:1;min-width:320px;">
    <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
      <label style="margin:0;">Confidence</label>
      <input type="number" name="conf" min="0.01" max="0.99" step="0.01" value="%s" style="width:100px;">
      <div class="muted" style="font-size:0.85rem;">0.01 - 0.99</div>
    </div>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
      <div class="muted" style="font-size:0.9rem;">Models:</div>
      %s
    </div>
    <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
      <button class="submit" type="submit">Connect</button>
      <span class="muted" style="font-size:0.88rem;">Quick:</span>
      <a class="submit" style="display:inline-block;padding:8px 14px;text-decoration:none;" href="%s">Local camera 0</a>
      <a class="submit" style="display:inline-block;padding:8px 14px;text-decoration:none;" href="%s">Local camera 1</a>
    </div>
  </form>
</div>""" % (escaped_rtsp, ("%.2f" % conf_value), "".join(checks), local0_href, local1_href)

    notice_banner = ""
    if camera_notice:
        notice_banner = '<div class="notice" style="margin-bottom:12px;">%s</div>' % html.escape(camera_notice)
    content = notify_block + notice_banner + yolo_html + form_html + stream_box + table_html
    return _dashboard_layout_html(username, "camera-detection", "Camera Detection", content)


def main() -> None:
    host = "127.0.0.1"
    port = 8766
    purge_days: int | None = None
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == "--purge-days" and i + 1 < len(argv):
            try:
                purge_days = int(argv[i + 1])
            except ValueError:
                print("Invalid --purge-days value", file=sys.stderr)
                sys.exit(2)
            i += 2
            continue
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

    if purge_days is not None:
        if purge_days < 0:
            print("--purge-days must be >= 0", file=sys.stderr)
            sys.exit(2)
        conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _init_db(conn)
        try:
            n_fr, n_cd, n_files = purge_old_portal_data(conn, purge_days)
            print(
                "Purged rows older than %d day(s): fire_reports=%d, camera_detections=%d; image files removed=%d"
                % (purge_days, n_fr, n_cd, n_files)
            )
            print("Database: %s" % _DB_PATH)
        finally:
            conn.close()
        return

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

    wa_state = {"last_msg": 0.0}
    wa_lock = threading.Lock()
    state = {
        "conn": conn,
        "cursor": cursor,
        "sessions": sessions,
        "sess_lock": sess_lock,
        "yolo_models": yolo_models,
        "yolo_model_names": yolo_model_names,
        "yolo_status_text": yolo_status_text,
        "wa_times": wa_state,
        "wa_lock": wa_lock,
        "rtsp_bridge": RtspBridge(
            state={"conn": conn, "cursor": cursor, "wa_times": wa_state, "wa_lock": wa_lock},
            models_map=yolo_models,
        ),
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

        def _send_json(self, obj: dict | list, status: int = 200) -> None:
            raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

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
                cleared = (qs.get("cleared") or [""])[0] == "1"
                notice = "All alert records have been cleared." if cleared else ""
                state["cursor"].execute(
                    """
                    SELECT created_at, source, severity, details, image_path, record_type, record_id
                    FROM (
                        SELECT created_at, 'Report Fire' AS source, severity, description AS details, image_path, 'report' AS record_type, id AS record_id
                        FROM fire_reports
                        UNION ALL
                        SELECT created_at, camera_name AS source, severity, details, image_path, 'detection' AS record_type, id AS record_id
                        FROM camera_detections
                    )
                    ORDER BY created_at DESC
                    LIMIT 10
                    """
                )
                rows = state["cursor"].fetchall()
                self._send_html(dashboard_page_html(u, rows, notice=notice))
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
                    ok_source = bool(_is_local_camera_source(rtsp_url) or rtsp_url.lower().startswith("rtsp://"))
                    if not ok_source:
                        stream_error = "Invalid camera source: use rtsp://… or a local index (e.g. 0, 1) or webcam / local."
                    elif cv2 is None:
                        stream_error = "OpenCV is not available. Install with: pip install opencv-python"
                state["cursor"].execute(
                    "SELECT created_at, camera_name, status, severity, COALESCE(details, '') FROM camera_detections ORDER BY created_at DESC LIMIT 300"
                )
                rows = state["cursor"].fetchall()
                state["cursor"].execute("SELECT COALESCE(MAX(id), 0) FROM camera_detections")
                poll_since_id = int(state["cursor"].fetchone()[0])
                wa_list = load_whatsapp_phones_list(state["cursor"])
                wa_lines = "\n".join(wa_list)
                wa_auto = load_whatsapp_auto_send(state["cursor"])
                stopped_ok = (qs.get("stopped") or [""])[0] == "1"
                cam_notice = "Camera capture and fire detection stopped." if stopped_ok else ""
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
                        poll_since_id=poll_since_id,
                        wa_phones_lines=wa_lines,
                        wa_auto_send=wa_auto,
                        camera_notice=cam_notice,
                    )
                )
                return

            if parsed.path == "/home/camera-detection/api/settings":
                u = session_user(self)
                if not u:
                    self.send_error(401, "Login required")
                    return
                phones = load_whatsapp_phones_list(state["cursor"])
                self._send_json({"phones": phones, "auto_send": load_whatsapp_auto_send(state["cursor"])})
                return

            if parsed.path == "/home/camera-detection/poll-detections":
                u = session_user(self)
                if not u:
                    self.send_error(401, "Login required")
                    return
                since_raw = (qs.get("since_id") or ["0"])[0]
                try:
                    since_id = int(since_raw)
                except ValueError:
                    since_id = 0
                state["cursor"].execute(
                    """
                    SELECT id, created_at, camera_name, status, severity, COALESCE(details, '')
                    FROM camera_detections WHERE id > ? ORDER BY id ASC LIMIT 40
                    """,
                    (since_id,),
                )
                rows_raw = state["cursor"].fetchall()
                state["cursor"].execute("SELECT COALESCE(MAX(id), 0) FROM camera_detections")
                max_id = int(state["cursor"].fetchone()[0])
                self._send_json(
                    {
                        "max_id": max_id,
                        "rows": [
                            {
                                "id": r[0],
                                "created_at": r[1],
                                "camera_name": r[2],
                                "status": r[3],
                                "severity": r[4],
                                "details": (r[5] or "")[:220],
                            }
                            for r in rows_raw
                        ],
                    }
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
                ok_source = bool(_is_local_camera_source(rtsp_url) or rtsp_url.lower().startswith("rtsp://"))
                if not ok_source:
                    self.send_error(400, "url must be rtsp://… or a local camera index (0, 1, …)")
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

            if parsed.path == "/home/camera-detection/stop":
                u = session_user(self)
                if not u:
                    self._redirect("/")
                    return
                try:
                    state["rtsp_bridge"].stop()
                except Exception:
                    pass
                self._redirect("/home/camera-detection?stopped=1")
                return

            if parsed.path == "/home/camera-detection/api/settings":
                u = session_user(self)
                if not u:
                    self.send_error(401, "Login required")
                    return
                n = int(self.headers.get("Content-Length", 0))
                if n <= 0 or n > 100_000:
                    self._send_json({"error": "Invalid body"}, 400)
                    return
                raw = self.rfile.read(n)
                try:
                    data = json.loads(raw.decode("utf-8"))
                except Exception:
                    self._send_json({"error": "Invalid JSON"}, 400)
                    return
                phones_out: list[str] = []
                if isinstance(data.get("phones_text"), str):
                    phones_out = parse_whatsapp_phones_blob(data["phones_text"])
                elif isinstance(data.get("phones"), list):
                    for x in data["phones"]:
                        nline = _normalize_whatsapp_phone_line(str(x).strip())
                        if nline and nline not in phones_out:
                            phones_out.append(nline)
                auto = bool(data.get("auto_send", False))
                try:
                    save_whatsapp_preferences(state["conn"], state["cursor"], phones_out, auto)
                    self._send_json({"saved": "Saved", "phones": phones_out, "auto_send": auto})
                except Exception:
                    self._send_json({"error": "Failed to save settings"}, 500)
                return

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

            if parsed.path == "/home/clear-all-alerts":
                u = session_user(self)
                if not u:
                    self._redirect("/")
                    return
                fields_clear = self._read_form()
                if fields_clear is None:
                    self.send_error(413)
                    return
                clear_all_alert_records(state["conn"])
                self._redirect("/home?cleared=1")
                return

            if parsed.path == "/home/report-delete":
                u = session_user(self)
                if not u:
                    self._redirect("/")
                    return
                fields = self._read_form()
                if fields is None:
                    self.send_error(413)
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
                            log_root = os.path.abspath(_LOG_DIR) + os.sep
                            if abs_path.startswith(log_root) and os.path.isfile(abs_path):
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

            if parsed.path == "/home/detection-delete":
                u = session_user(self)
                if not u:
                    self._redirect("/")
                    return
                fields = self._read_form()
                if fields is None:
                    self.send_error(413)
                    return
                det_raw = self._one(fields, "detection_id").strip()
                report_mode = self._one(fields, "report_mode").strip() or "all"
                report_day = self._one(fields, "report_day").strip()
                report_month = self._one(fields, "report_month").strip()
                try:
                    det_id = int(det_raw)
                except Exception:
                    self._redirect("/home/all-alerts")
                    return
                state["cursor"].execute("SELECT image_path FROM camera_detections WHERE id = ?", (det_id,))
                row = state["cursor"].fetchone()
                if row:
                    image_path = row[0]
                    state["cursor"].execute("DELETE FROM camera_detections WHERE id = ?", (det_id,))
                    state["conn"].commit()
                    try:
                        if image_path:
                            abs_path = os.path.abspath(image_path)
                            log_root = os.path.abspath(_LOG_DIR) + os.sep
                            if abs_path.startswith(log_root) and os.path.isfile(abs_path):
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

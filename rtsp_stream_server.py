#!/usr/bin/env python3
"""
Local RTSP → MJPEG bridge for the web camera page.
OpenCV reads RTSP; the browser shows the stream via <img> multipart MJPEG.

Run (from project folder or this directory):
  python python/rtsp_stream_server.py

Default: http://127.0.0.1:8765
Requires: pip install opencv-python (or opencv-python-headless)
"""

from __future__ import annotations

import os
import sys
import time
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

try:
    import cv2
except ImportError:
    print("Missing OpenCV. Install with: pip install opencv-python", file=sys.stderr)
    sys.exit(1)

# Prefer TCP for RTSP (more reliable on many cameras)
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
BOUNDARY = b"frame"


class RtspBridge:
    """Single background reader; many HTTP clients reuse the same JPEG feed."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cap: cv2.VideoCapture | None = None
        self._url: str | None = None
        self._latest_jpeg: bytes | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def ensure_stream(self, rtsp_url: str) -> None:
        with self._lock:
            if self._url == rtsp_url and self._thread and self._thread.is_alive():
                return
            self._stop_unlocked()
            self._url = rtsp_url
            self._stop.clear()
            self._thread = threading.Thread(target=self._run_capture, daemon=True)
            self._thread.start()

    def _run_capture(self) -> None:
        url = self._url
        if not url:
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
                time.sleep(0.05)
                continue
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                with self._lock:
                    self._latest_jpeg = buf.tobytes()
            time.sleep(1.0 / 25.0)
        try:
            cap.release()
        except Exception:
            pass
        with self._lock:
            self._cap = None

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def stop(self) -> None:
        with self._lock:
            self._stop_unlocked()

    def _stop_unlocked(self) -> None:
        self._stop.set()
        t = self._thread
        self._thread = None
        self._url = None
        self._latest_jpeg = None
        cap = self._cap
        self._cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if t and t.is_alive():
            t.join(timeout=2.0)


bridge = RtspBridge()


def _cors(handler: BaseHTTPRequestHandler) -> None:
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        _cors(self)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/health"):
            body = b'{"ok":true,"service":"rtsp-mjpeg-bridge"}\n'
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            _cors(self)
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/video_feed":
            qs = urllib.parse.parse_qs(parsed.query)
            raw = (qs.get("url") or [""])[0]
            rtsp_url = urllib.parse.unquote(raw).strip()
            if not rtsp_url.startswith("rtsp://"):
                self.send_error(400, "url must be rtsp://...")
                return
            bridge.ensure_stream(rtsp_url)
            # Wait briefly for first frame
            deadline = time.time() + 30.0
            while time.time() < deadline and bridge.get_jpeg() is None:
                time.sleep(0.05)
            if bridge.get_jpeg() is None:
                self.send_error(504, "RTSP not ready")
                return

            self.send_response(200)
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=" + BOUNDARY.decode(),
            )
            self.send_header("Cache-Control", "no-cache, no-store")
            _cors(self)
            self.end_headers()

            try:
                while True:
                    chunk = bridge.get_jpeg()
                    if chunk:
                        self.wfile.write(b"--" + BOUNDARY + b"\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(chunk)
                        self.wfile.write(b"\r\n")
                    time.sleep(1.0 / 20.0)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            return

        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/stop":
            bridge.stop()
            body = b'{"ok":true}\n'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            _cors(self)
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    host = os.environ.get("RTSP_BRIDGE_HOST", DEFAULT_HOST)
    port = int(os.environ.get("RTSP_BRIDGE_PORT", str(DEFAULT_PORT)))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"RTSP MJPEG bridge: http://{host}:{port}/")
    print("  GET /video_feed?url=<urlencoded rtsp url>")
    print("  POST /stop  — release camera")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop()
        server.server_close()


if __name__ == "__main__":
    main()

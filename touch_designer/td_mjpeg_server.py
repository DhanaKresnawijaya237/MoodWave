"""
MoodWave — TouchDesigner MJPEG Server
======================================
Serves the render output of a TOP as an MJPEG stream so the browser
can display it at http://localhost:9000/video.

HOW TO USE IN TOUCHDESIGNER
----------------------------
1. Create a Text DAT, paste this entire script into it.
2. Set TOP_PATH below to the path of your output TOP (default: /project1/out1).
3. Create an Execute DAT and turn ON BOTH callbacks below:
     - "Start" callback (runs once when project starts)
         exec(op('text1').text)
     - "Frame Start" callback (runs every frame — REQUIRED, this is what
       actually pushes frames into the stream)
         op('text1').module.update_frame()
   Replace 'text1' with the name of your Text DAT.
4. Save and re-open the project, or manually pulse the Execute DAT's Start.
5. You should see "[MoodWave] MJPEG server running on http://localhost:9000" in
   the Textport. Open http://localhost:9000/health — it should report
   "frames=ok". If it says "frames=stale" or "frames=none", the Frame Start
   callback is not wired.

To restart the server at any time, run:  start_server()  in a Script DAT.
"""

import socket
import socketserver
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import numpy as np
import cv2


class _DualStackHTTPServer(ThreadingHTTPServer):
    """Threaded HTTPServer that listens on both IPv4 and IPv6.

    - Threaded: the /video MJPEG stream is an infinite loop and would
      otherwise block all other requests (including /health) on the single
      worker thread, causing the browser to show "refused to connect"
      after the first connection.
    - Dual-stack: Windows often resolves 'localhost' to ::1 (IPv6); a plain
      AF_INET bind would silently miss those clients.
    """
    address_family = socket.AF_INET6
    daemon_threads = True       # don't block shutdown on hung streams
    allow_reuse_address = True  # avoid 'address already in use' on quick restarts

    def server_bind(self):
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()

# ── Config ──────────────────────────────────────────────────────────────────
TOP_PATH     = '/project1/out1'  # full path to your output TOP inside TD
PORT         = 9000              # must match TD_FEED_URL port in moodwave.html
JPEG_QUALITY = 75                # 1–95: lower = smaller/faster, higher = sharper
TARGET_FPS   = 30
# ────────────────────────────────────────────────────────────────────────────

# Shared frame buffer — written by main TD thread, read by server thread
_frame_lock   = threading.Lock()
_frame_buffer = None             # bytes or None
_last_frame_t = 0.0              # time.time() of last successful update_frame
_last_error   = None             # last update_frame error, surfaced via /health

_server        = None
_server_thread = None


def update_frame():
    """
    Call this every frame FROM THE MAIN TD THREAD (Execute DAT onFrameStart).
    Captures the TOP, encodes to JPEG, and stores in the shared buffer.
    """
    global _frame_buffer, _last_frame_t, _last_error
    try:
        top = op(TOP_PATH)
        if top is None:
            _last_error = f'TOP not found: {TOP_PATH}'
            return

        arr = top.numpyArray(delayed=False)
        if arr is None or arr.size == 0:
            _last_error = f'TOP {TOP_PATH} returned empty array'
            return

        # Flip vertically (TD origin is bottom-left)
        arr = arr[::-1]
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)

        if arr.shape[2] == 4:
            arr = arr[:, :, :3]

        arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        _, jpg = cv2.imencode('.jpg', arr_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        jpg_bytes = jpg.tobytes()

        with _frame_lock:
            _frame_buffer = jpg_bytes
        _last_frame_t = time.time()
        _last_error = None

    except Exception as e:
        _last_error = str(e)
        print(f'[MoodWave] update_frame error: {e}')


class _MJPEGHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # silence per-request access logs

    def do_GET(self):
        if self.path == '/health':
            self._health()
        elif self.path.startswith('/video'):
            self._stream()
        else:
            self.send_response(404)
            self.end_headers()

    def _health(self):
        # Diagnostic so the user can tell *why* the stream isn't working.
        with _frame_lock:
            has_frame = _frame_buffer is not None
        age = time.time() - _last_frame_t if _last_frame_t else None

        if not has_frame:
            status = 'frames=none (update_frame never ran — wire onFrameStart)'
        elif age is not None and age > 1.0:
            status = f'frames=stale ({age:.1f}s old — onFrameStart not firing)'
        else:
            status = 'frames=ok'

        if _last_error:
            status += f' · last_error={_last_error}'

        body = f'OK · top={TOP_PATH} · {status}'.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream(self):
        self.send_response(200)
        self.send_header('Content-Type',  'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        interval = 1.0 / TARGET_FPS
        try:
            while True:
                with _frame_lock:
                    jpg = _frame_buffer
                if jpg:
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(jpg)}\r\n\r\n'.encode())
                    self.wfile.write(jpg)
                    self.wfile.write(b'\r\n')
                    self.wfile.flush()
                time.sleep(interval)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # client disconnected — normal


def start_server():
    global _server, _server_thread

    # Shut down any existing server first — also close the socket, otherwise
    # re-running this script leaks LISTENING sockets on the same port.
    if _server is not None:
        try:
            _server.shutdown()
            _server.server_close()
        except Exception:
            pass
        _server = None

    # Dual-stack bind so 'localhost' works whether it resolves to 127.0.0.1
    # or ::1 (Windows commonly does the latter).
    _server = _DualStackHTTPServer(('::', PORT), _MJPEGHandler)
    _server_thread = threading.Thread(target=_server.serve_forever, daemon=True)
    _server_thread.start()
    print(f'[MoodWave] MJPEG server running on http://localhost:{PORT}')
    print(f'[MoodWave]   /health  →  status check')
    print(f'[MoodWave]   /video   →  MJPEG stream from {TOP_PATH}')


def stop_server():
    global _server
    if _server:
        _server.shutdown()
        _server.server_close()
        _server = None
        print('[MoodWave] MJPEG server stopped.')


# Auto-start when this script is executed
start_server()

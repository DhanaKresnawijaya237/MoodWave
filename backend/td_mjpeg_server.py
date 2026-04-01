"""
MoodWave — TouchDesigner MJPEG Server
======================================
Serves the render output of a TOP as an MJPEG stream so the browser
can display it at http://localhost:9000/video.

HOW TO USE IN TOUCHDESIGNER
----------------------------
1. Create a Text DAT, paste this entire script into it.
2. Create an Execute DAT:
     - Enable the "Start" pulse
     - In the Start callback, add:  exec(op('text1').text)
       (replace 'text1' with the name of your Text DAT)
3. Set TOP_PATH below to the path of your output TOP (default: /project1/out1).
4. Save and re-open the project, or manually pulse the Execute DAT's Start.
5. You should see "[MoodWave] MJPEG server running on http://localhost:9000" in
   the Textport.

To restart the server at any time, run:  start_server()  in a Script DAT.
"""

import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import cv2

# ── Config ──────────────────────────────────────────────────────────────────
TOP_PATH     = '/project1/out1'  # full path to your output TOP inside TD
PORT         = 9000              # must match TD_FEED_URL port in moodwave.html
JPEG_QUALITY = 75                # 1–95: lower = smaller/faster, higher = sharper
TARGET_FPS   = 30
# ────────────────────────────────────────────────────────────────────────────

# Shared frame buffer — written by main TD thread, read by server thread
_frame_lock   = threading.Lock()
_frame_buffer = None             # bytes or None

_server        = None
_server_thread = None


def update_frame():
    """
    Call this every frame FROM THE MAIN TD THREAD (e.g. Execute DAT onFrameStart).
    Captures out1, encodes to JPEG, and stores in the shared buffer.
    """
    global _frame_buffer
    try:
        top = op(TOP_PATH)
        if top is None:
            return

        arr = top.numpyArray(delayed=False)
        if arr is None or arr.size == 0:
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

    except Exception as e:
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
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(b'OK')

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

    # Shut down any existing server first
    if _server is not None:
        try:
            _server.shutdown()
        except Exception:
            pass
        _server = None

    _server = HTTPServer(('', PORT), _MJPEGHandler)
    _server_thread = threading.Thread(target=_server.serve_forever, daemon=True)
    _server_thread.start()
    print(f'[MoodWave] MJPEG server running on http://localhost:{PORT}')
    print(f'[MoodWave]   /health  →  status check')
    print(f'[MoodWave]   /video   →  MJPEG stream from {TOP_PATH}')


def stop_server():
    global _server
    if _server:
        _server.shutdown()
        _server = None
        print('[MoodWave] MJPEG server stopped.')


# Auto-start when this script is executed
start_server()

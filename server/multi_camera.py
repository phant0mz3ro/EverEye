"""
Multi-camera security system — PI 3 BASELINE.

Capture frames from each camera and display them in a grid, plus a
Manage Cameras tab to add/remove cameras from the GUI. Still no
recording, motion detection, web/remote view, discovery scanning,
settings, recordings browser, or WiFi tab.

This exists as a known-good floor after a round of optimization
attempts (hardware-encode recording, live-view gating) that turned
out to cost more than they saved and made the app laggy under load.
Confirmed baseline (capture+display only) sits under 25% CPU across
cores on the actual Pi 3 — the plan is to add features back one at a
time from here, checking CPU after each one, so if something makes it
worse again it's obvious which change did it.

Re-add order:
    1. Manage Cameras tab (add/remove cameras from the GUI) — DONE
    2. Recording — AsyncSegmentWriter + FfmpegSegmentWriter (pipes to the
       ffmpeg CLI, downscaled, off the capture thread) — DONE. Records
       continuously (unconditionally) for now since motion detection
       isn't back yet; gating on motion happens once step 3 lands.
    2b. Recordings tab (browse/play/delete clips, in-app player via
        cv2.VideoCapture + Tkinter) — DONE, pulled forward out of the
        original order (was step 6) since there's no point recording
        without a way to see the clips yet.
    3. Motion detection (cheap OpenCV frame-diff, no mediapipe/dlib) — DONE.
       Downscaled grayscale diff, checked every few frames, gates what
       capture_worker submits to the recorder when recording_mode is set
       to motion. A hold window keeps one motion event in a single clip
       instead of chopping it up. Settings tab mode selector (originally
       step 6) pulled forward alongside this since the mode has to be
       switchable somewhere for motion detection to be testable at all.
    4. Web/remote MJPEG view
    5. USB + network discovery scanning
    6. WiFi Setup tab

INITIAL_CAMERAS below still seeds cameras at startup, but the Manage
Cameras tab is now the actual way to add/remove them at runtime.
"""

import math
import multiprocessing
import os
import queue as queue_module
import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2

import discover_cameras as discovery
from wifi_tab import WiFiTab

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # quiets OpenCV's own
# "can't open camera by index" warnings — expected noise when probing a composite
# device's non-capture nodes (metadata/control), harmless as long as the real
# capture index (confirmed working via discover_cameras.py) opens fine
import numpy as np
import requests
from PIL import Image, ImageTk
from flask import Flask, Response, render_template_string

# ---------- config ----------

INITIAL_CAMERAS = [
    # {"name": "front_door", "type": "http", "source": "http://192.168.1.42:81/stream"},
     {"name": "usb_cam", "type": "usb", "source": 0},
]

THUMB_W = 320
THUMB_H = 240
GUI_REFRESH_MS = 120  # a security grid doesn't need 30fps — this is plenty and cheap

RECORDINGS_DIR = "recordings"
RECORDING_SEGMENT_SEC = .10 * 60  # new file every 30 minutes
RECORD_WIDTH = 640  # pixel count drives encode cost more than fps does — cap it
RECORD_FPS = 10  # approximate — actual pull rate varies, this just sets playback speed metadata

RECORDING_MODE_CONTINUOUS = 0
RECORDING_MODE_MOTION = 1

# Motion detection stays deliberately dumb — grayscale frame-diff on a small
# downscaled copy, checked every few frames rather than every frame. A
# background-subtraction model (MOG2 etc.) would cost meaningfully more per
# frame for accuracy this app doesn't need.
MOTION_CHECK_EVERY_N_FRAMES = 3      # skip most frames — motion doesn't need per-frame granularity
MOTION_DOWNSCALE_WIDTH = 160         # diff on a tiny frame; cost scales with pixel count
MOTION_DIFF_THRESHOLD = 25           # per-pixel grayscale delta to count as "changed"
MOTION_MIN_CHANGED_RATIO = 0.1      # fraction of downscaled pixels that must change to call it motion
MOTION_RECORD_HOLD_SEC = 8           # keep recording this long after the last motion hit, so a clip
                                      # doesn't chop itself into fragments during one continuous event

# The GUI only redraws every GUI_REFRESH_MS and recording only wants ~RECORD_FPS,
# so a network camera pushing MJPEG faster than this is pure wasted decode CPU —
# cap it, especially with multiple network cameras running as separate processes.
NETWORK_STREAM_MAX_FPS = 15
NETWORK_STREAM_MIN_INTERVAL = 1.0 / NETWORK_STREAM_MAX_FPS

registry = {}  # camera_name -> {out_queue, capture_p, latest, latest_time}
registry_lock = threading.Lock()

web_app = Flask(__name__)

# Basic embedded web dashboard template
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Pi Security - Select View</title>
    <style>
        body { background-color: #121212; color: #fff; font-family: sans-serif; margin: 0; padding: 20px; display: flex; gap: 20px; }
        .sidebar { width: 220px; background: #1e1e1e; padding: 15px; border-radius: 8px; }
        .sidebar h2 { font-size: 18px; margin-top: 0; }
        .cam-btn { display: block; width: 100%; padding: 10px; margin-bottom: 8px; background: #2a2a2a; color: white; border: none; border-radius: 4px; cursor: pointer; text-align: left; }
        .cam-btn:hover { background: #3a3a3a; }
        .main-view { flex-grow: 1; background: #1e1e1e; padding: 15px; border-radius: 8px; text-align: center; }
        img { max-width: 100%; height: auto; border-radius: 4px; background: #000; }
    </style>
</head>
<body>
    <div class="sidebar">
        <h2>Cameras</h2>
        {% for name in cameras %}
            <button class="cam-btn" onclick="selectCamera('{{ name }}')">{{ name }}</button>
        {% endfor %}
    </div>
    <div class="main-view">
        <h2 id="cam-title">Select a camera</h2>
        <img id="feed" src="" width="640" height="480" style="display:none;" />
    </div>

    <script>
        function selectCamera(name) {
            document.getElementById('cam-title').innerText = name;
            const img = document.getElementById('feed');
            img.src = "/video_feed/" + name;
            img.style.display = "inline-block";
        }
    </script>
</body>
</html>
"""

def generate_mjpeg_stream(camera_name):
    """Yields JPEG-encoded camera frames as a multipart MJPEG stream."""
    while True:
        with registry_lock:
            entry = registry.get(camera_name)
            frame = entry["latest"] if entry else None

        if frame is not None:
            # Encode frame to JPEG
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        # Rate-limit to ~15 FPS to conserve Pi 3 CPU during multiple viewer connections
        time.sleep(1.0 / 15)

@web_app.route('/')
def index():
    with registry_lock:
        cam_names = list(registry.keys())
    return render_template_string(HTML_PAGE, cameras=cam_names)

@web_app.route('/video_feed/<camera_name>')
def video_feed(camera_name):
    return Response(generate_mjpeg_stream(camera_name),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def run_web_server():
    # Listens on all interfaces on port 5000
    web_app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


# ---------- frame sources ----------

def frame_generator(url):
    """Pulls an MJPEG-over-HTTP stream and yields decoded BGR frames."""
    last_yield = 0.0
    while True:
        try:
            # (connect_timeout, read_timeout) — a short read timeout means a
            # stalled/starved connection (e.g. two concurrent camera clients
            # more than the camera's own webserver can serve) gets detected
            # and reconnected within a few seconds instead of just trickling
            # and looking "frozen" on the last good frame for a long stretch.
            resp = requests.get(url, stream=True, timeout=(5, 5))
            buf = b""
            for chunk in resp.iter_content(chunk_size=4096):
                buf += chunk
                start = buf.find(b"\xff\xd8")
                end = buf.find(b"\xff\xd9")
                if start != -1 and end != -1 and end > start:
                    jpg = buf[start:end + 2]
                    buf = buf[end + 2:]
                    now = time.time()
                    if now - last_yield < NETWORK_STREAM_MIN_INTERVAL:
                        continue  # arrived faster than needed — skip the decode entirely
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        last_yield = now
                        yield frame
        except requests.RequestException as e:
            print(f"Stream error ({url}): {e} — retrying in 2s")
            time.sleep(2)


def usb_frame_generator(device_index: int):
    cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open USB camera at index {device_index}")

    # Set resolution BEFORE format — some UVC firmwares (composite
    # Jieli-chipset ones especially) negotiate format differently
    # depending on what resolution was already requested.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Try MJPEG first — most cheap UVC webcams need it forced explicitly
    # rather than relying on OpenCV's default format autodetection — but
    # fall back to YUYV if the hardware rejects MJPEG outright rather than
    # failing the whole camera over a format preference.
    fourcc_mjpg = cv2.VideoWriter_fourcc(*"MJPG")
    if not cap.set(cv2.CAP_PROP_FOURCC, fourcc_mjpg):
        print(f"USB camera {device_index}: MJPEG format rejected, trying YUYV fallback...", flush=True)
        fourcc_yuyv = cv2.VideoWriter_fourcc(*"YUYV")
        cap.set(cv2.CAP_PROP_FOURCC, fourcc_yuyv)

    # Same cameras also commonly fail their first few reads right after
    # open/format-set while still initializing — checks actual frame
    # content, not just the ret flag, since a "successful" read can still
    # come back as a blank/all-zero frame during that warm-up window.
    warm_ok = False
    for attempt in range(15):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0 and np.count_nonzero(frame) > 0:
            warm_ok = True
            break
        print(f"USB camera {device_index}: warm-up read {attempt + 1}/15 invalid, retrying...", flush=True)
        time.sleep(0.2)
    if not warm_ok:
        cap.release()
        raise RuntimeError(f"USB camera {device_index} failed to produce valid frames")

    # Once running, a lone failed read is a normal hiccup for this class
    # of camera, not a reason to kill the whole feed — only give up after
    # several consecutive failures in a row, and reset that count the
    # moment a good frame comes through again.
    consecutive_failures = 0
    max_consecutive_failures = 20

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(f"USB camera {device_index}: {consecutive_failures} reads failed — stopping", flush=True)
                    break
                time.sleep(0.05)
                continue
            consecutive_failures = 0
            yield frame
    finally:
        cap.release()


def push_latest(q, item):
    """Keep only the newest item in a maxsize=1 queue — never blocks the sender."""
    try:
        q.get_nowait()
    except queue_module.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue_module.Full:
        pass


# ---------- recording ----------
# FfmpegSegmentWriter pipes raw frames to a per-segment ffmpeg subprocess
# for encoding, rather than going through cv2.VideoWriter — this Pi's
# OpenCV build could not open a VideoWriter under any codec, and the
# ffmpeg CLI binary was confirmed to work independently, so recording
# goes through it directly instead.
#
# AsyncSegmentWriter runs it on its own background thread behind a
# 2-slot dropping queue so a slow encoder can never stall the capture
# loop — piping to ffmpeg's stdin must never happen in the capture loop
# itself (an earlier h264_v4l2m2m attempt did exactly that and stalled
# capture whenever the encoder fell behind). Dropping frames from a
# recording is fine; stalling capture/display is not.

class FfmpegSegmentWriter:
    """Pipes raw BGR frames to a per-segment ffmpeg subprocess, downscaled
    to RECORD_WIDTH. Only ever driven from AsyncSegmentWriter's background
    thread — see the module comment above for why."""

    def __init__(self, camera_name):
        self.camera_name = camera_name
        self.proc = None
        self.frame_size = None  # (w, h) actually being encoded for the current segment
        self.segment_start = None
        self.temp_path = None
        self.final_path = None

    def _open_new_segment(self, w, h):
        self._close_current()
        cam_dir = os.path.join(RECORDINGS_DIR, self.camera_name)
        os.makedirs(cam_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # The moov atom only gets finalized once ffmpeg exits after we
        # close its stdin — recording under a temp name and renaming only
        # after that means the Recordings tab (which globs for finished
        # files) never sees the in-progress one. Same "moov atom not
        # found" hazard as with cv2.VideoWriter if this weren't done.
        self.final_path = os.path.join(cam_dir, f"{timestamp}.mp4")
        self.temp_path = self.final_path + ".rec"
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", "bgr24",
            "-video_size", f"{w}x{h}", "-framerate", str(RECORD_FPS),
            "-i", "-",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-f", "mp4",  # explicit — the .rec temp suffix on the output path means
                          # ffmpeg can't guess the muxer from the extension the way it
                          # normally would, and errors out with "Unable to choose an
                          # output format" instead of just writing the file
            self.temp_path,
        ]
        try:
            # stderr captured (not DEVNULL) for now so a write failure can
            # report ffmpeg's actual error instead of just "broken pipe" —
            # same diagnosis trick that found the "-f mp4" muxer issue
            # earlier. stdout stays DEVNULL; with -loglevel error ffmpeg
            # writes nothing there, so it can't fill up and block.
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except OSError as e:
            print(f"[{self.camera_name}] Couldn't start ffmpeg: {e} — recording disabled for this camera", flush=True)
            self.proc = None
            self.temp_path = None
            self.final_path = None
            return
        self.frame_size = (w, h)
        self.segment_start = time.time()
        print(f"[{self.camera_name}] Recording new segment: {self.final_path}", flush=True)

    def _close_current(self):
        if self.proc is not None:
            try:
                self.proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(f"[{self.camera_name}] ffmpeg didn't exit in time — killing it", flush=True)
                self.proc.kill()
                self.proc.wait()
            self.proc = None
            if self.temp_path and os.path.exists(self.temp_path):
                try:
                    os.replace(self.temp_path, self.final_path)
                except OSError as e:
                    print(f"[{self.camera_name}] Could not finalize segment {self.temp_path}: {e}", flush=True)
            self.temp_path = None
            self.final_path = None

    def write(self, frame):
        h, w = frame.shape[:2]
        h, w = int(h), int(w)  # frame.shape gives numpy ints; keep this native for consistency
        if w > RECORD_WIDTH:
            scale = RECORD_WIDTH / w
            new_h = max(2, int(h * scale))
            if new_h % 2:
                new_h -= 1  # even dims for the encoder's chroma subsampling
            frame = cv2.resize(frame, (RECORD_WIDTH, new_h))
            h, w = frame.shape[:2]
            h, w = int(h), int(w)

        segment_expired = (
            self.segment_start is not None
            and (time.time() - self.segment_start) >= RECORDING_SEGMENT_SEC
        )
        if self.proc is None or self.frame_size != (w, h) or segment_expired:
            self._open_new_segment(w, h)
        if self.proc is None:
            return  # this segment failed to open — drop the frame

        try:
            self.proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError) as e:
            # The write failing means ffmpeg already exited on its own —
            # grab whatever it printed to stderr so the actual cause shows
            # up here instead of just "broken pipe".
            stderr_tail = ""
            try:
                self.proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                self.proc.wait(timeout=2)
                if self.proc.stderr is not None:
                    stderr_tail = self.proc.stderr.read().decode(errors="replace").strip()
            except subprocess.TimeoutExpired:
                pass
            print(f"[{self.camera_name}] ffmpeg pipe write failed: {e} — "
                  f"ffmpeg exit code {self.proc.returncode}, stderr: {stderr_tail!r} — "
                  f"will reopen on the next frame", flush=True)
            self.proc = None

    def close(self):
        self._close_current()


class AsyncSegmentWriter:
    """
    Background thread wrapper around FfmpegSegmentWriter. submit() is the
    only method the capture loop touches, and it never blocks: if the
    writer thread is behind, the oldest queued frame is dropped to make
    room for the newest one, same idea as push_latest() for the display
    queue.
    """

    def __init__(self, camera_name):
        self.queue = queue_module.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(camera_name,), daemon=True)
        self.thread.start()

    def submit(self, frame):
        try:
            self.queue.put_nowait(frame)
        except queue_module.Full:
            try:
                self.queue.get_nowait()  # drop the oldest queued frame
            except queue_module.Empty:
                pass
            try:
                self.queue.put_nowait(frame)
            except queue_module.Full:
                pass  # writer thread grabbed it between our get and put — fine, skip this frame

    def _run(self, camera_name):
        writer = FfmpegSegmentWriter(camera_name)
        while not self.stop_event.is_set():
            try:
                frame = self.queue.get(timeout=0.5)
            except queue_module.Empty:
                continue
            try:
                writer.write(frame)
            except Exception as e:
                print(f"[{camera_name}] Recording write failed: {e}", flush=True)
        writer.close()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
class MotionDetector:
    """
    Cheap per-camera motion check: downscale to MOTION_DOWNSCALE_WIDTH,
    grayscale, absdiff against the previous checked frame, threshold, and
    see how much of the frame changed. Only actually runs the diff every
    MOTION_CHECK_EVERY_N_FRAMES-th call — the rest are free no-ops — so
    the per-frame cost in the capture loop is close to zero most of the
    time. Lives entirely inside one camera's capture process; no shared
    state with anything else.
    """

    def __init__(self):
        self._prev_gray = None
        self._frame_count = 0

    def check(self, frame):
        """Returns True only on a frame where motion was actually detected.
        Most calls return False either because it wasn't this camera's turn
        to be checked yet, or because nothing changed enough — callers that
        need a "recording gate" should combine this with their own hold/
        cooldown window rather than treating a lone False as "no motion"."""
        self._frame_count += 1
        if self._frame_count % MOTION_CHECK_EVERY_N_FRAMES != 0:
            return False

        h, w = frame.shape[:2]
        scale = MOTION_DOWNSCALE_WIDTH / w
        small = cv2.resize(frame, (MOTION_DOWNSCALE_WIDTH, max(2, int(h * scale))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return False

        diff = cv2.absdiff(gray, self._prev_gray)
        self._prev_gray = gray
        changed = cv2.countNonZero(cv2.threshold(diff, MOTION_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)[1])
        return (changed / diff.size) >= MOTION_MIN_CHANGED_RATIO


# ---------- capture process ----------
# One process per camera: pull frames, build a thumbnail, record, nothing else.

def capture_worker(camera_name, camera_type, source, out_queue, record_enabled, recording_mode):
    if camera_type == "usb":
        source_desc = f"USB device {source}"
        frames = usb_frame_generator(source)
    else:
        source_desc = source
        frames = frame_generator(source)

    print(f"[{camera_name}] Connecting to {source_desc} ...", flush=True)
    recorder = AsyncSegmentWriter(camera_name)
    motion_detector = MotionDetector()
    motion_hold_until = 0.0  # monotonic-ish wall clock deadline; recording stays open until this passes
    try:
        for frame in frames:
            thumb = cv2.resize(frame, (THUMB_W, THUMB_H))
            push_latest(out_queue, thumb)
            # Checked here, not inside AsyncSegmentWriter, so a disabled
            # toggle skips the submit entirely — no point queuing/copying a
            # full-res frame that's about to be thrown away.
            if not record_enabled.value:
                continue
            if recording_mode.value == RECORDING_MODE_MOTION:
                if motion_detector.check(frame):
                    if time.time() >= motion_hold_until:
                        # only prints on the idle→active edge, not every
                        # hit — cheap way to see how often it's actually
                        # triggering without flooding the log
                        print(f"[{camera_name}] motion detected — recording for {MOTION_RECORD_HOLD_SEC}s", flush=True)
                    motion_hold_until = time.time() + MOTION_RECORD_HOLD_SEC
                if time.time() >= motion_hold_until:
                    continue  # no recent motion — don't submit this frame
            recorder.submit(frame)  # full-res frame — AsyncSegmentWriter.submit() downscales it
    except Exception as e:
        print(f"[{camera_name}] Stopped: {e}", flush=True)
    finally:
        recorder.stop()


# ---------- camera lifecycle ----------

def start_camera(name, cam_type, source, registry_, record_enabled, recording_mode):
    out_queue = multiprocessing.Queue(maxsize=1)
    capture_p = multiprocessing.Process(
        target=capture_worker,
        args=(name, cam_type, source, out_queue, record_enabled, recording_mode),
    )
    capture_p.start()

    with registry_lock:
        registry_[name] = {
            "out_queue": out_queue,
            "capture_p": capture_p,
            "latest": None,
            "latest_time": None,
        }


def stop_camera(name, registry_):
    with registry_lock:
        entry = registry_.pop(name, None)
    if entry is None:
        return
    entry["capture_p"].terminate()
    entry["capture_p"].join()


# ---------- desktop: scrollable tab wrapper ----------

class ScrollableFrame(tk.Frame):
    """
    A Canvas + inner Frame + Scrollbar, wrapped up so any tab builder can
    just pack widgets into `.body` like a normal Frame and get vertical
    scrolling for free once the content overflows the window.

    Tkinter has no built-in scrollable frame — this is the standard
    pattern: the Canvas is what actually scrolls, `body` is a Frame drawn
    inside it, and the scrollregion is recalculated whenever `body`
    resizes. Mousewheel scrolling is bound only while the pointer is over
    this particular canvas, so scrolling one tab doesn't also scroll
    whatever's behind an adjacent one.
    """

    def __init__(self, parent):
        super().__init__(parent)
        canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.body = tk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=self.body, anchor="nw")

        def on_body_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        self.body.bind("<Configure>", on_body_configure)

        def on_canvas_configure(event):
            # Keep the inner frame exactly as wide as the visible canvas
            # so content doesn't hang off the right edge or leave dead
            # space — only height should ever need to scroll here.
            canvas.itemconfig(window_id, width=event.width)
        canvas.bind("<Configure>", on_canvas_configure)

        def on_mousewheel(event):
            # Linux (Raspberry Pi OS / X11) sends Button-4/5; Windows and
            # macOS send <MouseWheel> with event.delta instead — handling
            # both keeps this working during dev on a non-Pi machine too.
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")
            else:
                canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        def bind_wheel(_event):
            canvas.bind_all("<Button-4>", on_mousewheel)
            canvas.bind_all("<Button-5>", on_mousewheel)
            canvas.bind_all("<MouseWheel>", on_mousewheel)

        def unbind_wheel(_event):
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", bind_wheel)
        canvas.bind("<Leave>", unbind_wheel)


# ---------- desktop: Manage Cameras tab ----------

def build_manage_tab(parent, registry_, record_enabled, recording_mode):
    listbox = tk.Listbox(parent, height=8, exportselection=False)
    listbox.pack(padx=16, pady=(16, 6), fill="x")

    def refresh_listbox():
        listbox.delete(0, "end")
        with registry_lock:
            names = list(registry_.keys())
        for name in names:
            listbox.insert("end", name)

    refresh_listbox()

    remove_btn = tk.Button(parent, text="Remove Selected")
    remove_btn.pack(padx=16, pady=(0, 16), anchor="w")

    # ---- Scan for Cameras ----
    tk.Label(parent, text="Scan for Cameras", font=("Arial", 11, "bold")).pack(anchor="w", padx=16, pady=(4, 6))

    scan_frame = tk.Frame(parent)
    scan_frame.pack(padx=16, pady=(0, 6), fill="x")

    scan_results = []  # parallel to scan_listbox rows; each entry is a discover_cameras() dict

    scan_listbox = tk.Listbox(scan_frame, height=6, exportselection=False)
    scan_listbox.pack(fill="x")

    scan_status = tk.Label(parent, text="", fg="gray")
    scan_status.pack(padx=16, anchor="w")

    def format_result(r):
        if r["protocol"] == "usb":
            return f"USB — {r['name']} (index {r['index']})"
        if r["protocol"] == "rtsp":
            return f"RTSP — {r['ip']}:{r['port']} (not yet supported by this app)"
        return f"Network (MJPEG) — {r['ip']}:{r['port']}{r['stream_path']}"

    def run_scan():
        scan_status.config(text="Scanning network and USB — this can take a few seconds...")
        scan_listbox.delete(0, "end")
        scan_results.clear()

        def worker():
            # Runs off the GUI thread — discover_cameras() does its own
            # /24 sweep with a thread pool, but that's still several
            # seconds blocking, and discover_usb_cameras() opens every
            # /dev/video* node in turn, so both would freeze the GUI if
            # called directly from a button callback.
            net_found = discovery.discover_cameras()
            usb_found = discovery.discover_usb_cameras()
            results = net_found + usb_found

            def apply():
                scan_results.extend(results)
                for r in results:
                    scan_listbox.insert("end", format_result(r))
                scan_status.config(
                    text=f"Found {len(results)} camera(s)." if results
                    else "No cameras found on the network or via USB."
                )

            parent.after(0, apply)

        threading.Thread(target=worker, daemon=True).start()

    def use_selected():
        sel = scan_listbox.curselection()
        if not sel:
            status_label.config(text="Select a scanned camera first.")
            return
        r = scan_results[sel[0]]
        if r["protocol"] == "rtsp":
            status_label.config(text="RTSP cameras aren't supported by this app yet — pick a different result.")
            return
        if r["protocol"] == "usb":
            type_var.set("usb")
            source_entry.delete(0, "end")
            source_entry.insert(0, str(r["index"]))
            name_entry.delete(0, "end")
            name_entry.insert(0, f"usb_{r['index']}")
        else:
            type_var.set("http")
            source_entry.delete(0, "end")
            source_entry.insert(0, f"http://{r['ip']}:{r['port']}{r['stream_path']}")
            name_entry.delete(0, "end")
            name_entry.insert(0, f"cam_{r['ip'].replace('.', '_')}")
        status_label.config(text="Filled in below — rename if you'd like, then click Add.")

    scan_btn_row = tk.Frame(parent)
    scan_btn_row.pack(padx=16, pady=(0, 16), anchor="w")
    tk.Button(scan_btn_row, text="Scan", command=run_scan).pack(side="left", padx=(0, 8))
    tk.Button(scan_btn_row, text="Use Selected", command=use_selected).pack(side="left")

    tk.Label(parent, text="Add Camera", font=("Arial", 11, "bold")).pack(anchor="w", padx=16, pady=(4, 6))

    form = tk.Frame(parent)
    form.pack(padx=16, pady=(0, 6), fill="x")

    tk.Label(form, text="Name:").grid(row=0, column=0, sticky="w", pady=2)
    name_entry = tk.Entry(form, width=30)
    name_entry.grid(row=0, column=1, sticky="w", pady=2)

    tk.Label(form, text="Type:").grid(row=1, column=0, sticky="w", pady=2)
    type_var = tk.StringVar(value="http")
    type_menu = tk.OptionMenu(form, type_var, "http", "usb")
    type_menu.grid(row=1, column=1, sticky="w", pady=2)

    tk.Label(form, text="Source:").grid(row=2, column=0, sticky="w", pady=2)
    source_entry = tk.Entry(form, width=40)
    source_entry.grid(row=2, column=1, sticky="w", pady=2)
    tk.Label(
        form, text="(http: full stream URL — e.g. http://192.168.1.42:81/stream)\n"
                    "(usb: device index — e.g. 0)",
        fg="gray", justify="left",
    ).grid(row=3, column=1, sticky="w")

    status_label = tk.Label(parent, text="", fg="red")
    status_label.pack(padx=16, anchor="w")

    def add_camera():
        name = name_entry.get().strip()
        cam_type = type_var.get()
        source = source_entry.get().strip()

        with registry_lock:
            exists = name in registry_

        if not name or not source:
            status_label.config(text="Name and source are both required.")
            return
        if exists:
            status_label.config(text=f"A camera named '{name}' already exists.")
            return
        if cam_type == "usb":
            try:
                source = int(source)
            except ValueError:
                status_label.config(text="USB source must be a device index number (e.g. 0).")
                return

        status_label.config(text="")
        start_camera(name, cam_type, source, registry_, record_enabled, recording_mode)
        refresh_listbox()
        name_entry.delete(0, "end")
        source_entry.delete(0, "end")

    tk.Button(parent, text="Add", command=add_camera).pack(padx=16, pady=(0, 16), anchor="w")

    def remove_camera():
        sel = listbox.curselection()
        if not sel:
            status_label.config(text="Select a camera to remove first.")
            return
        name = listbox.get(sel[0])
        stop_camera(name, registry_)
        status_label.config(text="")
        refresh_listbox()

    remove_btn.config(command=remove_camera)


# ---------- desktop: Settings tab ----------

def build_settings_tab(parent, record_enabled, recording_mode):
    """
    Global settings: the record on/off master switch, plus the recording
    mode (continuous vs motion-triggered) that gates what capture_worker
    actually submits to the recorder for each camera.
    """
    body = tk.Frame(parent)
    body.pack(fill="both", expand=True, padx=16, pady=16)

    tk.Label(body, text="Recording", font=("Arial", 11, "bold")).pack(anchor="w", pady=(0, 6))

    record_var = tk.BooleanVar(value=bool(record_enabled.value))

    def on_toggle():
        with record_enabled.get_lock():
            record_enabled.value = 1 if record_var.get() else 0

    tk.Checkbutton(
        body, text="Record cameras (applies to all cameras)",
        variable=record_var, command=on_toggle,
    ).pack(anchor="w")

    tk.Label(
        body,
        text="Turning this off stops new footage from being saved for every\n"
             "camera. Existing clips are untouched — use the Recordings tab\n"
             "to browse or delete them.",
        fg="gray", justify="left",
    ).pack(anchor="w", pady=(6, 0))

    tk.Label(body, text="Recording mode", font=("Arial", 11, "bold")).pack(anchor="w", pady=(16, 6))

    mode_var = tk.IntVar(value=recording_mode.value)

    def on_mode_change():
        with recording_mode.get_lock():
            recording_mode.value = mode_var.get()

    tk.Radiobutton(
        body, text="Continuous — record everything, all the time",
        variable=mode_var, value=RECORDING_MODE_CONTINUOUS, command=on_mode_change,
    ).pack(anchor="w")
    tk.Radiobutton(
        body, text="Motion-triggered — only record when something moves",
        variable=mode_var, value=RECORDING_MODE_MOTION, command=on_mode_change,
    ).pack(anchor="w")

    tk.Label(
        body,
        text="Motion-triggered keeps recording for a few seconds after the\n"
             "last motion, so one event stays in a single clip. Applies to\n"
             "every camera; this setting only matters when recording above\n"
             "is on.",
        fg="gray", justify="left",
    ).pack(anchor="w", pady=(6, 0))


# ---------- desktop: Recordings tab ----------

def _format_time(seconds):
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    return f"{m}:{s:02d}"


class RecordingPlayer:
    """
    Embedded clip player — lives inside the Recordings tab itself (no
    Toplevel popup) so it plays inline the same way it will on the final
    Pi + monitor device. Backed by cv2.VideoCapture + a Tkinter Label for
    frames, with a Scale widget as the seek bar.
    """

    def __init__(self, parent, get_delay_ms):
        self.get_delay_ms = get_delay_ms
        self.cap = None
        self.playing = False
        self.seeking = False
        self.total_frames = 0
        self.fps = RECORD_FPS
        self.after_id = None
        self.photo = None  # keep a reference so Tk doesn't garbage-collect it

        self.frame = tk.Frame(parent)

        self.video_label = tk.Label(self.frame, bg="black")
        self.video_label.pack(fill="both", expand=True)

        controls = tk.Frame(self.frame)
        controls.pack(fill="x", pady=(6, 0))

        self.play_btn = tk.Button(controls, text="Play", command=self.toggle_play, state="disabled")
        self.play_btn.pack(side="left", padx=(0, 8))

        self.time_label = tk.Label(controls, text="0:00 / 0:00", width=12)
        self.time_label.pack(side="left", padx=(0, 8))

        self.seek_var = tk.DoubleVar(value=0)
        self.seek_scale = ttk.Scale(
            controls, from_=0, to=1, orient="horizontal",
            variable=self.seek_var, command=self._on_seek_drag,
        )
        self.seek_scale.pack(side="left", fill="x", expand=True)
        self.seek_scale.state(["disabled"])
        self.seek_scale.bind("<ButtonPress-1>", self._on_seek_press)
        self.seek_scale.bind("<ButtonRelease-1>", self._on_seek_release)

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def _cancel_scheduled(self):
        if self.after_id is not None:
            try:
                self.frame.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None

    def load(self, filepath, status_cb=None):
        self.stop()
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            if status_cb:
                status_cb(f"Could not open {os.path.basename(filepath)}.")
            return
        self.cap = cap
        self.fps = cap.get(cv2.CAP_PROP_FPS) or RECORD_FPS
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        self.seek_scale.state(["!disabled"])
        self.seek_scale.configure(to=max(self.total_frames - 1, 1))
        self.seek_var.set(0)
        self.play_btn.config(text="Pause", state="normal")

        self.playing = True
        self._schedule()

    def _schedule(self):
        self._cancel_scheduled()
        if not self.frame.winfo_exists():
            return
        self.after_id = self.frame.after(self.get_delay_ms(self.fps), self._update)

    def _update(self):
        self.after_id = None
        if self.cap is None or not self.frame.winfo_exists():
            return
        if self.playing and not self.seeking:
            ret, frame = self.cap.read()
            if not ret:
                # end of clip — pause on the last frame, seek bar at the end
                self.playing = False
                self.play_btn.config(text="Play")
                if self.total_frames:
                    self.seek_var.set(self.total_frames - 1)
            else:
                self._show_frame(frame)
                if not self.seeking:
                    self.seek_var.set(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self._update_time_label()
        if self.playing:
            self._schedule()

    def _show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.video_label.configure(image=self.photo)

    def _update_time_label(self):
        if self.cap is None or not self.fps:
            return
        pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES) / self.fps
        total = (self.total_frames / self.fps) if self.total_frames else 0
        self.time_label.config(text=f"{_format_time(pos)} / {_format_time(total)}")

    def toggle_play(self):
        if self.cap is None:
            return
        self.playing = not self.playing
        self.play_btn.config(text="Pause" if self.playing else "Play")
        if self.playing:
            self._schedule()

    def _on_seek_press(self, event):
        self.seeking = True

    def _on_seek_drag(self, value):
        # live-update the time label while dragging, without pulling frames
        if self.cap is not None and self.fps:
            pos = float(value) / self.fps
            total = (self.total_frames / self.fps) if self.total_frames else 0
            self.time_label.config(text=f"{_format_time(pos)} / {_format_time(total)}")

    def _on_seek_release(self, event):
        if self.cap is not None:
            frame_no = int(self.seek_var.get())
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = self.cap.read()
            if ret:
                self._show_frame(frame)
                # cap.read() advances one frame past frame_no — step back
                # so playback resumes from where the user actually dropped it
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            self._update_time_label()
        self.seeking = False
        if self.playing:
            self._schedule()

    def stop(self):
        self._cancel_scheduled()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.playing = False
        self.seeking = False
        self.total_frames = 0
        self.photo = None
        self.video_label.configure(image="")
        self.play_btn.config(text="Play", state="disabled")
        self.seek_var.set(0)
        self.seek_scale.configure(to=1)
        self.seek_scale.state(["disabled"])
        self.time_label.config(text="0:00 / 0:00")


def build_recordings_tab(parent, root):
    os.makedirs(RECORDINGS_DIR, exist_ok=True)

    body = tk.Frame(parent)
    body.pack(fill="both", expand=True, padx=16, pady=16)
    body.rowconfigure(0, weight=1)
    body.columnconfigure(0, weight=1)
    body.columnconfigure(1, weight=1)

    # Player sits on top, in the tab itself — no separate window pops up
    # when a clip is played.
    player = RecordingPlayer(body, get_delay_ms=lambda fps: int(1000 / fps) if fps else int(1000 / RECORD_FPS))
    player.frame.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=(0, 16))

    tk.Label(body, text="Camera:").grid(row=1, column=0, sticky="w")
    tk.Label(body, text="Clips:").grid(row=1, column=1, sticky="w", padx=(16, 0))

    camera_listbox = tk.Listbox(body, height=12, width=24, exportselection=False)
    camera_listbox.grid(row=2, column=0, sticky="ns")

    clip_listbox = tk.Listbox(body, height=12, width=36, exportselection=False)
    clip_listbox.grid(row=2, column=1, sticky="ns", padx=(16, 0))

    status_label = tk.Label(parent, text="", fg="red")
    status_label.pack(padx=16, anchor="w")

    def selected_camera():
        sel = camera_listbox.curselection()
        return camera_listbox.get(sel[0]) if sel else None

    def clip_path(camera_name, filename):
        return os.path.join(RECORDINGS_DIR, camera_name, filename)

    def refresh_clips():
        clip_listbox.delete(0, "end")
        camera_name = selected_camera()
        if camera_name is None:
            return
        cam_dir = os.path.join(RECORDINGS_DIR, camera_name)
        try:
            files = sorted(os.listdir(cam_dir), reverse=True)
        except FileNotFoundError:
            files = []
        for f in files:
            if f.lower().endswith((".mp4", ".avi")):
                clip_listbox.insert("end", f)

    def refresh_cameras():
        prev = selected_camera()
        camera_listbox.delete(0, "end")
        try:
            names = sorted(d for d in os.listdir(RECORDINGS_DIR)
                            if os.path.isdir(os.path.join(RECORDINGS_DIR, d)))
        except FileNotFoundError:
            names = []
        for name in names:
            camera_listbox.insert("end", name)
        if prev in names:
            camera_listbox.selection_set(names.index(prev))
        refresh_clips()

    camera_listbox.bind("<<ListboxSelect>>", lambda e: refresh_clips())

    def play_selected():
        camera_name = selected_camera()
        sel = clip_listbox.curselection()
        if camera_name is None or not sel:
            status_label.config(text="Select a clip to play first.")
            return
        status_label.config(text="")
        player.load(clip_path(camera_name, clip_listbox.get(sel[0])), status_cb=status_label.config)

    # double-click a clip to play it, same as the Play button
    clip_listbox.bind("<Double-Button-1>", lambda e: play_selected())

    def delete_selected():
        camera_name = selected_camera()
        sel = clip_listbox.curselection()
        if camera_name is None or not sel:
            status_label.config(text="Select a clip to delete first.")
            return
        filename = clip_listbox.get(sel[0])
        try:
            os.remove(clip_path(camera_name, filename))
        except OSError as e:
            status_label.config(text=f"Could not delete: {e}")
            return
        status_label.config(text="")
        refresh_clips()

    btn_row = tk.Frame(parent)
    btn_row.pack(padx=16, pady=(0, 16), anchor="w")
    tk.Button(btn_row, text="Refresh", command=refresh_cameras).pack(side="left", padx=(0, 8))
    tk.Button(btn_row, text="Play", command=play_selected).pack(side="left", padx=(0, 8))
    tk.Button(btn_row, text="Delete", command=delete_selected).pack(side="left")

    refresh_cameras()


# ---------- main process ----------

def main():
    # "spawn" not "fork": each camera process gets a genuinely fresh Python/
    # OpenCV startup instead of inheriting a copy of the parent's memory —
    # cv2.VideoCapture().open() can deadlock inside a forked child if OpenCV
    # set up any internal thread state in the parent before the fork, since
    # the child inherits a copy of that state without the actual threads
    # that owned it. This is a well-known fork+OpenCV interaction, and
    # matches "hangs forever right at open(), works fine standalone."
    multiprocessing.set_start_method("spawn")

    # Shared across the main process and every camera subprocess — a plain
    # Python bool on the Settings tab wouldn't be visible to capture_worker,
    # which runs in its own process. "b" = signed char; get_lock() guards
    # the read-modify-write from the GUI thread against a torn write.
    record_enabled = multiprocessing.Value("b", 1)
    # Same sharing rationale as record_enabled — a plain int on the Settings
    # tab wouldn't be visible to capture_worker's own process.
    recording_mode = multiprocessing.Value("b", RECORDING_MODE_CONTINUOUS)

    for cam in INITIAL_CAMERAS:
        start_camera(cam["name"], cam["type"], cam["source"], registry, record_enabled, recording_mode)

    root = tk.Tk()
    root.attributes('-fullscreen', True)
    root.title("Security Cameras — Pi 3 baseline")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    camera_tab = tk.Frame(notebook, bg="black")
    manage_tab = ScrollableFrame(notebook)
    settings_tab = ScrollableFrame(notebook)
    recordings_tab = ScrollableFrame(notebook)
    notebook.add(camera_tab, text="Camera View")
    notebook.add(manage_tab, text="Manage Cameras")
    notebook.add(settings_tab, text="Settings")
    notebook.add(recordings_tab, text="Recordings")
    build_manage_tab(manage_tab.body, registry, record_enabled, recording_mode)
    build_settings_tab(settings_tab.body, record_enabled, recording_mode)
    build_recordings_tab(recordings_tab.body, root)

    wifi_scroll = ScrollableFrame(notebook)
    notebook.add(wifi_scroll, text="WiFi Setup")
    wifi_tab = WiFiTab(wifi_scroll.body)
    wifi_tab.pack(fill="both", expand=True)

    video_label = tk.Label(camera_tab, bg="black")
    video_label.pack(fill="both", expand=True)
    photo_holder = {"image": None}
    canvas_cache = {"canvas": None, "shape": None}
    zoomed_cam = {"name": None}
    grid_layout = {"cols": 1, "names": []}  # so clicks can map back to a camera

    def on_video_click(event):
        if zoomed_cam["name"] is not None:
            zoomed_cam["name"] = None  # click while zoomed = go back to grid
            return
        cols = grid_layout["cols"]
        names = grid_layout["names"]
        if not names:
            return
        col = event.x // THUMB_W
        row = event.y // THUMB_H
        idx = row * cols + col
        if 0 <= idx < len(names):
            zoomed_cam["name"] = names[idx]

    video_label.bind("<Button-1>", on_video_click)

    def update_frame():
        with registry_lock:
            names = list(registry.keys())
        n = len(names)
        cols = max(1, math.ceil(math.sqrt(n))) if n else 1
        rows = max(1, math.ceil(n / cols)) if n else 1
        grid_layout["cols"] = cols
        grid_layout["names"] = names

        for cam_name in names:
            try:
                latest = registry[cam_name]["out_queue"].get_nowait()
                with registry_lock:
                    registry[cam_name]["latest"] = latest
                    registry[cam_name]["latest_time"] = time.time()
            except queue_module.Empty:
                pass

        if zoomed_cam["name"] is not None and zoomed_cam["name"] in registry:
            with registry_lock:
                data = registry[zoomed_cam["name"]]["latest"]
            if data is not None:
                disp_w = video_label.winfo_width()
                disp_h = video_label.winfo_height()
                if disp_w > 1 and disp_h > 1:
                    canvas = cv2.resize(data, (disp_w, disp_h))
                else:
                    canvas = data
            else:
                canvas = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
                cv2.putText(canvas, f"{zoomed_cam['name']}: connecting...",
                            (20, THUMB_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        else:
            shape = (rows * THUMB_H, cols * THUMB_W, 3)
            if canvas_cache["canvas"] is None or canvas_cache["shape"] != shape:
                canvas_cache["canvas"] = np.zeros(shape, dtype=np.uint8)
                canvas_cache["shape"] = shape
            canvas = canvas_cache["canvas"]
            grid_layout["canvas_w"] = cols * THUMB_W
            grid_layout["canvas_h"] = rows * THUMB_H

            if n == 0:
                cv2.putText(canvas, "No cameras — edit INITIAL_CAMERAS in multi_camera.py",
                            (10, THUMB_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            for idx, cam_name in enumerate(names):
                row, col = divmod(idx, cols)
                y0, x0 = row * THUMB_H, col * THUMB_W
                with registry_lock:
                    data = registry[cam_name]["latest"]
                if data is not None:
                    canvas[y0:y0 + THUMB_H, x0:x0 + THUMB_W] = data
                else:
                    cv2.putText(canvas, f"{cam_name}: connecting...",
                                (x0 + 20, y0 + THUMB_H // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        photo_holder["image"] = photo
        video_label.configure(image=photo)

        root.after(GUI_REFRESH_MS, update_frame)

    def on_close():
        for name in list(registry.keys()):
            stop_camera(name, registry)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    update_frame()
    # Inside main() before root.mainloop()
    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()
    root.mainloop()


if __name__ == "__main__":
    main()
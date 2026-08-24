"""
Multi-camera security system — PI 3 BASELINE.

Deliberately stripped down to the minimum: capture frames from each
camera and display them in a grid on screen. Nothing else. No
recording, no motion detection, no web/remote view, no discovery
scanning, no settings, no recordings browser, no WiFi tab.

This exists as a known-good floor after a round of optimization
attempts (hardware-encode recording, live-view gating) that turned
out to cost more than they saved and made the app laggy under load.
Rather than keep patching a stack that's hard to reason about, the
plan is: confirm THIS is smooth and cheap on the actual Pi 3 first,
then add features back one at a time, checking CPU after each one —
so if something makes it worse again, it's obvious which change did
it, instead of guessing across a pile of simultaneous changes.

Planned re-add order (roughly cheapest/safest first):
    1. Manage Cameras tab (add/remove cameras from the GUI, not just
       the hardcoded list below)
    2. Recording — AsyncSegmentWriter + CV2SegmentWriter (mp4v,
       downscaled, off the capture thread) from the previous round,
       since that part was validated as correctly decoupled — just
       re-add it in isolation and confirm on its own
    3. Motion detection (cheap OpenCV frame-diff, no mediapipe/dlib)
    4. Web/remote MJPEG view
    5. USB + network discovery scanning
    6. Settings tab, Recordings tab, in-app player
    7. WiFi Setup tab

Edit INITIAL_CAMERAS below to point at your cameras — this baseline
has no add/remove UI yet, on purpose, to keep the surface area small
while establishing the floor.
"""

import math
import multiprocessing
import queue as queue_module
import threading
import time
import tkinter as tk

import cv2

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # quiets OpenCV's own
# "can't open camera by index" warnings — expected noise when probing a composite
# device's non-capture nodes (metadata/control), harmless as long as the real
# capture index (confirmed working via discover_cameras.py) opens fine
import numpy as np
import requests
from PIL import Image, ImageTk

# ---------- config ----------

INITIAL_CAMERAS = [
    # {"name": "front_door", "type": "http", "source": "http://192.168.1.42:81/stream"},
    {"name": "usb_cam", "type": "usb", "source": 0},
]

THUMB_W = 320
THUMB_H = 240
GUI_REFRESH_MS = 120  # a security grid doesn't need 30fps — this is plenty and cheap

registry = {}  # camera_name -> {out_queue, capture_p, latest, latest_time}
registry_lock = threading.Lock()


# ---------- frame sources ----------

def frame_generator(url):
    """Pulls an MJPEG-over-HTTP stream and yields decoded BGR frames."""
    while True:
        try:
            resp = requests.get(url, stream=True, timeout=10)
            buf = b""
            for chunk in resp.iter_content(chunk_size=4096):
                buf += chunk
                start = buf.find(b"\xff\xd8")
                end = buf.find(b"\xff\xd9")
                if start != -1 and end != -1 and end > start:
                    jpg = buf[start:end + 2]
                    buf = buf[end + 2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
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


# ---------- capture process ----------
# One process per camera: pull frames, build a thumbnail, nothing else.

def capture_worker(camera_name, camera_type, source, out_queue):
    if camera_type == "usb":
        source_desc = f"USB device {source}"
        frames = usb_frame_generator(source)
    else:
        source_desc = source
        frames = frame_generator(source)

    print(f"[{camera_name}] Connecting to {source_desc} ...", flush=True)
    try:
        for frame in frames:
            thumb = cv2.resize(frame, (THUMB_W, THUMB_H))
            push_latest(out_queue, thumb)
    except Exception as e:
        print(f"[{camera_name}] Stopped: {e}", flush=True)


# ---------- camera lifecycle ----------

def start_camera(name, cam_type, source, registry_):
    out_queue = multiprocessing.Queue(maxsize=1)
    capture_p = multiprocessing.Process(
        target=capture_worker,
        args=(name, cam_type, source, out_queue),
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

    for cam in INITIAL_CAMERAS:
        start_camera(cam["name"], cam["type"], cam["source"], registry)

    root = tk.Tk()
    root.title("Security Cameras — Pi 3 baseline")

    video_label = tk.Label(root, bg="black")
    video_label.pack(fill="both", expand=True)
    photo_holder = {"image": None}
    canvas_cache = {"canvas": None, "shape": None}

    def update_frame():
        with registry_lock:
            names = list(registry.keys())
        n = len(names)
        cols = max(1, math.ceil(math.sqrt(n))) if n else 1
        rows = max(1, math.ceil(n / cols)) if n else 1

        for cam_name in names:
            try:
                latest = registry[cam_name]["out_queue"].get_nowait()
                with registry_lock:
                    registry[cam_name]["latest"] = latest
                    registry[cam_name]["latest_time"] = time.time()
            except queue_module.Empty:
                pass

        shape = (rows * THUMB_H, cols * THUMB_W, 3)
        if canvas_cache["canvas"] is None or canvas_cache["shape"] != shape:
            canvas_cache["canvas"] = np.zeros(shape, dtype=np.uint8)
            canvas_cache["shape"] = shape
        canvas = canvas_cache["canvas"]

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
    root.mainloop()


if __name__ == "__main__":
    main()

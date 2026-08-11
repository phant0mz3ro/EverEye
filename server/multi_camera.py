"""
Multi-camera security system — PI 3 / 1GB RAM TIER.

This is a deliberately lighter branch of the main codebase, built for
Raspberry Pi 3 hardware. Cuts made relative to the full version:

  - NO face recognition (dlib/face_recognition dropped entirely — it's
    the single heaviest thing in the original stack, both to build and
    to run on a Cortex-A53). This tier is motion-alert only, no identity.
  - NO mediapipe face detection either — replaced with plain OpenCV
    frame-differencing motion detection (detect_motion, already existed
    in the original file, just wasn't wired to anything executable).
  - Motion detection folded directly into each camera's own capture
    process, using cheap OpenCV frame-differencing instead of mediapipe
    or dlib. The original file's "one detector process per camera" was
    actually dead code already (never started, never wired up) — this
    makes real what was aspirational, rather than adding a second
    process type back in. Net result: one process per camera, same as
    what was actually running before, just with working motion alerts.
  - NO WebRTC/aiortc for the web view — aiortc pulls in native codec
    deps that are painful to build on ARM and does its own software
    video encode in the background. Replaced with plain MJPEG streaming
    (motion JPEG over HTTP), which is what most cheap IP cameras use
    anyway and costs far less CPU.
  - Recording only writes while motion is active (not continuously),
    to cut disk I/O and encode load.
  - Slower GUI refresh and detection cadence — tuned for A53, not a
    desktop CPU.

Desktop UI: one Tkinter window, ttk.Notebook top nav with "Camera View",
"Manage Cameras", and "WiFi Setup" tabs. Video renders into a Tkinter
Label (not a separate cv2 window).

Clicking a camera tile in the grid goes fullscreen (full-resolution,
single camera); clicking anywhere while fullscreen returns to the grid.

Web app: FastAPI + MJPEG live view, reachable from other devices, runs
as a background thread inside this same process.
"""

import json
import math
import multiprocessing
import os
import queue as queue_module
import threading
import time
import tkinter as tk
import uuid
from tkinter import ttk, messagebox

import cv2
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageTk
from starlette.requests import Request

from wifi_tab import WiFiTab

# ===================
# Seed cameras — more can be added live from Manage (desktop tab or web page)
# "type": "http" for ESP32-CAM streams, "usb" for local webcams
# ===================
INITIAL_CAMERAS = [
    {"name": "front_door", "type": "http", "source": "http://<esp32-ip-1>/stream"},
    {"name": "usb_test",   "type": "usb",  "source": 0},
]

MOTION_THRESHOLD = 25
MOTION_MIN_AREA = 2000
MOTION_SCALE = 0.4              # motion diffing runs on a downscaled frame — cheap on A53
DETECTION_INTERVAL_SEC = 1.5    # motion check cadence — original was 0.5s, too tight for a Pi 3
MOTION_ACTIVE_HOLD_SEC = 8.0    # keep "motion active" (and recording) this long after last trigger

THUMB_W = 320
THUMB_H = 240

RECORDINGS_DIR = "recordings"
RECORDING_SEGMENT_SEC = 30 * 60  # new file every 30 minutes, while motion-active
RECORD_FPS = 8                    # fps for "motion" recording mode
RECORD_FPS_CONTINUOUS = 4         # fps for "continuous" mode — kept low, it's running all the time

GUI_REFRESH_MS = 120   # was 30ms (~33fps) — a security grid doesn't need that; ~8fps is plenty
STALE_AFTER_SEC = 5    # no new frame in this long = considered offline on the web camera list
WEB_HOST = "0.0.0.0"
WEB_PORT = 8000
MJPEG_QUALITY = 70     # JPEG quality for the web stream — lower = less CPU/bandwidth

registry = {}  # camera_name -> {out_queue, live_queue, capture_p, latest, latest_time, latest_full_frame, motion}
registry_lock = threading.Lock()

profiles = None


def generate_id():
    return uuid.uuid4().hex[:8]


# ---------- disk persistence ----------

PROFILES_FILE = "profiles.json"


def load_profiles_from_disk():
    if not os.path.exists(PROFILES_FILE):
        return {}
    with open(PROFILES_FILE, "r") as f:
        return json.load(f)


def save_profiles_to_disk(profiles_):
    with open(PROFILES_FILE, "w") as f:
        json.dump(dict(profiles_), f, indent=2)


SETTINGS_FILE = "settings.json"
DEFAULT_SETTINGS = {"recording_mode": "motion"}  # "motion" or "continuous"

_settings_cache = {"mtime": None, "data": dict(DEFAULT_SETTINGS)}


def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        return dict(DEFAULT_SETTINGS)
    with open(SETTINGS_FILE, "r") as f:
        data = json.load(f)
    return {**DEFAULT_SETTINGS, **data}


def save_settings(settings_):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings_, f, indent=2)


def get_recording_mode():
    """
    Re-reads settings.json only when its mtime changes — cheap enough to
    call every frame from a capture process, and picks up a change made
    in the Settings tab within a second or two without restarting cameras.
    """
    try:
        mtime = os.path.getmtime(SETTINGS_FILE)
    except OSError:
        return DEFAULT_SETTINGS["recording_mode"]
    if mtime != _settings_cache["mtime"]:
        _settings_cache["mtime"] = mtime
        _settings_cache["data"] = load_settings()
    return _settings_cache["data"].get("recording_mode", DEFAULT_SETTINGS["recording_mode"])


# ---------- frame sources ----------

def frame_generator(url: str):
    stream = requests.get(url, stream=True, timeout=10)
    buffer = b""
    for chunk in stream.iter_content(chunk_size=1024):
        buffer += chunk
        start = buffer.find(b"\xff\xd8")
        end = buffer.find(b"\xff\xd9")
        if start != -1 and end != -1 and end > start:
            jpg_bytes = buffer[start:end + 2]
            buffer = buffer[end + 2:]
            frame = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                yield frame


def usb_frame_generator(device_index: int):
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open USB camera at index {device_index}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"USB camera {device_index} read failed — stopping")
                break
            yield frame
    finally:
        cap.release()


def detect_motion(prev_gray, curr_gray):
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return any(cv2.contourArea(c) >= MOTION_MIN_AREA for c in contours)


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


def open_new_segment(camera_name, frame_w, frame_h, fps):
    cam_dir = os.path.join(RECORDINGS_DIR, camera_name)
    os.makedirs(cam_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(cam_dir, f"{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filepath, fourcc, fps, (frame_w, frame_h))
    print(f"[{camera_name}] Recording new segment: {filepath}")
    return writer


# ---------- capture process ----------
# One process per camera: pulls frames, does cheap motion-diff (no
# mediapipe/dlib), records only while motion is active, builds the
# thumbnail for the desktop grid. This is the ENTIRE per-camera process
# now — there is no separate detector process on this tier.

def capture_worker(camera_name, camera_type, source, out_queue, live_queue):
    if camera_type == "usb":
        source_desc = f"USB device {source}"
        frames = usb_frame_generator(source)
    else:
        source_desc = source
        frames = frame_generator(source)

    writer = None
    writer_fps = None
    segment_start_time = 0.0
    prev_gray = None
    last_motion_check = 0.0
    last_motion_time = 0.0  # last time motion was actually seen

    print(f"[{camera_name}] Connecting to {source_desc} ...")
    try:
        for frame in frames:
            push_latest(live_queue, frame)  # full-res, for on-demand live view — cheap even with no viewer

            h, w = frame.shape[:2]
            now = time.time()

            # cheap motion check — downscaled grayscale diff, not every frame
            if now - last_motion_check >= DETECTION_INTERVAL_SEC:
                last_motion_check = now
                small = cv2.resize(frame, None, fx=MOTION_SCALE, fy=MOTION_SCALE)
                gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (21, 21), 0)
                if prev_gray is not None and detect_motion(prev_gray, gray):
                    last_motion_time = now
                prev_gray = gray

            motion_active = (now - last_motion_time) < MOTION_ACTIVE_HOLD_SEC
            mode = get_recording_mode()
            should_record = motion_active if mode == "motion" else True
            fps = RECORD_FPS if mode == "motion" else RECORD_FPS_CONTINUOUS

            # "motion" mode: record only while motion is (recently) active —
            # cuts disk I/O and encode load to a fraction of continuous.
            # "continuous" mode: always recording, at a lower fps to keep
            # the encode load manageable on a Pi 3.
            if should_record:
                if writer is None or (now - segment_start_time) >= RECORDING_SEGMENT_SEC or writer_fps != fps:
                    if writer is not None:
                        writer.release()
                    writer = open_new_segment(camera_name, w, h, fps)
                    writer_fps = fps
                    segment_start_time = now
                writer.write(frame)
            elif writer is not None:
                writer.release()
                writer = None

            thumb = cv2.resize(frame, (THUMB_W, THUMB_H))
            if motion_active:
                cv2.rectangle(thumb, (0, 0), (THUMB_W - 1, THUMB_H - 1), (0, 0, 255), 4)
                cv2.putText(thumb, "MOTION", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            push_latest(out_queue, {"frame": thumb, "motion": motion_active})
    except Exception as e:
        print(f"[{camera_name}] Stopped: {e}")
    finally:
        if writer is not None:
            writer.release()


# ---------- camera lifecycle (add/remove live, from desktop OR web) ----------

def start_camera(name, cam_type, source, registry_):
    out_queue = multiprocessing.Queue(maxsize=1)
    live_queue = multiprocessing.Queue(maxsize=1)

    capture_p = multiprocessing.Process(
        target=capture_worker,
        args=(name, cam_type, source, out_queue, live_queue),
    )
    capture_p.start()

    with registry_lock:
        registry_[name] = {
            "out_queue": out_queue,
            "live_queue": live_queue,
            "capture_p": capture_p,
            "latest": None,
            "latest_time": None,
            "latest_full_frame": None,
            "motion": False,
        }


def stop_camera(name, registry_):
    with registry_lock:
        entry = registry_.pop(name, None)
    if entry is None:
        return
    entry["capture_p"].terminate()
    entry["capture_p"].join()


# ---------- web app (FastAPI + MJPEG) ----------

web_app = FastAPI()
templates = Jinja2Templates(directory="live_view/templates")


def _mjpeg_frame_bytes(camera_name):
    """Generator yielding one multipart-JPEG chunk per frame for a camera's stream."""
    boundary = b"--frame"
    while True:
        with registry_lock:
            entry = registry.get(camera_name)
            frame = entry["latest_full_frame"] if entry else None
        if frame is None:
            time.sleep(0.1)
            continue
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY])
        if not ok:
            time.sleep(0.1)
            continue
        yield (
            boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
            + str(len(jpg)).encode() + b"\r\n\r\n" + jpg.tobytes() + b"\r\n"
        )
        time.sleep(1.0 / 12)  # ~12fps cap on the web stream — plenty for a live view, cheap to encode


@web_app.get("/")
async def index(request: Request):
    with registry_lock:
        names = list(registry.keys())
        camera_list = []
        for name in names:
            latest_time = registry[name].get("latest_time")
            online = latest_time is not None and (time.time() - latest_time) < STALE_AFTER_SEC
            camera_list.append({"name": name, "online": online})
    online_count = sum(1 for c in camera_list if c["online"])
    return templates.TemplateResponse(request, "index.html", {
        "cameras": camera_list,
        "online_count": online_count,
        "total_count": len(camera_list),
        "active_nav": "cameras",
    })


@web_app.get("/camera/{name}")
async def camera_page(request: Request, name: str):
    return templates.TemplateResponse(request, "viewer.html", {"camera_name": name, "active_nav": "cameras"})


@web_app.get("/stream/{name}")
async def stream_camera(name: str):
    return StreamingResponse(
        _mjpeg_frame_bytes(name), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@web_app.get("/manage")
async def manage_page(request: Request):
    with registry_lock:
        names = list(registry.keys())
    return templates.TemplateResponse(request, "manage.html", {"cameras": names, "error": None, "active_nav": "manage"})


@web_app.post("/cameras/add")
async def add_camera_route(
    request: Request,
    name: str = Form(...),
    cam_type: str = Form(..., alias="type"),
    source: str = Form(...),
):
    with registry_lock:
        exists = name in registry

    error = None
    if not name or not source:
        error = "Name and source are required."
    elif exists:
        error = f"A camera named '{name}' already exists."
    elif cam_type == "usb":
        try:
            source = int(source)
        except ValueError:
            error = "USB source must be a device index number (e.g. 0)."

    if error is not None:
        with registry_lock:
            names = list(registry.keys())
        return templates.TemplateResponse(
            request, "manage.html", {"cameras": names, "error": error, "active_nav": "manage"}, status_code=400
        )

    start_camera(name, cam_type, source, registry)
    return RedirectResponse(url="/manage", status_code=303)


@web_app.post("/cameras/remove")
async def remove_camera_route(name: str = Form(...)):
    stop_camera(name, registry)
    return RedirectResponse(url="/manage", status_code=303)


def run_web_server():
    uvicorn.run(web_app, host=WEB_HOST, port=WEB_PORT, log_level="warning")


# ---------- desktop: Manage Cameras tab ----------

def build_manage_tab(parent, registry_, profiles_):
    tk.Label(parent, text="Active Cameras", font=("Arial", 11, "bold")).pack(anchor="w", padx=16, pady=(16, 6))
    listbox = tk.Listbox(parent, height=8, exportselection=False)
    listbox.pack(padx=16, pady=(0, 10), fill="x")

    last_shown_names = {"names": None}

    def refresh_listbox():
        with registry_lock:
            names = list(registry_.keys())

        if names != last_shown_names["names"]:
            selected_name = None
            sel = listbox.curselection()
            if sel:
                selected_name = listbox.get(sel[0])

            listbox.delete(0, "end")
            for cam_name in names:
                listbox.insert("end", cam_name)

            if selected_name in names:
                listbox.selection_set(names.index(selected_name))

            last_shown_names["names"] = names

        parent.after(1000, refresh_listbox)

    def remove_selected():
        selection = listbox.curselection()
        if not selection:
            return
        cam_name = listbox.get(selection[0])
        stop_camera(cam_name, registry_)

    tk.Button(parent, text="Remove Selected", command=remove_selected).pack(padx=16, pady=(0, 20), anchor="w")

    tk.Label(parent, text="Add Camera", font=("Arial", 11, "bold")).pack(anchor="w", padx=16, pady=(0, 6))

    form = tk.Frame(parent)
    form.pack(padx=16, anchor="w")

    tk.Label(form, text="Name:").grid(row=0, column=0, sticky="w", pady=4)
    name_entry = tk.Entry(form, width=25)
    name_entry.grid(row=0, column=1, pady=4)

    tk.Label(form, text="Type:").grid(row=1, column=0, sticky="w", pady=4)
    type_var = tk.StringVar(value="http")
    tk.OptionMenu(form, type_var, "http", "usb").grid(row=1, column=1, sticky="w", pady=4)

    tk.Label(form, text="Source:").grid(row=2, column=0, sticky="w", pady=4)
    source_entry = tk.Entry(form, width=25)
    source_entry.grid(row=2, column=1, pady=4)

    status_label = tk.Label(parent, text="", fg="red")
    status_label.pack(padx=16, anchor="w")

    def submit():
        name = name_entry.get().strip()
        cam_type = type_var.get()
        source_raw = source_entry.get().strip()

        with registry_lock:
            exists = name in registry_

        if not name or not source_raw:
            status_label.config(text="Fill in all fields")
            return
        if exists:
            status_label.config(text="A camera with this name already exists")
            return

        source = source_raw
        if cam_type == "usb":
            try:
                source = int(source_raw)
            except ValueError:
                status_label.config(text="USB source must be a device index number (e.g. 0)")
                return

        start_camera(name, cam_type, source, registry_)
        status_label.config(text="")
        name_entry.delete(0, "end")
        source_entry.delete(0, "end")

    tk.Button(parent, text="Add Camera", command=submit).pack(padx=16, pady=10, anchor="w")


    # --- Camera Discovery ---
    import threading
    from discover_cameras import discover_cameras, discover_usb_cameras

    tk.Label(parent, text="Discover Cameras", font=("Arial", 11, "bold")).pack(anchor="w", padx=16, pady=(20, 6))

    discovery_frame = tk.Frame(parent)
    discovery_frame.pack(padx=16, pady=(0, 6), fill="x")

    scan_status_label = tk.Label(discovery_frame, text="", fg="gray")
    scan_status_label.pack(anchor="w")

    discovery_listbox = tk.Listbox(parent, height=5, exportselection=False)
    discovery_listbox.pack(padx=16, pady=(0, 6), fill="x")

    found_cameras = {"results": []}  # index -> camera dict, kept in sync with listbox rows

    def run_scan():
        scan_btn.config(state="disabled", text="Scanning...")
        usb_scan_btn.config(state="disabled")
        scan_status_label.config(text="Scanning local network, this can take a few seconds...")
        discovery_listbox.delete(0, "end")
        found_cameras["results"] = []

        def worker():
            results = discover_cameras()

            def update_ui():
                discovery_listbox.delete(0, "end")
                found_cameras["results"] = results
                if not results:
                    scan_status_label.config(text="No network cameras found.")
                else:
                    scan_status_label.config(text=f"Found {len(results)} camera(s). Select one, then click 'Use Selected'.")
                    for cam in results:
                        label = f"{cam['protocol'].upper()}  {cam['ip']}:{cam['port']}{cam['stream_path'] or ''}"
                        discovery_listbox.insert("end", label)
                scan_btn.config(state="normal", text="Scan Network")
                usb_scan_btn.config(state="normal")

            parent.after(0, update_ui)

        threading.Thread(target=worker, daemon=True).start()

    def run_usb_scan():
        scan_btn.config(state="disabled")
        usb_scan_btn.config(state="disabled", text="Scanning...")
        scan_status_label.config(text="Checking /dev/video* devices...")
        discovery_listbox.delete(0, "end")
        found_cameras["results"] = []

        def worker():
            results = discover_usb_cameras()

            def update_ui():
                discovery_listbox.delete(0, "end")
                found_cameras["results"] = results
                if not results:
                    scan_status_label.config(text="No USB cameras found.")
                else:
                    scan_status_label.config(text=f"Found {len(results)} USB camera(s). Select one, then click 'Use Selected'.")
                    for cam in results:
                        label = f"USB  {cam['name']} (index {cam['index']})"
                        discovery_listbox.insert("end", label)
                scan_btn.config(state="normal")
                usb_scan_btn.config(state="normal", text="Scan USB")

            parent.after(0, update_ui)

        threading.Thread(target=worker, daemon=True).start()

    def use_selected_discovery():
        selection = discovery_listbox.curselection()
        if not selection:
            scan_status_label.config(text="Select a camera from the list first.")
            return
        cam = found_cameras["results"][selection[0]]

        if cam["protocol"] == "usb":
            type_var.set("usb")
            source_entry.delete(0, "end")
            source_entry.insert(0, str(cam["index"]))
        elif cam["protocol"] == "mjpeg":
            type_var.set("http")
            source_entry.delete(0, "end")
            source_entry.insert(0, f"http://{cam['ip']}:{cam['port']}{cam['stream_path']}")
        else:  # rtsp
            type_var.set("http")  # adjust if/when a dedicated rtsp type exists in start_camera
            source_entry.delete(0, "end")
            source_entry.insert(0, f"rtsp://{cam['ip']}:{cam['port']}/")
            scan_status_label.config(text="RTSP path guessed — verify/adjust before adding, paths vary by camera.")

        name_entry.focus_set()  # cursor to Name field since that's the only thing left to fill in

    btn_row = tk.Frame(discovery_frame)
    btn_row.pack(anchor="w", pady=(4, 0))
    scan_btn = tk.Button(btn_row, text="Scan Network", command=run_scan)
    scan_btn.pack(side="left", padx=(0, 8))
    usb_scan_btn = tk.Button(btn_row, text="Scan USB", command=run_usb_scan)
    usb_scan_btn.pack(side="left", padx=(0, 8))
    tk.Button(btn_row, text="Use Selected", command=use_selected_discovery).pack(side="left")

    refresh_listbox()


# ---------- desktop: Settings tab ----------

def build_settings_tab(parent):
    tk.Label(parent, text="Recording Mode", font=("Arial", 11, "bold")).pack(anchor="w", padx=16, pady=(16, 6))

    current = load_settings().get("recording_mode", "motion")
    mode_var = tk.StringVar(value=current)

    tk.Radiobutton(
        parent, text="Motion-triggered (default) — records only while motion is detected, cheapest on a Pi 3",
        variable=mode_var, value="motion", justify="left",
    ).pack(anchor="w", padx=16, pady=2)
    tk.Radiobutton(
        parent, text=f"Continuous — always recording at {RECORD_FPS_CONTINUOUS}fps, more disk use and CPU",
        variable=mode_var, value="continuous", justify="left",
    ).pack(anchor="w", padx=16, pady=2)

    status_label = tk.Label(parent, text="", fg="green")
    status_label.pack(anchor="w", padx=16, pady=(10, 0))

    def apply_settings():
        settings_ = load_settings()
        settings_["recording_mode"] = mode_var.get()
        save_settings(settings_)
        status_label.config(text="Saved — cameras pick this up within a couple seconds, no restart needed.")

    tk.Button(parent, text="Save", command=apply_settings).pack(anchor="w", padx=16, pady=10)


# ---------- desktop: in-app recording playback ----------

def open_recording_player(root, filepath):
    """
    Plays a recorded .mp4 in its own Toplevel using cv2.VideoCapture +
    a Tkinter Label — no external player, no OS file-association
    dependency, since the final device runs on a plain connected
    monitor with just this app.
    """
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        messagebox.showerror("Playback error", f"Couldn't open:\n{filepath}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or RECORD_FPS
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_delay_ms = max(1, int(1000 / fps))

    win = tk.Toplevel(root)
    win.title(os.path.basename(filepath))

    video_label = tk.Label(win, bg="black")
    video_label.pack(fill="both", expand=True)
    photo_holder = {"image": None}

    controls = tk.Frame(win)
    controls.pack(fill="x", padx=8, pady=6)

    state = {"playing": True, "seeking": False, "after_id": None}

    play_btn = tk.Button(controls, text="Pause")
    play_btn.pack(side="left", padx=(0, 8))

    time_label = tk.Label(controls, text="0:00 / 0:00", width=12)
    time_label.pack(side="right")

    def frame_to_time(frame_idx):
        secs = frame_idx / fps if fps else 0
        return f"{int(secs // 60)}:{int(secs % 60):02d}"

    seek_var = tk.DoubleVar(value=0)
    seek_scale = tk.Scale(
        controls, from_=0, to=max(frame_count - 1, 0), orient="horizontal",
        variable=seek_var, showvalue=False,
    )
    seek_scale.pack(side="left", fill="x", expand=True, padx=8)

    def on_seek_press(event):
        state["seeking"] = True

    def on_seek_release(event):
        cap.set(cv2.CAP_PROP_POS_FRAMES, seek_var.get())
        state["seeking"] = False

    seek_scale.bind("<Button-1>", on_seek_press)
    seek_scale.bind("<ButtonRelease-1>", on_seek_release)

    def toggle_play():
        state["playing"] = not state["playing"]
        play_btn.config(text="Pause" if state["playing"] else "Play")

    play_btn.config(command=toggle_play)

    def show_frame():
        if state["seeking"]:
            win.after(frame_delay_ms, show_frame)
            return

        if state["playing"]:
            ok, frame = cap.read()
            if not ok:
                # end of clip — pause on last frame rather than closing
                state["playing"] = False
                play_btn.config(text="Play")
                win.after(frame_delay_ms, show_frame)
                return

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            photo_holder["image"] = photo
            video_label.configure(image=photo)

            current_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            seek_var.set(current_idx)
            time_label.config(text=f"{frame_to_time(current_idx)} / {frame_to_time(frame_count)}")

        state["after_id"] = win.after(frame_delay_ms, show_frame)

    def on_close():
        if state["after_id"] is not None:
            win.after_cancel(state["after_id"])
        cap.release()
        win.destroy()

    win.protocol("WM_DELETE_WINDOW", on_close)
    show_frame()


# ---------- desktop: Recordings tab ----------


def build_recordings_tab(parent):
    list_frame = tk.Frame(parent)
    list_frame.pack(fill="both", expand=True, padx=16, pady=16)

    left = tk.Frame(list_frame)
    left.pack(side="left", fill="y", padx=(0, 16))
    tk.Label(left, text="Cameras").pack(anchor="w")
    camera_listbox = tk.Listbox(left, height=15, width=20, exportselection=False)
    camera_listbox.pack(fill="y")

    right = tk.Frame(list_frame)
    right.pack(side="left", fill="both", expand=True)
    tk.Label(right, text="Recordings").pack(anchor="w")
    file_listbox = tk.Listbox(right, height=15, exportselection=False)
    file_listbox.pack(fill="both", expand=True)

    status_label = tk.Label(parent, text="", fg="red")
    status_label.pack(padx=16, anchor="w")

    state = {"files": []}  # index -> full path, kept in sync with file_listbox rows

    def refresh_cameras():
        camera_listbox.delete(0, "end")
        if not os.path.isdir(RECORDINGS_DIR):
            return
        for cam_name in sorted(os.listdir(RECORDINGS_DIR)):
            if os.path.isdir(os.path.join(RECORDINGS_DIR, cam_name)):
                camera_listbox.insert("end", cam_name)

    def refresh_files(event=None):
        file_listbox.delete(0, "end")
        state["files"] = []
        sel = camera_listbox.curselection()
        if not sel:
            return
        cam_name = camera_listbox.get(sel[0])
        cam_dir = os.path.join(RECORDINGS_DIR, cam_name)
        if not os.path.isdir(cam_dir):
            return
        for filename in sorted(os.listdir(cam_dir), reverse=True):
            if not filename.endswith(".mp4"):
                continue
            full_path = os.path.join(cam_dir, filename)
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            file_listbox.insert("end", f"{filename}  ({size_mb:.1f} MB)")
            state["files"].append(full_path)

    camera_listbox.bind("<<ListboxSelect>>", refresh_files)

    def play_selected():
        sel = file_listbox.curselection()
        if not sel:
            status_label.config(text="Select a recording first.")
            return
        full_path = state["files"][sel[0]]
        open_recording_player(parent.winfo_toplevel(), full_path)

    def delete_selected():
        sel = file_listbox.curselection()
        if not sel:
            status_label.config(text="Select a recording first.")
            return
        full_path = state["files"][sel[0]]
        try:
            os.remove(full_path)
            status_label.config(text="")
            refresh_files()
        except OSError as e:
            status_label.config(text=f"Couldn't delete: {e}")

    btn_row = tk.Frame(parent)
    btn_row.pack(padx=16, pady=(0, 10), anchor="w")
    tk.Button(btn_row, text="Play", command=play_selected).pack(side="left", padx=(0, 8))
    tk.Button(btn_row, text="Delete", command=delete_selected).pack(side="left", padx=(0, 8))
    tk.Button(btn_row, text="Refresh", command=refresh_cameras).pack(side="left")

    refresh_cameras()


# ---------- main process ----------

def main():
    global profiles

    multiprocessing.set_start_method("fork")

    profiles = load_profiles_from_disk()
    if not os.path.exists(SETTINGS_FILE):
        save_settings(dict(DEFAULT_SETTINGS))

    for cam in INITIAL_CAMERAS:
        start_camera(cam["name"], cam["type"], cam["source"], registry)

    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()

    # ---- single unified Tkinter window ----
    root = tk.Tk()
    root.title("Security Cameras")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    camera_tab = tk.Frame(notebook, bg="black")
    manage_tab = tk.Frame(notebook)
    settings_tab = tk.Frame(notebook)
    recordings_tab = tk.Frame(notebook)
    notebook.add(camera_tab, text="Camera View")
    notebook.add(manage_tab, text="Manage Cameras")
    notebook.add(recordings_tab, text="Recordings")
    notebook.add(settings_tab, text="Settings")
    notebook.add(WiFiTab(notebook), text="WiFi Setup")

    build_manage_tab(manage_tab, registry, profiles)
    build_recordings_tab(recordings_tab)
    build_settings_tab(settings_tab)

    # Video grid (or fullscreen single camera) renders into this Label as a
    # PIL/ImageTk photo — instead of a separate cv2.imshow window — this is
    # what lets it live inside the same window as the tabs.
    video_label = tk.Label(camera_tab, bg="black")
    video_label.pack(fill="both", expand=True)

    photo_holder = {"image": None}  # must keep a reference — Tkinter drops PhotoImages with no live reference
    canvas_cache = {"canvas": None, "shape": None}
    image_dims = {"w": None, "h": None}  # actual rendered image size — needed to undo tk.Label's auto-centering
    fullscreen_camera = None  # None = grid view; a camera name = fullscreen that camera

    def widget_to_image_coords(event):
        """
        tk.Label centers its image when the widget is larger than the image
        (which is almost always true once the window's resized/maximized) —
        event.x/event.y are relative to the WIDGET, not the image itself.
        Without this translation, clicks land increasingly off-target the
        bigger the window gets, which is exactly the fullscreen bug.
        """
        img_w, img_h = image_dims["w"], image_dims["h"]
        if img_w is None:
            return event.x, event.y
        offset_x = max(0, (video_label.winfo_width() - img_w) // 2)
        offset_y = max(0, (video_label.winfo_height() - img_h) // 2)
        return event.x - offset_x, event.y - offset_y

    def exit_fullscreen(event=None):
        nonlocal fullscreen_camera
        fullscreen_camera = None

    root.bind("<Escape>", exit_fullscreen)

    def on_click(event):
        nonlocal fullscreen_camera

        if fullscreen_camera is not None:
            fullscreen_camera = None
            return

        with registry_lock:
            names = list(registry.keys())
        if not names:
            return

        img_x, img_y = widget_to_image_coords(event)
        if img_x < 0 or img_y < 0:
            return  # click landed in the Label's padding area, not on the image itself

        cols = max(1, math.ceil(math.sqrt(len(names))))
        col, row = img_x // THUMB_W, img_y // THUMB_H
        idx = row * cols + col
        if idx >= len(names):
            return
        cam_name = names[idx]
        fullscreen_camera = cam_name

    video_label.bind("<Button-1>", on_click)

    def update_frame():
        nonlocal fullscreen_camera

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
                    registry[cam_name]["motion"] = latest.get("motion", False)
            except queue_module.Empty:
                pass

            # Full-res frame is only pulled when something actually needs
            # it — the desktop fullscreen view, or the web MJPEG stream
            # (which reads latest_full_frame directly from another thread,
            # so we keep it updated any time a camera exists, not just on
            # fullscreen — the web stream has no equivalent gate on a Pi 3
            # build since there's no per-viewer negotiation like WebRTC had).
            try:
                full_frame = registry[cam_name]["live_queue"].get_nowait()
                with registry_lock:
                    registry[cam_name]["latest_full_frame"] = full_frame
            except queue_module.Empty:
                pass

        if fullscreen_camera is not None and fullscreen_camera in names:
            with registry_lock:
                frame = registry[fullscreen_camera].get("latest_full_frame")
            display_frame = frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            shape = (rows * THUMB_H, cols * THUMB_W, 3)
            if canvas_cache["canvas"] is None or canvas_cache["shape"] != shape:
                canvas_cache["canvas"] = np.zeros(shape, dtype=np.uint8)
                canvas_cache["shape"] = shape
            canvas = canvas_cache["canvas"]

            if n == 0:
                cv2.putText(canvas, "No cameras — use Manage Cameras to add one",
                            (10, THUMB_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            for idx, cam_name in enumerate(names):
                row, col = divmod(idx, cols)
                y0, x0 = row * THUMB_H, col * THUMB_W
                with registry_lock:
                    data = registry[cam_name]["latest"]
                if data is not None:
                    canvas[y0:y0 + THUMB_H, x0:x0 + THUMB_W] = data["frame"]
                else:
                    cv2.putText(canvas, f"{cam_name}: connecting...",
                                (x0 + 20, y0 + THUMB_H // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            display_frame = canvas

        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        image_dims["h"], image_dims["w"] = rgb.shape[:2]
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        photo_holder["image"] = photo
        video_label.configure(image=photo)

        root.after(GUI_REFRESH_MS, update_frame)

    update_frame()

    def on_close():
        for cam_name in list(registry.keys()):
            stop_camera(cam_name, registry)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()

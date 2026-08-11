"""
Multi-camera security system.

Architecture per camera: TWO processes — capture_worker (lightweight,
pulls frames, records, builds thumbnails) and detector_worker (CPU-heavy
motion+face detection/recognition, isolated on its own core so it never
stalls capture or display).

Desktop UI: one Tkinter window, ttk.Notebook top nav with "Camera View"
and "Manage Cameras" tabs. Video renders into a Tkinter Label (not a
separate cv2 window) so everything lives in one app instance.

Clicking a camera tile in the grid goes fullscreen (full-resolution,
single camera); clicking anywhere while fullscreen returns to the grid.
Clicking a face box (grid or nowhere-near-fullscreen) still opens the
known/unknown recognition popups as before.

Web app: FastAPI + WebRTC (aiortc) live view, reachable from other
devices, with its own camera list / per-camera viewer / manage pages —
runs as a background thread inside this same process.

Known faces / profiles are shared across all detector processes via
multiprocessing.Manager, and across desktop + web via a shared registry.
"""

import json
import math
import multiprocessing
import os
import pickle
import queue as queue_module
import threading
import time
import tkinter as tk
import uuid
from concurrent.futures import ThreadPoolExecutor
from tkinter import ttk

import cv2
#import face_recognition
import mediapipe as mp
import numpy as np
import requests
import uvicorn
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
from fastapi import FastAPI, Form, WebSocket
from fastapi.responses import HTMLResponse, RedirectResponse
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

FACE_SAVE_DIR = "detected_faces"
KNOWN_FACES_DIR = "known_faces"
SAVE_COOLDOWN_SEC = 2.0
CROP_PADDING = 0.3

MOTION_THRESHOLD = 25
MOTION_MIN_AREA = 2000

ENCODINGS_FILE = "encodings.pkl"
PROFILES_FILE = "profiles.json"
RECOGNITION_TOLERANCE = 0.6

BOX_PERSIST_SEC = 4.0
DETECTION_INTERVAL_SEC = 0.5  # detector won't re-run more often than this
PROCESS_SCALE = 0.5           # detection/encoding runs on a half-size frame

THUMB_W = 320
THUMB_H = 240

RECORDINGS_DIR = "recordings"
RECORDING_SEGMENT_SEC = 30 * 60  # new file every 30 minutes
RECORD_FPS = 10  # approximate — actual pull rate varies, this just sets playback speed metadata

RTC_CONFIG = RTCConfiguration(iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")])
STALE_AFTER_SEC = 5  # no new frame in this long = considered offline on the web camera list
WEB_HOST = "0.0.0.0"
WEB_PORT = 8000

# How many active WebRTC viewers each camera currently has. The render
# loop only drains the (expensive, full-resolution) live_queue for a
# camera when this is > 0 OR the desktop app has it fullscreen — otherwise
# that unpickling cost was happening every frame for every camera
# regardless of whether anyone was actually watching.
viewer_counts = {}
viewer_counts_lock = threading.Lock()

registry = {}  # camera_name -> {out_queue, live_queue, stop_event, detector_p, capture_p, latest, latest_time, latest_full_frame}
registry_lock = threading.Lock()

# Same idea as registry — set once by main(), read by the web routes and
# desktop UI so camera add/remove works from anywhere, not just main()'s
# own local scope.
known_encodings = None
known_ids = None
profiles = None
manager = None


def generate_id():
    return uuid.uuid4().hex[:8]


# ---------- disk persistence ----------

def load_known_faces_from_disk():
    if not os.path.exists(ENCODINGS_FILE):
        print(f"No {ENCODINGS_FILE} found — run encode_known_faces.py first, "
              f"or enroll people live via the Unknown popup.")
        return [], []
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["ids"]


def save_known_faces_to_disk(known_encodings_, known_ids_):
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": list(known_encodings_), "ids": list(known_ids_)}, f)


def load_profiles_from_disk():
    if not os.path.exists(PROFILES_FILE):
        return {}
    with open(PROFILES_FILE, "r") as f:
        return json.load(f)


def save_profiles_to_disk(profiles_):
    with open(PROFILES_FILE, "w") as f:
        json.dump(dict(profiles_), f, indent=2)


def get_reference_photo(person_id):
    person_dir = os.path.join(KNOWN_FACES_DIR, person_id)
    if not os.path.isdir(person_dir):
        return None
    for filename in sorted(os.listdir(person_dir)):
        return os.path.join(person_dir, filename)
    return None


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


def open_new_segment(camera_name, frame_w, frame_h):
    cam_dir = os.path.join(RECORDINGS_DIR, camera_name)
    os.makedirs(cam_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(cam_dir, f"{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filepath, fourcc, RECORD_FPS, (frame_w, frame_h))
    print(f"[{camera_name}] Recording new segment: {filepath}")
    return writer


# ---------- detector process ----------
"""
def detector_worker(camera_name, known_encodings_, known_ids_, profiles_, frame_queue, boxes_queue, stop_event):
    os.makedirs(FACE_SAVE_DIR, exist_ok=True)
    save_executor = ThreadPoolExecutor(max_workers=1)

    mp_face_detection = mp.solutions.face_detection
    prev_gray = None
    last_detection_time = 0.0

    with mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.6,
    ) as detector:
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.2)
            except queue_module.Empty:
                continue

            now = time.time()
            if now - last_detection_time < DETECTION_INTERVAL_SEC:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)
            motion = False
            if prev_gray is not None:
                motion = detect_motion(prev_gray, gray_blurred)
            prev_gray = gray_blurred

            if not motion:
                continue

            last_detection_time = now

            small = cv2.resize(frame, None, fx=PROCESS_SCALE, fy=PROCESS_SCALE)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb_small)
            sh, sw = small.shape[:2]
            h, w = frame.shape[:2]
            new_boxes = []

            local_encodings = list(known_encodings_)
            local_ids = list(known_ids_)
            local_profiles = dict(profiles_)

            if results.detections:
                for detection in results.detections:
                    box = detection.location_data.relative_bounding_box
                    x1 = max(int(box.xmin * w), 0)
                    y1 = max(int(box.ymin * h), 0)
                    bw = int(box.width * w)
                    bh = int(box.height * h)
                    x2, y2 = min(x1 + bw, w), min(y1 + bh, h)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    sx1 = max(int(box.xmin * sw), 0)
                    sy1 = max(int(box.ymin * sh), 0)
                    sbw = int(box.width * sw)
                    sbh = int(box.height * sh)
                    sx2, sy2 = min(sx1 + sbw, sw), min(sy1 + sbh, sh)

                    face_location = [(sy1, sx2, sy2, sx1)]
                    person_id = None
                    display_name = "Unknown"
                    confidence = None

                    
                    encodings = face_recognition.face_encodings(
                        rgb_small, known_face_locations=face_location)

                    

                    if encodings and local_encodings:
                        distances = face_recognition.face_distance(local_encodings, encodings[0])
                        best_match_idx = int(np.argmin(distances))
                        best_distance = distances[best_match_idx]
                        if best_distance <= RECOGNITION_TOLERANCE:
                            person_id = local_ids[best_match_idx]
                            display_name = local_profiles.get(person_id, {}).get("name", person_id)
                            confidence = round(max(0.0, 1.0 - best_distance) * 100)
                    
                    pad_x = int(bw * CROP_PADDING)
                    pad_y = int(bh * CROP_PADDING)
                    cx1 = max(x1 - pad_x, 0)
                    cy1 = max(y1 - pad_y, 0)
                    cx2 = min(x2 + pad_x, w)
                    cy2 = min(y2 + pad_y, h)
                    crop = frame[cy1:cy2, cx1:cx2].copy()

                    new_boxes.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "id": person_id, "name": display_name, "confidence": confidence,
                        "crop": crop,
                    })

                    safe_name = display_name.replace(" ", "_")
                    id_tag = person_id or "unknown"
                    filename = os.path.join(
                        FACE_SAVE_DIR, f"{camera_name}_{safe_name}_{id_tag}_{int(now * 1000)}.jpg"
                    )
                    save_executor.submit(cv2.imwrite, filename, crop)
                    print(f"[{camera_name}] Saving: {filename}")

            if new_boxes:
                push_latest(boxes_queue, new_boxes)

    save_executor.shutdown(wait=False)
"""

# ---------- capture process ----------

def capture_worker(camera_name, camera_type, source, frame_queue, boxes_queue, out_queue, live_queue):
    if camera_type == "usb":
        source_desc = f"USB device {source}"
        frames = usb_frame_generator(source)
    else:
        source_desc = source
        frames = frame_generator(source)

    last_boxes = []
    last_boxes_time = 0.0
    writer = None
    segment_start_time = 0.0

    print(f"[{camera_name}] Connecting to {source_desc} ...")
    try:
        for frame in frames:
            push_latest(frame_queue, frame)
            push_latest(live_queue, frame)  # full-res, for on-demand live view — cheap even with no viewer

            h, w = frame.shape[:2]

            now = time.time()
            if writer is None or (now - segment_start_time) >= RECORDING_SEGMENT_SEC:
                if writer is not None:
                    writer.release()
                writer = open_new_segment(camera_name, w, h)
                segment_start_time = now
            writer.write(frame)

            try:
                last_boxes = boxes_queue.get_nowait()
                last_boxes_time = time.time()
            except queue_module.Empty:
                pass

            scale_x, scale_y = THUMB_W / w, THUMB_H / h
            thumb = cv2.resize(frame, (THUMB_W, THUMB_H))

            thumb_boxes = []
            if last_boxes and (time.time() - last_boxes_time) < BOX_PERSIST_SEC:
                for b in last_boxes:
                    tx1, ty1 = int(b["x1"] * scale_x), int(b["y1"] * scale_y)
                    tx2, ty2 = int(b["x2"] * scale_x), int(b["y2"] * scale_y)
                    color = (0, 255, 0) if b["id"] is not None else (0, 0, 255)
                    label = b["name"]
                    if b["confidence"] is not None:
                        label = f"{label} {b['confidence']}%"
                    cv2.rectangle(thumb, (tx1, ty1), (tx2, ty2), color, 2)
                    cv2.putText(thumb, label, (tx1, max(ty1 - 8, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                    thumb_boxes.append({
                        "x1": tx1, "y1": ty1, "x2": tx2, "y2": ty2,
                        "id": b["id"], "name": b["name"], "confidence": b["confidence"],
                        "crop": b["crop"],
                    })

            push_latest(out_queue, {"frame": thumb, "boxes": thumb_boxes})
    except Exception as e:
        print(f"[{camera_name}] Stopped: {e}")
    finally:
        if writer is not None:
            writer.release()


# ---------- popup UI (desktop, main process only) ----------

def open_known_popup(root, box, profiles_, lock):
    person_id = box["id"]
    profile = dict(profiles_.get(person_id, {"name": box["name"], "age": ""}))

    win = tk.Toplevel(root)
    win.title(profile.get("name", person_id))
    win.resizable(True, True)

    ref_path = get_reference_photo(person_id)
    if ref_path and os.path.exists(ref_path):
        img = Image.open(ref_path)
        img.thumbnail((200, 200))
        photo = ImageTk.PhotoImage(img)
        img_label = tk.Label(win, image=photo)
        img_label.image = photo
        img_label.pack(padx=10, pady=10)

    tk.Label(win, text=f"ID: {person_id}", fg="gray").pack()
    if box["confidence"] is not None:
        tk.Label(win, text=f"Match confidence: {box['confidence']}%", fg="gray").pack(pady=(0, 10))

    fields_frame = tk.Frame(win)
    fields_frame.pack(padx=10, pady=5, fill="both", expand=True)

    field_rows = []

    def add_row(key, value):
        row = tk.Frame(fields_frame)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=f"{key}:", width=10, anchor="w").pack(side="left")
        entry = tk.Entry(row)
        entry.insert(0, value)
        entry.pack(side="left", fill="x", expand=True)
        field_rows.append((key, entry))

    add_row("name", profile.get("name", ""))
    add_row("age", profile.get("age", ""))
    for key, value in profile.items():
        if key not in ("name", "age"):
            add_row(key, value)

    add_field_frame = tk.Frame(win)
    add_field_frame.pack(padx=10, pady=(5, 0), fill="x")
    tk.Label(add_field_frame, text="New field:").pack(side="left")
    new_key_entry = tk.Entry(add_field_frame, width=10)
    new_key_entry.pack(side="left", padx=(5, 2))
    new_val_entry = tk.Entry(add_field_frame, width=12)
    new_val_entry.pack(side="left", padx=2)

    def add_new_field():
        key = new_key_entry.get().strip()
        if not key:
            return
        add_row(key, new_val_entry.get().strip())
        new_key_entry.delete(0, "end")
        new_val_entry.delete(0, "end")

    tk.Button(add_field_frame, text="+", command=add_new_field).pack(side="left", padx=2)

    def save_profile():
        updated = {key: entry.get().strip() for key, entry in field_rows}
        with lock:
            profiles_[person_id] = updated
            save_profiles_to_disk(profiles_)
        win.destroy()

    tk.Button(win, text="Save", command=save_profile).pack(pady=10)

"""
def open_unknown_popup(root, box, known_encodings_, known_ids_, profiles_, lock):
    win = tk.Toplevel(root)
    win.title("Unknown — Enroll")
    win.resizable(True, True)

    crop_rgb = cv2.cvtColor(box["crop"], cv2.COLOR_BGR2RGB)
    img = Image.fromarray(crop_rgb)
    img.thumbnail((200, 200))
    photo = ImageTk.PhotoImage(img)
    img_label = tk.Label(win, image=photo)
    img_label.image = photo
    img_label.pack(padx=10, pady=10)

    tk.Label(win, text="Name:").pack(anchor="w", padx=10)
    name_entry = tk.Entry(win, width=30)
    name_entry.pack(padx=10, pady=(0, 5))

    tk.Label(win, text="Age:").pack(anchor="w", padx=10)
    age_entry = tk.Entry(win, width=30)
    age_entry.pack(padx=10, pady=(0, 10))

    status_label = tk.Label(win, text="", fg="red")
    status_label.pack()

    def enroll():
        new_name = name_entry.get().strip()
        if not new_name:
            status_label.config(text="Enter a name first")
            return

        encodings = face_recognition.face_encodings(crop_rgb)
        if not encodings:
            status_label.config(text="Couldn't encode this face — try a clearer capture")
            return

        person_id = generate_id()
        person_dir = os.path.join(KNOWN_FACES_DIR, person_id)
        os.makedirs(person_dir, exist_ok=True)
        photo_path = os.path.join(person_dir, f"{int(time.time())}.jpg")
        cv2.imwrite(photo_path, box["crop"])

        with lock:
            known_encodings_.append(encodings[0])
            known_ids_.append(person_id)
            save_known_faces_to_disk(known_encodings_, known_ids_)

            profiles_[person_id] = {"name": new_name, "age": age_entry.get().strip()}
            save_profiles_to_disk(profiles_)

        print(f"Enrolled new person: {new_name} (id={person_id}, visible to all cameras)")
        win.destroy()

    tk.Button(win, text="Enroll as known", command=enroll).pack(pady=(0, 10))

"""
# ---------- camera lifecycle (add/remove live, from desktop OR web) ----------

def start_camera(name, cam_type, source, known_encodings_, known_ids_, profiles_, registry_, manager_):
    frame_queue = multiprocessing.Queue(maxsize=1)
    boxes_queue = multiprocessing.Queue(maxsize=1)
    out_queue = multiprocessing.Queue(maxsize=1)
    live_queue = multiprocessing.Queue(maxsize=1)
    stop_event = manager_.Event()  # Manager-backed — avoids the raw-semaphore rebuild issue on spawn/fork edge cases

    """detector_p = multiprocessing.Process(
        target=detector_worker,
        args=(name, known_encodings_, known_ids_, profiles_, frame_queue, boxes_queue, stop_event),
    )"""
    capture_p = multiprocessing.Process(
        target=capture_worker,
        args=(name, cam_type, source, frame_queue, boxes_queue, out_queue, live_queue),
    )
    #detector_p.start()
    capture_p.start()

    with registry_lock:
        registry_[name] = {
            "out_queue": out_queue,
            "live_queue": live_queue,
            "stop_event": stop_event,
            #"detector_p": detector_p,
            "capture_p": capture_p,
            "latest": None,
            "latest_time": None,
            "latest_full_frame": None,
        }


def stop_camera(name, registry_):
    with registry_lock:
        entry = registry_.pop(name, None)
    if entry is None:
        return
    entry["stop_event"].set()
    #entry["detector_p"].terminate()
    entry["capture_p"].terminate()
    #entry["detector_p"].join()
    entry["capture_p"].join()


# ---------- web app (FastAPI + WebRTC) ----------

web_app = FastAPI()
templates = Jinja2Templates(directory="live_view/templates")
active_connections = set()


class CameraStreamTrack(VideoStreamTrack):
    def __init__(self, camera_name):
        super().__init__()
        self._camera_name = camera_name

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        with registry_lock:
            entry = registry.get(self._camera_name)
            frame = entry["latest_full_frame"] if entry else None

        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame


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

    start_camera(name, cam_type, source, known_encodings, known_ids, profiles, registry, manager)
    return RedirectResponse(url="/manage", status_code=303)


@web_app.post("/cameras/remove")
async def remove_camera_route(name: str = Form(...)):
    stop_camera(name, registry)
    return RedirectResponse(url="/manage", status_code=303)


@web_app.websocket("/ws/{name}")
async def signaling(websocket: WebSocket, name: str):
    await websocket.accept()
    pc = RTCPeerConnection(configuration=RTC_CONFIG)
    active_connections.add(pc)
    pc.addTrack(CameraStreamTrack(name))

    with viewer_counts_lock:
        viewer_counts[name] = viewer_counts.get(name, 0) + 1

    @pc.on("connectionstatechange")
    async def on_state_change():
        print(f"Connection state: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await pc.close()
            active_connections.discard(pc)

    try:
        raw = await websocket.receive_text()
        message = json.loads(raw)
        offer = RTCSessionDescription(sdp=message["sdp"], type=message["type"])
        await pc.setRemoteDescription(offer)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        await websocket.send_text(json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }))

        while True:
            await websocket.receive_text()
    except Exception as e:
        print(f"[{name}] Signaling closed: {e}")
    finally:
        await pc.close()
        active_connections.discard(pc)
        with viewer_counts_lock:
            viewer_counts[name] = max(0, viewer_counts.get(name, 1) - 1)


def run_web_server():
    uvicorn.run(web_app, host=WEB_HOST, port=WEB_PORT, log_level="warning")


# ---------- desktop: Manage Cameras tab ----------

def build_manage_tab(parent, registry_, known_encodings_, known_ids_, profiles_, manager_):
    tk.Label(parent, text="Active Cameras", font=("Arial", 11, "bold")).pack(anchor="w", padx=16, pady=(16, 6))
    listbox = tk.Listbox(parent, height=8)
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

        start_camera(name, cam_type, source, known_encodings_, known_ids_, profiles_, registry_, manager_)
        status_label.config(text="")
        name_entry.delete(0, "end")
        source_entry.delete(0, "end")

    tk.Button(parent, text="Add Camera", command=submit).pack(padx=16, pady=10, anchor="w")


    # --- Camera Discovery ---
    import threading
    from discover_cameras import discover_cameras

    tk.Label(parent, text="Discover Cameras", font=("Arial", 11, "bold")).pack(anchor="w", padx=16, pady=(20, 6))

    discovery_frame = tk.Frame(parent)
    discovery_frame.pack(padx=16, pady=(0, 6), fill="x")

    scan_status_label = tk.Label(discovery_frame, text="", fg="gray")
    scan_status_label.pack(anchor="w")

    discovery_listbox = tk.Listbox(parent, height=5)
    discovery_listbox.pack(padx=16, pady=(0, 6), fill="x")

    found_cameras = {"results": []}  # index -> camera dict, kept in sync with listbox rows

    def run_scan():
        scan_btn.config(state="disabled", text="Scanning...")
        scan_status_label.config(text="Scanning local network, this can take a few seconds...")
        discovery_listbox.delete(0, "end")
        found_cameras["results"] = []

        def worker():
            results = discover_cameras()

            def update_ui():
                discovery_listbox.delete(0, "end")
                found_cameras["results"] = results
                if not results:
                    scan_status_label.config(text="No cameras found on the network.")
                else:
                    scan_status_label.config(text=f"Found {len(results)} camera(s). Select one, then click 'Use Selected'.")
                    for cam in results:
                        label = f"{cam['protocol'].upper()}  {cam['ip']}:{cam['port']}{cam['stream_path'] or ''}"
                        discovery_listbox.insert("end", label)
                scan_btn.config(state="normal", text="Scan for Cameras")

            parent.after(0, update_ui)

        threading.Thread(target=worker, daemon=True).start()

    def use_selected_discovery():
        selection = discovery_listbox.curselection()
        if not selection:
            scan_status_label.config(text="Select a camera from the list first.")
            return
        cam = found_cameras["results"][selection[0]]

        if cam["protocol"] == "mjpeg":
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
    scan_btn = tk.Button(btn_row, text="Scan for Cameras", command=run_scan)
    scan_btn.pack(side="left", padx=(0, 8))
    tk.Button(btn_row, text="Use Selected", command=use_selected_discovery).pack(side="left")

    refresh_listbox()


# ---------- main process ----------

def main():
    global known_encodings, known_ids, profiles, manager

    multiprocessing.set_start_method("fork")

    manager = multiprocessing.Manager()
    initial_encodings, initial_ids = load_known_faces_from_disk()
    initial_profiles = load_profiles_from_disk()

    known_encodings = manager.list(initial_encodings)
    known_ids = manager.list(initial_ids)
    profiles = manager.dict(initial_profiles)
    lock = manager.Lock()

    for cam in INITIAL_CAMERAS:
        start_camera(cam["name"], cam["type"], cam["source"],
                     known_encodings, known_ids, profiles, registry, manager)

    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()

    # ---- single unified Tkinter window ----
    root = tk.Tk()
    root.title("Security Cameras")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    camera_tab = tk.Frame(notebook, bg="black")
    manage_tab = tk.Frame(notebook)
    notebook.add(camera_tab, text="Camera View")
    notebook.add(manage_tab, text="Manage Cameras")
    notebook.add(WiFiTab(notebook), text="WiFi Setup")

    build_manage_tab(manage_tab, registry, known_encodings, known_ids, profiles, manager)

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
            # Any click while fullscreen returns to the grid — kept simple
            # deliberately: no box-click interaction while fullscreen, since
            # box coordinates are computed for the small grid thumbnails and
            # don't line up with the full-resolution fullscreen frame.
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
        local_x, local_y = img_x - col * THUMB_W, img_y - row * THUMB_H

        with registry_lock:
            data = registry.get(cam_name, {}).get("latest")

        if data:
            for box in data.get("boxes", []):
                if box["x1"] <= local_x <= box["x2"] and box["y1"] <= local_y <= box["y2"]:
                    if box["id"] is None:
                        pass#open_unknown_popup(root, box, known_encodings, known_ids, profiles, lock)
                    else:
                        open_known_popup(root, box, profiles, lock)
                    return

        # No face box hit — go fullscreen on this camera instead
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
            except queue_module.Empty:
                pass

            with viewer_counts_lock:
                has_web_viewer = viewer_counts.get(cam_name, 0) > 0
            wants_full_res = has_web_viewer or (fullscreen_camera == cam_name)

            if wants_full_res:
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

        root.after(30, update_frame)

    update_frame()

    def on_close():
        for cam_name in list(registry.keys()):
            stop_camera(cam_name, registry)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
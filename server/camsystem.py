"""
Multi-camera security system with:
  - ID-based identity (not name-based) — two people can share a display
    name without being confused, since matching happens against a unique
    ID, and "name" is just one editable field on that ID's profile.
  - Match confidence shown on screen next to recognized names.
  - Cameras can be added/removed live from a control panel window,
    no restart needed.

Architecture per camera: TWO processes (capture + detector), same as
before — capture stays lightweight and never stutters, detector does the
expensive CNN encoding in full isolation on its own core.

Known faces / profiles are shared across ALL detector processes via
multiprocessing.Manager.
"""

import json
import math
import multiprocessing
import os
import pickle
import queue as queue_module
import time
import tkinter as tk
import uuid
from concurrent.futures import ThreadPoolExecutor
from tkinter import messagebox

import cv2
import face_recognition
import mediapipe as mp
import numpy as np
import requests
from PIL import Image, ImageTk

# ===================
# Seed cameras — more can be added live from the control panel afterward
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
DETECTION_INTERVAL_SEC = 0.5
PROCESS_SCALE = 0.5

THUMB_W = 320
THUMB_H = 240

RECORDINGS_DIR = "recordings"
RECORDING_SEGMENT_SEC = 5 * 60  # new file every 30 minutes
RECORD_FPS = 15  # approximate — actual pull rate varies, this just sets playback speed metadata


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


def save_known_faces_to_disk(known_encodings, known_ids):
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": list(known_encodings), "ids": list(known_ids)}, f)


def load_profiles_from_disk():
    if not os.path.exists(PROFILES_FILE):
        return {}
    with open(PROFILES_FILE, "r") as f:
        return json.load(f)


def save_profiles_to_disk(profiles):
    with open(PROFILES_FILE, "w") as f:
        json.dump(dict(profiles), f, indent=2)


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


def push_latest(q: multiprocessing.Queue, item):
    try:
        q.get_nowait()
    except queue_module.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue_module.Full:
        pass


# ---------- detector process ----------

def detector_worker(camera_name, known_encodings, known_ids, profiles, frame_queue, boxes_queue, stop_event):
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

            local_encodings = list(known_encodings)
            local_ids = list(known_ids)
            local_profiles = dict(profiles)

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
                    encodings = face_recognition.face_encodings(
                        rgb_small, known_face_locations=face_location
                    )

                    person_id = None
                    display_name = "Unknown"
                    confidence = None

                    if encodings and local_encodings:
                        distances = face_recognition.face_distance(local_encodings, encodings[0])
                        best_match_idx = int(np.argmin(distances))
                        best_distance = distances[best_match_idx]
                        if best_distance <= RECOGNITION_TOLERANCE:
                            person_id = local_ids[best_match_idx]
                            display_name = local_profiles.get(person_id, {}).get("name", person_id)
                            # Rough similarity score, not a calibrated probability —
                            # lower distance = better match, so invert it into a percentage
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


def open_new_segment(camera_name, frame_w, frame_h):
    cam_dir = os.path.join(RECORDINGS_DIR, camera_name)
    os.makedirs(cam_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(cam_dir, f"{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filepath, fourcc, RECORD_FPS, (frame_w, frame_h))
    print(f"[{camera_name}] Recording new segment: {filepath}")
    return writer


# ---------- capture process ----------

def capture_worker(camera_name, camera_type, source, frame_queue, boxes_queue, out_queue):
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


# ---------- popup UI (main process only) ----------

def open_known_popup(root, box, profiles, lock):
    person_id = box["id"]
    profile = dict(profiles.get(person_id, {"name": box["name"], "age": ""}))

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

    field_rows = []  # list of (key, tk.Entry) — read at Save time

    def add_row(key, value):
        row = tk.Frame(fields_frame)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=f"{key}:", width=10, anchor="w").pack(side="left")
        entry = tk.Entry(row)
        entry.insert(0, value)
        entry.pack(side="left", fill="x", expand=True)
        field_rows.append((key, entry))

    # Ensure name/age always show first, then any other custom fields
    add_row("name", profile.get("name", ""))
    add_row("age", profile.get("age", ""))
    for key, value in profile.items():
        if key not in ("name", "age"):
            add_row(key, value)

    # ---- add a brand new custom field ----
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
            profiles[person_id] = updated
            save_profiles_to_disk(profiles)
        win.destroy()

    tk.Button(win, text="Save", command=save_profile).pack(pady=10)


def open_unknown_popup(root, box, known_encodings, known_ids, profiles, lock):
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
            known_encodings.append(encodings[0])
            known_ids.append(person_id)
            save_known_faces_to_disk(known_encodings, known_ids)

            profiles[person_id] = {"name": new_name, "age": age_entry.get().strip()}
            save_profiles_to_disk(profiles)

        print(f"Enrolled new person: {new_name} (id={person_id}, visible to all cameras)")
        win.destroy()

    tk.Button(win, text="Enroll as known", command=enroll).pack(pady=(0, 10))


# ---------- camera lifecycle (add/remove live) ----------

def start_camera(name, cam_type, source, known_encodings, known_ids, profiles, registry, manager):
    """registry is a dict: name -> {process handles + queues}. Mutated in place."""
    frame_queue = multiprocessing.Queue(maxsize=1)
    boxes_queue = multiprocessing.Queue(maxsize=1)
    out_queue = multiprocessing.Queue(maxsize=1)
    stop_event = manager.Event()  # Manager-backed — avoids the raw-semaphore rebuild issue on spawn

    detector_p = multiprocessing.Process(
        target=detector_worker,
        args=(name, known_encodings, known_ids, profiles, frame_queue, boxes_queue, stop_event),
    )
    capture_p = multiprocessing.Process(
        target=capture_worker,
        args=(name, cam_type, source, frame_queue, boxes_queue, out_queue),
    )
    detector_p.start()
    capture_p.start()

    registry[name] = {
        "out_queue": out_queue,
        "stop_event": stop_event,
        "detector_p": detector_p,
        "capture_p": capture_p,
        "latest": None,
    }


def stop_camera(name, registry):
    entry = registry.pop(name, None)
    if entry is None:
        return
    entry["stop_event"].set()
    entry["detector_p"].terminate()
    entry["capture_p"].terminate()
    entry["detector_p"].join()
    entry["capture_p"].join()


# ---------- control panel (visible Tk window) ----------

def build_control_panel(root, registry, known_encodings, known_ids, profiles, manager):
    panel = tk.Toplevel(root)
    panel.title("Camera Control Panel")
    panel.geometry("300x300")

    tk.Label(panel, text="Active cameras:").pack(anchor="w", padx=10, pady=(10, 0))
    listbox = tk.Listbox(panel)
    listbox.pack(padx=10, pady=5, fill="both", expand=True)

    def refresh_listbox():
        listbox.delete(0, "end")
        for cam_name in registry:
            listbox.insert("end", cam_name)

    def remove_selected():
        selection = listbox.curselection()
        if not selection:
            return
        cam_name = listbox.get(selection[0])
        stop_camera(cam_name, registry)
        refresh_listbox()

    def open_add_form():
        form = tk.Toplevel(panel)
        form.title("Add Camera")

        tk.Label(form, text="Name:").pack(anchor="w", padx=10, pady=(10, 0))
        name_entry = tk.Entry(form, width=25)
        name_entry.pack(padx=10)

        tk.Label(form, text="Type:").pack(anchor="w", padx=10, pady=(10, 0))
        type_var = tk.StringVar(value="http")
        tk.OptionMenu(form, type_var, "http", "usb").pack(padx=10, anchor="w")

        tk.Label(form, text="Source (URL for http, device index for usb):").pack(
            anchor="w", padx=10, pady=(10, 0))
        source_entry = tk.Entry(form, width=25)
        source_entry.pack(padx=10)

        status = tk.Label(form, text="", fg="red")
        status.pack(pady=5)

        def submit():
            name = name_entry.get().strip()
            cam_type = type_var.get()
            source_raw = source_entry.get().strip()

            if not name or not source_raw:
                status.config(text="Fill in all fields")
                return
            if name in registry:
                status.config(text="A camera with this name already exists")
                return

            source = source_raw
            if cam_type == "usb":
                try:
                    source = int(source_raw)
                except ValueError:
                    status.config(text="USB source must be a device index number (e.g. 0)")
                    return

            start_camera(name, cam_type, source, known_encodings, known_ids, profiles, registry, manager)
            refresh_listbox()
            form.destroy()

        tk.Button(form, text="Add", command=submit).pack(pady=10)

    tk.Button(panel, text="Add Camera", command=open_add_form).pack(padx=10, pady=(0, 5), fill="x")
    tk.Button(panel, text="Remove Selected", command=remove_selected).pack(padx=10, pady=(0, 10), fill="x")

    refresh_listbox()
    return refresh_listbox


# ---------- main process ----------

def main():
    multiprocessing.set_start_method("fork")

    manager = multiprocessing.Manager()
    initial_encodings, initial_ids = load_known_faces_from_disk()
    initial_profiles = load_profiles_from_disk()

    known_encodings = manager.list(initial_encodings)
    known_ids = manager.list(initial_ids)
    profiles = manager.dict(initial_profiles)
    lock = manager.Lock()

    registry = {}  # camera_name -> {out_queue, stop_event, detector_p, capture_p, latest}
    for cam in INITIAL_CAMERAS:
        start_camera(cam["name"], cam["type"], cam["source"],
                     known_encodings, known_ids, profiles, registry, manager)

    root = tk.Tk()
    root.withdraw()  # only used to host Toplevels (control panel + popups)
    build_control_panel(root, registry, known_encodings, known_ids, profiles, manager)

    window_name = "Security Cameras"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        names = list(registry.keys())
        cols = max(1, math.ceil(math.sqrt(len(names))))
        col, row = x // THUMB_W, y // THUMB_H
        idx = row * cols + col
        if idx >= len(names):
            return
        cam_name = names[idx]
        local_x, local_y = x - col * THUMB_W, y - row * THUMB_H

        data = registry.get(cam_name, {}).get("latest")
        if not data:
            return
        for box in data.get("boxes", []):
            if box["x1"] <= local_x <= box["x2"] and box["y1"] <= local_y <= box["y2"]:
                if box["id"] is None:
                    open_unknown_popup(root, box, known_encodings, known_ids, profiles, lock)
                else:
                    open_known_popup(root, box, profiles, lock)
                break

    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            names = list(registry.keys())
            n = len(names)
            cols = max(1, math.ceil(math.sqrt(n))) if n else 1
            rows = max(1, math.ceil(n / cols)) if n else 1

            for cam_name in names:
                try:
                    registry[cam_name]["latest"] = registry[cam_name]["out_queue"].get_nowait()
                except queue_module.Empty:
                    pass

            canvas = np.zeros((rows * THUMB_H, cols * THUMB_W, 3), dtype=np.uint8)
            if n == 0:
                cv2.putText(canvas, "No cameras — use the control panel to add one",
                            (10, THUMB_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            for idx, cam_name in enumerate(names):
                row, col = divmod(idx, cols)
                y0, x0 = row * THUMB_H, col * THUMB_W
                data = registry[cam_name]["latest"]
                if data is not None:
                    canvas[y0:y0 + THUMB_H, x0:x0 + THUMB_W] = data["frame"]
                else:
                    cv2.putText(canvas, f"{cam_name}: connecting...",
                                (x0 + 20, y0 + THUMB_H // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            cv2.imshow(window_name, canvas)
            root.update()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        for cam_name in list(registry.keys()):
            stop_camera(cam_name, registry)
        root.destroy()


if __name__ == "__main__":
    main()

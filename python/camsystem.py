"""
Multi-camera, single-window version.

Each camera is now TWO processes, not one:
  - capture_worker: pulls frames from the stream, builds the display
    thumbnail, handles nothing CPU-heavy. Runs at full stream speed.
  - detector_worker: does motion detection + face detection + recognition
    + saving. Genuinely CPU-heavy (dlib CNN encoding). Runs in its own
    OS process, so it competes for CPU cycles on its own core, not the
    same GIL as capture_worker — capture never stutters when detection
    fires, because they're not even in the same process anymore.

They talk to each other via two small Queues (latest frame in, latest
boxes out) — not through shared memory, so no locking needed between them.

Known faces / profiles are shared across ALL detector processes via
multiprocessing.Manager — enroll on one camera, every camera recognizes
them right after.
"""

import json
import math
import multiprocessing
import os
import pickle
import queue as queue_module
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor

import cv2
import face_recognition
import mediapipe as mp
import numpy as np
import requests
from PIL import Image, ImageTk

# ===================
# Camera list — "type": "http" for ESP32-CAM streams, "usb" for local webcams
# For usb cameras, "source" is the device index (0, 1, ...) not a URL
# ===================
CAMERAS = [
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

DETECTION_INTERVAL_SEC = 0.5  # detector process won't re-run more often than this
PROCESS_SCALE = 0.5           # detection/encoding runs on a half-size frame

THUMB_W = 320
THUMB_H = 240


# ---------- disk persistence ----------

def load_known_faces_from_disk():
    if not os.path.exists(ENCODINGS_FILE):
        print(f"No {ENCODINGS_FILE} found — run encode_known_faces.py first.")
        return [], []
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]


def save_known_faces_to_disk(known_encodings, known_names):
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": list(known_encodings), "names": list(known_names)}, f)


def load_profiles_from_disk():
    if not os.path.exists(PROFILES_FILE):
        return {}
    with open(PROFILES_FILE, "r") as f:
        return json.load(f)


def save_profiles_to_disk(profiles):
    with open(PROFILES_FILE, "w") as f:
        json.dump(dict(profiles), f, indent=2)


def get_reference_photo(name):
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
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
    """Keep only the newest item in a maxsize=1 queue — never blocks the sender."""
    try:
        q.get_nowait()
    except queue_module.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue_module.Full:
        pass


# ---------- detector process: ALL the CPU-heavy work lives here, isolated ----------

def detector_worker(camera_name, known_encodings, known_names, frame_queue, boxes_queue, stop_event):
    os.makedirs(FACE_SAVE_DIR, exist_ok=True)
    save_executor = ThreadPoolExecutor(max_workers=1)  # disk writes still shouldn't block this loop either

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
            local_names = list(known_names)

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

                    name = "Unknown"
                    if encodings and local_encodings:
                        distances = face_recognition.face_distance(local_encodings, encodings[0])
                        best_match_idx = int(np.argmin(distances))
                        if distances[best_match_idx] <= RECOGNITION_TOLERANCE:
                            name = local_names[best_match_idx]

                    pad_x = int(bw * CROP_PADDING)
                    pad_y = int(bh * CROP_PADDING)
                    cx1 = max(x1 - pad_x, 0)
                    cy1 = max(y1 - pad_y, 0)
                    cx2 = min(x2 + pad_x, w)
                    cy2 = min(y2 + pad_y, h)
                    crop = frame[cy1:cy2, cx1:cx2].copy()

                    new_boxes.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "name": name, "crop": crop,
                    })

                    safe_name = name.replace(" ", "_")
                    filename = os.path.join(
                        FACE_SAVE_DIR, f"{camera_name}_{safe_name}_{int(now * 1000)}.jpg"
                    )
                    save_executor.submit(cv2.imwrite, filename, crop)
                    print(f"[{camera_name}] Saving: {filename}")

            if new_boxes:
                push_latest(boxes_queue, new_boxes)

    save_executor.shutdown(wait=False)


# ---------- capture process: lightweight, just pulls frames + displays ----------

def capture_worker(camera_name, camera_type, source, frame_queue, boxes_queue, out_queue):
    if camera_type == "usb":
        source_desc = f"USB device {source}"
        frames = usb_frame_generator(source)
    else:
        source_desc = source
        frames = frame_generator(source)

    last_boxes = []
    last_boxes_time = 0.0

    print(f"[{camera_name}] Connecting to {source_desc} ...")
    try:
        for frame in frames:
            push_latest(frame_queue, frame)  # hand off to the detector process, non-blocking

            try:
                last_boxes = boxes_queue.get_nowait()
                last_boxes_time = time.time()
            except queue_module.Empty:
                pass

            h, w = frame.shape[:2]
            scale_x, scale_y = THUMB_W / w, THUMB_H / h
            thumb = cv2.resize(frame, (THUMB_W, THUMB_H))

            thumb_boxes = []
            if last_boxes and (time.time() - last_boxes_time) < BOX_PERSIST_SEC:
                for b in last_boxes:
                    tx1, ty1 = int(b["x1"] * scale_x), int(b["y1"] * scale_y)
                    tx2, ty2 = int(b["x2"] * scale_x), int(b["y2"] * scale_y)
                    color = (0, 255, 0) if b["name"] != "Unknown" else (0, 0, 255)
                    cv2.rectangle(thumb, (tx1, ty1), (tx2, ty2), color, 2)
                    cv2.putText(thumb, b["name"], (tx1, max(ty1 - 8, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                    thumb_boxes.append({
                        "x1": tx1, "y1": ty1, "x2": tx2, "y2": ty2,
                        "name": b["name"], "crop": b["crop"],
                    })

            push_latest(out_queue, {"frame": thumb, "boxes": thumb_boxes})
    except Exception as e:
        print(f"[{camera_name}] Stopped: {e}")


# ---------- popup UI (runs in the main process only) ----------

def open_known_popup(root, box, profiles, lock):
    name = box["name"]
    win = tk.Toplevel(root)
    win.title(name)
    win.resizable(True, True)

    ref_path = get_reference_photo(name)
    if ref_path and os.path.exists(ref_path):
        img = Image.open(ref_path)
        img.thumbnail((220, 220))
        photo = ImageTk.PhotoImage(img)
        img_label = tk.Label(win, image=photo)
        img_label.image = photo
        img_label.pack(padx=10, pady=10)

    tk.Label(win, text=name, font=("Arial", 14, "bold")).pack(pady=(0, 10))

    tk.Label(win, text="Notes:").pack(anchor="w", padx=10)
    notes_box = tk.Text(win, width=35, height=6)
    notes_box.insert("1.0", dict(profiles.get(name, {})).get("notes", ""))
    notes_box.pack(padx=10, pady=(0, 10), fill="both", expand=True)

    def save_notes():
        with lock:
            entry = dict(profiles.get(name, {}))
            entry["notes"] = notes_box.get("1.0", "end").strip()
            profiles[name] = entry
            save_profiles_to_disk(profiles)
        win.destroy()

    tk.Button(win, text="Save", command=save_notes).pack(pady=(0, 10))


def open_unknown_popup(root, box, known_encodings, known_names, profiles, lock):
    win = tk.Toplevel(root)
    win.title("Unknown — Enroll")
    win.resizable(True, True)

    crop_rgb = cv2.cvtColor(box["crop"], cv2.COLOR_BGR2RGB)
    img = Image.fromarray(crop_rgb)
    img.thumbnail((220, 220))
    photo = ImageTk.PhotoImage(img)
    img_label = tk.Label(win, image=photo)
    img_label.image = photo
    img_label.pack(padx=10, pady=10)

    tk.Label(win, text="Name:").pack(anchor="w", padx=10)
    name_entry = tk.Entry(win, width=30)
    name_entry.pack(padx=10, pady=(0, 10))

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

        person_dir = os.path.join(KNOWN_FACES_DIR, new_name)
        os.makedirs(person_dir, exist_ok=True)
        photo_path = os.path.join(person_dir, f"{int(time.time())}.jpg")
        cv2.imwrite(photo_path, box["crop"])

        with lock:
            known_encodings.append(encodings[0])
            known_names.append(new_name)
            save_known_faces_to_disk(known_encodings, known_names)

            entry = dict(profiles.get(new_name, {"notes": ""}))
            profiles[new_name] = entry
            save_profiles_to_disk(profiles)

        print(f"Enrolled new person: {new_name} (now visible to all cameras)")
        win.destroy()

    tk.Button(win, text="Enroll as known", command=enroll).pack(pady=(0, 10))


# ---------- main process: combine grid, handle clicks, launch workers ----------

def main():
    multiprocessing.set_start_method("spawn")

    manager = multiprocessing.Manager()
    initial_encodings, initial_names = load_known_faces_from_disk()
    initial_profiles = load_profiles_from_disk()

    known_encodings = manager.list(initial_encodings)
    known_names = manager.list(initial_names)
    profiles = manager.dict(initial_profiles)
    lock = manager.Lock()

    n = len(CAMERAS)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    out_queues = {}
    processes = []
    stop_event = multiprocessing.Event()

    for cam in CAMERAS:
        frame_queue = multiprocessing.Queue(maxsize=1)
        boxes_queue = multiprocessing.Queue(maxsize=1)
        out_queue = multiprocessing.Queue(maxsize=1)
        out_queues[cam["name"]] = out_queue

        detector_p = multiprocessing.Process(
            target=detector_worker,
            args=(cam["name"], known_encodings, known_names, frame_queue, boxes_queue, stop_event),
        )
        capture_p = multiprocessing.Process(
            target=capture_worker,
            args=(cam["name"], cam["type"], cam["source"], frame_queue, boxes_queue, out_queue),
        )
        detector_p.start()
        capture_p.start()
        processes.extend([detector_p, capture_p])

    root = tk.Tk()
    root.withdraw()

    window_name = "Security Cameras"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    latest = {}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        col, row = x // THUMB_W, y // THUMB_H
        idx = row * cols + col
        if idx >= len(CAMERAS):
            return
        camera_name = CAMERAS[idx]["name"]
        local_x, local_y = x - col * THUMB_W, y - row * THUMB_H

        for box in latest.get(camera_name, {}).get("boxes", []):
            if box["x1"] <= local_x <= box["x2"] and box["y1"] <= local_y <= box["y2"]:
                if box["name"] == "Unknown":
                    open_unknown_popup(root, box, known_encodings, known_names, profiles, lock)
                else:
                    open_known_popup(root, box, profiles, lock)
                break

    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            for cam in CAMERAS:
                try:
                    latest[cam["name"]] = out_queues[cam["name"]].get_nowait()
                except queue_module.Empty:
                    pass

            canvas = np.zeros((rows * THUMB_H, cols * THUMB_W, 3), dtype=np.uint8)
            for idx, cam in enumerate(CAMERAS):
                row, col = divmod(idx, cols)
                y0, x0 = row * THUMB_H, col * THUMB_W
                data = latest.get(cam["name"])
                if data is not None:
                    canvas[y0:y0 + THUMB_H, x0:x0 + THUMB_W] = data["frame"]
                else:
                    cv2.putText(canvas, f"{cam['name']}: connecting...",
                                (x0 + 20, y0 + THUMB_H // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            cv2.imshow(window_name, canvas)
            root.update()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stop_event.set()
        cv2.destroyAllWindows()
        root.destroy()
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()

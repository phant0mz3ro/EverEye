"""
Pull frames from the ESP32-CAM's /stream endpoint, run motion-gated face
detection (MediaPipe) + recognition (face_recognition), and show live
boxes: green = known match, red = unknown.

Click any box to open a details popup:
  - Known person: shows their enrolled reference photo, name, and an
    editable notes field (saved to profiles.json).
  - Unknown person: lets you type a name and enroll them on the spot.
"""

import json
import os
import pickle
import time
import tkinter as tk

import cv2
import face_recognition
import mediapipe as mp
import numpy as np
import requests
from PIL import Image, ImageTk

STREAM_URL = "http://10.160.66.104/stream"  # replace with your device's IP
FACE_SAVE_DIR = "detected_faces"
KNOWN_FACES_DIR = "known_faces"
SAVE_COOLDOWN_SEC = 2.0
CROP_PADDING = 0.3

MOTION_THRESHOLD = 25
MOTION_MIN_AREA = 2000

ENCODINGS_FILE = "encodings.pkl"
PROFILES_FILE = "profiles.json"
RECOGNITION_TOLERANCE = 0.6

BOX_PERSIST_SEC = 4.0  # how long a box stays visible/clickable after last detection

mp_face_detection = mp.solutions.face_detection


# ---------- persistence helpers ----------

def load_known_faces():
    if not os.path.exists(ENCODINGS_FILE):
        print(f"No {ENCODINGS_FILE} found — run encode_known_faces.py first. "
              f"Continuing with everyone labeled 'Unknown'.")
        return [], []
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data['encodings'])} known face encodings")
    return data["encodings"], data["names"]


def save_known_faces(known_encodings, known_names):
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)


def load_profiles():
    if not os.path.exists(PROFILES_FILE):
        return {}
    with open(PROFILES_FILE, "r") as f:
        return json.load(f)


def save_profiles(profiles):
    with open(PROFILES_FILE, "w") as f:
        json.dump(profiles, f, indent=2)


def get_reference_photo(name):
    """Path to the first enrolled photo for a known person, or None."""
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    if not os.path.isdir(person_dir):
        return None
    for filename in sorted(os.listdir(person_dir)):
        return os.path.join(person_dir, filename)
    return None


# ---------- stream + motion ----------

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


def detect_motion(prev_gray, curr_gray):
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return any(cv2.contourArea(c) >= MOTION_MIN_AREA for c in contours)


# ---------- popup UI ----------

def open_known_popup(root, box, profiles):
    name = box["name"]
    win = tk.Toplevel(root)
    win.title(name)

    ref_path = get_reference_photo(name)
    if ref_path and os.path.exists(ref_path):
        img = Image.open(ref_path)
        img.thumbnail((220, 220))
        photo = ImageTk.PhotoImage(img)
        img_label = tk.Label(win, image=photo)
        img_label.image = photo  # keep a reference so it isn't garbage collected
        img_label.pack(padx=10, pady=10)

    tk.Label(win, text=name, font=("Arial", 14, "bold")).pack(pady=(0, 10))

    tk.Label(win, text="Notes:").pack(anchor="w", padx=10)
    notes_box = tk.Text(win, width=35, height=6)
    notes_box.insert("1.0", profiles.get(name, {}).get("notes", ""))
    notes_box.pack(padx=10, pady=(0, 10))

    def save_notes():
        profiles.setdefault(name, {})["notes"] = notes_box.get("1.0", "end").strip()
        save_profiles(profiles)
        win.destroy()

    tk.Button(win, text="Save", command=save_notes).pack(pady=(0, 10))


def open_unknown_popup(root, box, known_encodings, known_names, profiles):
    win = tk.Toplevel(root)
    win.title("Unknown — Enroll")

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

        known_encodings.append(encodings[0])
        known_names.append(new_name)
        save_known_faces(known_encodings, known_names)

        profiles.setdefault(new_name, {"notes": ""})
        save_profiles(profiles)

        print(f"Enrolled new person: {new_name}")
        win.destroy()

    tk.Button(win, text="Enroll as known", command=enroll).pack(pady=(0, 10))


def make_mouse_callback(click_state, root, known_encodings, known_names, profiles):
    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for box in click_state["boxes"]:
            if box["x1"] <= x <= box["x2"] and box["y1"] <= y <= box["y2"]:
                if box["name"] == "Unknown":
                    open_unknown_popup(root, box, known_encodings, known_names, profiles)
                else:
                    open_known_popup(root, box, profiles)
                break
    return on_mouse


# ---------- main loop ----------

def main():
    known_encodings, known_names = load_known_faces()
    profiles = load_profiles()
    os.makedirs(FACE_SAVE_DIR, exist_ok=True)

    last_save_time = 0.0
    prev_gray = None
    last_boxes = []
    last_boxes_time = 0.0

    root = tk.Tk()
    root.withdraw()  # no empty root window, just used to host popups + keep the Tk event loop alive

    click_state = {"boxes": []}
    cv2.namedWindow("ESP32-CAM")
    cv2.setMouseCallback(
        "ESP32-CAM",
        make_mouse_callback(click_state, root, known_encodings, known_names, profiles),
    )

    print(f"Connecting to {STREAM_URL} ...")
    try:
        with mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6,
        ) as detector:
            for frame in frame_generator(STREAM_URL):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)

                motion = False
                if prev_gray is not None:
                    motion = detect_motion(prev_gray, gray_blurred)
                prev_gray = gray_blurred

                if motion:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = detector.process(rgb_frame)
                    h, w = frame.shape[:2]
                    new_boxes = []

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

                            face_location = [(y1, x2, y2, x1)]
                            encodings = face_recognition.face_encodings(
                                rgb_frame, known_face_locations=face_location
                            )

                            name = "Unknown"
                            if encodings and known_encodings:
                                distances = face_recognition.face_distance(known_encodings, encodings[0])
                                best_match_idx = int(np.argmin(distances))
                                if distances[best_match_idx] <= RECOGNITION_TOLERANCE:
                                    name = known_names[best_match_idx]

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

                            now = time.time()
                            if now - last_save_time >= SAVE_COOLDOWN_SEC:
                                safe_name = name.replace(" ", "_")
                                filename = os.path.join(
                                    FACE_SAVE_DIR, f"{safe_name}_{int(now * 1000)}.jpg"
                                )
                                cv2.imwrite(filename, crop)
                                last_save_time = now
                                print(f"Saved face crop: {filename}")

                    if new_boxes:
                        last_boxes = new_boxes
                        last_boxes_time = time.time()

                # Draw persisted boxes (whether or not this exact frame had motion),
                # so there's always a window of time to actually click one.
                display_frame = frame
                if last_boxes and (time.time() - last_boxes_time) < BOX_PERSIST_SEC:
                    for b in last_boxes:
                        color = (0, 255, 0) if b["name"] != "Unknown" else (0, 0, 255)
                        cv2.rectangle(display_frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, 2)
                        cv2.putText(display_frame, b["name"], (b["x1"], b["y1"] - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    click_state["boxes"] = last_boxes
                else:
                    click_state["boxes"] = []

                cv2.imshow("ESP32-CAM", display_frame)
                root.update()  # non-blocking pump of the Tk event loop, keeps popups responsive
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except requests.exceptions.RequestException as e:
        print(f"Could not connect to stream: {e}")
        print("Check STREAM_URL and that the ESP32 is powered on and connected.")
    finally:
        cv2.destroyAllWindows()
        root.destroy()


if __name__ == "__main__":
    main()

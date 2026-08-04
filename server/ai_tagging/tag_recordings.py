"""
Standalone prototype: tag saved recordings with detected object classes
(car, person, dog, etc.) so they become searchable later.

Deliberately separate from multi_camera.py — this only reads already-saved
.mp4 files from recordings/, never touches the live camera pipeline. Safe
to test freely; nothing here can affect the running system.

Usage:
    pip install ultralytics
    python tag_recordings.py

Re-run anytime — already-tagged videos are skipped, so it's safe to run
repeatedly as new recordings appear.
"""

import os
import sqlite3

import cv2
from ultralytics import YOLO

RECORDINGS_DIR = "../recordings"  # adjust if this script moves relative to server/
DB_FILE = "detections.db"
SAMPLE_INTERVAL_SEC = 3   # analyze one frame every N seconds — full-frame analysis would be way too slow/redundant
CONFIDENCE_THRESHOLD = 0.4


def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera TEXT,
            video_file TEXT,
            timestamp_offset_sec REAL,
            label TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    return conn


def already_processed(conn, video_file):
    cur = conn.execute("SELECT 1 FROM detections WHERE video_file = ? LIMIT 1", (video_file,))
    return cur.fetchone() is not None


def process_video(model, conn, camera_name, video_path):
    filename = os.path.basename(video_path)
    if already_processed(conn, filename):
        print(f"Skipping {filename} (already tagged)")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    frame_interval = max(1, int(fps * SAMPLE_INTERVAL_SEC))

    frame_idx = 0
    tagged_count = 0
    print(f"Processing {filename} ...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp_sec = frame_idx / fps
            results = model(frame, verbose=False)[0]

            for box in results.boxes:
                confidence = float(box.conf[0])
                if confidence < CONFIDENCE_THRESHOLD:
                    continue
                label = model.names[int(box.cls[0])]
                conn.execute(
                    "INSERT INTO detections (camera, video_file, timestamp_offset_sec, label, confidence) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (camera_name, filename, timestamp_sec, label, confidence),
                )
                tagged_count += 1

        frame_idx += 1

    conn.commit()
    cap.release()
    print(f"  -> {tagged_count} detections saved")


def main():
    conn = init_db()
    model = YOLO("yolov8n.pt")  # auto-downloads (~6MB) on first run

    if not os.path.isdir(RECORDINGS_DIR):
        print(f"No recordings directory found at {RECORDINGS_DIR}")
        return

    for camera_name in sorted(os.listdir(RECORDINGS_DIR)):
        camera_dir = os.path.join(RECORDINGS_DIR, camera_name)
        if not os.path.isdir(camera_dir):
            continue
        for filename in sorted(os.listdir(camera_dir)):
            if filename.endswith(".mp4"):
                process_video(model, conn, camera_name, os.path.join(camera_dir, filename))

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()

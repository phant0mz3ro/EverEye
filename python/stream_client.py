"""
Step 4/5: pull frames from the ESP32-CAM's /stream endpoint and run
face detection on each frame using OpenCV's Haar cascade classifier.

Ships inside opencv-python already — no extra install, no version drama.
Detected faces are boxed on-screen and cropped to disk under detected_faces/
for use in the recognition step next.
"""

import os
import time

import cv2
import numpy as np
import requests

STREAM_URL = "http://10.76.39.104/stream"  # replace with your device's IP
FACE_SAVE_DIR = "detected_faces"
SAVE_COOLDOWN_SEC = 2.0  # avoid saving 30 near-identical crops per second

# Ships inside every opencv-python install under cv2.data.haarcascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def frame_generator(url: str):
    """Yields decoded BGR frames (numpy arrays) pulled from the MJPEG stream."""
    stream = requests.get(url, stream=True, timeout=10)
    buffer = b""

    for chunk in stream.iter_content(chunk_size=1024):
        buffer += chunk
        start = buffer.find(b"\xff\xd8")  # JPEG start-of-image marker
        end = buffer.find(b"\xff\xd9")    # JPEG end-of-image marker

        if start != -1 and end != -1 and end > start:
            jpg_bytes = buffer[start:end + 2]
            buffer = buffer[end + 2:]

            frame = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                yield frame


def main():
    os.makedirs(FACE_SAVE_DIR, exist_ok=True)
    last_save_time = 0.0

    print(f"Connecting to {STREAM_URL} ...")
    try:
        for frame in frame_generator(STREAM_URL):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,      # how much the image is scaled down at each step
                minNeighbors=5,       # higher = fewer false positives, may miss angled faces
                minSize=(60, 60),     # ignore tiny/far-away detections
            )

            now = time.time()
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if now - last_save_time >= SAVE_COOLDOWN_SEC:
                    face_crop = frame[y:y + h, x:x + w]
                    filename = os.path.join(
                        FACE_SAVE_DIR, f"face_{int(now * 1000)}.jpg"
                    )
                    cv2.imwrite(filename, face_crop)
                    last_save_time = now
                    print(f"Saved face crop: {filename}")

            cv2.imshow("ESP32-CAM", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except requests.exceptions.RequestException as e:
        print(f"Could not connect to stream: {e}")
        print("Check STREAM_URL and that the ESP32 is powered on and connected.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



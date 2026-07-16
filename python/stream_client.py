"""
Step 4/5: pull frames from the ESP32-CAM's /stream endpoint and run
face detection on each frame using MediaPipe (pinned to 0.10.9 — later
releases dropped the legacy `solutions` API in favor of the Tasks API).

Detected faces are boxed on-screen and cropped to disk under detected_faces/
for use in the recognition step next.
"""

import os
import time

import cv2
import mediapipe as mp
import numpy as np
import requests

STREAM_URL = "http://10.76.39.104/stream" # replace with your device's IP
FACE_SAVE_DIR = "detected_faces"
SAVE_COOLDOWN_SEC = 2.0  # avoid saving 30 near-identical crops per second
CROP_PADDING = 0.3  # expand crop by 30% on each side to capture full face incl. chin/forehead

MOTION_THRESHOLD = 25       # pixel intensity diff to count as "changed"
MOTION_MIN_AREA = 2000      # min contour area (px) to count as real motion, filters noise

mp_face_detection = mp.solutions.face_detection


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


def detect_motion(prev_gray, curr_gray):
    """Returns True if enough pixels changed between frames to count as motion."""
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)  # close small gaps in the diff blob
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return any(cv2.contourArea(c) >= MOTION_MIN_AREA for c in contours)


def main():
    os.makedirs(FACE_SAVE_DIR, exist_ok=True)
    last_save_time = 0.0
    prev_gray = None

    print(f"Connecting to {STREAM_URL} ...")
    try:
        with mp_face_detection.FaceDetection(
            model_selection=0,        # 0 = short-range model, best for faces within ~2m (fits indoor security use)
            min_detection_confidence=0.6,
        ) as detector:
            for frame in frame_generator(STREAM_URL):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)  # smooths sensor noise so it isn't flagged as motion

                motion = False
                if prev_gray is not None:
                    motion = detect_motion(prev_gray, gray_blurred)
                prev_gray = gray_blurred

                if not motion:
                    cv2.imshow("ESP32-CAM", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue  # skip face detection entirely on static frames — this is the compute saved

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.process(rgb_frame)

                h, w = frame.shape[:2]

                if results.detections:
                    for detection in results.detections:
                        box = detection.location_data.relative_bounding_box
                        x1 = max(int(box.xmin * w), 0)
                        y1 = max(int(box.ymin * h), 0)
                        bw = int(box.width * w)
                        bh = int(box.height * h)
                        x2, y2 = min(x1 + bw, w), min(y1 + bh, h)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        confidence = detection.score[0]
                        cv2.putText(frame, f"{confidence:.2f}", (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        now = time.time()
                        if now - last_save_time >= SAVE_COOLDOWN_SEC and x2 > x1 and y2 > y1:
                            # Pad the crop region outward so we capture the full face,
                            # not just the tight detection box (which clips chin/forehead)
                            pad_x = int(bw * CROP_PADDING)
                            pad_y = int(bh * CROP_PADDING)
                            cx1 = max(x1 - pad_x, 0)
                            cy1 = max(y1 - pad_y, 0)
                            cx2 = min(x2 + pad_x, w)
                            cy2 = min(y2 + pad_y, h)

                            face_crop = frame[cy1:cy2, cx1:cx2]
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



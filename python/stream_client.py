"""
Step 4: pull frames from the ESP32-CAM's existing /stream endpoint.
No firmware changes needed — the camera already serves MJPEG, this just
parses the multipart stream into individual frames we can process.

Face detection (step 5) hooks into the frame loop below.
"""

import cv2
import numpy as np
import requests

STREAM_URL = "http://10.76.39.104/stream"  # replace with your device's IP


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
    print(f"Connecting to {STREAM_URL} ...")
    try:
        for frame in frame_generator(STREAM_URL):
            # Step 5 will slot face detection in right here, per frame.
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

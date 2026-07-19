"""
LUMEN — multi-camera live view hub.

Extends the single-camera WebRTC proof of concept into a real list-view
site: "/" shows every configured camera with online/offline status,
clicking one takes you to its dedicated live feed at "/camera/{name}".

Each camera gets exactly ONE SharedCamera instance, created once at
startup — same fix as before, now scaled to N cameras instead of one.

Run with:  uvicorn camera_hub:app --host 0.0.0.0 --port 8000
"""

import json
import threading
import time

import cv2
import numpy as np
import requests
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

# ===================
# Cameras available in this hub
# ===================
CAMERAS = [
    {"name": "front_door", "type": "http", "source": "http://<esp32-ip-1>/stream"},
    {"name": "usb_test",   "type": "usb",  "source": 0},
]

RTC_CONFIG = RTCConfiguration(iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")])
STALE_AFTER_SEC = 5  # no new frame in this long = considered offline on the list page


class SharedCamera:
    """Opens the camera/stream exactly once — every viewer of this camera
    reads from here, regardless of how many are watching at once."""

    def __init__(self, source_type, source):
        self._source_type = source_type
        self._source = source
        self._latest_frame = None
        self._last_frame_time = 0.0
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        if self._source_type == "usb":
            cap = cv2.VideoCapture(self._source)
            if not cap.isOpened():
                print(f"Could not open USB camera at index {self._source}")
                return
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                with self._lock:
                    self._latest_frame = frame
                    self._last_frame_time = time.time()
        else:
            try:
                response = requests.get(self._source, stream=True, timeout=10)
                buffer = b""
                for chunk in response.iter_content(chunk_size=1024):
                    buffer += chunk
                    start = buffer.find(b"\xff\xd8")
                    end = buffer.find(b"\xff\xd9")
                    if start != -1 and end != -1 and end > start:
                        jpg_bytes = buffer[start:end + 2]
                        buffer = buffer[end + 2:]
                        frame = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self._lock:
                                self._latest_frame = frame
                                self._last_frame_time = time.time()
            except requests.exceptions.RequestException as e:
                print(f"Camera stream error: {e}")

    def get_latest_frame(self):
        with self._lock:
            return self._latest_frame

    def is_online(self):
        with self._lock:
            return (time.time() - self._last_frame_time) < STALE_AFTER_SEC


class CameraStreamTrack(VideoStreamTrack):
    """Reads from a SharedCamera — opens nothing of its own."""

    def __init__(self, camera: SharedCamera):
        super().__init__()
        self._camera = camera

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = self._camera.get_latest_frame()
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame


# One SharedCamera per configured camera, created once at startup
shared_cameras = {
    cam["name"]: SharedCamera(cam["type"], cam["source"]) for cam in CAMERAS
}

app = FastAPI()
templates = Jinja2Templates(directory="templates")
active_connections = set()


@app.get("/")
async def index(request: Request):
    camera_list = [
        {"name": cam["name"], "online": shared_cameras[cam["name"]].is_online()}
        for cam in CAMERAS
    ]
    online_count = sum(1 for c in camera_list if c["online"])
    return templates.TemplateResponse(request, "index.html", {
        "cameras": camera_list,
        "online_count": online_count,
        "total_count": len(camera_list),
    })


@app.get("/camera/{name}")
async def camera_page(request: Request, name: str):
    if name not in shared_cameras:
        return HTMLResponse(f"No camera named '{name}'", status_code=404)
    return templates.TemplateResponse(request, "viewer.html", {"camera_name": name})


@app.websocket("/ws/{name}")
async def signaling(websocket: WebSocket, name: str):
    await websocket.accept()
    if name not in shared_cameras:
        await websocket.close()
        return

    pc = RTCPeerConnection(configuration=RTC_CONFIG)
    active_connections.add(pc)

    track = CameraStreamTrack(shared_cameras[name])
    pc.addTrack(track)

    @pc.on("connectionstatechange")
    async def on_state_change():
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

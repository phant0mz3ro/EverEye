"""
Live view proof of concept: streams ONE camera to a browser over WebRTC.

This is deliberately separate from multi_camera.py — it isolates the new,
hard part (the WebRTC handshake itself) from everything you've already
built, so we can prove the networking works before wiring it into the
full multi-camera + detection system.

How it works, at a glance:
  1. Browser opens the page served at "/", loads viewer.html
  2. Browser's JS creates an RTCPeerConnection, generates an "offer"
     (a text blob describing what it wants to receive), sends it over
     a plain WebSocket to this server
  3. This server (using aiortc — WebRTC for Python) takes that offer,
     attaches our camera as a video track, creates an "answer",
     sends it back over the same WebSocket
  4. Once both sides have exchanged that handshake, video flows
     directly between browser and server (peer-to-peer) — the
     WebSocket was only ever used to set up the connection, not to
     carry video itself

Run with:  uvicorn webrtc_live_view:app --host 0.0.0.0 --port 8000
Then open http://<this-machine's-ip>:8000 in a browser on the same LAN.
"""

import asyncio
import json
import threading

import cv2
import numpy as np
import requests
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

# ===================
# Camera source — same pattern as multi_camera.py
# ===================
SOURCE_TYPE = "usb"  # "http" for ESP32-CAM, "usb" for local webcam
SOURCE = 0            # URL string for http, device index (int) for usb


class SharedCamera:
    """
    Opens the camera/stream exactly ONCE, regardless of how many browser
    tabs or devices connect to watch it. Every CameraStreamTrack just reads
    the latest frame from here — this is what lets multiple viewers watch
    the same camera simultaneously, instead of each connection fighting
    for exclusive hardware access to the same device.
    """

    def __init__(self, source_type, source):
        self._source_type = source_type
        self._source = source
        self._latest_frame = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

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
        else:
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

    def get_latest_frame(self):
        with self._lock:
            return self._latest_frame


shared_camera = SharedCamera(SOURCE_TYPE, SOURCE)  # opened once at server startup


class CameraStreamTrack(VideoStreamTrack):
    """Reads from the single SharedCamera — opens nothing of its own,
    so any number of viewers can connect without fighting over hardware."""

    def __init__(self, camera: SharedCamera):
        super().__init__()
        self._camera = camera

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        frame = self._camera.get_latest_frame()
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)  # blank until first real frame arrives

        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame


RTC_CONFIG = RTCConfiguration(iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")])

app = FastAPI()
active_connections = set()  # keep RTCPeerConnections alive — they'd get garbage collected otherwise


@app.get("/")
async def index():
    with open("viewer.html") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws")
async def signaling(websocket: WebSocket):
    await websocket.accept()
    pc = RTCPeerConnection(configuration=RTC_CONFIG)
    active_connections.add(pc)

    track = CameraStreamTrack(shared_camera)
    pc.addTrack(track)

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

        # keep the socket open so the connection doesn't get torn down
        while True:
            await websocket.receive_text()
    except Exception as e:
        print(f"Signaling closed: {e}")
    finally:
        await pc.close()
        active_connections.discard(pc)

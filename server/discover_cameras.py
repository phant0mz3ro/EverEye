"""
Camera discovery — network (subnet scan) and USB (v4l2 device probe).

Finds ANY camera on the local network serving MJPEG-over-HTTP or RTSP,
regardless of brand or firmware, plus any USB camera visible to the OS
via /dev/video*. No custom firmware, no brand-specific protocol, no
mDNS dependency for the network side — identifies network cameras
purely by behavior: does this IP respond like a camera stream?

Requires:
    pip install requests

Usage:
    from discover_cameras import discover_cameras, discover_usb_cameras

    found = discover_cameras()
    # [{"ip": "192.168.1.42", "port": 80, "stream_path": "/stream",
    #   "protocol": "mjpeg"}, ...]

    usb_found = discover_usb_cameras()
    # [{"index": 0, "name": "USB2.0 HD UVC WebCam", "protocol": "usb"}, ...]
"""

import concurrent.futures
import glob
import ipaddress
import re
import socket
import subprocess

import requests

# Common ports/paths different camera firmwares use for MJPEG streams.
# Extend this list as you encounter new camera types in the field.
COMMON_STREAM_CANDIDATES = [
    (80, "/stream"),
    (80, "/mjpeg"),
    (80, "/video"),
    (80, "/video.mjpg"),
    (8080, "/stream"),
    (8080, "/video"),
    (81, "/stream"),  # some ESP32-CAM builds serve stream on a second port
]

RTSP_PORT = 554  # common for more capable IP cameras (non-MJPEG)


def _get_local_subnet():
    """Best-effort guess at the local /24 subnet based on the Pi's own IP."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
    finally:
        s.close()
    return ipaddress.ip_network(f"{local_ip}/24", strict=False)


def _probe_host(ip):
    """
    Check one IP for a recognizable camera stream. Returns a result
    dict or None. Identifies cameras purely by response behavior
    (MJPEG multipart content-type, or an open RTSP port) — no
    assumption about brand, hostname, or firmware.
    """
    for port, path in COMMON_STREAM_CANDIDATES:
        url = f"http://{ip}:{port}{path}"
        try:
            resp = requests.get(url, stream=True, timeout=0.5)
            content_type = resp.headers.get("Content-Type", "")
            if "multipart/x-mixed-replace" in content_type:
                resp.close()
                return {
                    "ip": ip,
                    "port": port,
                    "stream_path": path,
                    "protocol": "mjpeg",
                }
            resp.close()
        except requests.RequestException:
            continue

    try:
        with socket.create_connection((ip, RTSP_PORT), timeout=0.3):
            return {
                "ip": ip,
                "port": RTSP_PORT,
                "stream_path": None,  # RTSP path conventions vary too much to guess generically
                "protocol": "rtsp",
            }
    except (socket.timeout, ConnectionRefusedError, OSError):
        pass

    return None


def discover_cameras(max_workers=50):
    """
    Probe every host on the local /24 subnet in parallel, looking for
    anything serving MJPEG or RTSP. Takes a few seconds depending on
    subnet size and network responsiveness.
    """
    network = _get_local_subnet()
    candidates = [str(ip) for ip in network.hosts()]

    found = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_probe_host, ip): ip for ip in candidates}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                found.append(result)
    return found


def _v4l2_device_name(device_path):
    """
    Best-effort human-readable name for a /dev/videoN node via v4l2-ctl,
    if it's installed (it usually is on Raspberry Pi OS — part of
    v4l-utils). Falls back to the device path if not available.
    """
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device_path, "--info"],
            capture_output=True, text=True, timeout=1,
        )
        match = re.search(r"Card type\s*:\s*(.+)", result.stdout)
        if match:
            return match.group(1).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return device_path


def _is_capture_device(device_path):
    """
    Some /dev/videoN nodes on a camera are metadata/control nodes, not
    the actual capture stream (common with UVC cameras that expose 2
    nodes per physical camera). Try opening it with OpenCV as the real
    filter — this matches what start_camera will actually do.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(device_path)
        ok = cap.isOpened()
        if ok:
            ok, _ = cap.read()
        cap.release()
        return ok
    except Exception:
        return False


def discover_usb_cameras():
    """
    Enumerate /dev/video* nodes and return the ones that actually open
    and produce a frame — i.e. usable as a `usb` camera source. Index
    matches what OpenCV/start_camera expects (the trailing number on
    the device path).
    """
    found = []
    for device_path in sorted(glob.glob("/dev/video*")):
        match = re.search(r"video(\d+)$", device_path)
        if not match:
            continue
        index = int(match.group(1))
        if not _is_capture_device(device_path):
            continue
        found.append({
            "index": index,
            "name": _v4l2_device_name(device_path),
            "protocol": "usb",
        })
    return found


if __name__ == "__main__":
    print("Scanning local subnet for network cameras (this may take a moment)...")
    cameras = discover_cameras()
    if not cameras:
        print("No network cameras found.")
    for cam in cameras:
        print(f"  {cam['protocol']} camera at {cam['ip']}:{cam['port']}{cam['stream_path'] or ''}")

    print("Scanning for USB cameras...")
    usb_cameras = discover_usb_cameras()
    if not usb_cameras:
        print("No USB cameras found.")
    for cam in usb_cameras:
        print(f"  USB camera: {cam['name']} (index {cam['index']})")

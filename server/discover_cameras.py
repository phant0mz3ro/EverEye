"""
Optimized Camera Discovery — fast network subnet scan and fast USB filtering.
"""

import concurrent.futures
import glob
import ipaddress
import os
import re
import socket
import subprocess
import time
import requests
import cv2
import numpy as np

COMMON_STREAM_CANDIDATES = [
    (80, "/stream"),
    (80, "/mjpeg"),
    (80, "/video"),
    (80, "/video.mjpg"),
    (8080, "/stream"),
    (8080, "/video"),
    (81, "/stream"),
]

RTSP_PORT = 554


def _get_local_subnet():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
    finally:
        s.close()
    return ipaddress.ip_network(f"{local_ip}/24", strict=False)


def _is_port_open(ip, port, timeout=0.15):
    """Fast check if a port is listening before sending full HTTP requests."""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def _probe_host(ip):
    """Probes host only on ports that actively accept TCP connections."""
    # Fast check standard HTTP/RTSP ports
    active_ports = {port for port, _ in COMMON_STREAM_CANDIDATES if _is_port_open(ip, port)}
    has_rtsp = _is_port_open(ip, RTSP_PORT) if RTSP_PORT not in active_ports else True

    for port, path in COMMON_STREAM_CANDIDATES:
        if port not in active_ports:
            continue
        url = f"http://{ip}:{port}{path}"
        try:
            resp = requests.get(url, stream=True, timeout=0.3)
            content_type = resp.headers.get("Content-Type", "")
            if "multipart/x-mixed-replace" in content_type:
                resp.close()
                return {"ip": ip, "port": port, "stream_path": path, "protocol": "mjpeg"}
            resp.close()
        except requests.RequestException:
            continue

    if has_rtsp:
        return {"ip": ip, "port": RTSP_PORT, "stream_path": None, "protocol": "rtsp"}

    return None


def discover_cameras(max_workers=100):
    """Probe network subnets using an expanded worker pool and fast pre-checks."""
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
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device_path, "--info"],
            capture_output=True, text=True, timeout=0.5,
        )
        match = re.search(r"Card type\s*:\s*(.+)", result.stdout)
        if match:
            return match.group(1).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return device_path


def _is_real_video_capture_node(device_path):
    """
    Fast sysfs/v4l2 check to filter out metadata/control nodes
    without invoking slow OpenCV initialization.
    """
    try:
        # Check v4l2 capabilities bitmask via v4l2-ctl if available
        res = subprocess.run(["v4l2-ctl", "-d", device_path, "--all"],
                             capture_output=True, text=True, timeout=0.5)
        if "Video Capture" not in res.stdout:
            return False
    except Exception:
        pass
    return True


def _is_capture_device(device_path):
    if not os.access(device_path, os.R_OK | os.W_OK):
        return False

    # Skip non-capture (metadata) nodes immediately
    if not _is_real_video_capture_node(device_path):
        return False

    try:
        cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))

        ok = False
        # Reduced from 15 attempts to 3 attempts (0.1s sleep) for scan speed
        for _ in range(3):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0 and np.count_nonzero(frame) > 0:
                ok = True
                break
            time.sleep(0.1)
        cap.release()
        return ok
    except Exception:
        return False


def discover_usb_cameras():
    found = []
    device_paths = sorted(glob.glob("/dev/video*"), key=lambda p: int(re.search(r"(\d+)$", p).group(1)))
    for device_path in device_paths:
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
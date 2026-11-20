"""
Generic network camera discovery via subnet scan.

Finds ANY camera on the local network serving MJPEG-over-HTTP or RTSP,
regardless of brand or firmware. No custom firmware, no brand-specific
protocol, no mDNS dependency — identifies cameras purely by behavior:
does this IP respond like a camera stream?

Requires:
    pip install requests

Usage:
    from discover_cameras import discover_cameras

    found = discover_cameras()
    # [{"ip": "192.168.1.42", "port": 80, "stream_path": "/stream",
    #   "protocol": "mjpeg"}, ...]
"""

import concurrent.futures
import ipaddress
import socket

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


def discover_usb_cameras(max_index=10):
    """
    Probe local USB/webcam device indices (/dev/videoN on Linux, or a
    plain index on Windows/Mac) for anything OpenCV can actually open
    AND read a frame from. Separate from the network scan above — no
    HTTP/RTSP involved, just local capture devices. cv2 is imported
    lazily here so the network-only path above stays free of the
    OpenCV dependency. Each index is isolated in its own try/except so
    one misbehaving device (a driver-level error, a device that hangs
    on open) doesn't abort the scan of the remaining indices.
    """
    import cv2

    found = []
    for index in range(max_index):
        cap = None
        try:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    found.append({"index": index, "protocol": "usb"})
        except Exception as e:
            print(f"USB scan: device index {index} raised {e} — skipping")
        finally:
            if cap is not None:
                cap.release()
    return found


def discover_cameras(max_workers=50):
    """
    Probe every host on the local /24 subnet in parallel, looking for
    anything serving MJPEG or RTSP. Takes a few seconds depending on
    subnet size and network responsiveness. Returns [] (rather than
    raising) if the local subnet can't even be determined — e.g. no
    network connection at all — so a caller doing both network and USB
    discovery still gets USB results instead of losing the whole scan.
    """
    try:
        network = _get_local_subnet()
    except OSError as e:
        print(f"Network scan: couldn't determine local subnet ({e}) — skipping network discovery")
        return []
    candidates = [str(ip) for ip in network.hosts()]

    found = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_probe_host, ip): ip for ip in candidates}
        for future in concurrent.futures.as_completed(futures):
            ip = futures[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"Network scan: probing {ip} raised {e} — skipping")
                continue
            if result:
                found.append(result)
    return found


if __name__ == "__main__":
    print("Scanning local subnet for cameras (this may take a moment)...")
    cameras = discover_cameras()
    if not cameras:
        print("No cameras found on the local network.")
    for cam in cameras:
        print(f"  {cam['protocol']} camera at {cam['ip']}:{cam['port']}{cam['stream_path'] or ''}")

    print("Scanning local USB devices...")
    usb_cameras = discover_usb_cameras()
    if not usb_cameras:
        print("No USB cameras found.")
    for cam in usb_cameras:
        print(f"  usb camera at device index {cam['index']}")

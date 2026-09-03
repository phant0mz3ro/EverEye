# Pi 3 Security Camera — Progress Log

## The goal

Take an existing multi-camera security system (Python/Tkinter GUI +
FastAPI web app, originally built around ESP32-CAMs) and get it running
well enough on a **Raspberry Pi 3 (1GB RAM)** to be marketable as a
standalone product — final form factor is the Pi connected to a
monitor, running the GUI directly, no VNC/SSH needed by an end user.

All of this work lives on a separate **`pi3-lite`** git branch, kept
apart from the main/full-feature version.

---

## Phase 1 — Trying to run the original app as-is

The original codebase assumed much more headroom than a Pi 3 has:

- **dlib / face_recognition** — wouldn't build. `/tmp` being a small
  tmpfs (208MB) broke the pip install; fixing `TMPDIR` got further,
  then the compiler got OOM-killed (`cc1plus` killed) on 1GB RAM. Added
  a 2GB swap file to get past that, but the build still took hours —
  no prebuilt aarch64 wheel available via piwheels, so it's a from-
  source compile no matter what.
- **mediapipe, aiortc/WebRTC** — same story: heavy, ARM build pain,
  slow.
- Running with face recognition commented out got the GUI up, but
  surfaced other problems (USB camera not found, a WiFi tab bug) that
  turned out to be symptoms of the Pi being generally overwhelmed by
  the app's architecture, not isolated bugs.

**Conclusion:** the Pi 3 needed a genuinely different, lighter build —
not incremental patches to the original. That's when `pi3-lite` was
created as its own branch.

---

## Phase 2 — Building the Pi 3 tier (first pass)

Stripped out the heaviest pieces and rebuilt the lighter equivalents:

- **Dropped entirely:** face recognition (dlib/face_recognition),
  mediapipe face detection, WebRTC (aiortc) for the web view.
- **Motion detection:** replaced with plain OpenCV frame-differencing,
  running inline in each camera's own capture process (no separate
  detector process — the original's "detector process per camera" was
  actually dead code that was never wired up).
- **Recording:** motion-triggered only (not continuous), to cut disk
  I/O and encode load.
- **Web view:** swapped WebRTC for plain MJPEG streaming — same
  remote-viewing feature, much lighter to serve.
- **USB camera discovery:** added alongside the existing network
  discovery, with a functional test (actually opens + reads a frame)
  to filter out a composite camera's non-capture nodes.
- **New tabs added:** Settings (recording mode: motion vs continuous),
  Recordings (browse/play/delete clips, with an in-app player using
  `cv2.VideoCapture` + Tkinter instead of shelling out to an OS
  player, since the final device has no external player to hand off
  to).

### Bugs fixed along the way
- Tkinter `Listbox` widgets were fighting over the X selection
  (`exportselection=True` by default) — selecting in one silently
  cleared the other's highlight. Fixed with `exportselection=False`.
- USB camera not detected: OpenCV's default backend/format
  autodetection failed against the camera's chipset (Jieli Technology
  composite device). Fixed by forcing the V4L2 backend explicitly and
  setting MJPG format before reading.

### CPU regression and rollback
- Testing showed ~70% average CPU across cores during motion.
- First fix attempt: hardware H264 encode via `ffmpeg`
  (`h264_v4l2m2m`) with software fallback, plus gating the full-res
  live-view frame push to only happen when something's actually
  watching.
- **This made it worse** — cores hit 100%, framerate cratered. Root
  cause: piping frames into `ffmpeg` via a blocking `stdin.write()`
  inside the capture loop meant a slow encoder (it silently fell back
  to software `libx264`, heavier than the original `mp4v`) stalled the
  *entire* capture loop waiting on it.
- **Rolled back:** removed `ffmpeg` entirely. Replaced with
  `AsyncSegmentWriter` — a small background thread per camera with a
  2-slot dropping queue, so a slow encoder loses frames from the
  recording instead of ever blocking capture. Also capped recording
  resolution at 640px wide (pixel count drives encode cost more than
  fps does) and kept the proven `mp4v` codec.

---

## Phase 3 — Strip back to a known-good floor, rebuild up

After the ffmpeg regression, decided to stop patching a stack that was
hard to reason about and instead **rebuild from a minimal, verified
floor** — confirm each layer is cheap before adding the next, instead
of guessing across a pile of simultaneous changes.

**Baseline (rebuilt from scratch):** capture frames from each camera,
show them in a Tkinter grid. Nothing else — no recording, no motion
detection, no web view, no discovery, no extra tabs.

### Getting the baseline itself working
- USB camera got stuck at "connecting..." with no error — several
  compounding causes found in sequence:
  1. A stale/orphaned process from an earlier run was holding
     `/dev/video0` open exclusively, blocking every new attempt.
  2. Once that was cleared, `cv2.VideoCapture().open()` still stalled
     — traced to `multiprocessing`'s `fork` start method combined with
     OpenCV; switched to `spawn` (each camera process gets a genuinely
     fresh Python/OpenCV startup).
  3. Read failures turned out to need format handling refinements:
     setting resolution *before* format, falling back to YUYV if the
     camera rejects MJPEG outright, and validating actual frame
     content (not just the `ret` flag — a "successful" read can still
     return a blank/all-zero frame during warm-up).
  4. Also added tolerance for transient single-frame read failures in
     steady state (only give up after several consecutive failures,
     not on the very first blip) — cheap composite USB cameras hiccup
     occasionally, that's normal, not fatal.
  5. Diagnostic prints from the camera's child process needed
     `flush=True` — otherwise buffered output made real (if slow)
     progress look like a silent hang.
- Same fixes ported into `discover_cameras.py`'s USB probing so
  discovery and actual capture behave consistently.

**Result: baseline confirmed under 25% CPU across cores** on the
actual Pi 3 hardware — capture + display alone is cheap. Everything
expensive lived in the features layered on top, not the core loop.

### Remote testing setup (VNC)
Needed a way to see the Pi's GUI without touching boot config (no
physical monitor available yet, and didn't want to force HDMI output
for something temporary). Landed on:
- **TigerVNC** (`tigervnc-standalone-server`) creating its own virtual
  display (`:1`) — didn't need a real monitor, unlike `x11vnc` which
  requires an existing display to share.
- Had to configure `~/.config/tigervnc/xstartup` (not the more commonly
  documented `~/.vnc/xstartup` — this TigerVNC version uses the XDG
  config path) to actually launch something in the session — without
  it, the virtual display started and had nothing to run, so it just
  exited immediately.
- Landed on Openbox (`openbox-session`) as a lightweight window
  manager once installed — the desktop originally flashed didn't
  include `startlxde` (Raspberry Pi OS dropped LXDE as default years
  ago), and Openbox fits the "keep it light" theme of the whole Pi 3
  build anyway.
- Had to explicitly bind with `-localhost no`, since TigerVNC defaults
  to only accepting connections from the Pi itself.

---

## Re-add plan (from the known-good floor)

1. ✅ **Manage Cameras tab** (add/remove cameras from the GUI) — done,
   confirmed CPU still under 30% with it added.
2. ⏭️ **Recording** — re-add `AsyncSegmentWriter` + `CV2SegmentWriter`
   (proven decoupled-from-capture design from Phase 2's rollback;
   `mp4v`, downscaled to 640px, off the capture thread). *Next up.*
3. **Motion detection** (cheap OpenCV frame-diff, no mediapipe/dlib).
4. **Web/remote MJPEG view.**
5. **USB + network discovery scanning** (already-hardened code from
   Phase 2/3, just needs re-wiring into the GUI).
6. **Settings tab, Recordings tab, in-app player.**
7. **WiFi Setup tab.**

Each step gets added in isolation and checked against CPU before
moving to the next — that discipline is what caught the ffmpeg
regression being a regression at all, rather than it just becoming
"how the app performs" by default.

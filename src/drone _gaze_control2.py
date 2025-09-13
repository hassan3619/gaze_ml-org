#!/usr/bin/env python3
"""
Shift-triggered gaze control for stepwise drone movement (DJITelloPy).

- SHIFT (press once): sample gaze (smoothed) from gaze_vector.txt, move for MOVE_TIME, then hover.
- 't': takeoff   |   'l': land
- 'm': start video stream (OpenCV window)   |   'n': stop video stream
- Ctrl+C: safe exit

Self-contained: no imports from extract_gaze_vector.py.
Make sure `GAZE_VECTOR_FILE` points to the file your extractor writes.
"""
### to be tested has smoother controlsssss

from __future__ import annotations
import os
import time
import threading
import math
import sys
from collections import Counter
from contextlib import suppress
from djitellopy import Tello

# conda activate tello_env
# python drone_gaze_control.py

# =====================
# CONFIG
# =====================

# Set this to the absolute path of your gaze_vector.txt if it's not in the same folder.
GAZE_VECTOR_FILE = os.path.join(os.path.dirname(__file__), "gaze_vector.txt")

SPEED = 20             # DJITelloPy RC units: -100..100 (start small: 20–30)
MOVE_TIME = 0.8        # seconds to apply the one-shot gaze move
HOVER_HOLD = 0.15      # seconds between idle keep-alives
AUTO_TAKEOFF = False   # True to auto-takeoff on start (usually False)

# Gaze smoothing (temporal)
VOTE_WINDOW = 8            # samples per SHIFT press
SAMPLE_INTERVAL = 0.04     # seconds between samples
MIN_DIRECTION_VOTES = 3    # prefer a direction if it has at least this many votes
CENTER_SUPPRESS_AFTER_MOVE_S = 0.4  # ignore "center" right after a move

# Anti-drift (when idle/hovering)
DRIFT_SPEED_CM_S = 15.0    # if |v| > this while idle, send braking pulse
DRIFT_BRAKE_RC = 15        # small RC correction to brake
DRIFT_BRAKE_TIME = 0.15    # seconds of brake pulse

# Map lowercase gaze -> Tello rc tuple: (left_right, forward_back, up_down, yaw)
GAZE_TO_RC = {
    "left":   (-SPEED, 0, 0, 0),
    "right":  ( SPEED, 0, 0, 0),
    "up":     ( 0, 0,  SPEED, 0),
    "down":   ( 0, 0, -SPEED, 0),
    "center": ( 0, 0, 0, 0),
}

# =====================
# Gaze helpers (inline, matching your classifier)
# =====================

def _read_last_gaze_line(path: str) -> str | None:
    """Return last non-empty line from gaze_vector.txt, or None."""
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        for line in reversed(lines):
            s = line.strip()
            if s:
                return s
        return None
    except FileNotFoundError:
        print(f"[GAZE] file not found: {path}")
        return None
    except Exception as e:
        print(f"[GAZE] read error: {e}")
        return None

def classify_gaze_like_yours(x: float, y: float) -> str:
    """
    Replicates your classify_gaze():
      - swap/invert: gaze_x = -y (horizontal), gaze_y = -x (vertical)
      - threshold box around center
      - pick axis with larger exceedance
      - return 'Left/Right/Up/Down' or 'Center'
    """
    threshold = 0.2  # kept from your script

    gaze_x = -y  # Horizontal (invert Y)
    gaze_y = -x  # Vertical   (invert X)

    if -threshold <= gaze_x <= threshold and -threshold <= gaze_y <= threshold:
        return "Center"

    dx = abs(gaze_x) - threshold
    dy = abs(gaze_y) - threshold

    if dx > dy:
        return "Left" if gaze_x < -threshold else "Right"
    else:
        return "Up" if gaze_y < -threshold else "Down"

def sample_gaze_label(window=VOTE_WINDOW, interval=SAMPLE_INTERVAL,
                      suppress_center_until: float = 0.0) -> str | None:
    """
    Collect multiple samples quickly and return a stable label (lowercase).
    - Majority vote across the window
    - If majority is 'center' but there are enough direction votes, prefer the direction
    - Optionally suppress 'center' within a small cooldown after a move
    """
    labels = []
    last_dbg = None

    deadline = time.time() + window * interval + 0.05
    while len(labels) < window and time.time() < deadline:
        raw = _read_last_gaze_line(GAZE_VECTOR_FILE)
        if raw:
            try:
                x_str, y_str = raw.split(",")
                x, y = float(x_str), float(y_str)
                lab = classify_gaze_like_yours(x, y).lower()
                labels.append(lab)
                last_dbg = (x, y, lab)
            except Exception as e:
                print(f"[GAZE] Parse error '{raw}': {e}")
        time.sleep(interval)

    if not labels:
        return None

    c = Counter(labels)
    winner, votes = c.most_common(1)[0]

    # Center suppression window
    now = time.time()
    if winner == "center" and now < suppress_center_until:
        c.pop("center", None)
        if c:
            winner, votes = c.most_common(1)[0]
        else:
            print("[GAZE] center suppressed; no stable direction")
            return None

    # Prefer a direction if it has enough votes even when center is majority
    if winner == "center":
        best_dir = None
        best_votes = 0
        for d in ("left", "right", "up", "down"):
            if c[d] > best_votes:
                best_dir, best_votes = d, c[d]
        if best_dir and best_votes >= MIN_DIRECTION_VOTES:
            winner, votes = best_dir, best_votes

    print(f"[GAZE] votes={dict(c)} -> winner='{winner}' ({votes})")
    if last_dbg:
        x, y, lab = last_dbg
        print(f"[GAZE] last raw=({x:.3f},{y:.3f}) last lab='{lab}'")

    return winner if winner in GAZE_TO_RC else None

# =====================
# Video streaming (OpenCV) - optional
# =====================
try:
    import cv2
except Exception:
    cv2 = None

class VideoViewer(threading.Thread):
    def __init__(self, tello: Tello):
        super().__init__(daemon=True)
        self.tello = tello
        self.running = False
        self._frame_read = None

    def run(self):
        if cv2 is None:
            print("[VIDEO] OpenCV not available; pip install opencv-python")
            return
        self._frame_read = self.tello.get_frame_read()
        self.running = True
        while self.running:
            frame = self._frame_read.frame
            if frame is None:
                time.sleep(0.01)
                continue
            cv2.imshow("Tello Stream (press 'n' here or hotkey)", frame)
            if cv2.waitKey(1) & 0xFF == ord('n'):
                self.stop()
                break
        with suppress(Exception):
            cv2.destroyAllWindows()

    def stop(self):
        self.running = False

# =====================
# Keyboard controller (Shift edge + t/l/m/n)
# =====================
from pynput import keyboard

class KeyController:
    """Keyboard: Shift (edge), 't' takeoff, 'l' land, 'm' stream on, 'n' stream off."""
    def __init__(self):
        self._shift_edge = False
        self._takeoff = False
        self._land = False
        self._stream_on = False
        self._stream_off = False
        self._lock = threading.Lock()
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.daemon = True
        self._listener.start()

    def _on_press(self, key):
        try:
            with self._lock:
                if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
                    self._shift_edge = True
                else:
                    ch = getattr(key, 'char', '')
                    if ch:
                        ch = ch.lower()
                        if ch == 't':
                            self._takeoff = True
                        elif ch == 'l':
                            self._land = True
                        elif ch == 'm':
                            self._stream_on = True
                        elif ch == 'n':
                            self._stream_off = True
        except Exception:
            pass

    def consume_shift_edge(self) -> bool:
        with self._lock:
            if self._shift_edge:
                self._shift_edge = False
                return True
            return False

    def consume_takeoff(self) -> bool:
        with self._lock:
            if self._takeoff:
                self._takeoff = False
                return True
            return False

    def consume_land(self) -> bool:
        with self._lock:
            if self._land:
                self._land = False
                return True
            return False

    def consume_stream_on(self) -> bool:
        with self._lock:
            if self._stream_on:
                self._stream_on = False
                return True
            return False

    def consume_stream_off(self) -> bool:
        with self._lock:
            if self._stream_off:
                self._stream_off = False
                return True
            return False

# =====================
# IMU/State helpers
# =====================

def _read_speeds_cm_s(tello: Tello):
    """
    Return (vgx, vgy, vgz) from current state (cm/s). If unavailable -> (0,0,0).
    NOTE (Tello axes): vgx forward/back, vgy left/right, vgz up/down (SDK units cm/s).
    """
    try:
        st = tello.get_current_state()
        if not st:
            return 0.0, 0.0, 0.0
        vgx = float(st.get("vgx", 0.0))
        vgy = float(st.get("vgy", 0.0))
        vgz = float(st.get("vgz", 0.0))
        return vgx, vgy, vgz
    except Exception:
        return 0.0, 0.0, 0.0

def _norm3(a, b, c):
    return math.sqrt(a*a + b*b + c*c)

# =====================
# Main
# =====================
def main():
    print("\n=== Shift-triggered Gaze Control (DJITelloPy) ===")
    print("SHIFT: step-move (smoothed)  |  t: takeoff  |  l: land")
    print("m: stream ON (OpenCV)  |  n: stream OFF")
    print(f"Reading gaze from: {GAZE_VECTOR_FILE}\n")

    tello = Tello()
    tello.connect()
    with suppress(Exception):
        print("Battery:", tello.get_battery(), "%")

    if AUTO_TAKEOFF:
        with suppress(Exception):
            tello.takeoff()

    keys = KeyController()
    viewer = None
    center_suppress_until = 0.0  # timestamp to ignore "center" votes after a move

    try:
        while True:
            # --- Hotkeys: takeoff/land
            if keys.consume_takeoff():
                print("[KEY] t → takeoff")
                with suppress(Exception):
                    tello.takeoff()
            if keys.consume_land():
                print("[KEY] l → land")
                with suppress(Exception):
                    tello.land()

            # --- Hotkeys: stream on/off
            if keys.consume_stream_on():
                if cv2 is None:
                    print("[VIDEO] OpenCV not installed: pip install opencv-python")
                else:
                    print("[VIDEO] stream ON")
                    with suppress(Exception):
                        tello.streamon()
                    if viewer is None or not viewer.is_alive():
                        viewer = VideoViewer(tello)
                        viewer.start()

            if keys.consume_stream_off():
                print("[VIDEO] stream OFF")
                if viewer is not None:
                    viewer.stop()
                    viewer = None
                with suppress(Exception):
                    tello.streamoff()

            # --- Idle hover keep-alive
            tello.send_rc_control(0, 0, 0, 0)

            # --- Drift check (when idle)
            vgx, vgy, vgz = _read_speeds_cm_s(tello)
            speed = _norm3(vgx, vgy, vgz)
            if speed > DRIFT_SPEED_CM_S:
                # apply a gentle brake opposite measured horizontal velocity
                brake_lr = -DRIFT_BRAKE_RC if vgy > 0 else (DRIFT_BRAKE_RC if vgy < 0 else 0)
                brake_fb = -DRIFT_BRAKE_RC if vgx > 0 else (DRIFT_BRAKE_RC if vgx < 0 else 0)
                print(f"[DRIFT] |v|={speed:.1f} cm/s -> brake lr={brake_lr} fb={brake_fb}")
                tello.send_rc_control(brake_lr, brake_fb, 0, 0)
                time.sleep(DRIFT_BRAKE_TIME)
                tello.send_rc_control(0, 0, 0, 0)

            # --- SHIFT edge → smoothed gaze vote + step move
            if keys.consume_shift_edge():
                label = sample_gaze_label(
                    window=VOTE_WINDOW,
                    interval=SAMPLE_INTERVAL,
                    suppress_center_until=center_suppress_until
                )
                if label is None:
                    print("[WARN] No stable gaze direction detected. Hovering.")
                else:
                    lr, fb, ud, yaw = GAZE_TO_RC[label]
                    print(f"[MOVE] {label} → rc=({lr},{fb},{ud},{yaw}) for {MOVE_TIME:.2f}s")
                    tello.send_rc_control(lr, fb, ud, yaw)
                    time.sleep(MOVE_TIME)
                    tello.send_rc_control(0, 0, 0, 0)
                    # Debounce: ignore "center" shortly after a move
                    center_suppress_until = time.time() + CENTER_SUPPRESS_AFTER_MOVE_S

            time.sleep(HOVER_HOLD)

    except KeyboardInterrupt:
        print("\n[CTRL+C] Landing and exiting...")
        with suppress(Exception):
            tello.land()
        if viewer is not None:
            viewer.stop()
        with suppress(Exception):
            tello.streamoff()
        tello.end()
        sys.exit(0)

    except Exception as e:
        print(f"[ERROR] {e}")
        with suppress(Exception):
            tello.land()
        if viewer is not None:
            viewer.stop()
        with suppress(Exception):
            tello.streamoff()
        tello.end()
        sys.exit(1)

if __name__ == "__main__":
    main()

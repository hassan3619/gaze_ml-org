#!/usr/bin/env python3
import os
import time
import json
import matplotlib.pyplot as plt

# ====== FILES ======
GAZE_VECTOR_FILE = "gaze_vector.txt"
CALIB_FILE = "calibration.json"

# ====== CLASSIFIER KNOBS (asymmetric, up-biased) ======
THRESH_X = 0.15     # Left/Right threshold
THRESH_UP = 0.12    # Easier to trigger Up
THRESH_DOWN = 0.18  # Slightly harder to trigger Down

# ====== CALIBRATION SETTINGS ======
CALIB_SECONDS = 2.0     # sample window when you press 'C'
MAX_HISTORY = 500       # points shown on the live plot

# ====== GLOBALS ======
last_modified_time = None

# Calibration offsets in "gaze" coordinates *after* your swap/invert:
#   gaze_x = -y_raw   (horizontal)
#   gaze_y = -x_raw   (vertical)
CALIB_OFFSET_X = 0.0
CALIB_OFFSET_Y = 0.0

# runtime state for calibration capture
_calib_collecting = False
_calib_start_time = 0.0
_calib_samples = []   # list of (gx, gy) tuples during capture

# ---------------- I/O helpers ----------------
def _load_calibration():
    global CALIB_OFFSET_X, CALIB_OFFSET_Y
    if os.path.isfile(CALIB_FILE):
        try:
            with open(CALIB_FILE, "r") as f:
                data = json.load(f)
            CALIB_OFFSET_X = float(data.get("offset_x", 0.0))
            CALIB_OFFSET_Y = float(data.get("offset_y", 0.0))
            print(f"[CALIB] Loaded offsets: x={CALIB_OFFSET_X:.3f}, y={CALIB_OFFSET_Y:.3f}")
        except Exception as e:
            print(f"[CALIB] Failed to load {CALIB_FILE}: {e}")

def _save_calibration():
    data = {"offset_x": CALIB_OFFSET_X, "offset_y": CALIB_OFFSET_Y}
    try:
        with open(CALIB_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[CALIB] Saved to {CALIB_FILE}: x={CALIB_OFFSET_X:.3f}, y={CALIB_OFFSET_Y:.3f}")
    except Exception as e:
        print(f"[CALIB] Failed to save calibration: {e}")

# ---------------- Data source ----------------
def read_last_gaze_vector():
    """Reads the last line (theta, phi) written by elg_run.py, only if file changed."""
    global last_modified_time
    try:
        current_modified_time = os.path.getmtime(GAZE_VECTOR_FILE)
        if current_modified_time != last_modified_time:
            with open(GAZE_VECTOR_FILE, 'r') as f:
                lines = f.readlines()
            last_modified_time = current_modified_time
            if lines:
                return lines[-1].strip()
        return None
    except FileNotFoundError:
        # Don't spam: only print once per few seconds if you want
        print("[WARN] Gaze vector file not found.")
        time.sleep(0.5)
        return None
    except Exception as e:
        print(f"[WARN] Read error: {e}")
        return None

# ---------------- Calibration ----------------
def _begin_calibration():
    global _calib_collecting, _calib_start_time, _calib_samples
    _calib_collecting = True
    _calib_start_time = time.time()
    _calib_samples = []
    print("[CALIB] Started. Keep looking at the CENTER of your screen...")

def _finish_calibration():
    global _calib_collecting, CALIB_OFFSET_X, CALIB_OFFSET_Y
    _calib_collecting = False
    if not _calib_samples:
        print("[CALIB] No samples collected.")
        return
    # average samples
    sx = sum(p[0] for p in _calib_samples) / len(_calib_samples)
    sy = sum(p[1] for p in _calib_samples) / len(_calib_samples)
    CALIB_OFFSET_X = sx
    CALIB_OFFSET_Y = sy
    print(f"[CALIB] Done. offsets: x={CALIB_OFFSET_X:.3f}, y={CALIB_OFFSET_Y:.3f}  "
          f"(samples={len(_calib_samples)})")

def reset_calibration():
    global CALIB_OFFSET_X, CALIB_OFFSET_Y
    CALIB_OFFSET_X = 0.0
    CALIB_OFFSET_Y = 0.0
    print("[CALIB] Reset to zeros.")

# ---------------- Mapping + classification ----------------
def map_raw_to_gaze(theta, phi):
    """
    Your file stores current_gaze = [theta, phi].
    In the previous script you used:
        gaze_x = -y_raw, gaze_y = -x_raw
    Here, let's treat raw (x_raw, y_raw) == (theta, phi) to keep naming consistent:
        gaze_x = -phi  (horizontal)   < 0 → left, >0 → right
        gaze_y = -theta (vertical)    < 0 → down, >0 → up
    """
    x_raw = theta
    y_raw = phi
    gaze_x = -y_raw
    gaze_y = -x_raw
    return gaze_x, gaze_y

def normalize_with_calibration(gaze_x, gaze_y):
    """Subtract calibration offsets so screen-center maps to (0,0)."""
    return gaze_x - CALIB_OFFSET_X, gaze_y - CALIB_OFFSET_Y

def classify_normalized(nx, ny):
    """
    Asymmetric thresholds: THRESH_UP < THRESH_DOWN
    """
    # Center check (use the tighter vertical threshold to avoid eating small Up)
    center_x_ok = (-THRESH_X <= nx <= THRESH_X)
    center_y_ok = (-THRESH_UP <= ny <= THRESH_UP)
    if center_x_ok and center_y_ok:
        return "Center"

    # Exceedances
    dx = max(0.0, abs(nx) - THRESH_X)
    if ny >= 0:
        dy = max(0.0, ny - THRESH_UP)      # Up
    else:
        dy = max(0.0, -ny - THRESH_DOWN)   # Down

    if dx > dy:
        return "Left" if nx < -THRESH_X else "Right"
    else:
        return "Up" if ny >= THRESH_UP else "Down"

# ---------------- Plotting ----------------
def get_color_for_direction(direction):
    return {
        "Center": "black",
        "Left": "orange",
        "Right": "cyan",
        "Up": "blue",
        "Down": "green"
    }.get(direction, "gray")

def draw_decision_boundaries(ax):
    # Vertical LR thresholds
    ax.axvline(x= THRESH_X, color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=-THRESH_X, color='gray', linestyle='--', linewidth=1)
    # Horizontal Up/Down thresholds (asymmetric)
    ax.axhline(y= THRESH_UP,   color='gray', linestyle='--', linewidth=1)
    ax.axhline(y=-THRESH_DOWN, color='gray', linestyle='--', linewidth=1)
    # Diagonals for context
    ax.plot([-1, 1], [-1, 1],  color='gray', linestyle=':', linewidth=1)
    ax.plot([-1, 1], [ 1,-1],  color='gray', linestyle=':', linewidth=1)

def update_plot(ax, xs, ys, dirs, status_text=""):
    ax.cla()
    # scatter
    for x, y, d in zip(xs, ys, dirs):
        ax.scatter(x, y, s=12, color=get_color_for_direction(d))
    draw_decision_boundaries(ax)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Horizontal (normalized)')
    ax.set_ylabel('Vertical (normalized)')
    ax.set_title('Real-time Gaze (normalized by calibration)')
    # On-plot HUD
    hud = (f"Offsets: x={CALIB_OFFSET_X:.2f}, y={CALIB_OFFSET_Y:.2f}   "
           f"[C] calibrate  [S] save  [R] reset   {status_text}")
    ax.text(0.01, 1.02, hud, transform=ax.transAxes, fontsize=9, va='bottom')
    plt.pause(0.001)

# ---------------- Matplotlib key handler ----------------
def on_key(event):
    if event.key is None:
        return
    k = event.key.lower()
    if k == 'c':
        # toggle calibration start (if already collecting, ignore; it will auto-finish)
        if not _calib_collecting:
            _begin_calibration()
    elif k == 's':
        _save_calibration()
    elif k == 'r':
        reset_calibration()

# ---------------- Main ----------------
def main():
    print("Gaze Vector Extraction + Calibration")
    print("Controls:  C = calibrate (2s at center)   S = save calibration   R = reset")
    _load_calibration()

    # plotting buffers (normalized points)
    xs, ys, dirs = [], [], []

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.canvas.mpl_connect('key_press_event', on_key)

    status = ""
    last_status_time = 0.0

    try:
        while True:
            raw_line = read_last_gaze_vector()
            # Keep status text briefly visible
            if time.time() - last_status_time > 2.5:
                status = ""

            if raw_line:
                try:
                    theta_str, phi_str = raw_line.split(",")
                    theta = float(theta_str.strip())
                    phi = float(phi_str.strip())

                    # map raw -> gaze, then apply calibration
                    gx, gy = map_raw_to_gaze(theta, phi)
                    nx, ny = normalize_with_calibration(gx, gy)
                    label = classify_normalized(nx, ny)

                    # collect samples for calibration window if active
                    if _calib_collecting:
                        _calib_samples.append((gx, gy))  # collect *pre*-normalized gaze
                        if time.time() - _calib_start_time >= CALIB_SECONDS:
                            _finish_calibration()
                            status = "Calibrated ✔  (press S to save)"
                            last_status_time = time.time()

                    # store for plotting (normalized stream)
                    xs.append(nx); ys.append(ny); dirs.append(label)
                    if len(xs) > MAX_HISTORY:
                        xs = xs[-MAX_HISTORY:]
                        ys = ys[-MAX_HISTORY:]
                        dirs = dirs[-MAX_HISTORY:]

                    update_plot(ax, xs, ys, dirs, status_text=status)

                except Exception as e:
                    print(f"[PARSE] Bad line '{raw_line}': {e}")
            else:
                # No new data yet — still update HUD so you see keypress effect.
                update_plot(ax, xs, ys, dirs, status_text=status)
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nExiting.")
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()

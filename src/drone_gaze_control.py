#!/usr/bin/env python3
"""
Shift-triggered AND optional continuous gaze control for DJI Tello (DJITelloPy).

Key upgrades vs. original:
  • Read gaze live over UDP from elg_run.py (no file polling), with staleness gating.
  • EMA + windowed averaging to smooth jitter.
  • Quick center-bias calibration + dead-zone to avoid drift.
  • Speed scales with gaze magnitude (subtle look = gentle nudge).
  • Step mode (press SHIFT to step-move) and toggleable continuous mode (press 'c').
  • Safety: stale → hover, emergency stop (SPACE), altitude clamp, debounced SHIFT.
  • Telemetry/logging to CSV; battery warnings.

Requirements:
  pip install djitellopy pynput
Make sure elg_run.py is running and broadcasting UDP JSON to 127.0.0.1:50555 of the form:
  {"t": <unix_time>, "gx": <float -1..1>, "gy": <float -1..1>, "ok": true}

Tested with Python 3.10+.
"""
# conda activate tello_env
# cd '/media/hn_97/VOLUME G/GazeML-org/src'
# python drone_gaze_control.py

from __future__ import annotations
import csv
import json
import os
import socket
import sys
import threading
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass

from djitellopy import Tello
from pynput import keyboard

# =====================
# CONFIG
# =====================
UDP_HOST = "127.0.0.1"
UDP_PORT = 50555
STALE_MS = 300                # consider gaze stale after this many ms
SMOOTH_WINDOW = 6             # moving average window for gaze smoothing
EMA_ALPHA = 0.35              # exponential component for smoothing
CENTER_SAMPLES = 80           # samples to collect for center calibration
CENTER_TIMEOUT = 2.0          # seconds to attempt center calibration
DEADZONE = 0.20               # neutral zone radius after centering (0..1)

STEP_MOVE_TIME = 0.80         # seconds of RC command in step mode
STEP_DEBOUNCE_MS = 150        # minimum time between step triggers

SPEED_MIN = 15                # min RC speed when outside deadzone
SPEED_MAX = 40                # max RC speed

CONTINUOUS_RATE_HZ = 12       # RC update rate in continuous mode
IDLE_KEEPALIVE_HZ = 5         # RC 0,0,0,0 keep-alive when idle (prevents failsafe)

ALT_MIN_CM = 30               # don't descend below this (approx)
ALT_MAX_CM = 150              # don't ascend above this (approx)
HEIGHT_QUERY_INTERVAL = 0.8   # seconds between tello.get_height() calls
BATTERY_WARN = 20             # % battery warning threshold
BATTERY_QUERY_INTERVAL = 2.0  # seconds

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

# =====================
# Gaze receiver (UDP)
# =====================
@dataclass
class GazeSample:
    gx: float = 0.0
    gy: float = 0.0
    ok: bool = False
    t: float = 0.0  # unix seconds

class GazeReceiver:
    def __init__(self, host: str = UDP_HOST, port: int = UDP_PORT):
        self._host = host
        self._port = port
        self._latest = GazeSample()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        try:
            # poke socket by sending an empty datagram so recvfrom unblocks
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.sendto(b"{}", (self._host, self._port))
        except Exception:
            pass
        self._thread.join(timeout=1.0)

    def _run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)
        try:
            sock.bind((self._host, self._port))
        except OSError:
            # If already bound elsewhere, still try to read (rare). You can change port if needed.
            pass
        while not self._stop.is_set():
            try:
                data, _ = sock.recvfrom(4096)
                msg = json.loads(data.decode("utf-8"))
                sample = GazeSample(
                    gx=float(msg.get("gx", 0.0)),
                    gy=float(msg.get("gy", 0.0)),
                    ok=bool(msg.get("ok", False)),
                    t=float(msg.get("t", time.time()))
                )
                with self._lock:
                    self._latest = sample
            except socket.timeout:
                continue
            except Exception:
                continue
        sock.close()

    def latest(self) -> GazeSample:
        with self._lock:
            return self._latest

# =====================
# Gaze processing
# =====================
class GazeFilter:
    def __init__(self, receiver: GazeReceiver):
        self.rx = receiver
        self.buf: deque[tuple[float, float]] = deque(maxlen=SMOOTH_WINDOW)
        self.center = (0.0, 0.0)
        self.deadzone = DEADZONE
        self._ema = None  # (gx, gy)

    def _append(self, gx: float, gy: float):
        self.buf.append((gx, gy))
        if self._ema is None:
            self._ema = (gx, gy)
        else:
            ex, ey = self._ema
            self._ema = (EMA_ALPHA * ex + (1 - EMA_ALPHA) * gx,
                         EMA_ALPHA * ey + (1 - EMA_ALPHA) * gy)

    def read_smoothed(self) -> tuple[float, float] | None:
        s = self.rx.latest()
        if not s.ok:
            return None
        if (time.time() - s.t) * 1000.0 > STALE_MS:
            return None
        self._append(s.gx, s.gy)
        # combine simple average with EMA for stability
        ax = sum(x for x, _ in self.buf) / len(self.buf)
        ay = sum(y for _, y in self.buf) / len(self.buf)
        if self._ema is None:
            return ax, ay
        ex, ey = self._ema
        return 0.5 * (ax + ex), 0.5 * (ay + ey)

    def calibrate_center(self, timeout: float = CENTER_TIMEOUT, target_samples: int = CENTER_SAMPLES):
        print("[calib] Hold neutral gaze…")
        t0 = time.time()
        xs: list[float] = []
        ys: list[float] = []
        while (time.time() - t0) < timeout and len(xs) < target_samples:
            s = self.rx.latest()
            if s.ok and (time.time() - s.t) * 1000.0 <= STALE_MS:
                xs.append(s.gx)
                ys.append(s.gy)
            time.sleep(0.01)
        if xs:
            self.center = (sum(xs) / len(xs), sum(ys) / len(ys))
            print(f"[calib] center set to {self.center}")
        else:
            print("[calib] failed (no fresh gaze)")

    def centered(self, gx: float, gy: float) -> tuple[float, float]:
        cx, cy = self.center
        return gx - cx, gy - cy

# =====================
# Keyboard controller
# =====================
class KeyController:
    def __init__(self):
        self._lock = threading.Lock()
        self._shift_edge = False
        self._takeoff = False
        self._land = False
        self._stop = False
        self._toggle_cont = False
        self._recalib = False
        self._emergency = False
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.daemon = True
        self._listener.start()

    def _on_press(self, key):
        try:
            with self._lock:
                if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
                    self._shift_edge = True
                elif key == keyboard.Key.space:
                    self._emergency = True
                elif key == keyboard.Key.esc:
                    self._stop = True
                else:
                    ch = getattr(key, 'char', '')
                    if not ch:
                        return
                    ch = ch.lower()
                    if ch == 't':
                        self._takeoff = True
                    elif ch == 'l':
                        self._land = True
                    elif ch == 'c':
                        self._toggle_cont = True
                    elif ch == 'g':
                        self._recalib = True
        except Exception:
            pass

    def consume_shift_edge(self) -> bool:
        with self._lock:
            v = self._shift_edge
            self._shift_edge = False
            return v

    def consume_takeoff(self) -> bool:
        with self._lock:
            v = self._takeoff
            self._takeoff = False
            return v

    def consume_land(self) -> bool:
        with self._lock:
            v = self._land
            self._land = False
            return v

    def consume_toggle_cont(self) -> bool:
        with self._lock:
            v = self._toggle_cont
            self._toggle_cont = False
            return v

    def consume_recalib(self) -> bool:
        with self._lock:
            v = self._recalib
            self._recalib = False
            return v

    def consume_emergency(self) -> bool:
        with self._lock:
            v = self._emergency
            self._emergency = False
            return v

    def should_stop(self) -> bool:
        with self._lock:
            return self._stop

# =====================
# Utility
# =====================
def ensure_logdir() -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOG_DIR, f"gaze_drone_{ts}.csv")
    return path


def scale_speed(mag: float) -> int:
    mag = max(0.0, min(1.0, mag))
    return int(SPEED_MIN + (SPEED_MAX - SPEED_MIN) * mag)


def classify_vector(gf: GazeFilter) -> tuple[str, tuple[int, int, int, int], int]:
    """Return (label, rc_tuple, speed). Label in {'left','right','up','down','center'}"""
    sm = gf.read_smoothed()
    if sm is None:
        return "center", (0, 0, 0, 0), 0
    gx, gy = gf.centered(*sm)

    # Deadzone check
    if abs(gx) < gf.deadzone and abs(gy) < gf.deadzone:
        return "center", (0, 0, 0, 0), 0

    # Exceedance beyond deadzone
    dx = abs(gx) - gf.deadzone
    dy = abs(gy) - gf.deadzone

    if dx >= dy:
        label = "left" if gx < 0 else "right"
        mag = min(1.0, dx / (1.0 - gf.deadzone))
        spd = scale_speed(mag)
        rc = (-spd, 0, 0, 0) if label == "left" else (spd, 0, 0, 0)
    else:
        label = "down" if gy < 0 else "up"
        mag = min(1.0, dy / (1.0 - gf.deadzone))
        spd = scale_speed(mag)
        rc = (0, 0, -spd, 0) if label == "down" else (0, 0, spd, 0)

    return label, rc, spd


# Clamp up/down command based on height
_last_height = 0
_last_h_query = 0.0

def clamp_ud_with_height(tello: Tello, rc: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    global _last_height, _last_h_query
    lr, fb, ud, yaw = rc
    now = time.time()
    if (now - _last_h_query) > HEIGHT_QUERY_INTERVAL:
        with suppress(Exception):
            _last_height = int(tello.get_height())  # cm
        _last_h_query = now
    # Clamp
    if ud > 0 and _last_height >= ALT_MAX_CM:
        ud = 0
    if ud < 0 and _last_height <= ALT_MIN_CM:
        ud = 0
    return (lr, fb, ud, yaw)


_last_batt = None
_last_b_query = 0.0

def poll_battery(tello: Tello):
    global _last_batt, _last_b_query
    now = time.time()
    if (now - _last_b_query) > BATTERY_QUERY_INTERVAL:
        with suppress(Exception):
            _last_batt = int(tello.get_battery())
        _last_b_query = now
        if _last_batt is not None:
            print(f"[battery] {_last_batt}%")
            if _last_batt <= BATTERY_WARN:
                print("[battery] WARNING: low battery")


# =====================
# Main
# =====================
def main():
    print("\n=== Gaze Control for Tello ===")
    print("t: takeoff | l: land | SHIFT: step move | c: toggle continuous | g: calibrate center | SPACE: emergency hover | ESC: quit")

    # Gaze pipeline
    rx = GazeReceiver()
    rx.start()
    gf = GazeFilter(rx)

    # Connect drone
    tello = Tello()
    tello.connect()
    with suppress(Exception):
        print("Battery:", tello.get_battery(), "%")

    keys = KeyController()

    # Logging
    log_path = ensure_logdir()
    log_f = open(log_path, "w", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    log.writerow(["ts","mode","label","gx","gy","speed","rc_lr","rc_fb","rc_ud","rc_yaw","height_cm","battery_%"])  # header

    # Initial calibration
    gf.calibrate_center(CENTER_TIMEOUT, CENTER_SAMPLES)

    continuous = False
    last_step_ms = 0.0

    try:
        last_idle = 0.0
        while not keys.should_stop():
            # periodic telemetry
            poll_battery(tello)

            # emergency hover
            if keys.consume_emergency():
                print("[EMERGENCY] Hover")
                with suppress(Exception):
                    tello.send_rc_control(0, 0, 0, 0)

            # takeoff/land
            if keys.consume_takeoff():
                print("[KEY] t → takeoff")
                with suppress(Exception):
                    tello.takeoff()
            if keys.consume_land():
                print("[KEY] l → land")
                with suppress(Exception):
                    tello.land()

            # toggle continuous
            if keys.consume_toggle_cont():
                continuous = not continuous
                mode = "CONTINUOUS" if continuous else "STEP"
                print(f"[mode] {mode}")

            # recalibrate
            if keys.consume_recalib():
                gf.calibrate_center(CENTER_TIMEOUT, CENTER_SAMPLES)

            # STEP MODE (edge-triggered SHIFT)
            if not continuous and keys.consume_shift_edge():
                now_ms = time.time() * 1000.0
                if (now_ms - last_step_ms) >= STEP_DEBOUNCE_MS:
                    label, rc, spd = classify_vector(gf)
                    rc = clamp_ud_with_height(tello, rc)
                    print(f"[STEP] {label} speed={spd} rc={rc} for {STEP_MOVE_TIME:.2f}s")
                    # log
                    sm = gf.read_smoothed()
                    gx, gy = sm if sm else (None, None)
                    with suppress(Exception):
                        log.writerow([time.time(), "step", label, gx, gy, spd, *rc, _last_height, _last_batt])
                        log_f.flush()
                    # execute
                    with suppress(Exception):
                        tello.send_rc_control(*rc)
                        time.sleep(STEP_MOVE_TIME)
                        tello.send_rc_control(0, 0, 0, 0)
                    last_step_ms = now_ms

            # CONTINUOUS MODE (send at fixed rate)
            if continuous:
                label, rc, spd = classify_vector(gf)
                rc = clamp_ud_with_height(tello, rc)
                # log
                sm = gf.read_smoothed()
                gx, gy = sm if sm else (None, None)
                with suppress(Exception):
                    log.writerow([time.time(), "cont", label, gx, gy, spd, *rc, _last_height, _last_batt])
                    log_f.flush()
                with suppress(Exception):
                    tello.send_rc_control(*rc)
                time.sleep(1.0 / CONTINUOUS_RATE_HZ)
            else:
                # idle keep-alive to prevent failsafe
                if (time.time() - last_idle) >= (1.0 / IDLE_KEEPALIVE_HZ):
                    with suppress(Exception):
                        tello.send_rc_control(0, 0, 0, 0)
                    last_idle = time.time()
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[CTRL+C] Exiting…")
    finally:
        # Cleanup
        rx.stop()
        with suppress(Exception):
            log_f.close()
        with suppress(Exception):
            tello.send_rc_control(0, 0, 0, 0)
            time.sleep(0.1)
        with suppress(Exception):
            tello.land()
        with suppress(Exception):
            tello.end()
        print(f"Logs saved to: {log_path}")


if __name__ == "__main__":
    main()

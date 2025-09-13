#!/usr/bin/env python3
"""
Blink + Gaze hybrid control for Tello (DJITelloPy)

Stability additions inspired by tello_control_updated.py:
- Gaze moves use *smoothed RC ramp* (anti-jerk) instead of a hard step.
- Precision mode (toggle with 'z') lowers speeds/angles.
- Landing confirmation via altitude; zero RC on touchdown.
- 1s global cooldown after each drone command (rotate/move/takeoff/land).
- (Optional) IMU freshness check hook (kept light here; can be expanded).
- Battery + Altitude overlays.
- LIVE video shown via Pygame window (960x720) for smoother display.

Control mapping (updated):
- Two DOUBLE blinks (within window) → TOGGLE TAKEOFF/LAND
- 't' → Start recording (stream must be ON)
- 'l' → Stop recording
- 'm' → Stream ON | 'n' → Stream OFF
- 'z' → Precision mode toggle
- SHIFT → one-shot smoothed gaze RC
"""
# conda activate tello_env
# cd '/media/hn_97/VOLUME G/GazeML-org/src'
# python drone_gaze_blink_control.py

from __future__ import annotations
import os
import sys
import time
import threading
import queue
import importlib.util
from contextlib import suppress
from typing import Optional

from djitellopy import Tello
from pynput import keyboard
import cv2
import pygame  # Pygame for display

# ---- blink events ----
BLINK_PATH = "/media/hn_97/VOLUME G/gaze-tracking-main/ablink.py"  # <= keep your path
BLINK_DIR = os.path.dirname(BLINK_PATH)
if BLINK_DIR not in sys.path:
    sys.path.insert(3, BLINK_DIR)
if not os.path.isfile(BLINK_PATH):
    raise FileNotFoundError(f"ablink.py not found at {BLINK_PATH}")
spec = importlib.util.spec_from_file_location("ablink_ext", BLINK_PATH)
blink_ext = importlib.util.module_from_spec(spec)
sys.modules["ablink_ext"] = blink_ext
spec.loader.exec_module(blink_ext)
blink_events = blink_ext.blink_events

# =====================
# CONFIG
# =====================

GAZE_VECTOR_FILE = os.path.join(os.path.dirname(__file__), "gaze_vector.txt")

# Nominal speeds (precision mode scales these down)
SPEED = 40            # RC magnitude for gaze step (cm/s up/down, L/R)
MOVE_TIME = 0.8       # seconds to apply gaze control (total)
HOVER_HOLD = 0.2
AUTO_TAKEOFF = False

ROTATE_DEG = 45
MOVE_DIST_CM = 30
CONSEC_WINDOW_S = 1.0                 # single-eye consecutive window
DOUBLE_CONFIRM_WINDOW_S = 1.0         # window to see 2x DOUBLE blinks

VIDEO_FPS = 30
TELEMETRY_INTERVAL_S = 1.0

# NEW: global cooldown + precision-mode factors + RC ramp config
CMD_COOLDOWN_S = 1.0
PRECISION_FACTOR = 0.5        # 50% of speeds/angles when precision mode is ON
RC_RAMP_FPS = 30              # ramp smoothing rate (Hz)
RC_RAMP_STEPS = int(MOVE_TIME * RC_RAMP_FPS)  # steps over which to ramp

# Pygame UI config
PG_WIDTH, PG_HEIGHT = 960, 720
PG_FPS = 30

# Logging / folders (created at startup)
BASE_DIR = os.path.join(os.path.dirname(__file__), "tello gaze log")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

# =====================
# Gaze helpers
# =====================

def _read_last_gaze_line(path: str) -> Optional[str]:
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        for line in reversed(lines):
            s = line.strip()
            if s:
                return s
    except FileNotFoundError:
        print(f"[GAZE] file not found: {path}")
    except Exception as e:
        print(f"[GAZE] read error: {e}")
    return None

def classify_gaze_like_yours(x: float, y: float) -> str:
    threshold = 0.2
    gaze_x = -y
    gaze_y = -x
    if -threshold <= gaze_x <= threshold and -threshold <= gaze_y <= threshold:
        return "Center"
    dx = abs(gaze_x) - threshold
    dy = abs(gaze_y) - threshold
    if dx > dy:
        return "Left" if gaze_x < -threshold else "Right"
    else:
        return "Up" if gaze_y < -threshold else "Down"

def read_and_classify_once(timeout: float = 0.7) -> Optional[str]:
    t0 = time.time()
    while time.time() - t0 < timeout:
        raw = _read_last_gaze_line(GAZE_VECTOR_FILE)
        if raw:
            try:
                x_str, y_str = raw.split(",")
                x, y = float(x_str), float(y_str)
                label = classify_gaze_like_yours(x, y).lower()
                if label in {"left","right","up","down","center"}:
                    print(f"[GAZE] raw=({x:.3f},{y:.3f}) → {label}")
                    return label
            except Exception as e:
                print(f"[GAZE] parse/classify error for '{raw}': {e}")
        time.sleep(0.05)
    return None

def rc_from_gaze(label: str, speed: int) -> tuple[int,int,int,int]:
    return {
        "left":   (-speed, 0, 0, 0),
        "right":  ( speed, 0, 0, 0),
        "up":     ( 0, 0,  speed, 0),
        "down":   ( 0, 0, -speed, 0),
        "center": ( 0, 0, 0, 0),
    }[label]

# =====================
# Keyboard
# =====================

class KeyController:
    """Keyboard: Shift(edge), 't' REC START, 'l' REC STOP, 'm' streamon, 'n' streamoff, 'z' precision toggle."""
    def __init__(self):
        self._shift_edge = False
        self._rec_start = False
        self._rec_stop = False
        self._stream_on = False
        self._stream_off = False
        self._precision_toggle = False
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
                            self._rec_start = True     # start recording
                        elif ch == 'l':
                            self._rec_stop = True      # stop recording
                        elif ch == 'm':
                            self._stream_on = True
                        elif ch == 'n':
                            self._stream_off = True
                        elif ch == 'z':
                            self._precision_toggle = True
        except Exception:
            pass

    def consume_shift_edge(self) -> bool:
        with self._lock:
            if self._shift_edge:
                self._shift_edge = False
                return True
            return False

    def consume_rec_start(self) -> bool:
        with self._lock:
            if self._rec_start:
                self._rec_start = False
                return True
            return False

    def consume_rec_stop(self) -> bool:
        with self._lock:
            if self._rec_stop:
                self._rec_stop = False
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

    def consume_precision_toggle(self) -> bool:
        with self._lock:
            if self._precision_toggle:
                self._precision_toggle = False
                return True
            return False

# =====================
# Flight + Session logging
# =====================

class FlightLogger:
    def __init__(self):
        self.flight_idx = 0
        self._log_fp = None
        self.video_idx = 0
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        self._session_path = os.path.join(LOGS_DIR, f"session_{ts}.txt")
        self._session_fp = open(self._session_path, "a", encoding="utf-8")
        self._write_session_line("== SESSION START ==")

    def _flight_tag(self) -> str:
        return f"flight_{self.flight_idx:03d}"

    def start_flight(self):
        self.flight_idx += 1
        self.video_idx = 0
        log_path = os.path.join(LOGS_DIR, f"{self._flight_tag()}.txt")
        self._log_fp = open(log_path, "a", encoding="utf-8")
        self._write_both("== TAKEOFF ==")
        os.makedirs(self.current_video_dir(), exist_ok=True)

    def end_flight(self):
        self._write_both("== LAND ==")
        if self._log_fp:
            with suppress(Exception):
                self._log_fp.flush()
                self._log_fp.close()
        self._log_fp = None

    def current_video_dir(self) -> str:
        if self.flight_idx == 0:
            return os.path.join(VIDEOS_DIR, "flight_000")
        return os.path.join(VIDEOS_DIR, self._flight_tag())

    def next_video_path(self) -> str:
        self.video_idx += 1
        os.makedirs(self.current_video_dir(), exist_ok=True)
        return os.path.join(self.current_video_dir(), f"video{self.video_idx}.avi")

    def log(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with suppress(Exception):
            self._session_fp.write(line + "\n")
            self._session_fp.flush()
        if self._log_fp:
            with suppress(Exception):
                self._log_fp.write(line + "\n")
                self._log_fp.flush()

    def _write_both(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with suppress(Exception):
            self._session_fp.write(line + "\n")
            self._session_fp.flush()
        if self._log_fp:
            with suppress(Exception):
                self._log_fp.write(line + "\n")
                self._log_fp.flush()

    def _write_session_line(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with suppress(Exception):
            self._session_fp.write(line + "\n")
            self._session_fp.flush()

    def close_session(self):
        self._write_session_line("== SESSION END ==")
        with suppress(Exception):
            self._session_fp.flush()
            self._session_fp.close()

# =====================
# Recording manager
# =====================

class VideoRecorder:
    def __init__(self, logger: FlightLogger, fps: int = VIDEO_FPS):
        self.fps = fps
        self.reader = None
        self.writer = None
        self.active = False
        self._th = None
        self._stop = threading.Event()
        self.logger = logger

    def set_reader(self, reader):
        self.reader = reader

    def start(self):
        if self.active:
            self.logger.log("[REC] already recording")
            return
        if self.reader is None:
            self.logger.log("[REC] no active video stream; cannot start recording.")
            return

        t0 = time.time()
        frame = None
        while time.time() - t0 < 2.0 and self.reader is not None:
            frame = getattr(self.reader, "frame", None)
            if frame is not None:
                break
            time.sleep(0.02)
        if frame is None:
            self.logger.log("[REC] no video frame available; cannot start recording.")
            return

        path = self.logger.next_video_path()
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'M','J','P','G')
        self.writer = cv2.VideoWriter(path, fourcc, self.fps, (w, h))
        if not self.writer.isOpened():
            self.logger.log("[REC] failed to open VideoWriter")
            self.writer = None
            return

        self._stop.clear()
        self.active = True
        self._th = threading.Thread(target=self._loop, name="TelloRecorder", daemon=True)
        self._th.start()
        self.logger.log(f"[REC] recording → {path}")

    def _loop(self):
        next_t = time.time()
        interval = 1.0 / max(self.fps, 1)
        while not self._stop.is_set():
            try:
                frame = None if self.reader is None else self.reader.frame
                if frame is not None and self.writer is not None:
                    self.writer.write(frame)  # write BGR
            except Exception:
                pass
            next_t += interval
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(min(sleep, interval))
            else:
                next_t = time.time()

    def stop(self):
        if not self.active:
            return
        self._stop.set()
        if self._th is not None:
            with suppress(Exception):
                self._th.join(timeout=1.0)
        with suppress(Exception):
            if self.writer is not None:
                self.writer.release()
        self.writer = None
        self.active = False
        self.logger.log("[REC] recording stopped")

    def close(self):
        self.stop()

# =====================
# Helpers: cooldown + RC ramp for smoothed gaze steps
# =====================

def cooldown_after_command(tello: Tello, secs: float = CMD_COOLDOWN_S):
    with suppress(Exception):
        tello.send_rc_control(0, 0, 0, 0)
    time.sleep(max(0.0, secs))

def smoothed_gaze_step(tello: Tello, rc_target: tuple[int,int,int,int], total_time: float = MOVE_TIME):
    """
    Smoothly ramp RC from 0 -> target -> 0 across total_time.
    Uses linear ramp with RC_RAMP_STEPS steps at RC_RAMP_FPS Hz.
    """
    steps = max(2, int(total_time * RC_RAMP_FPS))
    lr_t, fb_t, ud_t, yaw_t = rc_target
    for i in range(steps):
        a = (i + 1) / steps  # 0..1
        lr = int(lr_t * a); fb = int(fb_t * a); ud = int(ud_t * a); yw = int(yaw_t * a)
        with suppress(Exception):
            tello.send_rc_control(lr, fb, ud, yw)
        time.sleep(1.0 / RC_RAMP_FPS)
    # ramp down
    for i in range(steps):
        a = 1.0 - (i + 1) / steps
        lr = int(lr_t * a); fb = int(fb_t * a); ud = int(ud_t * a); yw = int(yaw_t * a)
        with suppress(Exception):
            tello.send_rc_control(lr, fb, ud, yw)
        time.sleep(1.0 / RC_RAMP_FPS)
    with suppress(Exception):
        tello.send_rc_control(0, 0, 0, 0)

# =====================
# Blink listener → command queue (debounced rotate + confirmed double-double takeoff/land)
# =====================

class BlinkToCommandBridge:
    def __init__(self, cmd_q: queue.Queue):
        self.cmd_q = cmd_q
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, name="BlinkListener", daemon=True)

        # Debounce state (for single-eye actions)
        self._lock = threading.Lock()
        self._pending = {"right": False, "left": False}
        self._timers = {"right": None, "left": None}

        # NEW: double-double confirmation state
        self._dd_pending = False
        self._dd_t0 = 0.0
        self._dd_timer = None

    def start(self):
        self._th.start()

    def stop(self):
        self._stop.set()
        # cancel any pending timers
        with self._lock:
            for side in ("right", "left"):
                t = self._timers.get(side)
                if t is not None:
                    t.cancel()
                    self._timers[side] = None
            if self._dd_timer is not None:
                self._dd_timer.cancel()
                self._dd_timer = None
            self._dd_pending = False

    def _emit(self, typ: str, **kw):
        self.cmd_q.put({"type": typ, **kw})

    def _schedule_rotate(self, side: str):
        """Schedule a rotate after the single-eye double window, unless a second blink cancels it."""
        def fire():
            with self._lock:
                if self._pending.get(side):
                    self._pending[side] = False
                    self._timers[side] = None
                    cw = (side == "right")
                    self._emit("rotate", cw=cw)
        t = threading.Timer(CONSEC_WINDOW_S, fire)
        t.daemon = True
        t.start()
        self._timers[side] = t

    def _handle_single_eye(self, side: str):
        with self._lock:
            if self._pending.get(side):
                # Second blink within window → cancel rotate and move
                t = self._timers.get(side)
                if t is not None:
                    t.cancel()
                    self._timers[side] = None
                self._pending[side] = False
                if side == "right":
                    self._emit("move_forward")
                else:
                    self._emit("move_back")
            else:
                # First blink → mark pending and schedule rotate
                self._pending[side] = True
                self._schedule_rotate(side)

    def _handle_double_double(self):
        """Require TWO 'double' events within DOUBLE_CONFIRM_WINDOW_S to trigger takeoff/land toggle."""
        with self._lock:
            now = time.time()
            if not self._dd_pending:
                # first double → arm
                self._dd_pending = True
                self._dd_t0 = now

                # start expiry timer
                def expire():
                    with self._lock:
                        self._dd_pending = False
                        self._dd_timer = None
                self._dd_timer = threading.Timer(DOUBLE_CONFIRM_WINDOW_S, expire)
                self._dd_timer.daemon = True
                self._dd_timer.start()
            else:
                # second double within window?
                if (now - self._dd_t0) <= DOUBLE_CONFIRM_WINDOW_S:
                    # confirmed → emit toggle & reset
                    if self._dd_timer is not None:
                        self._dd_timer.cancel()
                        self._dd_timer = None
                    self._dd_pending = False
                    self._emit("toggle_takeoff_land")
                else:
                    # window expired; treat this as first of a new pair
                    if self._dd_timer is not None:
                        self._dd_timer.cancel()
                    self._dd_t0 = now
                    def expire2():
                        with self._lock:
                            self._dd_pending = False
                            self._dd_timer = None
                    self._dd_timer = threading.Timer(DOUBLE_CONFIRM_WINDOW_S, expire2)
                    self._dd_timer.daemon = True
                    self._dd_timer.start()
                    self._dd_pending = True

    def _loop(self):
        try:
            for ev in blink_events(camera_index=0, visualize=False, yield_double=True):
                if self._stop.is_set():
                    break
                eye = ev.get("eye")

                if eye == "double":
                    # NEW: require *two* double blinks to toggle takeoff/land
                    self._handle_double_double()
                    continue

                if eye == "right":
                    self._handle_single_eye("right")
                elif eye == "left":
                    self._handle_single_eye("left")
        except Exception as e:
            print(f"[BLINK] listener error: {e}")

# =====================
# Main
# =====================

def main():
    print("\n=== Blink + Gaze Drone Control (DJITelloPy) — Stabilized ===")
    print("Two DOUBLE blinks → TOGGLE TAKEOFF/LAND")
    print("Right blink → CW 45° | Left blink → CCW 45° (precision halves angle)")
    print("Two quick RIGHT → Forward | Two quick LEFT → Backward")
    print("'t' start recording | 'l' stop recording (stream must be ON)")
    print("'z' precision mode | 'm' stream on | 'n' stream off")
    print("SHIFT → one-shot smoothed gaze RC")
    print(f"Reading gaze from: {GAZE_VECTOR_FILE}\n")

    # ==== Pygame init (Display + Font + Clock) ====
    pygame.init()
    screen = pygame.display.set_mode((PG_WIDTH, PG_HEIGHT))
    pygame.display.set_caption("Tello Live (Blink+Gaze)")
    font = pygame.font.SysFont("Arial", 24)
    clock = pygame.time.Clock()

    flight = FlightLogger()

    tello = Tello()
    tello.connect()

    # Startup battery
    try:
        batt = tello.get_battery()
        flight.log(f"[STARTUP] Battery: {batt}%")
    except Exception as e:
        flight.log(f"[STARTUP] Battery read failed: {e}")

    is_flying = False
    landing_in_progress = False
    landed_banner_until = 0.0

    stream_active = False
    frame_reader = None

    # telemetry cache
    battery_pct = None
    altitude_cm = None
    last_tel_t = 0.0

    precision_mode = False

    keys = KeyController()
    recorder = VideoRecorder(logger=flight)

    cmd_q: queue.Queue = queue.Queue()
    bridge = BlinkToCommandBridge(cmd_q)
    bridge.start()

    if AUTO_TAKEOFF:
        with suppress(Exception):
            tello.takeoff()
            cooldown_after_command(tello)
            is_flying = True
            flight.start_flight()

    try:
        while True:
            now = time.time()

            # Pygame event pump (keep window responsive; allow close button)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            # Precision toggle
            if keys.consume_precision_toggle():
                precision_mode = not precision_mode
                flight.log(f"[MODE] Precision: {'ON' if precision_mode else 'OFF'}")

            # Stream controls
            if keys.consume_stream_on() and not stream_active:
                with suppress(Exception):
                    tello.streamon()
                    cooldown_after_command(tello)
                time.sleep(0.2)
                try:
                    frame_reader = tello.get_frame_read()
                except Exception as e:
                    flight.log(f"[STREAM] get_frame_read failed: {e}")
                    frame_reader = None
                if frame_reader is not None:
                    stream_active = True
                    recorder.set_reader(frame_reader)
                    flight.log("[STREAM] streamon() (display via Pygame @ 960x720)")
                else:
                    flight.log("[STREAM] streamon() but no frame reader available")

            if keys.consume_stream_off() and stream_active:
                if recorder.active:
                    recorder.stop()
                with suppress(Exception):
                    tello.streamoff()
                    cooldown_after_command(tello)
                stream_active = False
                frame_reader = None
                recorder.set_reader(None)
                flight.log("[STREAM] streamoff() (Pygame window remains open)")

            # Recording via keys 't' (start) and 'l' (stop)
            if keys.consume_rec_start():
                if not stream_active or frame_reader is None:
                    flight.log("[REC] start ignored — streaming is OFF (press 'm' first).")
                else:
                    if recorder.active:
                        flight.log("[REC] already recording.")
                    else:
                        recorder.start()
                        flight.log("[REC] start (key 't').")

            if keys.consume_rec_stop():
                if recorder.active:
                    recorder.stop()
                    flight.log("[REC] stop (key 'l').")
                else:
                    flight.log("[REC] stop ignored — not recording.")

            # Idle keep-alive
            with suppress(Exception):
                tello.send_rc_control(0, 0, 0, 0)

            # SHIFT → smoothed gaze step (ramped RC)
            if keys.consume_shift_edge():
                label = read_and_classify_once(timeout=0.7)
                if label is None:
                    flight.log("[GAZE] No valid gaze direction detected. Hover.")
                else:
                    # apply precision scaling
                    spd = int(SPEED * (PRECISION_FACTOR if precision_mode else 1.0))
                    rc = rc_from_gaze(label, spd)
                    flight.log(f"[GAZE MOVE] {label} (precision={'ON' if precision_mode else 'OFF'}) → rc{rc} ramped over {MOVE_TIME:.2f}s")
                    with suppress(Exception):
                        smoothed_gaze_step(tello, rc, total_time=MOVE_TIME)
                    cooldown_after_command(tello)

            # Execute queued blink commands
            while not cmd_q.empty():
                cmd = cmd_q.get_nowait()
                ctype = cmd["type"]

                # Move/rotate only if flying
                if ctype in ("rotate", "move_forward", "move_back"):
                    if not is_flying:
                        flight.log(f"[CMD] {ctype} ignored (drone not flying).")
                        continue

                try:
                    if ctype == "toggle_takeoff_land":
                        # Two DOUBLE blinks → toggle
                        if not is_flying:
                            flight.log("[BLINK] Double x2 → TAKEOFF")
                            tello.takeoff()
                            cooldown_after_command(tello)
                            is_flying = True
                            flight.start_flight()
                        else:
                            flight.log("[BLINK] Double x2 → LAND (tracking altitude for touchdown)")
                            tello.land()
                            cooldown_after_command(tello)
                            landing_in_progress = True

                    elif ctype == "rotate":
                        base_deg = ROTATE_DEG
                        deg = int(base_deg * (PRECISION_FACTOR if precision_mode else 1.0))
                        cw = bool(cmd.get("cw", True))
                        if cw:
                            flight.log(f"[BLINK] Right → rotate CW {deg}° (precision={'ON' if precision_mode else 'OFF'})")
                            tello.rotate_clockwise(deg)
                        else:
                            flight.log(f"[BLINK] Left → rotate CCW {deg}° (precision={'ON' if precision_mode else 'OFF'})")
                            tello.rotate_counter_clockwise(deg)
                        cooldown_after_command(tello)

                    elif ctype == "move_forward":
                        d = int(MOVE_DIST_CM * (PRECISION_FACTOR if precision_mode else 1.0))
                        flight.log(f"[BLINK] Consecutive RIGHT → move forward {d} cm (precision={'ON' if precision_mode else 'OFF'})")
                        tello.move_forward(d)
                        cooldown_after_command(tello)

                    elif ctype == "move_back":
                        d = int(MOVE_DIST_CM * (PRECISION_FACTOR if precision_mode else 1.0))
                        flight.log(f"[BLINK] Consecutive LEFT → move backward {d} cm (precision={'ON' if precision_mode else 'OFF'})")
                        tello.move_back(d)
                        cooldown_after_command(tello)

                except Exception as e:
                    flight.log(f"[CMD] error during '{ctype}': {e}")

            # Landing progress check (baro-based)
            if landing_in_progress:
                try:
                    h = tello.get_height()
                    if isinstance(h, (int, float)) and h <= 5:
                        is_flying = False
                        landing_in_progress = False
                        landed_banner_until = now + 3.0
                        with suppress(Exception):
                            tello.send_rc_control(0, 0, 0, 0)
                        flight.end_flight()
                        flight.log("✅ Landed (alt<=5cm).")
                except Exception:
                    landing_in_progress = False  # stop polling on error

            # Telemetry refresh
            if now - last_tel_t >= TELEMETRY_INTERVAL_S:
                last_tel_t = now
                with suppress(Exception):
                    battery_pct = tello.get_battery()
                with suppress(Exception):
                    altitude_cm = tello.get_height()

            # ===== Live video + overlays (Pygame) =====
            if stream_active and frame_reader is not None:
                try:
                    bgr = frame_reader.frame  # OpenCV BGR frame
                    if bgr is not None:
                        # Convert to RGB for Pygame display
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                        # Proportional fit into 960x720 with smooth scaling
                        h, w = rgb.shape[:2]
                        scale = min(PG_WIDTH / w, PG_HEIGHT / h)
                        new_w, new_h = int(w * scale), int(h * scale)

                        surf_frame = pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")
                        surf_frame = pygame.transform.smoothscale(surf_frame, (new_w, new_h))

                        screen.fill((20, 20, 20))
                        x = (PG_WIDTH - new_w) // 2
                        y = (PG_HEIGHT - new_h) // 2
                        screen.blit(surf_frame, (x, y))
                    else:
                        screen.fill((20, 20, 20))
                except Exception:
                    screen.fill((20, 20, 20))
            else:
                # No stream: subtle background
                screen.fill((20, 20, 20))

            # === HUD Overlay (Battery / Altitude / Mode / LANDED) ===
            hud_rect = pygame.Rect(8, 8, 360, 110)
            pygame.draw.rect(screen, (0, 0, 0), hud_rect)

            btxt = f"Batt: {battery_pct if battery_pct is not None else '??'}%"
            atxt = f"Alt:  {altitude_cm if altitude_cm is not None else '??'} cm"
            mtxt = f"Mode: {'Precision' if precision_mode else 'Normal'}"
            screen.blit(pygame.font.Font.render(font, btxt, True, (50, 220, 255)), (16, 16))
            screen.blit(pygame.font.Font.render(font, atxt, True, (50, 255, 120)), (16, 44))
            screen.blit(pygame.font.Font.render(font, mtxt, True, (255, 220, 50)), (16, 72))

            if now < landed_banner_until:
                banner_rect = pygame.Rect(PG_WIDTH - 210, 8, 200, 36)
                pygame.draw.rect(screen, (0, 0, 0), banner_rect)
                screen.blit(pygame.font.Font.render(font, "LANDED", True, (0, 255, 0)), (PG_WIDTH - 190, 14))

            pygame.display.flip()
            clock.tick(PG_FPS)  # target ~30 FPS display

            time.sleep(HOVER_HOLD)

    except KeyboardInterrupt:
        print("\n[CTRL+C] Landing and exiting...")
        with suppress(Exception):
            if recorder.active:
                recorder.stop()
        with suppress(Exception):
            if is_flying:
                tello.land()
                cooldown_after_command(tello)
                is_flying = False
                flight.end_flight()
    finally:
        with suppress(Exception):
            flight.close_session()
        with suppress(Exception):
            tello.end()
        with suppress(Exception):
            pygame.quit()

if __name__ == "__main__":
    main()

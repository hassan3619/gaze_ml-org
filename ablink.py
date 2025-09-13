# blink.py
import time
import os
from datetime import datetime
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
# conda activate gaze_tracking_env
# cd '/media/hn_97/VOLUME G/gaze-tracking-main'
LEFT_EYE_INDICES  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [263, 387, 385, 362, 380, 373]

def eye_aspect_ratio(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3]) + 1e-6
    ear = (A + B) / (2.0 * C)
    return float(np.clip(ear, 0.0, 1.0))

class BlinkDetector:
    def __init__(
        self,
        ema_alpha=0.35,
        baseline_beta=0.02,
        open_ratio=0.75,
        close_ratio=0.62,
        min_close_ms=60,
        refractory_ms=220,
        double_window_ms=500
    ):
        self.ema_alpha = ema_alpha
        self.baseline_beta = baseline_beta
        self.open_ratio = open_ratio
        self.close_ratio = close_ratio
        self.min_close_ms = min_close_ms
        self.refractory_ms = refractory_ms
        self.double_window_ms = double_window_ms

        self.state = {"L": "OPEN", "R": "OPEN"}
        self.ema   = {"L": None,  "R": None}
        self.base  = {"L": None,  "R": None}
        self.t_cross_down = {"L": None, "R": None}
        self.t_last_blink = {"L": -1e9, "R": -1e9}

        self.pending_eye = None
        self.pending_time = -1e9

        self.count = {"L": 0, "R": 0, "D": 0}

    def _update_ema(self, key, val):
        prev = self.ema[key]
        self.ema[key] = val if prev is None else (self.ema_alpha * prev + (1 - self.ema_alpha) * val)
        return self.ema[key]

    def _update_baseline(self, key, ear_smooth):
        if self.base[key] is None:
            self.base[key] = ear_smooth
        else:
            if ear_smooth > self.base[key] * 0.7:
                self.base[key] = (1 - self.baseline_beta) * self.base[key] + self.baseline_beta * ear_smooth

    def _per_eye_step(self, key, ear_s, now_ms, candidates):
        base = self.base[key] if self.base[key] is not None else 0.3
        thr_close = base * self.close_ratio
        thr_open  = base * self.open_ratio

        st = self.state[key]
        refractory_ok = (now_ms - self.t_last_blink[key]) >= self.refractory_ms

        if st == "OPEN":
            if ear_s < thr_close and refractory_ok:
                self.state[key] = "CLOSING"
                self.t_cross_down[key] = now_ms

        elif st in ("CLOSING", "CLOSED"):
            if ear_s < thr_close:
                self.state[key] = "CLOSED"
            if ear_s > thr_open:
                if self.t_cross_down[key] is not None:
                    duration = now_ms - self.t_cross_down[key]
                    if duration >= self.min_close_ms and refractory_ok:
                        candidates.append(key)
                self.state[key] = "OPEN"
                self.t_cross_down[key] = None

        return thr_close, thr_open

    def update(self, left_ear_raw, right_ear_raw, now_ms=None):
        if now_ms is None:
            now_ms = time.time() * 1000.0

        ev = {
            "left_blink": False, "right_blink": False, "double_blink": False,
            "L_ear": None, "R_ear": None,
            "L_thr_close": None, "L_thr_open": None,
            "R_thr_close": None, "R_thr_open": None
        }

        L_ear_s = self._update_ema("L", left_ear_raw)
        R_ear_s = self._update_ema("R", right_ear_raw)
        self._update_baseline("L", L_ear_s)
        self._update_baseline("R", R_ear_s)

        candidates = []
        L_thr_close, L_thr_open = self._per_eye_step("L", L_ear_s, now_ms, candidates)
        R_thr_close, R_thr_open = self._per_eye_step("R", R_ear_s, now_ms, candidates)

        if self.pending_eye is not None and (now_ms - self.pending_time) > self.double_window_ms and len(candidates) == 0:
            eye = self.pending_eye
            if eye == "L":
                ev["left_blink"] = True
                self.count["L"] += 1
                self.t_last_blink["L"] = self.pending_time
            else:
                ev["right_blink"] = True
                self.count["R"] += 1
                self.t_last_blink["R"] = self.pending_time
            self.pending_eye = None

        if len(candidates) >= 2 or (
            self.pending_eye is not None and
            len(candidates) == 1 and
            candidates[0] != self.pending_eye and
            (now_ms - self.pending_time) <= self.double_window_ms
        ):
            ev["double_blink"] = True
            self.count["D"] += 1
            self.t_last_blink["L"] = now_ms
            self.t_last_blink["R"] = now_ms
            self.pending_eye = None

        elif len(candidates) == 1:
            eye_now = candidates[0]
            if self.pending_eye is None:
                self.pending_eye = eye_now
                self.pending_time = now_ms
            else:
                if eye_now == self.pending_eye:
                    if eye_now == "L":
                        ev["left_blink"] = True
                        self.count["L"] += 1
                        self.t_last_blink["L"] = self.pending_time
                    else:
                        ev["right_blink"] = True
                        self.count["R"] += 1
                        self.t_last_blink["R"] = self.pending_time
                    self.pending_eye = eye_now
                    self.pending_time = now_ms
                else:
                    if (now_ms - self.pending_time) > self.double_window_ms:
                        if self.pending_eye == "L":
                            ev["left_blink"] = True
                            self.count["L"] += 1
                            self.t_last_blink["L"] = self.pending_time
                        else:
                            ev["right_blink"] = True
                            self.count["R"] += 1
                            self.t_last_blink["R"] = self.pending_time
                        self.pending_eye = eye_now
                        self.pending_time = now_ms

        ev["L_ear"], ev["R_ear"] = L_ear_s, R_ear_s
        ev["L_thr_close"], ev["L_thr_open"] = L_thr_close, L_thr_open
        ev["R_thr_close"], ev["R_thr_open"] = R_thr_close, R_thr_open
        return ev

def _pts(landmarks, indices, w, h):
    return np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices], dtype=np.float32)

# -------------------- Logging helpers --------------------
def _default_log_path():
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"blink_log_{ts}.txt"

def _open_log(log_path):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    f = open(log_path, "a", buffering=1, encoding="utf-8")  # line-buffered
    f.write(f"# BLINK LOG SESSION START {datetime.now().isoformat(timespec='seconds')}\n")
    f.write("# format: <ISO-8601 time> <EVENT>\n")
    f.flush()
    return f
# ---------------------------------------------------------

def blink_events(camera_index: int = 0, visualize: bool = False, yield_double: bool = False,
                 log_path: Optional[str] = None,
                 cooldown_ms: int = 1000):
    """
    Yields {"eye": "left"|"right", "ts": <seconds>}  (and "double" if yield_double=True)
    Also logs every event to a .txt file with ISO timestamps.

    Extras added:
      • Prints "consecutive" if two DOUBLE blinks occur within 1.3 s.
      • Prints "consecutive left" when two LEFT blinks occur within 1 s.
      • Prints "consecutive right" when two RIGHT blinks occur within 1 s.
      • Adds a cooldown after obstructions clear to avoid false blinks.
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    det = BlinkDetector(
        ema_alpha=0.35,
        baseline_beta=0.02,
        open_ratio=0.75,
        close_ratio=0.62,
        min_close_ms=60,
        refractory_ms=220,
        double_window_ms=500
    )

    # set up logging
    if log_path is None:
        log_path = _default_log_path()
    log_file = _open_log(log_path)

    L_total = R_total = D_total = 0
    last_double_ts = None         # for 1 ms double-blink "consecutive"
    last_left_ts = None           # for 1 s consecutive left
    last_right_ts = None          # for 1 s consecutive right
    consecutive_eye_window = 1.0  # seconds

    # obstruction/cooldown tracking
    was_reliable_prev = False
    ignore_until_ms = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            L_ear_raw = R_ear_raw = None
            reliable = False

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                L_pts = _pts(lm, LEFT_EYE_INDICES, w, h)
                R_pts = _pts(lm, RIGHT_EYE_INDICES, w, h)

                if np.linalg.norm(L_pts[0] - L_pts[3]) > 2 and np.linalg.norm(R_pts[0] - R_pts[3]) > 2:
                    reliable = True
                    L_ear_raw = eye_aspect_ratio(L_pts)
                    R_ear_raw = eye_aspect_ratio(R_pts)

                    if visualize:
                        cv2.polylines(frame, [L_pts.astype(int)], True, (0, 255, 255), 1, cv2.LINE_AA)
                        cv2.polylines(frame, [R_pts.astype(int)], True, (0, 255, 255), 1, cv2.LINE_AA)

            now_s  = time.time()
            now_ms = now_s * 1000.0

            # transition: obstruction cleared -> start cooldown
            if (not was_reliable_prev) and reliable:
                ignore_until_ms = now_ms + cooldown_ms
            was_reliable_prev = reliable

            if L_ear_raw is not None and R_ear_raw is not None:
                in_cooldown = now_ms < ignore_until_ms

                ev = det.update(L_ear_raw, R_ear_raw, now_ms=now_ms)
                ts = now_s  # seconds (float)
                iso_now = datetime.now().isoformat(timespec="milliseconds")

                if in_cooldown:
                    ev["double_blink"] = False
                    ev["left_blink"] = False
                    ev["right_blink"] = False

                if ev["double_blink"]:
                    D_total += 1

                    if last_double_ts is not None and (ts - last_double_ts) <= 1:
                        print("consecutive")
                        log_file.write(f"{iso_now} CONSECUTIVE\n")
                        if visualize:
                            cv2.putText(frame, "CONSECUTIVE", (24, 230),
                                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

                    log_file.write(f"{iso_now} DOUBLE\n")
                    last_double_ts = ts

                    if yield_double:
                        yield {"eye": "double", "ts": ts}

                elif ev["left_blink"]:
                    L_total += 1

                    if last_left_ts is not None and (ts - last_left_ts) <= consecutive_eye_window:
                        print("consecutive left")
                        log_file.write(f"{iso_now} CONSECUTIVE_LEFT\n")
                        if visualize:
                            cv2.putText(frame, "CONSECUTIVE LEFT", (24, 260),
                                        cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA)

                    log_file.write(f"{iso_now} LEFT\n")
                    last_left_ts = ts
                    if not in_cooldown:
                        yield {"eye": "left", "ts": ts}

                elif ev["right_blink"]:
                    R_total += 1

                    if last_right_ts is not None and (ts - last_right_ts) <= consecutive_eye_window:
                        print("consecutive right")
                        log_file.write(f"{iso_now} CONSECUTIVE_RIGHT\n")
                        if visualize:
                            cv2.putText(frame, "CONSECUTIVE RIGHT", (24, 290),
                                        cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA)

                    log_file.write(f"{iso_now} RIGHT\n")
                    last_right_ts = ts
                    if not in_cooldown:
                        yield {"eye": "right", "ts": ts}

                if visualize:
                    def put(text, y, color=(255,255,255)):
                        cv2.putText(frame, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
                    put(f"L EAR: {ev['L_ear']:.3f}  (close>{ev['L_thr_close']:.3f}, open<{ev['L_thr_open']:.3f})", 40, (255,200,0))
                    put(f"R EAR: {ev['R_ear']:.3f}  (close>{ev['R_thr_close']:.3f}, open<{ev['R_thr_open']:.3f})", 70, (200,255,0))
                    put(f"Left blinks: {L_total}   Right blinks: {R_total}   Double: {D_total}", 100, (0,255,255))

                    if in_cooldown:
                        cv2.putText(frame, "COOLDOWN (obstruction cleared)", (24, 170),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,165,255), 2, cv2.LINE_AA)
                    else:
                        if ev["double_blink"]:
                            cv2.putText(frame, "DOUBLE BLINK!", (24, 200), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)
                        elif ev["left_blink"]:
                            cv2.putText(frame, "LEFT BLINK",  (24, 140), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                        elif ev["right_blink"]:
                            cv2.putText(frame, "RIGHT BLINK", (24, 170), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
            else:
                if visualize:
                    cv2.putText(frame, "Face/eyes not reliable", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)

            if visualize:
                cv2.imshow("MediaPipe Blink Detection (robust)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        if visualize:
            cv2.destroyAllWindows()
        # footer + close
        try:
            log_file.write(f"# totals L={L_total} R={R_total} D={D_total}\n")
            log_file.write(f"# BLINK LOG SESSION END {datetime.now().isoformat(timespec='seconds')}\n")
            log_file.close()
        except Exception:
            pass

if __name__ == "__main__":
    # Visualize directly if you run: python blink.py
    # Creates a new log file in the current folder by default.
    for e in blink_events(camera_index=0, visualize=True, yield_double=True):
        print(e)

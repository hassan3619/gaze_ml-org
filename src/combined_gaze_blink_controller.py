#!/usr/bin/env python3
"""
combined_gaze_blink_controller.py
---------------------------------
Single-process controller that merges the ELG gaze pipeline with robust blink
detection (EAR + debouncer), using the eyelid landmarks produced by ELG.

This file pulls the *logic* you already use in:
- elg_run.py  -> ELG model setup + per-frame landmark extraction + gaze mapping
- blink.py    -> BlinkDetector (EMA baseline, hysteresis, refractory, coalescing)

and integrates them into one real-time control loop suitable for drone commands.

Run:
  python combined_gaze_blink_controller.py --camera_id 0 --fps 60

Press 'q' to quit.
"""

import argparse
import os
import queue
import threading
import time
import socket
import json

import cv2 as cv
import numpy as np
import coloredlogs

# ==== Drone I/O (stub; replace with your controller if desired) ====
class DroneController:
    def __init__(self, enabled=False):
        self.enabled = enabled

    def send(self, cmd: str):
        if self.enabled:
            # TODO: call your real Tello controller here
            print(f"[DRONE] {cmd}")
        else:
            print(f"[SIM ] {cmd}")

# ==== Blink logic: copied/adapted from blink.py (no import side effects) ====
def eye_aspect_ratio_from_eyelid8(eyelid_pts: np.ndarray) -> float:
    """
    Compute an EAR-like ratio from ELG eyelid landmarks (first 8 points).
    The point order in ELG is around the eyelid; we approximate vertical
    distances by pairing roughly top/bottom points and width by lateral pair.

    Pairs used (indices may be tuned for your model's order):
      vertical A: (1,5)
      vertical B: (2,6)
      width   C: (0,4)

    Returns EAR in [0,1], larger=open, smaller=closed.
    """
    p = eyelid_pts
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[6])
    C = np.linalg.norm(p[0] - p[4]) + 1e-6
    ear = (A + B) / (2.0 * C)
    return float(np.clip(ear, 0.0, 1.0))

class BlinkDetector:
    """
    Robust blink detector with:
      - Exponential smoothing (EMA) of EAR
      - Online baseline learning (open state) + hysteresis (open/close ratios)
      - Min close duration + refractory period
      - Optional double-blink coalescing (kept simple here; per-eye not needed)
    """
    def __init__(
        self,
        ema_alpha=0.35,
        baseline_beta=0.02,
        open_ratio=0.75,
        close_ratio=0.62,
        min_close_ms=90,
        refractory_ms=280,
        double_window_ms=400
    ):
        self.ema_alpha = ema_alpha
        self.baseline_beta = baseline_beta
        self.open_ratio = open_ratio
        self.close_ratio = close_ratio
        self.min_close_ms = min_close_ms
        self.refractory_ms = refractory_ms
        self.double_window_ms = double_window_ms

        self.state = "OPEN"  # OPEN, CLOSING, CLOSED
        self.ema = None
        self.base = None
        self.t_cross_down = None
        self.t_last_blink = -1e12

    def _update_ema(self, val):
        prev = self.ema
        self.ema = val if prev is None else (self.ema_alpha * prev + (1 - self.ema_alpha) * val)
        return self.ema

    def _update_baseline(self, ear_s):
        if self.base is None:
            self.base = ear_s
        else:
            if ear_s > self.base * 0.7:
                self.base = (1 - self.baseline_beta) * self.base + self.baseline_beta * ear_s

    def update(self, ear_raw, now_ms=None):
        if now_ms is None:
            now_ms = time.time() * 1000.0

        ev = {"blink": False, "ear": None, "thr_close": None, "thr_open": None}

        ear_s = self._update_ema(ear_raw)
        self._update_baseline(ear_s)

        base = self.base if self.base is not None else 0.3
        thr_close = base * self.close_ratio
        thr_open  = base * self.open_ratio
        ev["ear"] = ear_s
        ev["thr_close"] = thr_close
        ev["thr_open"]  = thr_open

        refractory_ok = (now_ms - self.t_last_blink) >= self.refractory_ms

        if self.state == "OPEN":
            if ear_s < thr_close and refractory_ok:
                self.state = "CLOSING"
                self.t_cross_down = now_ms

        elif self.state in ("CLOSING", "CLOSED"):
            if ear_s < thr_close:
                self.state = "CLOSED"
            if ear_s > thr_open:
                if self.t_cross_down is not None:
                    dur = now_ms - self.t_cross_down
                    if dur >= self.min_close_ms and refractory_ok:
                        ev["blink"] = True
                        self.t_last_blink = now_ms
                self.state = "OPEN"
                self.t_cross_down = None

        return ev

# ==== ELG pipeline imports (from your project) ====
import tensorflow as tf
from datasources import Video, Webcam
from models import ELG
import util.gaze

# ==== Combined orchestrator (based on elg_run.py structure) ====
def main():
    parser = argparse.ArgumentParser(description='ELG gaze + blink controller')
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--camera_id', type=int, default=0)
    parser.add_argument('--from_video', type=str, default=None)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--fullscreen', action='store_true')
    parser.add_argument('--record_video', type=str, default=None)
    parser.add_argument('-v', type=str, default='info', choices=['debug','info','warning','error','critical'])
    args = parser.parse_args()

    coloredlogs.install(
        datefmt='%d/%m %H:%M',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level=args.v.upper(),
    )

    # UDP (optional: mirrors your elg_run.py broadcast)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    DEST = ("127.0.0.1", 50555)

    # TF session (GPU allow_growth)
    from tensorflow.python.client import device_lib
    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    gpu_available = False
    try:
        gpus = [d for d in device_lib.list_local_devices(config=session_config) if d.device_type == 'GPU']
        gpu_available = len(gpus) > 0
    except:
        pass

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Session(config=session_config) as session:
        # Data source
        if args.from_video:
            data_source = Video(
                args.from_video,
                tensorflow_session=session, batch_size=2,
                data_format='NCHW' if gpu_available else 'NHWC',
                eye_image_shape=(108, 180)
            )
        else:
            data_source = Webcam(
                tensorflow_session=session, batch_size=2,
                camera_id=args.camera_id, fps=args.fps,
                data_format='NCHW' if gpu_available else 'NHWC',
                eye_image_shape=(36, 60)
            )

        # Model
        model = ELG(
            session, train_data={'videostream': data_source},
            first_layer_stride=(3 if args.from_video else 1),
            num_modules=(3 if args.from_video else 2),
            num_feature_maps=(64 if args.from_video else 32),
            learning_schedule=[{'loss_terms_to_optimize': {'dummy': ['hourglass','radius']}}],
        )

        # Blink + control
        blink = BlinkDetector()
        drone = DroneController(enabled=False)  # set True after testing

        # Visualization
        inferred_q = queue.Queue()
        def _visualize():
            last_frame_index = 0
            gaze_histories = []
            if args.fullscreen:
                cv.namedWindow('vis', cv.WND_PROP_FULLSCREEN)
                cv.setWindowProperty('vis', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

            while True:
                if inferred_q.empty():
                    next_index = last_frame_index + 1
                    if next_index in data_source._frames:
                        fr = data_source._frames[next_index]
                        if 'faces' in fr and len(fr['faces']) == 0 and not args.headless:
                            cv.imshow('vis', fr['bgr'])
                        last_frame_index = next_index
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        return
                    continue

                out = inferred_q.get()
                bgr = None
                batch_size = out['heatmaps'].shape[0]
                for j in range(batch_size):
                    fidx = out['frame_index'][j]
                    if fidx not in data_source._frames:
                        continue
                    frame = data_source._frames[fidx]

                    # Usability checks like in elg_run.py
                    heatmaps_amax = np.amax(out['heatmaps'][j, :].reshape(-1, 18), axis=0)
                    can_use_eye = np.all(heatmaps_amax > 0.7)
                    can_use_eyelid = np.all(heatmaps_amax[0:8] > 0.75)
                    can_use_iris = np.all(heatmaps_amax[8:16] > 0.8)

                    eye_index = out['eye_index'][j]
                    bgr = frame['bgr']
                    eye = frame['eyes'][eye_index]
                    eye_image = eye['image']
                    eye_side = eye['side']
                    eye_landmarks = out['landmarks'][j, :]
                    eye_radius = out['radius'][j][0]
                    if eye_side == 'left':
                        eye_landmarks[:, 0] = eye_image.shape[1] - eye_landmarks[:, 0]
                        eye_image = np.fliplr(eye_image)

                    # Transform to frame coords (same as elg_run.py)
                    eye_landmarks = np.concatenate([eye_landmarks,
                                                    [[eye_landmarks[-1, 0] + eye_radius,
                                                      eye_landmarks[-1, 1]]]])
                    eye_landmarks = np.asmatrix(np.pad(eye_landmarks, ((0, 0), (0, 1)),
                                                       'constant', constant_values=1.0))
                    eye_landmarks = (eye_landmarks * eye['inv_landmarks_transform_mat'].T)[:, :2]
                    eye_landmarks = np.asarray(eye_landmarks)
                    eyelid_landmarks = eye_landmarks[0:8, :]
                    iris_landmarks   = eye_landmarks[8:16, :]
                    iris_centre      = eye_landmarks[16, :]
                    eyeball_centre   = eye_landmarks[17, :]
                    eyeball_radius   = np.linalg.norm(eye_landmarks[18, :] - eye_landmarks[17, :])

                    # Gaze direction (theta, phi) like in elg_run.py
                    i_x0, i_y0 = iris_centre
                    e_x0, e_y0 = eyeball_centre
                    theta = -np.arcsin(np.clip((i_y0 - e_y0) / eyeball_radius, -1.0, 1.0))
                    phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_radius * -np.cos(theta)),
                                            -1.0, 1.0))
                    current_gaze = np.array([theta, phi], dtype=np.float32)
                    gx = float(np.clip(phi / 0.35,  -1.0, 1.0))  # horiz
                    gy = float(np.clip(theta / 0.35, -1.0, 1.0)) # vert

                    # Blink EAR from ELG eyelid landmarks
                    ear = None
                    if can_use_eyelid:
                        ear = eye_aspect_ratio_from_eyelid8(eyelid_landmarks)
                        ev = blink.update(ear, now_ms=time.time() * 1000.0)

                        # Simple policy: if blink, issue a command based on gaze
                        if ev["blink"]:
                            if gx < -0.15:
                                drone.send("ccw 15")
                            elif gx > 0.15:
                                drone.send("cw 15")
                            elif gy < -0.15:
                                drone.send("up 20")
                            elif gy > 0.15:
                                drone.send("down 20")
                            else:
                                drone.send("forward 20")

                    # Debug overlay
                    if not args.headless:
                        cv.putText(bgr, f"gx:{gx:+.2f} gy:{gy:+.2f}", (10, 30),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                        if ear is not None:
                            cv.putText(bgr, f"EAR:{ear:.3f}", (10, 60),
                                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                        # Draw gaze ray
                        util.gaze.draw_gaze(bgr, iris_centre, current_gaze, length=120.0, thickness=1)
                        cv.imshow('vis', bgr)

                    # UDP broadcast (optional)
                    msg = {"t": time.time(), "gx": gx, "gy": gy, "ok": bool(can_use_eye)}
                    try:
                        sock.sendto(json.dumps(msg).encode("utf-8"), DEST)
                    except Exception:
                        pass

                    if cv.waitKey(1) & 0xFF == ord('q'):
                        return

        vis_thread = threading.Thread(target=_visualize, daemon=True)
        vis_thread.start()

        # Inference loop
        infer = model.inference_generator()
        while True:
            output = next(infer)
            for frame_index in np.unique(output['frame_index']):
                if frame_index not in data_source._frames:
                    continue
                frame = data_source._frames[frame_index]
                if 'inference' in frame['time']:
                    frame['time']['inference'] += output['inference_time']
                else:
                    frame['time']['inference'] = output['inference_time']
            inferred_q.put_nowait(output)

            if not vis_thread.is_alive() or not data_source._open:
                break

if __name__ == "__main__":
    main()

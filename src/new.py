#!/usr/bin/env python3
"""Main script for gaze direction inference from webcam feed."""
import argparse
import os
import queue
import threading
import time
import json



import coloredlogs
import cv2 as cv
import numpy as np
import tensorflow as tf

from datasources import Video, Webcam
from models import ELG
import util.gaze
import scipy.optimize

# conda activate gazeml

if __name__ == '__main__':

    # Set global log level
    parser = argparse.ArgumentParser(description='Demonstration of landmarks localization.')
    parser.add_argument('-v', type=str, help='logging level', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    parser.add_argument('--from_video', type=str, help='Use this video path instead of webcam')
    parser.add_argument('--record_video', type=str, help='Output path of video of demonstration.')
    parser.add_argument('--fullscreen', action='store_true')
    parser.add_argument('--headless', action='store_true')

    parser.add_argument('--fps', type=int, default=60, help='Desired sampling rate of webcam')
    parser.add_argument('--camera_id', type=int, default=0, help='ID of webcam to use')
    parser.add_argument('--calibrate', action='store_true', help='Run 9-point screen calibration before normal run')
    parser.add_argument('--save_calib', type=str, default='gaze_calib.npz', help='Path to save calibration mapping')
    parser.add_argument('--load_calib', type=str, help='Path to load existing calibration mapping')
    parser.add_argument('--screen_w', type=int, default=1920, help='Screen width in pixels')
    parser.add_argument('--screen_h', type=int, default=1080, help='Screen height in pixels')

    args = parser.parse_args()
    coloredlogs.install(
        datefmt='%d/%m %H:%M',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level=args.v.upper(),
    )

    # Check if GPU is available
    from tensorflow.python.client import device_lib
    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    gpu_available = False
    try:
        gpus = [d for d in device_lib.list_local_devices(config=session_config)
                if d.device_type == 'GPU']
        gpu_available = len(gpus) > 0
    except:
        pass

    # Initialize Tensorflow session
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Session(config=session_config) as session:

        # Declare some parameters
        batch_size = 2

        # Define webcam stream data source
        # Change data_format='NHWC' if not using CUDA
        if args.from_video:
            assert os.path.isfile(args.from_video)
            data_source = Video(args.from_video,
                                tensorflow_session=session, batch_size=batch_size,
                                data_format='NCHW' if gpu_available else 'NHWC',
                                eye_image_shape=(108, 180))
        else:
            data_source = Webcam(tensorflow_session=session, batch_size=batch_size,
                                 camera_id=args.camera_id, fps=args.fps,
                                 data_format='NCHW' if gpu_available else 'NHWC',
                                 eye_image_shape=(36, 60))

        # Define model
        if args.from_video:
            model = ELG(
                session, train_data={'videostream': data_source},
                first_layer_stride=3,
                num_modules=3,
                num_feature_maps=64,
                learning_schedule=[
                    {
                        'loss_terms_to_optimize': {'dummy': ['hourglass', 'radius']},
                    },
                ],
            )
        else:
            model = ELG(
                session, train_data={'videostream': data_source},
                first_layer_stride=1,
                num_modules=2,
                num_feature_maps=32,
                learning_schedule=[
                    {
                        'loss_terms_to_optimize': {'dummy': ['hourglass', 'radius']},
                    },
                ],
            )

        # Record output frames to file if requested
        if args.record_video:
            video_out = None
            video_out_queue = queue.Queue()
            video_out_should_stop = False
            video_out_done = threading.Condition()

            def _record_frame():
                global video_out
                last_frame_time = None
                out_fps = 30
                out_frame_interval = 1.0 / out_fps
                while not video_out_should_stop:
                    frame_index = video_out_queue.get()
                    if frame_index is None:
                        break
                    assert frame_index in data_source._frames
                    frame = data_source._frames[frame_index]['bgr']
                    h, w, _ = frame.shape
                    if video_out is None:
                        video_out = cv.VideoWriter(
                            args.record_video, cv.VideoWriter_fourcc(*'H264'),
                            out_fps, (w, h),
                        )
                    now_time = time.time()
                    if last_frame_time is not None:
                        time_diff = now_time - last_frame_time
                        while time_diff > 0.0:
                            video_out.write(frame)
                            time_diff -= out_frame_interval
                    last_frame_time = now_time
                video_out.release()
                with video_out_done:
                    video_out_done.notify_all()
            record_thread = threading.Thread(target=_record_frame, name='record')
            record_thread.daemon = True
            record_thread.start()

        # Begin visualization thread
        inferred_stuff_queue = queue.Queue()
                # === Calibration additions start ===
        class Poly2Calibrator:
            """
            2D polynomial (quadratic) mapping:
            [1, th, ph, th^2, th*ph, ph^2] -> x  and  -> y
            Fit with least squares, then predict screen pixels from [theta, phi].
            """
            def __init__(self):
                self.coef_x = None
                self.coef_y = None

            @staticmethod
            def _phi(th, ph):
                return np.array([1.0, th, ph, th*th, th*ph, ph*ph], dtype=np.float64)

            def fit(self, thetas, phis, xs, ys):
                # Build design matrix
                A = np.stack([self._phi(t, p) for t, p in zip(thetas, phis)], axis=0)  # [N,6]
                bx = np.asarray(xs, dtype=np.float64)
                by = np.asarray(ys, dtype=np.float64)
                self.coef_x, *_ = np.linalg.lstsq(A, bx, rcond=None)
                self.coef_y, *_ = np.linalg.lstsq(A, by, rcond=None)

            def predict(self, th, ph):
                f = self._phi(th, ph)
                x = float(f @ self.coef_x) if self.coef_x is not None else None
                y = float(f @ self.coef_y) if self.coef_y is not None else None
                return x, y

            def save(self, path):
                np.savez(path, coef_x=self.coef_x, coef_y=self.coef_y)

            def load(self, path):
                data = np.load(path)
                self.coef_x = data['coef_x']
                self.coef_y = data['coef_y']


        calib = Poly2Calibrator()
        calibrated = False
        current_gaze = None  # (theta, phi) each frame (smoothed)
        pred_point = None    # (x, y) on screen after mapping

        # 9 target points in 3x3 grid (normalized), scaled later by screen_w/h
        grid = [(0.15,0.15),(0.50,0.15),(0.85,0.15),
                (0.15,0.50),(0.50,0.50),(0.85,0.50),
                (0.15,0.85),(0.50,0.85),(0.85,0.85)]

        # State for calibration capture
        calib_enabled = False
        calib_idx = 0
        samples_per_target = 25          # frames to average per dot
        sampled = 0
        buf_th, buf_ph = [], []
        th_list, ph_list, x_list, y_list = [], [], [], []

        def draw_target(canvas, cx, cy, color=(0, 255, 255)):
            cv.circle(canvas, (int(cx), int(cy)), 16, color, 2, lineType=cv.LINE_AA)
            cv.circle(canvas, (int(cx), int(cy)), 3, color, -1, lineType=cv.LINE_AA)

        # === Calibration additions end ===

        def _visualize_output():
            last_frame_index = 0
            last_frame_time = time.time()
            fps_history = []
            all_gaze_histories = []
            global current_gaze, pred_point, calibrated, calib_enabled, calib_idx, sampled, buf_th, buf_ph, th_list, ph_list, x_list, y_list
            if args.load_calib:
                calib.load(args.load_calib)
                calibrated = True

            if args.fullscreen:
                cv.namedWindow('vis', cv.WND_PROP_FULLSCREEN)
                cv.setWindowProperty('vis', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

            while True:
                # If no output to visualize, show unannotated frame
                if inferred_stuff_queue.empty():
                    next_frame_index = last_frame_index + 1
                    if next_frame_index in data_source._frames:
                        next_frame = data_source._frames[next_frame_index]
                        if 'faces' in next_frame and len(next_frame['faces']) == 0:
                            if not args.headless:
                                cv.imshow('vis', next_frame['bgr'])
                            if args.record_video:
                                video_out_queue.put_nowait(next_frame_index)
                            last_frame_index = next_frame_index
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        return
                    continue

                # Get output from neural network and visualize
                output = inferred_stuff_queue.get()
                bgr = None
                for j in range(batch_size):
                    frame_index = output['frame_index'][j]
                    if frame_index not in data_source._frames:
                        continue
                    frame = data_source._frames[frame_index]

                    # Decide which landmarks are usable
                    heatmaps_amax = np.amax(output['heatmaps'][j, :].reshape(-1, 18), axis=0)
                    can_use_eye = np.all(heatmaps_amax > 0.7)
                    can_use_eyelid = np.all(heatmaps_amax[0:8] > 0.75)
                    can_use_iris = np.all(heatmaps_amax[8:16] > 0.8)

                    start_time = time.time()
                    eye_index = output['eye_index'][j]
                    bgr = frame['bgr']
                    eye = frame['eyes'][eye_index]
                    eye_image = eye['image']
                    eye_side = eye['side']
                    eye_landmarks = output['landmarks'][j, :]
                    eye_radius = output['radius'][j][0]
                    if eye_side == 'left':
                        eye_landmarks[:, 0] = eye_image.shape[1] - eye_landmarks[:, 0]
                        eye_image = np.fliplr(eye_image)

                    # Embed eye image and annotate for picture-in-picture
                    eye_upscale = 2
                    eye_image_raw = cv.cvtColor(cv.equalizeHist(eye_image), cv.COLOR_GRAY2BGR)
                    eye_image_raw = cv.resize(eye_image_raw, (0, 0), fx=eye_upscale, fy=eye_upscale)
                    eye_image_annotated = np.copy(eye_image_raw)
                    if can_use_eyelid:
                        cv.polylines(
                            eye_image_annotated,
                            [np.round(eye_upscale*eye_landmarks[0:8]).astype(np.int32)
                                                                     .reshape(-1, 1, 2)],
                            isClosed=True, color=(255, 255, 0), thickness=1, lineType=cv.LINE_AA,
                        )
                    if can_use_iris:
                        cv.polylines(
                            eye_image_annotated,
                            [np.round(eye_upscale*eye_landmarks[8:16]).astype(np.int32)
                                                                      .reshape(-1, 1, 2)],
                            isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv.LINE_AA,
                        )
                        cv.drawMarker(
                            eye_image_annotated,
                            tuple(np.round(eye_upscale*eye_landmarks[16, :]).astype(np.int32)),
                            color=(0, 255, 255), markerType=cv.MARKER_CROSS, markerSize=4,
                            thickness=1, line_type=cv.LINE_AA,
                        )
                    face_index = int(eye_index / 2)
                    eh, ew, _ = eye_image_raw.shape
                    v0 = face_index * 2 * eh
                    v1 = v0 + eh
                    v2 = v1 + eh
                    u0 = 0 if eye_side == 'left' else ew
                    u1 = u0 + ew
                    bgr[v0:v1, u0:u1] = eye_image_raw
                    bgr[v1:v2, u0:u1] = eye_image_annotated

                    # Visualize preprocessing results
                    frame_landmarks = (frame['smoothed_landmarks']
                                       if 'smoothed_landmarks' in frame
                                       else frame['landmarks'])
                    for f, face in enumerate(frame['faces']):
                        for landmark in frame_landmarks[f][:-1]:
                            cv.drawMarker(bgr, tuple(np.round(landmark).astype(np.int32)),
                                          color=(0, 0, 255), markerType=cv.MARKER_STAR,
                                          markerSize=2, thickness=1, line_type=cv.LINE_AA)
                        cv.rectangle(
                            bgr, tuple(np.round(face[:2]).astype(np.int32)),
                            tuple(np.round(np.add(face[:2], face[2:])).astype(np.int32)),
                            color=(0, 255, 255), thickness=1, lineType=cv.LINE_AA,
                        )

                    # Transform predictions
                    eye_landmarks = np.concatenate([eye_landmarks,
                                                    [[eye_landmarks[-1, 0] + eye_radius,
                                                      eye_landmarks[-1, 1]]]])
                    eye_landmarks = np.asmatrix(np.pad(eye_landmarks, ((0, 0), (0, 1)),
                                                       'constant', constant_values=1.0))
                    eye_landmarks = (eye_landmarks *
                                     eye['inv_landmarks_transform_mat'].T)[:, :2]
                    eye_landmarks = np.asarray(eye_landmarks)
                    eyelid_landmarks = eye_landmarks[0:8, :]
                    iris_landmarks = eye_landmarks[8:16, :]
                    iris_centre = eye_landmarks[16, :]
                    eyeball_centre = eye_landmarks[17, :]
                    eyeball_radius = np.linalg.norm(eye_landmarks[18, :] -
                                                    eye_landmarks[17, :])

                    # Smooth and visualize gaze direction
                    num_total_eyes_in_frame = len(frame['eyes'])
                    if len(all_gaze_histories) != num_total_eyes_in_frame:
                        all_gaze_histories = [list() for _ in range(num_total_eyes_in_frame)]
                    gaze_history = all_gaze_histories[eye_index]
                    if can_use_eye:
                        # Visualize landmarks
                        cv.drawMarker(  # Eyeball centre
                            bgr, tuple(np.round(eyeball_centre).astype(np.int32)),
                            color=(0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=4,
                            thickness=1, line_type=cv.LINE_AA,
                        )
                     #   cv.circle(  # Eyeball outline
                      #       bgr, tuple(np.round(eyeball_centre).astype(np.int32)),
                       #      int(np.round(eyeball_radius)), color=(0, 255, 0),
                        #     thickness=1, lineType=cv.LINE_AA,
                         #)
                        
                        # Draw "gaze"
                        from models.elg import estimate_gaze_from_landmarks
                        current_gaze = estimate_gaze_from_landmarks(
                             iris_landmarks, iris_centre, eyeball_centre, eyeball_radius)
                        i_x0, i_y0 = iris_centre
                        e_x0, e_y0 = eyeball_centre
                        theta = -np.arcsin(np.clip((i_y0 - e_y0) / eyeball_radius, -1.0, 1.0))
                        phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_radius * -np.cos(theta)),
                                                -1.0, 1.0))
                        current_gaze = np.array([theta, phi])
                        gaze_history.append(current_gaze)
                        gaze_history_max_len = 10
                        if len(gaze_history) > gaze_history_max_len:
                            gaze_history = gaze_history[-gaze_history_max_len:]

                        smoothed = np.mean(gaze_history, axis=0)  # [theta, phi]
                        current_gaze = smoothed  # expose for calibration/pointing

                        # Draw the gaze vector (as before)
                        util.gaze.draw_gaze(bgr, iris_centre, smoothed, length=120.0, thickness=1)

                        # If we have a calibration, predict the screen point and visualize on the video
                        if calibrated:
                            px, py = calib.predict(smoothed[0], smoothed[1])
                            pred_point = (px, py)
                            # Show a small HUD crosshair at bottom-right with predicted pixel (text only)
                            label = f"Pred: ({int(px):4d}, {int(py):4d})"
                            cv.putText(bgr, label, org=(20, 30),
                                    fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.6,
                                    color=(0, 0, 0), thickness=2, lineType=cv.LINE_AA)
                            cv.putText(bgr, label, org=(20, 30),
                                    fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.6,
                                    color=(255, 255, 255), thickness=1, lineType=cv.LINE_AA)

                        
                    else:
                        gaze_history.clear()

                    if can_use_eyelid:
                        cv.polylines(
                            bgr, [np.round(eyelid_landmarks).astype(np.int32).reshape(-1, 1, 2)],
                            isClosed=True, color=(255, 255, 0), thickness=1, lineType=cv.LINE_AA,
                        )

                    if can_use_iris:
                        cv.polylines(
                            bgr, [np.round(iris_landmarks).astype(np.int32).reshape(-1, 1, 2)],
                            isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv.LINE_AA,
                        )
                        cv.drawMarker(
                            bgr, tuple(np.round(iris_centre).astype(np.int32)),
                            color=(0, 255, 255), markerType=cv.MARKER_CROSS, markerSize=4,
                            thickness=1, line_type=cv.LINE_AA,
                        )

                    dtime = 1e3*(time.time() - start_time)
                    if 'visualization' not in frame['time']:
                        frame['time']['visualization'] = dtime
                    else:
                        frame['time']['visualization'] += dtime

                    def _dtime(before_id, after_id):
                        return int(1e3 * (frame['time'][after_id] - frame['time'][before_id]))

                    def _dstr(title, before_id, after_id):
                        return '%s: %dms' % (title, _dtime(before_id, after_id))

                    if eye_index == len(frame['eyes']) - 1:
                        # Calculate timings
                        frame['time']['after_visualization'] = time.time()
                        fps = int(np.round(1.0 / (time.time() - last_frame_time)))
                        fps_history.append(fps)
                        if len(fps_history) > 60:
                            fps_history = fps_history[-60:]
                        fps_str = '%d FPS' % np.mean(fps_history)
                        last_frame_time = time.time()
                        fh, fw, _ = bgr.shape
                        cv.putText(bgr, fps_str, org=(fw - 110, fh - 20),
                                   fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.8,
                                   color=(0, 0, 0), thickness=1, lineType=cv.LINE_AA)
                        cv.putText(bgr, fps_str, org=(fw - 111, fh - 21),
                                   fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.79,
                                   color=(255, 255, 255), thickness=1, lineType=cv.LINE_AA)
                        if not args.headless:
                            cv.imshow('vis', bgr)
                        last_frame_index = frame_index
                                                # === Calibration UI / capture ===
                        if args.calibrate and not calibrated:
                            calib_enabled = True

                        if calib_enabled:
                            # Create an overlay for the target on top of the camera feed
                            overlay = bgr.copy()
                            gw, gh = grid[calib_idx]
                            cx = int(args.screen_w * gw)
                            cy = int(args.screen_h * gh)
                            draw_target(overlay, cx, cy, (0, 255, 255))
                            alpha = 0.35
                            bgr = cv.addWeighted(overlay, alpha, bgr, 1 - alpha, 0)

                            # If we have a valid current_gaze this frame, accumulate
                            if current_gaze is not None and can_use_eye:
                                buf_th.append(float(current_gaze[0]))
                                buf_ph.append(float(current_gaze[1]))
                                sampled += 1

                            # When enough frames collected for this target -> store average and move on
                            if sampled >= samples_per_target:
                                th_list.append(float(np.mean(buf_th)))
                                ph_list.append(float(np.mean(buf_ph)))
                                x_list.append(cx)
                                y_list.append(cy)
                                buf_th, buf_ph = [], []
                                sampled = 0
                                calib_idx += 1

                                # Progress text
                                cv.putText(bgr, f"Captured {len(th_list)}/{len(grid)} points",
                                        org=(20, 60), fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.6,
                                        color=(0, 0, 0), thickness=2, lineType=cv.LINE_AA)
                                cv.putText(bgr, f"Captured {len(th_list)}/{len(grid)} points",
                                        org=(20, 60), fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.6,
                                        color=(0, 255, 255), thickness=1, lineType=cv.LINE_AA)

                            # If done with all targets, fit and save
                            if calib_idx >= len(grid):
                                calib.fit(th_list, ph_list, x_list, y_list)
                                calib.save(args.save_calib)
                                calibrated = True
                                calib_enabled = False
                                # Quick confirmation
                                cv.putText(bgr, f"Calibration saved to {args.save_calib}",
                                        org=(20, 90), fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.6,
                                        color=(0, 0, 0), thickness=2, lineType=cv.LINE_AA)
                                cv.putText(bgr, f"Calibration saved to {args.save_calib}",
                                        org=(20, 90), fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.6,
                                        color=(0, 255, 0), thickness=1, lineType=cv.LINE_AA)
                        # === end calibration chunk ===

                        # Define the file where the gaze vector will be written
                        gaze_vector_file = "gaze_vector.txt"

                        # After calculating the gaze vector, write the gaze data (theta, phi) to the file
                        gaze_data = f"{current_gaze[0]}, {current_gaze[1]}\n"  # Example using current gaze
                        # Alternatively, use smoothed_gaze
                        # gaze_data = f"{smoothed_gaze[0]}, {smoothed_gaze[1]}\n"

                        with open(gaze_vector_file, 'a') as f:
                            f.write(gaze_data)

                        
                        # Record frame?
                        if args.record_video:
                            video_out_queue.put_nowait(frame_index)

                        # Quit?
                        if cv.waitKey(1) & 0xFF == ord('q'):
                            return

                        # Print timings
                        if frame_index % 60 == 0:
                            latency = _dtime('before_frame_read', 'after_visualization')
                            processing = _dtime('after_frame_read', 'after_visualization')
                            timing_string = ', '.join([
                                _dstr('read', 'before_frame_read', 'after_frame_read'),
                                _dstr('preproc', 'after_frame_read', 'after_preprocessing'),
                                'infer: %dms' % int(frame['time']['inference']),
                                'vis: %dms' % int(frame['time']['visualization']),
                                'proc: %dms' % processing,
                                'latency: %dms' % latency,
                            ])
                            print('%08d [%s] %s' % (frame_index, fps_str, timing_string))

        visualize_thread = threading.Thread(target=_visualize_output, name='visualization')
        visualize_thread.daemon = True
        visualize_thread.start()

        # Do inference forever
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
            inferred_stuff_queue.put_nowait(output)

            if not visualize_thread.isAlive():
                break

            if not data_source._open:
                break

        # Close video recording
        if args.record_video and video_out is not None:
            video_out_should_stop = True
            video_out_queue.put_nowait(None)
            with video_out_done:
                video_out_done.wait()





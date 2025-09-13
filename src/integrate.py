import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# ---- ELG MODEL IMPORTS ----
from models import ELG
from datasources import Webcam
import util.gaze

# ---- MediaPipe Setup ----
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # MediaPipe left eye for EAR
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380] # MediaPipe right eye for EAR

def compute_EAR(eye_landmarks):
    # eye_landmarks: list of 6 mediapipe landmarks
    p2_p6 = np.linalg.norm(np.array([eye_landmarks[1].x, eye_landmarks[1].y]) - np.array([eye_landmarks[5].x, eye_landmarks[5].y]))
    p3_p5 = np.linalg.norm(np.array([eye_landmarks[2].x, eye_landmarks[2].y]) - np.array([eye_landmarks[4].x, eye_landmarks[4].y]))
    p1_p4 = np.linalg.norm(np.array([eye_landmarks[0].x, eye_landmarks[0].y]) - np.array([eye_landmarks[3].x, eye_landmarks[3].y]))
    ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
    return ear

BLINK_THRESHOLD = 0.23
CONSEC_FRAMES = 2
blink_count = 0
frame_counter = 0

# ---- ELG Setup ----
session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=session_config)
batch_size = 2
data_source = Webcam(tensorflow_session=session, batch_size=batch_size,
                     camera_id=0, fps=30,
                     data_format='NHWC',
                     eye_image_shape=(36, 60))
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
infer = model.inference_generator()

cv2.namedWindow("Gaze and Blink", cv2.WINDOW_NORMAL)

while True:
    # ----- Get Frame -----
    output = next(infer)
    frame_index = output['frame_index'][0]
    frame = data_source._frames[frame_index]
    bgr = frame['bgr'].copy()
    h, w = bgr.shape[:2]

    # ----- ELG Gaze & Iris -----
    eye_landmarks = output['landmarks'][0, :]
    eye_radius = output['radius'][0][0]
    eye = frame['eyes'][output['eye_index'][0]]
    eye_image = eye['image']
    eye_side = eye['side']
    if eye_side == 'left':
        eye_landmarks[:, 0] = eye_image.shape[1] - eye_landmarks[:, 0]
        eye_image = np.fliplr(eye_image)
    eye_landmarks = np.concatenate([eye_landmarks,
                                    [[eye_landmarks[-1, 0] + eye_radius,
                                      eye_landmarks[-1, 1]]]])
    eye_landmarks = np.asmatrix(np.pad(eye_landmarks, ((0, 0), (0, 1)), 'constant', constant_values=1.0))
    eye_landmarks = (eye_landmarks * eye['inv_landmarks_transform_mat'].T)[:, :2]
    eye_landmarks = np.asarray(eye_landmarks)
    eyelid_landmarks = eye_landmarks[0:8, :]
    iris_landmarks = eye_landmarks[8:16, :]
    iris_centre = eye_landmarks[16, :]
    eyeball_centre = eye_landmarks[17, :]
    eyeball_radius = np.linalg.norm(eye_landmarks[18, :] - eye_landmarks[17, :])

    # Draw ELG landmarks
    cv2.polylines(bgr, [np.round(eyelid_landmarks).astype(np.int32).reshape(-1, 1, 2)],
                  isClosed=True, color=(255, 255, 0), thickness=1)
    cv2.polylines(bgr, [np.round(iris_landmarks).astype(np.int32).reshape(-1, 1, 2)],
                  isClosed=True, color=(0, 255, 255), thickness=1)
    cv2.drawMarker(bgr, tuple(np.round(iris_centre).astype(np.int32)),
                   color=(0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=4, thickness=2)
    cv2.drawMarker(bgr, tuple(np.round(eyeball_centre).astype(np.int32)),
                   color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=4, thickness=2)

    # ----- Gaze Visualization -----
    theta = -np.arcsin(np.clip((iris_centre[1] - eyeball_centre[1]) / eyeball_radius, -1.0, 1.0))
    phi = np.arcsin(np.clip((iris_centre[0] - eyeball_centre[0]) / (eyeball_radius * -np.cos(theta)), -1.0, 1.0))
    current_gaze = np.array([theta, phi])
    util.gaze.draw_gaze(bgr, iris_centre, current_gaze, length=120.0, thickness=2)

    # ---- MediaPipe Blink Detection ----
    frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    left_EAR, right_EAR = 1.0, 1.0
    blink_detected = False
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye = [face_landmarks.landmark[i] for i in LEFT_EYE_INDICES]
            right_eye = [face_landmarks.landmark[i] for i in RIGHT_EYE_INDICES]
            left_EAR = compute_EAR(left_eye)
            right_EAR = compute_EAR(right_eye)
            if left_EAR < BLINK_THRESHOLD and right_EAR < BLINK_THRESHOLD:
                frame_counter += 1
            else:
                if frame_counter >= CONSEC_FRAMES:
                    blink_count += 1
                    blink_detected = True
                frame_counter = 0
    # Display blink info
    cv2.putText(bgr, f"Blinks: {blink_count}", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(bgr, f"Left EAR: {left_EAR:.3f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.putText(bgr, f"Right EAR: {right_EAR:.3f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    if blink_detected:
        cv2.putText(bgr, "Blink!", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Gaze and Blink", bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
session.close()
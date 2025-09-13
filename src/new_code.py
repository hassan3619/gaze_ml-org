import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                   refine_landmarks=True, min_detection_confidence=0.5)

# Call in your data frame loop:
results = face_mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

if results.multi_face_landmarks:
    landmarks = results.multi_face_landmarks[0]
    # Use iris landmarks: 468–478

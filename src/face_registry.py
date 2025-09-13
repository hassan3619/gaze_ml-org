# face_registry.py
import os, json, time, cv2, numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from ultralytics import YOLO
import insightface
from insightface.app import FaceAnalysis

DB_DIR = Path("profiles")
DB_DIR.mkdir(exist_ok=True)
FACE_DIR = DB_DIR / "faces"
FACE_DIR.mkdir(exist_ok=True)
DB_PATH = DB_DIR / "db.json"

def _load_db() -> Dict[str, List[List[float]]]:
    if DB_PATH.exists():
        with open(DB_PATH, "r") as f:
            return json.load(f)
    return {}

def _save_db(db: Dict[str, List[List[float]]]):
    with open(DB_PATH, "w") as f:
        json.dump(db, f)

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-9
    return v / n

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_normalize(a), _normalize(b)))

class FaceEncoder:
    def __init__(self):
        # One engine for BOTH detection (SCRFD) and embeddings (ArcFace)
        self.app = FaceAnalysis(name="buffalo_l")
        try:
            self.app.prepare(ctx_id=0)  # GPU if available
        except Exception:
            self.app.prepare(ctx_id=-1)  # CPU fallback


    def detect(self, frame: np.ndarray, conf: float = 0.3):
        """Return list of (x1, y1, x2, y2, score) using InsightFace detector."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb)
        out = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int).tolist()
            score = float(f.det_score)
            if score >= conf:
                out.append((x1, y1, x2, y2, score))
        return out
    def embed(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb)
        if not faces:
            return None
        f = max(faces, key=lambda x: x.det_score)
        return _normalize(f.normed_embedding.astype(np.float32))

class FaceRegistry:
    # --- create a per-user enrollment video writer ---


    def __init__(self, encoder: FaceEncoder, min_frames: int = 30):
        self.encoder = encoder
        self.min_frames = min_frames

    def add_user_from_camera(self, username: str, cam_index: int = 0, seconds: int = 7) -> bool:
        db = _load_db()
        username = username.strip()
        save_video = True
        save_frames = True  # toggle if you also want per-frame JPGs
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # AVI with MJPG is widely compatible
        video_path = str(FACE_DIR / f"{username}_enroll.avi")
        video_writer = None
        if not username:
            return False
        

        
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print("Camera open failed.")
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        start = time.time()
        embeddings = []
        saved = 0

        while time.time() - start < seconds:
            ok, frame = cap.read()
            # init writer lazily when we know frame size
            if save_video and video_writer is None and frame is not None:
                h, w = frame.shape[:2]
                video_writer = cv2.VideoWriter(video_path, fourcc, 30, (w, h))
                if not video_writer.isOpened():
                    print("[Enroll] Could not open video writer. Will skip video save.")
                    video_writer = None

            # write video
            if video_writer is not None:
                video_writer.write(frame)

            # optionally save some JPG frames
            if save_frames and len(embeddings) < 30:  # first 30 samples
                jpg_path = FACE_DIR / f"{username}_frame_{len(embeddings):03d}.jpg"
                cv2.imwrite(str(jpg_path), frame)

            if not ok: break

            dets = self.encoder.detect(frame, conf=0.3)
            if not dets:
                cv2.imshow("Add user - show your face", frame)
                cv2.putText(frame, f"samples: {len(embeddings)} / target: {self.min_frames}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
       
                if cv2.waitKey(1) == 27: break
                continue

            dets.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
            x1,y1,x2,y2,_ = dets[0]
            x1,y1 = max(0,x1), max(0,y1)
            pad = int(0.15 * max(x2-x1, y2-y1))
            x1p, y1p = max(0, x1-pad), max(0, y1-pad)
            x2p, y2p = min(frame.shape[1], x2+pad), min(frame.shape[0], y2+pad)
            crop = frame[y1p:y2p, x1p:x2p].copy()

            emb = self.encoder.embed(crop)
            if emb is not None:
                embeddings.append(emb.tolist())
                if saved < 5:
                    (FACE_DIR / f"{username}_{int(time.time())}_{saved}.jpg").write_bytes(cv2.imencode(".jpg", crop)[1].tobytes())
                    saved += 1

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"Collecting {username} ({len(embeddings)})",(10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            cv2.imshow("Add user - show your face", frame)
            if cv2.waitKey(1) == 27: break

        cap.release()
        cv2.destroyAllWindows()

        if len(embeddings) < self.min_frames:
            print("Not enough face samples; try again.")
            return False

        # Robust template: mean of inliers
        import numpy as np
        arr = np.array(embeddings, dtype=np.float32)
        mean = arr.mean(axis=0)
        sims = np.dot(arr, _normalize(mean))
        mask = sims >= (sims.mean() - 2*sims.std())
        arr = arr[mask]
        final = arr.tolist()

        db.setdefault(username, []).extend(final)
        with open(DB_PATH, "w") as f:
            json.dump(db, f)
        print(f"Added/updated user {username} with {len(final)} embeddings.")
        return True

    def list_users(self) -> List[str]:
        return list(_load_db().keys())

    def user_template(self, username: str) -> Optional[np.ndarray]:
        db = _load_db()
        if username not in db or len(db[username]) == 0:
            return None
        import numpy as np
        arr = np.array(db[username], dtype=np.float32)
        return _normalize(arr.mean(axis=0))

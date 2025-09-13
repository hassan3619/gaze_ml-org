# face_lock.py
import cv2, numpy as np, time
from typing import Optional, Tuple
from face_registry import FaceEncoder, FaceRegistry, _normalize

class FaceLockFilter:
    """
    Filters frames to the selected operator’s face ROI.
    - Uses detector+embed every det_interval frames.
    - Tracks in between with CSRT for stability/latency.
    - K hits to lock; M misses to unlock.
    """
    def __init__(
        self,
        target_name: str,
        confidence: float = 0.25,
        match_thresh: float = 0.5,
        det_interval: int = 6,
        k_lock: int = 3,
        m_unlock: int = 15,
    ):
        self.encoder = FaceEncoder()
        self.registry = FaceRegistry(self.encoder)
        self.target_name = target_name
        self.template = self.registry.user_template(target_name)
        if self.template is None:
            raise RuntimeError(f"No embeddings for user: {target_name}")

        self.confidence = confidence
        self.match_thresh = match_thresh
        self.det_interval = max(1, det_interval)
        self.k_lock = k_lock
        self.m_unlock = m_unlock

        self.frame_idx = 0
        self.tracker = None
        self.bbox_xyxy: Optional[Tuple[int,int,int,int]] = None
        self.lock_hits = 0
        self.miss_count = 0
        self.locked = False

    def _init_tracker(self, frame, box_xyxy):
        x1,y1,x2,y2 = box_xyxy
        w, h = x2-x1, y2-y1
        self.tracker = cv2.legacy.TrackerCSRT_create()
        self.tracker.init(frame, (x1,y1,w,h))

    def _tracker_update(self, frame):
        if self.tracker is None:
            return None
        ok, box = self.tracker.update(frame)
        if not ok:
            return None
        x,y,w,h = map(int, box)
        return (x, y, x+w, y+h)

    def _best_target_box(self, frame):
        dets = self.encoder.detect(frame, conf=self.confidence)
        best, best_score = None, -1.0
        for x1, y1, x2, y2, _ in dets:
            crop = frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
            if crop.size == 0: 
                continue
            emb = self.encoder.embed(crop)
            if emb is None: 
                continue
            sim = float(np.dot(_normalize(emb), self.template))
            if sim > best_score:
                best_score = sim
                best = (x1,y1,x2,y2)
        if best is None or best_score < self.match_thresh:
            return None
        return best

    def _clip(self, frame, xyxy):
        x1,y1,x2,y2 = xyxy
        H,W = frame.shape[:2]
        return (max(0,x1), max(0,y1), min(W,x2), min(H,y2))

    def process(self, frame):
        """
        Returns (roi_frame, bbox_xyxy or None, locked: bool, similarity_or_None).
        """
        self.frame_idx += 1
        box = None
        sim = None

        # Periodic detect+embed; otherwise track
        if self.frame_idx % self.det_interval == 0 or self.tracker is None:
            box = self._best_target_box(frame)
            if box is not None:
                self.bbox_xyxy = self._clip(frame, box)
                self._init_tracker(frame, self.bbox_xyxy)
                self.lock_hits = min(self.k_lock, self.lock_hits + 1)
                self.miss_count = 0
            else:
                self.miss_count += 1
        else:
            tbox = self._tracker_update(frame)
            if tbox is not None:
                self.bbox_xyxy = self._clip(frame, tbox)
                self.miss_count = 0
                # keep accumulating hits while tracking after a good detect
                self.lock_hits = min(self.k_lock, self.lock_hits + 1)
            else:
                self.miss_count += 1

        # Lock/unlock logic
        if not self.locked and self.lock_hits >= self.k_lock and self.bbox_xyxy is not None:
            self.locked = True
        if self.locked and self.miss_count >= self.m_unlock:
            self.locked = False
            self.lock_hits = 0
            self.tracker = None
            self.bbox_xyxy = None

        # Prepare output
        if self.bbox_xyxy is not None:
            x1,y1,x2,y2 = self.bbox_xyxy
            pad = int(0.1 * max(x2-x1, y2-y1))
            x1,y1,x2,y2 = x1-pad, y1-pad, x2+pad, y2+pad
            x1,y1,x2,y2 = self._clip(frame, (x1,y1,x2,y2))
            roi = frame[y1:y2, x1:x2]
            return (roi.copy() if roi.size else frame, (x1,y1,x2,y2), self.locked, sim)
        else:
            return (frame, None, False, sim)

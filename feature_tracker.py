import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrackerConfig:
    detector: str = "harris"    
    harris_block_size: int = 2
    harris_ksize: int = 3
    harris_k: float = 0.04
    harris_threshold: float = 0.01    
    harris_min_distance: int = 10 
    fast_threshold: int = 20
    orb_n_features: int = 500
    orb_scale_factor: float = 1.2
    orb_n_levels: int = 8
    max_features: int = 500
    lk_win_size: tuple = (21, 21)
    lk_max_level: int = 3
    lk_criteria: tuple = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    min_tracked: int = 50    


class FeatureTracker:

    def __init__(self, cfg: TrackerConfig = TrackerConfig()):
        self.cfg = cfg
        self.orb = cv2.ORB_create(
            nfeatures=cfg.orb_n_features,
            scaleFactor=cfg.orb_scale_factor,
            nlevels=cfg.orb_n_levels,
        )

    def detect(self, gray: np.ndarray) -> np.ndarray:
        if self.cfg.detector == "harris":
            return self._detect_harris(gray)
        elif self.cfg.detector == "fast":
            return self._detect_fast(gray)
        else:
            raise ValueError(f"Unknown detector: {self.cfg.detector}")

    def _detect_harris(self, gray: np.ndarray) -> np.ndarray:
        gray_f = np.float32(gray)
        response = cv2.cornerHarris(
            gray_f,
            self.cfg.harris_block_size,
            self.cfg.harris_ksize,
            self.cfg.harris_k,
        )
        response = cv2.dilate(response, None)
        threshold = self.cfg.harris_threshold * response.max()
        mask = response > threshold
        ys, xs = np.where(mask)
        responses = response[ys, xs]
        order = np.argsort(-responses)
        xs, ys = xs[order], ys[order]
        kept = []
        occupied = np.zeros(gray.shape[:2], dtype=bool)
        d = self.cfg.harris_min_distance
        for x, y in zip(xs, ys):
            if len(kept) >= self.cfg.max_features:
                break
            x0, y0 = int(x), int(y)
            r0, r1 = max(0, y0 - d), min(gray.shape[0], y0 + d + 1)
            c0, c1 = max(0, x0 - d), min(gray.shape[1], x0 + d + 1)
            if occupied[r0:r1, c0:c1].any():
                continue
            kept.append([x0, y0])
            occupied[r0:r1, c0:c1] = True

        if not kept:
            return np.empty((0, 2), dtype=np.float32)
        return np.array(kept, dtype=np.float32)

    def _detect_fast(self, gray: np.ndarray) -> np.ndarray:
        fast = cv2.FastFeatureDetector_create(threshold=self.cfg.fast_threshold)
        kps = fast.detect(gray, None)
        if not kps:
            return np.empty((0, 2), dtype=np.float32)
        kps = sorted(kps, key=lambda k: -k.response)[: self.cfg.max_features]
        pts = np.array([[k.pt[0], k.pt[1]] for k in kps], dtype=np.float32)
        return pts

    def describe(self, gray: np.ndarray, pts: np.ndarray):
        
        if len(pts) == 0:
            return [], None
        kps = [cv2.KeyPoint(float(p[0]), float(p[1]), 7) for p in pts]
        kps, descs = self.orb.compute(gray, kps)
        return kps, descs

    def track(
        self,
        gray_prev: np.ndarray,
        gray_curr: np.ndarray,
        pts_prev: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        if len(pts_prev) == 0:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty, np.zeros(0, dtype=bool)

        p0 = pts_prev.reshape(-1, 1, 2).astype(np.float32)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            gray_prev, gray_curr, p0, None,
            winSize=self.cfg.lk_win_size,
            maxLevel=self.cfg.lk_max_level,
            criteria=self.cfg.lk_criteria,
        )
        # Back-track for consistency check
        p0r, st_back, _ = cv2.calcOpticalFlowPyrLK(
            gray_curr, gray_prev, p1, None,
            winSize=self.cfg.lk_win_size,
            maxLevel=self.cfg.lk_max_level,
            criteria=self.cfg.lk_criteria,
        )
        fb_err = np.linalg.norm(p0r.reshape(-1, 2) - p0.reshape(-1, 2), axis=1)

        good = (st.ravel() == 1) & (st_back.ravel() == 1) & (fb_err < 2.0)
        return (
            pts_prev[good],
            p1.reshape(-1, 2)[good],
            good,
        )

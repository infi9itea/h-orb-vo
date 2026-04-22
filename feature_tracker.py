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
    ratio_thresh: float = 0.75 
    min_tracked: int = 50


class FeatureTracker:

    def __init__(self, cfg: TrackerConfig = TrackerConfig()):
        self.cfg = cfg
        self.orb = cv2.ORB_create(
            nfeatures=cfg.orb_n_features,
            scaleFactor=cfg.orb_scale_factor,
            nlevels=cfg.orb_n_levels,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)



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
            return [], None, np.empty((0, 2), dtype=np.float32)

        kps_in = [cv2.KeyPoint(float(p[0]), float(p[1]), 7) for p in pts]
        kps_out, descs = self.orb.compute(gray, kps_in)

        if not kps_out or descs is None:
            return [], None, np.empty((0, 2), dtype=np.float32)

        pts_out = np.array([[k.pt[0], k.pt[1]] for k in kps_out], dtype=np.float32)
        return kps_out, descs, pts_out



    def match(
        self,
        gray_prev: np.ndarray,
        gray_curr: np.ndarray,
        pts_prev: np.ndarray,
        descs_prev: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        if descs_prev is None or len(descs_prev) == 0:
            _, descs_prev, pts_prev = self.describe(gray_prev, pts_prev)
            if descs_prev is None:
                empty = np.empty((0, 2), dtype=np.float32)
                return empty, empty, None, empty

        pts_curr_all = self.detect(gray_curr)
        _, descs_curr, pts_curr_all = self.describe(gray_curr, pts_curr_all)

        if descs_curr is None or len(descs_curr) == 0:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty, descs_curr, empty

        raw = self.matcher.knnMatch(descs_prev, descs_curr, k=2)

        good_prev, good_curr = [], []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.cfg.ratio_thresh * n.distance:
                good_prev.append(pts_prev[m.queryIdx])
                good_curr.append(pts_curr_all[m.trainIdx])

        if not good_prev:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty, descs_curr, pts_curr_all

        return (
            np.array(good_prev, dtype=np.float32),
            np.array(good_curr, dtype=np.float32),
            descs_curr,
            pts_curr_all,
        )

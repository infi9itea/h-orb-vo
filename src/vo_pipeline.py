"""
Monocular Visual Odometry Pipeline — Week 1-2 Baseline
=======================================================
Harris corner detection → ORB descriptors → Hamming BFMatcher + ratio test
→ RANSAC Essential Matrix → recoverPose → Triangulation → Trajectory

Author: Week 1–2 baseline implementation
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CameraIntrinsics:
    """Pinhole camera model."""
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0,      0,      1]], dtype=np.float64)

    @property
    def dist(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.p1, self.p2], dtype=np.float64)


@dataclass
class VOFrame:
    """Holds everything computed for one frame."""
    idx: int
    image: np.ndarray
    gray: np.ndarray
    keypoints: list = field(default_factory=list)
    descriptors: Optional[np.ndarray] = None
    # Pose relative to first frame  (world ← camera)
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros((3, 1)))


@dataclass
class MatchResult:
    pts_prev: np.ndarray   # (N, 2) float32 points in previous frame
    pts_curr: np.ndarray   # (N, 2) float32 points in current frame
    n_raw: int             # matches before ratio test
    n_inliers: int         # matches after RANSAC
    R: np.ndarray          # recovered rotation
    t: np.ndarray          # recovered translation (unit scale)
    mask: np.ndarray       # RANSAC inlier boolean mask


# ---------------------------------------------------------------------------
# Stage 1 — Harris corner detection
# ---------------------------------------------------------------------------

class HarrisDetector:
    """
    Detect corners with Harris response, then convert to cv2.KeyPoint objects
    so that ORB can compute descriptors at those locations.

    Parameters
    ----------
    max_corners   : maximum number of features to retain
    quality_level : minimum Harris score relative to the best pixel
    min_distance  : minimum Euclidean distance between features (px)
    block_size    : neighbourhood size for Harris computation
    k             : Harris free parameter (typically 0.04–0.06)
    """

    def __init__(
        self,
        max_corners: int = 2000,
        quality_level: float = 0.01,
        min_distance: float = 7.0,
        block_size: int = 3,
        k: float = 0.04,
    ):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        self.k = k

    def detect(self, gray: np.ndarray) -> list:
        """Return a list of cv2.KeyPoint from Harris corner detection."""
        # Harris response map (float32 per pixel)
        harris_map = cv2.cornerHarris(
            gray.astype(np.float32),
            blockSize=self.block_size,
            ksize=3,
            k=self.k,
        )

        # goodFeaturesToTrack with Harris scorer — returns (N, 1, 2) array
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size,
            useHarrisDetector=True,
            k=self.k,
        )

        if corners is None:
            return []

        # Assign Harris response as keypoint response for sorting/filtering
        keypoints = []
        for c in corners.reshape(-1, 2):
            x, y = c
            xi, yi = int(round(x)), int(round(y))
            # Clamp to image bounds
            yi = max(0, min(yi, gray.shape[0] - 1))
            xi = max(0, min(xi, gray.shape[1] - 1))
            resp = float(harris_map[yi, xi])
            kp = cv2.KeyPoint(x=float(x), y=float(y), size=7.0, response=resp)
            keypoints.append(kp)

        # Sort descending by Harris response
        keypoints.sort(key=lambda k: k.response, reverse=True)
        return keypoints


# ---------------------------------------------------------------------------
# Stage 2 — ORB descriptor computation
# ---------------------------------------------------------------------------

class ORBDescriptor:
    """
    Compute ORB (Oriented FAST and Rotated BRIEF) descriptors at the given
    keypoint locations.  We intentionally separate detection (Harris) from
    description (ORB) so each can be swapped independently.
    """

    def __init__(self, n_features: int = 2000, scale_factor: float = 1.2, n_levels: int = 8):
        self._orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
        )

    def compute(self, gray: np.ndarray, keypoints: list):
        """
        Given a list of cv2.KeyPoint, compute ORB descriptors.
        Returns (keypoints, descriptors) — keypoints may be a subset
        if ORB drops some (e.g. too close to border).
        """
        kps, descs = self._orb.compute(gray, keypoints)
        return kps, descs


# ---------------------------------------------------------------------------
# Stage 3 — Hamming BFMatcher + Lowe's ratio test
# ---------------------------------------------------------------------------

class HammingMatcher:
    """
    Brute-force matching on Hamming distance (appropriate for binary ORB descriptors).
    Applies Lowe's ratio test to filter ambiguous matches.
    """

    def __init__(self, ratio_threshold: float = 0.75):
        self.ratio_threshold = ratio_threshold
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match(self, desc1: np.ndarray, desc2: np.ndarray) -> list:
        """
        Returns a list of cv2.DMatch that pass the ratio test.
        desc1 from previous frame, desc2 from current frame.
        """
        if desc1 is None or desc2 is None:
            return []
        if len(desc1) < 2 or len(desc2) < 2:
            return []

        # knnMatch returns two nearest neighbours per query descriptor
        raw = self._bf.knnMatch(desc1, desc2, k=2)

        good = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            # Lowe's ratio test: accept only if best match is distinctly better
            if m.distance < self.ratio_threshold * n.distance:
                good.append(m)

        return good


# ---------------------------------------------------------------------------
# Stage 4 + 5 — RANSAC Essential Matrix & pose recovery
# ---------------------------------------------------------------------------

class EssentialMatrixEstimator:
    """
    Estimate the Essential matrix from 2-D correspondences using RANSAC,
    then recover R and t via SVD + cheirality check.
    """

    def __init__(
        self,
        prob: float = 0.999,
        threshold: float = 1.0,
        min_inliers: int = 8,
    ):
        self.prob = prob
        self.threshold = threshold
        self.min_inliers = min_inliers

    def estimate(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        K: np.ndarray,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray],
               Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Parameters
        ----------
        pts1, pts2 : (N, 2) float32 correspondences
        K          : 3×3 camera intrinsic matrix

        Returns
        -------
        E, R, t, mask  — or (None, None, None, None) if estimation fails
        """
        if len(pts1) < self.min_inliers:
            log.warning("Too few correspondences (%d) for Essential matrix.", len(pts1))
            return None, None, None, None

        E, mask = cv2.findEssentialMat(
            pts1, pts2, K,
            method=cv2.RANSAC,
            prob=self.prob,
            threshold=self.threshold,
        )

        if E is None or mask is None:
            return None, None, None, None

        n_inliers = int(mask.sum())
        if n_inliers < self.min_inliers:
            log.warning("Only %d RANSAC inliers — skipping frame.", n_inliers)
            return None, None, None, None

        # Recover rotation and (unit-scale) translation
        _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

        return E, R, t, mask.ravel().astype(bool)


# ---------------------------------------------------------------------------
# Stage 6 — Triangulation
# ---------------------------------------------------------------------------

def triangulate(pts1: np.ndarray, pts2: np.ndarray,
                P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """
    Triangulate matched 2-D points into 3-D using the Direct Linear Transform.

    Parameters
    ----------
    pts1, pts2 : (N, 2) float32 in pixel coordinates
    P1, P2     : 3×4 projection matrices  (K @ [R | t])

    Returns
    -------
    pts3d : (N, 3) float64 3-D points in the reference frame of P1
    """
    # triangulatePoints expects (2, N) homogeneous image coordinates
    pts1_h = pts1.T.astype(np.float64)   # (2, N)
    pts2_h = pts2.T.astype(np.float64)

    pts4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)   # (4, N)

    # Convert homogeneous → Euclidean
    pts3d = (pts4d[:3] / pts4d[3]).T     # (N, 3)
    return pts3d


# ---------------------------------------------------------------------------
# Stage 7 — VO Pipeline (orchestrator)
# ---------------------------------------------------------------------------

class MonocularVOPipeline:
    """
    Full monocular visual odometry pipeline (Week 1–2 baseline).

    Usage
    -----
    pipeline = MonocularVOPipeline(intrinsics)
    for img in frame_sequence:
        pose = pipeline.process_frame(img)
        # pose is (R, t) of the current camera in the world frame
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        max_corners: int = 2000,
        ratio_threshold: float = 0.75,
        ransac_threshold: float = 1.0,
        ransac_prob: float = 0.999,
        min_inliers: int = 15,
    ):
        self.K = intrinsics.K
        self.dist = intrinsics.dist

        self.detector   = HarrisDetector(max_corners=max_corners)
        self.descriptor = ORBDescriptor(n_features=max_corners)
        self.matcher    = HammingMatcher(ratio_threshold=ratio_threshold)
        self.estimator  = EssentialMatrixEstimator(
            prob=ransac_prob,
            threshold=ransac_threshold,
            min_inliers=min_inliers,
        )

        # Accumulated global pose  (world ← camera)
        self._R_global = np.eye(3)
        self._t_global = np.zeros((3, 1))

        # Previous frame state
        self._prev_kps   = None
        self._prev_descs = None
        self._prev_gray  = None
        self._frame_idx  = 0

        # Trajectory log: list of (3,) translation vectors
        self.trajectory: list[np.ndarray] = [np.zeros(3)]

        # Per-frame diagnostics
        self.diagnostics: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Process one RGB or grayscale frame.

        Returns current (R_global, t_global) — pose in world coordinates.
        The first frame initialises the reference and returns identity pose.
        """
        gray = self._to_gray(image)
        gray = self._undistort(gray)

        # Stage 1: Harris detection
        kps = self.detector.detect(gray)

        # Stage 2: ORB description
        kps, descs = self.descriptor.compute(gray, kps)

        diag = {"frame": self._frame_idx, "n_kps": len(kps),
                "n_matches": 0, "n_inliers": 0, "status": "init"}

        if self._frame_idx == 0:
            # First frame — no motion to estimate
            self._store_prev(gray, kps, descs)
            self._frame_idx += 1
            self.diagnostics.append(diag)
            return self._R_global.copy(), self._t_global.copy()

        # Stage 3: Hamming matching + ratio test
        matches = self.matcher.match(self._prev_descs, descs)
        diag["n_matches"] = len(matches)

        if len(matches) < self.estimator.min_inliers:
            log.warning("Frame %d: too few matches (%d).", self._frame_idx, len(matches))
            diag["status"] = "few_matches"
            self._store_prev(gray, kps, descs)
            self.trajectory.append(self._t_global.ravel().copy())
            self._frame_idx += 1
            self.diagnostics.append(diag)
            return self._R_global.copy(), self._t_global.copy()

        # Extract matched point arrays
        pts_prev = np.float32([self._prev_kps[m.queryIdx].pt for m in matches])
        pts_curr = np.float32([kps[m.trainIdx].pt for m in matches])

        # Stage 4–5: Essential matrix + pose recovery
        E, R, t, mask = self.estimator.estimate(pts_prev, pts_curr, self.K)

        if R is None:
            diag["status"] = "ransac_fail"
            self._store_prev(gray, kps, descs)
            self.trajectory.append(self._t_global.ravel().copy())
            self._frame_idx += 1
            self.diagnostics.append(diag)
            return self._R_global.copy(), self._t_global.copy()

        n_inliers = int(mask.sum())
        diag["n_inliers"] = n_inliers
        diag["status"] = "ok"

        # Stage 6: Triangulation (inlier points only)
        inlier_prev = pts_prev[mask]
        inlier_curr = pts_curr[mask]

        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R, t])
        pts3d = triangulate(inlier_prev, inlier_curr, P1, P2)

        # Monocular scale: fix translation norm to 1 (no absolute scale)
        # In Week 5–8 you can use ground-truth or depth to recover scale.
        t_norm = np.linalg.norm(t)
        if t_norm > 1e-8:
            t = t / t_norm  # unit translation

        # Stage 7: Accumulate global pose
        # recoverPose returns (R, t) such that x_curr = R*x_prev + t
        # For world-frame tracking: x_prev = R^T * x_curr - R^T * t
        R_rel = R.T
        t_rel = -R.T @ t

        self._t_global = self._t_global + self._R_global @ t_rel
        self._R_global = self._R_global @ R_rel

        self.trajectory.append(self._t_global.ravel().copy())

        log.info(
            "Frame %4d | kps=%4d | matches=%4d | inliers=%4d | pos=[%.2f %.2f %.2f]",
            self._frame_idx, len(kps), len(matches), n_inliers,
            *self._t_global.ravel(),
        )

        self._store_prev(gray, kps, descs)
        self._frame_idx += 1
        self.diagnostics.append(diag)
        return self._R_global.copy(), self._t_global.copy()

    def get_trajectory(self) -> np.ndarray:
        """Return (N, 3) array of camera positions in world coordinates."""
        return np.array(self.trajectory)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()

    def _undistort(self, gray: np.ndarray) -> np.ndarray:
        if np.any(self.dist != 0):
            return cv2.undistort(gray, self.K, self.dist)
        return gray

    def _store_prev(self, gray, kps, descs):
        self._prev_gray  = gray
        self._prev_kps   = kps
        self._prev_descs = descs

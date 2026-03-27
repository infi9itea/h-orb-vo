"""
Pose estimation via the 5-point algorithm (Essential matrix), RANSAC,
and linear triangulation.
"""
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class PoseConfig:
    ransac_prob: float = 0.999
    ransac_threshold: float = 1.0   # pixels (for E matrix)
    min_inliers: int = 20
    triangulate_min_angle_deg: float = 1.0   # filter degenerate triangulations


class PoseEstimator:
    """
    Given matched 2-D points and camera intrinsics K, recover
    relative rotation R and translation t (up to scale).
    """

    def __init__(self, K: np.ndarray, cfg: PoseConfig = PoseConfig()):
        self.K = K
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Essential matrix + decomposition
    # ------------------------------------------------------------------
    def recover_pose(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
    ) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate R, t from matched point pairs using the 5-point algorithm.

        Returns
        -------
        success : bool
        R       : (3,3)
        t       : (3,1) unit vector
        mask    : (N,1) uint8 RANSAC inlier mask
        """
        if len(pts1) < 5:
            return False, np.eye(3), np.zeros((3, 1)), np.zeros((len(pts1), 1), dtype=np.uint8)

        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=self.cfg.ransac_prob,
            threshold=self.cfg.ransac_threshold,
        )
        if E is None or mask is None:
            return False, np.eye(3), np.zeros((3, 1)), np.zeros((len(pts1), 1), dtype=np.uint8)

        n_inliers = int(mask.sum())
        if n_inliers < self.cfg.min_inliers:
            return False, np.eye(3), np.zeros((3, 1)), mask

        n_pass, R, t, mask2 = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask.copy())
        if n_pass < self.cfg.min_inliers:
            return False, np.eye(3), np.zeros((3, 1)), mask

        return True, R, t, mask2

    # ------------------------------------------------------------------
    # Triangulation
    # ------------------------------------------------------------------
    def triangulate(
        self,
        R1: np.ndarray, t1: np.ndarray,
        R2: np.ndarray, t2: np.ndarray,
        pts1: np.ndarray, pts2: np.ndarray,
    ) -> np.ndarray:
        """
        Linear triangulation (DLT).

        Parameters
        ----------
        R1, t1 : pose of camera 1  (world-to-cam)
        R2, t2 : pose of camera 2  (world-to-cam)
        pts1   : (N,2) image points in camera 1
        pts2   : (N,2) image points in camera 2

        Returns
        -------
        pts3d  : (N,3) in world coordinates (may contain NaN for degenerate pts)
        """
        P1 = self.K @ np.hstack([R1, t1])
        P2 = self.K @ np.hstack([R2, t2])

        pts1_h = pts1.T.astype(np.float64)   # (2, N)
        pts2_h = pts2.T.astype(np.float64)

        pts4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)   # (4, N)
        pts4d /= pts4d[3:4]   # homogeneous -> euclidean
        pts3d = pts4d[:3].T   # (N, 3)

        # Filter by reprojection angle (depth check is done in pipeline)
        # mark degenerate as NaN
        z1 = (R1 @ pts3d.T + t1).T[:, 2]   # depth in cam1
        z2 = (R2 @ pts3d.T + t2).T[:, 2]   # depth in cam2
        bad = (z1 <= 0) | (z2 <= 0)
        pts3d[bad] = np.nan

        return pts3d

    # ------------------------------------------------------------------
    # Homography-based check (detect pure-rotation / planar scenes)
    # ------------------------------------------------------------------
    def is_degenerate(self, pts1: np.ndarray, pts2: np.ndarray) -> bool:
        """
        Returns True if the motion is likely pure-rotation or planar
        (H inliers > E inliers * 0.45 heuristic).
        """
        if len(pts1) < 8:
            return True
        _, mH = cv2.findHomography(pts1, pts2, cv2.RANSAC, 2.0)
        _, mE = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.99,
            threshold=1.0,
        )
        if mH is None or mE is None:
            return False
        ratio = mH.sum() / max(1, mE.sum())
        return ratio > 0.45

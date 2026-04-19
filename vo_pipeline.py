"""
Monocular Visual Odometry Pipeline
===================================
Wires together:
  FeatureTracker  (Harris/FAST + ORB + LK optical flow)
  PoseEstimator   (5-point algorithm, RANSAC, triangulation)
  Trajectory      (pose book-keeping)
  scale recovery  (GT-assisted, standard monocular VO protocol)

Also records per-frame inlier_counts and match_counts for diagnostics.
"""
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Iterator

from feature_tracker import FeatureTracker, TrackerConfig
from pose_estimator import PoseEstimator, PoseConfig
from trajectory import Trajectory, Frame, recover_scale_from_gt


@dataclass
class VOConfig:
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    use_gt_scale: bool = True
    min_baseline: float = 0.1
    verbose: bool = False


class MonocularVO:
    """
    Frame-to-frame monocular VO.

    Usage
    -----
    vo = MonocularVO(K, cfg)
    for img, _, gt_pose in dataset:
        vo.process_frame(img, gt_pose)
    traj         = vo.trajectory
    inlier_counts = vo.inlier_counts   # list[int], one per frame
    match_counts  = vo.match_counts    # list[int], one per frame
    """

    def __init__(self, K: np.ndarray, cfg: VOConfig = VOConfig()):
        self.K = K
        self.cfg = cfg
        self.tracker = FeatureTracker(cfg.tracker)
        self.estimator = PoseEstimator(K, cfg.pose)
        self.trajectory = Trajectory()

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None
        self._frame_idx: int = 0

        self._R = np.eye(3)
        self._t = np.zeros((3, 1))
        self._prev_gt_pos: Optional[np.ndarray] = None

        # Diagnostic counters — one entry per processed frame
        self.inlier_counts: list = []
        self.match_counts: list = []

        self.stats = []

    # ──────────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────────

    def process_frame(self, img: np.ndarray, gt_pose=None) -> dict:
        """
        Process a single frame.

        Parameters
        ----------
        img      : HxW uint8 greyscale
        gt_pose  : (4,4) world-to-cam  OR  (3,) camera origin in world

        Returns
        -------
        dict with R, t, n_features, n_inliers, success
        """
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Extract GT position
        gt_pos = None
        if gt_pose is not None:
            arr = np.array(gt_pose)
            if arr.shape == (4, 4):
                gt_pos = arr[:3, 3]
            else:
                gt_pos = arr.ravel()[:3]

        # ── First frame ─────────────────────────────────────────────────
        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_pts = self.tracker.detect(gray)
            self._prev_gt_pos = gt_pos
            frame = Frame(0, self._R.copy(), self._t.copy(), 0, len(self._prev_pts))
            self.trajectory.add(frame, gt_pos)
            self.inlier_counts.append(0)
            self.match_counts.append(len(self._prev_pts))
            self._frame_idx = 1
            return {"R": self._R, "t": self._t, "n_features": len(self._prev_pts),
                    "n_inliers": 0, "success": True, "frame": 0}

        # ── Track ────────────────────────────────────────────────────────
        pts_prev_ok, pts_curr_ok, _ = self.tracker.track(
            self._prev_gray, gray, self._prev_pts)
        n_tracked = len(pts_prev_ok)
        self.match_counts.append(n_tracked)

        # ── Re-detect if too few tracks ──────────────────────────────────
        if n_tracked < self.cfg.tracker.min_tracked:
            new_pts = self.tracker.detect(gray)
            self._prev_gray = gray
            self._prev_pts = new_pts
            self._prev_gt_pos = gt_pos
            frame = Frame(self._frame_idx, self._R.copy(), self._t.copy(), 0, len(new_pts))
            self.trajectory.add(frame, gt_pos)
            self.inlier_counts.append(0)
            self._frame_idx += 1
            if self.cfg.verbose:
                print(f"[{self._frame_idx}] Re-detected ({n_tracked} tracks)")
            return {"R": self._R, "t": self._t, "n_features": len(new_pts),
                    "n_inliers": 0, "success": False, "frame": self._frame_idx - 1}

        # ── Pose estimation (5-point + RANSAC) ───────────────────────────
        success, R_rel, t_rel, inlier_mask = self.estimator.recover_pose(
            pts_prev_ok, pts_curr_ok)
        n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        self.inlier_counts.append(n_inliers)

        if not success:
            if self.cfg.verbose:
                print(f"[{self._frame_idx}] Pose estimation failed")
            self._prev_gray = gray
            self._prev_pts = self.tracker.detect(gray)
            self._prev_gt_pos = gt_pos
            frame = Frame(self._frame_idx, self._R.copy(), self._t.copy(), 0, n_tracked)
            self.trajectory.add(frame, gt_pos)
            self._frame_idx += 1
            return {"R": self._R, "t": self._t, "n_features": n_tracked,
                    "n_inliers": 0, "success": False, "frame": self._frame_idx - 1}

        # ── Scale recovery ───────────────────────────────────────────────
        if self.cfg.use_gt_scale and gt_pos is not None and self._prev_gt_pos is not None:
            scale = recover_scale_from_gt(t_rel, self._prev_gt_pos, gt_pos)
        else:
            scale = 1.0

        # ── Compose poses ────────────────────────────────────────────────
        self._t = self._t + scale * (self._R @ t_rel)
        self._R = self._R @ R_rel.T

        # Re-detect on inlier set
        inlier_flat = inlier_mask.ravel().astype(bool)
        new_tracked = pts_curr_ok[inlier_flat]
        if len(new_tracked) < self.cfg.tracker.min_tracked:
            new_tracked = self.tracker.detect(gray)

        self._prev_gray = gray
        self._prev_pts = new_tracked
        self._prev_gt_pos = gt_pos

        frame = Frame(self._frame_idx, self._R.copy(), self._t.copy(), n_inliers, n_tracked)
        self.trajectory.add(frame, gt_pos)

        stat = {"frame": self._frame_idx, "R": self._R.copy(), "t": self._t.copy(),
                "n_features": n_tracked, "n_inliers": n_inliers,
                "scale": scale, "success": True}
        self.stats.append(stat)
        self._frame_idx += 1

        if self.cfg.verbose and self._frame_idx % 50 == 0:
            print(f"[{self._frame_idx}] t={self._t.ravel()}, "
                  f"inliers={n_inliers}/{n_tracked}, scale={scale:.3f}")

        return stat

    # ──────────────────────────────────────────────────────────────────────
    # Batch run
    # ──────────────────────────────────────────────────────────────────────

    def run(self, dataset_iter: Iterator, max_frames: int = None) -> Trajectory:
        """
        Convenience: iterate a dataset, process all frames.
        dataset_iter yields (img, K, gt_pose_or_None)
        """
        for i, (img, K, gt_pose) in enumerate(dataset_iter):
            if max_frames is not None and i >= max_frames:
                break
            self.process_frame(img, gt_pose)
        return self.trajectory

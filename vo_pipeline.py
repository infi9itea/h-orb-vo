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

    def __init__(self, K: np.ndarray, cfg: VOConfig = VOConfig()):
        self.K = K
        self.cfg = cfg
        self.tracker = FeatureTracker(cfg.tracker)
        self.estimator = PoseEstimator(K, cfg.pose)
        self.trajectory = Trajectory()

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None
        self._prev_descs: Optional[np.ndarray] = None  
        self._frame_idx: int = 0

        self._R = np.eye(3)
        self._t = np.zeros((3, 1))
        self._prev_gt_pos: Optional[np.ndarray] = None

        self.inlier_counts: list = []
        self.match_counts: list = []
        self.stats = []


    def _detect_and_describe(self, gray: np.ndarray):
        pts = self.tracker.detect(gray)
        _, descs, pts = self.tracker.describe(gray, pts)
        return pts, descs

    def _redetect(self, gray: np.ndarray, gt_pos):
        pts, descs = self._detect_and_describe(gray)
        self._prev_gray = gray
        self._prev_pts = pts
        self._prev_descs = descs
        self._prev_gt_pos = gt_pos



    def process_frame(self, img: np.ndarray, gt_pose=None) -> dict:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gt_pos = None
        if gt_pose is not None:
            arr = np.array(gt_pose)
            gt_pos = arr[:3, 3] if arr.shape == (4, 4) else arr.ravel()[:3]

        if self._prev_gray is None:
            self._prev_pts, self._prev_descs = self._detect_and_describe(gray)
            self._prev_gray = gray
            self._prev_gt_pos = gt_pos
            frame = Frame(0, self._R.copy(), self._t.copy(), 0, len(self._prev_pts))
            self.trajectory.add(frame, gt_pos)
            self.inlier_counts.append(0)
            self.match_counts.append(len(self._prev_pts))
            self._frame_idx = 1
            return {"R": self._R, "t": self._t,
                    "n_features": len(self._prev_pts),
                    "n_inliers": 0, "success": True, "frame": 0}

        pts_prev_ok, pts_curr_ok, descs_curr, pts_curr_all = self.tracker.match(
            self._prev_gray, gray,
            self._prev_pts, self._prev_descs,
        )
        n_matched = len(pts_prev_ok)
        self.match_counts.append(n_matched)

        if n_matched < self.cfg.tracker.min_tracked:
            self._redetect(gray, gt_pos)
            frame = Frame(self._frame_idx, self._R.copy(), self._t.copy(),
                          0, len(self._prev_pts))
            self.trajectory.add(frame, gt_pos)
            self.inlier_counts.append(0)
            self._frame_idx += 1
            if self.cfg.verbose:
                print(f"[{self._frame_idx}] Re-detected ({n_matched} matches)")
            return {"R": self._R, "t": self._t,
                    "n_features": len(self._prev_pts),
                    "n_inliers": 0, "success": False,
                    "frame": self._frame_idx - 1}

        success, R_rel, t_rel, inlier_mask = self.estimator.recover_pose(
            pts_prev_ok, pts_curr_ok)
        n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        self.inlier_counts.append(n_inliers)

        if not success:
            if self.cfg.verbose:
                print(f"[{self._frame_idx}] Pose estimation failed")
            self._redetect(gray, gt_pos)
            frame = Frame(self._frame_idx, self._R.copy(), self._t.copy(),
                          0, n_matched)
            self.trajectory.add(frame, gt_pos)
            self._frame_idx += 1
            return {"R": self._R, "t": self._t,
                    "n_features": n_matched,
                    "n_inliers": 0, "success": False,
                    "frame": self._frame_idx - 1}

        if self.cfg.use_gt_scale and gt_pos is not None and self._prev_gt_pos is not None:
            scale = recover_scale_from_gt(t_rel, self._prev_gt_pos, gt_pos)
        else:
            scale = 1.0

        self._t = self._t + scale * (self._R @ -t_rel)
        self._R = R_rel @ self._R


        self._prev_gray = gray
        self._prev_pts = pts_curr_all if pts_curr_all is not None and len(pts_curr_all) >= self.cfg.tracker.min_tracked \
                         else self.tracker.detect(gray)
        if len(self._prev_pts) != (len(pts_curr_all) if pts_curr_all is not None else 0):
            _, self._prev_descs, self._prev_pts = self.tracker.describe(gray, self._prev_pts)
        else:
            self._prev_descs = descs_curr
        self._prev_gt_pos = gt_pos

        frame = Frame(self._frame_idx, self._R.copy(), self._t.copy(),
                      n_inliers, n_matched)
        self.trajectory.add(frame, gt_pos)

        stat = {"frame": self._frame_idx, "R": self._R.copy(), "t": self._t.copy(),
                "n_features": n_matched, "n_inliers": n_inliers,
                "scale": scale, "success": True}
        self.stats.append(stat)
        self._frame_idx += 1

        if self.cfg.verbose and self._frame_idx % 50 == 0:
            print(f"[{self._frame_idx}] t={self._t.ravel()}, "
                  f"inliers={n_inliers}/{n_matched}, scale={scale:.3f}")

        return stat

    def run(self, dataset_iter: Iterator, max_frames: int = None) -> Trajectory:
        for i, (img, K, gt_pose) in enumerate(dataset_iter):
            if max_frames is not None and i >= max_frames:
                break
            self.process_frame(img, gt_pose)
        return self.trajectory

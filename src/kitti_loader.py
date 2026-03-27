"""
KITTI Dataset Loader
====================
Supports KITTI odometry benchmark sequences (00–10).
Downloads/loads calibration, images, and ground-truth poses.

Directory structure expected:
    <root>/
        sequences/
            00/
                image_0/  (left grayscale)
                image_2/  (left colour, optional)
                calib.txt
                times.txt
        poses/
            00.txt
            ...
"""

import os
import glob
import numpy as np
from pathlib import Path
from typing import Iterator, Optional
import cv2


class KITTISequence:
    """
    Loads a single KITTI odometry sequence.

    Parameters
    ----------
    root      : path to the KITTI odometry dataset root
    seq_id    : sequence number as a 2-char string, e.g. "00"
    camera_id : 0 = left grey, 1 = right grey, 2 = left colour
    """

    def __init__(self, root: str, seq_id: str, camera_id: int = 0):
        self.root      = Path(root)
        self.seq_id    = seq_id.zfill(2)
        self.camera_id = camera_id

        self.seq_dir  = self.root / "sequences" / self.seq_id
        self.img_dir  = self.seq_dir / f"image_{camera_id}"
        self.calib    = self._load_calib()
        self.poses_gt = self._load_poses()   # None if poses/ not present
        self.times    = self._load_times()

        self._image_paths = sorted(
            glob.glob(str(self.img_dir / "*.png"))
        )
        if not self._image_paths:
            raise FileNotFoundError(f"No images found in {self.img_dir}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def K(self) -> np.ndarray:
        """3×3 intrinsic matrix for the chosen camera."""
        P = self.calib[f"P{self.camera_id}"]   # 3×4
        return P[:3, :3]

    @property
    def intrinsics(self):
        from vo_pipeline import CameraIntrinsics
        K = self.K
        return CameraIntrinsics(
            fx=K[0, 0], fy=K[1, 1],
            cx=K[0, 2], cy=K[1, 2],
        )

    def __len__(self):
        return len(self._image_paths)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[np.ndarray]:
        for path in self._image_paths:
            yield cv2.imread(path)

    def get_frame(self, idx: int) -> np.ndarray:
        return cv2.imread(self._image_paths[idx])

    # ------------------------------------------------------------------
    # Pose utilities
    # ------------------------------------------------------------------

    def get_gt_xyz(self) -> Optional[np.ndarray]:
        """
        Return (N, 3) ground-truth camera positions in world coordinates,
        or None if no ground-truth is available.
        """
        if self.poses_gt is None:
            return None
        return self.poses_gt[:, :3, 3]

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_calib(self) -> dict:
        calib_file = self.seq_dir / "calib.txt"
        if not calib_file.exists():
            raise FileNotFoundError(f"calib.txt not found at {calib_file}")

        calib = {}
        with open(calib_file) as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, values = line.split(":", 1)
                key = key.strip()
                vals = list(map(float, values.split()))
                if key.startswith("P"):
                    calib[key] = np.array(vals).reshape(3, 4)
                elif key == "Tr":
                    calib[key] = np.array(vals).reshape(3, 4)
        return calib

    def _load_poses(self) -> Optional[np.ndarray]:
        poses_dir = self.root / "poses"
        pose_file = poses_dir / f"{self.seq_id}.txt"
        if not pose_file.exists():
            return None

        poses = []
        with open(pose_file) as f:
            for line in f:
                vals = list(map(float, line.split()))
                P = np.array(vals).reshape(3, 4)
                # Extend to 4×4 homogeneous
                P4 = np.eye(4)
                P4[:3, :] = P
                poses.append(P4)
        return np.array(poses)   # (N, 4, 4)

    def _load_times(self) -> Optional[np.ndarray]:
        times_file = self.seq_dir / "times.txt"
        if not times_file.exists():
            return None
        times = []
        with open(times_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    times.append(float(line))
        return np.array(times)


# ---------------------------------------------------------------------------
# Convenience: build a pipeline ready to run on a KITTI sequence
# ---------------------------------------------------------------------------

def build_pipeline_for_kitti(kitti_seq: KITTISequence, **kwargs):
    """Return a MonocularVOPipeline pre-configured for this KITTI sequence."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from vo_pipeline import MonocularVOPipeline
    return MonocularVOPipeline(kitti_seq.intrinsics, **kwargs)

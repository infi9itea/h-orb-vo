"""
Dataset loaders for KITTI (odometry) and EuRoC MAV.
Both yield (image_path, K, gt_pose_or_None) tuples.
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Optional


# ──────────────────────────────────────────────────────────────────────────────
# KITTI Odometry
# ──────────────────────────────────────────────────────────────────────────────

class KITTISequence:
    """
    Directory layout expected:
        <root>/sequences/<seq>/image_0/*.png
        <root>/sequences/<seq>/calib.txt
        <root>/poses/<seq>.txt             (optional, for GT)
    """

    def __init__(self, root: str, seq: str, cam: int = 0):
        self.root = Path(root)
        self.seq = seq
        self.cam = cam

        img_dir = self.root / "sequences" / seq / f"image_{cam}"
        self.image_paths = sorted(img_dir.glob("*.png"))
        if not self.image_paths:
            self.image_paths = sorted(img_dir.glob("*.jpg"))
        assert self.image_paths, f"No images found in {img_dir}"

        self.K = self._load_K()
        self.gt_poses = self._load_gt()  # list of (4,4) or None

    def _load_K(self) -> np.ndarray:
        calib_path = self.root / "sequences" / self.seq / "calib.txt"
        with open(calib_path) as f:
            for line in f:
                if line.startswith(f"P{self.cam}:"):
                    vals = list(map(float, line.strip().split()[1:]))
                    P = np.array(vals).reshape(3, 4)
                    return P[:3, :3]
        raise RuntimeError(f"Calibration for P{self.cam} not found")

    def _load_gt(self) -> Optional[list]:
        gt_path = self.root / "poses" / f"{self.seq}.txt"
        if not gt_path.exists():
            return None
        poses = []
        with open(gt_path) as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                T = np.eye(4)
                T[:3] = np.array(vals).reshape(3, 4)
                poses.append(T)
        return poses

    def __len__(self):
        return len(self.image_paths)

    def __iter__(self) -> Iterator[tuple]:
        for i, img_path in enumerate(self.image_paths):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            gt = self.gt_poses[i] if self.gt_poses is not None else None
            yield img, self.K, gt

    def gt_position(self, idx: int) -> Optional[np.ndarray]:
        if self.gt_poses is None:
            return None
        return self.gt_poses[idx][:3, 3]


# ──────────────────────────────────────────────────────────────────────────────
# EuRoC MAV
# ──────────────────────────────────────────────────────────────────────────────

class EuRoCSequence:
    """
    Directory layout expected (ASL dataset format):
        <root>/cam0/data/*.png
        <root>/cam0/sensor.yaml
        <root>/state_groundtruth_estimate0/data.csv   (optional GT)
    """

    def __init__(self, root: str, cam: str = "cam0"):
        self.root = Path(root)
        self.cam = cam

        img_dir = self.root / cam / "data"
        self.image_paths = sorted(img_dir.glob("*.png"))
        assert self.image_paths, f"No images found in {img_dir}"

        self.K, self.dist = self._load_calib()
        self.gt_poses = self._load_gt()  # dict ts -> (3,) position or None

    def _load_calib(self):
        """Parse sensor.yaml for intrinsics."""
        yaml_path = self.root / self.cam / "sensor.yaml"
        K = np.eye(3)
        dist = np.zeros(4)
        if not yaml_path.exists():
            print(f"[warn] sensor.yaml not found, using identity K")
            return K, dist
        import yaml
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        intr = cfg.get("intrinsics", [458.654, 457.296, 367.215, 248.375])
        K[0, 0] = intr[0]; K[1, 1] = intr[1]
        K[0, 2] = intr[2]; K[1, 2] = intr[3]
        dist_coeffs = cfg.get("distortion_coefficients", [0]*4)
        dist = np.array(dist_coeffs[:4])
        return K, dist

    def _load_gt(self) -> Optional[dict]:
        gt_csv = self.root / "state_groundtruth_estimate0" / "data.csv"
        if not gt_csv.exists():
            return None
        gt = {}
        with open(gt_csv) as f:
            next(f)  # header
            for line in f:
                vals = line.strip().split(",")
                ts = int(vals[0])
                pos = np.array([float(v) for v in vals[1:4]])
                gt[ts] = pos
        return gt

    def __len__(self):
        return len(self.image_paths)

    def __iter__(self) -> Iterator[tuple]:
        for img_path in self.image_paths:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # Undistort
            img = cv2.undistort(img, self.K, self.dist)
            ts = int(img_path.stem)
            gt_pos = None
            if self.gt_poses:
                # Find nearest GT timestamp
                nearest = min(self.gt_poses.keys(), key=lambda t: abs(t - ts))
                if abs(nearest - ts) < 1e7:   # within 10 ms
                    gt_pos = self.gt_poses[nearest]
            yield img, self.K, gt_pos

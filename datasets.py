import os
import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Optional

class KITTISequence:

    def __init__(self, root: str, seq: str, cam: int = 0):
        self.root = Path(root)
        self.seq = seq
        self.cam = cam

        # Try specified camera first, then fall back to others if needed
        img_dir = self.root / "sequences" / seq / f"image_{self.cam}"
        self.image_paths = self._find_images(img_dir)

        if not self.image_paths:
            # Common camera indices: 0, 1 (grayscale), 2, 3 (color)
            for fallback_cam in [0, 1, 2, 3]:
                if fallback_cam == self.cam:
                    continue
                img_dir = self.root / "sequences" / seq / f"image_{fallback_cam}"
                self.image_paths = self._find_images(img_dir)
                if self.image_paths:
                    print(f"[KITTI] Camera {self.cam} not found, falling back to camera {fallback_cam}")
                    self.cam = fallback_cam
                    break

        assert self.image_paths, f"No images found for sequence {seq} in {self.root / 'sequences' / seq}"

        self.K = self._load_K()
        self.gt_poses = self._load_gt()  # list of (4,4) or None

    def _find_images(self, img_dir: Path) -> list:
        paths = sorted(img_dir.glob("*.png"))
        if not paths:
            paths = sorted(img_dir.glob("*.jpg"))
        return paths

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


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n < 1e-12:
        return np.eye(3)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


class EuRoCSequence:
    """Loader for EuRoC MAV dataset sequences.

    Expected directory layout (either works):
        <root>/mav0/cam0/data/*.png          (standard)
        <root>/cam0/data/*.png               (flat)

    Ground-truth is read from:
        <root>/mav0/state_groundtruth_estimate0/data.csv
    """

    def __init__(self, root: str):
        self.root = Path(root)

        cam_dir = self._find_cam_dir()
        img_dir = cam_dir / "data"
        assert img_dir.is_dir(), f"Image directory not found: {img_dir}"

        self.image_paths = sorted(img_dir.glob("*.png"))
        if not self.image_paths:
            self.image_paths = sorted(img_dir.glob("*.jpg"))
        assert self.image_paths, f"No images found in {img_dir}"

        self.K = self._load_K(cam_dir)
        self.gt_poses = self._load_gt()

    # ------------------------------------------------------------------
    # Directory discovery
    # ------------------------------------------------------------------

    def _find_cam_dir(self) -> Path:
        for rel in ["mav0/cam0", "cam0"]:
            d = self.root / rel
            if d.is_dir():
                return d
        raise FileNotFoundError(
            f"cam0 directory not found under {self.root}. "
            f"Expected <root>/mav0/cam0/ or <root>/cam0/"
        )

    # ------------------------------------------------------------------
    # Intrinsics
    # ------------------------------------------------------------------

    def _load_K(self, cam_dir: Path) -> np.ndarray:
        sensor_path = cam_dir / "sensor.yaml"
        if sensor_path.exists():
            try:
                import yaml
                with open(sensor_path) as f:
                    data = yaml.safe_load(f)
                intrinsics = data.get("intrinsics", None)
                if intrinsics and len(intrinsics) == 4:
                    fu, fv, cu, cv = intrinsics
                    return np.array(
                        [[fu, 0, cu], [0, fv, cv], [0, 0, 1]],
                        dtype=np.float64,
                    )
            except Exception as e:
                print(f"[EuRoC] Warning: failed to parse {sensor_path}: {e}")

        # Default EuRoC cam0 intrinsics (Machine Hall sequences)
        print("[EuRoC] Using default cam0 intrinsics (MH sequences)")
        return np.array(
            [[458.654, 0.0, 367.215],
             [0.0, 457.296, 248.375],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Ground-truth loading & timestamp matching
    # ------------------------------------------------------------------

    def _load_gt(self) -> Optional[list]:
        for gt_rel in [
            "mav0/state_groundtruth_estimate0/data.csv",
            "state_groundtruth_estimate0/data.csv",
        ]:
            gt_path = self.root / gt_rel
            if gt_path.exists():
                return self._match_gt_to_images(gt_path)
        return None

    def _match_gt_to_images(self, gt_path: Path) -> Optional[list]:
        """Parse GT CSV and match each image to its nearest GT pose by timestamp."""
        gt_timestamps, gt_poses = [], []
        with open(gt_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                vals = line.split(",")
                if len(vals) < 8:
                    continue
                gt_timestamps.append(int(vals[0]))
                px, py, pz = float(vals[1]), float(vals[2]), float(vals[3])
                qw, qx, qy, qz = (
                    float(vals[4]), float(vals[5]),
                    float(vals[6]), float(vals[7]),
                )
                T = np.eye(4)
                T[:3, :3] = _quat_to_rot(qw, qx, qy, qz)
                T[:3, 3] = [px, py, pz]
                gt_poses.append(T)

        if not gt_poses:
            return None

        gt_ts = np.array(gt_timestamps)
        img_ts = np.array([int(p.stem) for p in self.image_paths])

        # Binary-search for nearest GT entry per image timestamp
        indices = np.searchsorted(gt_ts, img_ts)
        indices = np.clip(indices, 1, len(gt_ts) - 1)
        left = np.abs(img_ts - gt_ts[indices - 1])
        right = np.abs(img_ts - gt_ts[indices])
        indices[left < right] -= 1

        return [gt_poses[i] for i in indices]

    # ------------------------------------------------------------------
    # Sequence interface
    # ------------------------------------------------------------------

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
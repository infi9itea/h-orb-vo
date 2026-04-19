import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Frame:
    idx: int
    R: np.ndarray          
    t: np.ndarray      
    n_inliers: int = 0
    n_features: int = 0


class Trajectory:

    def __init__(self):
        self.frames: list[Frame] = []
        self.gt_positions: list[np.ndarray] = []   # (3,) each

    def add(self, frame: Frame, gt_pos: Optional[np.ndarray] = None):
        self.frames.append(frame)
        if gt_pos is not None:
            self.gt_positions.append(gt_pos)

    def estimated_positions(self) -> np.ndarray:
        return np.array([f.t.ravel() for f in self.frames])

    def gt_positions_array(self) -> Optional[np.ndarray]:
        if not self.gt_positions:
            return None
        return np.array(self.gt_positions)


def recover_scale_from_gt(
    t_unit: np.ndarray,
    t_prev_gt: np.ndarray,
    t_curr_gt: np.ndarray,
) -> float:

    gt_disp = np.linalg.norm(t_curr_gt - t_prev_gt)
    return float(gt_disp) if gt_disp > 1e-6 else 1.0

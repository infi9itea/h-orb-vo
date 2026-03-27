"""
Visualization utilities for monocular VO.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional


def draw_tracks(
    img: np.ndarray,
    pts_prev: np.ndarray,
    pts_curr: np.ndarray,
    color: tuple = (0, 200, 0),
) -> np.ndarray:
    """Draw optical flow tracks on an image."""
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    for p0, p1 in zip(pts_prev, pts_curr):
        x0, y0 = int(p0[0]), int(p0[1])
        x1, y1 = int(p1[0]), int(p1[1])
        cv2.line(vis, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
        cv2.circle(vis, (x1, y1), 2, (0, 100, 255), -1)
    return vis


def plot_trajectory_2d(
    est_pos: np.ndarray,
    gt_pos: Optional[np.ndarray] = None,
    title: str = "Top-down Trajectory (X-Z plane)",
    save_path: Optional[str] = None,
) -> None:
    """Plot estimated vs GT trajectory top-down."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(est_pos[:, 0], est_pos[:, 2], "b-", lw=1.2, label="Estimated", alpha=0.85)
    if gt_pos is not None:
        ax.plot(gt_pos[:, 0], gt_pos[:, 2], "r--", lw=1.2, label="Ground Truth", alpha=0.85)
    ax.plot(est_pos[0, 0], est_pos[0, 2], "go", ms=8, label="Start")
    ax.plot(est_pos[-1, 0], est_pos[-1, 2], "rs", ms=8, label="End (est)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved trajectory plot → {save_path}")
    plt.show()


def plot_ate_rpe(ate_dict: dict, rpe_dict: dict, save_path: Optional[str] = None):
    """Summary plot: ATE error over frames, RPE distribution."""
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig)

    # ATE per frame
    ax1 = fig.add_subplot(gs[0])
    errors = ate_dict.get("errors", [])
    ax1.plot(errors, color="#2563eb", lw=1)
    ax1.axhline(ate_dict["rmse"], color="red", ls="--", label=f'RMSE={ate_dict["rmse"]:.3f} m')
    ax1.set_title("ATE per Frame"); ax1.set_xlabel("Frame"); ax1.set_ylabel("Error (m)")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # RPE histogram
    ax2 = fig.add_subplot(gs[1])
    te = rpe_dict.get("trans_errors", [])
    ax2.hist(te, bins=40, color="#16a34a", edgecolor="white", alpha=0.85)
    ax2.axvline(rpe_dict["trans_rmse"], color="red", ls="--",
                label=f'RMSE={rpe_dict["trans_rmse"]:.3f} m')
    ax2.set_title("RPE Translation Distribution"); ax2.set_xlabel("Error (m)"); ax2.set_ylabel("Count")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle("Monocular VO Evaluation", fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved metrics plot → {save_path}")
    plt.show()


def print_metrics(ate: dict, rpe: dict, seq: str = ""):
    header = f"=== {seq} ===" if seq else "=== Results ==="
    print(header)
    print(f"  ATE  RMSE:       {ate['rmse']:.4f} m")
    print(f"  ATE  Mean±Std:   {ate['mean']:.4f} ± {ate['std']:.4f} m")
    print(f"  RPE  Trans RMSE: {rpe['trans_rmse']:.4f} m")
    print(f"  RPE  Rot   RMSE: {rpe['rot_rmse']:.4f} °")
    print()

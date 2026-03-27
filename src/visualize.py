"""
Visualization Utilities
=======================
- draw_features       : overlay keypoints on a frame
- draw_matches        : side-by-side match visualization
- plot_trajectory_2d  : top-down X-Z trajectory plot
- plot_trajectory_3d  : 3-D trajectory (matplotlib)
- plot_error_curves   : ATE / RPE over time
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — swap to "TkAgg" / "Qt5Agg" for GUI
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# OpenCV overlays (returns BGR images for cv2 display / saving)
# ---------------------------------------------------------------------------

def draw_features(
    image: np.ndarray,
    keypoints: list,
    color: tuple = (0, 255, 0),
    radius: int = 3,
) -> np.ndarray:
    """Draw circles at keypoint locations."""
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for kp in keypoints:
        x, y = map(int, kp.pt)
        cv2.circle(out, (x, y), radius, color, 1, cv2.LINE_AA)
    return out


def draw_optical_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    pts_prev: np.ndarray,
    pts_curr: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draw optical-flow arrows on the current frame.
    Green = inlier, Red = outlier (based on mask).
    """
    h, w = curr_gray.shape
    canvas = cv2.cvtColor(curr_gray, cv2.COLOR_GRAY2BGR)

    if mask is None:
        mask = np.ones(len(pts_prev), dtype=bool)

    for i, (p, c) in enumerate(zip(pts_prev, pts_curr)):
        p_i = tuple(map(int, p))
        c_i = tuple(map(int, c))
        color = (0, 200, 0) if mask[i] else (0, 0, 200)
        cv2.arrowedLine(canvas, p_i, c_i, color, 1, cv2.LINE_AA, tipLength=0.3)
        cv2.circle(canvas, c_i, 2, color, -1)

    return canvas


def draw_feature_tracks(
    image: np.ndarray,
    tracks: list[list[tuple]],
    n_recent: int = 8,
) -> np.ndarray:
    """
    Draw the last n_recent positions of each tracked feature as a polyline.
    tracks: list of lists, each inner list is [(x, y), ...] across frames
    """
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    colors = plt.cm.hsv(np.linspace(0, 1, max(len(tracks), 1)))

    for i, track in enumerate(tracks):
        pts = track[-n_recent:]
        color = tuple(int(c * 255) for c in colors[i % len(colors)][:3])[::-1]  # BGR
        for j in range(1, len(pts)):
            cv2.line(out, tuple(map(int, pts[j-1])), tuple(map(int, pts[j])),
                     color, 1, cv2.LINE_AA)
        if pts:
            cv2.circle(out, tuple(map(int, pts[-1])), 2, color, -1)

    return out


# ---------------------------------------------------------------------------
# Matplotlib plots (save to file or return figure)
# ---------------------------------------------------------------------------

def plot_trajectory_2d(
    traj_est: np.ndarray,
    traj_gt: Optional[np.ndarray] = None,
    title: str = "Trajectory (top-down X-Z)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Top-down view of the trajectory (X-Z plane, as in KITTI convention).

    Parameters
    ----------
    traj_est : (N, 3) estimated positions
    traj_gt  : (N, 3) ground-truth positions (optional)
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)

    # Plot estimated trajectory
    ax.plot(traj_est[:, 0], traj_est[:, 2],
            color="#E85D24", linewidth=1.5, label="Estimated", zorder=3)
    ax.scatter(traj_est[0, 0], traj_est[0, 2],
               marker="o", s=80, color="#E85D24", zorder=4, label="Start (est)")

    if traj_gt is not None:
        n = min(len(traj_est), len(traj_gt))
        ax.plot(traj_gt[:n, 0], traj_gt[:n, 2],
                color="#3B8BD4", linewidth=1.5, linestyle="--",
                label="Ground truth", zorder=2)
        ax.scatter(traj_gt[0, 0], traj_gt[0, 2],
                   marker="o", s=80, color="#3B8BD4", zorder=4, label="Start (GT)")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.legend(fontsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_trajectory_3d(
    traj_est: np.ndarray,
    traj_gt: Optional[np.ndarray] = None,
    title: str = "3-D Trajectory",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """3-D trajectory plot."""
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")

    ax.plot(traj_est[:, 0], traj_est[:, 2], traj_est[:, 1],
            color="#E85D24", linewidth=1.4, label="Estimated")

    if traj_gt is not None:
        n = min(len(traj_est), len(traj_gt))
        ax.plot(traj_gt[:n, 0], traj_gt[:n, 2], traj_gt[:n, 1],
                color="#3B8BD4", linewidth=1.4, linestyle="--", label="Ground truth")

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_zlabel("Y")
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ate_over_time(
    traj_est: np.ndarray,
    traj_gt: np.ndarray,
    title: str = "ATE over time",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot per-frame position error over the sequence."""
    from metrics import align_trajectories_umeyama
    n = min(len(traj_est), len(traj_gt))
    est_aligned, *_ = align_trajectories_umeyama(
        traj_est[:n], traj_gt[:n], with_scale=True
    )
    errors = np.linalg.norm(est_aligned - traj_gt[:n], axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(errors, color="#E85D24", linewidth=1.2)
    ax.fill_between(range(len(errors)), 0, errors, alpha=0.25, color="#E85D24")
    ax.axhline(errors.mean(), color="#3B8BD4", linestyle="--",
               linewidth=1, label=f"Mean ATE = {errors.mean():.3f} m")

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Frame")
    ax.set_ylabel("ATE (m)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_diagnostics(
    diagnostics: list[dict],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot per-frame pipeline diagnostics: n_keypoints, n_matches, n_inliers.
    """
    frames    = [d["frame"]     for d in diagnostics]
    n_kps     = [d["n_kps"]     for d in diagnostics]
    n_matches = [d["n_matches"] for d in diagnostics]
    n_inliers = [d["n_inliers"] for d in diagnostics]

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(frames, n_kps,     color="#3B8BD4"); axes[0].set_ylabel("Keypoints")
    axes[1].plot(frames, n_matches, color="#639922"); axes[1].set_ylabel("Matches")
    axes[2].plot(frames, n_inliers, color="#E85D24"); axes[2].set_ylabel("Inliers")
    axes[2].set_xlabel("Frame")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Pipeline diagnostics", fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional


def draw_tracks(img, pts_prev, pts_curr, color=(0, 200, 0)):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    for p0, p1 in zip(pts_prev, pts_curr):
        cv2.line(vis, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), color, 1, cv2.LINE_AA)
        cv2.circle(vis, (int(p1[0]), int(p1[1])), 2, (0, 100, 255), -1)
    return vis


def save_harris_corners(img, pts, save_path, radius=4, color=(0, 0, 255)):

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    for p in pts:
        cv2.circle(vis, (int(p[0]), int(p[1])), radius, color, -1)
    cv2.imwrite(save_path, vis)
    print(f"Saved Harris corners -> {save_path}")


def save_feature_matches(img1, img2, pts1, pts2, save_path, max_lines=40):

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    vis1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if img1.ndim == 2 else img1.copy()
    vis2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if img2.ndim == 2 else img2.copy()
    canvas[:h1, :w1] = vis1
    canvas[:h2, w1:] = vis2

    n = min(len(pts1), len(pts2), max_lines)
    rng = np.random.default_rng(0)
    for i in range(n):
        color = tuple(int(c) for c in rng.integers(50, 255, 3))
        p1 = (int(pts1[i, 0]), int(pts1[i, 1]))
        p2 = (int(pts2[i, 0]) + w1, int(pts2[i, 1]))
        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, color, -1)
        cv2.circle(canvas, p2, 3, color, -1)

    cv2.imwrite(save_path, canvas)
    print(f"Saved feature matches -> {save_path}")


def plot_trajectory_2d(est_pos, gt_pos=None, title="Trajectory", save_path=None):

    fig, ax = plt.subplots(figsize=(8, 8))
    est_x = -est_pos[:, 0]
    est_z = est_pos[:, 2]
    if gt_pos is not None:
        ax.plot(gt_pos[:, 0], gt_pos[:, 2], color="#1f77b4", lw=1.5, label="Ground Truth")
    ax.plot(est_pos[:, 0], est_pos[:, 2], color="#ff7f0e", lw=1.5, label="Estimated")
    ax.set_title(title)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved trajectory plot -> {save_path}")
    plt.close(fig)


def save_trajectory_csv(est_pos, save_path):
    np.savetxt(save_path, est_pos, delimiter=",", header="x,y,z", comments="")
    print(f"Saved trajectory CSV -> {save_path}")


def plot_error_per_frame(est_pos, gt_pos, title="Trajectory Error", save_path=None):
    n = min(len(est_pos), len(gt_pos))
    errors = np.linalg.norm(est_pos[:n] - gt_pos[:n], axis=1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(errors, color="#1f77b4", lw=1.2)
    ax.set_title(title)
    ax.set_xlabel("Frame"); ax.set_ylabel("Error (meters)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved error plot -> {save_path}")
    plt.close(fig)


def plot_inliers_per_frame(inlier_counts, title="RANSAC Inliers per Frame", save_path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(inlier_counts, color="#1f77b4", lw=1.0)
    ax.set_title(title)
    ax.set_xlabel("Frame"); ax.set_ylabel("Inliers")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved inliers plot -> {save_path}")
    plt.close(fig)


def plot_matches_per_frame(match_counts, title="Feature Matches per Frame", save_path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(match_counts, color="#1f77b4", lw=1.0)
    ax.set_title(title)
    ax.set_xlabel("Frame"); ax.set_ylabel("Matches")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved matches plot -> {save_path}")
    plt.close(fig)


def plot_ate_rpe(ate_dict, rpe_dict, save_path=None):
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    errors = ate_dict.get("errors", [])
    ax1.plot(errors, color="#2563eb", lw=1)
    ax1.axhline(ate_dict["rmse"], color="red", ls="--", label=f'RMSE={ate_dict["rmse"]:.3f} m')
    ax1.set_title("ATE per Frame"); ax1.set_xlabel("Frame"); ax1.set_ylabel("Error (m)")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    te = rpe_dict.get("trans_errors", [])
    if len(te):
        ax2.hist(te, bins=40, color="#16a34a", edgecolor="white", alpha=0.85)
        ax2.axvline(rpe_dict["trans_rmse"], color="red", ls="--",
                    label=f'RMSE={rpe_dict["trans_rmse"]:.3f} m')
        ax2.legend()
    ax2.set_title("RPE Translation Distribution"); ax2.set_xlabel("Error (m)"); ax2.set_ylabel("Count")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Monocular VO Evaluation", fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved metrics plot -> {save_path}")
    plt.close(fig)


def print_metrics(ate, rpe, seq=""):
    header = f"=== {seq} ===" if seq else "=== Results ==="
    print(header)
    print(f"  ATE  RMSE:       {ate['rmse']:.4f} m")
    print(f"  ATE  Mean+-Std:  {ate['mean']:.4f} +- {ate['std']:.4f} m")
    print(f"  RPE  Trans RMSE: {rpe['trans_rmse']:.4f} m")
    print(f"  RPE  Rot   RMSE: {rpe['rot_rmse']:.4f} deg")
    print()

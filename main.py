"""
Main entry point for the monocular VO pipeline.

Usage
-----
# Run on KITTI:
python main.py --dataset kitti --root /path/to/kitti --seq 00

# Run on EuRoC:
python main.py --dataset euroc --root /path/to/MH_01_easy

# Quick synthetic smoke-test (no dataset needed):
python main.py --test
"""
import sys
import os
import argparse
import numpy as np
import cv2

# ── make local imports work ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from feature_tracker import FeatureTracker, TrackerConfig
from pose_estimator import PoseEstimator, PoseConfig
from trajectory import Trajectory
from metrics import compute_ate, compute_rpe
from vo_pipeline import MonocularVO, VOConfig
from visualization import plot_trajectory_2d, plot_ate_rpe, print_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic smoke-test
# ──────────────────────────────────────────────────────────────────────────────

def _render_scene(R_cw, t_cw, K, pts3d, img_size=(640, 480)):
    """Project 3-D points to a synthetic image for testing."""
    h, w = img_size[1], img_size[0]
    img = np.zeros((h, w), dtype=np.uint8)
    for p in pts3d:
        p_cam = R_cw @ p.reshape(3, 1) + t_cw
        if p_cam[2] <= 0:
            continue
        px = K @ p_cam
        px /= px[2]
        x, y = int(px[0, 0]), int(px[1, 0])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), 3, 255, -1)
    # Add some noise texture so Harris works
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    return img


def run_synthetic_test():
    print("=" * 60)
    print("SYNTHETIC SMOKE-TEST")
    print("=" * 60)

    np.random.seed(42)
    W, H = 640, 480
    fx = fy = 500.0
    K = np.array([[fx, 0, W / 2],
                  [0, fy, H / 2],
                  [0,  0,     1]], dtype=np.float64)

    # Random 3-D scene (box of points in front of camera)
    N_pts = 200
    pts3d = np.random.uniform([-5, -3, 8], [5, 3, 20], (N_pts, 3))

    # Synthetic trajectory: camera moves forward+right with small yaw
    n_frames = 60
    gt_positions = []
    frames_data = []

    for i in range(n_frames):
        angle = np.radians(i * 0.5)         # 0 → 30°
        R_y = np.array([[np.cos(angle), 0, np.sin(angle)],
                        [0,             1, 0            ],
                        [-np.sin(angle),0, np.cos(angle)]])
        t_world = np.array([[i * 0.15], [0.0], [0.0]])   # move right
        # world-to-cam:
        R_cw = R_y.T
        t_cw = -R_y.T @ t_world

        img = _render_scene(R_cw, t_cw, K, pts3d)
        gt_pos = t_world.ravel()
        frames_data.append((img, K, gt_pos))
        gt_positions.append(gt_pos)

    # ── Run VO pipeline ─────────────────────────────────────────────────
    cfg = VOConfig(verbose=True)
    cfg.tracker.detector = "harris"
    cfg.tracker.max_features = 300

    vo = MonocularVO(K, cfg)
    for img, K_frame, gt_pos in frames_data:
        vo.process_frame(img, gt_pos)

    traj = vo.trajectory
    est_pos = traj.estimated_positions()
    gt_arr = np.array(gt_positions)

    # align length
    n = min(len(est_pos), len(gt_arr))
    est_pos, gt_arr = est_pos[:n], gt_arr[:n]

    ate = compute_ate(est_pos, gt_arr)
    
    Rs_est = [f.R for f in traj.frames[:n]]
    ts_est = [f.t for f in traj.frames[:n]]
    # Build GT Rs/ts from gt_positions (pure translation in this test)
    Rs_gt = [np.eye(3)] * n
    ts_gt = [p.reshape(3, 1) for p in gt_arr]
    rpe = compute_rpe(Rs_est, ts_est, Rs_gt, ts_gt)

    print_metrics(ate, rpe, seq="Synthetic")

    # Plot (X-Z plane = X vs forward-Z, but our test uses X translation)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(est_pos[:, 0], est_pos[:, 2], "b-", label="Estimated")
        ax.plot(gt_arr[:, 0], gt_arr[:, 2], "r--", label="Ground Truth")
        ax.set_title("Synthetic Test: Trajectory (X-Z)")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = "synthetic_trajectory.png"
        plt.savefig(out, dpi=150)
        print(f"Plot saved → {out}")
    except Exception as e:
        print(f"Plot failed: {e}")

    return ate, rpe


# ──────────────────────────────────────────────────────────────────────────────
# Real-dataset runners
# ──────────────────────────────────────────────────────────────────────────────

def run_kitti(root: str, seq: str, detector: str = "harris", max_frames: int = None):
    from datasets import KITTISequence
    print(f"\nLoading KITTI seq {seq} from {root} …")
    ds = KITTISequence(root, seq)
    print(f"  {len(ds)} frames | K={ds.K.diagonal()[:2]}")

    cfg = VOConfig(verbose=True)
    cfg.tracker.detector = detector
    vo = MonocularVO(ds.K, cfg)
    vo.run(iter(ds), max_frames=max_frames)

    traj = vo.trajectory
    est_pos = traj.estimated_positions()
    gt_pos = traj.gt_positions_array()

    if gt_pos is not None:
        n = min(len(est_pos), len(gt_pos))
        ate = compute_ate(est_pos[:n], gt_pos[:n])
        Rs_est = [f.R for f in traj.frames[:n]]
        ts_est = [f.t for f in traj.frames[:n]]
        Rs_gt, ts_gt = [], []
        for T in ds.gt_poses[:n]:
            Rs_gt.append(T[:3, :3])
            ts_gt.append(T[:3, 3:4])
        rpe = compute_rpe(Rs_est, ts_est, Rs_gt, ts_gt)
        print_metrics(ate, rpe, seq=f"KITTI-{seq}")
        return ate, rpe, traj
    else:
        print("No GT available — skipping metric computation")
        return None, None, traj


def run_euroc(root: str, detector: str = "harris", max_frames: int = None):
    from datasets import EuRoCSequence
    print(f"\nLoading EuRoC from {root} …")
    ds = EuRoCSequence(root)
    print(f"  {len(ds)} frames")

    cfg = VOConfig(verbose=True)
    cfg.tracker.detector = detector
    vo = MonocularVO(ds.K, cfg)
    vo.run(iter(ds), max_frames=max_frames)

    traj = vo.trajectory
    est_pos = traj.estimated_positions()
    gt_pos = traj.gt_positions_array()

    if gt_pos is not None:
        n = min(len(est_pos), len(gt_pos))
        ate = compute_ate(est_pos[:n], gt_pos[:n])
        Rs_est = [f.R for f in traj.frames[:n]]
        ts_est = [f.t for f in traj.frames[:n]]
        Rs_gt = [np.eye(3)] * n
        ts_gt = [p.reshape(3, 1) for p in gt_pos[:n]]
        rpe = compute_rpe(Rs_est, ts_est, Rs_gt, ts_gt)
        print_metrics(ate, rpe, seq="EuRoC")
        return ate, rpe, traj
    else:
        print("No GT available")
        return None, None, traj


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Monocular VO Pipeline")
    parser.add_argument("--test", action="store_true", help="Run synthetic smoke-test")
    parser.add_argument("--dataset", choices=["kitti", "euroc"], default="kitti")
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--seq", type=str, default="00", help="KITTI sequence id")
    parser.add_argument("--detector", choices=["harris", "fast"], default="harris")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    if args.test:
        run_synthetic_test()
    elif args.dataset == "kitti":
        run_kitti(args.root, args.seq, args.detector, args.max_frames)
    elif args.dataset == "euroc":
        run_euroc(args.root, args.detector, args.max_frames)


if __name__ == "__main__":
    main()

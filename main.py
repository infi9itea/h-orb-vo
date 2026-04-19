import sys
import os
import argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))

from feature_tracker import FeatureTracker, TrackerConfig
from pose_estimator import PoseEstimator, PoseConfig
from trajectory import Trajectory
from metrics import compute_ate, compute_rpe
from vo_pipeline import MonocularVO, VOConfig
from visualization import (
    plot_trajectory_2d, save_trajectory_csv, plot_error_per_frame,
    plot_inliers_per_frame, plot_matches_per_frame,
    save_harris_corners, save_feature_matches,
    plot_ate_rpe, print_metrics,
)


def _save_all_diagnostics(vo, traj, seq, gt_positions_list, ds_frames, out_dir):
    """
    Save every diagnostic output after a run.

    Parameters
    ----------
    vo               : MonocularVO instance (has .inlier_counts, .match_counts)
    traj             : Trajectory instance
    seq              : string label e.g. "00" or "synthetic"
    gt_positions_list: list of (3,) arrays or None
    ds_frames        : list of raw (img, K, gt) tuples for the first 2 frames
    out_dir          : output directory path
    """
    os.makedirs(out_dir, exist_ok=True)

    est_pos = traj.estimated_positions()
    gt_arr  = traj.gt_positions_array()

    plot_trajectory_2d(
        est_pos, gt_arr,
        title=f"Trajectory Sequence {seq}",
        save_path=os.path.join(out_dir, f"trajectory_{seq}.png"),
    )
    save_trajectory_csv(est_pos, os.path.join(out_dir, f"trajectory_{seq}.csv"))

    if gt_arr is not None:
        n = min(len(est_pos), len(gt_arr))
        plot_error_per_frame(
            est_pos[:n], gt_arr[:n],
            title="Trajectory Error",
            save_path=os.path.join(out_dir, f"error_{seq}.png"),
        )

    if vo.inlier_counts:
        plot_inliers_per_frame(
            vo.inlier_counts,
            title="RANSAC Inliers per Frame",
            save_path=os.path.join(out_dir, f"inliers_{seq}.png"),
        )
    if vo.match_counts:
        plot_matches_per_frame(
            vo.match_counts,
            title="Feature Matches per Frame",
            save_path=os.path.join(out_dir, f"matches_{seq}.png"),
        )

    if ds_frames and len(ds_frames) >= 1:
        tracker = vo.tracker
        img0 = ds_frames[0][0]
        if img0 is not None:
            pts0 = tracker.detect(img0)
            save_harris_corners(
                img0, pts0,
                save_path=os.path.join(out_dir, "harris_corners.png"),
            )
    if ds_frames and len(ds_frames) >= 2:
        img1 = ds_frames[1][0]
        if img1 is not None:
            pts1 = tracker.detect(img1)
            save_harris_corners(
                img1, pts1,
                save_path=os.path.join(out_dir, "harris_corners1.png"),
            )

    if ds_frames and len(ds_frames) >= 2:
        img0 = ds_frames[0][0]
        img1 = ds_frames[1][0]
        if img0 is not None and img1 is not None:
            pts0 = tracker.detect(img0)
            _, pts1_tracked, _ = tracker.track(img0, img1, pts0)
            pts0_tracked = pts0[:len(pts1_tracked)]
            save_feature_matches(
                img0, img1, pts0_tracked, pts1_tracked,
                save_path=os.path.join(out_dir, f"feature_matches_{seq}.png"),
            )

    print(f"\nAll outputs saved to: {out_dir}/")



def _render_scene(R_cw, t_cw, K, pts3d, img_size=(640, 480)):
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
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    return cv2.add(img, noise)


def run_synthetic_test(out_dir="."):
    print("=" * 60)
    print("SYNTHETIC SMOKE-TEST")
    print("=" * 60)

    np.random.seed(42)
    W, H = 640, 480
    fx = fy = 500.0
    K = np.array([[fx, 0, W/2], [0, fy, H/2], [0, 0, 1]], dtype=np.float64)

    pts3d = np.random.uniform([-5, -3, 8], [5, 3, 20], (200, 3))
    n_frames = 60
    gt_positions, frames_data = [], []

    for i in range(n_frames):
        angle = np.radians(i * 0.5)
        R_y = np.array([[np.cos(angle), 0, np.sin(angle)],
                        [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)]])
        t_world = np.array([[i * 0.15], [0.0], [0.0]])
        R_cw, t_cw = R_y.T, -R_y.T @ t_world
        img = _render_scene(R_cw, t_cw, K, pts3d)
        gt_pos = t_world.ravel()
        frames_data.append((img, K, gt_pos))
        gt_positions.append(gt_pos)

    cfg = VOConfig(verbose=True)
    cfg.tracker.detector = "harris"
    cfg.tracker.max_features = 300
    vo = MonocularVO(K, cfg)
    for img, K_frame, gt_pos in frames_data:
        vo.process_frame(img, gt_pos)

    traj = vo.trajectory
    est_pos = traj.estimated_positions()
    gt_arr = np.array(gt_positions)
    n = min(len(est_pos), len(gt_arr))
    est_pos, gt_arr = est_pos[:n], gt_arr[:n]

    ate = compute_ate(est_pos, gt_arr)
    Rs_est = [f.R for f in traj.frames[:n]]
    ts_est = [f.t for f in traj.frames[:n]]
    rpe = compute_rpe(Rs_est, ts_est, [np.eye(3)]*n, [p.reshape(3,1) for p in gt_arr])
    print_metrics(ate, rpe, seq="Synthetic")

    _save_all_diagnostics(vo, traj, "synthetic", gt_positions, frames_data[:2], out_dir)
    return ate, rpe



def run_kitti(root, seq, detector="harris", max_frames=None, out_dir="."):
    from datasets import KITTISequence
    print(f"\nLoading KITTI seq {seq} from {root} ...")
    ds = KITTISequence(root, seq)
    print(f"  {len(ds)} frames | K={ds.K.diagonal()[:2]}")

    cfg = VOConfig(verbose=True)
    cfg.tracker.detector = detector
    vo = MonocularVO(ds.K, cfg)

    frames_iter = iter(ds)
    first_frames = []
    for i, item in enumerate(ds):
        if i < 2:
            first_frames.append(item)

    vo.run(iter(ds), max_frames=max_frames)
    traj = vo.trajectory
    est_pos = traj.estimated_positions()
    gt_pos  = traj.gt_positions_array()

    ate, rpe = None, None
    if gt_pos is not None and ds.gt_poses is not None:
        n = min(len(est_pos), len(gt_pos))
        ate = compute_ate(est_pos[:n], gt_pos[:n])
        Rs_est = [f.R for f in traj.frames[:n]]
        ts_est = [f.t for f in traj.frames[:n]]
        Rs_gt  = [T[:3, :3]  for T in ds.gt_poses[:n]]
        ts_gt  = [T[:3, 3:4] for T in ds.gt_poses[:n]]
        rpe = compute_rpe(Rs_est, ts_est, Rs_gt, ts_gt)
        print_metrics(ate, rpe, seq=f"KITTI-{seq}")
    else:
        print("No GT available — skipping metric computation")

    _save_all_diagnostics(vo, traj, seq, None, first_frames, out_dir)
    return ate, rpe, traj


def run_euroc(root, detector="harris", max_frames=None, out_dir="."):
    from datasets import EuRoCSequence
    print(f"\nLoading EuRoC from {root} ...")
    ds = EuRoCSequence(root)
    print(f"  {len(ds)} frames")

    cfg = VOConfig(verbose=True)
    cfg.tracker.detector = detector
    vo = MonocularVO(ds.K, cfg)

    first_frames = [item for i, item in enumerate(ds) if i < 2]
    vo.run(iter(ds), max_frames=max_frames)
    traj  = vo.trajectory
    est_pos = traj.estimated_positions()
    gt_pos  = traj.gt_positions_array()

    ate, rpe = None, None
    if gt_pos is not None:
        n = min(len(est_pos), len(gt_pos))
        ate = compute_ate(est_pos[:n], gt_pos[:n])
        Rs_est = [f.R for f in traj.frames[:n]]
        ts_est = [f.t for f in traj.frames[:n]]
        rpe = compute_rpe(Rs_est, ts_est, [np.eye(3)]*n, [p.reshape(3,1) for p in gt_pos[:n]])
        print_metrics(ate, rpe, seq="EuRoC")
    else:
        print("No GT available")

    _save_all_diagnostics(vo, traj, "euroc", None, first_frames, out_dir)
    return ate, rpe, traj


def main():
    parser = argparse.ArgumentParser(description="Monocular VO Pipeline")
    parser.add_argument("--test",    action="store_true", help="Run synthetic smoke-test")
    parser.add_argument("--dataset", choices=["kitti", "euroc"], default="kitti")
    parser.add_argument("--root",    type=str, default="")
    parser.add_argument("--seq",     type=str, default="00", help="KITTI sequence id")
    parser.add_argument("--detector",choices=["harris", "fast"], default="harris")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=".", help="Directory for output files")
    args = parser.parse_args()

    if args.test:
        run_synthetic_test(out_dir=args.out_dir)
    elif args.dataset == "kitti":
        run_kitti(args.root, args.seq, args.detector, args.max_frames, out_dir=args.out_dir)
    elif args.dataset == "euroc":
        run_euroc(args.root, args.detector, args.max_frames, out_dir=args.out_dir)


if __name__ == "__main__":
    main()

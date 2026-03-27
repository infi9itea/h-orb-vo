"""
run_kitti.py — Run the monocular VO pipeline on one or more KITTI sequences.

Usage
-----
# Single sequence
python run_kitti.py --data /path/to/kitti --seq 00

# Multiple sequences
python run_kitti.py --data /path/to/kitti --seq 00 01 02 03 04 05

# Save results to a directory
python run_kitti.py --data /path/to/kitti --seq 00 --out results/

# Visualise live (requires a display)
python run_kitti.py --data /path/to/kitti --seq 00 --live
"""

import sys
import argparse
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from vo_pipeline  import MonocularVOPipeline, CameraIntrinsics
from kitti_loader import KITTISequence, build_pipeline_for_kitti
from metrics      import compute_ate, compute_rpe, compute_drift, print_metrics
from visualize    import plot_trajectory_2d, plot_ate_over_time, plot_diagnostics


# ---------------------------------------------------------------------------
# Hard-coded KITTI sequence lengths (for progress display)
# ---------------------------------------------------------------------------
SEQ_LENGTHS = {
    "00": 4541, "01": 1101, "02": 4661, "03": 801,  "04": 271,
    "05": 2761, "06": 1101, "07": 1101, "08": 4071, "09": 1591,
    "10": 1201,
}


def run_sequence(
    kitti_root: str,
    seq_id: str,
    out_dir: Path,
    live: bool = False,
    max_frames: int = None,
) -> dict:
    """Run the VO pipeline on one KITTI sequence. Returns evaluation metrics."""

    print(f"\n{'='*60}")
    print(f"  KITTI Sequence {seq_id}")
    print(f"{'='*60}")

    seq     = KITTISequence(kitti_root, seq_id, camera_id=0)
    pipeline = build_pipeline_for_kitti(seq)

    import cv2

    t_start = time.time()
    n_frames = min(len(seq), max_frames or len(seq))

    for idx in range(n_frames):
        frame = seq.get_frame(idx)
        if frame is None:
            print(f"  [warn] Could not read frame {idx}")
            continue

        pipeline.process_frame(frame)

        if idx % 100 == 0:
            elapsed = time.time() - t_start
            fps = (idx + 1) / elapsed if elapsed > 0 else 0
            pos = pipeline.trajectory[-1] if pipeline.trajectory else [0, 0, 0]
            print(f"  Frame {idx:4d}/{n_frames}  |  fps={fps:.1f}  |  "
                  f"pos=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")

        if live:
            traj = np.array(pipeline.trajectory)
            # Simple live window (top-down view)
            canvas = np.ones((500, 500, 3), dtype=np.uint8) * 30
            if len(traj) > 1:
                pts_scaled = traj[:, [0, 2]]  # X-Z
                scale = 1.0
                origin = np.array([250, 250])
                pts_cv = (pts_scaled * scale + origin).astype(int)
                for j in range(1, len(pts_cv)):
                    p1 = tuple(np.clip(pts_cv[j-1], 0, 499))
                    p2 = tuple(np.clip(pts_cv[j],   0, 499))
                    cv2.line(canvas, p1, p2, (0, 200, 100), 1)
            cv2.putText(canvas, f"Seq {seq_id} frame {idx}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("VO trajectory", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if live:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\n  Done. {n_frames} frames in {elapsed:.1f}s  ({n_frames/elapsed:.1f} fps)")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    traj_est = pipeline.get_trajectory()
    traj_gt  = seq.get_gt_xyz()

    metrics = {}
    if traj_gt is not None:
        n = min(len(traj_est), len(traj_gt))
        ate   = compute_ate(traj_est[:n], traj_gt[:n])
        rpe   = compute_rpe(traj_est[:n], traj_gt[:n], delta=1)
        drift = compute_drift(traj_est[:n], traj_gt[:n])
        metrics = {"ate": ate, "rpe": rpe, "drift": drift}
        print_metrics(ate, rpe, drift, label=f"Sequence {seq_id}")
    else:
        print("  [info] No ground-truth poses found — skipping metric computation.")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    seq_out = out_dir / seq_id
    seq_out.mkdir(parents=True, exist_ok=True)

    # Save trajectory as text  (same format as KITTI poses)
    traj_path = seq_out / "trajectory_est.txt"
    np.savetxt(traj_path, traj_est, fmt="%.6f")
    print(f"  Trajectory saved → {traj_path}")

    # Save plots
    fig_traj = plot_trajectory_2d(
        traj_est, traj_gt,
        title=f"KITTI {seq_id} — trajectory",
        save_path=str(seq_out / "trajectory.png"),
    )

    if traj_gt is not None:
        fig_ate = plot_ate_over_time(
            traj_est[:n], traj_gt[:n],
            title=f"KITTI {seq_id} — ATE over time",
            save_path=str(seq_out / "ate_curve.png"),
        )

    fig_diag = plot_diagnostics(
        pipeline.diagnostics,
        save_path=str(seq_out / "diagnostics.png"),
    )
    print(f"  Plots saved → {seq_out}/")

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Monocular VO on KITTI")
    parser.add_argument("--data",       required=True, help="KITTI odometry root directory")
    parser.add_argument("--seq",        nargs="+", default=["00"],
                        help="Sequence ID(s) to run, e.g. 00 01 02")
    parser.add_argument("--out",        default="results", help="Output directory")
    parser.add_argument("--live",       action="store_true", help="Show live trajectory window")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit number of frames per sequence (for quick tests)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for seq_id in args.seq:
        seq_id = seq_id.zfill(2)
        try:
            m = run_sequence(
                kitti_root=args.data,
                seq_id=seq_id,
                out_dir=out_dir,
                live=args.live,
                max_frames=args.max_frames,
            )
            all_metrics[seq_id] = m
        except Exception as e:
            print(f"  [ERROR] Sequence {seq_id} failed: {e}")
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------
    # Summary table across all sequences
    # ------------------------------------------------------------------
    if any("ate" in v for v in all_metrics.values()):
        print("\n" + "="*70)
        print(f"  {'Seq':>4}  {'ATE RMSE':>10}  {'RPE trans':>10}  {'Drift %':>9}  {'GT dist':>9}")
        print("="*70)
        for seq_id, m in all_metrics.items():
            if "ate" not in m:
                continue
            print(f"  {seq_id:>4}  {m['ate']['rmse']:>10.3f}  "
                  f"{m['rpe']['trans_rmse']:>10.4f}  "
                  f"{m['drift']['translational_drift_pct']:>9.2f}  "
                  f"{m['drift']['total_distance_m']:>9.1f}")
        print("="*70)

    print(f"\nAll results written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

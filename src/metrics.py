"""
Trajectory Evaluation Metrics
==============================
Implements:
  - ATE  : Absolute Trajectory Error  (RMSE of aligned positions)
  - RPE  : Relative Pose Error        (per-step or per-n-step)
  - Translational drift %
  - Rotational drift (deg/m)

All functions follow the conventions of:
  Sturm et al. "A Benchmark for the Evaluation of RGB-D SLAM Systems" IROS 2012
  Geiger et al. KITTI benchmark
"""

import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------

def align_trajectories_umeyama(
    traj_est: np.ndarray,
    traj_gt: np.ndarray,
    with_scale: bool = False,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Align estimated trajectory to ground truth using the Umeyama method.

    Parameters
    ----------
    traj_est    : (N, 3) estimated positions
    traj_gt     : (N, 3) ground-truth positions
    with_scale  : if True, also estimate scale factor (for monocular VO)

    Returns
    -------
    traj_aligned : (N, 3) aligned estimated positions
    scale        : scalar scale factor (1.0 if with_scale=False)
    R            : 3×3 rotation aligning est → gt
    t            : 3-vector translation
    """
    assert traj_est.shape == traj_gt.shape, "Trajectories must have same shape"
    N = traj_est.shape[0]

    mu_est = traj_est.mean(0)
    mu_gt  = traj_gt.mean(0)

    # Centre both trajectories
    est_c = traj_est - mu_est
    gt_c  = traj_gt  - mu_gt

    var_est = np.mean(np.sum(est_c ** 2, axis=1))

    # Cross-covariance matrix
    W = (gt_c.T @ est_c) / N

    U, S, Vt = np.linalg.svd(W)
    det_sign = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, det_sign])

    R = U @ D @ Vt
    scale = float(np.sum(S * np.diag(D)) / var_est) if with_scale else 1.0
    t = mu_gt - scale * R @ mu_est

    traj_aligned = (scale * (R @ traj_est.T)).T + t
    return traj_aligned, scale, R, t


# ---------------------------------------------------------------------------
# ATE — Absolute Trajectory Error
# ---------------------------------------------------------------------------

def compute_ate(
    traj_est: np.ndarray,
    traj_gt: np.ndarray,
    align: bool = True,
    with_scale: bool = True,
) -> dict:
    """
    Compute ATE (RMSE of position errors after Sim(3) alignment).

    Returns
    -------
    dict with keys: rmse, mean, median, std, min, max  (all in metres)
    """
    n = min(len(traj_est), len(traj_gt))
    est = traj_est[:n]
    gt  = traj_gt[:n]

    if align:
        est, scale, R, t = align_trajectories_umeyama(est, gt, with_scale=with_scale)

    errors = np.linalg.norm(est - gt, axis=1)
    return {
        "rmse":   float(np.sqrt(np.mean(errors ** 2))),
        "mean":   float(np.mean(errors)),
        "median": float(np.median(errors)),
        "std":    float(np.std(errors)),
        "min":    float(np.min(errors)),
        "max":    float(np.max(errors)),
    }


# ---------------------------------------------------------------------------
# RPE — Relative Pose Error
# ---------------------------------------------------------------------------

def compute_rpe(
    traj_est: np.ndarray,
    traj_gt: np.ndarray,
    delta: int = 1,
) -> dict:
    """
    Compute RPE (relative pose error) over windows of `delta` frames.

    For monocular VO we measure:
      - translational RPE (RMSE, metres per sub-sequence)
      - rotational RPE   (RMSE, degrees per sub-sequence)

    Parameters
    ----------
    delta : step size in frames (1 = consecutive frames, 100 = every 100 frames)

    Returns
    -------
    dict with trans_rmse, rot_rmse (deg), trans_mean, rot_mean
    """
    n = min(len(traj_est), len(traj_gt))
    trans_errors = []
    rot_errors   = []

    for i in range(n - delta):
        j = i + delta

        # Relative motion in ground truth  (gt_i → gt_j)
        dt_gt = traj_gt[j] - traj_gt[i]

        # Relative motion in estimate
        dt_est = traj_est[j] - traj_est[i]

        # Translational error: difference in relative displacements
        trans_err = np.linalg.norm(dt_est - dt_gt)
        trans_errors.append(trans_err)

        # Rotational error: we need full SE(3) for rotation — approximate with
        # cross-product angle between displacement vectors
        norm_gt  = np.linalg.norm(dt_gt)
        norm_est = np.linalg.norm(dt_est)
        if norm_gt > 1e-6 and norm_est > 1e-6:
            cos_theta = np.clip(
                np.dot(dt_gt / norm_gt, dt_est / norm_est), -1.0, 1.0
            )
            rot_err = np.degrees(np.arccos(cos_theta))
        else:
            rot_err = 0.0
        rot_errors.append(rot_err)

    trans_errors = np.array(trans_errors)
    rot_errors   = np.array(rot_errors)

    return {
        "trans_rmse": float(np.sqrt(np.mean(trans_errors ** 2))),
        "trans_mean": float(np.mean(trans_errors)),
        "rot_rmse":   float(np.sqrt(np.mean(rot_errors ** 2))),
        "rot_mean":   float(np.mean(rot_errors)),
        "delta":      delta,
    }


# ---------------------------------------------------------------------------
# Drift metrics (KITTI style)
# ---------------------------------------------------------------------------

def compute_drift(
    traj_est: np.ndarray,
    traj_gt: np.ndarray,
) -> dict:
    """
    KITTI-style drift:
      - translational_drift_pct : mean translational error / distance travelled * 100
      - rotational_drift_deg_m  : mean rotational error (deg) per metre

    Note: for monocular VO we compute over 100-200m sub-sequences.
    """
    n = min(len(traj_est), len(traj_gt))

    total_dist  = 0.0
    total_t_err = 0.0
    total_r_err = 0.0
    count = 0

    for i in range(n - 1):
        step_gt  = np.linalg.norm(traj_gt[i + 1] - traj_gt[i])
        if step_gt < 1e-6:
            continue

        step_est = np.linalg.norm(traj_est[i + 1] - traj_est[i])
        t_err    = abs(step_est - step_gt)

        total_dist  += step_gt
        total_t_err += t_err
        count += 1

    if total_dist < 1e-6:
        return {"translational_drift_pct": 0.0, "rotational_drift_deg_m": 0.0,
                "total_distance_m": 0.0}

    return {
        "translational_drift_pct": 100.0 * total_t_err / total_dist,
        "rotational_drift_deg_m":  0.0,   # requires full SE(3) — set during Week 5
        "total_distance_m":        total_dist,
    }


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------

def print_metrics(ate: dict, rpe: dict, drift: dict, label: str = "Results"):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  ATE RMSE              : {ate['rmse']:.4f} m")
    print(f"  ATE mean              : {ate['mean']:.4f} m")
    print(f"  ATE std               : {ate['std']:.4f} m")
    print(f"  RPE translational RMSE: {rpe['trans_rmse']:.4f} m  (delta={rpe['delta']})")
    print(f"  RPE rotational RMSE   : {rpe['rot_rmse']:.4f} deg")
    print(f"  Transl. drift         : {drift['translational_drift_pct']:.2f} %")
    print(f"  Total distance (GT)   : {drift['total_distance_m']:.1f} m")
    print(f"{'='*50}\n")

"""
Standard monocular VO evaluation metrics.
  - ATE  : Absolute Trajectory Error  (RMSE of position after Sim(3) alignment)
  - RPE  : Relative Pose Error        (translation and rotation, configurable delta)
"""
import numpy as np
from scipy.linalg import logm


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """
    Least-squares Sim(3) alignment of src → dst.
    src, dst : (N, 3)
    Returns s, R, t  such that  dst ≈ s * R @ src.T + t
    """
    assert src.shape == dst.shape and src.ndim == 2 and src.shape[1] == 3
    n = src.shape[0]
    mu_src = src.mean(0)
    mu_dst = dst.mean(0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    var_src = (src_c ** 2).sum() / n
    Sigma = (dst_c.T @ src_c) / n

    U, D, Vt = np.linalg.svd(Sigma)
    det_sign = np.linalg.det(U) * np.linalg.det(Vt)
    S = np.eye(3)
    if det_sign < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    s = (D * S.diagonal()).sum() / var_src if with_scale and var_src > 0 else 1.0
    t = mu_dst - s * R @ mu_src
    return s, R, t


# ──────────────────────────────────────────────────────────────────────────────
# ATE
# ──────────────────────────────────────────────────────────────────────────────

def compute_ate(
    est: np.ndarray,
    gt: np.ndarray,
    align: bool = True,
) -> dict:
    """
    Absolute Trajectory Error (position RMSE after Sim(3) alignment).

    Parameters
    ----------
    est : (N, 3)  estimated positions
    gt  : (N, 3)  ground-truth positions
    align : whether to perform Sim(3) alignment (True for monocular)

    Returns
    -------
    dict with keys: rmse, mean, std, max, min  (all in metres)
    """
    assert len(est) == len(gt), "est and gt must have same length"
    if align:
        s, R, t = _umeyama(est, gt)
        est_aligned = (s * (R @ est.T)).T + t
    else:
        est_aligned = est

    errors = np.linalg.norm(est_aligned - gt, axis=1)
    return {
        "rmse": float(np.sqrt((errors ** 2).mean())),
        "mean": float(errors.mean()),
        "std": float(errors.std()),
        "max": float(errors.max()),
        "min": float(errors.min()),
        "errors": errors,
        "est_aligned": est_aligned,
    }


# ──────────────────────────────────────────────────────────────────────────────
# RPE
# ──────────────────────────────────────────────────────────────────────────────

def _rot_error_deg(R_err: np.ndarray) -> float:
    """Rotation angle of a rotation matrix (degrees)."""
    val = (np.trace(R_err) - 1.0) / 2.0
    val = np.clip(val, -1.0, 1.0)
    return float(np.degrees(np.arccos(val)))


def compute_rpe(
    est_Rs: list,
    est_ts: list,
    gt_Rs: list,
    gt_ts: list,
    delta: int = 1,
) -> dict:
    """
    Relative Pose Error at stride `delta`.

    Parameters
    ----------
    est_Rs, est_ts : lists of (3,3) and (3,1) cam-in-world poses
    gt_Rs,  gt_ts  : same for ground truth
    delta          : frame stride for relative motion (default 1 = consecutive)

    Returns
    -------
    dict with trans_rmse (m), trans_mean, rot_mean (deg), rot_rmse (deg)
    """
    n = len(est_Rs)
    trans_errs, rot_errs = [], []

    for i in range(n - delta):
        j = i + delta

        # Relative ground-truth motion  (world frame, cam-in-world convention)
        # T_rel_gt  = T_gt_j^{-1} * T_gt_i  in camera convention
        # We store cam-in-world:  P_world = R_c2w @ P_cam + t_c2w
        # world-to-cam:           R_w2c = R_c2w.T,  t_w2c = -R_c2w.T @ t_c2w

        def rel_pose(R_a, t_a, R_b, t_b):
            """Relative transform from frame-a to frame-b (world coords)."""
            # T_rel = T_b^{-1} @ T_a
            R_rel = R_b.T @ R_a
            t_rel = R_b.T @ (t_a - t_b)
            return R_rel, t_rel

        R_gt_rel, t_gt_rel = rel_pose(gt_Rs[i], gt_ts[i], gt_Rs[j], gt_ts[j])
        R_es_rel, t_es_rel = rel_pose(est_Rs[i], est_ts[i], est_Rs[j], est_ts[j])

        # Scale est translation to match gt (monocular only)
        gt_norm = np.linalg.norm(t_gt_rel)
        es_norm = np.linalg.norm(t_es_rel)
        if es_norm > 1e-9:
            t_es_rel_scaled = t_es_rel * (gt_norm / es_norm)
        else:
            t_es_rel_scaled = t_es_rel

        t_err = np.linalg.norm(t_es_rel_scaled - t_gt_rel)
        R_err = R_es_rel.T @ R_gt_rel
        r_err = _rot_error_deg(R_err)

        trans_errs.append(t_err)
        rot_errs.append(r_err)

    trans_errs = np.array(trans_errs)
    rot_errs = np.array(rot_errs)

    return {
        "trans_rmse": float(np.sqrt((trans_errs ** 2).mean())),
        "trans_mean": float(trans_errs.mean()),
        "trans_std": float(trans_errs.std()),
        "rot_rmse": float(np.sqrt((rot_errs ** 2).mean())),
        "rot_mean": float(rot_errs.mean()),
        "rot_std": float(rot_errs.std()),
        "trans_errors": trans_errs,
        "rot_errors": rot_errs,
    }

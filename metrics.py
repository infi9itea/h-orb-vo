import numpy as np


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



def compute_ate(
    est: np.ndarray,
    gt: np.ndarray,
    align: bool = True,
) -> dict:

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
    val = (np.trace(R_err) - 1.0) / 2.0
    val = np.clip(val, -1.0, 1.0)
    return float(np.degrees(np.arccos(val)))


def _rel_pose(R_a, t_a, R_b, t_b):
    R_rel = R_b.T @ R_a
    t_rel = R_b.T @ (t_a - t_b)
    return R_rel, t_rel


def compute_rpe(
    est_Rs: list,
    est_ts: list,
    gt_Rs: list,
    gt_ts: list,
    delta: int = 1,
) -> dict:
    n = len(est_Rs)
    trans_errs, rot_errs = [], []

    for i in range(n - delta):
        j = i + delta

        R_gt_rel, t_gt_rel = _rel_pose(gt_Rs[i], gt_ts[i], gt_Rs[j], gt_ts[j])
        R_es_rel, t_es_rel = _rel_pose(est_Rs[i], est_ts[i], est_Rs[j], est_ts[j])

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

    # BUG FIX: guard against empty arrays (n <= delta case)
    if not trans_errs:
        return {
            "trans_rmse": 0.0, "trans_mean": 0.0, "trans_std": 0.0,
            "rot_rmse": 0.0, "rot_mean": 0.0, "rot_std": 0.0,
            "trans_errors": np.array([]), "rot_errors": np.array([]),
        }

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

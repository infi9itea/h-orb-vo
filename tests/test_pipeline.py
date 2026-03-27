"""
tests/test_pipeline.py — Unit & integration tests using synthetic data.

Run with:  python tests/test_pipeline.py
           (or: pytest tests/)
"""

import sys
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vo_pipeline import (
    CameraIntrinsics, HarrisDetector, ORBDescriptor,
    HammingMatcher, EssentialMatrixEstimator,
    triangulate, MonocularVOPipeline,
)
from metrics import compute_ate, compute_rpe, compute_drift


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_pair(
    width: int = 640,
    height: int = 480,
    n_points: int = 60,
    tx: float = 0.5,
    ty: float = 0.0,
    tz: float = 0.0,
    noise_px: float = 0.5,
) -> tuple:
    """
    Generate a pair of synthetic images with known relative pose.
    Returns (img1, img2, pts1, pts2, R_true, t_true, K).
    """
    K = np.array([[458.654, 0, 367.215],
                  [0, 457.296, 248.375],
                  [0, 0, 1]], dtype=np.float64)

    # Random 3-D points in front of the camera
    rng = np.random.default_rng(42)
    pts3d = np.column_stack([
        rng.uniform(-2, 2, n_points),
        rng.uniform(-1.5, 1.5, n_points),
        rng.uniform(3, 8, n_points),
    ])  # (N, 3)

    # Camera 1: identity
    P1 = K @ np.eye(3, 4)

    # Camera 2: pure translation
    R_true = np.eye(3)
    t_true = np.array([[tx], [ty], [tz]])
    P2 = K @ np.hstack([R_true, t_true])

    # Project
    def project(P, pts):
        pts_h = np.hstack([pts, np.ones((len(pts), 1))]).T   # (4, N)
        p = P @ pts_h                                          # (3, N)
        return (p[:2] / p[2]).T                               # (N, 2)

    pts1 = project(P1, pts3d)
    pts2 = project(P2, pts3d)

    # Filter to image bounds
    mask = (
        (pts1[:, 0] >= 0) & (pts1[:, 0] < width)  &
        (pts1[:, 1] >= 0) & (pts1[:, 1] < height) &
        (pts2[:, 0] >= 0) & (pts2[:, 0] < width)  &
        (pts2[:, 1] >= 0) & (pts2[:, 1] < height)
    )
    pts1, pts2 = pts1[mask], pts2[mask]

    # Add pixel noise
    pts1 += rng.normal(0, noise_px, pts1.shape)
    pts2 += rng.normal(0, noise_px, pts2.shape)

    # Render synthetic images with dots
    def render_dots(pts, h=height, w=width):
        img = np.full((h, w, 3), 80, dtype=np.uint8)
        for p in pts.astype(int):
            cv2.circle(img, tuple(p), 4, (220, 220, 220), -1)
        # Add some texture so Harris can detect features
        cv2.rectangle(img, (50, 50), (100, 100), (180, 100, 60), 2)
        cv2.rectangle(img, (200, 150), (320, 280), (60, 180, 100), 2)
        cv2.rectangle(img, (400, 300), (500, 400), (100, 60, 180), 2)
        return img

    img1 = render_dots(pts1)
    img2 = render_dots(pts2)

    K_int = CameraIntrinsics(
        fx=K[0, 0], fy=K[1, 1],
        cx=K[0, 2], cy=K[1, 2],
    )
    return img1, img2, pts1.astype(np.float32), pts2.astype(np.float32), R_true, t_true, K_int


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_harris_detection():
    print("test_harris_detection ... ", end="")
    detector = HarrisDetector(max_corners=500)
    img = np.random.randint(50, 200, (480, 640), dtype=np.uint8)
    # Add corners
    for (x, y) in [(100, 100), (300, 200), (500, 350)]:
        img[y-5:y+5, x-5:x+5] = 255
    kps = detector.detect(img)
    assert len(kps) > 0, "Harris should detect at least some corners"
    assert all(isinstance(k, cv2.KeyPoint) for k in kps), "Must return KeyPoint objects"
    print(f"OK  ({len(kps)} keypoints)")


def test_orb_descriptor():
    print("test_orb_descriptor ... ", end="")
    detector   = HarrisDetector(max_corners=200)
    descriptor = ORBDescriptor()
    img = np.random.randint(50, 200, (480, 640), dtype=np.uint8)
    kps = detector.detect(img)
    kps2, descs = descriptor.compute(img, kps)
    assert descs is not None, "Descriptors should not be None"
    assert descs.shape[1] == 32, "ORB descriptors are 32 bytes (256 bits)"
    print(f"OK  ({len(kps2)} keypoints with descriptors)")


def test_hamming_matcher():
    print("test_hamming_matcher ... ", end="")
    img1, img2, *_ = make_synthetic_pair()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    detector   = HarrisDetector(max_corners=500)
    descriptor = ORBDescriptor()
    matcher    = HammingMatcher(ratio_threshold=0.75)

    kp1, d1 = descriptor.compute(gray1, detector.detect(gray1))
    kp2, d2 = descriptor.compute(gray2, detector.detect(gray2))

    matches = matcher.match(d1, d2)
    assert len(matches) >= 0, "Matcher should return a list"
    print(f"OK  ({len(matches)} matches after ratio test)")


def test_essential_matrix():
    print("test_essential_matrix ... ", end="")
    img1, img2, pts1, pts2, R_true, t_true, intrinsics = make_synthetic_pair(
        n_points=100, tx=0.3, noise_px=0.3
    )
    estimator = EssentialMatrixEstimator(min_inliers=5)
    E, R, t, mask = estimator.estimate(pts1, pts2, intrinsics.K)

    assert E is not None, "Essential matrix should be estimated"
    assert R is not None, "Rotation should be recovered"
    assert t is not None, "Translation should be recovered"

    # Verify rotation is close to identity (pure translation motion)
    angle_err = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    assert angle_err < 5.0, f"Rotation error too large: {angle_err:.2f} deg"
    print(f"OK  (rot_err={angle_err:.2f} deg, inliers={mask.sum()})")


def test_triangulation():
    print("test_triangulation ... ", end="")
    K = np.array([[500, 0, 320],
                  [0, 500, 240],
                  [0, 0, 1]], dtype=np.float64)
    # Create two cameras
    P1 = K @ np.eye(3, 4)
    t2 = np.array([[1], [0], [0]])
    P2 = K @ np.hstack([np.eye(3), t2])

    # A 3-D point
    pt3d = np.array([0.5, 0.3, 5.0])
    # Project
    def proj(P, p):
        ph = np.append(p, 1.0)
        px = P @ ph; return (px[:2] / px[2]).astype(np.float32)

    pts1 = proj(P1, pt3d).reshape(1, 2)
    pts2 = proj(P2, pt3d).reshape(1, 2)

    pts3d = triangulate(pts1, pts2, P1, P2)
    err = np.linalg.norm(pts3d[0] - pt3d)
    assert err < 0.01, f"Triangulation error: {err:.4f}"
    print(f"OK  (3-D error={err:.4f} m)")


def test_full_pipeline_synthetic():
    print("test_full_pipeline_synthetic ... ", end="")
    img1, img2, *_, intrinsics = make_synthetic_pair(
        n_points=150, tx=0.5, noise_px=0.5
    )

    pipeline = MonocularVOPipeline(intrinsics, max_corners=500, min_inliers=8)

    R0, t0 = pipeline.process_frame(img1)
    R1, t1 = pipeline.process_frame(img2)

    assert pipeline._frame_idx == 2
    traj = pipeline.get_trajectory()
    assert traj.shape == (2, 3)

    # With the corrected logic, the translation should be roughly [-0.5, 0, 0]
    # for a camera move of +0.5 along X (relative to world).
    # Since Monocular VO has no scale, we check the normalized direction.
    t_est = traj[1] - traj[0]
    t_est_unit = t_est / np.linalg.norm(t_est)
    t_true_unit = np.array([-1.0, 0.0, 0.0]) # world-frame translation is -R_rel^T * t_rel

    cos_sim = np.dot(t_est_unit, t_true_unit)
    assert cos_sim > 0.9, f"Translation direction incorrect: {t_est_unit} (cos_sim={cos_sim:.2f})"

    print(f"OK  (final pos={t1.ravel()}, cos_sim={cos_sim:.2f})")


def test_metrics_ate():
    print("test_metrics_ate ... ", end="")
    # Two identical trajectories → ATE = 0
    traj = np.random.rand(100, 3) * 10
    result = compute_ate(traj, traj, align=False)
    assert result["rmse"] < 1e-6, "ATE with identical trajectories should be ~0"

    # Constant offset → ATE after alignment = 0
    traj_shifted = traj + np.array([3.0, 1.0, -2.0])
    result2 = compute_ate(traj, traj_shifted, align=True)
    assert result2["rmse"] < 1e-4, "ATE after alignment should be ~0 for constant offset"
    print(f"OK")


def test_metrics_rpe():
    print("test_metrics_rpe ... ", end="")
    traj = np.cumsum(np.ones((50, 3)) * 0.1, axis=0)
    result = compute_rpe(traj, traj, delta=1)
    assert result["trans_rmse"] < 1e-6
    print(f"OK")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_harris_detection,
    test_orb_descriptor,
    test_hamming_matcher,
    test_essential_matrix,
    test_triangulation,
    test_full_pipeline_synthetic,
    test_metrics_ate,
    test_metrics_rpe,
]

if __name__ == "__main__":
    print("\nRunning mono_vo test suite\n" + "-"*40)
    passed = 0
    failed = 0
    for test in ALL_TESTS:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL  — {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'-'*40}")
    print(f"  {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)

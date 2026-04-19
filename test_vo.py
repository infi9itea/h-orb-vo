import numpy as np
import cv2
import pytest
from vo_pipeline import MonocularVO, VOConfig
from main import run_synthetic_test

def test_vo_instantiation():
    """Ensure the MonocularVO class can be instantiated."""
    K = np.eye(3)
    cfg = VOConfig()
    vo = MonocularVO(K, cfg)
    assert vo is not None
    assert vo.K.shape == (3, 3)

def test_synthetic_run():
    """Test the synthetic smoke-test for basic functionality."""
    # This might take a few seconds, but ensures all parts (tracker, estimator, trajectory) work together.
    ate, rpe = run_synthetic_test()

    # Simple sanity checks on the results
    assert "rmse" in ate
    assert "trans_rmse" in rpe
    assert ate["rmse"] < 2.0  # Synthetic test should be reasonably accurate

import numpy as np
import cv2
import pytest
from vo_pipeline import MonocularVO, VOConfig
from main import run_synthetic_test

def test_vo_instantiation():

    K = np.eye(3)
    cfg = VOConfig()
    vo = MonocularVO(K, cfg)
    assert vo is not None
    assert vo.K.shape == (3, 3)

def test_synthetic_run():

    ate, rpe = run_synthetic_test()

    assert "rmse" in ate
    assert "trans_rmse" in rpe
    assert ate["rmse"] < 2.0 

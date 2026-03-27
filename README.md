# Monocular Visual Odometry — Week 1–2 Baseline

Harris + ORB + RANSAC Essential Matrix pipeline implemented in Python/OpenCV.

## Project structure

```
mono_vo/
├── src/
│   ├── vo_pipeline.py   # Core pipeline: all 7 stages
│   ├── kitti_loader.py  # KITTI sequence reader + calibration parser
│   ├── metrics.py       # ATE, RPE, translational drift, rotational drift
│   └── visualize.py     # Trajectory plots, feature overlays, diagnostics
├── tests/
│   └── test_pipeline.py # Unit tests (synthetic data, no KITTI needed)
├── run_kitti.py         # CLI entry point for KITTI evaluation
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Run tests (no KITTI needed)

```bash
python tests/test_pipeline.py
```

## Run on KITTI

Download the [KITTI odometry dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).
Unzip so the structure is:
```
/path/to/kitti/
    sequences/00/image_0/*.png
    sequences/00/calib.txt
    poses/00.txt
```

Then:
```bash
# Single sequence
python run_kitti.py --data /path/to/kitti --seq 00

# All sequences 00–10
python run_kitti.py --data /path/to/kitti --seq 00 01 02 03 04 05 06 07 08 09 10

# Quick test (100 frames)
python run_kitti.py --data /path/to/kitti --seq 00 --max-frames 100
```

Results (trajectories, plots, metrics) are saved to `results/<seq_id>/`.

## Pipeline stages

| Stage | Class / function | File |
|---|---|---|
| 1. Harris corner detection | `HarrisDetector` | `vo_pipeline.py` |
| 2. ORB descriptor computation | `ORBDescriptor` | `vo_pipeline.py` |
| 3. Hamming BFMatcher + ratio test | `HammingMatcher` | `vo_pipeline.py` |
| 4. RANSAC Essential matrix | `EssentialMatrixEstimator` | `vo_pipeline.py` |
| 5. Pose recovery (R, t) | `EssentialMatrixEstimator.estimate()` | `vo_pipeline.py` |
| 6. Triangulation | `triangulate()` | `vo_pipeline.py` |
| 7. Trajectory accumulation | `MonocularVOPipeline.process_frame()` | `vo_pipeline.py` |

## Key parameters

| Parameter | Default | Notes |
|---|---|---|
| `max_corners` | 2000 | Max Harris corners per frame |
| `ratio_threshold` | 0.75 | Lowe's ratio test threshold |
| `ransac_threshold` | 1.0 px | RANSAC inlier threshold |
| `ransac_prob` | 0.999 | RANSAC confidence |
| `min_inliers` | 15 | Min inliers to accept a frame |

## Week 3–4 extension point

To add your novelty (e.g. adaptive regional thresholding), subclass `HarrisDetector`
or pass a custom detector to `MonocularVOPipeline`:

```python
class AdaptiveRegionalHarris(HarrisDetector):
    def detect(self, gray):
        # Divide image into regions, threshold each independently
        ...
```

## Metrics

- **ATE** : Absolute Trajectory Error (RMSE after Sim(3) alignment)
- **RPE** : Relative Pose Error (per-frame translational + rotational)
- **Drift %** : translational drift as percentage of distance travelled

## Known limitations (Week 1–2 baseline)

- **Monocular scale ambiguity**: translation is normalised to unit length each frame.
  Scale recovery requires ground-truth or stereo/depth — addressed in Week 5.
- **No loop closure**: drift accumulates over long sequences.
- **No bundle adjustment**: no global refinement of landmark positions.

# Monocular Visual Odometry (VO) Pipeline

This repository implements a frame-to-frame monocular VO pipeline for the KITTI (odometry) and EuRoC MAV datasets. The goal is to estimate the camera's trajectory through space using a sequence of images.

## Features

- **Feature Tracking**: Harris or FAST corner detection + Lucas-Kanade optical flow.
- **Pose Estimation**: 5-point algorithm (via Essential Matrix), RANSAC, and relative motion decomposition.
- **Scale Recovery**: For monocular VO, the scale is recovered using ground-truth displacement (following the standard monocular VO protocol).
- **Evaluation**: Computes Absolute Trajectory Error (ATE) and Relative Pose Error (RPE).
- **Visualization**: Trajectory plots (top-down X-Z plane) and error histograms.

## Requirements

Install dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Usage

### Command-Line Interface (CLI)

#### Run synthetic smoke-test:
```bash
python main.py --test
```

#### Run on KITTI:
```bash
python main.py --dataset kitti --root /path/to/kitti --seq 00 --max-frames 500
```

#### Run on EuRoC:
```bash
python main.py --dataset euroc --root /path/to/MH_01_easy
```

### Jupyter Notebook

A `demo.ipynb` notebook is provided to demonstrate the pipeline, which is especially useful for environments like Kaggle. It includes the synthetic test and a skeleton for running on KITTI.

## Repository Structure

- `main.py`: Main entry point and synthetic test runner.
- `vo_pipeline.py`: Core `MonocularVO` pipeline class.
- `feature_tracker.py`: Feature detection and tracking.
- `pose_estimator.py`: Pose recovery and triangulation.
- `trajectory.py`: Trajectory book-keeping and scale recovery.
- `datasets.py`: Loaders for KITTI and EuRoC datasets.
- `metrics.py`: ATE and RPE metric implementations.
- `visualization.py`: Utilities for drawing tracks and plotting trajectories.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

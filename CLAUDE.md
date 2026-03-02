# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python project for multi-object detection and tracking using Ultralytics YOLO. The main use cases are running YOLO inference on video files/streams with trajectory visualization, and fine-tuning YOLO models on custom datasets (VisDrone, COCO).

## Setup

No package manager files exist. Install dependencies manually:

```bash
pip install ultralytics opencv-python numpy
```

Pre-trained model weights are checked into the repo: `yolo26n.pt`, `yolo11n.pt`, `yolo26n-pose.pt`, `yolo26n-seg.pt`. The best fine-tuned weights live at `runs/detect/train3/weights/best.pt`.

## Running Scripts

```bash
# Basic tracking with trajectory history (primary example)
python example_tracking_overtime.py

# Streaming variant (more memory-efficient)
python example_tracking_overtime_stream.py

# Frame-by-frame tracking (non-streaming)
python example_persisting_tracking.py

# Train on VisDrone dataset
python example_train.py
```

Each script has video source paths and model paths commented at the top — edit those to switch inputs.

## Architecture

All scripts follow the same pipeline:

1. **Load model**: `YOLO("yolo26n.pt")` or a fine-tuned checkpoint
2. **Filter classes**: `class2detect = [i for i, name in model.names.items() if name == "person"]`
3. **Run tracking**: `model.track(source=..., stream=True, persist=True, tracker="bytetrack.yaml")`
4. **Visualize**: `result.plot()` draws boxes; OpenCV `polylines` draws trajectory history stored in a `defaultdict`

**Trackers**: `bytetrack.yaml` (used for inference) and `botsort.yaml` (used during training runs).

**Datasets**:
- `datasets/coco8/` — 8-image mini COCO for quick validation
- `datasets/coco/` — Full COCO 2017
- `datasets/VisDrone/` — Drone imagery dataset; training config at `VisDrone.yaml`

## Device Configuration

- **Inference**: use `device="mps"` on Apple Silicon Macs
- **Training**: use `device="cpu"` — MPS has a shape-mismatch bug in Ultralytics' TAL assigner that causes training to fail

## Key Parameters

- `vid_stride=2` — process every 2nd frame for speed
- `imgsz=640` — standard inference size
- `persist=True` — required for consistent track IDs across frames
- Track history length: 30 points per track (see `if len(track) > 30`)

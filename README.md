# RobotTracking

Multi-object detection and tracking for FRC (FIRST Robotics Competition) robots using Ultralytics YOLO and TrOCR. Tracks a single robot by bumper number through a broadcast video and renders its trajectory on a top-down field map.

## Features

- YOLO-based robot detection with ByteTrack for persistent tracking
- TrOCR bumper number recognition with vote-based confirmation
- Top-down field trajectory via perspective homography
- Interactive setup UI: field corner selection, sub-camera ROI, robot click-to-initialize
- Handles occlusion and track-ID reassignment robustly

## Setup

```bash
pip install ultralytics opencv-python numpy torch transformers pillow
```

Pre-trained base weights are included: `yolo26n.pt`, `yolo11n.pt`.
Fine-tuned FRC robot detector: `runs/detect/robot_detector3/weights/best.pt`

## Main Script

**`plot_robot_traces_trocr.py`** — primary tracking script (TrOCR, best accuracy)

Edit the config section at the top:

```python
VIDEO_SOURCE      = "path/to/your/video.mp4"
TARGET_NUMBER     = "2073"                          # bumper number to track
ALL_ROBOT_NUMBERS = ["2910","1323","4272","2073","4414","1690"]  # all 6 robots
FIELD_CORNERS     = [(142,73),(1777,73),(1777,985),(142,985)]    # or None for interactive
BLUE_CAM_ROI      = None   # (x1,y1,x2,y2) bottom-left sub-camera, or None
RED_CAM_ROI       = None   # (x1,y1,x2,y2) bottom-right sub-camera, or None
```

Run:
```bash
python plot_robot_traces_trocr.py
```

### 3-Phase Workflow

| Phase | What happens |
|-------|-------------|
| **Phase 1** | Click 4 field corners (TL→TR→BR→BL) for homography. Skipped if `FIELD_CORNERS` is set. |
| **Phase 1.5** | Drag ROI boxes over the blue and red sub-camera insets. Skipped if `BLUE_CAM_ROI`/`RED_CAM_ROI` are set. |
| **Phase 2** | Scrub to a good frame with the seekbar, click on the target robot, press Enter. |
| **Phase 3** | Tracking runs. Two windows: live video with bounding boxes, and top-down field map with trajectory. Press **Q** to quit. |

A tracking summary (total time, time tracked, % coverage) is printed on exit.

## Other Scripts

| Script | Description |
|--------|-------------|
| `plot_robot_traces.py` | EasyOCR variant |
| `plot_robot_traces_paddleocr.py` | PaddleOCR variant |
| `plot_robot_traces_template.py` | Template matching variant (no OCR) |
| `track_robots_by_number.py` | Labels all robots on screen by number |
| `train_robot_detector.py` | Fine-tune YOLO on FRC robot images |
| `download_frc_dataset.py` | Download FRC robot image dataset |
| `score_tracker.py` | Score overlay tracking |
| `example_tracking_overtime.py` | Basic YOLO tracking with trajectory history |

## Device Notes

- **Inference**: `device="mps"` (Apple Silicon) or `"cuda"` / `"cpu"`
- **Training**: use `device="cpu"` — MPS has a shape-mismatch bug in Ultralytics' TAL assigner
- TrOCR runs on MPS automatically when available

## How Re-ID Works

When the target robot disappears and reappears:

1. **Priority 1**: re-attach to any track already confirmed as the target number (within `LOST_RADIUS=200px`)
2. **Priority 2**: fall back to the nearest unconfirmed track (within tight `UNCONFIRMED_RADIUS=60px`)
3. Tracks confirmed as a *different* robot are fully excluded
4. OCR runs every 3 frames on all detections so other robots get confirmed quickly, shrinking the pool of ambiguous candidates

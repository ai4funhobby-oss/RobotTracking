"""
Fine-tune yolo26n.pt to detect red and blue FRC robots.

Run download_frc_dataset.py first, then:
    python train_robot_detector.py

Trained weights will be at:
    runs/detect/robot_detector/weights/best.pt
"""

from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Use CPU — MPS has a shape-mismatch bug in Ultralytics' TAL assigner during training.
# For faster training, use a CUDA machine and set device="0".
results = model.train(
    data="datasets/frc-robots_2023/data.yaml",
    epochs=50,
    imgsz=640,
    device="mps",
    batch=16,
    name="robot_detector",
    patience=10,          # stop early if val loss plateaus
)

print("\nTraining complete.")
print("Best weights: runs/detect/robot_detector/weights/best.pt")

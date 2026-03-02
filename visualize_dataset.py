"""
Visualize YOLO-format dataset annotations.
Press any key to advance, 'q' to quit.
"""

import cv2
import numpy as np
from pathlib import Path

IMAGES_DIR = Path("datasets/frc-robots_2023/train/images")
LABELS_DIR = Path("datasets/frc-robots_2023/train/labels")
# IMAGES_DIR = Path("datasets/frc-robots2/train/images")
# LABELS_DIR = Path("datasets/frc-robots2/train/labels")
CLASS_NAMES = ["blue", "red"]
COLORS = {0: (220, 100, 0), 1: (0, 60, 220)}  # BGR: blue, red

for img_path in sorted(IMAGES_DIR.glob("*.jpg")):
    label_path = LABELS_DIR / img_path.with_suffix(".txt").name
    if not label_path.exists():
        continue

    frame = cv2.imread(str(img_path))
    h, w = frame.shape[:2]

    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        cls = int(parts[0])
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        color = COLORS.get(cls, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, CLASS_NAMES[cls], (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Dataset Viewer", frame)
    key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()

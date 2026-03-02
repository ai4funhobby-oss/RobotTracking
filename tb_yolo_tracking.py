from ultralytics import YOLO
import cv2

# 1. Load the model (YOLO11 is the current 2026 peak)
model = YOLO("yolo11n.pt") 

# 2. Run tracking on a video file or webcam
# 'persist=True' tells the model to remember IDs across frames
results = model.track(source="robot_video.mp4", show=True, tracker="bytetrack.yaml", persist=True)

# The 'results' object now contains:
# - Bounding boxes [x1, y1, x2, y2]
# - Track IDs (e.g., ID: 1, ID: 2)
# - Confidence scores
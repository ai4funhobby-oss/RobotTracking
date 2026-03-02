import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture("video_sources/Einstein Final Tiebreaker - 2025 FIRST Championship.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 120)
ret, frame = cap.read(); cap.release()
print("Frame shape:", frame.shape)

model = YOLO("runs/detect/robot_detector3/weights/best.pt")
res = model(frame, imgsz=640, device="mps", verbose=False)[0]
for box in res.boxes.xyxy.cpu():
    x1, y1, x2, y2 = map(int, box)
    print(f"  center ({(x1+x2)//2}, {(y1+y2)//2}), size {x2-x1}×{y2-y1}")
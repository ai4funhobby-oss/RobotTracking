import cv2
import numpy as np
import easyocr
from collections import defaultdict, Counter
from ultralytics import YOLO

PALETTE = [
    (0, 255, 0),
    (0, 128, 255),
    (255, 0, 128),
    (255, 255, 0),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (255, 0, 0),
]
OCR_EVERY_N_FRAMES = 5
VOTE_THRESHOLD = 3

VIDEO_SOURCE = "/Users/ai4hobby/Playground/RobotTracking/video_sources/Einstein Final 1 - 2025 FIRST Championship.mp4"
VIDEO_SOURCE = "/Users/ai4hobby/Playground/RobotTracking/video_sources/Einstein Final Tiebreaker - 2025 FIRST Championship.mp4"

# After training, switch to: "runs/detect/robot_detector/weights/best.pt"
MODEL_WEIGHTS = "yolo26n.pt"
MODEL_WEIGHTS = "runs/detect/robot_detector3/weights/best.pt"

model = YOLO(MODEL_WEIGHTS)
reader = easyocr.Reader(['en'], gpu=True)

print(model.names)
input()
# After training, filter to robot classes only:
#   robot_classes = [i for i, n in model.names.items() if "robot" in n.lower()]
# and pass classes=robot_classes to model.track() below.

results = model.track(
    source=VIDEO_SOURCE,
    stream=True,
    persist=True,
    imgsz=640,
    vid_stride=1,
    device="mps",
    tracker="bytetrack.yaml",
)

confirmed = {}               # track_id → number string
votes = defaultdict(Counter)
track_history = defaultdict(list)
frame_count = 0

for result in results:
    frame_count += 1
    if not (result.boxes and result.boxes.is_track):
        cv2.imshow("Robot Tracker", result.orig_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    frame = result.orig_img.copy()
    boxes_xyxy = result.boxes.xyxy.cpu()
    boxes_xywh = result.boxes.xywh.cpu()
    track_ids = result.boxes.id.int().cpu().tolist()

    # --- OCR pass (only on unconfirmed tracks, every N frames) ---
    if frame_count % OCR_EVERY_N_FRAMES == 0:
        for box_xyxy, track_id in zip(boxes_xyxy, track_ids):
            if track_id in confirmed:
                continue
            x1, y1, x2, y2 = map(int, box_xyxy)
            h = y2 - y1
            crop = frame[y1 + int(h * 0.6):y2, x1:x2]
            if crop.size == 0:
                continue
            crop_up = cv2.resize(crop, None, fx=2, fy=2)
            texts = reader.readtext(crop_up, detail=0, allowlist='0123456789')
            for text in texts:
                text = text.strip()
                if text:
                    votes[track_id][text] += 1
                    top_num, top_count = votes[track_id].most_common(1)[0]
                    if top_count >= VOTE_THRESHOLD:
                        confirmed[track_id] = top_num

    # --- Draw only confirmed tracks ---
    for box_xyxy, box_xywh, track_id in zip(boxes_xyxy, boxes_xywh, track_ids):
        if track_id not in confirmed:
            continue
        label = confirmed[track_id]
        color = PALETTE[track_id % len(PALETTE)]
        x1, y1, x2, y2 = map(int, box_xyxy)
        cx, cy = float(box_xywh[0]), float(box_xywh[1])

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f"#{label}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        track_history[track_id].append((cx, cy))
        if len(track_history[track_id]) > 40:
            track_history[track_id].pop(0)
        if len(track_history[track_id]) >= 2:
            pts = np.array(track_history[track_id], np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], False, (255, 255, 255), 3)

    cv2.imshow("Robot Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

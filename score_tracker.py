import cv2
import numpy as np
import easyocr
from collections import defaultdict, Counter
from ultralytics import YOLO

# --- Config ---
VIDEO_SOURCE = "/Users/ai4hobby/Playground/RobotTracking/video_sources/Einstein Final Tiebreaker - 2025 FIRST Championship.mp4"
MODEL_WEIGHTS = "runs/detect/robot_detector3/weights/best.pt"

OCR_EVERY_N_FRAMES = 5       # robot bumper OCR cadence
SCORE_OCR_EVERY_N = 10       # scoreboard OCR cadence
VOTE_THRESHOLD = 3            # frames needed to confirm a bumper number
ACTIVITY_WINDOW = 60          # frames of movement history for attribution

# Scoreboard score crop regions (1920x1080 frame, y=50:130 strip)
BLUE_SCORE_ROI = (760, 50, 930, 130)
RED_SCORE_ROI  = (1010, 50, 1180, 130)

# Alliance colors (BGR)
BLUE_COLOR = (220, 100, 0)
RED_COLOR  = (0, 60, 220)

# Per-track palette fallback (used before alliance is known)
PALETTE = [
    (0, 255, 0), (0, 128, 255), (255, 0, 128), (255, 255, 0),
    (0, 255, 255), (255, 128, 0), (128, 0, 255), (255, 0, 0),
]

# --- Init ---
model = YOLO(MODEL_WEIGHTS)
reader = easyocr.Reader(['en'], gpu=True)

# robot_classes = [i for i, n in model.names.items() if 'robot' in n.lower()]
# print("Robot class IDs:", {i: model.names[i] for i in robot_classes})

results = model.track(
    source=VIDEO_SOURCE,
    stream=True,
    persist=True,
    imgsz=640,
    vid_stride=2,
    device="mps",
    tracker="bytetrack.yaml",
    # classes=robot_classes,
)

# --- State ---
confirmed      = {}                  # track_id → bumper number string
votes          = defaultdict(Counter)
track_history  = defaultdict(list)   # track_id → [(cx, cy), ...]
track_alliance = {}                  # track_id → 'blue' | 'red'
robot_points   = defaultdict(int)    # bumper number → accumulated points
activity       = defaultdict(list)   # track_id → recent displacements

score_buf_blue: list[int] = []
score_buf_red:  list[int] = []
last_blue: int | None = None
last_red:  int | None = None
frame_count = 0


def ocr_score(crop) -> int | None:
    if crop.size == 0:
        return None
    up = cv2.resize(crop, None, fx=2, fy=2)
    texts = reader.readtext(up, detail=0, allowlist='0123456789')
    for t in texts:
        t = t.strip()
        if t.isdigit():
            return int(t)
    return None


def smooth_score(buf: list[int], new_val: int | None, window: int = 3) -> int | None:
    """Accept reading only if consistent across recent frames."""
    if new_val is not None:
        buf.append(new_val)
    if len(buf) > window:
        buf.pop(0)
    if not buf:
        return None
    return Counter(buf).most_common(1)[0][0]


def most_active_robot(alliance: str) -> str | None:
    """Return bumper number of most recently active robot on the given alliance."""
    candidates = [
        tid for tid, num in confirmed.items()
        if track_alliance.get(tid) == alliance
    ]
    if not candidates:
        return None
    # pick the one with highest recent displacement sum
    best = max(candidates, key=lambda tid: sum(activity[tid][-ACTIVITY_WINDOW:]))
    return confirmed[best]


def draw_scoreboard(frame, blue: int | None, red: int | None):
    """Draw a running totals panel in the bottom-left corner."""
    panel_x, panel_y = 20, frame.shape[0] - 20
    entries = sorted(robot_points.items(), key=lambda x: -x[1])
    for num, pts in entries:
        # find alliance for this number
        alliance = next(
            (track_alliance.get(tid) for tid, n in confirmed.items() if n == num),
            None,
        )
        color = BLUE_COLOR if alliance == 'blue' else RED_COLOR if alliance == 'red' else (200, 200, 200)
        text = f"#{num}: {pts} pts"
        panel_y -= 30
        cv2.putText(frame, text, (panel_x, panel_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Alliance totals from scoreboard OCR
    if blue is not None:
        cv2.putText(frame, f"Blue total: {blue}", (panel_x, panel_y - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLUE_COLOR, 2)
    if red is not None:
        cv2.putText(frame, f"Red total: {red}", (panel_x, panel_y - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED_COLOR, 2)


# --- Main loop ---
for result in results:
    frame_count += 1

    if not (result.boxes and result.boxes.is_track):
        cv2.imshow("Score Tracker", result.orig_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    frame      = result.orig_img.copy()
    boxes_xyxy = result.boxes.xyxy.cpu()
    boxes_xywh = result.boxes.xywh.cpu()
    track_ids  = result.boxes.id.int().cpu().tolist()
    classes    = result.boxes.cls.int().cpu().tolist()

    # --- Record alliance for each track ---
    for cls, tid in zip(classes, track_ids):
        if tid not in track_alliance:
            name = model.names[cls]
            if 'blue' in name:
                track_alliance[tid] = 'blue'
            elif 'red' in name:
                track_alliance[tid] = 'red'

    # --- Scoreboard OCR ---
    if frame_count % SCORE_OCR_EVERY_N == 0:
        x1, y1, x2, y2 = BLUE_SCORE_ROI
        new_blue = ocr_score(frame[y1:y2, x1:x2])
        x1, y1, x2, y2 = RED_SCORE_ROI
        new_red = ocr_score(frame[y1:y2, x1:x2])

        cur_blue = smooth_score(score_buf_blue, new_blue)
        cur_red  = smooth_score(score_buf_red,  new_red)

        # Detect increase and attribute
        if last_blue is not None and cur_blue is not None and cur_blue > last_blue:
            robot = most_active_robot('blue')
            if robot:
                robot_points[robot] += cur_blue - last_blue
        if last_red is not None and cur_red is not None and cur_red > last_red:
            robot = most_active_robot('red')
            if robot:
                robot_points[robot] += cur_red - last_red

        if cur_blue is not None:
            last_blue = cur_blue
        if cur_red is not None:
            last_red = cur_red

    # --- Bumper OCR (unconfirmed tracks only) ---
    if frame_count % OCR_EVERY_N_FRAMES == 0:
        for box_xyxy, tid in zip(boxes_xyxy, track_ids):
            if tid in confirmed:
                continue
            x1, y1, x2, y2 = map(int, box_xyxy)
            h = y2 - y1
            crop = frame[y1 + int(h * 0.6):y2, x1:x2]
            if crop.size == 0:
                continue
            up = cv2.resize(crop, None, fx=2, fy=2)
            texts = reader.readtext(up, detail=0, allowlist='0123456789')
            for t in texts:
                t = t.strip()
                if t:
                    votes[tid][t] += 1
                    top, count = votes[tid].most_common(1)[0]
                    if count >= VOTE_THRESHOLD:
                        confirmed[tid] = top

    # --- Update activity & trajectory ---
    for box_xywh, tid in zip(boxes_xywh, track_ids):
        cx, cy = float(box_xywh[0]), float(box_xywh[1])
        hist = track_history[tid]
        if hist:
            dx = cx - hist[-1][0]
            dy = cy - hist[-1][1]
            activity[tid].append((dx**2 + dy**2) ** 0.5)
        hist.append((cx, cy))
        if len(hist) > 40:
            hist.pop(0)

    # --- Draw confirmed tracks ---
    for box_xyxy, tid in zip(boxes_xyxy, track_ids):
        if tid not in confirmed:
            continue
        label    = confirmed[tid]
        alliance = track_alliance.get(tid)
        color    = BLUE_COLOR if alliance == 'blue' else RED_COLOR if alliance == 'red' else PALETTE[tid % len(PALETTE)]

        x1, y1, x2, y2 = map(int, box_xyxy)
        pts_label = f"#{label}  {robot_points[label]}pts"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, pts_label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        hist = track_history[tid]
        if len(hist) >= 2:
            pts = np.array(hist, np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], False, (255, 255, 255), 2)

    draw_scoreboard(frame, last_blue, last_red)

    cv2.imshow("Score Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

"""
2D field trace plotter — single-robot focus mode (OCR: PaddleOCR).

pip install paddlepaddle paddleocr

Phase 1: Seek to a frame with the field visible.
         Click 4 corners: Top-Left → Top-Right → Bottom-Right → Bottom-Left
         Press Enter.
Phase 2: Click on the robot you want to track, press Enter.
Phase 3: Tracking runs. Robot trajectory is drawn on a top-down field map.

Press Q to quit.
"""

import cv2
import numpy as np
from paddleocr import PaddleOCR
from collections import defaultdict, Counter
from ultralytics import YOLO

# --- Config ---
VIDEO_SOURCE       = "/Users/ai4hobby/Playground/RobotTracking/video_sources/Einstein Final Tiebreaker - 2025 FIRST Championship.mp4"
MODEL_WEIGHTS      = "runs/detect/robot_detector3/weights/best.pt"
MAP_W, MAP_H       = 900, 450
OCR_EVERY_N_FRAMES = 5
VOTE_THRESHOLD     = 3
TARGET_NUMBER      = "2073"
ALL_ROBOT_NUMBERS  = ["2910","1323","4272","2073","4414","1690"]
LOST_RADIUS        = 200   # pixels — proximity window for re-ID after track loss
# Hard-code field corners to skip Phase 1 (TL, TR, BR, BL in pixel coords).
# Set to None to use the interactive corner-clicking UI instead.
FIELD_CORNERS      = [(142, 73), (1777, 73), (1777, 985), (142, 985)]
SKIP_SECONDS       = 4    # seconds to skip before showing the Phase 2 click frame

BLUE_COLOR = (220, 100,   0)
RED_COLOR  = (  0,  60, 220)
GRAY_COLOR = (160, 160, 160)

# --- Load model & OCR ---
model      = YOLO(MODEL_WEIGHTS)
ocr_engine = PaddleOCR(use_textline_orientation=False, lang='en')
clahe      = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
print("Classes:", model.names)


def alliance_color(cls_id):
    name = model.names.get(cls_id, "")
    if "blue" in name.lower(): return BLUE_COLOR
    if "red"  in name.lower(): return RED_COLOR
    return GRAY_COLOR


def ocr_texts(img):
    """Run PaddleOCR and return a list of recognised text strings."""
    results = ocr_engine.predict(img)
    texts = []
    for res in (results or []):
        if hasattr(res, 'rec_texts'):          # v3 object API
            texts.extend(res.rec_texts)
        elif isinstance(res, dict):            # v3 dict API
            texts.extend(res.get('rec_texts', []))
        elif isinstance(res, list):            # legacy nested-list API
            for line in res:
                if line and len(line) >= 2:
                    texts.append(line[1][0])
    return texts


def best_known_match(ocr_text, known_numbers):
    """Snap OCR text to the closest known robot number (edit distance ≤ 2)."""
    def edit_dist(a, b):
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n + 1):
                dp[j], prev = min(dp[j] + 1, dp[j-1] + 1,
                                  prev + (a[i-1] != b[j-1])), dp[j]
        return dp[n]
    best = min(known_numbers, key=lambda n: edit_dist(ocr_text, n))
    return best if edit_dist(ocr_text, best) <= 2 else None


# --------------------------------------------------------------------------
# Phase 1: seek to field view, click 4 corners  (skipped if FIELD_CORNERS set)
# --------------------------------------------------------------------------
cap = cv2.VideoCapture(VIDEO_SOURCE)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
current_frame = [0]


def read_frame(idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, f = cap.read()
    return f if ret else None


if FIELD_CORNERS:
    corners = list(FIELD_CORNERS)
    print(f"Using hard-coded field corners: {corners}")
    phase2_img = read_frame(int(fps * SKIP_SECONDS))
    cap.release()
else:
    corners = []

    def redraw(frame_idx):
        f = read_frame(frame_idx)
        if f is None:
            return
        display = f.copy()
        for i, (cx, cy) in enumerate(corners):
            cv2.circle(display, (cx, cy), 8, (0, 255, 0), -1)
            cv2.putText(display, ["TL","TR","BR","BL"][i], (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        msg = f"Seek to field view. Click TL->TR->BR->BL. Enter to start.  [{int(frame_idx/fps)}s]"
        cv2.putText(display, msg, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow("Define field corners", display)

    def on_seekbar(val):
        corners.clear()
        current_frame[0] = val
        redraw(val)

    def on_corner_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append((x, y))
            redraw(current_frame[0])
            if len(corners) == 4:
                print(f"All 4 corners set — press Enter to start tracking.")
                print(f"To skip this step next time, set FIELD_CORNERS = {corners}")

    cv2.namedWindow("Define field corners")
    cv2.setMouseCallback("Define field corners", on_corner_click)
    cv2.createTrackbar("Frame", "Define field corners", 0, total_frames - 1, on_seekbar)
    redraw(0)

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 13 and len(corners) == 4:
            break
        if key == ord("q"):
            cap.release(); cv2.destroyAllWindows(); raise SystemExit

    phase2_img = read_frame(current_frame[0])   # save before cap is released
    cap.release()
    cv2.destroyWindow("Define field corners")

# Homography: clicked corners → top-down map
src_pts = np.array(corners, dtype=np.float32)
dst_pts = np.array([[0, 0], [MAP_W, 0], [MAP_W, MAP_H], [0, MAP_H]], dtype=np.float32)
H, _ = cv2.findHomography(src_pts, dst_pts)


def frame_to_map(x, y):
    pt = np.array([[[x, y]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, H)
    return int(mapped[0][0][0]), int(mapped[0][0][1])


def make_field_canvas():
    canvas = np.full((MAP_H, MAP_W, 3), (34, 85, 34), dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (MAP_W - 1, MAP_H - 1), (255, 255, 255), 2)
    cv2.line(canvas, (MAP_W // 2, 0), (MAP_W // 2, MAP_H), (255, 255, 255), 1)
    return canvas


# --------------------------------------------------------------------------
# Phase 2: click on the target robot to set initial position
# --------------------------------------------------------------------------
init_click = [None]
if phase2_img is None:
    phase2_img = np.zeros((720, 1280, 3), dtype=np.uint8)


def on_target_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        init_click[0] = (x, y)


cv2.namedWindow("Click target robot")
cv2.setMouseCallback("Click target robot", on_target_click)

while True:
    disp = phase2_img.copy()
    cv2.putText(disp, f"Click on robot #{TARGET_NUMBER}, then press Enter.",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    if init_click[0]:
        cv2.circle(disp, init_click[0], 12, (0, 255, 255), 2)
        cv2.putText(disp, "Press Enter to confirm", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Click target robot", disp)
    key = cv2.waitKey(50) & 0xFF
    if key == 13 and init_click[0]:
        break
    if key == ord("q"):
        cv2.destroyAllWindows(); raise SystemExit

cv2.destroyWindow("Click target robot")

# --------------------------------------------------------------------------
# Phase 3: tracking loop
# --------------------------------------------------------------------------
results = model.track(
    source=VIDEO_SOURCE,
    stream=True,
    persist=True,
    imgsz=640,
    vid_stride=2,
    device="mps",
    tracker="bytetrack.yaml",
)

track_color     = {}
confirmed       = {}              # track_id → confirmed number string
votes           = defaultdict(Counter)
field_canvas    = make_field_canvas()
frame_count     = 0
target_track_id = None            # current YOLO track ID for the target
last_frame_pos  = None            # (cx, cy) in video-frame coords, last seen
target_traj     = []              # unified map trajectory (survives track-ID changes)

for result in results:
    frame_count += 1

    if not (result.boxes and result.boxes.is_track):
        frame = result.orig_img
    else:
        frame      = result.orig_img.copy()
        boxes_xyxy = result.boxes.xyxy.cpu()
        boxes_xywh = result.boxes.xywh.cpu()
        track_ids  = result.boxes.id.int().cpu().tolist()
        classes    = result.boxes.cls.int().cpu().tolist()

        # Center of each detection in frame coords
        centers = {tid: (float(xywh[0]), float(xywh[1]))
                   for xywh, tid in zip(boxes_xywh, track_ids)}

        def _dist(pos, tid):
            cx, cy = centers[tid]
            return ((cx - pos[0])**2 + (cy - pos[1])**2) ** 0.5

        # --- Target assignment / re-assignment by proximity ---
        if target_track_id not in track_ids:
            ref = last_frame_pos or init_click[0]
            if ref and track_ids:
                candidates = [t for t in track_ids
                              if confirmed.get(t) in (None, TARGET_NUMBER)]
                if candidates:
                    best = min(candidates, key=lambda t: _dist(ref, t))
                    if _dist(ref, best) < LOST_RADIUS:
                        target_track_id = best

        # Update last known frame position
        if target_track_id in track_ids:
            last_frame_pos = centers[target_track_id]

        # --- OCR: all unconfirmed tracks ---
        if frame_count % OCR_EVERY_N_FRAMES == 0:
            for box_xyxy, tid in zip(boxes_xyxy, track_ids):
                if tid in confirmed:
                    continue
                x1, y1, x2, y2 = map(int, box_xyxy)
                h = y2 - y1
                crop = frame[y1 + int(h * 0.6):y2, x1:x2]
                if crop.size == 0:
                    continue
                gray     = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                enhanced = clahe.apply(gray)
                crop_up  = cv2.resize(enhanced, None, fx=3, fy=3,
                                      interpolation=cv2.INTER_CUBIC)
                # PaddleOCR expects a BGR image; convert grayscale back to BGR
                crop_bgr = cv2.cvtColor(crop_up, cv2.COLOR_GRAY2BGR)
                for t in ocr_texts(crop_bgr):
                    t = t.strip()
                    if not t:
                        continue
                    if ALL_ROBOT_NUMBERS:
                        t = best_known_match(t, ALL_ROBOT_NUMBERS)
                        if t is None:
                            continue
                    else:
                        if not (t == TARGET_NUMBER or
                                (len(t) >= 3 and t in TARGET_NUMBER)):
                            continue
                        t = TARGET_NUMBER
                    votes[tid][t] += 1
                    top, count = votes[tid].most_common(1)[0]
                    if count >= VOTE_THRESHOLD:
                        confirmed[tid] = top

        # --- Draw all boxes on video; highlight target ---
        for box_xyxy, tid, cls in zip(boxes_xyxy, track_ids, classes):
            if tid not in track_color:
                track_color[tid] = alliance_color(cls)
            x1, y1, x2, y2 = map(int, box_xyxy)
            if tid == target_track_id:
                status = "OCR OK" if confirmed.get(tid) == TARGET_NUMBER else "tracking"
                cv2.rectangle(frame, (x1, y1), (x2, y2), track_color[tid], 3)
                cv2.putText(frame, f"#{TARGET_NUMBER} [{status}]", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, track_color[tid], 2)
            else:
                label = f"#{confirmed[tid]}" if tid in confirmed else f"ID {tid}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), GRAY_COLOR, 1)
                cv2.putText(frame, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY_COLOR, 1)

        # --- Update unified target trajectory ---
        if target_track_id in track_ids:
            idx = track_ids.index(target_track_id)
            bx  = float(boxes_xywh[idx][0])
            by  = float(boxes_xyxy[idx][3])
            mx, my = frame_to_map(bx, by)
            if 0 <= mx < MAP_W and 0 <= my < MAP_H:
                target_traj.append((mx, my))

        # --- Draw field canvas ---
        field_canvas = make_field_canvas()
        if len(target_traj) >= 2:
            color = track_color.get(target_track_id, GRAY_COLOR)
            pts   = np.array(target_traj, np.int32).reshape(-1, 1, 2)
            cv2.polylines(field_canvas, [pts], False, color, 2)
            cv2.circle(field_canvas, target_traj[-1], 8, color, -1)
            cv2.putText(field_canvas, f"#{TARGET_NUMBER}",
                        (target_traj[-1][0] + 7, target_traj[-1][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Video", frame)
    cv2.imshow("Field Map", field_canvas)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

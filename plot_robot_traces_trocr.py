"""
2D field trace plotter — single-robot focus mode (OCR: TrOCR).

pip install transformers pillow

Phase 1: Seek to a frame with the field visible (skipped if FIELD_CORNERS set).
         Click 4 corners: Top-Left → Top-Right → Bottom-Right → Bottom-Left
         Press Enter.
Phase 2: Click on the robot you want to track, press Enter.
Phase 3: Tracking runs. Robot trajectory is drawn on a top-down field map.

Press Q to quit.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from collections import defaultdict, Counter
from ultralytics import YOLO

# --- Config ---
VIDEO_SOURCE       = "/Users/ai4hobby/Playground/RobotTracking/video_sources/Einstein Final Tiebreaker - 2025 FIRST Championship.mp4"
MODEL_WEIGHTS      = "runs/detect/robot_detector3/weights/best.pt"
TROCR_MODEL        = "microsoft/trocr-base-printed"   # or trocr-large-printed for more accuracy
MAP_W, MAP_H       = 900, 450
OCR_EVERY_N_FRAMES = 3    # run frequently so other robots get confirmed quickly
VOTE_THRESHOLD     = 3
TARGET_NUMBER      = "2073"
ALL_ROBOT_NUMBERS  = ["2910","1323","4272","2073","4414","1690"]
LOST_RADIUS        = 200
UNCONFIRMED_RADIUS = 60    # tighter radius for re-assigning unconfirmed tracks
FIELD_CORNERS      = [(142, 73), (1777, 73), (1777, 985), (142, 985)]
BLUE_CAM_ROI       = None  # (x1, y1, x2, y2) bottom-left sub-camera, or None for interactive setup
RED_CAM_ROI        = None  # (x1, y1, x2, y2) bottom-right sub-camera, or None for interactive setup
SKIP_SECONDS       = 4

BLUE_COLOR = (220, 100,   0)
RED_COLOR  = (  0,  60, 220)
GRAY_COLOR = (160, 160, 160)

# --- Load YOLO & TrOCR ---
model       = YOLO(MODEL_WEIGHTS)
trocr_proc  = TrOCRProcessor.from_pretrained(TROCR_MODEL)
trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL)
device      = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
trocr_model = trocr_model.to(device)
trocr_model.eval()
print("Classes:", model.names)
print(f"TrOCR model loaded: {TROCR_MODEL} on {device}")


def alliance_color(cls_id):
    name = model.names.get(cls_id, "")
    if "blue" in name.lower(): return BLUE_COLOR
    if "red"  in name.lower(): return RED_COLOR
    return GRAY_COLOR


def ocr_text(crop_bgr):
    """Run TrOCR on a BGR crop and return the recognised string (digits only)."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    pixel_values = trocr_proc(pil_img, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = trocr_model.generate(pixel_values)
    text = trocr_proc.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return "".join(c for c in text if c.isdigit())


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
        corners.clear(); current_frame[0] = val; redraw(val)

    def on_corner_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append((x, y))
            redraw(current_frame[0])
            if len(corners) == 4:
                print(f"All 4 corners set — press Enter to start tracking.")
                print(f"To skip next time, set FIELD_CORNERS = {corners}")

    cv2.namedWindow("Define field corners")
    cv2.setMouseCallback("Define field corners", on_corner_click)
    cv2.createTrackbar("Frame", "Define field corners", 0, total_frames - 1, on_seekbar)
    redraw(0)

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 13 and len(corners) == 4: break
        if key == ord("q"):
            cap.release(); cv2.destroyAllWindows(); raise SystemExit

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
# Phase 1.5: select sub-camera regions (skipped if ROIs are hard-coded)
# --------------------------------------------------------------------------
if BLUE_CAM_ROI and RED_CAM_ROI:
    blue_roi = BLUE_CAM_ROI
    red_roi  = RED_CAM_ROI
    print(f"Using hard-coded sub-camera ROIs: blue={blue_roi}, red={red_roi}")
else:
    _cap_roi = cv2.VideoCapture(VIDEO_SOURCE)
    _cap_roi.set(cv2.CAP_PROP_POS_FRAMES, int(fps * SKIP_SECONDS))
    _ret, _roi_frame = _cap_roi.read()
    _cap_roi.release()
    if not _ret:
        _roi_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    print("Select the BLUE sub-camera region (bottom-left). Press Enter/Space to confirm, C to skip.")
    r = cv2.selectROI("Select Blue Camera ROI", _roi_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Blue Camera ROI")
    blue_roi = (int(r[0]), int(r[1]), int(r[0]+r[2]), int(r[1]+r[3])) if r[2] > 0 and r[3] > 0 else None

    print("Select the RED sub-camera region (bottom-right). Press Enter/Space to confirm, C to skip.")
    r = cv2.selectROI("Select Red Camera ROI", _roi_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Red Camera ROI")
    red_roi = (int(r[0]), int(r[1]), int(r[0]+r[2]), int(r[1]+r[3])) if r[2] > 0 and r[3] > 0 else None

    if blue_roi:
        print(f"Blue camera ROI: {blue_roi}")
    if red_roi:
        print(f"Red camera ROI:  {red_roi}")
    print("To skip this step next time, set BLUE_CAM_ROI and RED_CAM_ROI in the config.")


def _in_subcam(box_xyxy):
    """True if the detection center falls inside a sub-camera region."""
    cx = (float(box_xyxy[0]) + float(box_xyxy[2])) / 2
    cy = (float(box_xyxy[1]) + float(box_xyxy[3])) / 2
    for roi in (blue_roi, red_roi):
        if roi and roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]:
            return True
    return False


# --------------------------------------------------------------------------
# Phase 2: seek to a good frame, click the target robot to set initial position
# --------------------------------------------------------------------------
cap2        = cv2.VideoCapture(VIDEO_SOURCE)
seek2       = [int(fps * SKIP_SECONDS)]
cur_f2      = [None]
last_idx2   = [-1]
init_click  = [None]


def on_seek2(val):
    seek2[0] = val
    init_click[0] = None   # reset click when seeking


def on_click2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        init_click[0] = (x, y)


cv2.namedWindow("Click target robot")
cv2.setMouseCallback("Click target robot", on_click2)
cv2.createTrackbar("Frame", "Click target robot",
                   seek2[0], total_frames - 1, on_seek2)

while True:
    idx = seek2[0]
    if idx != last_idx2[0]:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f2 = cap2.read()
        if ret:
            cur_f2[0] = f2
        last_idx2[0] = idx

    if cur_f2[0] is None:
        cv2.waitKey(50)
        continue

    disp = cur_f2[0].copy()
    cv2.putText(disp,
                f"Seek & click robot #{TARGET_NUMBER}, then press Enter.  [{int(idx/fps):.0f}s]",
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
        cap2.release(); cv2.destroyAllWindows(); raise SystemExit

cap2.release()
cv2.destroyWindow("Click target robot")

# --------------------------------------------------------------------------
# Phase 3: tracking loop
# --------------------------------------------------------------------------
results = model.track(
    source=VIDEO_SOURCE,
    stream=True,
    persist=True,
    # imgsz=640,
    imgsz=1280,
    vid_stride=2,
    device="mps",
    tracker="bytetrack.yaml",
)

track_color      = {}
confirmed        = {}
votes            = defaultdict(Counter)
field_canvas     = make_field_canvas()
frame_count      = 0
target_track_id  = None
last_frame_pos   = None
target_traj      = []
tracked_frames   = 0   # frames where target was actively tracked

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

        centers = {tid: (float(xywh[0]), float(xywh[1]))
                   for xywh, tid in zip(boxes_xywh, track_ids)}

        subcam_mask    = [_in_subcam(b) for b in boxes_xyxy]
        main_track_ids = [t for t, sc in zip(track_ids, subcam_mask) if not sc]

        def _dist(pos, tid):
            cx, cy = centers[tid]
            return ((cx - pos[0])**2 + (cy - pos[1])**2) ** 0.5

        # --- Target assignment / re-assignment by proximity (main camera only) ---
        if target_track_id not in main_track_ids:
            ref = last_frame_pos or init_click[0]
            if ref and main_track_ids:
                # 1st priority: track already confirmed as TARGET_NUMBER
                confirmed_targets = [t for t in main_track_ids
                                     if confirmed.get(t) == TARGET_NUMBER]
                if confirmed_targets:
                    best = min(confirmed_targets, key=lambda t: _dist(ref, t))
                    if _dist(ref, best) < LOST_RADIUS:
                        target_track_id = best
                else:
                    # 2nd priority: unconfirmed track, but only within tight radius
                    unconfirmed = [t for t in main_track_ids
                                   if confirmed.get(t) is None]
                    if unconfirmed:
                        best = min(unconfirmed, key=lambda t: _dist(ref, t))
                        if _dist(ref, best) < UNCONFIRMED_RADIUS:
                            target_track_id = best

        if target_track_id in main_track_ids:
            last_frame_pos = centers[target_track_id]
            tracked_frames += 1

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
                # Upscale for TrOCR (larger image = better recognition)
                crop_up = cv2.resize(crop, None, fx=3, fy=3,
                                     interpolation=cv2.INTER_CUBIC)
                t = ocr_text(crop_up)
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

        # --- Draw all boxes; highlight target ---
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

        # --- Draw sub-camera region outlines ---
        for roi, lbl, col in [(blue_roi, "Blue Cam", BLUE_COLOR), (red_roi, "Red Cam", RED_COLOR)]:
            if roi:
                cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), col, 2)
                cv2.putText(frame, lbl, (roi[0] + 4, roi[1] + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        # --- Update unified target trajectory ---
        if target_track_id in main_track_ids:
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

# --- Tracking summary ---
vid_stride    = 2
secs_per_frame = vid_stride / fps
total_time    = frame_count * secs_per_frame
tracked_time  = tracked_frames * secs_per_frame
pct           = (tracked_time / total_time * 100) if total_time > 0 else 0.0

def fmt(s):
    return f"{int(s // 60)}m {s % 60:.1f}s"

print(f"\n--- Tracking summary for #{TARGET_NUMBER} ---")
print(f"  Total video processed : {fmt(total_time)}  ({frame_count} frames)")
print(f"  Time tracked          : {fmt(tracked_time)}  ({tracked_frames} frames)")
print(f"  Coverage              : {pct:.1f}%")

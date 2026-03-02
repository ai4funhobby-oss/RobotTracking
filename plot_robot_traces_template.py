"""
2D field trace plotter — single-robot focus mode (re-ID: template matching).

No OCR. Phase 2 lets you seek the video and draw a rectangle around the number
on the robot. Those proportional crop coordinates are reused every frame so the
comparison region is always consistent. A rolling template bank keeps the last
N templates from stable tracking, so re-ID adapts to lighting / angle changes.

Phase 1: Seek to a frame with the field visible (skipped if FIELD_CORNERS set).
         Click 4 corners: Top-Left → Top-Right → Bottom-Right → Bottom-Left
         Press Enter.
Phase 2: Seek to a clear view of the target robot. Click it, then drag a
         rectangle around the number. Press Space/Enter to confirm.
Phase 3: Tracking. Robot trajectory drawn on a top-down field map.

Press Q to quit.
"""

import cv2
import numpy as np
from ultralytics import YOLO

# --- Config ---
VIDEO_SOURCE          = "/Users/ai4hobby/Playground/RobotTracking/video_sources/Einstein Final Tiebreaker - 2025 FIRST Championship.mp4"
MODEL_WEIGHTS         = "runs/detect/robot_detector3/weights/best.pt"
MAP_W, MAP_H          = 900, 450
TARGET_NUMBER         = "2073"       # display label only
LOST_RADIUS           = 200          # px — proximity window for re-ID
CLOSE_RADIUS          = 60           # px — "very close": lower match threshold
MATCH_THRESHOLD       = 0.35         # normalised cross-correlation for re-ID
MATCH_THRESHOLD_CLOSE = 0.15         # threshold when candidate is within CLOSE_RADIUS
TEMPLATE_W, TEMPLATE_H = 80, 40     # fixed template size
TEMPLATE_BANK_SIZE    = 6            # max templates kept
UPDATE_TEMPLATE_EVERY = 25           # frames of stable tracking before bank update
FIELD_CORNERS         = [(142, 73), (1777, 73), (1777, 985), (142, 985)]
SKIP_SECONDS          = 4

BLUE_COLOR = (220, 100,   0)
RED_COLOR  = (  0,  60, 220)
GRAY_COLOR = (160, 160, 160)

# --- Load model ---
model = YOLO(MODEL_WEIGHTS)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
print("Classes:", model.names)


def alliance_color(cls_id):
    name = model.names.get(cls_id, "")
    if "blue" in name.lower(): return BLUE_COLOR
    if "red"  in name.lower(): return RED_COLOR
    return GRAY_COLOR


def number_crop(frame, box_xyxy, props):
    """Extract the number region using stored proportional coordinates."""
    x1, y1, x2, y2 = map(int, box_xyxy)
    bh, bw = y2 - y1, x2 - x1
    cy1 = y1 + int(props[0] * bh)
    cy2 = y1 + int(props[1] * bh)
    cx1 = x1 + int(props[2] * bw)
    cx2 = x1 + int(props[3] * bw)
    crop = frame[cy1:cy2, cx1:cx2]
    return crop if crop.size > 0 else None


def to_template(crop):
    """Preprocess a crop into a fixed-size grayscale template."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)
    return cv2.resize(enhanced, (TEMPLATE_W, TEMPLATE_H),
                      interpolation=cv2.INTER_CUBIC)


def best_match(crop, bank):
    """Best normalised cross-correlation score across the template bank."""
    if not bank or crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)
    resized = cv2.resize(enhanced, (TEMPLATE_W, TEMPLATE_H),
                         interpolation=cv2.INTER_CUBIC)
    return max(
        float(cv2.matchTemplate(resized, t, cv2.TM_CCOEFF_NORMED).max())
        for t in bank
    )


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
                print(f"All 4 corners set — press Enter.")
                print(f"To skip next time: FIELD_CORNERS = {corners}")

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
# Phase 2: seek, click robot, draw rectangle around number → template + props
# --------------------------------------------------------------------------
cap2 = cv2.VideoCapture(VIDEO_SOURCE)
seek2        = [int(fps * SKIP_SECONDS)]
last_pred    = [-1]
cached_boxes = [[]]          # [[x1,y1,x2,y2,cls], ...]
selected_box = [None]
current_f2   = [None]

# Template state (set at end of Phase 2)
init_click     = [None]
crop_props     = [None]      # (y1_frac, y2_frac, x1_frac, x2_frac) relative to bbox
template_bank  = []


def on_seek2(val):
    seek2[0] = val
    selected_box[0] = None


def on_click2(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN or not cached_boxes[0]:
        return
    best_i, best_d = None, float('inf')
    for i, (x1, y1, x2, y2, _) in enumerate(cached_boxes[0]):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        d = ((cx - x)**2 + (cy - y)**2) ** 0.5
        if d < best_d:
            best_d, best_i = d, i
    if best_i is not None and best_d < 150:
        selected_box[0] = cached_boxes[0][best_i]


cv2.namedWindow("Capture template")
cv2.setMouseCallback("Capture template", on_click2)
cv2.createTrackbar("Frame", "Capture template",
                   seek2[0], total_frames - 1, on_seek2)

while True:
    idx = seek2[0]
    if idx != last_pred[0]:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f2 = cap2.read()
        if ret:
            current_f2[0] = f2
            r = model.predict(f2, imgsz=640, device="mps", verbose=False)[0]
            cached_boxes[0] = (
                [[int(b[0]), int(b[1]), int(b[2]), int(b[3]), int(c)]
                 for b, c in zip(r.boxes.xyxy.cpu(), r.boxes.cls.int().cpu())]
                if r.boxes and len(r.boxes) else []
            )
        last_pred[0] = idx

    if current_f2[0] is None:
        cv2.waitKey(50); continue

    disp = current_f2[0].copy()
    for x1, y1, x2, y2, cls in cached_boxes[0]:
        cv2.rectangle(disp, (x1, y1), (x2, y2), alliance_color(cls), 1)
    if selected_box[0] is not None:
        x1, y1, x2, y2, _ = selected_box[0]
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(disp, "SELECTED — press Enter to draw number region",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(disp,
                f"Seek to clear view of #{TARGET_NUMBER}. Click robot, Enter to draw region.  "
                f"[{int(idx/fps):.0f}s]",
                (10, disp.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imshow("Capture template", disp)

    key = cv2.waitKey(50) & 0xFF
    if key == 13 and selected_box[0] is not None:
        bx1, by1, bx2, by2, _ = selected_box[0]
        bh, bw = by2 - by1, bx2 - bx1
        # Zoom the robot region for easier drawing
        robot_crop = current_f2[0][by1:by2, bx1:bx2]
        scale = max(1, 300 // max(bh, 1))
        zoomed = cv2.resize(robot_crop, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_CUBIC)
        cv2.putText(zoomed, "Drag around NUMBER, Space/Enter to confirm",
                    (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        rx, ry, rw, rh = cv2.selectROI("Select number region", zoomed,
                                        fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select number region")
        if rw > 0 and rh > 0:
            # Store crop proportions relative to the bbox
            crop_props[0] = (
                ry / (bh * scale),           # y1_frac
                (ry + rh) / (bh * scale),    # y2_frac
                rx / (bw * scale),           # x1_frac
                (rx + rw) / (bw * scale),    # x2_frac
            )
            number_img = zoomed[ry:ry + rh, rx:rx + rw]
            template_bank.append(to_template(number_img))
            init_click[0] = ((bx1 + bx2) // 2, (by1 + by2) // 2)
            print(f"Template captured for #{TARGET_NUMBER} at frame {idx}. "
                  f"Crop props: {[f'{v:.2f}' for v in crop_props[0]]}")
            break
    if key == ord("q"):
        cap2.release(); cv2.destroyAllWindows(); raise SystemExit

cap2.release()
cv2.destroyWindow("Capture template")

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
field_canvas    = make_field_canvas()
frame_count     = 0
target_track_id = None
last_frame_pos  = None
target_traj     = []
stable_frames   = 0     # consecutive frames the target has been tracked

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

        def _dist(pos, tid):
            cx, cy = centers[tid]
            return ((cx - pos[0])**2 + (cy - pos[1])**2) ** 0.5

        # --- Target assignment / re-assignment ---
        if target_track_id not in track_ids:
            stable_frames = 0
            ref = last_frame_pos or init_click[0]
            if ref and track_ids:
                for tid in sorted(track_ids, key=lambda t: _dist(ref, t)):
                    d = _dist(ref, tid)
                    if d >= LOST_RADIUS:
                        break
                    # Initial assignment: no template yet, just take closest
                    if not template_bank:
                        target_track_id = tid
                        break
                    # Re-assignment: template match with proximity-based threshold
                    thresh = MATCH_THRESHOLD_CLOSE if d < CLOSE_RADIUS else MATCH_THRESHOLD
                    crop = number_crop(frame, boxes_xyxy[track_ids.index(tid)], crop_props[0])
                    if best_match(crop, template_bank) >= thresh:
                        target_track_id = tid
                        break
        else:
            stable_frames += 1
            # Auto-update template bank during stable tracking
            if stable_frames % UPDATE_TEMPLATE_EVERY == 0:
                idx = track_ids.index(target_track_id)
                crop = number_crop(frame, boxes_xyxy[idx], crop_props[0])
                if crop is not None:
                    template_bank.append(to_template(crop))
                    if len(template_bank) > TEMPLATE_BANK_SIZE:
                        template_bank.pop(0)

        # Update last known frame position
        if target_track_id in track_ids:
            last_frame_pos = centers[target_track_id]

        # --- Draw all boxes; highlight target ---
        for box_xyxy, tid, cls in zip(boxes_xyxy, track_ids, classes):
            if tid not in track_color:
                track_color[tid] = alliance_color(cls)
            x1, y1, x2, y2 = map(int, box_xyxy)
            if tid == target_track_id:
                cv2.rectangle(frame, (x1, y1), (x2, y2), track_color[tid], 3)
                cv2.putText(frame, f"#{TARGET_NUMBER}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, track_color[tid], 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), GRAY_COLOR, 1)
                cv2.putText(frame, f"ID {tid}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY_COLOR, 1)

        # --- Update unified target trajectory ---
        if target_track_id in track_ids:
            i   = track_ids.index(target_track_id)
            bx  = float(boxes_xywh[i][0])
            by  = float(boxes_xyxy[i][3])
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

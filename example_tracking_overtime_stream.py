from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO26 model (change to your custom model if needed)
# model = YOLO("yolo26n.pt")
model = YOLO("/Users/ai4hobby/Playground/RobotTracking/runs/detect/train3/weights/best.pt")
car_classes = [i for i, name in model.names.items() if name == "car"]

# YouTube video URL as the source
video_source = "https://youtu.be/vZot9cgPKjI"
video_source = "https://youtu.be/JPk0U_NXfmo?si=bkQ3bkRTbW4puUk8"

# Store the track history
track_history = defaultdict(lambda: [])

# Use Ultralytics streaming API to read frames from the YouTube source
# Tune for speed: smaller image size + skip frames
results = model.track(
    source=video_source,
    stream=True,
    persist=True,
    classes=car_classes,
    imgsz=480,    # smaller than default 640 → faster
    vid_stride=2  # process every 2nd frame → faster
)

for result in results:
    # Get the boxes and track IDs
    if result.boxes and result.boxes.is_track:
        boxes = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()

        # Visualize the result on the frame
        frame = result.plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 20:  # retain fewer points for slightly less drawing work
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=6)

        # Display the annotated frame
        cv2.imshow("YOLO26 Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
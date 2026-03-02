from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO26 model (change to your custom model if needed)
model = YOLO("yolo26n.pt")
# model = YOLO("/Users/ai4hobby/Playground/RobotTracking/runs/detect/train3/weights/best.pt")
# car_classes = [i for i, name in model.names.items() if name == "car"]
# human_classes = [i for i, name in model.names.items() if name == "human"]
class2detect = [i for i, name in model.names.items() if name == "person"]

# Local MP4 video as the source
# video_source = "/Users/ai4hobby/Playground/RobotTracking/video_sources/Einstein Final 1 - 2025 FIRST Championship.mp4"
# video_source = "/Users/ai4hobby/Playground/RobotTracking/video_sources/Einstein Final Tiebreaker - 2025 FIRST Championship.mp4"
# video_source = '/Users/ai4hobby/Playground/RobotTracking/video_sources/Qualification 3 - Week 0.mp4'
video_source = '/Users/ai4hobby/Playground/RobotTracking/video_sources/people walking around a shopping mall - 4K free stock video.mp4'
# video_source = '/Users/ai4hobby/Playground/RobotTracking/video_sources/Womens Short Track Speed Skating 1500m Final 2024.mp4'
# video_source = '/Users/ai4hobby/Playground/RobotTracking/video_sources/Home Town Hero  1000m Women Final  Seoul 2024  ShortTrackWorldTour.mp4'
# video_source = '/Users/ai4hobby/Playground/RobotTracking/video_sources/South Korea battles to 7th womens short track relay title.mp4'
# video_source = "https://youtu.be/JPk0U_NXfmo?si=bkQ3bkRTbW4puUk8"

# Store the track history
track_history = defaultdict(lambda: [])

# Use Ultralytics streaming API to read frames from the YouTube source
# Tune for speed: smaller image size + skip frames
results = model.track(
    source=video_source,
    stream=True,
    persist=True,
    classes=class2detect,
    imgsz=640,    # smaller than default 640 → faster
    vid_stride=2,  # process every 2nd frame → faster
    device="mps",
    tracker="bytetrack.yaml"
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
            if len(track) > 30:  # retain fewer points for slightly less drawing work
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
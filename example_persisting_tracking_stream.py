from ultralytics import YOLO
import cv2

model = YOLO("/Users/ai4hobby/Playground/RobotTracking/runs/detect/train3/weights/best.pt")

car_classes = [i for i, name in model.names.items() if name == "car"]
video_source = "https://youtu.be/vZot9cgPKjI"
video_source = "https://youtu.be/JPk0U_NXfmo?si=bkQ3bkRTbW4puUk8"
results = model.track(
    source=video_source,
    stream=True,     # yield frames as a generator
    persist=True,    # keep track IDs across frames
    show=False,       # we'll show with OpenCV ourselves
    classes=car_classes,
    device="mps"
)

for r in results:
    annotated_frame = r.plot()
    cv2.imshow("YOLO26 Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
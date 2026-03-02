from ultralytics import YOLO

# Load an official or custom model
# model = YOLO("yolo26n.pt")  # Load an official Detect model
# model = YOLO("yolo26n-seg.pt")  # Load an official Segment model
# model = YOLO("yolo26n-pose.pt")  # Load an official Pose model
model = YOLO("/Users/ai4hobby/Playground/RobotTracking/runs/detect/train3/weights/best.pt")  # Load a custom-trained model

# Build a list of all class IDs except "person"
# non_person_classes = [i for i, name in model.names.items() if name != "person"]
car_classes = [i for i, name in model.names.items() if name == "car"]
print(model.names)
input()

# Perform tracking with the model
# results = model.track("https://youtu.be/LNwODJXcvt4", show=True)  # Tracking with default tracker
# results = model.track("https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # with ByteTrack
results = model.track(
        "https://youtu.be/vZot9cgPKjI", 
        show=True, 
        device="mps", 
        classes=car_classes
)  # Tracking with default tracker

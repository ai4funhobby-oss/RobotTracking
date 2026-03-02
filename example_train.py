from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.yaml")  # build a new model from YAML
model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo26n.yaml").load("yolo26n.pt")  # build from YAML and transfer weights

# # Train the model
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device="mps")

# Train the model Drone dataset

# Use CPU to avoid MPS shape-mismatch bug in Ultralytics training loss (TAL assigner).
# MPS can be used for inference; for faster training use a CUDA machine.
results = model.train(data="VisDrone.yaml", epochs=100, imgsz=640, device="mps")    #error due to the compatibility issue
# results = model.train(data="VisDrone.yaml", epochs=100, imgsz=640, device="cpu")
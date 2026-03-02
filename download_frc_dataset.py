"""
Download an FRC robot dataset from Roboflow Universe in YOLOv8 format.

Steps:
1. pip install roboflow
2. Paste your API key below (roboflow.com → Settings → API)
3. Run: python download_frc_dataset.py

The dataset will land in datasets/frc-robots/ with a data.yaml ready for training.

Recommended datasets (pick one by uncommenting):
  - FRC v2025 by team 611  (2025 game, Blue Robot / Red Robot)
  - 2024 FRC by WorBots    (1482 images, red robot / blue robot + game pieces)
"""

from roboflow import Roboflow

API_KEY = "SdggTvUvo4eU0zDMJMUc"

rf = Roboflow(api_key=API_KEY)

# --- Your private workspace ---
# project = rf.workspace("ai4hobbys-workspace").project("frc-robots-fx5cu-sp5h4")
project = rf.workspace("ai4hobbys-workspace").project("frc-2023-hwue2-vtp95")

# List available versions, then download the latest
versions = project.versions()
print(f"Available versions: {[v.version for v in versions]}")

latest = max(versions, key=lambda v: v.version)
print(f"Downloading version {latest.version}...")

dataset = latest.download("yolov8", location="datasets/frc-robots_2023")

print(f"\nDataset downloaded to: datasets/frc-robots_2023/")
print(f"data.yaml path for training: datasets/frc-robots_2023/data.yaml")

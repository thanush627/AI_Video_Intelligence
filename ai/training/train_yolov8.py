from ultralytics import YOLO
from pathlib import Path

# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_YAML = PROJECT_ROOT / "ai" / "training" / "data.yaml"

RUNS_DIR = PROJECT_ROOT / "ai" / "training" / "runs"

RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# LOAD MODEL
# ==========================================================

print("=" * 60)
print("Loading YOLOv8m...")
print("=" * 60)

model = YOLO("yolov8m.pt")

# ==========================================================
# START TRAINING
# ==========================================================

print("\nStarting Training...\n")

model.train(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=640,
    batch=16,
    workers=2,
    project=str(RUNS_DIR),
    name="yolov8_custom",
    exist_ok=True
)

print("\nTraining Completed Successfully!")
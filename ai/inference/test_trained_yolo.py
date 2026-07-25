from pathlib import Path
from ultralytics import YOLO
import time

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "ai"
    / "models"
    / "yolo"
    / "trained"
    / "best.pt"
)

VIDEO_PATH = PROJECT_ROOT / "test_videos" / "test.mp4"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "yolo_test"

# ============================================================
# VERIFY FILES
# ============================================================

print("=" * 70)
print("TRAINED YOLO VIDEO TEST")
print("=" * 70)

print("Model :", MODEL_PATH)
print("Video :", VIDEO_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}")

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Video not found:\n{VIDEO_PATH}")

# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(str(MODEL_PATH))

print("\nModel loaded successfully.")
print("Starting video detection...\n")

# ============================================================
# RUN DETECTION
# ============================================================

start_time = time.time()

results = model.predict(
    source=str(VIDEO_PATH),

    # Detection settings
    imgsz=640,
    conf=0.25,
    iou=0.45,

    # Save annotated video
    save=True,

    # Output location
    project=str(OUTPUT_DIR),
    name="prediction",
    exist_ok=True,

    # Show progress
    verbose=True
)

elapsed = time.time() - start_time

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("VIDEO DETECTION COMPLETED")
print("=" * 70)

print("Processing time :", round(elapsed, 2), "seconds")
print("Frames processed:", len(results))

print("\nOutput folder:")
print(OUTPUT_DIR / "prediction")

print("\nOpen the generated video to check the bounding boxes.")
print("=" * 70)
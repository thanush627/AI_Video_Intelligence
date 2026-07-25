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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "bytetrack_test"

# ============================================================
# VERIFY FILES
# ============================================================

print("=" * 70)
print("YOLO + BYTETRACK TEST")
print("=" * 70)

print("Model :", MODEL_PATH)
print("Video :", VIDEO_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}")

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Video not found:\n{VIDEO_PATH}")

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = YOLO(str(MODEL_PATH))

print("\nModel loaded successfully.")
print("Starting ByteTrack tracking...\n")

# ============================================================
# RUN YOLO + BYTETRACK
# ============================================================

start_time = time.time()

results = model.track(
    source=str(VIDEO_PATH),

    # Tracking
    tracker="bytetrack.yaml",
    persist=True,

    # Detection
    imgsz=640,
    conf=0.25,
    iou=0.45,

    # Save annotated video
    save=True,

    # Output
    project=str(OUTPUT_DIR),
    name="prediction",
    exist_ok=True,

    # Avoid storing every frame result in RAM
    stream=True,

    verbose=True
)

# ============================================================
# PROCESS STREAM
# ============================================================

frame_count = 0
frames_with_tracks = 0
unique_track_ids = set()

for result in results:
    frame_count += 1

    if result.boxes is not None and result.boxes.id is not None:
        track_ids = result.boxes.id.int().cpu().tolist()

        frames_with_tracks += 1
        unique_track_ids.update(track_ids)

elapsed = time.time() - start_time

# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("BYTETRACK TEST COMPLETED")
print("=" * 70)

print("Frames processed   :", frame_count)
print("Frames with tracks :", frames_with_tracks)
print("Unique track IDs   :", len(unique_track_ids))
print("Track IDs          :", sorted(unique_track_ids))
print("Processing time    :", round(elapsed, 2), "seconds")

print("\nOutput folder:")
print(OUTPUT_DIR / "prediction")

print("=" * 70)
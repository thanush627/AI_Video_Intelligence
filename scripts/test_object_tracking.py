import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from ai.pipeline.object_tracker import ObjectTracker


CONFIG_PATH = (
    PROJECT_ROOT
    / "ai"
    / "configs"
    / "phase3_config.yaml"
)

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8",
) as file:
    config = yaml.safe_load(file)


MODEL_PATH = (
    PROJECT_ROOT
    / config["paths"]["model"]
)

VIDEO_PATH = (
    PROJECT_ROOT
    / config["paths"]["test_video"]
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / config["paths"]["output_root"]
    / "tracking_test"
)


detector_config = config["detection"]
tracking_config = config["tracking"]


tracker = ObjectTracker(
    model_path=MODEL_PATH,
    confidence_threshold=detector_config[
        "confidence_threshold"
    ],
    iou_threshold=detector_config[
        "iou_threshold"
    ],
    image_size=detector_config["image_size"],
    tracker=tracking_config["tracker"],
    device=detector_config["device"],
)


print("\nRunning YOLO + ByteTrack...\n")


results = tracker.track(
    video_path=VIDEO_PATH,
    output_dir=OUTPUT_DIR,
)


print("\n" + "=" * 70)
print("OBJECT TRACKING TEST")
print("=" * 70)

print(
    "Processed frames   :",
    results["processed_frames"],
)

print(
    "Frames with tracks :",
    results["frames_with_tracks"],
)

print(
    "Unique tracks      :",
    results["unique_tracks"],
)

print(
    "Metadata file      :",
    results["metadata_path"],
)

print("=" * 70)
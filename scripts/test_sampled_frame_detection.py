import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from ai.inference.sampled_frame_detector import (
    SampledFrameDetector,
)


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

FRAME_METADATA_PATH = (
    PROJECT_ROOT
    / config["paths"]["output_root"]
    / "frame_extraction_test"
    / "frame_metadata.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / config["paths"]["output_root"]
    / "detection_test"
)


detection_config = config["detection"]


detector = SampledFrameDetector(
    model_path=MODEL_PATH,
    confidence_threshold=detection_config[
        "confidence_threshold"
    ],
    iou_threshold=detection_config[
        "iou_threshold"
    ],
    image_size=detection_config[
        "image_size"
    ],
    device=detection_config["device"],
)


print("\nRunning YOLO detection on sampled frames...\n")


results = detector.detect(
    frame_metadata_path=FRAME_METADATA_PATH,
    output_dir=OUTPUT_DIR,
)


print("\n" + "=" * 70)
print("SAMPLED FRAME DETECTION TEST")
print("=" * 70)

print(
    "Processed frames :",
    results["processed_frames"],
)

print(
    "Total detections :",
    results["total_detections"],
)

print(
    "Class counts     :",
    results["class_counts"],
)

print(
    "Metadata file    :",
    results["metadata_path"],
)

print("=" * 70)
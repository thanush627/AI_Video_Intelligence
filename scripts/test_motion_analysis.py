import sys
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


import yaml

from ai.event_generation.motion_analyzer import (
    MotionAnalyzer,
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


TRACK_METADATA_PATH = (
    PROJECT_ROOT
    / config["paths"]["output_root"]
    / "tracking_test"
    / "track_metadata.json"
)


RELIABILITY_METADATA_PATH = (
    PROJECT_ROOT
    / config["paths"]["output_root"]
    / "track_reliability_test"
    / "track_reliability.json"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / config["paths"]["output_root"]
    / "motion_analysis_test"
)


SOURCE_FPS = 29.97002997002997

FRAME_WIDTH = 2560

FRAME_HEIGHT = 1440


analyzer = MotionAnalyzer(
    track_metadata_path=(
        TRACK_METADATA_PATH
    ),
    reliability_metadata_path=(
        RELIABILITY_METADATA_PATH
    ),
    output_dir=OUTPUT_DIR,
    source_fps=SOURCE_FPS,
    frame_width=FRAME_WIDTH,
    frame_height=FRAME_HEIGHT,
    include_review=True,
)


print(
    "\nRunning track motion "
    "analysis...\n"
)


results = analyzer.analyze()


print("\n" + "=" * 70)

print(
    "MOTION ANALYSIS TEST"
)

print("=" * 70)


print(
    "Analyzed tracks            :",
    results["analyzed_tracks"],
)


print(
    "Skipped reject tracks      :",
    results[
        "skipped_reject_tracks"
    ],
)


print(
    "Skipped insufficient tracks:",
    results[
        "skipped_insufficient_tracks"
    ],
)


print(
    "Motion state counts        :",
    results[
        "motion_state_counts"
    ],
)


print(
    "Direction counts           :",
    results[
        "direction_counts"
    ],
)


print(
    "Metadata file              :",
    results["metadata_path"],
)


print("=" * 70)
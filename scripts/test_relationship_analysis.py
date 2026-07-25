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

from ai.event_generation.relationship_analyzer import (
    RelationshipAnalyzer,
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


OUTPUT_ROOT = (
    PROJECT_ROOT
    / config["paths"]["output_root"]
)


TRACK_METADATA_PATH = (
    OUTPUT_ROOT
    / "tracking_test"
    / "track_metadata.json"
)


RELIABILITY_METADATA_PATH = (
    OUTPUT_ROOT
    / "track_reliability_test"
    / "track_reliability.json"
)


MOTION_METADATA_PATH = (
    OUTPUT_ROOT
    / "motion_analysis_test"
    / "motion_metadata.json"
)


OUTPUT_DIR = (
    OUTPUT_ROOT
    / "relationship_analysis_test"
)


analyzer = RelationshipAnalyzer(
    track_metadata_path=(
        TRACK_METADATA_PATH
    ),
    reliability_metadata_path=(
        RELIABILITY_METADATA_PATH
    ),
    motion_metadata_path=(
        MOTION_METADATA_PATH
    ),
    output_dir=OUTPUT_DIR,
    frame_width=2560,
    frame_height=1440,
    include_review=True,
    near_threshold=0.08,
    moving_together_threshold=0.06,
    minimum_shared_frames=3,
)


print(
    "\nRunning object-to-object "
    "spatial relationship analysis...\n"
)


results = analyzer.analyze()


print("\n" + "=" * 70)

print(
    "RELATIONSHIP ANALYSIS TEST"
)

print("=" * 70)


print(
    "Usable tracks          :",
    results["usable_tracks"],
)


print(
    "Frames with objects    :",
    results["frames_with_objects"],
)


print(
    "Analyzed track pairs   :",
    results["analyzed_pairs"],
)


print(
    "Detected relationships :",
    results[
        "detected_relationships"
    ],
)


print(
    "Relationship counts    :",
    results[
        "relationship_type_counts"
    ],
)


print(
    "Metadata file          :",
    results["metadata_path"],
)


print("=" * 70)
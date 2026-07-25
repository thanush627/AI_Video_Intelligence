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

from ai.event_generation.spatial_analyzer import (
    SpatialAnalyzer,
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


MOTION_METADATA_PATH = (
    PROJECT_ROOT
    / config["paths"]["output_root"]
    / "motion_analysis_test"
    / "motion_metadata.json"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / config["paths"]["output_root"]
    / "spatial_analysis_test"
)


FRAME_WIDTH = 2560

FRAME_HEIGHT = 1440


analyzer = SpatialAnalyzer(
    motion_metadata_path=(
        MOTION_METADATA_PATH
    ),
    output_dir=OUTPUT_DIR,
    frame_width=FRAME_WIDTH,
    frame_height=FRAME_HEIGHT,
)


print(
    "\nRunning single-object "
    "spatial reasoning...\n"
)


results = analyzer.analyze()


print("\n" + "=" * 70)

print(
    "SPATIAL ANALYSIS TEST"
)

print("=" * 70)


print(
    "Analyzed tracks     :",
    results["analyzed_tracks"],
)


print(
    "Skipped tracks      :",
    results["skipped_tracks"],
)


print(
    "Start region counts :",
    results["start_region_counts"],
)


print(
    "End region counts   :",
    results["end_region_counts"],
)


print(
    "Metadata file       :",
    results["metadata_path"],
)


print("=" * 70)
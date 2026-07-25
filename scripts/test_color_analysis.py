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

from ai.pipeline.color_analyzer import (
    ColorAnalyzer,
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


REPRESENTATIVE_METADATA_PATH = (
    PROJECT_ROOT
    / config["paths"]["output_root"]
    / "representative_selection_test"
    / "representative_crops.json"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / config["paths"]["output_root"]
    / "color_analysis_object_aware"
)


analyzer = ColorAnalyzer(
    representative_metadata_path=(
        REPRESENTATIVE_METADATA_PATH
    ),
    output_dir=OUTPUT_DIR,
)


print(
    "\nRunning object-aware "
    "colour analysis...\n"
)


results = analyzer.analyze()


print("\n" + "=" * 70)

print(
    "OBJECT-AWARE COLOUR ANALYSIS"
)

print("=" * 70)


print(
    "Analyzed tracks     :",
    results["analyzed_tracks"],
)


print(
    "Unknown tracks      :",
    results["unknown_tracks"],
)


print(
    "Final colour counts :",
    results["final_color_counts"],
)


print(
    "Metadata file       :",
    results["metadata_path"],
)


print("=" * 70)
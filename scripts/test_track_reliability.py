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

from ai.pipeline.track_reliability import (
    TrackReliabilityFilter,
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
    / "track_reliability_test"
)


analyzer = TrackReliabilityFilter(
    representative_metadata_path=(
        REPRESENTATIVE_METADATA_PATH
    ),
    output_dir=OUTPUT_DIR,
)


print(
    "\nRunning track reliability "
    "analysis...\n"
)


results = analyzer.analyze()


print("\n" + "=" * 70)

print(
    "TRACK RELIABILITY TEST"
)

print("=" * 70)


print(
    "Total tracks        :",
    results["total_tracks"],
)


print(
    "Status counts       :",
    results["status_counts"],
)


print(
    "Class status counts :"
)


for class_name, counts in (
    results[
        "class_status_counts"
    ].items()
):

    print(
        f"  {class_name:10} : "
        f"{counts}"
    )


print(
    "Metadata file       :",
    results["metadata_path"],
)


print("=" * 70)
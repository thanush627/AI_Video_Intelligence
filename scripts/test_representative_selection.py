import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from ai.pipeline.representative_selector import (
    RepresentativeCropSelector,
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


VIDEO_PATH = (
    PROJECT_ROOT
    / config["paths"]["test_video"]
)

TRACK_METADATA_PATH = (
    PROJECT_ROOT
    / config["paths"]["output_root"]
    / "tracking_test"
    / "track_metadata.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / config["paths"]["output_root"]
    / "representative_selection_test"
)


selection_config = config[
    "representative_selection"
]


selector = RepresentativeCropSelector(
    video_path=VIDEO_PATH,
    track_metadata_path=TRACK_METADATA_PATH,
    output_dir=OUTPUT_DIR,
    max_crops_per_track=selection_config[
        "max_crops_per_track"
    ],
    min_crop_width=selection_config[
        "min_crop_width"
    ],
    min_crop_height=selection_config[
        "min_crop_height"
    ],
    weights=selection_config["weights"],
)


print(
    "\nSelecting representative crops...\n"
)


results = selector.select()


print("\n" + "=" * 70)
print("REPRESENTATIVE CROP SELECTION TEST")
print("=" * 70)

print(
    "Input tracks       :",
    results["input_tracks"],
)

print(
    "Tracks with crops  :",
    results["tracks_with_crops"],
)

print(
    "Total saved crops  :",
    results["total_saved_crops"],
)

print(
    "Crops directory    :",
    results["crops_directory"],
)

print(
    "Metadata file      :",
    results["metadata_path"],
)

print("=" * 70)
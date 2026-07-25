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

from ai.event_generation.atomic_event_generator import (
    AtomicEventGenerator,
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


COLOR_METADATA_PATH = (
    PROJECT_ROOT
    / config["paths"]["output_root"]
    / "color_analysis_object_aware"
    / "color_metadata.json"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / config["paths"]["output_root"]
    / "atomic_event_generation_test"
)


SOURCE_FPS = 29.97002997002997


generator = AtomicEventGenerator(
    track_metadata_path=(
        TRACK_METADATA_PATH
    ),
    reliability_metadata_path=(
        RELIABILITY_METADATA_PATH
    ),
    color_metadata_path=(
        COLOR_METADATA_PATH
    ),
    output_dir=OUTPUT_DIR,
    source_fps=SOURCE_FPS,
    include_review=True,
)


print(
    "\nGenerating atomic "
    "spatiotemporal events...\n"
)


results = generator.generate()


print("\n" + "=" * 70)

print(
    "ATOMIC EVENT GENERATION TEST"
)

print("=" * 70)


print(
    "Total events          :",
    results["total_events"],
)


print(
    "Usable tracks         :",
    results["usable_tracks"],
)


print(
    "Skipped reject tracks :",
    results[
        "skipped_reject_tracks"
    ],
)


print(
    "Skipped no-time tracks:",
    results[
        "skipped_no_time_tracks"
    ],
)


print(
    "Track status counts   :",
    results[
        "track_status_counts"
    ],
)


print(
    "Event type counts     :",
    results[
        "event_type_counts"
    ],
)


print(
    "Metadata file         :",
    results["metadata_path"],
)


print("=" * 70)
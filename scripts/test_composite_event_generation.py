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

from ai.event_generation.composite_event_generator import (
    CompositeEventGenerator,
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


ATOMIC_EVENTS_PATH = (
    OUTPUT_ROOT
    / "atomic_event_generation_test"
    / "atomic_events.json"
)


COLOR_METADATA_PATH = (
    OUTPUT_ROOT
    / "color_analysis_object_aware"
    / "color_metadata.json"
)


MOTION_METADATA_PATH = (
    OUTPUT_ROOT
    / "motion_analysis_test"
    / "motion_metadata.json"
)


SPATIAL_METADATA_PATH = (
    OUTPUT_ROOT
    / "spatial_analysis_test"
    / "spatial_metadata.json"
)


RELATIONSHIP_METADATA_PATH = (
    OUTPUT_ROOT
    / "relationship_analysis_test"
    / "relationship_metadata.json"
)


OUTPUT_DIR = (
    OUTPUT_ROOT
    / "composite_event_generation_test"
)


generator = CompositeEventGenerator(
    atomic_events_path=(
        ATOMIC_EVENTS_PATH
    ),
    color_metadata_path=(
        COLOR_METADATA_PATH
    ),
    motion_metadata_path=(
        MOTION_METADATA_PATH
    ),
    spatial_metadata_path=(
        SPATIAL_METADATA_PATH
    ),
    relationship_metadata_path=(
        RELATIONSHIP_METADATA_PATH
    ),
    output_dir=OUTPUT_DIR,
)


print(
    "\nGenerating composite "
    "semantic events...\n"
)


results = generator.generate()


print("\n" + "=" * 70)

print(
    "COMPOSITE EVENT GENERATION TEST"
)

print("=" * 70)


print(
    "Total events       :",
    results["total_events"],
)


print(
    "Event type counts  :",
    results[
        "event_type_counts"
    ],
)


print(
    "Metadata file      :",
    results["metadata_path"],
)


print("=" * 70)
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

from ai.retrieval.event_quality_filter import (
    EventQualityFilter,
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


COMPOSITE_EVENTS_PATH = (
    OUTPUT_ROOT
    / "composite_event_generation_test"
    / "composite_events.json"
)


OUTPUT_DIR = (
    OUTPUT_ROOT
    / "event_quality_filter_test"
)


quality_filter = EventQualityFilter(
    composite_events_path=(
        COMPOSITE_EVENTS_PATH
    ),
    output_dir=OUTPUT_DIR,
    minimum_track_duration=0.10,
    minimum_relationship_frames=4,
    minimum_quality_score=0.50,
    include_review=True,
)


print(
    "\nFiltering composite events "
    "and generating retrieval documents...\n"
)


results = quality_filter.process()


print("\n" + "=" * 70)

print(
    "EVENT QUALITY FILTER TEST"
)

print("=" * 70)


print(
    "Input events          :",
    results["input_events"],
)


print(
    "Accepted events       :",
    results["accepted_events"],
)


print(
    "Rejected events       :",
    results["rejected_events"],
)


print(
    "Accepted type counts  :",
    results["accepted_type_counts"],
)


print(
    "Quality label counts  :",
    results["quality_label_counts"],
)


print(
    "Rejected reasons      :",
    results["rejected_reason_counts"],
)


print(
    "Retrieval documents   :",
    results["retrieval_documents"],
)


print(
    "Filtered events file  :",
    results["filtered_path"],
)


print(
    "Retrieval docs file   :",
    results["retrieval_path"],
)


print("=" * 70)
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

from ai.pipeline.visual_track_verifier import (
    VisualTrackVerifier,
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


RELIABILITY_METADATA_PATH = (
    PROJECT_ROOT
    / config["paths"]["output_root"]
    / "track_reliability_test"
    / "track_reliability.json"
)


MODEL_PATH = (
    PROJECT_ROOT
    / config["paths"]["model"]
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / config["paths"]["output_root"]
    / "visual_track_verification_test"
)


verifier = VisualTrackVerifier(
    reliability_metadata_path=(
        RELIABILITY_METADATA_PATH
    ),
    model_path=MODEL_PATH,
    output_dir=OUTPUT_DIR,
    confidence_threshold=0.20,
)


print(
    "\nRunning visual crop-class "
    "verification...\n"
)


results = verifier.analyze()


print("\n" + "=" * 70)

print(
    "VISUAL TRACK VERIFICATION TEST"
)

print("=" * 70)


print(
    "Total tracks  :",
    results["total_tracks"],
)


print(
    "Status counts :",
    results["status_counts"],
)


print(
    "Metadata file :",
    results["metadata_path"],
)


print("=" * 70)
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from ai.preprocessing.adaptive_sampler import (
    AdaptiveFrameSampler,
)

from ai.preprocessing.frame_extractor import (
    FrameExtractor,
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / config["paths"]["output_root"]
    / "frame_extraction_test"
)


motion_config = (
    config["adaptive_sampling"]["motion"]
)

sampling_config = (
    config["adaptive_sampling"]["sampling"]
)


sampler = AdaptiveFrameSampler(
    video_path=VIDEO_PATH,
    low_threshold=motion_config["low_threshold"],
    high_threshold=motion_config["high_threshold"],
    low_motion_fps=sampling_config["low_motion_fps"],
    medium_motion_fps=sampling_config[
        "medium_motion_fps"
    ],
    high_motion_fps=sampling_config[
        "high_motion_fps"
    ],
)


print("\nRunning adaptive sampling...")

sampling_results = sampler.sample()


extractor = FrameExtractor(
    video_path=VIDEO_PATH,
    output_dir=OUTPUT_DIR,
)


print("Extracting selected frames...")

extraction_results = extractor.extract(
    sampling_results["sampled_frames"]
)


print("\n" + "=" * 70)
print("FRAME EXTRACTION TEST")
print("=" * 70)

print(
    "Source frames       :",
    sampling_results["total_source_frames"],
)

print(
    "Selected frames     :",
    sampling_results["sampled_frame_count"],
)

print(
    "Saved frames        :",
    extraction_results["saved_frame_count"],
)

print(
    "Frames directory    :",
    extraction_results["frames_directory"],
)

print(
    "Metadata file       :",
    extraction_results["metadata_path"],
)

print("=" * 70)
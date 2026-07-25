import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from ai.preprocessing.adaptive_sampler import AdaptiveFrameSampler


CONFIG_PATH = (
    PROJECT_ROOT
    / "ai"
    / "configs"
    / "phase3_config.yaml"
)

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


VIDEO_PATH = PROJECT_ROOT / config["paths"]["test_video"]

motion_config = config["adaptive_sampling"]["motion"]
sampling_config = config["adaptive_sampling"]["sampling"]


sampler = AdaptiveFrameSampler(
    video_path=VIDEO_PATH,
    low_threshold=motion_config["low_threshold"],
    high_threshold=motion_config["high_threshold"],
    low_motion_fps=sampling_config["low_motion_fps"],
    medium_motion_fps=sampling_config["medium_motion_fps"],
    high_motion_fps=sampling_config["high_motion_fps"],
)

results = sampler.sample()


print("\n" + "=" * 70)
print("ADAPTIVE FRAME SAMPLING TEST")
print("=" * 70)

print("Source FPS           :", results["source_fps"])
print("Total source frames  :", results["total_source_frames"])
print("Sampled frames       :", results["sampled_frame_count"])
print("Motion counts        :", results["motion_counts"])

print("\nFIRST 10 SAMPLED FRAMES")
print("-" * 70)

for frame in results["sampled_frames"][:10]:
    print(
        f"Frame {frame['frame_index']:4} | "
        f"Time {frame['timestamp_seconds']:6.2f}s | "
        f"Motion {frame['motion_score']:6.2f} | "
        f"Level {frame['motion_level']:6} | "
        f"Sampling {frame['sampling_fps']:.1f} FPS"
    )

print("=" * 70)
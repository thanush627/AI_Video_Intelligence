import sys
from pathlib import Path

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.preprocessing.video_analyzer import VideoAnalyzer

VIDEO_PATH = PROJECT_ROOT / "test_videos" / "test.mp4"

analyzer = VideoAnalyzer(VIDEO_PATH)
metadata = analyzer.analyze()

print("\n" + "=" * 60)
print("VIDEO ANALYSIS")
print("=" * 60)

for key, value in metadata.items():
    print(f"{key:20}: {value}")

print("=" * 60)
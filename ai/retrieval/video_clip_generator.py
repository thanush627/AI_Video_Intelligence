import os
import subprocess
from typing import Optional

class VideoClipGenerator:
    def __init__(self, output_dir: str = "uploads/clips", padding: float = 1.5):
        self.output_dir = output_dir
        self.padding = padding
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_clip(self, video_path: str, start_time: float, end_time: float, output_path: Optional[str] = None) -> Optional[str]:
        if not os.path.exists(video_path):
            print(f"[FFmpeg Error] Source video not found: {video_path}")
            return None

        padded_start = max(0.0, start_time - self.padding)
        duration = (end_time - start_time) + (2 * self.padding)

        if not output_path:
            filename = f"clip_{int(start_time)}_{int(end_time)}.mp4"
            output_path = os.path.join(self.output_dir, filename)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",                     # Overwrite output files
            "-ss", str(padded_start),  # Fast seek before input
            "-i", video_path,
            "-t", str(duration),      # Duration
            "-c:v", "libx264",        # Ensure H.264 video encoding
            "-c:a", "aac",            # AAC audio
            "-preset", "fast",
            output_path
        ]

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return output_path
        except Exception as e:
            print(f"[FFmpeg Clip Extraction Failed]: {e}")
            return None
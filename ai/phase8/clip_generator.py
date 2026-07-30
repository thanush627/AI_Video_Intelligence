import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


class Phase8ClipGenerator:
    def __init__(self, output_dir: str = "uploads/clips"):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ffmpeg_path = shutil.which("ffmpeg")
        if self.ffmpeg_path is None:
            raise RuntimeError("FFmpeg executable not found. Install ffmpeg before running phase 8.")

    def _safe_float(self, value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _format_clip_name(self, event_id: str) -> str:
        return f"{event_id}.mp4"

    def _format_thumbnail_name(self, event_id: str) -> str:
        return f"{event_id}_thumbnail.jpg"

    def generate_clip(
        self,
        video_path: str,
        event_id: str,
        start_time: float,
        end_time: float,
        track_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        start_time = self._safe_float(start_time)
        end_time = self._safe_float(end_time)
        duration = max(0.0, end_time - start_time)

        if duration <= 0:
            raise ValueError("Clip duration must be positive")

        clip_path = self.output_dir / self._format_clip_name(event_id)
        thumbnail_path = self.output_dir / self._format_thumbnail_name(event_id)

        if clip_path.exists():
            clip_path.unlink()
        if thumbnail_path.exists():
            thumbnail_path.unlink()

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-ss",
            str(start_time),
            "-i",
            str(video_path),
            "-to",
            str(duration),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(clip_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        thumb_cmd = [
            self.ffmpeg_path,
            "-y",
            "-i",
            str(clip_path),
            "-vf",
            "scale=320:-1",
            "-frames:v",
            "1",
            str(thumbnail_path),
        ]
        subprocess.run(thumb_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        metadata_payload = {
            "event_id": event_id,
            "track_id": track_id,
            "video_path": str(video_path),
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "clip_path": str(clip_path),
            "thumbnail_path": str(thumbnail_path),
            "metadata": metadata or {},
        }

        with open(self.output_dir / f"{event_id}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=2)

        with open(self.output_dir / f"{event_id}_statistics.json", "w", encoding="utf-8") as f:
            json.dump({
                "event_id": event_id,
                "duration": duration,
                "clip_file": str(clip_path),
                "thumbnail_file": str(thumbnail_path),
            }, f, indent=2)

        return metadata_payload

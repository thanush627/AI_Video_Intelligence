import os
import shutil
import subprocess
from datetime import timedelta
from pathlib import Path


class VideoClipGenerator:

    def __init__(
        self,
        output_directory,
        context_before_seconds=2.0,
        context_after_seconds=2.0,
        minimum_clip_duration_seconds=1.0,
        overwrite=True,
    ):

        self.output_directory = Path(
            output_directory
        ).resolve()

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.context_before_seconds = float(
            context_before_seconds
        )

        self.context_after_seconds = float(
            context_after_seconds
        )

        self.minimum_clip_duration_seconds = float(
            minimum_clip_duration_seconds
        )

        self.overwrite = bool(
            overwrite
        )

        self.ffmpeg_path = shutil.which(
            "ffmpeg"
        )

        if self.ffmpeg_path is None:
            raise RuntimeError(
                "FFmpeg executable not found."
            )


    @staticmethod
    def _safe_float(
        value,
        default=0.0,
    ):
        try:
            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return default


    @staticmethod
    def _format_timestamp(
        seconds,
    ):
        seconds = max(
            0.0,
            float(seconds),
        )

        td = timedelta(
            seconds=seconds
        )

        total_seconds = int(
            td.total_seconds()
        )

        milliseconds = int(
            round(
                (seconds - total_seconds)
                * 1000
            )
        )

        hours = (
            total_seconds
            // 3600
        )

        minutes = (
            total_seconds
            % 3600
        ) // 60

        secs = (
            total_seconds
            % 60
        )

        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{secs:02d}."
            f"{milliseconds:03d}"
        )


    def _clip_filename(
        self,
        event_id,
    ):
        return (
            f"{event_id}.mp4"
        )


    def _clip_path(
        self,
        event_id,
    ):
        return (
            self.output_directory
            / self._clip_filename(
                event_id
            )
        )


    def _build_ffmpeg_command(
        self,
        video_path,
        clip_start,
        duration,
        output_path,
    ):

        command = [
            self.ffmpeg_path,
            "-y",
            "-ss",
            str(
                round(
                    clip_start,
                    3,
                )
            ),
            "-i",
            str(video_path),
            "-t",
            str(
                round(
                    duration,
                    3,
                )
            ),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-preset",
            "fast",
            str(output_path),
        ]

        return command
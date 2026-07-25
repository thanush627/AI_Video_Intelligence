import json
import math
from pathlib import Path

import cv2


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

        self.context_before_seconds = max(
            0.0,
            self._safe_float(
                context_before_seconds,
                default=2.0,
            ),
        )

        self.context_after_seconds = max(
            0.0,
            self._safe_float(
                context_after_seconds,
                default=2.0,
            ),
        )

        self.minimum_clip_duration_seconds = max(
            0.1,
            self._safe_float(
                minimum_clip_duration_seconds,
                default=1.0,
            ),
        )

        self.overwrite = bool(
            overwrite
        )


    def _safe_float(
        self,
        value,
        default=None,
    ):
        try:
            value = float(value)

            if not math.isfinite(
                value
            ):
                return default

            return value

        except (
            TypeError,
            ValueError,
        ):
            return default


    def _safe_int(
        self,
        value,
        default=0,
    ):
        try:
            return int(value)

        except (
            TypeError,
            ValueError,
        ):
            return default


    def _sanitize_filename(
        self,
        value,
    ):
        value = str(
            value
        ).strip()

        if not value:
            value = "event"

        safe_characters = []

        for character in value:

            if (
                character.isalnum()
                or character in (
                    "-",
                    "_",
                )
            ):
                safe_characters.append(
                    character
                )

            else:
                safe_characters.append(
                    "_"
                )

        sanitized = "".join(
            safe_characters
        )

        while "__" in sanitized:
            sanitized = sanitized.replace(
                "__",
                "_",
            )

        return (
            sanitized.strip(
                "_"
            )
            or "event"
        )


    def _format_timestamp(
        self,
        seconds,
    ):
        seconds = max(
            0.0,
            self._safe_float(
                seconds,
                default=0.0,
            ),
        )

        minutes = int(
            seconds // 60
        )

        remaining_seconds = (
            seconds
            - (
                minutes * 60
            )
        )

        return (
            f"{minutes:02d}:"
            f"{remaining_seconds:06.3f}"
        )


    def _get_video_metadata(
        self,
        video_path,
    ):
        video_path = Path(
            video_path
        ).resolve()

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video file does not exist: "
                f"{video_path}"
            )

        capture = cv2.VideoCapture(
            str(video_path)
        )

        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open video: "
                f"{video_path}"
            )

        fps = self._safe_float(
            capture.get(
                cv2.CAP_PROP_FPS
            ),
            default=0.0,
        )

        frame_count = self._safe_int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            ),
            default=0,
        )

        width = self._safe_int(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            ),
            default=0,
        )

        height = self._safe_int(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            ),
            default=0,
        )

        capture.release()

        if fps <= 0:
            raise RuntimeError(
                "Invalid source video FPS."
            )

        if frame_count <= 0:
            raise RuntimeError(
                "Invalid source video frame count."
            )

        if (
            width <= 0
            or height <= 0
        ):
            raise RuntimeError(
                "Invalid source video resolution."
            )

        duration_seconds = (
            frame_count / fps
        )

        return {
            "video_path": str(
                video_path
            ),
            "video_name": (
                video_path.name
            ),
            "fps": fps,
            "frame_count": (
                frame_count
            ),
            "width": width,
            "height": height,
            "duration_seconds": (
                duration_seconds
            ),
        }


    def _normalize_event_times(
        self,
        start_time_seconds,
        end_time_seconds,
        video_duration_seconds,
        context_before_seconds,
        context_after_seconds,
    ):
        start_time = self._safe_float(
            start_time_seconds
        )

        end_time = self._safe_float(
            end_time_seconds
        )

        if (
            start_time is None
            or end_time is None
        ):
            raise ValueError(
                "Event start and end times "
                "must be valid numbers."
            )

        if start_time > end_time:
            start_time, end_time = (
                end_time,
                start_time,
            )

        start_time = max(
            0.0,
            start_time,
        )

        end_time = max(
            0.0,
            end_time,
        )

        video_duration_seconds = max(
            0.0,
            self._safe_float(
                video_duration_seconds,
                default=0.0,
            ),
        )

        if start_time >= (
            video_duration_seconds
        ):
            raise ValueError(
                "Event starts outside the "
                "source video duration."
            )

        end_time = min(
            end_time,
            video_duration_seconds,
        )

        context_before_seconds = max(
            0.0,
            self._safe_float(
                context_before_seconds,
                default=0.0,
            ),
        )

        context_after_seconds = max(
            0.0,
            self._safe_float(
                context_after_seconds,
                default=0.0,
            ),
        )

        clip_start = max(
            0.0,
            start_time
            - context_before_seconds,
        )

        clip_end = min(
            video_duration_seconds,
            end_time
            + context_after_seconds,
        )

        current_duration = (
            clip_end - clip_start
        )

        if current_duration < (
            self.minimum_clip_duration_seconds
        ):
            missing_duration = (
                self.minimum_clip_duration_seconds
                - current_duration
            )

            left_extension = (
                missing_duration / 2.0
            )

            right_extension = (
                missing_duration / 2.0
            )

            clip_start = max(
                0.0,
                clip_start
                - left_extension,
            )

            clip_end = min(
                video_duration_seconds,
                clip_end
                + right_extension,
            )

            current_duration = (
                clip_end - clip_start
            )

            if current_duration < (
                self.minimum_clip_duration_seconds
            ):

                if clip_start <= 0.0:
                    clip_end = min(
                        video_duration_seconds,
                        self.minimum_clip_duration_seconds,
                    )

                elif clip_end >= (
                    video_duration_seconds
                ):
                    clip_start = max(
                        0.0,
                        video_duration_seconds
                        - self.minimum_clip_duration_seconds,
                    )

        if clip_end <= clip_start:
            raise ValueError(
                "Generated clip has an "
                "invalid time range."
            )

        return {
            "event_start_time_seconds": (
                start_time
            ),
            "event_end_time_seconds": (
                end_time
            ),
            "clip_start_time_seconds": (
                clip_start
            ),
            "clip_end_time_seconds": (
                clip_end
            ),
            "clip_duration_seconds": (
                clip_end - clip_start
            ),
        }


    def _create_video_writer(
        self,
        output_path,
        fps,
        width,
        height,
    ):
        codec_candidates = [
            "mp4v",
            "avc1",
        ]

        for codec_name in codec_candidates:

            fourcc = cv2.VideoWriter_fourcc(
                *codec_name
            )

            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (
                    width,
                    height,
                ),
            )

            if writer.isOpened():
                return (
                    writer,
                    codec_name,
                )

            writer.release()

        raise RuntimeError(
            "Could not create MP4 video writer."
        )


    def generate_clip(
        self,
        video_path,
        event_id,
        start_time_seconds,
        end_time_seconds,
        context_before_seconds=None,
        context_after_seconds=None,
    ):
        video_metadata = (
            self._get_video_metadata(
                video_path
            )
        )

        if context_before_seconds is None:
            context_before_seconds = (
                self.context_before_seconds
            )

        if context_after_seconds is None:
            context_after_seconds = (
                self.context_after_seconds
            )

        time_data = (
            self._normalize_event_times(
                start_time_seconds=(
                    start_time_seconds
                ),
                end_time_seconds=(
                    end_time_seconds
                ),
                video_duration_seconds=(
                    video_metadata[
                        "duration_seconds"
                    ]
                ),
                context_before_seconds=(
                    context_before_seconds
                ),
                context_after_seconds=(
                    context_after_seconds
                ),
            )
        )

        safe_event_id = (
            self._sanitize_filename(
                event_id
            )
        )

        output_filename = (
            f"{safe_event_id}"
            f"_"
            f"{time_data['clip_start_time_seconds']:.3f}"
            f"_to_"
            f"{time_data['clip_end_time_seconds']:.3f}"
            f".mp4"
        )

        output_path = (
            self.output_directory
            / output_filename
        )

        if (
            output_path.exists()
            and not self.overwrite
        ):
            return {
                "success": True,
                "status": "existing",
                "event_id": str(
                    event_id
                ),
                "clip_path": str(
                    output_path
                ),
                **time_data,
            }

        capture = cv2.VideoCapture(
            video_metadata[
                "video_path"
            ]
        )

        if not capture.isOpened():
            raise RuntimeError(
                "Could not reopen source video."
            )

        fps = video_metadata[
            "fps"
        ]

        start_frame = max(
            0,
            int(
                math.floor(
                    time_data[
                        "clip_start_time_seconds"
                    ]
                    * fps
                )
            ),
        )

        end_frame = min(
            video_metadata[
                "frame_count"
            ],
            int(
                math.ceil(
                    time_data[
                        "clip_end_time_seconds"
                    ]
                    * fps
                )
            ),
        )

        if end_frame <= start_frame:
            capture.release()

            raise RuntimeError(
                "Calculated clip frame range "
                "is invalid."
            )

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            start_frame,
        )

        writer = None

        try:
            writer, codec_name = (
                self._create_video_writer(
                    output_path=(
                        output_path
                    ),
                    fps=fps,
                    width=(
                        video_metadata[
                            "width"
                        ]
                    ),
                    height=(
                        video_metadata[
                            "height"
                        ]
                    ),
                )
            )

            frames_written = 0

            current_frame = (
                start_frame
            )

            while current_frame < end_frame:

                success, frame = (
                    capture.read()
                )

                if not success:
                    break

                writer.write(
                    frame
                )

                frames_written += 1

                current_frame += 1

        finally:
            capture.release()

            if writer is not None:
                writer.release()

        if frames_written <= 0:

            if output_path.exists():
                output_path.unlink()

            raise RuntimeError(
                "No frames were written "
                "to the output clip."
            )

        actual_duration_seconds = (
            frames_written / fps
        )

        result = {
            "success": True,
            "status": "generated",
            "event_id": str(
                event_id
            ),
            "source_video_path": (
                video_metadata[
                    "video_path"
                ]
            ),
            "source_video_name": (
                video_metadata[
                    "video_name"
                ]
            ),
            "clip_path": str(
                output_path.resolve()
            ),
            "clip_filename": (
                output_path.name
            ),
            "codec": codec_name,
            "fps": round(
                fps,
                6,
            ),
            "width": (
                video_metadata[
                    "width"
                ]
            ),
            "height": (
                video_metadata[
                    "height"
                ]
            ),
            "start_frame": (
                start_frame
            ),
            "end_frame": (
                start_frame
                + frames_written
                - 1
            ),
            "frames_written": (
                frames_written
            ),
            "event_start_time_seconds": round(
                time_data[
                    "event_start_time_seconds"
                ],
                4,
            ),
            "event_end_time_seconds": round(
                time_data[
                    "event_end_time_seconds"
                ],
                4,
            ),
            "clip_start_time_seconds": round(
                time_data[
                    "clip_start_time_seconds"
                ],
                4,
            ),
            "clip_end_time_seconds": round(
                time_data[
                    "clip_end_time_seconds"
                ],
                4,
            ),
            "requested_clip_duration_seconds": round(
                time_data[
                    "clip_duration_seconds"
                ],
                4,
            ),
            "actual_clip_duration_seconds": round(
                actual_duration_seconds,
                4,
            ),
            "clip_start_timestamp": (
                self._format_timestamp(
                    time_data[
                        "clip_start_time_seconds"
                    ]
                )
            ),
            "clip_end_timestamp": (
                self._format_timestamp(
                    time_data[
                        "clip_end_time_seconds"
                    ]
                )
            ),
        }

        return result


    def generate_from_retrieval_response(
        self,
        video_path,
        retrieval_response,
        maximum_clips=None,
    ):
        if not isinstance(
            retrieval_response,
            dict,
        ):
            raise TypeError(
                "retrieval_response must "
                "be a dictionary."
            )

        results = retrieval_response.get(
            "results",
            [],
        )

        if not isinstance(
            results,
            list,
        ):
            results = []

        if maximum_clips is not None:
            maximum_clips = max(
                0,
                self._safe_int(
                    maximum_clips,
                    default=0,
                ),
            )

            results = results[
                :maximum_clips
            ]

        generated_clips = []

        failed_clips = []

        for result in results:

            if not isinstance(
                result,
                dict,
            ):
                continue

            event_id = result.get(
                "event_id",
                "event",
            )

            try:
                clip_result = (
                    self.generate_clip(
                        video_path=(
                            video_path
                        ),
                        event_id=(
                            event_id
                        ),
                        start_time_seconds=(
                            result.get(
                                "start_time_seconds"
                            )
                        ),
                        end_time_seconds=(
                            result.get(
                                "end_time_seconds"
                            )
                        ),
                    )
                )

                generated_clips.append(
                    clip_result
                )

            except Exception as error:
                failed_clips.append(
                    {
                        "event_id": str(
                            event_id
                        ),
                        "error": str(
                            error
                        ),
                    }
                )

        return {
            "success": (
                len(failed_clips) == 0
            ),
            "query": retrieval_response.get(
                "query",
                "",
            ),
            "requested_events": len(
                results
            ),
            "generated_clip_count": len(
                generated_clips
            ),
            "failed_clip_count": len(
                failed_clips
            ),
            "clips": generated_clips,
            "failures": failed_clips,
        }


    def save_metadata(
        self,
        metadata,
        output_path,
    ):
        output_path = Path(
            output_path
        ).resolve()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=2,
                ensure_ascii=False,
            )

        return output_path
import json
from collections import Counter
from pathlib import Path


class AtomicEventGenerator:

    HUMAN_CLASSES = {
        "pedestrian",
        "people",
    }

    def __init__(
        self,
        track_metadata_path,
        reliability_metadata_path,
        color_metadata_path,
        output_dir,
        source_fps,
        include_review=True,
    ):
        self.track_metadata_path = Path(
            track_metadata_path
        )

        self.reliability_metadata_path = Path(
            reliability_metadata_path
        )

        self.color_metadata_path = Path(
            color_metadata_path
        )

        self.output_dir = Path(
            output_dir
        )

        self.source_fps = float(
            source_fps
        )

        self.include_review = bool(
            include_review
        )

        for path in [
            self.track_metadata_path,
            self.reliability_metadata_path,
            self.color_metadata_path,
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required metadata not found: {path}"
                )

        if self.source_fps <= 0:
            raise ValueError(
                "source_fps must be greater than 0"
            )

    @staticmethod
    def _load_json(path):

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    @staticmethod
    def _track_key(track_id):

        return str(
            int(track_id)
        )

    @staticmethod
    def _extract_track_id(
        track_key,
        track,
    ):

        value = track.get(
            "track_id",
            track_key,
        )

        return int(value)

    @staticmethod
    def _extract_observations(track):

        possible_keys = [
            "observations",
            "track_history",
            "detections",
            "frames",
        ]

        for key in possible_keys:

            value = track.get(key)

            if isinstance(value, list):
                return value

        return []

    @staticmethod
    def _extract_frame_index(
        observation,
    ):

        possible_keys = [
            "frame_index",
            "frame_id",
            "frame_number",
            "frame",
        ]

        for key in possible_keys:

            value = observation.get(key)

            if isinstance(
                value,
                (int, float),
            ):
                return int(value)

        return None

    @staticmethod
    def _extract_timestamp(
        observation,
    ):

        possible_keys = [
            "timestamp_seconds",
            "timestamp",
            "time_seconds",
            "time",
        ]

        for key in possible_keys:

            value = observation.get(key)

            if isinstance(
                value,
                (int, float),
            ):
                return float(value)

        return None

    def _extract_temporal_bounds(
        self,
        track,
    ):

        observations = (
            self._extract_observations(
                track
            )
        )

        frame_indices = []

        timestamps = []

        for observation in observations:

            if not isinstance(
                observation,
                dict,
            ):
                continue

            frame_index = (
                self._extract_frame_index(
                    observation
                )
            )

            timestamp = (
                self._extract_timestamp(
                    observation
                )
            )

            if frame_index is not None:
                frame_indices.append(
                    frame_index
                )

            if timestamp is not None:
                timestamps.append(
                    timestamp
                )

        if frame_indices:

            first_frame = min(
                frame_indices
            )

            last_frame = max(
                frame_indices
            )

        else:

            first_frame = track.get(
                "first_frame"
            )

            last_frame = track.get(
                "last_frame"
            )

        if timestamps:

            start_time = min(
                timestamps
            )

            end_time = max(
                timestamps
            )

        elif (
            first_frame is not None
            and last_frame is not None
        ):

            start_time = (
                float(first_frame)
                / self.source_fps
            )

            end_time = (
                float(last_frame)
                / self.source_fps
            )

        else:

            return None

        duration = max(
            0.0,
            end_time - start_time,
        )

        return {
            "first_frame": (
                int(first_frame)
                if first_frame is not None
                else None
            ),
            "last_frame": (
                int(last_frame)
                if last_frame is not None
                else None
            ),
            "start_time": round(
                start_time,
                4,
            ),
            "end_time": round(
                end_time,
                4,
            ),
            "duration": round(
                duration,
                4,
            ),
            "observation_count": len(
                observations
            ),
        }

    @staticmethod
    def _find_track(
        tracks,
        track_id,
    ):

        target_id = int(
            track_id
        )

        for track_key, track in (
            tracks.items()
        ):

            current_id = track.get(
                "track_id",
                track_key,
            )

            try:

                if int(current_id) == target_id:
                    return track

            except (
                TypeError,
                ValueError,
            ):
                continue

        return None

    def _get_reliability(
        self,
        reliability_tracks,
        track_id,
    ):

        track = self._find_track(
            reliability_tracks,
            track_id,
        )

        if track is None:

            return {
                "status": "reject",
                "score": 0.0,
            }

        reliability = track.get(
            "reliability",
            {}
        )

        return {
            "status": reliability.get(
                "status",
                "reject",
            ),
            "score": float(
                reliability.get(
                    "reliability_score",
                    0.0,
                )
            ),
        }

    def _get_color(
        self,
        color_tracks,
        track_id,
        class_name,
    ):

        track = self._find_track(
            color_tracks,
            track_id,
        )

        if track is None:

            return {
                "description": "",
                "attributes": {},
            }

        analysis = track.get(
            "final_color_analysis",
            {},
        )

        if class_name in self.HUMAN_CLASSES:

            upper = analysis.get(
                "upper_body",
                {},
            ).get(
                "primary_color",
                "unknown",
            )

            lower = analysis.get(
                "lower_body",
                {},
            ).get(
                "primary_color",
                "unknown",
            )

            attributes = {
                "upper_body_color": upper,
                "lower_body_color": lower,
            }

            parts = []

            if upper != "unknown":

                parts.append(
                    f"{upper} upper clothing"
                )

            if lower != "unknown":

                parts.append(
                    f"{lower} lower clothing"
                )

            if parts:

                description = (
                    " with "
                    + " and ".join(parts)
                )

            else:

                description = ""

        else:

            object_color = analysis.get(
                "object_color",
                {},
            )

            primary = object_color.get(
                "primary_color",
                "unknown",
            )

            confidence = float(
                object_color.get(
                    "confidence",
                    0.0,
                )
            )

            attributes = {
                "primary_color": primary,
                "color_confidence": (
                    confidence
                ),
            }

            if primary != "unknown":

                description = (
                    f"{primary} "
                )

            else:

                description = ""

        return {
            "description": description,
            "attributes": attributes,
        }

    @staticmethod
    def _event_id(
        event_number,
    ):

        return (
            f"event_"
            f"{event_number:06d}"
        )

    def _create_observed_event(
        self,
        event_number,
        track_id,
        class_name,
        temporal,
        reliability,
        color,
    ):

        start_time = temporal[
            "start_time"
        ]

        end_time = temporal[
            "end_time"
        ]

        duration = temporal[
            "duration"
        ]

        if class_name in self.HUMAN_CLASSES:

            subject_text = (
                f"{class_name}"
                f"{color['description']}"
            )

        else:

            subject_text = (
                f"{color['description']}"
                f"{class_name}"
            )

        text = (
            f"{subject_text} observed "
            f"from {start_time:.2f}s "
            f"to {end_time:.2f}s "
            f"for {duration:.2f}s"
        )

        return {
            "event_id": self._event_id(
                event_number
            ),
            "event_type": (
                "object_observed"
            ),
            "track_id": track_id,
            "class_name": class_name,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "first_frame": temporal[
                "first_frame"
            ],
            "last_frame": temporal[
                "last_frame"
            ],
            "reliability_status": (
                reliability["status"]
            ),
            "reliability_score": round(
                reliability["score"],
                4,
            ),
            "attributes": color[
                "attributes"
            ],
            "event_text": text,
        }

    def _create_boundary_event(
        self,
        event_number,
        event_type,
        track_id,
        class_name,
        timestamp,
        frame_index,
        reliability,
        color,
    ):

        if class_name in self.HUMAN_CLASSES:

            subject_text = (
                f"{class_name}"
                f"{color['description']}"
            )

        else:

            subject_text = (
                f"{color['description']}"
                f"{class_name}"
            )

        if event_type == "object_entered":

            action_text = (
                "entered the scene"
            )

        else:

            action_text = (
                "exited the scene"
            )

        text = (
            f"{subject_text} "
            f"{action_text} "
            f"at {timestamp:.2f}s"
        )

        return {
            "event_id": self._event_id(
                event_number
            ),
            "event_type": event_type,
            "track_id": track_id,
            "class_name": class_name,
            "timestamp": round(
                timestamp,
                4,
            ),
            "frame_index": frame_index,
            "reliability_status": (
                reliability["status"]
            ),
            "reliability_score": round(
                reliability["score"],
                4,
            ),
            "attributes": color[
                "attributes"
            ],
            "event_text": text,
        }

    def generate(self):

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        tracking_data = self._load_json(
            self.track_metadata_path
        )

        reliability_data = self._load_json(
            self.reliability_metadata_path
        )

        color_data = self._load_json(
            self.color_metadata_path
        )

        tracking_tracks = (
            tracking_data.get(
                "tracks",
                tracking_data,
            )
        )

        reliability_tracks = (
            reliability_data.get(
                "tracks",
                {},
            )
        )

        color_tracks = (
            color_data.get(
                "tracks",
                {},
            )
        )

        events = []

        skipped_reject = 0
        skipped_no_time = 0

        track_status_counts = Counter()
        event_type_counts = Counter()

        event_number = 1

        for track_key, track in (
            tracking_tracks.items()
        ):

            track_id = (
                self._extract_track_id(
                    track_key,
                    track,
                )
            )

            class_name = track.get(
                "class_name",
                "unknown",
            )

            reliability = (
                self._get_reliability(
                    reliability_tracks,
                    track_id,
                )
            )

            status = reliability[
                "status"
            ]

            if status == "reject":

                skipped_reject += 1
                continue

            if (
                status == "review"
                and not self.include_review
            ):

                skipped_reject += 1
                continue

            temporal = (
                self._extract_temporal_bounds(
                    track
                )
            )

            if temporal is None:

                skipped_no_time += 1
                continue

            color = self._get_color(
                color_tracks,
                track_id,
                class_name,
            )

            track_status_counts[
                status
            ] += 1

            observed_event = (
                self._create_observed_event(
                    event_number,
                    track_id,
                    class_name,
                    temporal,
                    reliability,
                    color,
                )
            )

            events.append(
                observed_event
            )

            event_type_counts[
                "object_observed"
            ] += 1

            event_number += 1


            entered_event = (
                self._create_boundary_event(
                    event_number,
                    "object_entered",
                    track_id,
                    class_name,
                    temporal["start_time"],
                    temporal["first_frame"],
                    reliability,
                    color,
                )
            )

            events.append(
                entered_event
            )

            event_type_counts[
                "object_entered"
            ] += 1

            event_number += 1


            exited_event = (
                self._create_boundary_event(
                    event_number,
                    "object_exited",
                    track_id,
                    class_name,
                    temporal["end_time"],
                    temporal["last_frame"],
                    reliability,
                    color,
                )
            )

            events.append(
                exited_event
            )

            event_type_counts[
                "object_exited"
            ] += 1

            event_number += 1


            print(
                f"Track {track_id:3} | "
                f"{class_name:10} | "
                f"{status.upper():8} | "
                f"{temporal['start_time']:6.2f}s "
                f"→ "
                f"{temporal['end_time']:6.2f}s | "
                f"{observed_event['event_text']}"
            )

        events.sort(
            key=lambda event: (
                event.get(
                    "start_time",
                    event.get(
                        "timestamp",
                        0.0,
                    ),
                ),
                event["event_id"],
            )
        )

        output_data = {
            "summary": {
                "total_events": len(
                    events
                ),
                "usable_tracks": sum(
                    track_status_counts.values()
                ),
                "skipped_reject_tracks": (
                    skipped_reject
                ),
                "skipped_no_time_tracks": (
                    skipped_no_time
                ),
                "track_status_counts": dict(
                    track_status_counts
                ),
                "event_type_counts": dict(
                    event_type_counts
                ),
                "source_fps": (
                    self.source_fps
                ),
            },
            "events": events,
        }

        output_path = (
            self.output_dir
            / "atomic_events.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                output_data,
                file,
                indent=2,
            )

        return {
            "total_events": len(
                events
            ),
            "usable_tracks": sum(
                track_status_counts.values()
            ),
            "skipped_reject_tracks": (
                skipped_reject
            ),
            "skipped_no_time_tracks": (
                skipped_no_time
            ),
            "track_status_counts": dict(
                track_status_counts
            ),
            "event_type_counts": dict(
                event_type_counts
            ),
            "metadata_path": str(
                output_path
            ),
        }
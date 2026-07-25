import json
from collections import Counter
from pathlib import Path


class CompositeEventGenerator:

    def __init__(
        self,
        atomic_events_path,
        color_metadata_path,
        motion_metadata_path,
        spatial_metadata_path,
        relationship_metadata_path,
        output_dir,
    ):
        self.atomic_events_path = Path(atomic_events_path)
        self.color_metadata_path = Path(color_metadata_path)
        self.motion_metadata_path = Path(motion_metadata_path)
        self.spatial_metadata_path = Path(spatial_metadata_path)
        self.relationship_metadata_path = Path(relationship_metadata_path)
        self.output_dir = Path(output_dir)

        for path in [
            self.atomic_events_path,
            self.color_metadata_path,
            self.motion_metadata_path,
            self.spatial_metadata_path,
            self.relationship_metadata_path,
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required metadata not found: {path}"
                )

    @staticmethod
    def _load_json(path):
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _find_track(tracks, track_id):
        target_id = int(track_id)

        for key, track in tracks.items():
            current_id = track.get("track_id", key)

            try:
                if int(current_id) == target_id:
                    return track
            except (TypeError, ValueError):
                continue

        return None

    @staticmethod
    def _event_id(number):
        return f"composite_event_{number:06d}"

    @staticmethod
    def _clean_region(region):
        return str(region).replace("_", " ")

    @staticmethod
    def _get_colour_attributes(colour_track):
        if not colour_track:
            return {}

        final_analysis = colour_track.get(
            "final_color_analysis",
            {},
        )

        if not isinstance(final_analysis, dict):
            return {}

        attributes = {}

        # Vehicle / object colour
        object_color_data = final_analysis.get(
            "object_color"
        )

        if isinstance(object_color_data, dict):
            primary_color = object_color_data.get(
                "primary_color"
            )

            if (
                primary_color
                and primary_color != "unknown"
            ):
                attributes["object_color"] = primary_color

        # Human upper-body colour
        upper_body_data = final_analysis.get(
            "upper_body"
        )

        if isinstance(upper_body_data, dict):
            upper_color = upper_body_data.get(
                "primary_color"
            )

            if (
                upper_color
                and upper_color != "unknown"
            ):
                attributes["upper_color"] = upper_color

        # Human lower-body colour
        lower_body_data = final_analysis.get(
            "lower_body"
        )

        if isinstance(lower_body_data, dict):
            lower_color = lower_body_data.get(
                "primary_color"
            )

            if (
                lower_color
                and lower_color != "unknown"
            ):
                attributes["lower_color"] = lower_color

        return attributes

    @staticmethod
    def _object_phrase(
        class_name,
        colour_attributes,
    ):
        object_color = colour_attributes.get(
            "object_color"
        )

        if (
            object_color
            and object_color != "unknown"
        ):
            return f"{object_color} {class_name}"

        upper = colour_attributes.get(
            "upper_color"
        )

        lower = colour_attributes.get(
            "lower_color"
        )

        if upper and lower:
            return (
                f"{class_name} wearing "
                f"{upper} upper clothing "
                f"and {lower} lower clothing"
            )

        if upper:
            return (
                f"{class_name} wearing "
                f"{upper} upper clothing"
            )

        if lower:
            return (
                f"{class_name} wearing "
                f"{lower} lower clothing"
            )

        return class_name

    def _build_track_description(
        self,
        class_name,
        colour_attributes,
        motion,
        spatial,
    ):
        object_phrase = self._object_phrase(
            class_name,
            colour_attributes,
        )

        motion_state = motion.get(
            "motion_state",
            "unknown",
        )

        direction = motion.get(
            "direction",
            "unknown",
        )

        start_region = spatial.get(
            "start_region",
            {},
        ).get(
            "combined",
            "unknown",
        )

        end_region = spatial.get(
            "end_region",
            {},
        ).get(
            "combined",
            "unknown",
        )

        start_text = self._clean_region(
            start_region
        )

        end_text = self._clean_region(
            end_region
        )

        if motion_state == "stationary":
            return (
                f"{object_phrase} remained "
                f"stationary in the "
                f"{start_text} region"
            )

        direction_text = direction.replace(
            "_",
            " ",
        )

        if (
            start_region != "unknown"
            and end_region != "unknown"
            and start_region != end_region
        ):
            return (
                f"{object_phrase} moved "
                f"from the {start_text} region "
                f"to the {end_text} region"
            )

        if direction not in {
            "unknown",
            "stationary",
        }:
            return (
                f"{object_phrase} moved "
                f"{direction_text} within "
                f"the {start_text} region"
            )

        return (
            f"{object_phrase} was observed "
            f"in the {start_text} region"
        )

    @staticmethod
    def _relationship_description(
        relationship,
    ):
        class_a = relationship.get(
            "track_a_class",
            "object",
        )

        class_b = relationship.get(
            "track_b_class",
            "object",
        )

        types = relationship.get(
            "relationship_types",
            [],
        )

        if "moving_together" in types:
            return (
                f"A {class_a} and a "
                f"{class_b} moved together"
            )

        if "human_near_vehicle" in types:
            return (
                f"A person was near a "
                f"{class_b if class_b != 'pedestrian' else class_a}"
            )

        if "overlapping" in types:
            return (
                f"A {class_a} and a "
                f"{class_b} overlapped "
                f"while visible together"
            )

        if "near" in types:
            return (
                f"A {class_a} and a "
                f"{class_b} remained near "
                f"each other"
            )

        return (
            f"A {class_a} and a "
            f"{class_b} were visible "
            f"together"
        )

    @staticmethod
    def _safe_relationship_timestamps(
        relationship,
    ):
        first_timestamp = relationship.get(
            "first_timestamp"
        )

        last_timestamp = relationship.get(
            "last_timestamp"
        )

        if (
            first_timestamp is None
            or last_timestamp is None
        ):
            return (
                first_timestamp,
                last_timestamp,
                None,
            )

        try:
            first_timestamp = float(
                first_timestamp
            )

            last_timestamp = float(
                last_timestamp
            )

        except (TypeError, ValueError):
            return (
                first_timestamp,
                last_timestamp,
                None,
            )

        start_time_seconds = min(
            first_timestamp,
            last_timestamp,
        )

        end_time_seconds = max(
            first_timestamp,
            last_timestamp,
        )

        duration_seconds = max(
            0.0,
            end_time_seconds
            - start_time_seconds,
        )

        return (
            start_time_seconds,
            end_time_seconds,
            duration_seconds,
        )

    def generate(self):
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        atomic_data = self._load_json(
            self.atomic_events_path
        )

        colour_data = self._load_json(
            self.color_metadata_path
        )

        motion_data = self._load_json(
            self.motion_metadata_path
        )

        spatial_data = self._load_json(
            self.spatial_metadata_path
        )

        relationship_data = self._load_json(
            self.relationship_metadata_path
        )

        atomic_events = atomic_data.get(
            "events",
            [],
        )

        colour_tracks = colour_data.get(
            "tracks",
            {},
        )

        motion_tracks = motion_data.get(
            "tracks",
            {},
        )

        spatial_tracks = spatial_data.get(
            "tracks",
            {},
        )

        relationships = relationship_data.get(
            "relationships",
            [],
        )

        observed_events = {}

        for event in atomic_events:
            if (
                event.get("event_type")
                != "object_observed"
            ):
                continue

            track_id = event.get("track_id")

            if track_id is None:
                continue

            observed_events[int(track_id)] = event

        composite_events = []
        event_type_counts = Counter()
        event_number = 1

        # -------------------------------------------------
        # TRACK SEMANTIC EVENTS
        # -------------------------------------------------

        for track_id, atomic_event in (
            observed_events.items()
        ):
            motion_track = self._find_track(
                motion_tracks,
                track_id,
            )

            spatial_track = self._find_track(
                spatial_tracks,
                track_id,
            )

            colour_track = self._find_track(
                colour_tracks,
                track_id,
            )

            if (
                motion_track is None
                or spatial_track is None
            ):
                continue

            class_name = atomic_event.get(
                "class_name",
                "unknown",
            )

            motion = motion_track.get(
                "motion",
                {},
            )

            colour_attributes = (
                self._get_colour_attributes(
                    colour_track
                )
            )

            description = (
                self._build_track_description(
                    class_name,
                    colour_attributes,
                    motion,
                    spatial_track,
                )
            )

            # Correct atomic event timestamp mapping
            start_time = atomic_event.get(
                "start_time"
            )

            end_time = atomic_event.get(
                "end_time"
            )

            duration = atomic_event.get(
                "duration"
            )

            # Safe fallback if duration is missing
            if (
                duration is None
                and start_time is not None
                and end_time is not None
            ):
                duration = max(
                    0.0,
                    float(end_time)
                    - float(start_time),
                )

            event = {
                "event_id": self._event_id(
                    event_number
                ),
                "event_type": (
                    "track_semantic_event"
                ),
                "track_ids": [
                    track_id
                ],
                "class_names": [
                    class_name
                ],
                "start_time_seconds": (
                    start_time
                ),
                "end_time_seconds": (
                    end_time
                ),
                "duration_seconds": (
                    duration
                ),
                "reliability_status": (
                    atomic_event.get(
                        "reliability_status",
                        "unknown",
                    )
                ),
                "colour_attributes": (
                    colour_attributes
                ),
                "motion_state": motion.get(
                    "motion_state",
                    "unknown",
                ),
                "motion_direction": motion.get(
                    "direction",
                    "unknown",
                ),
                "start_region": spatial_track.get(
                    "start_region"
                ),
                "end_region": spatial_track.get(
                    "end_region"
                ),
                "region_transition": (
                    spatial_track.get(
                        "region_transition"
                    )
                ),
                "description": description,
            }

            composite_events.append(event)

            event_type_counts[
                "track_semantic_event"
            ] += 1

            print(
                f"Event {event_number:3} | "
                f"Track {track_id:3} | "
                f"{description}"
            )

            event_number += 1

        # -------------------------------------------------
        # RELATIONSHIP EVENTS
        # -------------------------------------------------

        for relationship in relationships:
            description = (
                self._relationship_description(
                    relationship
                )
            )

            (
                start_time_seconds,
                end_time_seconds,
                duration_seconds,
            ) = self._safe_relationship_timestamps(
                relationship
            )

            event = {
                "event_id": self._event_id(
                    event_number
                ),
                "event_type": (
                    "relationship_event"
                ),
                "track_ids": [
                    relationship.get(
                        "track_a_id"
                    ),
                    relationship.get(
                        "track_b_id"
                    ),
                ],
                "class_names": [
                    relationship.get(
                        "track_a_class"
                    ),
                    relationship.get(
                        "track_b_class"
                    ),
                ],
                "relationship_types": (
                    relationship.get(
                        "relationship_types",
                        [],
                    )
                ),
                "start_time_seconds": (
                    start_time_seconds
                ),
                "end_time_seconds": (
                    end_time_seconds
                ),
                "duration_seconds": (
                    duration_seconds
                ),
                "shared_frames": (
                    relationship.get(
                        "shared_frames"
                    )
                ),
                "near_ratio": (
                    relationship.get(
                        "near_ratio"
                    )
                ),
                "overlap_ratio": (
                    relationship.get(
                        "overlap_ratio"
                    )
                ),
                "description": description,
            }

            composite_events.append(event)

            event_type_counts[
                "relationship_event"
            ] += 1

            print(
                f"Event {event_number:3} | "
                f"Relationship | "
                f"{description}"
            )

            event_number += 1

        output_data = {
            "summary": {
                "total_events": len(
                    composite_events
                ),
                "event_type_counts": dict(
                    event_type_counts
                ),
                "track_events": (
                    event_type_counts.get(
                        "track_semantic_event",
                        0,
                    )
                ),
                "relationship_events": (
                    event_type_counts.get(
                        "relationship_event",
                        0,
                    )
                ),
            },
            "events": composite_events,
        }

        output_path = (
            self.output_dir
            / "composite_events.json"
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
                composite_events
            ),
            "event_type_counts": dict(
                event_type_counts
            ),
            "metadata_path": str(
                output_path
            ),
        }
import json
from collections import Counter
from pathlib import Path


class SpatialAnalyzer:

    def __init__(
        self,
        motion_metadata_path,
        output_dir,
        frame_width,
        frame_height,
    ):
        self.motion_metadata_path = Path(
            motion_metadata_path
        )

        self.output_dir = Path(
            output_dir
        )

        self.frame_width = float(
            frame_width
        )

        self.frame_height = float(
            frame_height
        )

        if not self.motion_metadata_path.exists():
            raise FileNotFoundError(
                f"Motion metadata not found: "
                f"{self.motion_metadata_path}"
            )

        if (
            self.frame_width <= 0
            or self.frame_height <= 0
        ):
            raise ValueError(
                "Frame width and height "
                "must be greater than 0"
            )

    @staticmethod
    def _load_json(path):

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    def _horizontal_region(self, x):

        normalized_x = (
            float(x)
            / self.frame_width
        )

        if normalized_x < 1 / 3:
            return "left"

        if normalized_x < 2 / 3:
            return "center"

        return "right"

    def _vertical_region(self, y):

        normalized_y = (
            float(y)
            / self.frame_height
        )

        if normalized_y < 1 / 3:
            return "top"

        if normalized_y < 2 / 3:
            return "middle"

        return "bottom"

    def _scene_region(
        self,
        x,
        y,
    ):

        horizontal = (
            self._horizontal_region(x)
        )

        vertical = (
            self._vertical_region(y)
        )

        return {
            "horizontal": horizontal,
            "vertical": vertical,
            "combined": (
                f"{horizontal}_{vertical}"
            ),
            "normalized_x": round(
                float(x)
                / self.frame_width,
                4,
            ),
            "normalized_y": round(
                float(y)
                / self.frame_height,
                4,
            ),
        }

    @staticmethod
    def _region_transition(
        start_region,
        end_region,
    ):

        start_combined = (
            start_region["combined"]
        )

        end_combined = (
            end_region["combined"]
        )

        if (
            start_combined
            == end_combined
        ):
            return (
                f"remained_in_"
                f"{start_combined}"
            )

        return (
            f"{start_combined}"
            f"_to_"
            f"{end_combined}"
        )

    @staticmethod
    def _horizontal_transition(
        start_region,
        end_region,
    ):

        start = start_region[
            "horizontal"
        ]

        end = end_region[
            "horizontal"
        ]

        if start == end:
            return f"remained_{start}"

        return f"{start}_to_{end}"

    @staticmethod
    def _vertical_transition(
        start_region,
        end_region,
    ):

        start = start_region[
            "vertical"
        ]

        end = end_region[
            "vertical"
        ]

        if start == end:
            return f"remained_{start}"

        return f"{start}_to_{end}"

    @staticmethod
    def _describe_spatial_motion(
        class_name,
        motion_state,
        start_region,
        end_region,
    ):

        start_text = (
            start_region["combined"]
            .replace("_", " ")
        )

        end_text = (
            end_region["combined"]
            .replace("_", " ")
        )

        if (
            start_region["combined"]
            == end_region["combined"]
        ):

            if motion_state == "stationary":

                return (
                    f"{class_name} remained "
                    f"stationary in the "
                    f"{start_text} region"
                )

            return (
                f"{class_name} remained "
                f"within the "
                f"{start_text} region"
            )

        return (
            f"{class_name} moved from "
            f"the {start_text} region "
            f"to the {end_text} region"
        )

    def analyze(self):

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        motion_data = self._load_json(
            self.motion_metadata_path
        )

        motion_tracks = (
            motion_data.get(
                "tracks",
                {},
            )
        )

        output_tracks = {}

        start_region_counts = Counter()
        end_region_counts = Counter()
        transition_counts = Counter()

        skipped_tracks = 0

        for track_key, track in (
            motion_tracks.items()
        ):

            track_id = int(
                track.get(
                    "track_id",
                    track_key,
                )
            )

            class_name = track.get(
                "class_name",
                "unknown",
            )

            motion = track.get(
                "motion",
                {},
            )

            start_position = motion.get(
                "start_position"
            )

            end_position = motion.get(
                "end_position"
            )

            if (
                not isinstance(
                    start_position,
                    dict,
                )
                or not isinstance(
                    end_position,
                    dict,
                )
            ):
                skipped_tracks += 1
                continue

            if (
                "x" not in start_position
                or "y" not in start_position
                or "x" not in end_position
                or "y" not in end_position
            ):
                skipped_tracks += 1
                continue

            start_region = (
                self._scene_region(
                    start_position["x"],
                    start_position["y"],
                )
            )

            end_region = (
                self._scene_region(
                    end_position["x"],
                    end_position["y"],
                )
            )

            transition = (
                self._region_transition(
                    start_region,
                    end_region,
                )
            )

            horizontal_transition = (
                self._horizontal_transition(
                    start_region,
                    end_region,
                )
            )

            vertical_transition = (
                self._vertical_transition(
                    start_region,
                    end_region,
                )
            )

            motion_state = motion.get(
                "motion_state",
                "unknown",
            )

            description = (
                self._describe_spatial_motion(
                    class_name,
                    motion_state,
                    start_region,
                    end_region,
                )
            )

            start_region_counts[
                start_region["combined"]
            ] += 1

            end_region_counts[
                end_region["combined"]
            ] += 1

            transition_counts[
                transition
            ] += 1

            output_tracks[
                str(track_id)
            ] = {
                "track_id": track_id,
                "class_name": class_name,
                "reliability_status": (
                    track.get(
                        "reliability_status",
                        "unknown",
                    )
                ),
                "motion_state": (
                    motion_state
                ),
                "motion_direction": (
                    motion.get(
                        "direction",
                        "unknown",
                    )
                ),
                "start_region": (
                    start_region
                ),
                "end_region": (
                    end_region
                ),
                "region_transition": (
                    transition
                ),
                "horizontal_transition": (
                    horizontal_transition
                ),
                "vertical_transition": (
                    vertical_transition
                ),
                "spatial_description": (
                    description
                ),
            }

            print(
                f"Track {track_id:3} | "
                f"{class_name:10} | "
                f"{start_region['combined']:14} "
                f"→ "
                f"{end_region['combined']:14} | "
                f"{transition}"
            )

        output_data = {
            "summary": {
                "analyzed_tracks": len(
                    output_tracks
                ),
                "skipped_tracks": (
                    skipped_tracks
                ),
                "start_region_counts": dict(
                    start_region_counts
                ),
                "end_region_counts": dict(
                    end_region_counts
                ),
                "transition_counts": dict(
                    transition_counts
                ),
                "frame_width": (
                    self.frame_width
                ),
                "frame_height": (
                    self.frame_height
                ),
            },
            "tracks": output_tracks,
        }

        output_path = (
            self.output_dir
            / "spatial_metadata.json"
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
            "analyzed_tracks": len(
                output_tracks
            ),
            "skipped_tracks": (
                skipped_tracks
            ),
            "start_region_counts": dict(
                start_region_counts
            ),
            "end_region_counts": dict(
                end_region_counts
            ),
            "transition_counts": dict(
                transition_counts
            ),
            "metadata_path": str(
                output_path
            ),
        }
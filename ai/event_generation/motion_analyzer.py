import json
import math
from collections import Counter
from pathlib import Path


class MotionAnalyzer:

    def __init__(
        self,
        track_metadata_path,
        reliability_metadata_path,
        output_dir,
        source_fps,
        frame_width,
        frame_height,
        include_review=True,
    ):
        self.track_metadata_path = Path(
            track_metadata_path
        )

        self.reliability_metadata_path = Path(
            reliability_metadata_path
        )

        self.output_dir = Path(output_dir)

        self.source_fps = float(source_fps)

        self.frame_width = float(frame_width)
        self.frame_height = float(frame_height)

        self.include_review = bool(
            include_review
        )

        for path in [
            self.track_metadata_path,
            self.reliability_metadata_path,
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required metadata not found: {path}"
                )

        if self.source_fps <= 0:
            raise ValueError(
                "source_fps must be greater than 0"
            )

        self.frame_diagonal = math.sqrt(
            self.frame_width ** 2
            + self.frame_height ** 2
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
    def _find_track(
        tracks,
        track_id,
    ):
        target_id = int(track_id)

        for key, track in tracks.items():

            current_id = track.get(
                "track_id",
                key,
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
    def _extract_bbox(observation):

        # Your actual tracking metadata format:
        # "bounding_box": {
        #     "x1": ...,
        #     "y1": ...,
        #     "x2": ...,
        #     "y2": ...
        # }

        bounding_box = observation.get(
            "bounding_box"
        )

        if isinstance(
            bounding_box,
            dict,
        ):
            required_keys = [
                "x1",
                "y1",
                "x2",
                "y2",
            ]

            if all(
                key in bounding_box
                for key in required_keys
            ):
                return [
                    float(
                        bounding_box["x1"]
                    ),
                    float(
                        bounding_box["y1"]
                    ),
                    float(
                        bounding_box["x2"]
                    ),
                    float(
                        bounding_box["y2"]
                    ),
                ]

        # Fallback for list-based metadata

        for key in [
            "bbox",
            "box",
            "xyxy",
        ]:

            value = observation.get(key)

            if (
                isinstance(value, list)
                and len(value) >= 4
            ):
                return [
                    float(value[0]),
                    float(value[1]),
                    float(value[2]),
                    float(value[3]),
                ]

        return None

    @staticmethod
    def _bbox_center(bbox):

        x1, y1, x2, y2 = bbox

        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        )

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
            {},
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

    def _build_trajectory(self, track):

        observations = (
            self._extract_observations(track)
        )

        trajectory = []

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

            bbox = self._extract_bbox(
                observation
            )

            if (
                frame_index is None
                or bbox is None
            ):
                continue

            center_x, center_y = (
                self._bbox_center(bbox)
            )

            trajectory.append(
                {
                    "frame_index": frame_index,
                    "timestamp": (
                        frame_index
                        / self.source_fps
                    ),
                    "center_x": center_x,
                    "center_y": center_y,
                    "bbox": bbox,
                }
            )

        trajectory.sort(
            key=lambda item: (
                item["frame_index"]
            )
        )

        return trajectory

    @staticmethod
    def _distance(point_a, point_b):

        dx = (
            point_b["center_x"]
            - point_a["center_x"]
        )

        dy = (
            point_b["center_y"]
            - point_a["center_y"]
        )

        return math.sqrt(
            dx ** 2 + dy ** 2
        )

    def _classify_direction(
        self,
        dx,
        dy,
        displacement,
    ):
        normalized_displacement = (
            displacement
            / max(
                self.frame_diagonal,
                1.0,
            )
        )

        if normalized_displacement < 0.015:
            return "stationary"

        angle = math.degrees(
            math.atan2(-dy, dx)
        )

        if -22.5 <= angle < 22.5:
            return "left_to_right"

        if 22.5 <= angle < 67.5:
            return "bottom_left_to_top_right"

        if 67.5 <= angle < 112.5:
            return "bottom_to_top"

        if 112.5 <= angle < 157.5:
            return "bottom_right_to_top_left"

        if (
            angle >= 157.5
            or angle < -157.5
        ):
            return "right_to_left"

        if -157.5 <= angle < -112.5:
            return "top_right_to_bottom_left"

        if -112.5 <= angle < -67.5:
            return "top_to_bottom"

        return "top_left_to_bottom_right"

    def _motion_state(
        self,
        normalized_displacement,
        normalized_path_distance,
    ):
        if (
            normalized_displacement < 0.015
            and normalized_path_distance < 0.025
        ):
            return "stationary"

        if normalized_path_distance < 0.060:
            return "slow_moving"

        return "moving"

    @staticmethod
    def _motion_consistency(
        trajectory,
    ):
        if len(trajectory) < 3:
            return 0.0

        direction_vectors = []

        for index in range(
            1,
            len(trajectory),
        ):
            dx = (
                trajectory[index]["center_x"]
                - trajectory[index - 1]["center_x"]
            )

            dy = (
                trajectory[index]["center_y"]
                - trajectory[index - 1]["center_y"]
            )

            magnitude = math.sqrt(
                dx ** 2 + dy ** 2
            )

            if magnitude <= 1e-8:
                continue

            direction_vectors.append(
                (
                    dx / magnitude,
                    dy / magnitude,
                )
            )

        if not direction_vectors:
            return 0.0

        mean_x = sum(
            vector[0]
            for vector in direction_vectors
        ) / len(direction_vectors)

        mean_y = sum(
            vector[1]
            for vector in direction_vectors
        ) / len(direction_vectors)

        consistency = math.sqrt(
            mean_x ** 2
            + mean_y ** 2
        )

        return max(
            0.0,
            min(1.0, consistency),
        )

    def _analyze_trajectory(
        self,
        trajectory,
    ):
        if len(trajectory) < 2:
            return {
                "motion_state": "insufficient_data",
                "direction": "unknown",
                "trajectory_points": len(
                    trajectory
                ),
            }

        start = trajectory[0]
        end = trajectory[-1]

        dx = (
            end["center_x"]
            - start["center_x"]
        )

        dy = (
            end["center_y"]
            - start["center_y"]
        )

        net_displacement = math.sqrt(
            dx ** 2 + dy ** 2
        )

        path_distance = 0.0

        segment_speeds = []

        for index in range(
            1,
            len(trajectory),
        ):
            previous = trajectory[index - 1]
            current = trajectory[index]

            distance = self._distance(
                previous,
                current,
            )

            frame_gap = (
                current["frame_index"]
                - previous["frame_index"]
            )

            if frame_gap <= 0:
                continue

            time_gap = (
                frame_gap
                / self.source_fps
            )

            speed = (
                distance
                / max(time_gap, 1e-8)
            )

            path_distance += distance

            segment_speeds.append(
                speed
            )

        duration = max(
            end["timestamp"]
            - start["timestamp"],
            0.0,
        )

        average_speed = (
            path_distance
            / duration
            if duration > 0
            else 0.0
        )

        maximum_speed = (
            max(segment_speeds)
            if segment_speeds
            else 0.0
        )

        normalized_displacement = (
            net_displacement
            / max(
                self.frame_diagonal,
                1.0,
            )
        )

        normalized_path_distance = (
            path_distance
            / max(
                self.frame_diagonal,
                1.0,
            )
        )

        normalized_average_speed = (
            average_speed
            / max(
                self.frame_diagonal,
                1.0,
            )
        )

        direction = (
            self._classify_direction(
                dx,
                dy,
                net_displacement,
            )
        )

        motion_state = (
            self._motion_state(
                normalized_displacement,
                normalized_path_distance,
            )
        )

        if direction == "stationary":
            motion_state = "stationary"

        if motion_state == "stationary":
            direction = "stationary"

        consistency = (
            self._motion_consistency(
                trajectory
            )
        )

        return {
            "motion_state": motion_state,
            "direction": direction,
            "trajectory_points": len(
                trajectory
            ),
            "start_frame": start[
                "frame_index"
            ],
            "end_frame": end[
                "frame_index"
            ],
            "start_time": round(
                start["timestamp"],
                4,
            ),
            "end_time": round(
                end["timestamp"],
                4,
            ),
            "duration": round(
                duration,
                4,
            ),
            "start_position": {
                "x": round(
                    start["center_x"],
                    2,
                ),
                "y": round(
                    start["center_y"],
                    2,
                ),
            },
            "end_position": {
                "x": round(
                    end["center_x"],
                    2,
                ),
                "y": round(
                    end["center_y"],
                    2,
                ),
            },
            "horizontal_displacement": round(
                dx,
                2,
            ),
            "vertical_displacement": round(
                dy,
                2,
            ),
            "net_displacement_pixels": round(
                net_displacement,
                2,
            ),
            "path_distance_pixels": round(
                path_distance,
                2,
            ),
            "normalized_displacement": round(
                normalized_displacement,
                6,
            ),
            "normalized_path_distance": round(
                normalized_path_distance,
                6,
            ),
            "average_speed_pixels_per_second": round(
                average_speed,
                2,
            ),
            "maximum_speed_pixels_per_second": round(
                maximum_speed,
                2,
            ),
            "normalized_average_speed": round(
                normalized_average_speed,
                6,
            ),
            "motion_consistency": round(
                consistency,
                4,
            ),
        }

    def analyze(self):

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

        output_tracks = {}

        motion_state_counts = Counter()
        direction_counts = Counter()

        skipped_reject = 0
        skipped_insufficient = 0

        for track_key, track in (
            tracking_tracks.items()
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

            reliability = (
                self._get_reliability(
                    reliability_tracks,
                    track_id,
                )
            )

            status = reliability["status"]

            if status == "reject":
                skipped_reject += 1
                continue

            if (
                status == "review"
                and not self.include_review
            ):
                skipped_reject += 1
                continue

            trajectory = (
                self._build_trajectory(
                    track
                )
            )

            motion = (
                self._analyze_trajectory(
                    trajectory
                )
            )

            if (
                motion["motion_state"]
                == "insufficient_data"
            ):
                skipped_insufficient += 1

            motion_state_counts[
                motion["motion_state"]
            ] += 1

            direction_counts[
                motion["direction"]
            ] += 1

            output_tracks[str(track_id)] = {
                "track_id": track_id,
                "class_name": class_name,
                "reliability_status": status,
                "reliability_score": round(
                    reliability["score"],
                    4,
                ),
                "motion": motion,
                "trajectory": trajectory,
            }

            print(
                f"Track {track_id:3} | "
                f"{class_name:10} | "
                f"{status.upper():8} | "
                f"{motion['motion_state']:17} | "
                f"{motion['direction']}"
            )

        output_data = {
            "summary": {
                "analyzed_tracks": len(
                    output_tracks
                ),
                "skipped_reject_tracks": (
                    skipped_reject
                ),
                "skipped_insufficient_tracks": (
                    skipped_insufficient
                ),
                "motion_state_counts": dict(
                    motion_state_counts
                ),
                "direction_counts": dict(
                    direction_counts
                ),
                "source_fps": self.source_fps,
                "frame_width": self.frame_width,
                "frame_height": self.frame_height,
            },
            "tracks": output_tracks,
        }

        output_path = (
            self.output_dir
            / "motion_metadata.json"
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
            "skipped_reject_tracks": (
                skipped_reject
            ),
            "skipped_insufficient_tracks": (
                skipped_insufficient
            ),
            "motion_state_counts": dict(
                motion_state_counts
            ),
            "direction_counts": dict(
                direction_counts
            ),
            "metadata_path": str(
                output_path
            ),
        }
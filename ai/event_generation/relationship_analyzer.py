import json
import math
from collections import Counter, defaultdict
from pathlib import Path


class RelationshipAnalyzer:

    HUMAN_CLASSES = {
        "pedestrian",
        "people",
    }

    VEHICLE_CLASSES = {
        "car",
        "van",
        "truck",
        "bus",
        "bicycle",
    }

    def __init__(
        self,
        track_metadata_path,
        reliability_metadata_path,
        motion_metadata_path,
        output_dir,
        frame_width,
        frame_height,
        include_review=True,
        near_threshold=0.08,
        moving_together_threshold=0.06,
        minimum_shared_frames=3,
    ):
        self.track_metadata_path = Path(
            track_metadata_path
        )

        self.reliability_metadata_path = Path(
            reliability_metadata_path
        )

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

        self.include_review = bool(
            include_review
        )

        self.near_threshold = float(
            near_threshold
        )

        self.moving_together_threshold = float(
            moving_together_threshold
        )

        self.minimum_shared_frames = int(
            minimum_shared_frames
        )

        self.frame_diagonal = math.sqrt(
            self.frame_width ** 2
            + self.frame_height ** 2
        )

        for path in [
            self.track_metadata_path,
            self.reliability_metadata_path,
            self.motion_metadata_path,
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required metadata not found: {path}"
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
    def _extract_bbox(observation):

        bounding_box = observation.get(
            "bounding_box"
        )

        if isinstance(
            bounding_box,
            dict,
        ):
            required = [
                "x1",
                "y1",
                "x2",
                "y2",
            ]

            if all(
                key in bounding_box
                for key in required
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

        return None

    @staticmethod
    def _bbox_center(bbox):

        x1, y1, x2, y2 = bbox

        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        )

    @staticmethod
    def _iou(
        box_a,
        box_b,
    ):
        x1 = max(
            box_a[0],
            box_b[0],
        )

        y1 = max(
            box_a[1],
            box_b[1],
        )

        x2 = min(
            box_a[2],
            box_b[2],
        )

        y2 = min(
            box_a[3],
            box_b[3],
        )

        intersection_width = max(
            0.0,
            x2 - x1,
        )

        intersection_height = max(
            0.0,
            y2 - y1,
        )

        intersection = (
            intersection_width
            * intersection_height
        )

        area_a = max(
            0.0,
            box_a[2] - box_a[0],
        ) * max(
            0.0,
            box_a[3] - box_a[1],
        )

        area_b = max(
            0.0,
            box_b[2] - box_b[0],
        ) * max(
            0.0,
            box_b[3] - box_b[1],
        )

        union = (
            area_a
            + area_b
            - intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union

    def _normalized_distance(
        self,
        box_a,
        box_b,
    ):
        center_a = self._bbox_center(
            box_a
        )

        center_b = self._bbox_center(
            box_b
        )

        distance = math.sqrt(
            (
                center_b[0]
                - center_a[0]
            ) ** 2
            + (
                center_b[1]
                - center_a[1]
            ) ** 2
        )

        return (
            distance
            / self.frame_diagonal
        )

    def _get_reliability_status(
        self,
        reliability_tracks,
        track_id,
    ):
        track = self._find_track(
            reliability_tracks,
            track_id,
        )

        if track is None:
            return "reject"

        reliability = track.get(
            "reliability",
            {}
        )

        return reliability.get(
            "status",
            "reject",
        )

    def _usable_track(
        self,
        status,
    ):
        if status == "reliable":
            return True

        if (
            status == "review"
            and self.include_review
        ):
            return True

        return False

    @staticmethod
    def _same_motion_direction(
        motion_a,
        motion_b,
    ):
        direction_a = motion_a.get(
            "direction",
            "unknown",
        )

        direction_b = motion_b.get(
            "direction",
            "unknown",
        )

        invalid = {
            "unknown",
            "stationary",
        }

        return (
            direction_a == direction_b
            and direction_a not in invalid
        )

    @staticmethod
    def _relationship_id(
        number,
    ):
        return (
            f"relationship_"
            f"{number:06d}"
        )

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

        motion_data = self._load_json(
            self.motion_metadata_path
        )

        tracking_tracks = tracking_data.get(
            "tracks",
            tracking_data,
        )

        reliability_tracks = (
            reliability_data.get(
                "tracks",
                {},
            )
        )

        motion_tracks = motion_data.get(
            "tracks",
            {},
        )

        usable_tracks = {}

        for track_key, track in (
            tracking_tracks.items()
        ):
            track_id = int(
                track.get(
                    "track_id",
                    track_key,
                )
            )

            status = (
                self._get_reliability_status(
                    reliability_tracks,
                    track_id,
                )
            )

            if not self._usable_track(
                status
            ):
                continue

            usable_tracks[track_id] = {
                "track": track,
                "status": status,
            }

        frame_objects = defaultdict(
            list
        )

        for track_id, data in (
            usable_tracks.items()
        ):
            track = data["track"]

            class_name = track.get(
                "class_name",
                "unknown",
            )

            observations = track.get(
                "observations",
                [],
            )

            for observation in observations:

                frame_index = observation.get(
                    "frame_index"
                )

                bbox = self._extract_bbox(
                    observation
                )

                if (
                    frame_index is None
                    or bbox is None
                ):
                    continue

                frame_objects[
                    int(frame_index)
                ].append(
                    {
                        "track_id": track_id,
                        "class_name": (
                            class_name
                        ),
                        "bbox": bbox,
                        "timestamp": (
                            observation.get(
                                "timestamp_seconds"
                            )
                        ),
                    }
                )

        pair_statistics = defaultdict(
            lambda: {
                "shared_frames": 0,
                "near_frames": 0,
                "overlap_frames": 0,
                "distance_sum": 0.0,
                "minimum_distance": 1.0,
                "maximum_iou": 0.0,
                "first_frame": None,
                "last_frame": None,
                "first_timestamp": None,
                "last_timestamp": None,
            }
        )

        for frame_index, objects in (
            frame_objects.items()
        ):
            for first_index in range(
                len(objects)
            ):
                for second_index in range(
                    first_index + 1,
                    len(objects),
                ):
                    object_a = objects[
                        first_index
                    ]

                    object_b = objects[
                        second_index
                    ]

                    pair_key = tuple(
                        sorted(
                            [
                                object_a[
                                    "track_id"
                                ],
                                object_b[
                                    "track_id"
                                ],
                            ]
                        )
                    )

                    distance = (
                        self._normalized_distance(
                            object_a["bbox"],
                            object_b["bbox"],
                        )
                    )

                    iou = self._iou(
                        object_a["bbox"],
                        object_b["bbox"],
                    )

                    stats = pair_statistics[
                        pair_key
                    ]

                    stats[
                        "shared_frames"
                    ] += 1

                    stats[
                        "distance_sum"
                    ] += distance

                    stats[
                        "minimum_distance"
                    ] = min(
                        stats[
                            "minimum_distance"
                        ],
                        distance,
                    )

                    stats[
                        "maximum_iou"
                    ] = max(
                        stats["maximum_iou"],
                        iou,
                    )

                    if (
                        distance
                        <= self.near_threshold
                    ):
                        stats[
                            "near_frames"
                        ] += 1

                    if iou > 0.0:
                        stats[
                            "overlap_frames"
                        ] += 1

                    if (
                        stats["first_frame"]
                        is None
                    ):
                        stats[
                            "first_frame"
                        ] = frame_index

                        stats[
                            "first_timestamp"
                        ] = object_a.get(
                            "timestamp"
                        )

                    stats[
                        "last_frame"
                    ] = frame_index

                    stats[
                        "last_timestamp"
                    ] = object_a.get(
                        "timestamp"
                    )

        relationships = []

        relationship_counts = Counter()

        relationship_number = 1

        for pair_key, stats in (
            pair_statistics.items()
        ):
            track_a_id = pair_key[0]
            track_b_id = pair_key[1]

            if (
                stats["shared_frames"]
                < self.minimum_shared_frames
            ):
                continue

            track_a = usable_tracks[
                track_a_id
            ]["track"]

            track_b = usable_tracks[
                track_b_id
            ]["track"]

            class_a = track_a.get(
                "class_name",
                "unknown",
            )

            class_b = track_b.get(
                "class_name",
                "unknown",
            )

            average_distance = (
                stats["distance_sum"]
                / stats["shared_frames"]
            )

            near_ratio = (
                stats["near_frames"]
                / stats["shared_frames"]
            )

            overlap_ratio = (
                stats["overlap_frames"]
                / stats["shared_frames"]
            )

            detected_relationships = []

            if near_ratio >= 0.50:
                detected_relationships.append(
                    "near"
                )

            if overlap_ratio >= 0.30:
                detected_relationships.append(
                    "overlapping"
                )

            human_vehicle = (
                (
                    class_a
                    in self.HUMAN_CLASSES
                    and class_b
                    in self.VEHICLE_CLASSES
                )
                or
                (
                    class_b
                    in self.HUMAN_CLASSES
                    and class_a
                    in self.VEHICLE_CLASSES
                )
            )

            if (
                human_vehicle
                and near_ratio >= 0.50
            ):
                detected_relationships.append(
                    "human_near_vehicle"
                )

            motion_a_track = (
                self._find_track(
                    motion_tracks,
                    track_a_id,
                )
            )

            motion_b_track = (
                self._find_track(
                    motion_tracks,
                    track_b_id,
                )
            )

            if (
                motion_a_track is not None
                and motion_b_track is not None
            ):
                motion_a = (
                    motion_a_track.get(
                        "motion",
                        {},
                    )
                )

                motion_b = (
                    motion_b_track.get(
                        "motion",
                        {},
                    )
                )

                same_direction = (
                    self._same_motion_direction(
                        motion_a,
                        motion_b,
                    )
                )

                if (
                    same_direction
                    and average_distance
                    <= self.moving_together_threshold
                    and stats[
                        "shared_frames"
                    ] >= 5
                ):
                    detected_relationships.append(
                        "moving_together"
                    )

            if not detected_relationships:
                continue

            for relationship_type in (
                detected_relationships
            ):
                relationship_counts[
                    relationship_type
                ] += 1

            relationship = {
                "relationship_id": (
                    self._relationship_id(
                        relationship_number
                    )
                ),
                "track_a_id": track_a_id,
                "track_a_class": class_a,
                "track_b_id": track_b_id,
                "track_b_class": class_b,
                "relationship_types": (
                    detected_relationships
                ),
                "shared_frames": (
                    stats["shared_frames"]
                ),
                "near_frames": (
                    stats["near_frames"]
                ),
                "overlap_frames": (
                    stats["overlap_frames"]
                ),
                "near_ratio": round(
                    near_ratio,
                    4,
                ),
                "overlap_ratio": round(
                    overlap_ratio,
                    4,
                ),
                "average_normalized_distance": (
                    round(
                        average_distance,
                        6,
                    )
                ),
                "minimum_normalized_distance": (
                    round(
                        stats[
                            "minimum_distance"
                        ],
                        6,
                    )
                ),
                "maximum_iou": round(
                    stats["maximum_iou"],
                    4,
                ),
                "first_frame": (
                    stats["first_frame"]
                ),
                "last_frame": (
                    stats["last_frame"]
                ),
                "first_timestamp": (
                    stats[
                        "first_timestamp"
                    ]
                ),
                "last_timestamp": (
                    stats[
                        "last_timestamp"
                    ]
                ),
            }

            relationships.append(
                relationship
            )

            print(
                f"Tracks "
                f"{track_a_id:3} "
                f"({class_a:10}) + "
                f"{track_b_id:3} "
                f"({class_b:10}) | "
                f"Shared: "
                f"{stats['shared_frames']:3} | "
                f"Near: "
                f"{near_ratio:.2f} | "
                f"{', '.join(detected_relationships)}"
            )

            relationship_number += 1

        output_data = {
            "summary": {
                "usable_tracks": len(
                    usable_tracks
                ),
                "frames_with_objects": len(
                    frame_objects
                ),
                "analyzed_pairs": len(
                    pair_statistics
                ),
                "detected_relationships": (
                    len(relationships)
                ),
                "relationship_type_counts": (
                    dict(
                        relationship_counts
                    )
                ),
                "near_threshold": (
                    self.near_threshold
                ),
                "minimum_shared_frames": (
                    self.minimum_shared_frames
                ),
            },
            "relationships": relationships,
        }

        output_path = (
            self.output_dir
            / "relationship_metadata.json"
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
            "usable_tracks": len(
                usable_tracks
            ),
            "frames_with_objects": len(
                frame_objects
            ),
            "analyzed_pairs": len(
                pair_statistics
            ),
            "detected_relationships": len(
                relationships
            ),
            "relationship_type_counts": dict(
                relationship_counts
            ),
            "metadata_path": str(
                output_path
            ),
        }
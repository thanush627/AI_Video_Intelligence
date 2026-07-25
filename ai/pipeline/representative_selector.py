import cv2
import json

from pathlib import Path
from collections import defaultdict


class RepresentativeCropSelector:
    def __init__(
        self,
        video_path,
        track_metadata_path,
        output_dir,
        max_crops_per_track=3,
        min_crop_width=40,
        min_crop_height=40,
        weights=None,
    ):
        self.video_path = Path(video_path)
        self.track_metadata_path = Path(track_metadata_path)
        self.output_dir = Path(output_dir)
        self.crops_dir = self.output_dir / "crops"

        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {self.video_path}"
            )

        if not self.track_metadata_path.exists():
            raise FileNotFoundError(
                f"Track metadata not found: "
                f"{self.track_metadata_path}"
            )

        self.max_crops_per_track = max_crops_per_track
        self.min_crop_width = min_crop_width
        self.min_crop_height = min_crop_height

        self.weights = weights or {
            "confidence": 0.40,
            "sharpness": 0.25,
            "size": 0.20,
            "center": 0.15,
        }

    @staticmethod
    def _sharpness_score(crop):
        if crop.size == 0:
            return 0.0

        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY,
        )

        variance = cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()

        # Normalize to 0–1.
        return min(float(variance) / 1000.0, 1.0)

    @staticmethod
    def _size_score(
        crop_width,
        crop_height,
        frame_width,
        frame_height,
    ):
        frame_area = max(
            frame_width * frame_height,
            1,
        )

        crop_area = crop_width * crop_height

        # Objects covering 20% or more of the frame
        # receive the maximum size score.
        return min(
            crop_area / (frame_area * 0.20),
            1.0,
        )

    @staticmethod
    def _center_score(
        x1,
        y1,
        x2,
        y2,
        frame_width,
        frame_height,
    ):
        object_center_x = (x1 + x2) / 2.0
        object_center_y = (y1 + y2) / 2.0

        frame_center_x = frame_width / 2.0
        frame_center_y = frame_height / 2.0

        dx = (
            object_center_x - frame_center_x
        ) / max(frame_center_x, 1)

        dy = (
            object_center_y - frame_center_y
        ) / max(frame_center_y, 1)

        normalized_distance = min(
            (dx * dx + dy * dy) ** 0.5,
            1.0,
        )

        return 1.0 - normalized_distance

    def _calculate_final_score(
        self,
        confidence,
        sharpness,
        size,
        center,
    ):
        return (
            self.weights["confidence"] * confidence
            + self.weights["sharpness"] * sharpness
            + self.weights["size"] * size
            + self.weights["center"] * center
        )
    
    def _build_frame_lookup(
        self,
        track_data,
        frame_width,
        frame_height,
    ):

        frame_lookup = defaultdict(list)

        for track_key, track in track_data["tracks"].items():

            for observation in track["observations"]:

                box = observation["bounding_box"]

                x1 = max(0, int(box["x1"]))
                y1 = max(0, int(box["y1"]))
                x2 = min(frame_width, int(box["x2"]))
                y2 = min(frame_height, int(box["y2"]))

                crop_width = x2 - x1
                crop_height = y2 - y1

                if (
                    crop_width < self.min_crop_width
                    or crop_height < self.min_crop_height
                ):
                    continue

                frame_lookup[
                    int(observation["frame_index"])
                ].append(
                    {
                        "track_key": track_key,
                        "track": track,
                        "observation": observation,
                        "bbox": (
                            x1,
                            y1,
                            x2,
                            y2,
                        ),
                    }
                )

        return frame_lookup

    def select(self):
        self.crops_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            self.track_metadata_path,
            "r",
            encoding="utf-8",
        ) as file:
            track_data = json.load(file)

        cap = cv2.VideoCapture(
            str(self.video_path)
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video: {self.video_path}"
            )

        frame_width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        frame_height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        ##########################################################
        # BUILD FRAME LOOKUP
        ##########################################################

        frame_lookup = self._build_frame_lookup(
            track_data,
            frame_width,
            frame_height,
        )
        frame_cache = {}
        selected_tracks = {}
        total_saved_crops = 0

        for track_key, track in track_data["tracks"].items():
            candidates = []

            for observation in track[
                "observations"
            ]:
                frame_index = int(
                    observation["frame_index"]
                )

                box = observation["bounding_box"]

                x1 = max(0, int(box["x1"]))
                y1 = max(0, int(box["y1"]))
                x2 = min(
                    frame_width,
                    int(box["x2"]),
                )
                y2 = min(
                    frame_height,
                    int(box["y2"]),
                )

                crop_width = x2 - x1
                crop_height = y2 - y1

                if (
                    crop_width < self.min_crop_width
                    or crop_height < self.min_crop_height
                ):
                    continue

                ##########################################################
                # FRAME CACHE
                ##########################################################

                if frame_index not in frame_cache:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index,)
                    success, frame = cap.read()
                    if not success:
                        continue

                    frame_cache[frame_index] = frame

                frame = frame_cache[frame_index]


                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                confidence = float(
                    observation["confidence"]
                )

                sharpness = self._sharpness_score(
                    crop
                )

                size = self._size_score(
                    crop_width,
                    crop_height,
                    frame_width,
                    frame_height,
                )

                center = self._center_score(
                    x1,
                    y1,
                    x2,
                    y2,
                    frame_width,
                    frame_height,
                )

                final_score = (
                    self._calculate_final_score(
                        confidence,
                        sharpness,
                        size,
                        center,
                    )
                )

                candidates.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": float(
                            observation[
                                "timestamp_seconds"
                            ]
                        ),
                        "confidence": confidence,
                        "sharpness_score": sharpness,
                        "size_score": size,
                        "center_score": center,
                        "final_score": final_score,
                        "bounding_box": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        },
                        "crop": crop.copy(),
                    }
                )

            candidates.sort(
                key=lambda item: item["final_score"],
                reverse=True,
            )

            best_candidates = candidates[
                : self.max_crops_per_track
            ]

            if not best_candidates:
                continue

            track_id = track["track_id"]
            class_name = track["class_name"]

            track_folder = (
                self.crops_dir
                / f"{class_name}_track_{track_id}"
            )

            track_folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            saved_crops = []

            for rank, candidate in enumerate(
                best_candidates,
                start=1,
            ):
                filename = (
                    f"rank_{rank}"
                    f"_frame_{candidate['frame_index']:06d}"
                    f".jpg"
                )

                crop_path = track_folder / filename

                saved = cv2.imwrite(
                    str(crop_path),
                    candidate["crop"],
                )

                if not saved:
                    continue

                total_saved_crops += 1

                saved_crops.append(
                    {
                        "rank": rank,
                        "frame_index": candidate[
                            "frame_index"
                        ],
                        "timestamp_seconds": candidate[
                            "timestamp_seconds"
                        ],
                        "confidence": candidate[
                            "confidence"
                        ],
                        "sharpness_score": candidate[
                            "sharpness_score"
                        ],
                        "size_score": candidate[
                            "size_score"
                        ],
                        "center_score": candidate[
                            "center_score"
                        ],
                        "final_score": candidate[
                            "final_score"
                        ],
                        "bounding_box": candidate[
                            "bounding_box"
                        ],
                        "crop_path": str(crop_path),
                    }
                )

            if saved_crops:
                selected_tracks[track_key] = {
                    "track_id": track_id,
                    "class_id": track["class_id"],
                    "class_name": class_name,
                    "observation_count": track[
                        "observation_count"
                    ],
                    "selected_crop_count": len(
                        saved_crops
                    ),
                    "selected_crops": saved_crops,
                }

            print(
                f"Track {track_id:3} | "
                f"{class_name:10} | "
                f"Observations: "
                f"{track['observation_count']:3} | "
                f"Saved: {len(saved_crops)}"
            )

        cap.release()

        metadata = {
            "summary": {
                "input_tracks": len(
                    track_data["tracks"]
                ),
                "tracks_with_crops": len(
                    selected_tracks
                ),
                "total_saved_crops": (
                    total_saved_crops
                ),
            },
            "tracks": selected_tracks,
        }

        metadata_path = (
            self.output_dir
            / "representative_crops.json"
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=2,
            )

        return {
            "input_tracks": len(
                track_data["tracks"]
            ),
            "tracks_with_crops": len(
                selected_tracks
            ),
            "total_saved_crops": total_saved_crops,
            "crops_directory": str(
                self.crops_dir
            ),
            "metadata_path": str(
                metadata_path
            ),
        }
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


class ObjectTracker:
    def __init__(
        self,
        model_path,
        confidence_threshold=0.35,
        iou_threshold=0.50,
        image_size=640,
        tracker="bytetrack.yaml",
        device="auto",
    ):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.tracker = tracker
        self.device = None if device == "auto" else device

        self.model = YOLO(str(self.model_path))

    def track(self, video_path, output_dir):
        video_path = Path(video_path)
        output_dir = Path(output_dir)

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video: {video_path}"
            )

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        cap.release()

        results = self.model.track(
            source=str(video_path),
            tracker=self.tracker,
            persist=True,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            stream=True,
            verbose=False,
        )

        tracks = {}
        processed_frames = 0
        frames_with_tracks = 0

        for frame_index, result in enumerate(results):
            processed_frames += 1

            if (
                result.boxes is None
                or result.boxes.id is None
            ):
                continue

            frames_with_tracks += 1

            track_ids = (
                result.boxes.id
                .int()
                .cpu()
                .tolist()
            )

            class_ids = (
                result.boxes.cls
                .int()
                .cpu()
                .tolist()
            )

            confidences = (
                result.boxes.conf
                .cpu()
                .tolist()
            )

            boxes = (
                result.boxes.xyxy
                .cpu()
                .tolist()
            )

            timestamp_seconds = (
                frame_index / source_fps
                if source_fps > 0
                else 0.0
            )

            for (
                track_id,
                class_id,
                confidence,
                box,
            ) in zip(
                track_ids,
                class_ids,
                confidences,
                boxes,
            ):
                class_name = self.model.names[
                    class_id
                ]

                x1, y1, x2, y2 = box

                observation = {
                    "frame_index": frame_index,
                    "timestamp_seconds": float(
                        timestamp_seconds
                    ),
                    "confidence": float(confidence),
                    "bounding_box": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    },
                }

                track_key = str(track_id)

                if track_key not in tracks:
                    tracks[track_key] = {
                        "track_id": track_id,
                        "class_id": class_id,
                        "class_name": class_name,
                        "first_seen_seconds": float(
                            timestamp_seconds
                        ),
                        "last_seen_seconds": float(
                            timestamp_seconds
                        ),
                        "observation_count": 0,
                        "observations": [],
                    }

                tracks[track_key][
                    "last_seen_seconds"
                ] = float(timestamp_seconds)

                tracks[track_key][
                    "observation_count"
                ] += 1

                tracks[track_key][
                    "observations"
                ].append(observation)

            if processed_frames % 50 == 0:
                print(
                    f"Processed "
                    f"{processed_frames}/{total_frames} "
                    f"frames | "
                    f"Tracks: {len(tracks)}"
                )

        for track in tracks.values():
            track["duration_seconds"] = float(
                track["last_seen_seconds"]
                - track["first_seen_seconds"]
            )

        metadata = {
            "video": {
                "video_path": str(video_path),
                "source_fps": float(source_fps),
                "total_frames": total_frames,
                "processed_frames": processed_frames,
                "frames_with_tracks": frames_with_tracks,
            },
            "summary": {
                "unique_tracks": len(tracks),
            },
            "tracks": tracks,
        }

        metadata_path = (
            output_dir / "track_metadata.json"
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
            "processed_frames": processed_frames,
            "frames_with_tracks": frames_with_tracks,
            "unique_tracks": len(tracks),
            "metadata_path": str(metadata_path),
        }
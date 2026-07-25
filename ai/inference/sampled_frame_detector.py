import json
from pathlib import Path

from ultralytics import YOLO


class SampledFrameDetector:
    def __init__(
        self,
        model_path,
        confidence_threshold=0.35,
        iou_threshold=0.50,
        image_size=640,
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

        # Ultralytics does not use "auto" as a direct device value.
        # None lets it automatically select GPU if available,
        # otherwise CPU.
        self.device = None if device == "auto" else device

        self.model = YOLO(str(self.model_path))

    def detect(
        self,
        frame_metadata_path,
        output_dir,
    ):
        frame_metadata_path = Path(frame_metadata_path)
        output_dir = Path(output_dir)

        if not frame_metadata_path.exists():
            raise FileNotFoundError(
                f"Frame metadata not found: "
                f"{frame_metadata_path}"
            )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            frame_metadata_path,
            "r",
            encoding="utf-8",
        ) as file:
            frame_metadata = json.load(file)

        all_results = []
        total_detections = 0
        detected_classes = {}

        for frame_info in frame_metadata:
            image_path = Path(frame_info["image_path"])

            if not image_path.exists():
                print(
                    f"Warning: Missing frame: {image_path}"
                )
                continue

            results = self.model.predict(
                source=str(image_path),
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )

            result = results[0]
            detections = []

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(
                        box.cls.item()
                    )

                    class_name = self.model.names[
                        class_id
                    ]

                    confidence = float(
                        box.conf.item()
                    )

                    x1, y1, x2, y2 = (
                        box.xyxy[0]
                        .cpu()
                        .tolist()
                    )

                    detection = {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bounding_box": {
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2),
                        },
                    }

                    detections.append(detection)
                    total_detections += 1

                    detected_classes[class_name] = (
                        detected_classes.get(
                            class_name,
                            0,
                        )
                        + 1
                    )

            frame_result = {
                "sample_number": frame_info[
                    "sample_number"
                ],
                "frame_index": frame_info[
                    "frame_index"
                ],
                "timestamp_seconds": frame_info[
                    "timestamp_seconds"
                ],
                "motion_level": frame_info[
                    "motion_level"
                ],
                "image_path": str(image_path),
                "detection_count": len(detections),
                "detections": detections,
            }

            all_results.append(frame_result)

            print(
                f"Frame {frame_info['frame_index']:4} | "
                f"Detections: {len(detections)}"
            )

        output_path = (
            output_dir
            / "detection_metadata.json"
        )

        output_data = {
            "summary": {
                "processed_frames": len(all_results),
                "total_detections": total_detections,
                "class_counts": detected_classes,
            },
            "frames": all_results,
        }

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
            "processed_frames": len(all_results),
            "total_detections": total_detections,
            "class_counts": detected_classes,
            "metadata_path": str(output_path),
        }
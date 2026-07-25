import cv2
import json
from pathlib import Path


class FrameExtractor:
    def __init__(self, video_path, output_dir):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / "frames"

        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {self.video_path}"
            )

    def extract(self, sampled_frames):
        self.frames_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        cap = cv2.VideoCapture(str(self.video_path))

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video: {self.video_path}"
            )

        saved_metadata = []

        for sample_number, sample in enumerate(
            sampled_frames,
            start=1,
        ):
            frame_index = int(sample["frame_index"])

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_index,
            )

            success, frame = cap.read()

            if not success:
                print(
                    f"Warning: Could not read frame "
                    f"{frame_index}"
                )
                continue

            filename = (
                f"frame_{sample_number:06d}"
                f"_src_{frame_index:06d}.jpg"
            )

            output_path = self.frames_dir / filename

            saved = cv2.imwrite(
                str(output_path),
                frame,
            )

            if not saved:
                print(
                    f"Warning: Could not save "
                    f"{output_path}"
                )
                continue

            metadata = {
                "sample_number": sample_number,
                "frame_index": frame_index,
                "timestamp_seconds": float(
                    sample["timestamp_seconds"]
                ),
                "motion_score": float(
                    sample["motion_score"]
                ),
                "motion_level": sample["motion_level"],
                "sampling_fps": float(
                    sample["sampling_fps"]
                ),
                "image_path": str(output_path),
            }

            saved_metadata.append(metadata)

        cap.release()

        metadata_path = (
            self.output_dir
            / "frame_metadata.json"
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                saved_metadata,
                file,
                indent=2,
            )

        return {
            "saved_frame_count": len(saved_metadata),
            "frames_directory": str(self.frames_dir),
            "metadata_path": str(metadata_path),
            "frames": saved_metadata,
        }
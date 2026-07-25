import cv2
from pathlib import Path


class AdaptiveFrameSampler:
    def __init__(
        self,
        video_path,
        low_threshold=2.0,
        high_threshold=12.0,
        low_motion_fps=1.0,
        medium_motion_fps=2.0,
        high_motion_fps=5.0,
    ):
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {self.video_path}"
            )

        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        self.low_motion_fps = low_motion_fps
        self.medium_motion_fps = medium_motion_fps
        self.high_motion_fps = high_motion_fps

    def _calculate_motion(self, previous_gray, current_gray):
        difference = cv2.absdiff(previous_gray, current_gray)
        return float(difference.mean())

    def _select_sampling_fps(self, motion_score):
        if motion_score < self.low_threshold:
            return self.low_motion_fps, "low"

        if motion_score < self.high_threshold:
            return self.medium_motion_fps, "medium"

        return self.high_motion_fps, "high"

    def sample(self):
        cap = cv2.VideoCapture(str(self.video_path))

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video: {self.video_path}"
            )

        source_fps = cap.get(cv2.CAP_PROP_FPS)

        if source_fps <= 0:
            cap.release()
            raise RuntimeError("Invalid source video FPS.")

        sampled_frames = []

        previous_gray = None
        last_sampled_time = -float("inf")
        frame_index = 0

        motion_counts = {
            "low": 0,
            "medium": 0,
            "high": 0,
        }

        while True:
            success, frame = cap.read()

            if not success:
                break

            timestamp_seconds = frame_index / source_fps

            # Smaller grayscale frame makes motion analysis faster
            motion_frame = cv2.resize(
                frame,
                (320, 180),
                interpolation=cv2.INTER_AREA,
            )

            current_gray = cv2.cvtColor(
                motion_frame,
                cv2.COLOR_BGR2GRAY,
            )

            if previous_gray is None:
                motion_score = 0.0
            else:
                motion_score = self._calculate_motion(
                    previous_gray,
                    current_gray,
                )

            sampling_fps, motion_level = (
                self._select_sampling_fps(motion_score)
            )

            sampling_interval = 1.0 / sampling_fps

            if (
                timestamp_seconds - last_sampled_time
                >= sampling_interval
            ):
                sampled_frames.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": timestamp_seconds,
                        "motion_score": motion_score,
                        "motion_level": motion_level,
                        "sampling_fps": sampling_fps,
                    }
                )

                motion_counts[motion_level] += 1
                last_sampled_time = timestamp_seconds

            previous_gray = current_gray
            frame_index += 1

        cap.release()

        return {
            "source_fps": float(source_fps),
            "total_source_frames": frame_index,
            "sampled_frame_count": len(sampled_frames),
            "motion_counts": motion_counts,
            "sampled_frames": sampled_frames,
        }
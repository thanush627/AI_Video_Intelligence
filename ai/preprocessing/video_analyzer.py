import cv2
from pathlib import Path


class VideoAnalyzer:
    def __init__(self, video_path):
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {self.video_path}"
            )

    def analyze(self):
        cap = cv2.VideoCapture(str(self.video_path))

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video: {self.video_path}"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = frame_count / fps if fps > 0 else 0

        cap.release()

        return {
            "video_name": self.video_path.name,
            "video_path": str(self.video_path),
            "fps": float(fps),
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_seconds": duration,
            "duration_minutes": duration / 60,
        }
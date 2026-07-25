from pathlib import Path
from typing import List
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ImageLoader:
    def __init__(self, image_extensions=None):
        self.image_extensions = image_extensions or [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        ]

    def load_image(self, image_path: str):
        try:
            image = Image.open(image_path).convert("RGB")
            return image
        except Exception as e:
            logger.error(f"Failed to load {image_path}: {e}")
            return None

    def load_images(self, image_paths: List[str]):
        images = []
        valid_paths = []

        for path in image_paths:
            image = self.load_image(path)
            if image is not None:
                images.append(image)
                valid_paths.append(path)

        return images, valid_paths

    def get_track_images(self, track_folder: str):
        track_folder = Path(track_folder)

        images = sorted([
            str(file)
            for file in track_folder.iterdir()
            if file.suffix.lower() in self.image_extensions
        ])

        return images

    def get_all_tracks(self, representative_crop_dir: str):
        representative_crop_dir = Path(representative_crop_dir)

        tracks = sorted([
            folder
            for folder in representative_crop_dir.iterdir()
            if folder.is_dir()
        ])

        return tracks

    def load_track(self, track_folder: str):
        image_paths = self.get_track_images(track_folder)
        images, valid_paths = self.load_images(image_paths)

        return {
            "track_folder": track_folder,
            "image_paths": valid_paths,
            "images": images,
            "num_images": len(images)
        }
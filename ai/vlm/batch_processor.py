from typing import List, Iterator
from PIL import Image
import math


class BatchProcessor:
    def __init__(self, batch_size: int = 16):
        self.batch_size = batch_size

    def create_batches(
        self,
        images: List[Image.Image],
        image_paths: List[str]
    ) -> Iterator[dict]:

        total = len(images)

        for i in range(0, total, self.batch_size):
            yield {
                "batch_id": i // self.batch_size,
                "images": images[i:i + self.batch_size],
                "image_paths": image_paths[i:i + self.batch_size],
                "batch_size": len(images[i:i + self.batch_size])
            }

    def total_batches(self, total_images: int) -> int:
        return math.ceil(total_images / self.batch_size)

    def process_track(self, track_data: dict):

        return self.create_batches(
            track_data["images"],
            track_data["image_paths"]
        )

    def update_batch_size(self, batch_size: int):
        self.batch_size = batch_size
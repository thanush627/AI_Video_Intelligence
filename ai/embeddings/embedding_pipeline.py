import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ai.embeddings.embedding_generator import EmbeddingGenerator


class EmbeddingPipeline:

    def __init__(self):

        self.generator = EmbeddingGenerator()

    def generate_embeddings(
        self,
        crops_directory,
        output_directory,
    ):

        crops_directory = Path(crops_directory)
        output_directory = Path(output_directory)

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        embeddings = []
        metadata = []

        image_count = 0

        image_files = sorted(
            crops_directory.rglob("*.jpg")
        )

        print(f"\nFound {len(image_files)} images.\n")

        for image_file in tqdm(image_files):

            embedding = self.generator.generate_image_embedding(
                image_file
            )

            embeddings.append(embedding)

            track_id = image_file.parent.name

            rank = 1

            if image_file.name.startswith("rank_2"):
                rank = 2
            elif image_file.name.startswith("rank_3"):
                rank = 3

            embedding_id = f"{track_id}_rank_{rank}"

            metadata.append(
                {
                    "embedding_id": embedding_id,
                    "embedding_index": image_count,
                    "track_id": track_id,
                    "video_id": "test_video",
                    "rank": rank,
                    "image_name": image_file.name,
                    "image_path": str(image_file),
                }
            )

            image_count += 1

        embeddings = np.array(
            embeddings,
            dtype=np.float32,
        )

        np.save(
            output_directory / "image_embeddings.npy",
            embeddings,
        )

        with open(
            output_directory / "embedding_metadata.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                metadata,
                f,
                indent=2,
            )

        print("\nDone!")

        print(
            f"Embeddings : {embeddings.shape}"
        )

        print(
            f"Metadata : {len(metadata)}"
        )
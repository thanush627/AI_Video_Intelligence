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
        phase5_track_metadata_file = (
            Path("outputs")
            / "phase5"
            / "clean_track_metadata.json"
        )
        phase4_track_metadata_file = (
            Path("outputs")
            / "phase4"
            / "track_metadata.json"
        )

        track_metadata_file = (
            phase5_track_metadata_file
            if phase5_track_metadata_file.exists()
            else phase4_track_metadata_file
        )

        with open(track_metadata_file, "r", encoding="utf-8") as f:
            track_metadata = json.load(f)

        track_lookup = {}

        for item in track_metadata:
            track_lookup[item["track_id"]] = item
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

            track_info = track_lookup.get(track_id, {})

            metadata.append(
                {
                    "embedding_id": embedding_id,
                    "embedding_index": image_count,
                    "track_id": track_id,
                    "video_id": "test_video",
                    "rank": rank,
                    "image_name": image_file.name,
                    "image_path": str(image_file),

                    # ---------- Semantic Metadata ----------
                    "class_names": track_info.get(
                        "object_type",
                        ""
                    ),

                    "event_type": "object_event",

                    "upper_body_color": track_info.get(
                        "colors",
                        {}
                    ).get(
                        "upper_body",
                        ""
                    ),

                    "lower_body_color": track_info.get(
                        "colors",
                        {}
                    ).get(
                        "lower_body",
                        ""
                    ),

                    "attributes": ", ".join(
                        track_info.get(
                            "attributes",
                            []
                        )
                    ),

                    "action": track_info.get(
                        "action",
                        ""
                    ),

                    "orientation": track_info.get(
                        "orientation",
                        ""
                    ),

                    "visibility": track_info.get(
                        "visibility",
                        ""
                    ),

                    "start_time_seconds": track_info.get(
                        "start_time_seconds"
                    ),

                    "end_time_seconds": track_info.get(
                        "end_time_seconds"
                    ),

                    "duration_seconds": track_info.get(
                        "duration_seconds"
                    ),

                    "timestamp": track_info.get(
                        "timestamp"
                    ) or track_info.get(
                        "start_timestamp"
                    ),

                    "start_timestamp": track_info.get(
                        "start_timestamp"
                    ),

                    "end_timestamp": track_info.get(
                        "end_timestamp"
                    ),

                    "quality_score": track_info.get(
                        "confidence",
                        {}
                    ).get(
                        "object",
                        0.0
                    ),

                    "description": (
                        f"{track_info.get('object_type','')} "
                        f"{track_info.get('action','')} "
                        f"{track_info.get('orientation','')} "
                        f"{track_info.get('visibility','')} "
                        f"{track_info.get('timestamp','')}"
                    ).strip()
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
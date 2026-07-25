from pathlib import Path
import json
import numpy as np
import chromadb


class ChromaManager:

    def __init__(
        self,
        db_path="database/chromadb",
        collection_name="image_embeddings",
    ):

        self.client = chromadb.PersistentClient(
            path=db_path
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine"
            },
        )

    def load_embeddings(
        self,
        embedding_file,
        metadata_file,
    ):

        embeddings = np.load(
            embedding_file
        )

        with open(
            metadata_file,
            "r",
            encoding="utf-8",
        ) as f:

            metadata = json.load(f)

        return embeddings, metadata

    def store(
        self,
        embedding_file,
        metadata_file,
    ):

        embeddings, metadata = self.load_embeddings(
            embedding_file,
            metadata_file,
        )

        print(
            f"Storing {len(metadata)} embeddings..."
        )

        for index, item in enumerate(metadata):

            self.collection.add(

                ids=[
                    item["embedding_id"]
                ],

                embeddings=[
                    embeddings[index].tolist()
                ],

                metadatas=[
                    item
                ],

            )

        print(
            "Finished storing embeddings."
        )

        print(
            "Collection Count:",
            self.collection.count()
        )
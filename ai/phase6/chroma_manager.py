from pathlib import Path

import chromadb
from chromadb.config import Settings


class ChromaManager:
    """
    Handles ChromaDB creation and collection management.
    """

    def __init__(self):

        root = Path(__file__).resolve().parents[2]

        self.db_path = root / "outputs" / "phase6" / "chromadb"

        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False
            )
        )

        self.collection = self.client.get_or_create_collection(
            name="video_embeddings",
            metadata={
                "description": "AI Video Intelligence CLIP Embeddings"
            }
        )

    def get_collection(self):
        return self.collection

    def reset_collection(self):

        try:
            self.client.delete_collection("video_embeddings")
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name="video_embeddings"
        )

        return self.collection
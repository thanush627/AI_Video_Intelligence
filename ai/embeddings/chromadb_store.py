import json
from pathlib import Path

import chromadb
import numpy as np


class ChromaDBEventStore:

    def __init__(
        self,
        database_directory,
        collection_name="event_vectors",
    ):
        self.database_directory = Path(
            database_directory
        )

        self.database_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.collection_name = collection_name

        print(
            f"ChromaDB directory: "
            f"{self.database_directory.resolve()}"
        )

        self.client = chromadb.PersistentClient(
            path=str(
                self.database_directory.resolve()
            )
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": (
                        "Phase 3 semantic video event embeddings"
                    ),
                    "embedding_model": "ViT-B-32",
                    "embedding_dimension": 512,
                    "distance_metric": "cosine",
                    "hnsw:space": "cosine",
                },
            )
        )

    @staticmethod
    def _load_json(file_path):
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    @staticmethod
    def _safe_list(value):
        if value is None:
            return []

        if isinstance(value, list):
            return value

        return [value]

    @staticmethod
    def _safe_float(value, default=-1.0):
        if value is None:
            return default

        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_string(value, default="unknown"):
        if value is None:
            return default

        text = str(value).strip()

        return text if text else default

    def load_embeddings(
        self,
        embeddings_file,
    ):
        embeddings_file = Path(embeddings_file)

        if not embeddings_file.exists():
            raise FileNotFoundError(
                f"Embeddings file not found: "
                f"{embeddings_file}"
            )

        embeddings = np.load(
            embeddings_file
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D array."
            )

        return embeddings.astype(
            np.float32
        )

    def prepare_records(
        self,
        embedding_metadata_file,
        retrieval_documents_file,
        embeddings,
    ):
        embedding_data = self._load_json(
            embedding_metadata_file
        )

        retrieval_data = self._load_json(
            retrieval_documents_file
        )

        embedding_documents = embedding_data.get(
            "documents",
            [],
        )

        if isinstance(retrieval_data, dict):
            retrieval_documents = (
                retrieval_data.get("documents")
                or retrieval_data.get(
                    "retrieval_documents"
                )
                or retrieval_data.get("events")
                or []
            )
        elif isinstance(retrieval_data, list):
            retrieval_documents = retrieval_data
        else:
            retrieval_documents = []

        if not embedding_documents:
            raise ValueError(
                "No embedding metadata documents found."
            )

        if not retrieval_documents:
            raise ValueError(
                "No retrieval documents found."
            )

        if len(embedding_documents) != len(embeddings):
            raise ValueError(
                "Embedding metadata count does not match "
                "embedding array count."
            )

        if len(retrieval_documents) != len(embeddings):
            raise ValueError(
                "Retrieval document count does not match "
                "embedding array count."
            )

        ids = []
        documents = []
        metadatas = []
        embedding_vectors = []

        for index in range(len(embeddings)):

            embedding_document = (
                embedding_documents[index]
            )

            retrieval_document = (
                retrieval_documents[index]
            )

            document_id = (
                embedding_document.get("document_id")
                or retrieval_document.get("document_id")
                or retrieval_document.get("event_id")
                or f"event_document_{index:06d}"
            )

            event_id = (
                retrieval_document.get("event_id")
                or embedding_document.get("event_id")
                or document_id
            )

            text = (
                embedding_document.get("text")
                or retrieval_document.get("text")
                or retrieval_document.get("document")
                or retrieval_document.get("description")
                or ""
            )

            if not text:
                raise ValueError(
                    f"Missing text for record {index}"
                )

            source_metadata = retrieval_document.get(
                "metadata",
                {},
            )

            if not isinstance(
                source_metadata,
                dict,
            ):
                source_metadata = {}

            track_ids = self._safe_list(
                source_metadata.get(
                    "track_ids",
                    retrieval_document.get(
                        "track_ids"
                    ),
                )
            )

            class_names = self._safe_list(
                source_metadata.get(
                    "class_names",
                    retrieval_document.get(
                        "class_names"
                    ),
                )
            )

            # ------------------------------------------
            # Read valid event timestamps
            # ------------------------------------------

            start_time_seconds = self._safe_float(
                source_metadata.get(
                    "start_time_seconds",
                    retrieval_document.get(
                        "start_time_seconds"
                    ),
                )
            )

            end_time_seconds = self._safe_float(
                source_metadata.get(
                    "end_time_seconds",
                    retrieval_document.get(
                        "end_time_seconds"
                    ),
                )
            )

            # ------------------------------------------
            # Read duration if it already exists
            # ------------------------------------------

            raw_duration = source_metadata.get(
                "duration_seconds",
                retrieval_document.get(
                    "duration_seconds"
                ),
            )

            # ------------------------------------------
            # Calculate missing duration
            # ------------------------------------------

            if raw_duration is not None:
                duration_seconds = self._safe_float(
                    raw_duration
                )

            elif (
                start_time_seconds >= 0
                and end_time_seconds >= 0
            ):
                duration_seconds = round(
                    end_time_seconds
                    - start_time_seconds,
                    4,
                )

            else:
                duration_seconds = -1.0

            metadata = {
                "event_id": self._safe_string(
                    event_id
                ),
                "event_type": self._safe_string(
                    source_metadata.get(
                        "event_type",
                        retrieval_document.get(
                            "event_type"
                        ),
                    )
                ),
                "start_time_seconds": (
                    start_time_seconds
                ),
                "end_time_seconds": (
                    end_time_seconds
                ),
                "duration_seconds": (
                    duration_seconds
                ),
                "quality_score": (
                    self._safe_float(
                        source_metadata.get(
                            "quality_score",
                            retrieval_document.get(
                                "quality_score"
                            ),
                        ),
                        default=0.0,
                    )
                ),
                "quality_label": (
                    self._safe_string(
                        source_metadata.get(
                            "quality_label",
                            retrieval_document.get(
                                "quality_label"
                            ),
                        )
                    )
                ),
                "track_ids": ",".join(
                    str(track_id)
                    for track_id in track_ids
                    if track_id is not None
                ),
                "class_names": ",".join(
                    str(class_name)
                    for class_name in class_names
                    if class_name is not None
                ),
                "embedding_index": int(index),
            }

            ids.append(
                str(document_id)
            )

            documents.append(
                str(text)
            )

            metadatas.append(
                metadata
            )

            embedding_vectors.append(
                embeddings[index].tolist()
            )

        return {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "embeddings": embedding_vectors,
        }

    def store_records(
        self,
        records,
        batch_size=100,
    ):
        total_records = len(
            records["ids"]
        )

        if total_records == 0:
            raise ValueError(
                "No records available for storage."
            )

        for start in range(
            0,
            total_records,
            batch_size,
        ):
            end = min(
                start + batch_size,
                total_records,
            )

            self.collection.upsert(
                ids=records["ids"][start:end],
                documents=records[
                    "documents"
                ][start:end],
                metadatas=records[
                    "metadatas"
                ][start:end],
                embeddings=records[
                    "embeddings"
                ][start:end],
            )

            print(
                f"Stored {end}/{total_records} records"
            )

    def index_events(
        self,
        embeddings_file,
        embedding_metadata_file,
        retrieval_documents_file,
    ):
        print("\nLoading CLIP embeddings...")

        embeddings = self.load_embeddings(
            embeddings_file
        )

        print(
            f"Loaded embeddings: "
            f"{embeddings.shape}"
        )

        print(
            "\nPreparing ChromaDB records..."
        )

        records = self.prepare_records(
            embedding_metadata_file=(
                embedding_metadata_file
            ),
            retrieval_documents_file=(
                retrieval_documents_file
            ),
            embeddings=embeddings,
        )

        print(
            f"Prepared records: "
            f"{len(records['ids'])}"
        )

        print(
            "\nStoring records in ChromaDB..."
        )

        self.store_records(
            records
        )

        final_count = self.collection.count()

        return {
            "input_embeddings": int(
                embeddings.shape[0]
            ),
            "embedding_dimension": int(
                embeddings.shape[1]
            ),
            "prepared_records": len(
                records["ids"]
            ),
            "collection_count": int(
                final_count
            ),
            "collection_name": (
                self.collection_name
            ),
            "database_directory": str(
                self.database_directory.resolve()
            ),
        }

    def get_sample_records(
        self,
        limit=5,
    ):
        count = self.collection.count()

        actual_limit = min(
            limit,
            count,
        )

        if actual_limit == 0:
            return {
                "ids": [],
                "documents": [],
                "metadatas": [],
            }

        return self.collection.get(
            limit=actual_limit,
            include=[
                "documents",
                "metadatas",
            ],
        )
import sys
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from ai.embeddings.chromadb_store import (
    ChromaDBEventStore,
)


EMBEDDINGS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "event_embedding_test"
    / "event_embeddings.npy"
)

EMBEDDING_METADATA_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "event_embedding_test"
    / "event_embedding_metadata.json"
)

RETRIEVAL_DOCUMENTS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "event_quality_filter_test"
    / "retrieval_documents.json"
)

DATABASE_DIRECTORY = (
    PROJECT_ROOT
    / "database"
    / "chromadb"
    / "phase3_events"
)


def main():

    print("\n" + "=" * 70)
    print("CHROMADB EVENT STORAGE TEST")
    print("=" * 70)

    store = ChromaDBEventStore(
        database_directory=DATABASE_DIRECTORY,
        collection_name="event_vectors",
    )

    result = store.index_events(
        embeddings_file=EMBEDDINGS_FILE,
        embedding_metadata_file=(
            EMBEDDING_METADATA_FILE
        ),
        retrieval_documents_file=(
            RETRIEVAL_DOCUMENTS_FILE
        ),
    )

    print("\n" + "=" * 70)
    print("CHROMADB STORAGE TEST")
    print("=" * 70)

    print(
        f"Input embeddings    : "
        f"{result['input_embeddings']}"
    )

    print(
        f"Embedding dimension : "
        f"{result['embedding_dimension']}"
    )

    print(
        f"Prepared records    : "
        f"{result['prepared_records']}"
    )

    print(
        f"Collection count    : "
        f"{result['collection_count']}"
    )

    print(
        f"Collection name     : "
        f"{result['collection_name']}"
    )

    print(
        f"Database directory  : "
        f"{result['database_directory']}"
    )

    print("=" * 70)

    sample = store.get_sample_records(
        limit=5
    )

    print("\nFIRST 5 STORED RECORDS")
    print("-" * 70)

    for index in range(
        len(sample["ids"])
    ):
        print(
            f"\nRecord {index + 1}"
        )

        print(
            f"ID       : "
            f"{sample['ids'][index]}"
        )

        print(
            f"Document : "
            f"{sample['documents'][index]}"
        )

        print(
            f"Metadata : "
            f"{sample['metadatas'][index]}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ai.embeddings.event_embedder import EventEmbedder


INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "event_quality_filter_test"
    / "retrieval_documents.json"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "event_embedding_test"
)


def main():

    print("\n" + "=" * 70)
    print("CLIP EVENT EMBEDDING TEST")
    print("=" * 70)

    embedder = EventEmbedder(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        batch_size=16,
    )

    result = embedder.generate_embeddings(
        input_file=INPUT_FILE,
        output_directory=OUTPUT_DIRECTORY,
    )

    embeddings = result["embeddings"]

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    print("\n" + "=" * 70)
    print("EVENT EMBEDDING TEST")
    print("=" * 70)

    print(
        f"Input documents     : "
        f"{len(result['documents'])}"
    )

    print(
        f"Generated embeddings: "
        f"{embeddings.shape[0]}"
    )

    print(
        f"Embedding dimension : "
        f"{embeddings.shape[1]}"
    )

    print(
        f"Embedding dtype     : "
        f"{embeddings.dtype}"
    )

    print(
        f"Average vector norm : "
        f"{norms.mean():.6f}"
    )

    print(
        f"Minimum vector norm : "
        f"{norms.min():.6f}"
    )

    print(
        f"Maximum vector norm : "
        f"{norms.max():.6f}"
    )

    print(
        f"Embeddings file     : "
        f"{result['embeddings_file']}"
    )

    print(
        f"Metadata file       : "
        f"{result['metadata_file']}"
    )

    print("=" * 70)

    print("\nFIRST 5 EMBEDDED DOCUMENTS")
    print("-" * 70)

    for index, text in enumerate(
        result["texts"][:5]
    ):
        print(
            f"{index + 1:2d}. {text}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
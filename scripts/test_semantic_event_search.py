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


from ai.retrieval.semantic_event_search import (
    SemanticEventSearch,
)


DATABASE_DIRECTORY = (
    PROJECT_ROOT
    / "database"
    / "chromadb"
    / "phase3_events"
)

COLLECTION_NAME = "event_vectors"


TEST_QUERIES = [
    "blue bus between 5 and 8 seconds",
    "pedestrians after 10 seconds",
    "vehicles before 5 seconds",
    "white car in the first 3 seconds",
    "pedestrian moving left to right between 7 and 9 seconds",
]

def safe_float(
    value,
    default=0.0,
):
    try:
        if value is None:
            return float(default)

        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return float(default)


def print_results(
    query,
    results,
):
    print("\n" + "=" * 70)

    print(
        f"QUERY: {query}"
    )

    print("=" * 70)

    if not results:
        print(
            "No matching events found."
        )
        return

    for result in results:

        rank = result.get(
            "rank",
            "-",
        )

        event_id = result.get(
            "event_id",
            "unknown",
        )

        hybrid_score = safe_float(
            result.get(
                "hybrid_score",
                result.get(
                    "final_score",
                    result.get(
                        "score",
                        result.get(
                            "similarity",
                            0.0,
                        ),
                    ),
                ),
            )
        )

        clip_similarity = safe_float(
            result.get(
                "clip_similarity",
                result.get(
                    "similarity",
                    0.0,
                ),
            )
        )

        rerank_adjustment = (
            hybrid_score
            - clip_similarity
        )

        class_names = result.get(
            "class_names",
            "",
        )

        track_ids = result.get(
            "track_ids",
            "",
        )

        start_time = safe_float(
            result.get(
                "start_time_seconds",
                0.0,
            )
        )

        end_time = safe_float(
            result.get(
                "end_time_seconds",
                0.0,
            )
        )

        duration = safe_float(
            result.get(
                "duration_seconds",
                max(
                    0.0,
                    end_time
                    - start_time,
                ),
            )
        )

        quality_label = result.get(
            "quality_label",
            "unknown",
        )

        quality_score = safe_float(
            result.get(
                "quality_score",
                0.0,
            )
        )

        document = result.get(
            "document",
            "",
        )

        print(
            f"\nRank {rank}"
        )

        print(
            f"Event ID          : "
            f"{event_id}"
        )

        print(
            f"Hybrid Score      : "
            f"{hybrid_score:.4f}"
        )

        print(
            f"CLIP Similarity   : "
            f"{clip_similarity:.4f}"
        )

        print(
            f"Rerank Adjustment : "
            f"{rerank_adjustment:+.4f}"
        )

        print(
            f"Class             : "
            f"{class_names}"
        )

        print(
            f"Track IDs         : "
            f"{track_ids}"
        )

        print(
            f"Time              : "
            f"{start_time:.2f}s"
            f" -> "
            f"{end_time:.2f}s"
        )

        print(
            f"Duration          : "
            f"{duration:.2f}s"
        )

        print(
            f"Quality           : "
            f"{quality_label} "
            f"({quality_score:.3f})"
        )

        print(
            f"Document          : "
            f"{document}"
        )


def main():

    print("\n" + "=" * 70)

    print(
        "SEMANTIC EVENT SEARCH TEST"
    )

    print("=" * 70)

    search_engine = SemanticEventSearch(
        database_directory=(
            DATABASE_DIRECTORY
        ),
        collection_name=(
            COLLECTION_NAME
        ),
    )

    for query in TEST_QUERIES:

        results = search_engine.search(
            query=query,
            top_k=5,
        )

        print_results(
            query=query,
            results=results,
        )

    print("\n" + "=" * 70)

    print(
        "SEMANTIC SEARCH TEST COMPLETE"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
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

from ai.retrieval.query_orchestrator import (
    QueryOrchestrator,
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
    (
        "pedestrian moving left to right "
        "between 7 and 9 seconds"
    ),
    "pedestrians near each other",
    "pedestrians overlapping",
]


def safe_float(
    value,
    default=0.0,
):
    try:
        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return default


def print_result(
    result,
):
    print(
        f"\nRank {result['rank']}"
    )

    print(
        f"Event ID          : "
        f"{result.get('event_id')}"
    )

    print(
        f"Event Type        : "
        f"{result.get('event_type')}"
    )

    print(
        f"Hybrid Score      : "
        f"{safe_float(result.get('hybrid_score')):.4f}"
    )

    print(
        f"CLIP Similarity   : "
        f"{safe_float(result.get('similarity')):.4f}"
    )

    print(
        f"Rerank Adjustment : "
        f"{safe_float(result.get('rerank_adjustment')):+.4f}"
    )

    print(
        f"Class             : "
        f"{result.get('class_names')}"
    )

    print(
        f"Track IDs         : "
        f"{result.get('track_ids')}"
    )

    print(
        f"Time              : "
        f"{safe_float(result.get('start_time_seconds')):.2f}s"
        f" -> "
        f"{safe_float(result.get('end_time_seconds')):.2f}s"
    )

    print(
        f"Duration          : "
        f"{safe_float(result.get('duration_seconds')):.2f}s"
    )

    print(
        f"Quality           : "
        f"{result.get('quality_label')} "
        f"("
        f"{safe_float(result.get('quality_score')):.3f}"
        f")"
    )

    print(
        f"Document          : "
        f"{result.get('document')}"
    )


def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "UNIFIED QUERY ORCHESTRATOR TEST"
    )

    print(
        "=" * 70
    )

    search_engine = SemanticEventSearch(
        database_directory=(
            DATABASE_DIRECTORY
        ),
        collection_name=(
            COLLECTION_NAME
        ),
    )

    orchestrator = QueryOrchestrator(
        search_engine=search_engine
    )

    for query in TEST_QUERIES:

        response = orchestrator.search(
            query=query,
            top_k=5,
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"QUERY: {query}"
        )

        print(
            "=" * 70
        )

        print(
            f"Detected Intent   : "
            f"{response['intent']}"
        )

        print(
            f"Intent Confidence : "
            f"{response['intent_confidence']:.2f}"
        )

        print(
            f"Intent Terms      : "
            f"{response['intent_terms']}"
        )

        print(
            f"Raw Results       : "
            f"{response['raw_result_count']}"
        )

        print(
            f"Filtered Results  : "
            f"{response['filtered_result_count']}"
        )

        print(
            f"Final Results     : "
            f"{response['result_count']}"
        )

        if not response[
            "results"
        ]:
            print(
                "\nNo matching events found."
            )

            continue

        for result in response[
            "results"
        ]:
            print_result(
                result
            )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "QUERY ORCHESTRATOR TEST COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
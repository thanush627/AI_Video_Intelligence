import json
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

from ai.retrieval.retrieval_response_builder import (
    RetrievalResponseBuilder,
)


DATABASE_DIRECTORY = (
    PROJECT_ROOT
    / "database"
    / "chromadb"
    / "phase3_events"
)

COLLECTION_NAME = "event_vectors"

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "retrieval_response_builder_test"
)

OUTPUT_FILE = (
    OUTPUT_DIRECTORY
    / "retrieval_responses.json"
)


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
    "purple truck between 50 and 60 seconds",
]


def print_response(
    response,
):
    print(
        "\n"
        + "=" * 70
    )

    print(
        f"QUERY: {response['query']}"
    )

    print(
        "=" * 70
    )

    print(
        f"Success           : "
        f"{response['success']}"
    )

    print(
        f"Intent            : "
        f"{response['intent']}"
    )

    print(
        f"Intent Confidence : "
        f"{response['intent_confidence']:.2f}"
    )

    print(
        f"Match Found       : "
        f"{response['match_found']}"
    )

    print(
        f"Result Count      : "
        f"{response['result_count']}"
    )

    print(
        f"Summary           : "
        f"{response['summary']}"
    )

    if not response[
        "results"
    ]:
        return

    print(
        "\nFINAL RESULTS"
    )

    print(
        "-" * 70
    )

    for result in response[
        "results"
    ]:

        print(
            f"\nRank {result['rank']}"
        )

        print(
            f"Event ID     : "
            f"{result['event_id']}"
        )

        print(
            f"Event Type   : "
            f"{result['event_type']}"
        )

        print(
            f"Description  : "
            f"{result['description']}"
        )

        print(
            f"Classes      : "
            f"{result['class_names']}"
        )

        print(
            f"Track IDs    : "
            f"{result['track_ids']}"
        )

        print(
            f"Time         : "
            f"{result['start_timestamp']}"
            f" -> "
            f"{result['end_timestamp']}"
        )

        print(
            f"Seconds      : "
            f"{result['start_time_seconds']:.2f}"
            f"s -> "
            f"{result['end_time_seconds']:.2f}"
            f"s"
        )

        print(
            f"Duration     : "
            f"{result['duration_seconds']:.2f}s"
        )

        print(
            f"Hybrid Score : "
            f"{result['hybrid_score']:.4f}"
        )

        print(
            f"Quality      : "
            f"{result['quality_label']} "
            f"("
            f"{result['quality_score']:.3f}"
            f")"
        )


def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RETRIEVAL RESPONSE BUILDER TEST"
    )

    print(
        "=" * 70
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
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

    response_builder = (
        RetrievalResponseBuilder()
    )

    all_responses = []

    for query in TEST_QUERIES:

        orchestrator_response = (
            orchestrator.search(
                query=query,
                top_k=5,
            )
        )

        final_response = (
            response_builder.build(
                orchestrator_response
            )
        )

        all_responses.append(
            final_response
        )

        print_response(
            final_response
        )

    output_data = {
        "test_name": (
            "retrieval_response_builder"
        ),
        "query_count": len(
            TEST_QUERIES
        ),
        "responses": all_responses,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output_data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RETRIEVAL RESPONSE BUILDER TEST COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Queries tested : "
        f"{len(TEST_QUERIES)}"
    )

    print(
        f"Output file    : "
        f"{OUTPUT_FILE}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
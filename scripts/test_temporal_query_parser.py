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


from ai.retrieval.temporal_query_parser import (
    TemporalQueryParser,
)


TEST_QUERIES = [
    "blue bus between 5 and 8 seconds",
    "pedestrians after 10 seconds",
    "vehicles before 5 seconds",
    "events from 8 to 12 seconds",
    "white car in the first 3 seconds",
    "pedestrian moving left to right",
]


def main():

    parser = TemporalQueryParser()

    print("\n" + "=" * 70)

    print(
        "TEMPORAL QUERY PARSER TEST"
    )

    print("=" * 70)

    for query in TEST_QUERIES:

        result = parser.parse(
            query
        )

        print(
            f"\nQuery          : "
            f"{query}"
        )

        print(
            f"Semantic Query : "
            f"{result['semantic_query']}"
        )

        print(
            f"Has Filter     : "
            f"{result['has_temporal_filter']}"
        )

        print(
            f"Temporal Filter: "
            f"{result['temporal_filter']}"
        )

    print("\n" + "=" * 70)

    print(
        "TEMPORAL QUERY PARSER TEST COMPLETE"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
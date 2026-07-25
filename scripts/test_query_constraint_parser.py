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


from ai.retrieval.query_constraint_parser import (
    QueryConstraintParser,
)


TEST_QUERIES = [
    "blue bus",
    "white car",
    "person wearing blue clothes",
    "pedestrian moving left to right",
    "vehicle on the right side",
    "stationary black bus",
    "red van moving bottom right to top left",
    "pedestrian in the center bottom region",
    "vehicles before 5 seconds",
    "blue bus between 5 and 8 seconds",
]


def main():

    print("\n" + "=" * 70)

    print(
        "QUERY CONSTRAINT PARSER TEST"
    )

    print("=" * 70)

    parser = QueryConstraintParser()

    for query in TEST_QUERIES:

        result = parser.parse(
            query
        )

        constraints = result[
            "constraints"
        ]

        print(
            f"\nQuery              : "
            f"{query}"
        )

        print(
            f"Has Constraints    : "
            f"{result['has_constraints']}"
        )

        print(
            f"Object Classes     : "
            f"{constraints['object_classes']}"
        )

        print(
            f"Class Group        : "
            f"{constraints['class_group']}"
        )

        print(
            f"Allowed Classes    : "
            f"{constraints['allowed_classes']}"
        )

        print(
            f"Colors             : "
            f"{constraints['colors']}"
        )

        print(
            f"Motion States      : "
            f"{constraints['motion_states']}"
        )

        print(
            f"Motion Directions  : "
            f"{constraints['motion_directions']}"
        )

        print(
            f"Spatial Regions    : "
            f"{constraints['spatial_regions']}"
        )

        print(
            f"Broad Regions      : "
            f"{constraints['broad_spatial_regions']}"
        )

    print("\n" + "=" * 70)

    print(
        "QUERY CONSTRAINT PARSER TEST COMPLETE"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
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


from ai.retrieval.end_to_end_retrieval_pipeline import (
    EndToEndRetrievalPipeline,
)


DATABASE_DIRECTORY = (
    PROJECT_ROOT
    / "database"
    / "chromadb"
    / "phase3_events"
)

COLLECTION_NAME = "event_vectors"

SOURCE_VIDEO = (
    PROJECT_ROOT
    / "test_videos"
    / "test.mp4"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "end_to_end_retrieval_test"
)

CLIPS_DIRECTORY = (
    OUTPUT_DIRECTORY
    / "clips"
)

RESULTS_DIRECTORY = (
    OUTPUT_DIRECTORY
    / "results"
)


TEST_CASES = [
    {
        "name": "blue_bus_temporal",
        "query": (
            "blue bus between 5 and 8 seconds"
        ),
        "top_k": 5,
        "maximum_clips": 1,
    },
    {
        "name": "pedestrian_motion_temporal",
        "query": (
            "pedestrian moving left to right "
            "between 7 and 9 seconds"
        ),
        "top_k": 5,
        "maximum_clips": 3,
    },
    {
        "name": "relationship_overlap",
        "query": (
            "pedestrians overlapping"
        ),
        "top_k": 5,
        "maximum_clips": 3,
    },
    {
        "name": "no_result",
        "query": (
            "purple truck between "
            "50 and 60 seconds"
        ),
        "top_k": 5,
        "maximum_clips": 1,
    },
]


def print_result(
    result,
):
    summary = result[
        "summary"
    ]

    retrieval = result[
        "retrieval"
    ]

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"QUERY: {result['query']}"
    )

    print(
        "=" * 70
    )

    print(
        f"Pipeline Success  : "
        f"{result['success']}"
    )

    print(
        f"Intent            : "
        f"{retrieval['intent']}"
    )

    print(
        f"Match Found       : "
        f"{summary['match_found']}"
    )

    print(
        f"Retrieved Events  : "
        f"{summary['retrieved_event_count']}"
    )

    print(
        f"Generated Clips   : "
        f"{summary['generated_clip_count']}"
    )

    print(
        f"Failed Clips      : "
        f"{summary['failed_clip_count']}"
    )

    print(
        f"Processing Time   : "
        f"{summary['processing_time_seconds']:.4f}s"
    )

    print(
        f"Response Summary  : "
        f"{retrieval['summary']}"
    )

    print(
        "\nSTAGE TIMINGS"
    )

    print(
        "-" * 70
    )

    for stage_name, duration in (
        result["timings"].items()
    ):
        print(
            f"{stage_name:<30}: "
            f"{duration:.4f}s"
        )

    if not retrieval[
        "results"
    ]:
        print(
            "\nNo matching events found."
        )

        return

    print(
        "\nFINAL RESULTS"
    )

    print(
        "-" * 70
    )

    for event in retrieval[
        "results"
    ]:

        print(
            f"\nRank {event['rank']}"
        )

        print(
            f"Event ID      : "
            f"{event['event_id']}"
        )

        print(
            f"Event Type    : "
            f"{event['event_type']}"
        )

        print(
            f"Description   : "
            f"{event['description']}"
        )

        print(
            f"Event Time    : "
            f"{event['start_time_seconds']:.2f}s"
            f" -> "
            f"{event['end_time_seconds']:.2f}s"
        )

        print(
            f"Hybrid Score  : "
            f"{event['hybrid_score']:.4f}"
        )

        evidence_clip = event.get(
            "evidence_clip",
            {},
        )

        print(
            f"Clip Available: "
            f"{evidence_clip.get('available')}"
        )

        print(
            f"Clip Status   : "
            f"{evidence_clip.get('status')}"
        )

        if evidence_clip.get(
            "available"
        ):
            print(
                f"Clip Time     : "
                f"{evidence_clip.get('clip_start_time_seconds'):.2f}s"
                f" -> "
                f"{evidence_clip.get('clip_end_time_seconds'):.2f}s"
            )

            print(
                f"Clip Path     : "
                f"{evidence_clip.get('clip_path')}"
            )


def validate_result(
    test_case,
    result,
):
    query = test_case[
        "query"
    ]

    retrieval = result[
        "retrieval"
    ]

    summary = result[
        "summary"
    ]

    if (
        "purple truck"
        in query.lower()
    ):
        assert (
            summary[
                "match_found"
            ]
            is False
        )

        assert (
            summary[
                "retrieved_event_count"
            ]
            == 0
        )

        assert (
            summary[
                "generated_clip_count"
            ]
            == 0
        )

        return

    assert (
        summary[
            "match_found"
        ]
        is True
    )

    assert (
        summary[
            "retrieved_event_count"
        ]
        > 0
    )

    assert (
        summary[
            "failed_clip_count"
        ]
        == 0
    )

    expected_clip_count = min(
        summary[
            "retrieved_event_count"
        ],
        test_case[
            "maximum_clips"
        ],
    )

    assert (
        summary[
            "generated_clip_count"
        ]
        == expected_clip_count
    )

    for index, event in enumerate(
        retrieval[
            "results"
        ]
    ):

        evidence_clip = event.get(
            "evidence_clip",
            {},
        )

        if index < expected_clip_count:

            assert (
                evidence_clip.get(
                    "available"
                )
                is True
            )

            clip_path = Path(
                evidence_clip[
                    "clip_path"
                ]
            )

            assert (
                clip_path.exists()
            )

            assert (
                clip_path.stat().st_size
                > 0
            )

        else:

            assert (
                evidence_clip.get(
                    "status"
                )
                == "not_requested"
            )


def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL END-TO-END RETRIEVAL PIPELINE TEST"
    )

    print(
        "=" * 70
    )

    if not SOURCE_VIDEO.exists():
        raise FileNotFoundError(
            "Source test video was not found:\n"
            f"{SOURCE_VIDEO}"
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    pipeline = (
        EndToEndRetrievalPipeline(
            database_directory=(
                DATABASE_DIRECTORY
            ),
            collection_name=(
                COLLECTION_NAME
            ),
            clip_output_directory=(
                CLIPS_DIRECTORY
            ),
            context_before_seconds=2.0,
            context_after_seconds=2.0,
            minimum_clip_duration_seconds=1.0,
            overwrite_clips=True,
        )
    )

    passed_tests = 0

    failed_tests = 0

    for test_case in TEST_CASES:

        print(
            "\n"
            + "#" * 70
        )

        print(
            f"RUNNING TEST: "
            f"{test_case['name']}"
        )

        print(
            "#" * 70
        )

        try:
            result = pipeline.run(
                query=(
                    test_case[
                        "query"
                    ]
                ),
                video_path=(
                    SOURCE_VIDEO
                ),
                top_k=(
                    test_case[
                        "top_k"
                    ]
                ),
                maximum_clips=(
                    test_case[
                        "maximum_clips"
                    ]
                ),
                generate_clips=True,
            )

            validate_result(
                test_case=(
                    test_case
                ),
                result=result,
            )

            output_file = (
                RESULTS_DIRECTORY
                / (
                    test_case[
                        "name"
                    ]
                    + ".json"
                )
            )

            pipeline.save_result(
                result=result,
                output_path=(
                    output_file
                ),
            )

            print_result(
                result
            )

            print(
                "\nTEST STATUS: PASSED"
            )

            print(
                f"Result File: "
                f"{output_file}"
            )

            passed_tests += 1

        except Exception as error:

            failed_tests += 1

            print(
                "\nTEST STATUS: FAILED"
            )

            print(
                f"Error: {error}"
            )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "END-TO-END RETRIEVAL TEST SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Total tests  : "
        f"{len(TEST_CASES)}"
    )

    print(
        f"Passed tests : "
        f"{passed_tests}"
    )

    print(
        f"Failed tests : "
        f"{failed_tests}"
    )

    print(
        f"Clips folder : "
        f"{CLIPS_DIRECTORY}"
    )

    print(
        f"Results      : "
        f"{RESULTS_DIRECTORY}"
    )

    print(
        "=" * 70
    )

    if failed_tests > 0:
        raise RuntimeError(
            f"{failed_tests} end-to-end "
            "test(s) failed."
        )

    print(
        "\nFINAL END-TO-END RETRIEVAL "
        "PIPELINE PASSED"
    )


if __name__ == "__main__":
    main()
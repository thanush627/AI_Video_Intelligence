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


from ai.pipeline.phase3_production_pipeline import (
    Phase3ProductionPipeline,
)


SOURCE_VIDEO = (
    PROJECT_ROOT
    / "test_videos"
    / "test.mp4"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "ai"
    / "configs"
    / "phase3_config.yaml"
)

DATABASE_DIRECTORY = (
    PROJECT_ROOT
    / "database"
    / "chromadb"
    / "phase3_events"
)

COLLECTION_NAME = "event_vectors"


def validate_file(
    path,
    label,
):
    path = Path(
        path
    )

    assert path.exists(), (
        f"{label} does not exist: "
        f"{path}"
    )

    assert path.is_file(), (
        f"{label} is not a file: "
        f"{path}"
    )

    assert path.stat().st_size > 0, (
        f"{label} is empty: "
        f"{path}"
    )

    return path


def main():

    print(
        "\n"
        + "=" * 78
    )

    print(
        "PHASE 3 PRODUCTION PIPELINE TEST"
    )

    print(
        "=" * 78
    )

    if not SOURCE_VIDEO.exists():
        raise FileNotFoundError(
            "Test video was not found:\n"
            f"{SOURCE_VIDEO}"
        )

    pipeline = (
        Phase3ProductionPipeline(
            project_root=PROJECT_ROOT,
            config_path=CONFIG_PATH,
            database_directory=(
                DATABASE_DIRECTORY
            ),
            collection_name=(
                COLLECTION_NAME
            ),
            clean_run=True,
        )
    )

    result = pipeline.run(
        video_path=SOURCE_VIDEO
    )

    print(
        "\nValidating final pipeline "
        "outputs..."
    )

    assert (
        result[
            "success"
        ]
        is True
    )

    assert (
        result[
            "stage_count"
        ]
        == 16
    )

    assert len(
        result[
            "completed_stages"
        ]
    ) == 16

    assert (
        result[
            "summary"
        ][
            "unique_tracks"
        ]
        > 0
    )

    assert (
        result[
            "summary"
        ][
            "composite_events"
        ]
        > 0
    )

    assert (
        result[
            "summary"
        ][
            "accepted_events"
        ]
        > 0
    )

    assert (
        result[
            "summary"
        ][
            "retrieval_documents"
        ]
        > 0
    )

    assert (
        result[
            "summary"
        ][
            "indexed_records"
        ]
        > 0
    )

    validate_file(
        result[
            "important_outputs"
        ][
            "track_metadata"
        ],
        "Track metadata",
    )

    validate_file(
        result[
            "important_outputs"
        ][
            "composite_events"
        ],
        "Composite events",
    )

    validate_file(
        result[
            "important_outputs"
        ][
            "retrieval_documents"
        ],
        "Retrieval documents",
    )

    validate_file(
        result[
            "important_outputs"
        ][
            "embeddings_file"
        ],
        "Embeddings file",
    )

    validate_file(
        result[
            "important_outputs"
        ][
            "embedding_metadata"
        ],
        "Embedding metadata",
    )

    manifest_path = validate_file(
        result[
            "manifest_path"
        ],
        "Pipeline manifest",
    )

    with open(
        manifest_path,
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(
            file
        )

    assert (
        manifest[
            "success"
        ]
        is True
    )

    assert (
        manifest[
            "stage_count"
        ]
        == 16
    )

    assert (
        manifest[
            "summary"
        ][
            "indexed_records"
        ]
        == result[
            "summary"
        ][
            "indexed_records"
        ]
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "PHASE 3 PRODUCTION PIPELINE TEST PASSED"
    )

    print(
        "=" * 78
    )

    print(
        "Completed stages    :",
        len(
            result[
                "completed_stages"
            ]
        ),
    )

    print(
        "Unique tracks       :",
        result[
            "summary"
        ][
            "unique_tracks"
        ],
    )

    print(
        "Composite events    :",
        result[
            "summary"
        ][
            "composite_events"
        ],
    )

    print(
        "Accepted events     :",
        result[
            "summary"
        ][
            "accepted_events"
        ],
    )

    print(
        "Indexed records     :",
        result[
            "summary"
        ][
            "indexed_records"
        ],
    )

    print(
        "Collection count    :",
        result[
            "summary"
        ][
            "collection_count"
        ],
    )

    print(
        "Manifest            :",
        result[
            "manifest_path"
        ],
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()
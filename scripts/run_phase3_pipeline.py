import argparse
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


def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete Phase 3 "
            "AI video intelligence pipeline."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
        help=(
            "Path to the source video."
        ),
    )

    parser.add_argument(
        "--config",
        default=(
            "ai/configs/phase3_config.yaml"
        ),
        help=(
            "Path to the Phase 3 "
            "configuration file."
        ),
    )

    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Optional custom output root."
        ),
    )

    parser.add_argument(
        "--database",
        default=(
            "database/chromadb/phase3_events"
        ),
        help=(
            "ChromaDB database directory."
        ),
    )

    parser.add_argument(
        "--collection",
        default="event_vectors",
        help=(
            "ChromaDB collection name."
        ),
    )

    parser.add_argument(
        "--keep-existing-run",
        action="store_true",
        help=(
            "Do not delete an existing "
            "production run folder."
        ),
    )

    return parser


def resolve_project_path(
    value,
):
    if value is None:
        return None

    path = Path(
        value
    )

    if not path.is_absolute():
        path = (
            PROJECT_ROOT
            / path
        )

    return path.resolve()


def main():

    parser = build_parser()

    args = parser.parse_args()

    video_path = resolve_project_path(
        args.video
    )

    config_path = resolve_project_path(
        args.config
    )

    output_root = resolve_project_path(
        args.output_root
    )

    database_directory = (
        resolve_project_path(
            args.database
        )
    )

    print(
        "\nPreparing Phase 3 "
        "production pipeline..."
    )

    pipeline = (
        Phase3ProductionPipeline(
            project_root=PROJECT_ROOT,
            config_path=config_path,
            output_root=output_root,
            database_directory=(
                database_directory
            ),
            collection_name=(
                args.collection
            ),
            clean_run=(
                not args.keep_existing_run
            ),
        )
    )

    try:
        result = pipeline.run(
            video_path=video_path
        )

    except KeyboardInterrupt:

        print(
            "\nPipeline cancelled by user."
        )

        raise SystemExit(
            130
        )

    except Exception as error:

        print(
            "\n"
            + "=" * 78
        )

        print(
            "PHASE 3 PIPELINE FAILED"
        )

        print(
            "=" * 78
        )

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        print(
            "=" * 78
        )

        raise

    print(
        "\nPipeline result:"
    )

    print(
        "Success       :",
        result[
            "success"
        ],
    )

    print(
        "Source video  :",
        result[
            "source_video"
        ],
    )

    print(
        "Run folder    :",
        result[
            "run_root"
        ],
    )

    print(
        "Manifest      :",
        result[
            "manifest_path"
        ],
    )

    print(
        "\nThe video is now indexed "
        "and ready for natural-language "
        "retrieval."
    )


if __name__ == "__main__":
    main()
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


from ai.retrieval.video_clip_generator import (
    VideoClipGenerator,
)


RETRIEVAL_RESPONSE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "retrieval_response_builder_test"
    / "retrieval_responses.json"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "video_clip_generator_test"
)

CLIPS_DIRECTORY = (
    OUTPUT_DIRECTORY
    / "clips"
)

METADATA_FILE = (
    OUTPUT_DIRECTORY
    / "clip_generation_metadata.json"
)


VIDEO_SEARCH_DIRECTORIES = [
    PROJECT_ROOT / "videos",
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "inputs",
    PROJECT_ROOT / "input",
    PROJECT_ROOT / "test_videos",
    PROJECT_ROOT / "samples",
    PROJECT_ROOT,
]


VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".mpeg",
    ".mpg",
    ".m4v",
}


TARGET_QUERY = (
    "blue bus between 5 and 8 seconds"
)


def find_source_video():

    ignored_directories = {
        "venv",
        ".venv",
        ".git",
        "node_modules",
        "outputs",
        "database",
    }

    candidates = []

    for directory in (
        VIDEO_SEARCH_DIRECTORIES
    ):

        if not directory.exists():
            continue

        if directory == PROJECT_ROOT:

            for path in directory.iterdir():

                if (
                    path.is_file()
                    and path.suffix.lower()
                    in VIDEO_EXTENSIONS
                ):
                    candidates.append(
                        path.resolve()
                    )

            continue

        for path in directory.rglob("*"):

            if not path.is_file():
                continue

            if path.suffix.lower() not in (
                VIDEO_EXTENSIONS
            ):
                continue

            if any(
                ignored_name
                in path.parts
                for ignored_name
                in ignored_directories
            ):
                continue

            candidates.append(
                path.resolve()
            )

    unique_candidates = []

    seen_paths = set()

    for candidate in candidates:

        candidate_key = str(
            candidate
        ).lower()

        if candidate_key in seen_paths:
            continue

        seen_paths.add(
            candidate_key
        )

        unique_candidates.append(
            candidate
        )

    if not unique_candidates:
        raise FileNotFoundError(
            "\nNo source video was found.\n"
            "Place the original test video in one "
            "of these folders:\n"
            "  videos\\\n"
            "  inputs\\\n"
            "  test_videos\\\n"
            "  samples\\\n"
            "Then run this script again."
        )

    unique_candidates.sort(
        key=lambda path: (
            path.stat().st_mtime
        ),
        reverse=True,
    )

    print(
        "\nDetected source videos:"
    )

    for index, path in enumerate(
        unique_candidates,
        start=1,
    ):
        print(
            f"{index}. {path}"
        )

    selected_video = (
        unique_candidates[0]
    )

    print(
        "\nAutomatically selected:"
    )

    print(
        selected_video
    )

    return selected_video


def load_target_response():

    if not RETRIEVAL_RESPONSE_FILE.exists():
        raise FileNotFoundError(
            "Step 25 retrieval response file "
            "was not found:\n"
            f"{RETRIEVAL_RESPONSE_FILE}"
        )

    with open(
        RETRIEVAL_RESPONSE_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(
            file
        )

    responses = data.get(
        "responses",
        [],
    )

    for response in responses:

        if response.get(
            "query"
        ) == TARGET_QUERY:
            return response

    raise ValueError(
        "Target blue bus query was not "
        "found in the Step 25 output."
    )


def print_clip_result(
    clip,
):

    print(
        "\n"
        + "-" * 70
    )

    print(
        f"Event ID       : "
        f"{clip['event_id']}"
    )

    print(
        f"Status         : "
        f"{clip['status']}"
    )

    print(
        f"Source Video   : "
        f"{clip['source_video_name']}"
    )

    print(
        f"Event Time     : "
        f"{clip['event_start_time_seconds']:.2f}s"
        f" -> "
        f"{clip['event_end_time_seconds']:.2f}s"
    )

    print(
        f"Clip Time      : "
        f"{clip['clip_start_time_seconds']:.2f}s"
        f" -> "
        f"{clip['clip_end_time_seconds']:.2f}s"
    )

    print(
        f"Clip Timestamp : "
        f"{clip['clip_start_timestamp']}"
        f" -> "
        f"{clip['clip_end_timestamp']}"
    )

    print(
        f"Frames Written : "
        f"{clip['frames_written']}"
    )

    print(
        f"FPS            : "
        f"{clip['fps']:.3f}"
    )

    print(
        f"Resolution     : "
        f"{clip['width']}"
        f"x"
        f"{clip['height']}"
    )

    print(
        f"Codec          : "
        f"{clip['codec']}"
    )

    print(
        f"Clip Duration  : "
        f"{clip['actual_clip_duration_seconds']:.2f}s"
    )

    print(
        f"Clip Path      : "
        f"{clip['clip_path']}"
    )


def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "VIDEO EVIDENCE CLIP GENERATION TEST"
    )

    print(
        "=" * 70
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_video = (
        find_source_video()
    )

    retrieval_response = (
        load_target_response()
    )

    print(
        "\nTarget Query:"
    )

    print(
        retrieval_response[
            "query"
        ]
    )

    print(
        "\nRetrieved Events:"
    )

    print(
        retrieval_response[
            "result_count"
        ]
    )

    generator = VideoClipGenerator(
        output_directory=(
            CLIPS_DIRECTORY
        ),
        context_before_seconds=2.0,
        context_after_seconds=2.0,
        minimum_clip_duration_seconds=1.0,
        overwrite=True,
    )

    batch_result = (
        generator.generate_from_retrieval_response(
            video_path=(
                source_video
            ),
            retrieval_response=(
                retrieval_response
            ),
            maximum_clips=5,
        )
    )

    for clip in batch_result[
        "clips"
    ]:
        print_clip_result(
            clip
        )

    if batch_result[
        "failures"
    ]:

        print(
            "\nFAILED CLIPS"
        )

        print(
            "-" * 70
        )

        for failure in batch_result[
            "failures"
        ]:

            print(
                f"{failure['event_id']} "
                f"-> "
                f"{failure['error']}"
            )

    metadata = {
        "test_name": (
            "video_clip_generator"
        ),
        "source_video": str(
            source_video
        ),
        "target_query": (
            TARGET_QUERY
        ),
        "batch_result": (
            batch_result
        ),
    }

    generator.save_metadata(
        metadata=metadata,
        output_path=(
            METADATA_FILE
        ),
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "VIDEO CLIP GENERATION TEST"
    )

    print(
        "=" * 70
    )

    print(
        f"Requested events : "
        f"{batch_result['requested_events']}"
    )

    print(
        f"Generated clips  : "
        f"{batch_result['generated_clip_count']}"
    )

    print(
        f"Failed clips     : "
        f"{batch_result['failed_clip_count']}"
    )

    print(
        f"Clips directory  : "
        f"{CLIPS_DIRECTORY}"
    )

    print(
        f"Metadata file    : "
        f"{METADATA_FILE}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
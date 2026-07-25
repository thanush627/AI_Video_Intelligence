import json
import traceback
from pathlib import Path

from ai.pipeline.phase6_pipeline import Phase6Pipeline


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_DIRECTORY = "database/chromadb"
COLLECTION_NAME = "image_embeddings"
CLIP_OUTPUT_DIRECTORY = "clips"

VIDEO_PATH = VIDEO_PATH = r"C:\Users\thanu\AI_Video_Intelligence\test_videos\test.mp4"

QUERY = "person wearing red shirt"

TOP_K = 5
MAXIMUM_CLIPS = 5
GENERATE_CLIPS = True

OUTPUT_DIRECTORY = "outputs"
OUTPUT_JSON = f"{OUTPUT_DIRECTORY}/phase6_result.json"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("PHASE 6 END-TO-END TEST")
    print("=" * 80)

    try:

        pipeline = Phase6Pipeline(
            database_directory=DATABASE_DIRECTORY,
            collection_name=COLLECTION_NAME,
            clip_output_directory=CLIP_OUTPUT_DIRECTORY,
        )

        print("\nPipeline initialized successfully.\n")

        print("Health Check")
        print("-" * 80)

        health = pipeline.health()

        print(json.dumps(
            health,
            indent=4,
        ))

        print("\nRunning Retrieval...\n")

        result = pipeline.run(
            query=QUERY,
            video_path=VIDEO_PATH,
            output_json=OUTPUT_JSON,
            top_k=TOP_K,
            maximum_clips=MAXIMUM_CLIPS,
            generate_clips=GENERATE_CLIPS,
        )

        pipeline.print_summary(result)

        print("\nDetailed Result\n")
        print(json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ))

        print("\nJSON exported to")
        print(Path(OUTPUT_JSON).resolve())

        print("\nTEST PASSED")

    except Exception as e:

        print("\nTEST FAILED\n")

        print(type(e).__name__)
        print(e)

        traceback.print_exc()


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()
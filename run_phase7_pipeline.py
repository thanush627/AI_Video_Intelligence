import os
import sys

# Ensure root directory is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.pipeline.phase7_pipeline import Phase7Pipeline

def main():
    print("="*60)
    print("      AI VIDEO INTELLIGENCE - PHASE 7 RETRIEVAL TEST")
    print("="*60)

    test_video = "test_videos/test.mp4" if os.path.exists("test_videos/test.mp4") else None

    # Initialize Phase 7 Pipeline
    pipeline = Phase7Pipeline(config_path="ai/configs/phase7.yaml")

    test_queries = [
        "find bicycle",
        "bicycle",
        "bus"
    ]

    for q in test_queries:
        output = pipeline.query(natural_language_query=q, video_path=test_video)
        print("\n--- QUERY RESULT ---")
        print(output["final_response"])
        if output.get("clip_paths"):
            print("Extracted Clips:", output["clip_paths"])
        print("-" * 60)

if __name__ == "__main__":
    main()
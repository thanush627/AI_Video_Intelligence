import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from ai.retrieval.end_to_end_retrieval_pipeline import (
    EndToEndRetrievalPipeline,
)


class Phase6Pipeline:
    """
    Phase 6 Production Pipeline

    Responsibilities
    ----------------
    1. Initialize retrieval system
    2. Validate inputs
    3. Execute retrieval
    4. Save results
    5. Return standardized response
    """

    def __init__(
        self,
        database_directory,
        collection_name,
        clip_output_directory,
        context_before_seconds=2.0,
        context_after_seconds=2.0,
        minimum_clip_duration_seconds=1.0,
        overwrite_clips=True,
    ):

        self.database_directory = Path(database_directory).resolve()
        self.collection_name = collection_name
        self.clip_output_directory = Path(
            clip_output_directory
        ).resolve()

        self.logger = self._create_logger()

        self.logger.info("Initializing Phase 6 Pipeline...")

        self._validate_paths()

        self.pipeline = EndToEndRetrievalPipeline(
            database_directory=self.database_directory,
            collection_name=self.collection_name,
            clip_output_directory=self.clip_output_directory,
            context_before_seconds=context_before_seconds,
            context_after_seconds=context_after_seconds,
            minimum_clip_duration_seconds=minimum_clip_duration_seconds,
            overwrite_clips=overwrite_clips,
        )

        self.logger.info("Phase 6 Pipeline Ready.")

    #########################################################
    # LOGGER
    #########################################################

    def _create_logger(self):

        logger = logging.getLogger("Phase6Pipeline")

        if logger.handlers:
            return logger

        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s : %(message)s"
        )

        console = logging.StreamHandler()
        console.setFormatter(formatter)

        logger.addHandler(console)

        return logger

    #########################################################
    # VALIDATION
    #########################################################

    def _validate_paths(self):

        if not self.database_directory.exists():

            raise FileNotFoundError(
                f"Database directory not found:\n"
                f"{self.database_directory}"
            )

        self.clip_output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not str(self.collection_name).strip():

            raise ValueError(
                "Collection name cannot be empty."
            )

    #########################################################
    # HEALTH CHECK
    #########################################################

    def health(self):

        return {
            "status": "healthy",
            "pipeline": "Phase6",
            "database_directory": str(
                self.database_directory
            ),
            "collection_name": self.collection_name,
            "clip_output_directory": str(
                self.clip_output_directory
            ),
        }

    #########################################################
    # INPUT VALIDATION
    #########################################################

    def _validate_query(self, query):

        query = str(query).strip()

        if len(query) == 0:

            raise ValueError(
                "Query cannot be empty."
            )

        return query

    def _validate_video(self, video_path):

        video_path = Path(video_path).resolve()

        if not video_path.exists():

            raise FileNotFoundError(
                f"Video not found:\n{video_path}"
            )

        if not video_path.is_file():

            raise ValueError(
                "Video path must point to a file."
            )

        return video_path

    #########################################################
    # SAVE RESULT
    #########################################################

    def save_result(
        self,
        result,
        output_json,
    ):

        output_json = Path(output_json)

        output_json.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_json,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                result,
                f,
                indent=2,
                ensure_ascii=False,
            )

        self.logger.info(
            f"Saved result to {output_json}"
        )

        return output_json

        #########################################################
    # SEARCH
    #########################################################

    def search(
        self,
        query,
        video_path,
        top_k=5,
        maximum_clips=None,
        generate_clips=True,
    ):
        """
        Execute a semantic search using the
        EndToEndRetrievalPipeline.
        """

        start = time.perf_counter()

        query = self._validate_query(query)

        if generate_clips:
            video_path = self._validate_video(video_path)

        self.logger.info(f"Query : {query}")

        result = self.pipeline.run(
            query=query,
            video_path=video_path,
            top_k=top_k,
            maximum_clips=maximum_clips,
            generate_clips=generate_clips,
        )

        elapsed = round(
            time.perf_counter() - start,
            4,
        )

        result["phase6_processing_time"] = elapsed

        self.logger.info(
            f"Completed in {elapsed} sec"
        )

        return result

    #########################################################
    # BATCH SEARCH
    #########################################################

    def batch_search(
        self,
        queries,
        video_path,
        top_k=5,
        maximum_clips=None,
        generate_clips=True,
    ):

        outputs = []

        for query in queries:

            try:

                outputs.append(
                    self.search(
                        query=query,
                        video_path=video_path,
                        top_k=top_k,
                        maximum_clips=maximum_clips,
                        generate_clips=generate_clips,
                    )
                )

            except Exception as e:

                outputs.append(
                    {
                        "success": False,
                        "query": query,
                        "error": str(e),
                    }
                )

        return outputs

    #########################################################
    # RUN
    #########################################################

    def run(
        self,
        query,
        video_path,
        output_json=None,
        top_k=5,
        maximum_clips=None,
        generate_clips=True,
    ):

        result = self.search(
            query=query,
            video_path=video_path,
            top_k=top_k,
            maximum_clips=maximum_clips,
            generate_clips=generate_clips,
        )

        if output_json is not None:

            self.save_result(
                result,
                output_json,
            )

        return result

    #########################################################
    # PRINT SUMMARY
    #########################################################

    def print_summary(
        self,
        result,
    ):

        print("\n" + "=" * 70)
        print("PHASE 6 RETRIEVAL SUMMARY")
        print("=" * 70)

        print(
            f"Query : {result.get('query')}"
        )

        summary = result.get(
            "summary",
            {}
        )

        print(
            f"Match Found : {summary.get('match_found')}"
        )

        print(
            f"Retrieved Events : "
            f"{summary.get('retrieved_event_count')}"
        )

        print(
            f"Generated Clips : "
            f"{summary.get('generated_clip_count')}"
        )

        print(
            f"Processing Time : "
            f"{summary.get('processing_time_seconds')} sec"
        )

        print("=" * 70)

    #########################################################
    # EXPORT
    #########################################################

    def export_results(
        self,
        result,
        output_directory,
    ):

        output_directory = Path(
            output_directory
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_file = (
            output_directory /
            "retrieval_result.json"
        )

        self.save_result(
            result,
            json_file,
        )

        return json_file
    
if __name__ == "__main__":

    pipeline = Phase6Pipeline(
        database_directory="database/chromadb",
        collection_name="video_events",
        clip_output_directory="clips",
    )

    result = pipeline.run(
        query="person wearing red shirt",
        video_path="videos/sample.mp4",
        output_json="outputs/result.json",
        top_k=5,
        maximum_clips=5,
        generate_clips=True,
    )

    pipeline.print_summary(result)

    print("\nFinished Successfully.")
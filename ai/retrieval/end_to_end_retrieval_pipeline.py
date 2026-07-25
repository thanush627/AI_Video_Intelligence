import json
import math
import time
from pathlib import Path

from ai.retrieval.semantic_event_search import (
    SemanticEventSearch,
)

from ai.retrieval.query_orchestrator import (
    QueryOrchestrator,
)

from ai.retrieval.retrieval_response_builder import (
    RetrievalResponseBuilder,
)

from ai.retrieval.video_clip_generator import (
    VideoClipGenerator,
)


class EndToEndRetrievalPipeline:

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
        self.database_directory = Path(
            database_directory
        ).resolve()

        self.collection_name = str(
            collection_name
        ).strip()

        self.clip_output_directory = Path(
            clip_output_directory
        ).resolve()

        self.clip_output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.collection_name:
            raise ValueError(
                "collection_name cannot be empty."
            )

        print(
            "\nInitializing end-to-end "
            "retrieval pipeline..."
        )

        self.search_engine = (
            SemanticEventSearch(
                database_directory=(
                    self.database_directory
                ),
                collection_name=(
                    self.collection_name
                ),
            )
        )

        self.orchestrator = (
            QueryOrchestrator(
                search_engine=(
                    self.search_engine
                )
            )
        )

        self.response_builder = (
            RetrievalResponseBuilder()
        )

        self.clip_generator = (
            VideoClipGenerator(
                output_directory=(
                    self.clip_output_directory
                ),
                context_before_seconds=(
                    context_before_seconds
                ),
                context_after_seconds=(
                    context_after_seconds
                ),
                minimum_clip_duration_seconds=(
                    minimum_clip_duration_seconds
                ),
                overwrite=(
                    overwrite_clips
                ),
            )
        )

        print(
            "End-to-end retrieval pipeline ready."
        )


    def _safe_float(
        self,
        value,
        default=0.0,
    ):
        try:
            value = float(
                value
            )

            if not math.isfinite(
                value
            ):
                return default

            return value

        except (
            TypeError,
            ValueError,
        ):
            return default


    def _safe_int(
        self,
        value,
        default=0,
    ):
        try:
            return int(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            return default


    def _clean_query(
        self,
        query,
    ):
        query = str(
            query
        ).strip()

        query = " ".join(
            query.split()
        )

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        return query


    def _attach_clips_to_results(
        self,
        retrieval_response,
        clip_batch_result,
    ):
        results = retrieval_response.get(
            "results",
            [],
        )

        clips = clip_batch_result.get(
            "clips",
            [],
        )

        failures = clip_batch_result.get(
            "failures",
            [],
        )

        clips_by_event_id = {
            str(
                clip.get(
                    "event_id",
                    "",
                )
            ): clip
            for clip in clips
        }

        failures_by_event_id = {
            str(
                failure.get(
                    "event_id",
                    "",
                )
            ): failure
            for failure in failures
        }

        updated_results = []

        for result in results:

            updated_result = dict(
                result
            )

            event_id = str(
                result.get(
                    "event_id",
                    "",
                )
            )

            clip = clips_by_event_id.get(
                event_id
            )

            failure = (
                failures_by_event_id.get(
                    event_id
                )
            )

            if clip is not None:

                updated_result[
                    "evidence_clip"
                ] = {
                    "available": True,
                    "status": clip.get(
                        "status"
                    ),
                    "clip_path": clip.get(
                        "clip_path"
                    ),
                    "clip_filename": (
                        clip.get(
                            "clip_filename"
                        )
                    ),
                    "clip_start_time_seconds": (
                        clip.get(
                            "clip_start_time_seconds"
                        )
                    ),
                    "clip_end_time_seconds": (
                        clip.get(
                            "clip_end_time_seconds"
                        )
                    ),
                    "clip_duration_seconds": (
                        clip.get(
                            "actual_clip_duration_seconds"
                        )
                    ),
                    "clip_start_timestamp": (
                        clip.get(
                            "clip_start_timestamp"
                        )
                    ),
                    "clip_end_timestamp": (
                        clip.get(
                            "clip_end_timestamp"
                        )
                    ),
                }

            elif failure is not None:

                updated_result[
                    "evidence_clip"
                ] = {
                    "available": False,
                    "status": "failed",
                    "error": failure.get(
                        "error"
                    ),
                }

            else:

                updated_result[
                    "evidence_clip"
                ] = {
                    "available": False,
                    "status": (
                        "not_requested"
                    ),
                }

            updated_results.append(
                updated_result
            )

        retrieval_response[
            "results"
        ] = updated_results

        if updated_results:

            retrieval_response[
                "best_match"
            ] = dict(
                updated_results[0]
            )

        else:

            retrieval_response[
                "best_match"
            ] = None

        return retrieval_response


    def _build_pipeline_summary(
        self,
        query,
        retrieval_response,
        clip_batch_result,
        processing_time_seconds,
    ):
        result_count = (
            retrieval_response.get(
                "result_count",
                0,
            )
        )

        generated_clip_count = (
            clip_batch_result.get(
                "generated_clip_count",
                0,
            )
        )

        failed_clip_count = (
            clip_batch_result.get(
                "failed_clip_count",
                0,
            )
        )

        return {
            "query": query,
            "match_found": (
                result_count > 0
            ),
            "retrieved_event_count": (
                result_count
            ),
            "generated_clip_count": (
                generated_clip_count
            ),
            "failed_clip_count": (
                failed_clip_count
            ),
            "processing_time_seconds": round(
                processing_time_seconds,
                4,
            ),
        }


    def run(
        self,
        query,
        video_path,
        top_k=5,
        maximum_clips=None,
        generate_clips=True,
    ):
        pipeline_start_time = (
            time.perf_counter()
        )

        query = self._clean_query(
            query
        )

        video_path = Path(
            video_path
        ).resolve()

        if generate_clips:

            if not video_path.exists():
                raise FileNotFoundError(
                    "Source video does not exist: "
                    f"{video_path}"
                )

            if not video_path.is_file():
                raise ValueError(
                    "Source video path is not "
                    "a file."
                )

        top_k = max(
            1,
            self._safe_int(
                top_k,
                default=5,
            ),
        )

        if maximum_clips is None:
            maximum_clips = top_k

        maximum_clips = max(
            0,
            self._safe_int(
                maximum_clips,
                default=top_k,
            ),
        )

        stage_timings = {}

        search_start_time = (
            time.perf_counter()
        )

        orchestrator_response = (
            self.orchestrator.search(
                query=query,
                top_k=top_k,
            )
        )

        stage_timings[
            "retrieval_seconds"
        ] = round(
            time.perf_counter()
            - search_start_time,
            4,
        )

        response_start_time = (
            time.perf_counter()
        )

        retrieval_response = (
            self.response_builder.build(
                orchestrator_response
            )
        )

        stage_timings[
            "response_building_seconds"
        ] = round(
            time.perf_counter()
            - response_start_time,
            4,
        )

        if (
            generate_clips
            and retrieval_response.get(
                "match_found",
                False,
            )
            and maximum_clips > 0
        ):

            clip_start_time = (
                time.perf_counter()
            )

            clip_batch_result = (
                self.clip_generator
                .generate_from_retrieval_response(
                    video_path=(
                        video_path
                    ),
                    retrieval_response=(
                        retrieval_response
                    ),
                    maximum_clips=(
                        maximum_clips
                    ),
                )
            )

            stage_timings[
                "clip_generation_seconds"
            ] = round(
                time.perf_counter()
                - clip_start_time,
                4,
            )

        else:

            clip_batch_result = {
                "success": True,
                "query": query,
                "requested_events": 0,
                "generated_clip_count": 0,
                "failed_clip_count": 0,
                "clips": [],
                "failures": [],
            }

            stage_timings[
                "clip_generation_seconds"
            ] = 0.0

        retrieval_response = (
            self._attach_clips_to_results(
                retrieval_response=(
                    retrieval_response
                ),
                clip_batch_result=(
                    clip_batch_result
                ),
            )
        )

        processing_time_seconds = (
            time.perf_counter()
            - pipeline_start_time
        )

        stage_timings[
            "total_seconds"
        ] = round(
            processing_time_seconds,
            4,
        )

        pipeline_summary = (
            self._build_pipeline_summary(
                query=query,
                retrieval_response=(
                    retrieval_response
                ),
                clip_batch_result=(
                    clip_batch_result
                ),
                processing_time_seconds=(
                    processing_time_seconds
                ),
            )
        )

        success = (
            retrieval_response.get(
                "success",
                False,
            )
            and (
                clip_batch_result.get(
                    "failed_clip_count",
                    0,
                )
                == 0
            )
        )

        return {
            "success": success,
            "pipeline": (
                "end_to_end_event_retrieval"
            ),
            "source_video": (
                str(video_path)
            ),
            "query": query,
            "summary": (
                pipeline_summary
            ),
            "timings": (
                stage_timings
            ),
            "retrieval": (
                retrieval_response
            ),
            "clip_generation": (
                clip_batch_result
            ),
        }


    def save_result(
        self,
        result,
        output_path,
    ):
        output_path = Path(
            output_path
        ).resolve()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                result,
                file,
                indent=2,
                ensure_ascii=False,
            )

        return output_path
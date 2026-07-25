import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from ai.preprocessing.video_analyzer import (
    VideoAnalyzer,
)

from ai.preprocessing.adaptive_sampler import (
    AdaptiveFrameSampler,
)

from ai.preprocessing.frame_extractor import (
    FrameExtractor,
)

from ai.inference.sampled_frame_detector import (
    SampledFrameDetector,
)

from ai.pipeline.object_tracker import (
    ObjectTracker,
)

from ai.pipeline.representative_selector import (
    RepresentativeCropSelector,
)

from ai.pipeline.track_reliability import (
    TrackReliabilityFilter,
)

from ai.pipeline.color_analyzer import (
    ColorAnalyzer,
)

from ai.event_generation.motion_analyzer import (
    MotionAnalyzer,
)

from ai.event_generation.spatial_analyzer import (
    SpatialAnalyzer,
)

from ai.event_generation.relationship_analyzer import (
    RelationshipAnalyzer,
)

from ai.event_generation.atomic_event_generator import (
    AtomicEventGenerator,
)

from ai.event_generation.composite_event_generator import (
    CompositeEventGenerator,
)

from ai.retrieval.event_quality_filter import (
    EventQualityFilter,
)

from ai.embeddings.event_embedder import (
    EventEmbedder,
)

from ai.embeddings.chromadb_store import (
    ChromaDBEventStore,
)


class Phase3ProductionPipeline:

    STAGE_NAMES = [
        "video_analysis",
        "adaptive_sampling",
        "frame_extraction",
        "sampled_frame_detection",
        "object_tracking",
        "representative_selection",
        "track_reliability",
        "color_analysis",
        "motion_analysis",
        "spatial_analysis",
        "relationship_analysis",
        "atomic_event_generation",
        "composite_event_generation",
        "event_quality_filter",
        "event_embeddings",
        "chromadb_storage",
    ]

    def __init__(
        self,
        project_root,
        config_path=None,
        output_root=None,
        database_directory=None,
        collection_name="event_vectors",
        clean_run=True,
    ):
        self.project_root = Path(
            project_root
        ).resolve()

        if config_path is None:
            config_path = (
                self.project_root
                / "ai"
                / "configs"
                / "phase3_config.yaml"
            )

        self.config_path = Path(
            config_path
        ).resolve()

        if not self.config_path.exists():
            raise FileNotFoundError(
                "Phase 3 configuration file "
                "was not found:\n"
                f"{self.config_path}"
            )

        with open(
            self.config_path,
            "r",
            encoding="utf-8",
        ) as file:
            self.config = (
                yaml.safe_load(file)
                or {}
            )

        if output_root is None:
            configured_output_root = (
                self.config
                .get(
                    "paths",
                    {},
                )
                .get(
                    "output_root",
                    "outputs/phase3",
                )
            )

            output_root = (
                self.project_root
                / configured_output_root
            )

        self.output_root = Path(
            output_root
        ).resolve()

        if database_directory is None:
            database_directory = (
                self.project_root
                / "database"
                / "chromadb"
                / "phase3_events"
            )

        self.database_directory = Path(
            database_directory
        ).resolve()

        self.collection_name = str(
            collection_name
        ).strip()

        if not self.collection_name:
            raise ValueError(
                "collection_name cannot be empty."
            )

        self.clean_run = bool(
            clean_run
        )

        self.run_root = None

        self.stage_directories = {}

        self.video_path = None

        self.video_metadata = {}

        self.source_fps = None

        self.frame_width = None

        self.frame_height = None

        self.total_frames = None

        self.video_duration_seconds = None

        self.results = {}

        self.timings = {}

        self.pipeline_start_time = None


    def _safe_float(
        self,
        value,
        default=None,
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
        default=None,
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


    def _slugify(
        self,
        value,
    ):
        value = str(
            value
        ).strip().lower()

        cleaned = []

        for character in value:

            if character.isalnum():
                cleaned.append(
                    character
                )

            elif character in (
                "-",
                "_",
                " ",
            ):
                cleaned.append(
                    "_"
                )

        slug = "".join(
            cleaned
        )

        while "__" in slug:
            slug = slug.replace(
                "__",
                "_",
            )

        slug = slug.strip(
            "_"
        )

        if not slug:
            slug = "video"

        return slug


    def _get_config(
        self,
        section,
        key,
        default=None,
    ):
        section_data = (
            self.config.get(
                section,
                {},
            )
        )

        return section_data.get(
            key,
            default,
        )


    def _resolve_model_path(
        self,
    ):
        model_value = (
            self.config
            .get(
                "paths",
                {},
            )
            .get(
                "model"
            )
        )

        if not model_value:
            raise KeyError(
                "Missing paths.model in "
                "phase3_config.yaml"
            )

        model_path = Path(
            model_value
        )

        if not model_path.is_absolute():
            model_path = (
                self.project_root
                / model_path
            )

        model_path = (
            model_path.resolve()
        )

        if not model_path.exists():
            raise FileNotFoundError(
                "YOLO model was not found:\n"
                f"{model_path}"
            )

        return model_path


    def _prepare_run(
        self,
        video_path,
    ):
        self.video_path = Path(
            video_path
        ).resolve()

        if not self.video_path.exists():
            raise FileNotFoundError(
                "Source video was not found:\n"
                f"{self.video_path}"
            )

        if not self.video_path.is_file():
            raise ValueError(
                "Source video path is not a file."
            )

        video_name = self._slugify(
            self.video_path.stem
        )

        self.run_root = (
            self.output_root
            / "production_runs"
            / video_name
        )

        if (
            self.clean_run
            and self.run_root.exists()
        ):
            shutil.rmtree(
                self.run_root
            )

        self.run_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        directory_names = {
            "frame_extraction": (
                "01_frame_extraction"
            ),
            "sampled_frame_detection": (
                "02_sampled_frame_detection"
            ),
            "object_tracking": (
                "03_object_tracking"
            ),
            "representative_selection": (
                "04_representative_selection"
            ),
            "track_reliability": (
                "05_track_reliability"
            ),
            "color_analysis": (
                "06_color_analysis"
            ),
            "motion_analysis": (
                "07_motion_analysis"
            ),
            "spatial_analysis": (
                "08_spatial_analysis"
            ),
            "relationship_analysis": (
                "09_relationship_analysis"
            ),
            "atomic_event_generation": (
                "10_atomic_event_generation"
            ),
            "composite_event_generation": (
                "11_composite_event_generation"
            ),
            "event_quality_filter": (
                "12_event_quality_filter"
            ),
            "event_embeddings": (
                "13_event_embeddings"
            ),
        }

        self.stage_directories = {}

        for (
            stage_name,
            directory_name,
        ) in directory_names.items():

            directory = (
                self.run_root
                / directory_name
            )

            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.stage_directories[
                stage_name
            ] = directory


    def _read_video_properties(
        self,
    ):
        capture = cv2.VideoCapture(
            str(self.video_path)
        )

        if not capture.isOpened():
            raise RuntimeError(
                "OpenCV could not open video:\n"
                f"{self.video_path}"
            )

        try:
            fps = capture.get(
                cv2.CAP_PROP_FPS
            )

            width = capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )

            height = capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )

            total_frames = capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )

        finally:
            capture.release()

        fps = self._safe_float(
            fps,
            0.0,
        )

        width = self._safe_int(
            width,
            0,
        )

        height = self._safe_int(
            height,
            0,
        )

        total_frames = self._safe_int(
            total_frames,
            0,
        )

        if fps <= 0:
            raise RuntimeError(
                "Invalid source video FPS."
            )

        if (
            width <= 0
            or height <= 0
        ):
            raise RuntimeError(
                "Invalid source video resolution."
            )

        self.source_fps = fps

        self.frame_width = width

        self.frame_height = height

        self.total_frames = total_frames

        if total_frames > 0:
            self.video_duration_seconds = (
                total_frames
                / fps
            )

        else:
            self.video_duration_seconds = (
                0.0
            )


    def _print_stage_header(
        self,
        stage_number,
        title,
    ):
        print(
            "\n"
            + "=" * 78
        )

        print(
            f"STAGE {stage_number:02d}/16 "
            f"- {title}"
        )

        print(
            "=" * 78
        )


    def _run_stage(
        self,
        stage_name,
        stage_number,
        title,
        function,
    ):
        self._print_stage_header(
            stage_number,
            title,
        )

        start_time = (
            time.perf_counter()
        )

        try:
            result = function()

            duration = (
                time.perf_counter()
                - start_time
            )

            self.results[
                stage_name
            ] = result

            self.timings[
                stage_name
            ] = round(
                duration,
                4,
            )

            print(
                f"\nSTATUS: PASSED"
            )

            print(
                f"TIME  : "
                f"{duration:.4f}s"
            )

            return result

        except Exception as error:

            duration = (
                time.perf_counter()
                - start_time
            )

            self.timings[
                stage_name
            ] = round(
                duration,
                4,
            )

            print(
                "\nSTATUS: FAILED"
            )

            print(
                f"ERROR : "
                f"{type(error).__name__}: "
                f"{error}"
            )

            raise RuntimeError(
                f"Phase 3 stage failed: "
                f"{stage_name}"
            ) from error


    def _stage_video_analysis(
        self,
    ):
        analyzer = VideoAnalyzer(
            self.video_path
        )

        metadata = analyzer.analyze()

        self._read_video_properties()

        self.video_metadata = dict(
            metadata
        )

        self.video_metadata.update(
            {
                "source_fps": (
                    self.source_fps
                ),
                "frame_width": (
                    self.frame_width
                ),
                "frame_height": (
                    self.frame_height
                ),
                "total_frames": (
                    self.total_frames
                ),
                "duration_seconds": (
                    self.video_duration_seconds
                ),
            }
        )

        print(
            f"Video      : "
            f"{self.video_path.name}"
        )

        print(
            f"FPS        : "
            f"{self.source_fps:.6f}"
        )

        print(
            f"Resolution : "
            f"{self.frame_width}x"
            f"{self.frame_height}"
        )

        print(
            f"Frames     : "
            f"{self.total_frames}"
        )

        print(
            f"Duration   : "
            f"{self.video_duration_seconds:.2f}s"
        )

        return self.video_metadata


    def _stage_adaptive_sampling(
        self,
    ):
        motion_config = self.config[
            "adaptive_sampling"
        ][
            "motion"
        ]

        sampling_config = self.config[
            "adaptive_sampling"
        ][
            "sampling"
        ]

        sampler = AdaptiveFrameSampler(
            video_path=self.video_path,
            low_threshold=motion_config[
                "low_threshold"
            ],
            high_threshold=motion_config[
                "high_threshold"
            ],
            low_motion_fps=sampling_config[
                "low_motion_fps"
            ],
            medium_motion_fps=sampling_config[
                "medium_motion_fps"
            ],
            high_motion_fps=sampling_config[
                "high_motion_fps"
            ],
        )

        result = sampler.sample()

        print(
            "Source frames  :",
            result[
                "total_source_frames"
            ],
        )

        print(
            "Sampled frames :",
            result[
                "sampled_frame_count"
            ],
        )

        print(
            "Motion counts  :",
            result[
                "motion_counts"
            ],
        )

        return result


    def _stage_frame_extraction(
        self,
    ):
        sampling_result = self.results[
            "adaptive_sampling"
        ]

        output_directory = (
            self.stage_directories[
                "frame_extraction"
            ]
        )

        extractor = FrameExtractor(
            video_path=self.video_path,
            output_dir=output_directory,
        )

        result = extractor.extract(
            sampling_result[
                "sampled_frames"
            ]
        )

        print(
            "Saved frames  :",
            result[
                "saved_frame_count"
            ],
        )

        print(
            "Metadata file :",
            result[
                "metadata_path"
            ],
        )

        return result


    def _stage_sampled_detection(
        self,
    ):
        detection_config = (
            self.config[
                "detection"
            ]
        )

        detector = SampledFrameDetector(
            model_path=(
                self._resolve_model_path()
            ),
            confidence_threshold=(
                detection_config[
                    "confidence_threshold"
                ]
            ),
            iou_threshold=(
                detection_config[
                    "iou_threshold"
                ]
            ),
            image_size=(
                detection_config[
                    "image_size"
                ]
            ),
            device=(
                detection_config[
                    "device"
                ]
            ),
        )

        frame_metadata_path = Path(
            self.results[
                "frame_extraction"
            ][
                "metadata_path"
            ]
        )

        result = detector.detect(
            frame_metadata_path=(
                frame_metadata_path
            ),
            output_dir=(
                self.stage_directories[
                    "sampled_frame_detection"
                ]
            ),
        )

        print(
            "Processed frames :",
            result[
                "processed_frames"
            ],
        )

        print(
            "Total detections :",
            result[
                "total_detections"
            ],
        )

        print(
            "Class counts     :",
            result[
                "class_counts"
            ],
        )

        return result


    def _stage_object_tracking(
        self,
    ):
        detection_config = (
            self.config[
                "detection"
            ]
        )

        tracking_config = (
            self.config[
                "tracking"
            ]
        )

        tracker = ObjectTracker(
            model_path=(
                self._resolve_model_path()
            ),
            confidence_threshold=(
                detection_config[
                    "confidence_threshold"
                ]
            ),
            iou_threshold=(
                detection_config[
                    "iou_threshold"
                ]
            ),
            image_size=(
                detection_config[
                    "image_size"
                ]
            ),
            tracker=(
                tracking_config[
                    "tracker"
                ]
            ),
            device=(
                detection_config[
                    "device"
                ]
            ),
        )

        result = tracker.track(
            video_path=self.video_path,
            output_dir=(
                self.stage_directories[
                    "object_tracking"
                ]
            ),
        )

        print(
            "Processed frames :",
            result[
                "processed_frames"
            ],
        )

        print(
            "Unique tracks    :",
            result[
                "unique_tracks"
            ],
        )

        print(
            "Metadata file    :",
            result[
                "metadata_path"
            ],
        )

        return result


    def _stage_representative_selection(
        self,
    ):
        selection_config = (
            self.config[
                "representative_selection"
            ]
        )

        track_metadata_path = Path(
            self.results[
                "object_tracking"
            ][
                "metadata_path"
            ]
        )

        selector = RepresentativeCropSelector(
            video_path=self.video_path,
            track_metadata_path=(
                track_metadata_path
            ),
            output_dir=(
                self.stage_directories[
                    "representative_selection"
                ]
            ),
            max_crops_per_track=(
                selection_config[
                    "max_crops_per_track"
                ]
            ),
            min_crop_width=(
                selection_config[
                    "min_crop_width"
                ]
            ),
            min_crop_height=(
                selection_config[
                    "min_crop_height"
                ]
            ),
            weights=(
                selection_config[
                    "weights"
                ]
            ),
        )

        result = selector.select()

        print(
            "Input tracks      :",
            result[
                "input_tracks"
            ],
        )

        print(
            "Tracks with crops :",
            result[
                "tracks_with_crops"
            ],
        )

        print(
            "Saved crops       :",
            result[
                "total_saved_crops"
            ],
        )

        return result


    def _stage_track_reliability(
        self,
    ):
        representative_metadata_path = Path(
            self.results[
                "representative_selection"
            ][
                "metadata_path"
            ]
        )

        analyzer = TrackReliabilityFilter(
            representative_metadata_path=(
                representative_metadata_path
            ),
            output_dir=(
                self.stage_directories[
                    "track_reliability"
                ]
            ),
        )

        result = analyzer.analyze()

        print(
            "Total tracks  :",
            result[
                "total_tracks"
            ],
        )

        print(
            "Status counts :",
            result[
                "status_counts"
            ],
        )

        return result


    def _stage_color_analysis(
        self,
    ):
        representative_metadata_path = Path(
            self.results[
                "representative_selection"
            ][
                "metadata_path"
            ]
        )

        analyzer = ColorAnalyzer(
            representative_metadata_path=(
                representative_metadata_path
            ),
            output_dir=(
                self.stage_directories[
                    "color_analysis"
                ]
            ),
        )

        result = analyzer.analyze()

        print(
            "Analyzed tracks :",
            result[
                "analyzed_tracks"
            ],
        )

        print(
            "Unknown tracks  :",
            result[
                "unknown_tracks"
            ],
        )

        return result


    def _stage_motion_analysis(
        self,
    ):
        analyzer = MotionAnalyzer(
            track_metadata_path=Path(
                self.results[
                    "object_tracking"
                ][
                    "metadata_path"
                ]
            ),
            reliability_metadata_path=Path(
                self.results[
                    "track_reliability"
                ][
                    "metadata_path"
                ]
            ),
            output_dir=(
                self.stage_directories[
                    "motion_analysis"
                ]
            ),
            source_fps=(
                self.source_fps
            ),
            frame_width=(
                self.frame_width
            ),
            frame_height=(
                self.frame_height
            ),
            include_review=True,
        )

        result = analyzer.analyze()

        print(
            "Analyzed tracks :",
            result[
                "analyzed_tracks"
            ],
        )

        print(
            "Motion states   :",
            result[
                "motion_state_counts"
            ],
        )

        return result


    def _stage_spatial_analysis(
        self,
    ):
        analyzer = SpatialAnalyzer(
            motion_metadata_path=Path(
                self.results[
                    "motion_analysis"
                ][
                    "metadata_path"
                ]
            ),
            output_dir=(
                self.stage_directories[
                    "spatial_analysis"
                ]
            ),
            frame_width=(
                self.frame_width
            ),
            frame_height=(
                self.frame_height
            ),
        )

        result = analyzer.analyze()

        print(
            "Analyzed tracks :",
            result[
                "analyzed_tracks"
            ],
        )

        print(
            "Skipped tracks  :",
            result[
                "skipped_tracks"
            ],
        )

        return result


    def _stage_relationship_analysis(
        self,
    ):
        analyzer = RelationshipAnalyzer(
            track_metadata_path=Path(
                self.results[
                    "object_tracking"
                ][
                    "metadata_path"
                ]
            ),
            reliability_metadata_path=Path(
                self.results[
                    "track_reliability"
                ][
                    "metadata_path"
                ]
            ),
            motion_metadata_path=Path(
                self.results[
                    "motion_analysis"
                ][
                    "metadata_path"
                ]
            ),
            output_dir=(
                self.stage_directories[
                    "relationship_analysis"
                ]
            ),
            frame_width=(
                self.frame_width
            ),
            frame_height=(
                self.frame_height
            ),
            include_review=True,
            near_threshold=0.08,
            moving_together_threshold=0.06,
            minimum_shared_frames=3,
        )

        result = analyzer.analyze()

        print(
            "Usable tracks          :",
            result[
                "usable_tracks"
            ],
        )

        print(
            "Detected relationships :",
            result[
                "detected_relationships"
            ],
        )

        return result


    def _stage_atomic_events(
        self,
    ):
        generator = AtomicEventGenerator(
            track_metadata_path=Path(
                self.results[
                    "object_tracking"
                ][
                    "metadata_path"
                ]
            ),
            reliability_metadata_path=Path(
                self.results[
                    "track_reliability"
                ][
                    "metadata_path"
                ]
            ),
            color_metadata_path=Path(
                self.results[
                    "color_analysis"
                ][
                    "metadata_path"
                ]
            ),
            output_dir=(
                self.stage_directories[
                    "atomic_event_generation"
                ]
            ),
            source_fps=(
                self.source_fps
            ),
            include_review=True,
        )

        result = generator.generate()

        print(
            "Total events  :",
            result[
                "total_events"
            ],
        )

        print(
            "Usable tracks :",
            result[
                "usable_tracks"
            ],
        )

        return result


    def _stage_composite_events(
        self,
    ):
        generator = CompositeEventGenerator(
            atomic_events_path=Path(
                self.results[
                    "atomic_event_generation"
                ][
                    "metadata_path"
                ]
            ),
            color_metadata_path=Path(
                self.results[
                    "color_analysis"
                ][
                    "metadata_path"
                ]
            ),
            motion_metadata_path=Path(
                self.results[
                    "motion_analysis"
                ][
                    "metadata_path"
                ]
            ),
            spatial_metadata_path=Path(
                self.results[
                    "spatial_analysis"
                ][
                    "metadata_path"
                ]
            ),
            relationship_metadata_path=Path(
                self.results[
                    "relationship_analysis"
                ][
                    "metadata_path"
                ]
            ),
            output_dir=(
                self.stage_directories[
                    "composite_event_generation"
                ]
            ),
        )

        result = generator.generate()

        print(
            "Total events      :",
            result[
                "total_events"
            ],
        )

        print(
            "Event type counts :",
            result[
                "event_type_counts"
            ],
        )

        return result


    def _stage_quality_filter(
        self,
    ):
        quality_filter = EventQualityFilter(
            composite_events_path=Path(
                self.results[
                    "composite_event_generation"
                ][
                    "metadata_path"
                ]
            ),
            output_dir=(
                self.stage_directories[
                    "event_quality_filter"
                ]
            ),
            minimum_track_duration=0.10,
            minimum_relationship_frames=4,
            minimum_quality_score=0.50,
            include_review=True,
        )

        result = quality_filter.process()

        print(
            "Input events        :",
            result[
                "input_events"
            ],
        )

        print(
            "Accepted events     :",
            result[
                "accepted_events"
            ],
        )

        print(
            "Rejected events     :",
            result[
                "rejected_events"
            ],
        )

        print(
            "Retrieval documents :",
            result[
                "retrieval_documents"
            ],
        )

        return result


    def _stage_embeddings(
        self,
    ):
        embedder = EventEmbedder(
            model_name="ViT-B-32",
            pretrained=(
                "laion2b_s34b_b79k"
            ),
            batch_size=16,
        )

        result = (
            embedder.generate_embeddings(
                input_file=Path(
                    self.results[
                        "event_quality_filter"
                    ][
                        "retrieval_path"
                    ]
                ),
                output_directory=(
                    self.stage_directories[
                        "event_embeddings"
                    ]
                ),
            )
        )

        embeddings = result[
            "embeddings"
        ]

        print(
            "Generated embeddings :",
            embeddings.shape[
                0
            ],
        )

        print(
            "Embedding dimension  :",
            embeddings.shape[
                1
            ],
        )

        return result


    def _stage_chromadb_storage(
        self,
    ):
        embedding_result = self.results[
            "event_embeddings"
        ]

        quality_result = self.results[
            "event_quality_filter"
        ]

        store = ChromaDBEventStore(
            database_directory=(
                self.database_directory
            ),
            collection_name=(
                self.collection_name
            ),
        )

        result = store.index_events(
            embeddings_file=Path(
                embedding_result[
                    "embeddings_file"
                ]
            ),
            embedding_metadata_file=Path(
                embedding_result[
                    "metadata_file"
                ]
            ),
            retrieval_documents_file=Path(
                quality_result[
                    "retrieval_path"
                ]
            ),
        )

        print(
            "Prepared records  :",
            result[
                "prepared_records"
            ],
        )

        print(
            "Collection count  :",
            result[
                "collection_count"
            ],
        )

        print(
            "Collection name   :",
            result[
                "collection_name"
            ],
        )

        return result


    def _json_safe(
        self,
        value,
    ):
        if isinstance(
            value,
            Path,
        ):
            return str(
                value
            )

        if isinstance(
            value,
            dict,
        ):
            return {
                str(key): self._json_safe(
                    item
                )
                for key, item
                in value.items()
            }

        if isinstance(
            value,
            (
                list,
                tuple,
            ),
        ):
            return [
                self._json_safe(
                    item
                )
                for item in value
            ]

        if hasattr(
            value,
            "tolist",
        ):
            return value.tolist()

        return value


    def _build_manifest(
        self,
    ):
        total_seconds = (
            time.perf_counter()
            - self.pipeline_start_time
        )

        return {
            "success": True,
            "pipeline": (
                "phase3_production_pipeline"
            ),
            "completed_at": (
                datetime.now()
                .isoformat(
                    timespec="seconds"
                )
            ),
            "source_video": (
                str(self.video_path)
            ),
            "run_root": (
                str(self.run_root)
            ),
            "database_directory": (
                str(
                    self.database_directory
                )
            ),
            "collection_name": (
                self.collection_name
            ),
            "video": {
                "fps": (
                    self.source_fps
                ),
                "frame_width": (
                    self.frame_width
                ),
                "frame_height": (
                    self.frame_height
                ),
                "total_frames": (
                    self.total_frames
                ),
                "duration_seconds": (
                    self.video_duration_seconds
                ),
            },
            "stage_count": 16,
            "completed_stages": (
                list(
                    self.results.keys()
                )
            ),
            "timings": (
                self.timings
            ),
            "total_processing_seconds": (
                round(
                    total_seconds,
                    4,
                )
            ),
            "important_outputs": {
                "track_metadata": (
                    self.results[
                        "object_tracking"
                    ][
                        "metadata_path"
                    ]
                ),
                "composite_events": (
                    self.results[
                        "composite_event_generation"
                    ][
                        "metadata_path"
                    ]
                ),
                "retrieval_documents": (
                    self.results[
                        "event_quality_filter"
                    ][
                        "retrieval_path"
                    ]
                ),
                "embeddings_file": (
                    self.results[
                        "event_embeddings"
                    ][
                        "embeddings_file"
                    ]
                ),
                "embedding_metadata": (
                    self.results[
                        "event_embeddings"
                    ][
                        "metadata_file"
                    ]
                ),
            },
            "summary": {
                "unique_tracks": (
                    self.results[
                        "object_tracking"
                    ][
                        "unique_tracks"
                    ]
                ),
                "composite_events": (
                    self.results[
                        "composite_event_generation"
                    ][
                        "total_events"
                    ]
                ),
                "accepted_events": (
                    self.results[
                        "event_quality_filter"
                    ][
                        "accepted_events"
                    ]
                ),
                "retrieval_documents": (
                    self.results[
                        "event_quality_filter"
                    ][
                        "retrieval_documents"
                    ]
                ),
                "indexed_records": (
                    self.results[
                        "chromadb_storage"
                    ][
                        "prepared_records"
                    ]
                ),
                "collection_count": (
                    self.results[
                        "chromadb_storage"
                    ][
                        "collection_count"
                    ]
                ),
            },
        }


    def run(
        self,
        video_path,
    ):
        self.pipeline_start_time = (
            time.perf_counter()
        )

        self.results = {}

        self.timings = {}

        self._prepare_run(
            video_path
        )

        print(
            "\n"
            + "=" * 78
        )

        print(
            "PHASE 3 PRODUCTION PIPELINE"
        )

        print(
            "=" * 78
        )

        print(
            f"Source video : "
            f"{self.video_path}"
        )

        print(
            f"Run folder   : "
            f"{self.run_root}"
        )

        print(
            f"Database     : "
            f"{self.database_directory}"
        )

        self._run_stage(
            "video_analysis",
            1,
            "VIDEO ANALYSIS",
            self._stage_video_analysis,
        )

        self._run_stage(
            "adaptive_sampling",
            2,
            "ADAPTIVE FRAME SAMPLING",
            self._stage_adaptive_sampling,
        )

        self._run_stage(
            "frame_extraction",
            3,
            "FRAME EXTRACTION",
            self._stage_frame_extraction,
        )

        self._run_stage(
            "sampled_frame_detection",
            4,
            "SAMPLED FRAME DETECTION",
            self._stage_sampled_detection,
        )

        self._run_stage(
            "object_tracking",
            5,
            "YOLO + BYTETRACK TRACKING",
            self._stage_object_tracking,
        )

        self._run_stage(
            "representative_selection",
            6,
            "REPRESENTATIVE CROP SELECTION",
            self._stage_representative_selection,
        )

        self._run_stage(
            "track_reliability",
            7,
            "TRACK RELIABILITY",
            self._stage_track_reliability,
        )

        self._run_stage(
            "color_analysis",
            8,
            "OBJECT-AWARE COLOR ANALYSIS",
            self._stage_color_analysis,
        )

        self._run_stage(
            "motion_analysis",
            9,
            "MOTION ANALYSIS",
            self._stage_motion_analysis,
        )

        self._run_stage(
            "spatial_analysis",
            10,
            "SPATIAL ANALYSIS",
            self._stage_spatial_analysis,
        )

        self._run_stage(
            "relationship_analysis",
            11,
            "RELATIONSHIP ANALYSIS",
            self._stage_relationship_analysis,
        )

        self._run_stage(
            "atomic_event_generation",
            12,
            "ATOMIC EVENT GENERATION",
            self._stage_atomic_events,
        )

        self._run_stage(
            "composite_event_generation",
            13,
            "COMPOSITE EVENT GENERATION",
            self._stage_composite_events,
        )

        self._run_stage(
            "event_quality_filter",
            14,
            "EVENT QUALITY FILTER",
            self._stage_quality_filter,
        )

        self._run_stage(
            "event_embeddings",
            15,
            "CLIP EVENT EMBEDDINGS",
            self._stage_embeddings,
        )

        self._run_stage(
            "chromadb_storage",
            16,
            "CHROMADB INDEXING",
            self._stage_chromadb_storage,
        )

        manifest = (
            self._build_manifest()
        )

        manifest_path = (
            self.run_root
            / "phase3_pipeline_manifest.json"
        )

        with open(
            manifest_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self._json_safe(
                    manifest
                ),
                file,
                indent=2,
                ensure_ascii=False,
            )

        manifest[
            "manifest_path"
        ] = str(
            manifest_path
        )

        print(
            "\n"
            + "=" * 78
        )

        print(
            "PHASE 3 PRODUCTION PIPELINE COMPLETE"
        )

        print(
            "=" * 78
        )

        print(
            "Completed stages    : 16/16"
        )

        print(
            "Unique tracks       :",
            manifest[
                "summary"
            ][
                "unique_tracks"
            ],
        )

        print(
            "Composite events    :",
            manifest[
                "summary"
            ][
                "composite_events"
            ],
        )

        print(
            "Accepted events     :",
            manifest[
                "summary"
            ][
                "accepted_events"
            ],
        )

        print(
            "Indexed records     :",
            manifest[
                "summary"
            ][
                "indexed_records"
            ],
        )

        print(
            "Collection count    :",
            manifest[
                "summary"
            ][
                "collection_count"
            ],
        )

        print(
            "Total processing    : "
            f"{manifest['total_processing_seconds']:.2f}s"
        )

        print(
            "Manifest            :",
            manifest_path,
        )

        print(
            "=" * 78
        )

        return manifest
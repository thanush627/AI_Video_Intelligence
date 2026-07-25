from pathlib import Path
import re

import chromadb
import open_clip
import torch

from ai.retrieval.temporal_query_parser import (
    TemporalQueryParser,
)


class SemanticEventSearch:

    PERSON_CLASSES = {
        "pedestrian",
    }

    VEHICLE_CLASSES = {
        "car",
        "bus",
        "van",
        "truck",
        "bicycle",
        "motorcycle",
    }

    CLASS_ALIASES = {
        "person": "pedestrian",
        "persons": "pedestrian",
        "people": "pedestrian",
        "pedestrian": "pedestrian",
        "pedestrians": "pedestrian",

        "car": "car",
        "cars": "car",

        "bus": "bus",
        "buses": "bus",

        "van": "van",
        "vans": "van",

        "truck": "truck",
        "trucks": "truck",

        "bicycle": "bicycle",
        "bicycles": "bicycle",
        "bike": "bicycle",
        "bikes": "bicycle",

        "motorcycle": "motorcycle",
        "motorcycles": "motorcycle",
        "motorbike": "motorcycle",
        "motorbikes": "motorcycle",
    }

    GROUP_ALIASES = {
        "vehicle": "vehicle",
        "vehicles": "vehicle",
    }

    COLORS = {
        "black",
        "white",
        "gray",
        "grey",
        "blue",
        "red",
        "green",
        "yellow",
        "orange",
        "pink",
        "purple",
        "brown",
    }

    DIRECTIONS = {
        "top_left_to_bottom_right": [
            "top left to bottom right",
            "top-left to bottom-right",
        ],
        "top_right_to_bottom_left": [
            "top right to bottom left",
            "top-right to bottom-left",
        ],
        "bottom_left_to_top_right": [
            "bottom left to top right",
            "bottom-left to top-right",
        ],
        "bottom_right_to_top_left": [
            "bottom right to top left",
            "bottom-right to top-left",
        ],
        "left_to_right": [
            "left to right",
            "left-to-right",
        ],
        "right_to_left": [
            "right to left",
            "right-to-left",
        ],
        "top_to_bottom": [
            "top to bottom",
            "top-to-bottom",
        ],
        "bottom_to_top": [
            "bottom to top",
            "bottom-to-top",
        ],
    }

    EXACT_REGIONS = {
        "left_top": [
            "left top region",
            "top left region",
        ],
        "center_top": [
            "center top region",
            "top center region",
        ],
        "right_top": [
            "right top region",
            "top right region",
        ],
        "left_middle": [
            "left middle region",
            "middle left region",
        ],
        "center_middle": [
            "center middle region",
            "middle center region",
        ],
        "right_middle": [
            "right middle region",
            "middle right region",
        ],
        "left_bottom": [
            "left bottom region",
            "bottom left region",
        ],
        "center_bottom": [
            "center bottom region",
            "bottom center region",
        ],
        "right_bottom": [
            "right bottom region",
            "bottom right region",
        ],
    }

    BROAD_REGIONS = {
        "left": [
            "left side",
            "left region",
            "on the left",
        ],
        "right": [
            "right side",
            "right region",
            "on the right",
        ],
        "center": [
            "center side",
            "centre side",
            "central region",
            "in the center",
            "in the centre",
        ],
        "top": [
            "top side",
            "top region",
            "at the top",
        ],
        "bottom": [
            "bottom side",
            "bottom region",
            "at the bottom",
        ],
    }

    MOVING_WORDS = {
        "moving",
        "moved",
        "walking",
        "walked",
        "traveling",
        "travelling",
        "going",
    }

    STATIONARY_PHRASES = {
        "stationary",
        "standing",
        "stopped",
        "still",
        "not moving",
        "remained stationary",
    }

    RELATIONSHIPS = {
        "near": [
            "near",
            "nearby",
            "close to",
        ],
        "overlapping": [
            "overlap",
            "overlapping",
            "overlapped",
        ],
        "together": [
            "together",
        ],
    }


    def __init__(
        self,
        database_directory,
        collection_name="event_vectors",
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device=None,
    ):

        self.database_directory = Path(
            database_directory
        )

        if not self.database_directory.exists():

            raise FileNotFoundError(
                f"ChromaDB directory not found: "
                f"{self.database_directory}"
            )

        self.collection_name = (
            collection_name
        )

        self.model_name = (
            model_name
        )

        self.pretrained = (
            pretrained
        )

        if device is None:

            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        else:

            self.device = device

        self.temporal_parser = (
            TemporalQueryParser()
        )

        print(
            f"Loading CLIP model: "
            f"{self.model_name}"
        )

        print(
            f"Device: {self.device}"
        )

        self.model, _, _ = (
            open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device,
            )
        )

        self.model.eval()

        self.tokenizer = (
            open_clip.get_tokenizer(
                self.model_name
            )
        )

        print(
            f"Opening ChromaDB: "
            f"{self.database_directory.resolve()}"
        )

        self.client = (
            chromadb.PersistentClient(
                path=str(
                    self.database_directory.resolve()
                )
            )
        )

        self.collection = (
            self.client.get_collection(
                name=self.collection_name
            )
        )

        print(
            f"Collection: "
            f"{self.collection_name}"
        )

        print(
            f"Indexed events: "
            f"{self.collection.count()}"
        )


    @staticmethod
    def _normalize_text(
        text,
    ):

        text = str(
            text or ""
        ).lower()

        text = text.replace(
            "_",
            " ",
        )

        text = text.replace(
            "-",
            " ",
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()


    @staticmethod
    def _safe_float(
        value,
        default=-1.0,
    ):

        try:

            if value is None:

                return float(
                    default
                )

            return float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return float(
                default
            )


    @staticmethod
    def _distance_to_similarity(
        distance,
    ):

        if distance is None:

            return 0.0

        similarity = (
            1.0
            - float(
                distance
            )
        )

        similarity = max(
            -1.0,
            min(
                1.0,
                similarity,
            ),
        )

        return round(
            similarity,
            4,
        )


    def encode_query(
        self,
        query,
    ):

        query = str(
            query
        ).strip()

        if not query:

            raise ValueError(
                "Search query cannot be empty."
            )

        tokens = (
            self.tokenizer(
                [query]
            )
            .to(
                self.device
            )
        )

        with torch.no_grad():

            query_embedding = (
                self.model.encode_text(
                    tokens
                )
            )

            query_embedding = (
                query_embedding
                / query_embedding.norm(
                    dim=-1,
                    keepdim=True,
                )
            )

        return (
            query_embedding
            .cpu()
            .numpy()
            .astype(
                "float32"
            )[0]
        )


    def _parse_query(
        self,
        query,
    ):

        query_text = (
            self._normalize_text(
                query
            )
        )

        parsed = {
            "classes": [],
            "class_group": None,
            "colors": [],
            "motion_state": None,
            "direction": None,
            "exact_regions": [],
            "broad_regions": [],
            "relationships": [],
            "relationship_requested": False,
            "has_hard_constraints": False,
        }

        for (
            alias,
            canonical_class,
        ) in self.CLASS_ALIASES.items():

            if re.search(
                rf"\b"
                rf"{re.escape(alias)}"
                rf"\b",
                query_text,
            ):

                parsed[
                    "classes"
                ].append(
                    canonical_class
                )

        parsed["classes"] = list(
            dict.fromkeys(
                parsed["classes"]
            )
        )

        for (
            alias,
            group_name,
        ) in self.GROUP_ALIASES.items():

            if re.search(
                rf"\b"
                rf"{re.escape(alias)}"
                rf"\b",
                query_text,
            ):

                parsed[
                    "class_group"
                ] = group_name

                break

        for color in self.COLORS:

            if re.search(
                rf"\b"
                rf"{re.escape(color)}"
                rf"\b",
                query_text,
            ):

                if color == "grey":

                    color = "gray"

                parsed[
                    "colors"
                ].append(
                    color
                )

        parsed["colors"] = list(
            dict.fromkeys(
                parsed["colors"]
            )
        )

        for (
            direction,
            phrases,
        ) in self.DIRECTIONS.items():

            matched = any(
                self._normalize_text(
                    phrase
                )
                in query_text

                for phrase in phrases
            )

            if matched:

                parsed[
                    "direction"
                ] = direction

                parsed[
                    "motion_state"
                ] = "moving"

                break

        stationary_match = any(
            phrase in query_text

            for phrase
            in self.STATIONARY_PHRASES
        )

        if stationary_match:

            parsed[
                "motion_state"
            ] = "stationary"

        elif (
            parsed[
                "motion_state"
            ]
            is None
        ):

            moving_match = any(
                re.search(
                    rf"\b"
                    rf"{re.escape(word)}"
                    rf"\b",
                    query_text,
                )

                for word
                in self.MOVING_WORDS
            )

            if moving_match:

                parsed[
                    "motion_state"
                ] = "moving"

        for (
            region,
            phrases,
        ) in self.EXACT_REGIONS.items():

            matched = any(
                self._normalize_text(
                    phrase
                )
                in query_text

                for phrase in phrases
            )

            if matched:

                parsed[
                    "exact_regions"
                ].append(
                    region
                )

        parsed[
            "exact_regions"
        ] = list(
            dict.fromkeys(
                parsed[
                    "exact_regions"
                ]
            )
        )

        for (
            region,
            phrases,
        ) in self.BROAD_REGIONS.items():

            matched = any(
                self._normalize_text(
                    phrase
                )
                in query_text

                for phrase in phrases
            )

            if matched:

                parsed[
                    "broad_regions"
                ].append(
                    region
                )

        parsed[
            "broad_regions"
        ] = list(
            dict.fromkeys(
                parsed[
                    "broad_regions"
                ]
            )
        )

        for (
            relationship,
            phrases,
        ) in self.RELATIONSHIPS.items():

            matched = any(
                self._normalize_text(
                    phrase
                )
                in query_text

                for phrase in phrases
            )

            if matched:

                parsed[
                    "relationships"
                ].append(
                    relationship
                )

        parsed[
            "relationships"
        ] = list(
            dict.fromkeys(
                parsed[
                    "relationships"
                ]
            )
        )

        parsed[
            "relationship_requested"
        ] = bool(
            parsed[
                "relationships"
            ]
        )

        parsed[
            "has_hard_constraints"
        ] = any(
            [
                bool(
                    parsed[
                        "classes"
                    ]
                ),
                (
                    parsed[
                        "class_group"
                    ]
                    is not None
                ),
                bool(
                    parsed[
                        "colors"
                    ]
                ),
                (
                    parsed[
                        "motion_state"
                    ]
                    is not None
                ),
                (
                    parsed[
                        "direction"
                    ]
                    is not None
                ),
                bool(
                    parsed[
                        "exact_regions"
                    ]
                ),
                bool(
                    parsed[
                        "broad_regions"
                    ]
                ),
                parsed[
                    "relationship_requested"
                ],
            ]
        )

        return parsed


    def _get_event_classes(
        self,
        metadata,
    ):

        class_names = str(
            metadata.get(
                "class_names",
                "",
            )
        )

        event_classes = set()

        for class_name in (
            class_names.split(
                ","
            )
        ):

            class_name = (
                self._normalize_text(
                    class_name
                )
            )

            if not class_name:

                continue

            canonical_class = (
                self.CLASS_ALIASES.get(
                    class_name,
                    class_name,
                )
            )

            event_classes.add(
                canonical_class
            )

        return event_classes


    def _class_matches(
        self,
        parsed_query,
        metadata,
    ):

        event_classes = (
            self._get_event_classes(
                metadata
            )
        )

        requested_classes = set(
            parsed_query[
                "classes"
            ]
        )

        if requested_classes:

            if not (
                event_classes
                & requested_classes
            ):

                return False

        class_group = (
            parsed_query[
                "class_group"
            ]
        )

        if (
            class_group
            == "vehicle"
        ):

            if not (
                event_classes
                & self.VEHICLE_CLASSES
            ):

                return False

        return True


    def _color_matches(
        self,
        parsed_query,
        document,
    ):

        requested_colors = (
            parsed_query[
                "colors"
            ]
        )

        if not requested_colors:

            return True

        document_text = (
            self._normalize_text(
                document
            )
        )

        return any(
            re.search(
                rf"\b"
                rf"{re.escape(color)}"
                rf"\b",
                document_text,
            )

            for color
            in requested_colors
        )


    def _motion_matches(
        self,
        parsed_query,
        document,
        metadata,
    ):

        requested_state = (
            parsed_query[
                "motion_state"
            ]
        )

        if requested_state is None:

            return True

        event_type = (
            self._normalize_text(
                metadata.get(
                    "event_type",
                    "",
                )
            )
        )

        if (
            event_type
            == "relationship event"
        ):

            return False

        document_text = (
            self._normalize_text(
                document
            )
        )

        stationary_match = (
            "stationary"
            in document_text
            or "standing"
            in document_text
            or "stopped"
            in document_text
        )

        moving_match = (
            " moved "
            in f" {document_text} "
            or "moving"
            in document_text
            or "walking"
            in document_text
            or "walked"
            in document_text
        )

        if (
            requested_state
            == "stationary"
        ):

            return stationary_match

        if (
            requested_state
            == "moving"
        ):

            return (
                moving_match
                and not stationary_match
            )

        return True


    def _direction_matches(
        self,
        parsed_query,
        document,
    ):

        direction = (
            parsed_query[
                "direction"
            ]
        )

        if direction is None:

            return True

        document_text = (
            self._normalize_text(
                document
            )
        )

        direction_text = (
            self._normalize_text(
                direction
            )
        )

        return (
            direction_text
            in document_text
        )


    def _region_matches(
        self,
        parsed_query,
        document,
    ):

        document_text = (
            self._normalize_text(
                document
            )
        )

        exact_regions = (
            parsed_query[
                "exact_regions"
            ]
        )

        if exact_regions:

            exact_match = any(
                self._normalize_text(
                    region
                )
                in document_text

                for region
                in exact_regions
            )

            if not exact_match:

                return False

        for broad_region in (
            parsed_query[
                "broad_regions"
            ]
        ):

            if (
                broad_region
                == "left"
            ):

                region_match = any(
                    region
                    in document_text

                    for region in [
                        "left top region",
                        "left middle region",
                        "left bottom region",
                    ]
                )

            elif (
                broad_region
                == "right"
            ):

                region_match = any(
                    region
                    in document_text

                    for region in [
                        "right top region",
                        "right middle region",
                        "right bottom region",
                    ]
                )

            elif (
                broad_region
                == "center"
            ):

                region_match = any(
                    region
                    in document_text

                    for region in [
                        "center top region",
                        "center middle region",
                        "center bottom region",
                    ]
                )

            elif (
                broad_region
                == "top"
            ):

                region_match = any(
                    region
                    in document_text

                    for region in [
                        "left top region",
                        "center top region",
                        "right top region",
                    ]
                )

            elif (
                broad_region
                == "bottom"
            ):

                region_match = any(
                    region
                    in document_text

                    for region in [
                        "left bottom region",
                        "center bottom region",
                        "right bottom region",
                    ]
                )

            else:

                region_match = True

            if not region_match:

                return False

        return True


    def _relationship_matches(
        self,
        parsed_query,
        document,
        metadata,
    ):

        if not parsed_query[
            "relationship_requested"
        ]:

            return True

        event_type = (
            self._normalize_text(
                metadata.get(
                    "event_type",
                    "",
                )
            )
        )

        if (
            event_type
            != "relationship event"
        ):

            return False

        document_text = (
            self._normalize_text(
                document
            )
        )

        for relationship in (
            parsed_query[
                "relationships"
            ]
        ):

            if (
                relationship
                == "near"
            ):

                if (
                    "near"
                    not in document_text
                ):

                    return False

            elif (
                relationship
                == "overlapping"
            ):

                if (
                    "overlap"
                    not in document_text
                ):

                    return False

            elif (
                relationship
                == "together"
            ):

                continue

        return True


    def _matches_hard_constraints(
        self,
        parsed_query,
        document,
        metadata,
    ):

        if not self._class_matches(
            parsed_query,
            metadata,
        ):

            return False

        if not self._color_matches(
            parsed_query,
            document,
        ):

            return False

        if not self._motion_matches(
            parsed_query,
            document,
            metadata,
        ):

            return False

        if not self._direction_matches(
            parsed_query,
            document,
        ):

            return False

        if not self._region_matches(
            parsed_query,
            document,
        ):

            return False

        if not self._relationship_matches(
            parsed_query,
            document,
            metadata,
        ):

            return False

        return True


    def _matches_temporal_filter(
        self,
        start_time,
        end_time,
        temporal_filter,
    ):

        if temporal_filter is None:

            return True

        if (
            start_time < 0
            or end_time < 0
        ):

            return False

        if start_time > end_time:

            start_time, end_time = (
                end_time,
                start_time,
            )

        filter_type = (
            temporal_filter.get(
                "type"
            )
        )

        filter_start = (
            temporal_filter.get(
                "start_time_seconds"
            )
        )

        filter_end = (
            temporal_filter.get(
                "end_time_seconds"
            )
        )

        if (
            filter_type
            == "range"
        ):

            if (
                filter_start is None
                or filter_end is None
            ):

                return True

            filter_start = float(
                filter_start
            )

            filter_end = float(
                filter_end
            )

            if (
                filter_start
                > filter_end
            ):

                (
                    filter_start,
                    filter_end,
                ) = (
                    filter_end,
                    filter_start,
                )

            return (
                end_time
                >= filter_start
                and start_time
                <= filter_end
            )

        if (
            filter_type
            == "after"
        ):

            if filter_start is None:

                return True

            return (
                end_time
                >= float(
                    filter_start
                )
            )

        if (
            filter_type
            == "before"
        ):

            if filter_end is None:

                return True

            return (
                start_time
                <= float(
                    filter_end
                )
            )

        return True


    def _calculate_hybrid_score(
        self,
        similarity,
        parsed_query,
        metadata,
    ):

        score = float(
            similarity
        )

        quality_score = (
            self._safe_float(
                metadata.get(
                    "quality_score"
                ),
                default=0.0,
            )
        )

        quality_score = max(
            0.0,
            min(
                1.0,
                quality_score,
            ),
        )

        score += (
            0.08
            * quality_score
        )

        if parsed_query[
            "classes"
        ]:

            score += 0.12

        if parsed_query[
            "class_group"
        ]:

            score += 0.08

        if parsed_query[
            "colors"
        ]:

            score += 0.12

        if parsed_query[
            "motion_state"
        ]:

            score += 0.10

        if parsed_query[
            "direction"
        ]:

            score += 0.18

        if parsed_query[
            "exact_regions"
        ]:

            score += 0.12

        if parsed_query[
            "broad_regions"
        ]:

            score += 0.08

        if parsed_query[
            "relationship_requested"
        ]:

            score += 0.12

        return round(
            score,
            4,
        )


    def search(
        self,
        query,
        top_k=5,
        minimum_similarity=None,
        where=None,
    ):

        query = str(
            query
        ).strip()

        if not query:

            raise ValueError(
                "Search query cannot be empty."
            )

        collection_count = (
            self.collection.count()
        )

        if collection_count == 0:

            return []

        requested_top_k = int(
            top_k
        )

        if requested_top_k <= 0:

            raise ValueError(
                "top_k must be greater than 0."
            )

        actual_top_k = min(
            requested_top_k,
            collection_count,
        )

        temporal_result = (
            self.temporal_parser.parse(
                query
            )
        )

        semantic_query = str(
            temporal_result.get(
                "semantic_query",
                query,
            )
        ).strip()

        if not semantic_query:

            semantic_query = "events"

        has_temporal_filter = bool(
            temporal_result.get(
                "has_temporal_filter",
                False,
            )
        )

        temporal_filter = (
            temporal_result.get(
                "temporal_filter"
            )
        )

        parsed_query = (
            self._parse_query(
                semantic_query
            )
        )

        query_embedding = (
            self.encode_query(
                semantic_query
            )
        )

        if (
            has_temporal_filter
            or parsed_query[
                "has_hard_constraints"
            ]
        ):

            candidate_count = (
                collection_count
            )

        else:

            candidate_count = min(
                max(
                    actual_top_k * 8,
                    30,
                ),
                collection_count,
            )

        query_arguments = {
            "query_embeddings": [
                query_embedding.tolist()
            ],
            "n_results": (
                candidate_count
            ),
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if where:

            query_arguments[
                "where"
            ] = where

        raw_results = (
            self.collection.query(
                **query_arguments
            )
        )

        ids = raw_results.get(
            "ids",
            [[]],
        )[0]

        documents = raw_results.get(
            "documents",
            [[]],
        )[0]

        metadatas = raw_results.get(
            "metadatas",
            [[]],
        )[0]

        distances = raw_results.get(
            "distances",
            [[]],
        )[0]

        results = []

        for (
            event_id,
            document,
            metadata,
            distance,
        ) in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):

            metadata = (
                metadata
                or {}
            )

            document = (
                document
                or ""
            )

            similarity = (
                self._distance_to_similarity(
                    distance
                )
            )

            if (
                minimum_similarity
                is not None
                and similarity
                < float(
                    minimum_similarity
                )
            ):

                continue

            start_time = (
                self._safe_float(
                    metadata.get(
                        "start_time_seconds"
                    ),
                    default=-1.0,
                )
            )

            end_time = (
                self._safe_float(
                    metadata.get(
                        "end_time_seconds"
                    ),
                    default=-1.0,
                )
            )

            if (
                start_time >= 0
                and end_time >= 0
                and start_time > end_time
            ):

                start_time, end_time = (
                    end_time,
                    start_time,
                )

            duration = (
                self._safe_float(
                    metadata.get(
                        "duration_seconds"
                    ),
                    default=-1.0,
                )
            )

            if (
                duration < 0
                and start_time >= 0
                and end_time >= 0
            ):

                duration = max(
                    0.0,
                    end_time
                    - start_time,
                )

            quality_score = (
                self._safe_float(
                    metadata.get(
                        "quality_score"
                    ),
                    default=0.0,
                )
            )

            if has_temporal_filter:

                if not (
                    self._matches_temporal_filter(
                        start_time=(
                            start_time
                        ),
                        end_time=(
                            end_time
                        ),
                        temporal_filter=(
                            temporal_filter
                        ),
                    )
                ):

                    continue

            if parsed_query[
                "has_hard_constraints"
            ]:

                if not (
                    self._matches_hard_constraints(
                        parsed_query=(
                            parsed_query
                        ),
                        document=(
                            document
                        ),
                        metadata=(
                            metadata
                        ),
                    )
                ):

                    continue

            hybrid_score = (
                self._calculate_hybrid_score(
                    similarity=(
                        similarity
                    ),
                    parsed_query=(
                        parsed_query
                    ),
                    metadata=(
                        metadata
                    ),
                )
            )

            rerank_adjustment = round(
                hybrid_score
                - similarity,
                4,
            )

            result = {
                "event_id": event_id,

                "document": document,

                "similarity": (
                    similarity
                ),

                "hybrid_score": (
                    hybrid_score
                ),

                "rerank_adjustment": (
                    rerank_adjustment
                ),

                "distance": round(
                    float(
                        distance
                    ),
                    4,
                ),

                "event_type": (
                    metadata.get(
                        "event_type",
                        "unknown",
                    )
                ),

                "class_names": (
                    metadata.get(
                        "class_names",
                        "",
                    )
                ),

                "track_ids": (
                    metadata.get(
                        "track_ids",
                        "",
                    )
                ),

                "start_time_seconds": (
                    start_time
                ),

                "end_time_seconds": (
                    end_time
                ),

                "duration_seconds": (
                    duration
                ),

                "quality_score": (
                    quality_score
                ),

                "quality_label": (
                    metadata.get(
                        "quality_label",
                        "unknown",
                    )
                ),

                "semantic_query": (
                    semantic_query
                ),

                "has_temporal_filter": (
                    has_temporal_filter
                ),

                "temporal_filter": (
                    temporal_filter
                ),

                "parsed_constraints": (
                    parsed_query
                ),

                "metadata": (
                    metadata
                ),
            }

            results.append(
                result
            )

        results.sort(
            key=lambda item: (
                float(
                    item.get(
                        "hybrid_score",
                        0.0,
                    )
                ),
                float(
                    item.get(
                        "similarity",
                        0.0,
                    )
                ),
                float(
                    item.get(
                        "quality_score",
                        0.0,
                    )
                ),
            ),
            reverse=True,
        )

        final_results = (
            results[
                :actual_top_k
            ]
        )

        for (
            rank,
            result,
        ) in enumerate(
            final_results,
            start=1,
        ):

            result[
                "rank"
            ] = rank

        return final_results
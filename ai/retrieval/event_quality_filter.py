import json
from collections import Counter
from pathlib import Path


class EventQualityFilter:

    def __init__(
        self,
        composite_events_path,
        output_dir,
        minimum_track_duration=0.10,
        minimum_relationship_frames=4,
        minimum_quality_score=0.50,
        include_review=True,
    ):
        self.composite_events_path = Path(
            composite_events_path
        )

        self.output_dir = Path(
            output_dir
        )

        self.minimum_track_duration = float(
            minimum_track_duration
        )

        self.minimum_relationship_frames = int(
            minimum_relationship_frames
        )

        self.minimum_quality_score = float(
            minimum_quality_score
        )

        self.include_review = bool(
            include_review
        )

        if not self.composite_events_path.exists():
            raise FileNotFoundError(
                f"Composite events not found: "
                f"{self.composite_events_path}"
            )

    @staticmethod
    def _load_json(path):

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    @staticmethod
    def _clamp(value):

        return max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )

    @staticmethod
    def _quality_label(score):

        if score >= 0.80:
            return "high"

        if score >= 0.60:
            return "medium"

        return "low"

    @staticmethod
    def _has_colour(event):

        attributes = event.get(
            "colour_attributes",
            {},
        )

        if not isinstance(
            attributes,
            dict,
        ):
            return False

        return any(
            value
            and value != "unknown"
            for value in attributes.values()
        )

    @staticmethod
    def _has_motion(event):

        motion_state = event.get(
            "motion_state",
            "unknown",
        )

        motion_direction = event.get(
            "motion_direction",
            "unknown",
        )

        return (
            motion_state != "unknown"
            or motion_direction != "unknown"
        )

    @staticmethod
    def _has_spatial_information(event):

        start_region = event.get(
            "start_region"
        )

        end_region = event.get(
            "end_region"
        )

        return (
            isinstance(
                start_region,
                dict,
            )
            and isinstance(
                end_region,
                dict,
            )
        )

    def _score_track_event(
        self,
        event,
    ):

        score = 0.0

        reasons = []

        reliability_status = event.get(
            "reliability_status",
            "unknown",
        )

        duration = float(
            event.get(
                "duration_seconds",
                0.0,
            )
            or 0.0
        )

        if reliability_status == "reliable":

            score += 0.40

            reasons.append(
                "reliable_track"
            )

        elif (
            reliability_status == "review"
            and self.include_review
        ):

            score += 0.22

            reasons.append(
                "review_track"
            )

        else:

            reasons.append(
                "weak_reliability"
            )

        if duration >= 1.0:

            score += 0.25

            reasons.append(
                "long_duration"
            )

        elif duration >= 0.50:

            score += 0.20

            reasons.append(
                "medium_duration"
            )

        elif duration >= 0.20:

            score += 0.12

            reasons.append(
                "short_valid_duration"
            )

        elif (
            duration
            >= self.minimum_track_duration
        ):

            score += 0.06

            reasons.append(
                "minimum_valid_duration"
            )

        else:

            reasons.append(
                "too_short"
            )

        if self._has_colour(event):

            score += 0.15

            reasons.append(
                "colour_attributes"
            )

        if self._has_motion(event):

            score += 0.10

            reasons.append(
                "motion_attributes"
            )

        if self._has_spatial_information(
            event
        ):

            score += 0.10

            reasons.append(
                "spatial_attributes"
            )

        score = self._clamp(
            score
        )

        hard_reject = (
            duration
            < self.minimum_track_duration
        )

        return (
            score,
            reasons,
            hard_reject,
        )

    def _score_relationship_event(
        self,
        event,
    ):

        score = 0.0

        reasons = []

        shared_frames = int(
            event.get(
                "shared_frames",
                0,
            )
            or 0
        )

        near_ratio = float(
            event.get(
                "near_ratio",
                0.0,
            )
            or 0.0
        )

        overlap_ratio = float(
            event.get(
                "overlap_ratio",
                0.0,
            )
            or 0.0
        )

        relationship_types = event.get(
            "relationship_types",
            [],
        )

        if shared_frames >= 15:

            score += 0.40

            reasons.append(
                "strong_temporal_support"
            )

        elif shared_frames >= 8:

            score += 0.32

            reasons.append(
                "good_temporal_support"
            )

        elif shared_frames >= 5:

            score += 0.24

            reasons.append(
                "moderate_temporal_support"
            )

        elif (
            shared_frames
            >= self.minimum_relationship_frames
        ):

            score += 0.16

            reasons.append(
                "minimum_temporal_support"
            )

        else:

            reasons.append(
                "insufficient_temporal_support"
            )

        if near_ratio >= 0.80:

            score += 0.25

            reasons.append(
                "strong_near_evidence"
            )

        elif near_ratio >= 0.50:

            score += 0.15

            reasons.append(
                "moderate_near_evidence"
            )

        if overlap_ratio >= 0.50:

            score += 0.20

            reasons.append(
                "strong_overlap_evidence"
            )

        elif overlap_ratio >= 0.30:

            score += 0.12

            reasons.append(
                "moderate_overlap_evidence"
            )

        if (
            "moving_together"
            in relationship_types
        ):

            score += 0.15

            reasons.append(
                "moving_together"
            )

        if (
            "human_near_vehicle"
            in relationship_types
        ):

            score += 0.15

            reasons.append(
                "human_vehicle_interaction"
            )

        if relationship_types:

            score += 0.10

            reasons.append(
                "semantic_relationship"
            )

        score = self._clamp(
            score
        )

        hard_reject = (
            shared_frames
            < self.minimum_relationship_frames
        )

        return (
            score,
            reasons,
            hard_reject,
        )

    @staticmethod
    def _build_retrieval_document(
        event,
    ):

        description = str(
            event.get(
                "description",
                "",
            )
        ).strip()

        if not description:
            return ""

        if not description.endswith(
            "."
        ):
            description += "."

        start_time = event.get(
            "start_time_seconds"
        )

        end_time = event.get(
            "end_time_seconds"
        )

        if (
            start_time is not None
            and end_time is not None
        ):

            description += (
                f" The event occurred from "
                f"{float(start_time):.2f} seconds "
                f"to {float(end_time):.2f} seconds."
            )

        return description

    @staticmethod
    def _build_filter_metadata(
        event,
        quality_score,
        quality_label,
    ):

        metadata = {
            "event_id": event.get(
                "event_id"
            ),
            "event_type": event.get(
                "event_type"
            ),
            "quality_score": round(
                quality_score,
                4,
            ),
            "quality_label": (
                quality_label
            ),
            "track_ids": event.get(
                "track_ids",
                [],
            ),
            "class_names": event.get(
                "class_names",
                [],
            ),
            "start_time_seconds": (
                event.get(
                    "start_time_seconds"
                )
            ),
            "end_time_seconds": (
                event.get(
                    "end_time_seconds"
                )
            ),
        }

        if (
            event.get("event_type")
            == "track_semantic_event"
        ):

            metadata.update(
                {
                    "reliability_status": (
                        event.get(
                            "reliability_status"
                        )
                    ),
                    "colour_attributes": (
                        event.get(
                            "colour_attributes",
                            {},
                        )
                    ),
                    "motion_state": (
                        event.get(
                            "motion_state"
                        )
                    ),
                    "motion_direction": (
                        event.get(
                            "motion_direction"
                        )
                    ),
                    "region_transition": (
                        event.get(
                            "region_transition"
                        )
                    ),
                }
            )

        if (
            event.get("event_type")
            == "relationship_event"
        ):

            metadata.update(
                {
                    "relationship_types": (
                        event.get(
                            "relationship_types",
                            [],
                        )
                    ),
                    "shared_frames": (
                        event.get(
                            "shared_frames",
                            0,
                        )
                    ),
                    "near_ratio": (
                        event.get(
                            "near_ratio",
                            0.0,
                        )
                    ),
                    "overlap_ratio": (
                        event.get(
                            "overlap_ratio",
                            0.0,
                        )
                    ),
                }
            )

        return metadata

    def process(self):

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        composite_data = self._load_json(
            self.composite_events_path
        )

        events = composite_data.get(
            "events",
            [],
        )

        accepted_events = []

        rejected_events = []

        retrieval_documents = []

        accepted_type_counts = Counter()

        rejected_reason_counts = Counter()

        quality_label_counts = Counter()

        for event in events:

            event_type = event.get(
                "event_type",
                "unknown",
            )

            if (
                event_type
                == "track_semantic_event"
            ):

                (
                    quality_score,
                    quality_reasons,
                    hard_reject,
                ) = self._score_track_event(
                    event
                )

            elif (
                event_type
                == "relationship_event"
            ):

                (
                    quality_score,
                    quality_reasons,
                    hard_reject,
                ) = (
                    self._score_relationship_event(
                        event
                    )
                )

            else:

                quality_score = 0.0

                quality_reasons = [
                    "unsupported_event_type"
                ]

                hard_reject = True

            quality_label = (
                self._quality_label(
                    quality_score
                )
            )

            accepted = (
                not hard_reject
                and quality_score
                >= self.minimum_quality_score
            )

            processed_event = dict(
                event
            )

            processed_event[
                "quality_score"
            ] = round(
                quality_score,
                4,
            )

            processed_event[
                "quality_label"
            ] = quality_label

            processed_event[
                "quality_reasons"
            ] = quality_reasons

            if accepted:

                accepted_events.append(
                    processed_event
                )

                accepted_type_counts[
                    event_type
                ] += 1

                quality_label_counts[
                    quality_label
                ] += 1

                retrieval_document = (
                    self._build_retrieval_document(
                        event
                    )
                )

                if retrieval_document:

                    retrieval_documents.append(
                        {
                            "document_id": (
                                event.get(
                                    "event_id"
                                )
                            ),
                            "text": (
                                retrieval_document
                            ),
                            "metadata": (
                                self._build_filter_metadata(
                                    event,
                                    quality_score,
                                    quality_label,
                                )
                            ),
                        }
                    )

                print(
                    f"KEEP   | "
                    f"{event.get('event_id')} | "
                    f"Score {quality_score:.3f} | "
                    f"{quality_label.upper():6} | "
                    f"{event.get('description')}"
                )

            else:

                if hard_reject:

                    rejection_reason = (
                        "hard_reject"
                    )

                else:

                    rejection_reason = (
                        "below_quality_threshold"
                    )

                processed_event[
                    "rejection_reason"
                ] = rejection_reason

                rejected_events.append(
                    processed_event
                )

                rejected_reason_counts[
                    rejection_reason
                ] += 1

                print(
                    f"REJECT | "
                    f"{event.get('event_id')} | "
                    f"Score {quality_score:.3f} | "
                    f"{rejection_reason} | "
                    f"{event.get('description')}"
                )

        filtered_output = {
            "summary": {
                "input_events": len(
                    events
                ),
                "accepted_events": len(
                    accepted_events
                ),
                "rejected_events": len(
                    rejected_events
                ),
                "acceptance_rate": round(
                    (
                        len(accepted_events)
                        / len(events)
                    )
                    if events
                    else 0.0,
                    4,
                ),
                "accepted_type_counts": dict(
                    accepted_type_counts
                ),
                "quality_label_counts": dict(
                    quality_label_counts
                ),
                "rejected_reason_counts": dict(
                    rejected_reason_counts
                ),
                "minimum_quality_score": (
                    self.minimum_quality_score
                ),
                "minimum_track_duration": (
                    self.minimum_track_duration
                ),
                "minimum_relationship_frames": (
                    self.minimum_relationship_frames
                ),
            },
            "accepted_events": (
                accepted_events
            ),
            "rejected_events": (
                rejected_events
            ),
        }

        retrieval_output = {
            "summary": {
                "document_count": len(
                    retrieval_documents
                ),
                "source": (
                    "filtered_composite_events"
                ),
            },
            "documents": (
                retrieval_documents
            ),
        }

        filtered_path = (
            self.output_dir
            / "filtered_events.json"
        )

        retrieval_path = (
            self.output_dir
            / "retrieval_documents.json"
        )

        with open(
            filtered_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                filtered_output,
                file,
                indent=2,
            )

        with open(
            retrieval_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                retrieval_output,
                file,
                indent=2,
            )

        return {
            "input_events": len(
                events
            ),
            "accepted_events": len(
                accepted_events
            ),
            "rejected_events": len(
                rejected_events
            ),
            "accepted_type_counts": dict(
                accepted_type_counts
            ),
            "quality_label_counts": dict(
                quality_label_counts
            ),
            "rejected_reason_counts": dict(
                rejected_reason_counts
            ),
            "retrieval_documents": len(
                retrieval_documents
            ),
            "filtered_path": str(
                filtered_path
            ),
            "retrieval_path": str(
                retrieval_path
            ),
        }
import json
from collections import Counter
from pathlib import Path


class TrackReliabilityFilter:

    HUMAN_CLASSES = {
        "pedestrian",
        "people",
    }

    VEHICLE_CLASSES = {
        "car",
        "van",
        "truck",
        "bus",
    }

    BICYCLE_CLASSES = {
        "bicycle",
    }

    def __init__(
        self,
        representative_metadata_path,
        output_dir,
    ):
        self.metadata_path = Path(
            representative_metadata_path
        )

        self.output_dir = Path(output_dir)

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Representative metadata not found: "
                f"{self.metadata_path}"
            )

    @staticmethod
    def _clamp(value):

        return max(
            0.0,
            min(1.0, float(value)),
        )

    @staticmethod
    def _get_observation_count(track):

        possible_keys = [
            "observation_count",
            "observations",
            "num_observations",
            "total_observations",
            "track_length",
        ]

        for key in possible_keys:

            value = track.get(key)

            if isinstance(value, (int, float)):
                return int(value)

        return len(
            track.get(
                "selected_crops",
                []
            )
        )

    @staticmethod
    def _extract_crop_score(crop):

        possible_keys = [
            "final_score",
            "score",
            "quality_score",
            "selection_score",
        ]

        for key in possible_keys:

            value = crop.get(key)

            if isinstance(value, (int, float)):
                return float(value)

        return 0.5

    @staticmethod
    def _extract_confidence(crop):

        possible_keys = [
            "confidence",
            "conf",
            "detection_confidence",
        ]

        for key in possible_keys:

            value = crop.get(key)

            if isinstance(value, (int, float)):
                return float(value)

        return 0.5

    def _persistence_score(
        self,
        observation_count,
        class_name,
    ):

        if class_name in self.VEHICLE_CLASSES:
            target = 8

        elif class_name in self.BICYCLE_CLASSES:
            target = 6

        else:
            target = 5

        return self._clamp(
            observation_count / target
        )

    def _crop_consistency_score(
        self,
        selected_crops,
    ):

        if not selected_crops:
            return 0.0

        scores = [
            self._extract_crop_score(crop)
            for crop in selected_crops
        ]

        average_score = (
            sum(scores) / len(scores)
        )

        if len(scores) == 1:
            consistency = 0.55

        else:

            score_range = (
                max(scores) - min(scores)
            )

            consistency = self._clamp(
                1.0 - score_range
            )

        return self._clamp(
            0.70 * average_score
            + 0.30 * consistency
        )

    def _confidence_score(
        self,
        selected_crops,
    ):

        if not selected_crops:
            return 0.0

        confidences = [
            self._extract_confidence(crop)
            for crop in selected_crops
        ]

        return self._clamp(
            sum(confidences)
            / len(confidences)
        )

    def _crop_count_score(
        self,
        selected_crops,
    ):

        crop_count = len(selected_crops)

        return self._clamp(
            crop_count / 3.0
        )

    def _class_penalty(
        self,
        class_name,
        observation_count,
        crop_count,
    ):

        penalty = 0.0

        reasons = []

        if (
            class_name in self.BICYCLE_CLASSES
            and observation_count < 4
        ):

            penalty += 0.20

            reasons.append(
                "short_bicycle_track"
            )

        if (
            class_name in self.VEHICLE_CLASSES
            and observation_count <= 2
        ):

            penalty += 0.18

            reasons.append(
                "very_short_vehicle_track"
            )

        if (
            class_name == "people"
            and observation_count <= 2
        ):

            penalty += 0.15

            reasons.append(
                "short_group_track"
            )

        if crop_count == 1:

            penalty += 0.08

            reasons.append(
                "single_representative_crop"
            )

        return penalty, reasons

    def _evaluate_track(
        self,
        track,
    ):

        class_name = track.get(
            "class_name",
            "unknown",
        )

        selected_crops = track.get(
            "selected_crops",
            [],
        )

        observation_count = (
            self._get_observation_count(
                track
            )
        )

        crop_count = len(
            selected_crops
        )

        persistence_score = (
            self._persistence_score(
                observation_count,
                class_name,
            )
        )

        crop_consistency_score = (
            self._crop_consistency_score(
                selected_crops
            )
        )

        confidence_score = (
            self._confidence_score(
                selected_crops
            )
        )

        crop_count_score = (
            self._crop_count_score(
                selected_crops
            )
        )

        base_score = (
            0.35 * persistence_score
            + 0.30 * crop_consistency_score
            + 0.20 * confidence_score
            + 0.15 * crop_count_score
        )

        penalty, penalty_reasons = (
            self._class_penalty(
                class_name,
                observation_count,
                crop_count,
            )
        )

        reliability_score = self._clamp(
            base_score - penalty
        )

        reasons = list(
            penalty_reasons
        )

        if persistence_score < 0.30:

            reasons.append(
                "weak_temporal_persistence"
            )

        if crop_consistency_score < 0.35:

            reasons.append(
                "weak_crop_quality"
            )

        if confidence_score < 0.35:

            reasons.append(
                "low_detection_confidence"
            )

        if reliability_score >= 0.65:

            status = "reliable"

        elif reliability_score >= 0.40:

            status = "review"

        else:

            status = "reject"

        if not reasons:

            reasons.append(
                "no_major_quality_issue"
            )

        return {
            "reliability_score": round(
                reliability_score,
                4,
            ),
            "status": status,
            "observation_count": (
                observation_count
            ),
            "representative_crop_count": (
                crop_count
            ),
            "component_scores": {
                "persistence": round(
                    persistence_score,
                    4,
                ),
                "crop_consistency": round(
                    crop_consistency_score,
                    4,
                ),
                "detection_confidence": round(
                    confidence_score,
                    4,
                ),
                "crop_count": round(
                    crop_count_score,
                    4,
                ),
            },
            "penalty": round(
                penalty,
                4,
            ),
            "reasons": reasons,
        }

    def analyze(self):

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        output_tracks = {}

        status_counts = Counter()

        class_status_counts = {}

        for track_key, track in (
            data["tracks"].items()
        ):

            evaluation = (
                self._evaluate_track(
                    track
                )
            )

            class_name = track.get(
                "class_name",
                "unknown",
            )

            status = evaluation[
                "status"
            ]

            status_counts[
                status
            ] += 1

            if class_name not in (
                class_status_counts
            ):

                class_status_counts[
                    class_name
                ] = {
                    "reliable": 0,
                    "review": 0,
                    "reject": 0,
                }

            class_status_counts[
                class_name
            ][status] += 1

            output_tracks[
                track_key
            ] = {
                "track_id": track[
                    "track_id"
                ],
                "class_name": class_name,
                "reliability": evaluation,
                "selected_crops": track.get(
                    "selected_crops",
                    [],
                ),
            }

            print(
                f"Track "
                f"{track['track_id']:3} | "
                f"{class_name:10} | "
                f"Score "
                f"{evaluation['reliability_score']:.3f} | "
                f"{status.upper():8} | "
                f"Obs "
                f"{evaluation['observation_count']:3}"
            )

        output_data = {
            "summary": {
                "total_tracks": len(
                    output_tracks
                ),
                "status_counts": dict(
                    status_counts
                ),
                "class_status_counts": (
                    class_status_counts
                ),
            },
            "tracks": output_tracks,
        }

        output_path = (
            self.output_dir
            / "track_reliability.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                output_data,
                file,
                indent=2,
            )

        return {
            "total_tracks": len(
                output_tracks
            ),
            "status_counts": dict(
                status_counts
            ),
            "class_status_counts": (
                class_status_counts
            ),
            "metadata_path": str(
                output_path
            ),
        }
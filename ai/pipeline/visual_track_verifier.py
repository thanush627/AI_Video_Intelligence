import json
from collections import Counter
from pathlib import Path

from ultralytics import YOLO


class VisualTrackVerifier:

    CLASS_GROUPS = {
        "pedestrian": {
            "pedestrian",
            "people",
        },
        "people": {
            "pedestrian",
            "people",
        },
        "car": {
            "car",
        },
        "van": {
            "van",
        },
        "truck": {
            "truck",
        },
        "bus": {
            "bus",
        },
        "bicycle": {
            "bicycle",
        },
    }

    def __init__(
        self,
        reliability_metadata_path,
        model_path,
        output_dir,
        confidence_threshold=0.20,
    ):
        self.reliability_path = Path(
            reliability_metadata_path
        )

        self.model_path = Path(
            model_path
        )

        self.output_dir = Path(
            output_dir
        )

        self.confidence_threshold = (
            confidence_threshold
        )

        if not self.reliability_path.exists():
            raise FileNotFoundError(
                f"Reliability metadata not found: "
                f"{self.reliability_path}"
            )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found: "
                f"{self.model_path}"
            )

        print(
            f"Loading model: {self.model_path}"
        )

        self.model = YOLO(
            str(self.model_path)
        )

        self.class_names = (
            self.model.names
        )

    def _allowed_classes(
        self,
        expected_class,
    ):
        return self.CLASS_GROUPS.get(
            expected_class,
            {expected_class},
        )

    def _verify_crop(
        self,
        crop_path,
        expected_class,
    ):
        crop_path = Path(
            crop_path
        )

        if not crop_path.exists():
            return {
                "verified": False,
                "expected_class": expected_class,
                "detected_classes": [],
                "best_matching_confidence": 0.0,
                "reason": "crop_not_found",
            }

        results = self.model.predict(
            source=str(crop_path),
            conf=self.confidence_threshold,
            imgsz=640,
            verbose=False,
            device="cpu",
        )

        if not results:
            return {
                "verified": False,
                "expected_class": expected_class,
                "detected_classes": [],
                "best_matching_confidence": 0.0,
                "reason": "no_prediction_result",
            }

        result = results[0]

        detections = []

        allowed_classes = self._allowed_classes(
            expected_class
        )

        best_matching_confidence = 0.0

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                class_name = (
                    self.class_names[
                        class_id
                    ]
                )

                detections.append(
                    {
                        "class_name": class_name,
                        "confidence": round(
                            confidence,
                            4,
                        ),
                    }
                )

                if (
                    class_name
                    in allowed_classes
                ):
                    best_matching_confidence = max(
                        best_matching_confidence,
                        confidence,
                    )

        verified = (
            best_matching_confidence
            >= self.confidence_threshold
        )

        if verified:
            reason = "expected_class_redetected"

        elif detections:
            reason = (
                "different_class_detected"
            )

        else:
            reason = (
                "no_object_redetected"
            )

        return {
            "verified": verified,
            "expected_class": expected_class,
            "detected_classes": detections,
            "best_matching_confidence": round(
                best_matching_confidence,
                4,
            ),
            "reason": reason,
        }

    def _verify_track(
        self,
        track,
    ):
        expected_class = track[
            "class_name"
        ]

        crops = track.get(
            "selected_crops",
            []
        )

        crop_results = []

        verified_count = 0

        confidence_sum = 0.0

        for crop in crops:

            crop_path = crop.get(
                "crop_path"
            )

            if not crop_path:
                continue

            verification = (
                self._verify_crop(
                    crop_path,
                    expected_class,
                )
            )

            crop_results.append(
                {
                    "crop_path": crop_path,
                    **verification,
                }
            )

            if verification["verified"]:

                verified_count += 1

                confidence_sum += (
                    verification[
                        "best_matching_confidence"
                    ]
                )

        total_crops = len(
            crop_results
        )

        if total_crops == 0:

            verification_ratio = 0.0

        else:

            verification_ratio = (
                verified_count
                / total_crops
            )

        if verified_count == 0:

            average_match_confidence = 0.0

        else:

            average_match_confidence = (
                confidence_sum
                / verified_count
            )

        visual_score = (
            0.65 * verification_ratio
            + 0.35 * average_match_confidence
        )

        return {
            "visual_score": round(
                visual_score,
                4,
            ),
            "verified_crop_count": (
                verified_count
            ),
            "total_crop_count": (
                total_crops
            ),
            "verification_ratio": round(
                verification_ratio,
                4,
            ),
            "average_match_confidence": round(
                average_match_confidence,
                4,
            ),
            "crop_verifications": (
                crop_results
            ),
        }

    @staticmethod
    def _combine_status(
        numerical_status,
        numerical_score,
        visual_score,
        verified_crop_count,
    ):
        if numerical_status == "reject":

            return (
                "reject",
                "numerical_reliability_failed",
            )

        if verified_crop_count == 0:

            return (
                "reject",
                "no_crop_visually_verified",
            )

        final_score = (
            0.55 * numerical_score
            + 0.45 * visual_score
        )

        if (
            numerical_status == "reliable"
            and visual_score >= 0.55
            and final_score >= 0.60
        ):

            return (
                "reliable",
                "numerical_and_visual_checks_passed",
            )

        if (
            visual_score >= 0.30
            and final_score >= 0.40
        ):

            return (
                "review",
                "partial_visual_support",
            )

        return (
            "reject",
            "insufficient_visual_support",
        )

    def analyze(self):

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            self.reliability_path,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        output_tracks = {}

        status_counts = Counter()

        for track_key, track in (
            data["tracks"].items()
        ):

            numerical = track[
                "reliability"
            ]

            visual = self._verify_track(
                track
            )

            final_status, reason = (
                self._combine_status(
                    numerical_status=(
                        numerical["status"]
                    ),
                    numerical_score=(
                        numerical[
                            "reliability_score"
                        ]
                    ),
                    visual_score=(
                        visual["visual_score"]
                    ),
                    verified_crop_count=(
                        visual[
                            "verified_crop_count"
                        ]
                    ),
                )
            )

            final_score = (
                0.55
                * numerical[
                    "reliability_score"
                ]
                + 0.45
                * visual[
                    "visual_score"
                ]
            )

            status_counts[
                final_status
            ] += 1

            output_tracks[
                track_key
            ] = {
                "track_id": track[
                    "track_id"
                ],
                "class_name": track[
                    "class_name"
                ],
                "numerical_reliability": (
                    numerical
                ),
                "visual_verification": (
                    visual
                ),
                "final_score": round(
                    final_score,
                    4,
                ),
                "final_status": (
                    final_status
                ),
                "final_reason": reason,
                "selected_crops": track.get(
                    "selected_crops",
                    [],
                ),
            }

            print(
                f"Track "
                f"{track['track_id']:3} | "
                f"{track['class_name']:10} | "
                f"Num "
                f"{numerical['reliability_score']:.3f} | "
                f"Visual "
                f"{visual['visual_score']:.3f} | "
                f"Verified "
                f"{visual['verified_crop_count']}/"
                f"{visual['total_crop_count']} | "
                f"{final_status.upper()}"
            )

        output_data = {
            "summary": {
                "total_tracks": len(
                    output_tracks
                ),
                "status_counts": dict(
                    status_counts
                ),
            },
            "tracks": output_tracks,
        }

        output_path = (
            self.output_dir
            / "visual_track_verification.json"
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
            "metadata_path": str(
                output_path
            ),
        }
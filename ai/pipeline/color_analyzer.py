import cv2
import json
import numpy as np

from collections import defaultdict
from pathlib import Path


class ColorAnalyzer:

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

    COLOR_PROTOTYPES = {
        "black": np.array([25, 25, 25]),
        "white": np.array([225, 225, 225]),
        "gray": np.array([125, 125, 125]),
        "red": np.array([190, 45, 45]),
        "orange": np.array([220, 120, 35]),
        "yellow": np.array([220, 200, 45]),
        "green": np.array([55, 140, 70]),
        "blue": np.array([55, 95, 175]),
        "purple": np.array([125, 65, 150]),
        "pink": np.array([210, 115, 155]),
        "brown": np.array([105, 70, 45]),
        "beige": np.array([190, 175, 140]),
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
                f"Metadata not found: "
                f"{self.metadata_path}"
            )

    @staticmethod
    def _crop_region(
        image,
        top,
        bottom,
        left,
        right,
    ):
        height, width = image.shape[:2]

        y1 = int(height * top)
        y2 = int(height * bottom)

        x1 = int(width * left)
        x2 = int(width * right)

        region = image[
            max(0, y1):min(height, y2),
            max(0, x1):min(width, x2),
        ]

        if region.size == 0:
            return image

        return region

    @staticmethod
    def _image_quality(image):

        if image is None or image.size == 0:
            return 0.0

        height, width = image.shape[:2]

        if width < 40 or height < 40:
            return 0.0

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        sharpness = cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()

        sharpness_score = min(
            sharpness / 500.0,
            1.0,
        )

        size_score = min(
            (width * height) / 40000.0,
            1.0,
        )

        return (
            0.65 * sharpness_score
            + 0.35 * size_score
        )

    @staticmethod
    def _foreground_mask(image):

        height, width = image.shape[:2]

        if width < 20 or height < 20:
            return np.full(
                (height, width),
                255,
                dtype=np.uint8,
            )

        mask = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        rectangle = (
            max(1, int(width * 0.05)),
            max(1, int(height * 0.05)),
            max(1, int(width * 0.90)),
            max(1, int(height * 0.90)),
        )

        background_model = np.zeros(
            (1, 65),
            np.float64,
        )

        foreground_model = np.zeros(
            (1, 65),
            np.float64,
        )

        try:

            cv2.grabCut(
                image,
                mask,
                rectangle,
                background_model,
                foreground_model,
                3,
                cv2.GC_INIT_WITH_RECT,
            )

            foreground = np.where(
                (
                    (mask == cv2.GC_FGD)
                    | (mask == cv2.GC_PR_FGD)
                ),
                255,
                0,
            ).astype(np.uint8)

            foreground_ratio = (
                cv2.countNonZero(foreground)
                / max(height * width, 1)
            )

            if (
                foreground_ratio < 0.10
                or foreground_ratio > 0.95
            ):
                return np.full(
                    (height, width),
                    255,
                    dtype=np.uint8,
                )

            return foreground

        except cv2.error:

            return np.full(
                (height, width),
                255,
                dtype=np.uint8,
            )

    @staticmethod
    def _color_name_from_hsv(h, s, v):

        if v < 45:
            return "black"

        if s < 35:

            if v > 190:
                return "white"

            if v > 65:
                return "gray"

            return "black"

        if s < 70:

            if v > 200:
                return "white"

            if v > 80:
                return "gray"

        if h < 10 or h >= 170:
            return "red"

        if h < 22:
            return "orange"

        if h < 35:
            return "yellow"

        if h < 85:
            return "green"

        if h < 130:
            return "blue"

        if h < 160:
            return "purple"

        return "pink"

    def _analyze_pixels(
        self,
        image,
        use_foreground=True,
    ):

        if image is None or image.size == 0:

            return self._unknown_result(
                "empty_region"
            )

        resized = cv2.resize(
            image,
            (160, 160),
            interpolation=cv2.INTER_AREA,
        )

        hsv = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2HSV,
        )

        if use_foreground:

            foreground_mask = (
                self._foreground_mask(resized)
            )

        else:

            foreground_mask = np.full(
                hsv.shape[:2],
                255,
                dtype=np.uint8,
            )

        height, width = hsv.shape[:2]

        center_mask = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        cv2.rectangle(
            center_mask,
            (
                int(width * 0.08),
                int(height * 0.08),
            ),
            (
                int(width * 0.92),
                int(height * 0.92),
            ),
            255,
            -1,
        )

        valid_mask = cv2.bitwise_and(
            foreground_mask,
            center_mask,
        )

        pixels = hsv[
            valid_mask > 0
        ]

        if len(pixels) < 150:

            return self._unknown_result(
                "insufficient_pixels"
            )

        votes = defaultdict(float)

        for pixel in pixels:

            h = int(pixel[0])
            s = int(pixel[1])
            v = int(pixel[2])

            color = self._color_name_from_hsv(
                h,
                s,
                v,
            )

            weight = 1.0

            if color == "black":
                weight *= 0.45

            elif color == "gray":
                weight *= 0.85

            elif color == "white":
                weight *= 1.10

            elif s > 90:
                weight *= 1.20

            votes[color] += weight

        if not votes:

            return self._unknown_result(
                "no_color_votes"
            )

        ranked = sorted(
            votes.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        total = max(
            sum(votes.values()),
            1e-8,
        )

        normalized = {
            color: score / total
            for color, score in ranked
        }

        primary = ranked[0][0]

        secondary = (
            ranked[1][0]
            if len(ranked) > 1
            else primary
        )

        confidence = normalized[primary]

        if confidence < 0.28:

            primary = "unknown"

        return {
            "primary_color": primary,
            "secondary_color": secondary,
            "confidence": round(
                confidence,
                4,
            ),
            "color_scores": {
                color: round(score, 4)
                for color, score
                in list(
                    normalized.items()
                )[:4]
            },
            "reason": "valid",
        }

    @staticmethod
    def _unknown_result(reason):

        return {
            "primary_color": "unknown",
            "secondary_color": "unknown",
            "confidence": 0.0,
            "color_scores": {},
            "reason": reason,
        }

    def _analyze_vehicle(self, image):

        height, width = image.shape[:2]

        aspect_ratio = (
            width / max(height, 1)
        )

        if aspect_ratio >= 1.35:

            region = self._crop_region(
                image,
                0.18,
                0.72,
                0.08,
                0.92,
            )

            view_type = "side_or_wide"

        else:

            region = self._crop_region(
                image,
                0.12,
                0.68,
                0.10,
                0.90,
            )

            view_type = "front_or_rear"

        result = self._analyze_pixels(
            region,
            use_foreground=True,
        )

        result["view_type"] = view_type

        return result

    def _analyze_human(self, image):

        height, width = image.shape[:2]

        aspect_ratio = (
            height / max(width, 1)
        )

        if aspect_ratio < 1.15:

            return {
                "upper_body": (
                    self._unknown_result(
                        "poor_human_crop"
                    )
                ),
                "lower_body": (
                    self._unknown_result(
                        "poor_human_crop"
                    )
                ),
            }

        upper = self._crop_region(
            image,
            0.18,
            0.55,
            0.20,
            0.80,
        )

        lower = self._crop_region(
            image,
            0.55,
            0.94,
            0.20,
            0.80,
        )

        upper_result = self._analyze_pixels(
            upper,
            use_foreground=False,
        )

        lower_result = self._analyze_pixels(
            lower,
            use_foreground=False,
        )

        return {
            "upper_body": upper_result,
            "lower_body": lower_result,
        }

    def _analyze_bicycle(self, image):

        height, width = image.shape[:2]

        if width < 70 or height < 70:

            return self._unknown_result(
                "bicycle_too_small"
            )

        region = self._crop_region(
            image,
            0.35,
            0.92,
            0.08,
            0.92,
        )

        result = self._analyze_pixels(
            region,
            use_foreground=True,
        )

        if result["confidence"] < 0.40:

            return self._unknown_result(
                "unreliable_bicycle_color"
            )

        return result

    @staticmethod
    def _aggregate_results(
        weighted_results,
    ):

        votes = defaultdict(float)

        valid_count = 0

        for result, crop_weight in (
            weighted_results
        ):

            if (
                result["primary_color"]
                == "unknown"
            ):
                continue

            valid_count += 1

            for color, score in (
                result[
                    "color_scores"
                ].items()
            ):

                votes[color] += (
                    score * crop_weight
                )

        if not votes:

            result = ColorAnalyzer._unknown_result(
                "no_reliable_crops"
            )

            result["valid_crop_count"] = 0

            return result

        ranked = sorted(
            votes.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        total = max(
            sum(votes.values()),
            1e-8,
        )

        normalized = {
            color: score / total
            for color, score in ranked
        }

        primary = ranked[0][0]

        secondary = (
            ranked[1][0]
            if len(ranked) > 1
            else primary
        )

        confidence = normalized[primary]

        if confidence < 0.32:

            primary = "unknown"

        return {
            "primary_color": primary,
            "secondary_color": secondary,
            "confidence": round(
                confidence,
                4,
            ),
            "color_scores": {
                color: round(score, 4)
                for color, score
                in list(
                    normalized.items()
                )[:4]
            },
            "valid_crop_count": valid_count,
            "reason": "track_consensus",
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

        analyzed_tracks = {}

        final_color_counts = defaultdict(int)

        unknown_tracks = 0

        for track_key, track in (
            data["tracks"].items()
        ):

            class_name = track["class_name"]

            crop_results = []

            object_results = []

            upper_results = []

            lower_results = []

            for crop_info in (
                track["selected_crops"]
            ):

                crop_path = Path(
                    crop_info["crop_path"]
                )

                image = cv2.imread(
                    str(crop_path)
                )

                if image is None:
                    continue

                quality = self._image_quality(
                    image
                )

                selector_score = float(
                    crop_info.get(
                        "final_score",
                        0.5,
                    )
                )

                crop_weight = max(
                    0.05,
                    (
                        0.60 * quality
                        + 0.40 * selector_score
                    ),
                )

                if class_name in (
                    self.HUMAN_CLASSES
                ):

                    result = (
                        self._analyze_human(
                            image
                        )
                    )

                    upper_results.append(
                        (
                            result["upper_body"],
                            crop_weight,
                        )
                    )

                    lower_results.append(
                        (
                            result["lower_body"],
                            crop_weight,
                        )
                    )

                elif class_name in (
                    self.BICYCLE_CLASSES
                ):

                    result = (
                        self._analyze_bicycle(
                            image
                        )
                    )

                    object_results.append(
                        (
                            result,
                            crop_weight,
                        )
                    )

                else:

                    result = (
                        self._analyze_vehicle(
                            image
                        )
                    )

                    object_results.append(
                        (
                            result,
                            crop_weight,
                        )
                    )

                crop_results.append(
                    {
                        "rank": crop_info[
                            "rank"
                        ],
                        "crop_path": str(
                            crop_path
                        ),
                        "crop_quality": round(
                            quality,
                            4,
                        ),
                        "crop_weight": round(
                            crop_weight,
                            4,
                        ),
                        "analysis": result,
                    }
                )

            if not crop_results:
                continue

            if class_name in (
                self.HUMAN_CLASSES
            ):

                final_upper = (
                    self._aggregate_results(
                        upper_results
                    )
                )

                final_lower = (
                    self._aggregate_results(
                        lower_results
                    )
                )

                final_analysis = {
                    "upper_body": final_upper,
                    "lower_body": final_lower,
                }

                for result in (
                    final_upper,
                    final_lower,
                ):

                    final_color_counts[
                        result["primary_color"]
                    ] += 1

                if (
                    final_upper[
                        "primary_color"
                    ] == "unknown"
                    and final_lower[
                        "primary_color"
                    ] == "unknown"
                ):
                    unknown_tracks += 1

                summary = (
                    f"upper="
                    f"{final_upper['primary_color']} | "
                    f"lower="
                    f"{final_lower['primary_color']}"
                )

            else:

                final_object = (
                    self._aggregate_results(
                        object_results
                    )
                )

                final_analysis = {
                    "object_color": (
                        final_object
                    )
                }

                final_color_counts[
                    final_object[
                        "primary_color"
                    ]
                ] += 1

                if (
                    final_object[
                        "primary_color"
                    ]
                    == "unknown"
                ):
                    unknown_tracks += 1

                summary = (
                    f"color="
                    f"{final_object['primary_color']}"
                )

            analyzed_tracks[track_key] = {
                "track_id": track["track_id"],
                "class_name": class_name,
                "final_color_analysis": (
                    final_analysis
                ),
                "crop_color_results": (
                    crop_results
                ),
            }

            print(
                f"Track "
                f"{track['track_id']:3} | "
                f"{class_name:10} | "
                f"{summary}"
            )

        output_data = {
            "summary": {
                "analyzed_tracks": len(
                    analyzed_tracks
                ),
                "unknown_tracks": (
                    unknown_tracks
                ),
                "final_color_counts": dict(
                    final_color_counts
                ),
            },
            "tracks": analyzed_tracks,
        }

        output_path = (
            self.output_dir
            / "color_metadata.json"
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
            "analyzed_tracks": len(
                analyzed_tracks
            ),
            "unknown_tracks": unknown_tracks,
            "final_color_counts": dict(
                final_color_counts
            ),
            "metadata_path": str(
                output_path
            ),
        }
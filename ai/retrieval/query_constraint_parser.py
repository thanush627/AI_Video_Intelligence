import re


class QueryConstraintParser:

    def __init__(self):

        self.class_aliases = {
            "pedestrian": [
                "pedestrian",
                "pedestrians",
                "person",
                "persons",
                "people",
                "man",
                "men",
                "woman",
                "women",
                "human",
                "humans",
            ],
            "car": [
                "car",
                "cars",
                "automobile",
                "automobiles",
            ],
            "bus": [
                "bus",
                "buses",
            ],
            "van": [
                "van",
                "vans",
            ],
            "truck": [
                "truck",
                "trucks",
                "lorry",
                "lorries",
            ],
            "bicycle": [
                "bicycle",
                "bicycles",
                "bike",
                "bikes",
                "cycle",
                "cycles",
                "cyclist",
                "cyclists",
            ],
            "motorcycle": [
                "motorcycle",
                "motorcycles",
                "motorbike",
                "motorbikes",
            ],
        }

        self.vehicle_words = [
            "vehicle",
            "vehicles",
        ]

        self.vehicle_classes = [
            "car",
            "bus",
            "van",
            "truck",
            "bicycle",
            "motorcycle",
        ]

        self.colors = [
            "black",
            "white",
            "gray",
            "grey",
            "red",
            "blue",
            "green",
            "yellow",
            "orange",
            "brown",
            "purple",
            "pink",
        ]

        self.color_normalization = {
            "grey": "gray",
        }

        self.motion_state_patterns = {
            "stationary": [
                r"\bstationary\b",
                r"\bstanding still\b",
                r"\bnot moving\b",
                r"\bstopped\b",
                r"\bparked\b",
            ],
            "moving": [
                r"\bmoving\b",
                r"\bmoved\b",
                r"\btravelling\b",
                r"\btraveling\b",
            ],
            "slow_moving": [
                r"\bslow moving\b",
                r"\bslowly moving\b",
                r"\bmoving slowly\b",
            ],
        }

        self.direction_patterns = {
            "left_to_right": [
                r"\bleft to right\b",
                r"\bfrom left to right\b",
            ],
            "right_to_left": [
                r"\bright to left\b",
                r"\bfrom right to left\b",
            ],
            "top_to_bottom": [
                r"\btop to bottom\b",
                r"\bfrom top to bottom\b",
            ],
            "bottom_to_top": [
                r"\bbottom to top\b",
                r"\bfrom bottom to top\b",
            ],
            "top_left_to_bottom_right": [
                r"\btop left to bottom right\b",
                r"\bfrom top left to bottom right\b",
            ],
            "top_right_to_bottom_left": [
                r"\btop right to bottom left\b",
                r"\bfrom top right to bottom left\b",
            ],
            "bottom_left_to_top_right": [
                r"\bbottom left to top right\b",
                r"\bfrom bottom left to top right\b",
            ],
            "bottom_right_to_top_left": [
                r"\bbottom right to top left\b",
                r"\bfrom bottom right to top left\b",
            ],
        }

        self.region_patterns = {
            "left_top": [
                r"\bleft top\b",
                r"\btop left region\b",
                r"\btop left side\b",
            ],
            "center_top": [
                r"\bcenter top\b",
                r"\btop center\b",
                r"\btop middle\b",
            ],
            "right_top": [
                r"\bright top\b",
                r"\btop right region\b",
                r"\btop right side\b",
            ],
            "left_middle": [
                r"\bleft middle\b",
                r"\bmiddle left\b",
            ],
            "center_middle": [
                r"\bcenter middle\b",
                r"\bmiddle center\b",
                r"\bcenter of the frame\b",
                r"\bcentre of the frame\b",
            ],
            "right_middle": [
                r"\bright middle\b",
                r"\bmiddle right\b",
            ],
            "left_bottom": [
                r"\bleft bottom\b",
                r"\bbottom left region\b",
                r"\bbottom left side\b",
            ],
            "center_bottom": [
                r"\bcenter bottom\b",
                r"\bbottom center\b",
                r"\bbottom middle\b",
            ],
            "right_bottom": [
                r"\bright bottom\b",
                r"\bbottom right region\b",
                r"\bbottom right side\b",
            ],
        }

        self.broad_region_patterns = {
            "left": [
                r"\bleft side\b",
                r"\bleft region\b",
                r"\bon the left\b",
            ],
            "center": [
                r"\bcenter region\b",
                r"\bcentre region\b",
                r"\bin the center\b",
                r"\bin the centre\b",
            ],
            "right": [
                r"\bright side\b",
                r"\bright region\b",
                r"\bon the right\b",
            ],
            "top": [
                r"\btop region\b",
                r"\bat the top\b",
            ],
            "bottom": [
                r"\bbottom region\b",
                r"\bat the bottom\b",
            ],
        }


    def _clean_query(
        self,
        query,
    ):

        query = str(
            query
        ).lower().strip()

        query = query.replace(
            "-",
            " ",
        )

        query = re.sub(
            r"\s+",
            " ",
            query,
        )

        return query


    def _contains_phrase(
        self,
        query,
        phrase,
    ):

        pattern = (
            r"\b"
            + re.escape(
                phrase
            )
            + r"\b"
        )

        return bool(
            re.search(
                pattern,
                query,
                flags=re.IGNORECASE,
            )
        )


    def _extract_classes(
        self,
        query,
    ):

        detected_classes = []

        vehicle_requested = any(
            self._contains_phrase(
                query,
                word,
            )
            for word
            in self.vehicle_words
        )

        if vehicle_requested:

            return {
                "object_classes": [],
                "class_group": "vehicle",
                "allowed_classes": list(
                    self.vehicle_classes
                ),
            }

        for (
            class_name,
            aliases,
        ) in self.class_aliases.items():

            found = any(
                self._contains_phrase(
                    query,
                    alias,
                )
                for alias
                in aliases
            )

            if found:

                detected_classes.append(
                    class_name
                )

        return {
            "object_classes": (
                detected_classes
            ),
            "class_group": None,
            "allowed_classes": [],
        }


    def _extract_colors(
        self,
        query,
    ):

        detected_colors = []

        for color in self.colors:

            if self._contains_phrase(
                query,
                color,
            ):

                normalized_color = (
                    self.color_normalization.get(
                        color,
                        color,
                    )
                )

                if (
                    normalized_color
                    not in detected_colors
                ):

                    detected_colors.append(
                        normalized_color
                    )

        return detected_colors


    def _extract_pattern_values(
        self,
        query,
        pattern_map,
    ):

        detected_values = []

        for (
            value,
            patterns,
        ) in pattern_map.items():

            for pattern in patterns:

                if re.search(
                    pattern,
                    query,
                    flags=re.IGNORECASE,
                ):

                    detected_values.append(
                        value
                    )

                    break

        return detected_values


    def _extract_motion_states(
        self,
        query,
    ):

        states = self._extract_pattern_values(
            query,
            self.motion_state_patterns,
        )

        if (
            "slow_moving"
            in states
            and "moving"
            in states
        ):

            states.remove(
                "moving"
            )

        return states


    def _extract_motion_directions(
        self,
        query,
    ):

        return self._extract_pattern_values(
            query,
            self.direction_patterns,
        )


    def _extract_spatial_regions(
        self,
        query,
    ):

        exact_regions = (
            self._extract_pattern_values(
                query,
                self.region_patterns,
            )
        )

        if exact_regions:

            return {
                "spatial_regions": (
                    exact_regions
                ),
                "broad_spatial_regions": [],
            }

        broad_regions = (
            self._extract_pattern_values(
                query,
                self.broad_region_patterns,
            )
        )

        return {
            "spatial_regions": [],
            "broad_spatial_regions": (
                broad_regions
            ),
        }


    def parse(
        self,
        query,
    ):

        cleaned_query = self._clean_query(
            query
        )

        class_data = self._extract_classes(
            cleaned_query
        )

        spatial_data = (
            self._extract_spatial_regions(
                cleaned_query
            )
        )

        constraints = {
            "object_classes": (
                class_data[
                    "object_classes"
                ]
            ),
            "class_group": (
                class_data[
                    "class_group"
                ]
            ),
            "allowed_classes": (
                class_data[
                    "allowed_classes"
                ]
            ),
            "colors": (
                self._extract_colors(
                    cleaned_query
                )
            ),
            "motion_states": (
                self._extract_motion_states(
                    cleaned_query
                )
            ),
            "motion_directions": (
                self._extract_motion_directions(
                    cleaned_query
                )
            ),
            "spatial_regions": (
                spatial_data[
                    "spatial_regions"
                ]
            ),
            "broad_spatial_regions": (
                spatial_data[
                    "broad_spatial_regions"
                ]
            ),
        }

        has_constraints = any(
            [
                bool(
                    constraints[
                        "object_classes"
                    ]
                ),
                bool(
                    constraints[
                        "class_group"
                    ]
                ),
                bool(
                    constraints[
                        "colors"
                    ]
                ),
                bool(
                    constraints[
                        "motion_states"
                    ]
                ),
                bool(
                    constraints[
                        "motion_directions"
                    ]
                ),
                bool(
                    constraints[
                        "spatial_regions"
                    ]
                ),
                bool(
                    constraints[
                        "broad_spatial_regions"
                    ]
                ),
            ]
        )

        return {
            "original_query": str(
                query
            ).strip(),
            "normalized_query": (
                cleaned_query
            ),
            "has_constraints": (
                has_constraints
            ),
            "constraints": (
                constraints
            ),
        }
import re


class TemporalQueryParser:

    def __init__(self):
        pass


    def _safe_float(
        self,
        value,
    ):
        try:
            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return None


    def _clean_query(
        self,
        query,
    ):
        query = str(
            query
        ).strip()

        query = re.sub(
            r"\s+",
            " ",
            query,
        )

        return query


    def _normalize_unit(
        self,
        value,
        unit,
    ):
        value = self._safe_float(
            value
        )

        if value is None:
            return None

        unit = str(
            unit
        ).lower()

        if unit.startswith(
            "min"
        ):
            return value * 60.0

        return value


    def _extract_between(
        self,
        query,
    ):
        pattern = (
            r"\bbetween\s+"
            r"(\d+(?:\.\d+)?)\s*"
            r"(seconds?|secs?|s|minutes?|mins?|m)?"
            r"\s+and\s+"
            r"(\d+(?:\.\d+)?)\s*"
            r"(seconds?|secs?|s|minutes?|mins?|m)?"
        )

        match = re.search(
            pattern,
            query,
            flags=re.IGNORECASE,
        )

        if not match:
            return None

        start_value = match.group(1)
        start_unit = match.group(2)

        end_value = match.group(3)
        end_unit = match.group(4)

        if start_unit is None:
            start_unit = (
                end_unit
                or "seconds"
            )

        if end_unit is None:
            end_unit = (
                start_unit
                or "seconds"
            )

        start_time = self._normalize_unit(
            start_value,
            start_unit,
        )

        end_time = self._normalize_unit(
            end_value,
            end_unit,
        )

        if (
            start_time is None
            or end_time is None
        ):
            return None

        start_time, end_time = sorted(
            [
                start_time,
                end_time,
            ]
        )

        return {
            "type": "range",
            "start_time_seconds": (
                start_time
            ),
            "end_time_seconds": (
                end_time
            ),
            "matched_text": (
                match.group(0)
            ),
        }


    def _extract_from_to(
        self,
        query,
    ):
        pattern = (
            r"\bfrom\s+"
            r"(\d+(?:\.\d+)?)\s*"
            r"(seconds?|secs?|s|minutes?|mins?|m)?"
            r"\s+(?:to|-)\s+"
            r"(\d+(?:\.\d+)?)\s*"
            r"(seconds?|secs?|s|minutes?|mins?|m)?"
        )

        match = re.search(
            pattern,
            query,
            flags=re.IGNORECASE,
        )

        if not match:
            return None

        start_unit = (
            match.group(2)
            or match.group(4)
            or "seconds"
        )

        end_unit = (
            match.group(4)
            or match.group(2)
            or "seconds"
        )

        start_time = self._normalize_unit(
            match.group(1),
            start_unit,
        )

        end_time = self._normalize_unit(
            match.group(3),
            end_unit,
        )

        start_time, end_time = sorted(
            [
                start_time,
                end_time,
            ]
        )

        return {
            "type": "range",
            "start_time_seconds": (
                start_time
            ),
            "end_time_seconds": (
                end_time
            ),
            "matched_text": (
                match.group(0)
            ),
        }


    def _extract_after(
        self,
        query,
    ):
        pattern = (
            r"\b(?:after|later\s+than)\s+"
            r"(\d+(?:\.\d+)?)\s*"
            r"(seconds?|secs?|s|minutes?|mins?|m)?"
        )

        match = re.search(
            pattern,
            query,
            flags=re.IGNORECASE,
        )

        if not match:
            return None

        time_value = self._normalize_unit(
            match.group(1),
            match.group(2)
            or "seconds",
        )

        return {
            "type": "after",
            "start_time_seconds": (
                time_value
            ),
            "end_time_seconds": None,
            "matched_text": (
                match.group(0)
            ),
        }


    def _extract_before(
        self,
        query,
    ):
        pattern = (
            r"\b(?:before|earlier\s+than)\s+"
            r"(\d+(?:\.\d+)?)\s*"
            r"(seconds?|secs?|s|minutes?|mins?|m)?"
        )

        match = re.search(
            pattern,
            query,
            flags=re.IGNORECASE,
        )

        if not match:
            return None

        time_value = self._normalize_unit(
            match.group(1),
            match.group(2)
            or "seconds",
        )

        return {
            "type": "before",
            "start_time_seconds": None,
            "end_time_seconds": (
                time_value
            ),
            "matched_text": (
                match.group(0)
            ),
        }


    def _extract_first(
        self,
        query,
    ):
        pattern = (
            r"\b(?:in\s+)?the\s+first\s+"
            r"(\d+(?:\.\d+)?)\s*"
            r"(seconds?|secs?|s|minutes?|mins?|m)?"
        )

        match = re.search(
            pattern,
            query,
            flags=re.IGNORECASE,
        )

        if not match:
            return None

        end_time = self._normalize_unit(
            match.group(1),
            match.group(2)
            or "seconds",
        )

        return {
            "type": "range",
            "start_time_seconds": 0.0,
            "end_time_seconds": (
                end_time
            ),
            "matched_text": (
                match.group(0)
            ),
        }


    def parse(
        self,
        query,
    ):
        original_query = self._clean_query(
            query
        )

        temporal_data = None

        extractors = [
            self._extract_between,
            self._extract_from_to,
            self._extract_first,
            self._extract_after,
            self._extract_before,
        ]

        for extractor in extractors:

            temporal_data = extractor(
                original_query
            )

            if temporal_data:
                break

        if temporal_data is None:

            return {
                "original_query": (
                    original_query
                ),
                "semantic_query": (
                    original_query
                ),
                "has_temporal_filter": False,
                "temporal_filter": None,
            }

        matched_text = temporal_data.get(
            "matched_text",
            "",
        )

        semantic_query = re.sub(
            re.escape(
                matched_text
            ),
            " ",
            original_query,
            flags=re.IGNORECASE,
        )

        semantic_query = re.sub(
            r"\s+",
            " ",
            semantic_query,
        ).strip()

        if not semantic_query:
            semantic_query = "events"

        temporal_data.pop(
            "matched_text",
            None,
        )

        return {
            "original_query": (
                original_query
            ),
            "semantic_query": (
                semantic_query
            ),
            "has_temporal_filter": True,
            "temporal_filter": (
                temporal_data
            ),
        }
from datetime import timedelta


class RetrievalResponseBuilder:

    def __init__(self):
        pass


    def _safe_float(
        self,
        value,
        default=0.0,
    ):
        try:
            return float(value)

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
            return int(value)

        except (
            TypeError,
            ValueError,
        ):
            return default


    def _format_timestamp(
        self,
        seconds,
    ):
        seconds = max(
            0.0,
            self._safe_float(seconds),
        )

        total_milliseconds = round(
            seconds * 1000
        )

        hours = (
            total_milliseconds
            // 3_600_000
        )

        remaining = (
            total_milliseconds
            % 3_600_000
        )

        minutes = (
            remaining
            // 60_000
        )

        remaining = (
            remaining
            % 60_000
        )

        whole_seconds = (
            remaining
            // 1000
        )

        milliseconds = (
            remaining
            % 1000
        )

        if hours > 0:
            return (
                f"{hours:02d}:"
                f"{minutes:02d}:"
                f"{whole_seconds:02d}."
                f"{milliseconds:03d}"
            )

        return (
            f"{minutes:02d}:"
            f"{whole_seconds:02d}."
            f"{milliseconds:03d}"
        )


    def _parse_csv_values(
        self,
        value,
    ):
        if value is None:
            return []

        if isinstance(
            value,
            (list, tuple, set),
        ):
            return [
                str(item).strip()
                for item in value
                if str(item).strip()
            ]

        text = str(value).strip()

        if not text:
            return []

        return [
            item.strip()
            for item in text.split(",")
            if item.strip()
        ]


    def _parse_track_ids(
        self,
        value,
    ):
        values = self._parse_csv_values(
            value
        )

        track_ids = []

        for item in values:
            try:
                track_ids.append(
                    int(item)
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

        return track_ids


    def _build_event_result(
        self,
        result,
    ):
        start_time = self._safe_float(
            result.get(
                "start_time_seconds"
            )
        )

        end_time = self._safe_float(
            result.get(
                "end_time_seconds"
            )
        )

        if end_time < start_time:
            start_time, end_time = (
                end_time,
                start_time,
            )

        duration = max(
            0.0,
            end_time - start_time,
        )

        stored_duration = result.get(
            "duration_seconds"
        )

        if stored_duration is not None:
            stored_duration = (
                self._safe_float(
                    stored_duration
                )
            )

            if stored_duration >= 0:
                duration = stored_duration

        event_type = str(
            result.get(
                "event_type",
                "",
            )
        ).strip()

        class_names = (
            self._parse_csv_values(
                result.get(
                    "class_names"
                )
            )
        )

        track_ids = (
            self._parse_track_ids(
                result.get(
                    "track_ids"
                )
            )
        )

        document = str(
            result.get(
                "document",
                "",
            )
        ).strip()

        return {
            "rank": self._safe_int(
                result.get(
                    "rank"
                ),
                default=0,
            ),
            "event_id": str(
                result.get(
                    "event_id",
                    "",
                )
            ),
            "event_type": event_type,
            "description": document,
            "class_names": class_names,
            "track_ids": track_ids,
            "start_time_seconds": round(
                start_time,
                4,
            ),
            "end_time_seconds": round(
                end_time,
                4,
            ),
            "duration_seconds": round(
                duration,
                4,
            ),
            "start_timestamp": (
                self._format_timestamp(
                    start_time
                )
            ),
            "end_timestamp": (
                self._format_timestamp(
                    end_time
                )
            ),
            "similarity": round(
                self._safe_float(
                    result.get(
                        "similarity"
                    )
                ),
                6,
            ),
            "hybrid_score": round(
                self._safe_float(
                    result.get(
                        "hybrid_score"
                    )
                ),
                6,
            ),
            "rerank_adjustment": round(
                self._safe_float(
                    result.get(
                        "rerank_adjustment"
                    )
                ),
                6,
            ),
            "quality_score": round(
                self._safe_float(
                    result.get(
                        "quality_score"
                    )
                ),
                6,
            ),
            "quality_label": str(
                result.get(
                    "quality_label",
                    "",
                )
            ),
        }


    def _build_summary(
        self,
        query,
        intent,
        results,
    ):
        result_count = len(
            results
        )

        if result_count == 0:
            return (
                f'No matching events were found '
                f'for "{query}".'
            )

        if result_count == 1:
            best_result = results[0]

            return (
                f'Found 1 matching event for '
                f'"{query}". '
                f'The best match is: '
                f'{best_result["description"]}'
            )

        best_result = results[0]

        return (
            f'Found {result_count} matching events '
            f'for "{query}". '
            f'The best match is: '
            f'{best_result["description"]}'
        )


    def _build_best_match(
        self,
        results,
    ):
        if not results:
            return None

        best_result = dict(
            results[0]
        )

        return best_result


    def build(
        self,
        orchestrator_response,
    ):
        if not isinstance(
            orchestrator_response,
            dict,
        ):
            raise TypeError(
                "orchestrator_response must be a dictionary."
            )

        query = str(
            orchestrator_response.get(
                "query",
                "",
            )
        ).strip()

        intent = str(
            orchestrator_response.get(
                "intent",
                "general_event",
            )
        ).strip()

        raw_results = (
            orchestrator_response.get(
                "results",
                [],
            )
        )

        if not isinstance(
            raw_results,
            list,
        ):
            raw_results = []

        results = [
            self._build_event_result(
                result
            )
            for result in raw_results
            if isinstance(
                result,
                dict,
            )
        ]

        result_count = len(
            results
        )

        best_match = (
            self._build_best_match(
                results
            )
        )

        summary = self._build_summary(
            query=query,
            intent=intent,
            results=results,
        )

        return {
            "success": True,
            "query": query,
            "intent": intent,
            "intent_confidence": round(
                self._safe_float(
                    orchestrator_response.get(
                        "intent_confidence"
                    )
                ),
                4,
            ),
            "intent_terms": list(
                orchestrator_response.get(
                    "intent_terms",
                    [],
                )
                or []
            ),
            "match_found": (
                result_count > 0
            ),
            "result_count": (
                result_count
            ),
            "summary": summary,
            "best_match": best_match,
            "results": results,
        }
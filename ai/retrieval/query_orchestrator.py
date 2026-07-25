import re


class QueryOrchestrator:

    RELATIONSHIP_TERMS = {
        "near",
        "nearby",
        "close to",
        "close together",
        "next to",
        "beside",
        "together",
        "overlap",
        "overlapping",
        "intersect",
        "relationship",
        "interacting",
        "interaction",
        "with each other",
    }

    OBJECT_TERMS = {
        "car",
        "cars",
        "bus",
        "buses",
        "van",
        "vans",
        "truck",
        "trucks",
        "bicycle",
        "bicycles",
        "bike",
        "bikes",
        "motorcycle",
        "motorcycles",
        "pedestrian",
        "pedestrians",
        "person",
        "people",
        "vehicle",
        "vehicles",
    }


    def __init__(
        self,
        search_engine,
    ):
        self.search_engine = search_engine


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


    def _contains_phrase(
        self,
        query,
        phrase,
    ):
        query = query.lower()
        phrase = phrase.lower()

        if " " in phrase:
            return phrase in query

        return bool(
            re.search(
                rf"\b{re.escape(phrase)}\b",
                query,
            )
        )


    def detect_intent(
        self,
        query,
    ):
        query = self._clean_query(
            query
        )

        query_lower = query.lower()

        relationship_matches = [
            term
            for term in self.RELATIONSHIP_TERMS
            if self._contains_phrase(
                query_lower,
                term,
            )
        ]

        if relationship_matches:
            return {
                "intent": (
                    "relationship_event"
                ),
                "confidence": 1.0,
                "matched_terms": sorted(
                    relationship_matches
                ),
            }

        object_matches = [
            term
            for term in self.OBJECT_TERMS
            if self._contains_phrase(
                query_lower,
                term,
            )
        ]

        if object_matches:
            return {
                "intent": (
                    "object_event"
                ),
                "confidence": 1.0,
                "matched_terms": sorted(
                    object_matches
                ),
            }

        return {
            "intent": (
                "general_event"
            ),
            "confidence": 0.5,
            "matched_terms": [],
        }


    def _event_type(
        self,
        result,
    ):
        event_type = result.get(
            "event_type"
        )

        if event_type:
            return str(
                event_type
            ).strip().lower()

        metadata = result.get(
            "metadata",
            {},
        )

        if isinstance(
            metadata,
            dict,
        ):
            event_type = metadata.get(
                "event_type"
            )

        if event_type:
            return str(
                event_type
            ).strip().lower()

        return ""


    def _filter_by_intent(
        self,
        results,
        intent,
    ):
        if intent == "object_event":

            object_results = [
                result
                for result in results
                if self._event_type(
                    result
                )
                == "track_semantic_event"
            ]

            return object_results

        if intent == "relationship_event":

            relationship_results = [
                result
                for result in results
                if self._event_type(
                    result
                )
                == "relationship_event"
            ]

            return relationship_results

        return results


    def _deduplicate(
        self,
        results,
    ):
        unique_results = []

        seen_event_ids = set()

        for result in results:

            event_id = str(
                result.get(
                    "event_id",
                    "",
                )
            )

            if not event_id:
                continue

            if event_id in seen_event_ids:
                continue

            seen_event_ids.add(
                event_id
            )

            unique_results.append(
                result
            )

        return unique_results


    def _normalize_result(
        self,
        result,
        rank,
    ):
        normalized = dict(
            result
        )

        normalized["rank"] = rank

        return normalized


    def search(
        self,
        query,
        top_k=5,
    ):
        query = self._clean_query(
            query
        )

        if not query:
            return {
                "query": query,
                "intent": (
                    "general_event"
                ),
                "intent_confidence": 0.0,
                "intent_terms": [],
                "result_count": 0,
                "results": [],
            }

        intent_data = self.detect_intent(
            query
        )

        intent = intent_data[
            "intent"
        ]

        retrieval_k = max(
            top_k * 5,
            25,
        )

        raw_results = (
            self.search_engine.search(
                query=query,
                top_k=retrieval_k,
            )
        )

        if raw_results is None:
            raw_results = []

        filtered_results = (
            self._filter_by_intent(
                results=raw_results,
                intent=intent,
            )
        )

        filtered_results = (
            self._deduplicate(
                filtered_results
            )
        )

        final_results = (
            filtered_results[
                :top_k
            ]
        )

        final_results = [
            self._normalize_result(
                result=result,
                rank=index,
            )
            for index, result in enumerate(
                final_results,
                start=1,
            )
        ]

        return {
            "query": query,
            "intent": intent,
            "intent_confidence": (
                intent_data[
                    "confidence"
                ]
            ),
            "intent_terms": (
                intent_data[
                    "matched_terms"
                ]
            ),
            "raw_result_count": len(
                raw_results
            ),
            "filtered_result_count": len(
                filtered_results
            ),
            "result_count": len(
                final_results
            ),
            "results": final_results,
        }
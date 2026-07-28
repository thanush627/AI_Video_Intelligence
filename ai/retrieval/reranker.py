from typing import Dict, List


class ReRanker:
    """
    Re-ranks hybrid retrieval results using weighted scoring
    and removes duplicate events.
    """

    def __init__(
        self,
        similarity_weight: float = 0.40,
        metadata_weight: float = 0.30,
        confidence_weight: float = 0.20,
        temporal_weight: float = 0.10,
    ):

        self.similarity_weight = similarity_weight
        self.metadata_weight = metadata_weight
        self.confidence_weight = confidence_weight
        self.temporal_weight = temporal_weight

    @staticmethod
    def _safe_float(value, default=0.0):

        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _metadata_score(self, result: Dict) -> float:

        metadata = result.get("metadata", {})

        score = metadata.get("metadata_score", 1.0)

        return self._safe_float(score, 1.0)

    def _confidence_score(self, result: Dict) -> float:

        metadata = result.get("metadata", {})

        score = metadata.get("confidence", 1.0)

        return self._safe_float(score, 1.0)

    def _temporal_score(self, result: Dict) -> float:

        metadata = result.get("metadata", {})

        score = metadata.get("temporal_score", 1.0)

        return self._safe_float(score, 1.0)

    def _compute_score(self, result: Dict) -> float:

        similarity = self._safe_float(
            result.get("vector_score", result.get("final_score", 0.0))
        )

        metadata_score = self._metadata_score(result)

        confidence = self._confidence_score(result)

        temporal = self._temporal_score(result)

        final_score = (
            similarity * self.similarity_weight
            + metadata_score * self.metadata_weight
            + confidence * self.confidence_weight
            + temporal * self.temporal_weight
        )

        return round(final_score, 4)

    def rerank(
        self,
        candidates: List[Dict]
    ) -> List[Dict]:

        ranked = []

        for candidate in candidates:

            candidate = candidate.copy()

            candidate["ranking_score"] = self._compute_score(candidate)

            ranked.append(candidate)

        ranked.sort(
            key=lambda x: x["ranking_score"],
            reverse=True
        )

        unique_results = []
        seen = set()

        for result in ranked:

            event_id = result.get("event_id")

            if event_id in seen:
                continue

            seen.add(event_id)

            unique_results.append(result)

        return unique_results


if __name__ == "__main__":

    candidates = [
        {
            "event_id": 1,
            "vector_score": 0.91,
            "metadata": {
                "confidence": 0.95,
                "metadata_score": 1.0,
                "temporal_score": 0.90,
            },
        },
        {
            "event_id": 2,
            "vector_score": 0.82,
            "metadata": {
                "confidence": 0.87,
                "metadata_score": 0.90,
                "temporal_score": 0.85,
            },
        },
        {
            "event_id": 1,
            "vector_score": 0.88,
            "metadata": {
                "confidence": 0.80,
                "metadata_score": 0.85,
                "temporal_score": 0.80,
            },
        },
    ]

    reranker = ReRanker()

    results = reranker.rerank(candidates)

    print("=" * 80)

    for result in results:
        print(result)
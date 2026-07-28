"""
Phase 7

Metadata Retriever

Retrieves matching tracks from semantic metadata.
"""

from typing import List

from .planner import RetrievalPlan
from .retrieval_result import RetrievalResult
from .data_loader import RetrievalDataLoader


class MetadataRetriever:

    def __init__(self, loader: RetrievalDataLoader):
        self.loader = loader

    def retrieve(self, plan: RetrievalPlan) -> List[RetrievalResult]:

        results: List[RetrievalResult] = []

        tracks = self.loader.semantic_metadata.get("tracks", [])

        for track in tracks:

            # --------------------------------------------------
            # Object Filter
            # --------------------------------------------------

            object_type = str(
                track.get("object_type", "")
            ).lower()

            if (
                plan.object_type
                and object_type != plan.object_type.lower()
            ):
                continue

            # --------------------------------------------------
            # Color Filter
            # --------------------------------------------------

            if plan.color:

                colors = []

                color_data = track.get("colors", {})

                if isinstance(color_data, dict):

                    upper = color_data.get("upper_body", "")
                    lower = color_data.get("lower_body", "")

                    if upper:
                        colors.append(upper.lower())

                    if lower:
                        colors.append(lower.lower())

                elif isinstance(color_data, list):

                    colors.extend(
                        str(c).lower()
                        for c in color_data
                        if c
                    )

                if plan.color.lower() not in colors:
                    continue

            # --------------------------------------------------
            # Action Filter
            # --------------------------------------------------

            if plan.action:

                action = str(
                    track.get("action", "")
                ).lower()

                if action != plan.action.lower():
                    continue

            # --------------------------------------------------
            # Attribute Filter
            # --------------------------------------------------

            if plan.attributes:

                attributes = track.get("attributes", [])

                if isinstance(attributes, str):
                    attributes = [attributes]

                attributes = [
                    str(a).lower()
                    for a in attributes
                    if a
                ]

                matched = all(
                    attr.lower() in attributes
                    for attr in plan.attributes
                )

                if not matched:
                    continue

            # --------------------------------------------------
            # Confidence
            # --------------------------------------------------

            confidence = track.get("confidence", {})

            object_confidence = 0.0

            if isinstance(confidence, dict):

                object_confidence = float(
                    confidence.get("object", 0.0)
                )

            elif isinstance(confidence, (int, float)):

                object_confidence = float(confidence)

            # --------------------------------------------------
            # Build Retrieval Result
            # --------------------------------------------------

            result = RetrievalResult(

                track_id=track.get("track_id", ""),

                video_id=track.get("video_id", ""),

                event_id=track.get("event_id"),

                object_type=track.get("object_type", ""),

                image_name=track.get("image_name"),

                image_path=track.get("image_path"),

                event_type=track.get(
                    "event_type",
                    "object_event"
                ),

                timestamp=track.get("timestamp"),

                start_time=track.get("start_time"),

                end_time=track.get("end_time"),

                similarity=1.0,

                confidence=object_confidence,

                quality_score=float(
                    track.get("quality_score", 0.0)
                ),

                metadata=track,

                source="metadata"

            )

            results.append(result)

        # --------------------------------------------------
        # Sort Results
        # --------------------------------------------------

        results.sort(

            key=lambda x: (
                x.confidence,
                x.quality_score
            ),

            reverse=True

        )

        return results[:plan.top_k]


# ----------------------------------------------------------
# Testing
# ----------------------------------------------------------

if __name__ == "__main__":

    from .planner import QueryPlanner
    from .query_parser import QueryParser

    loader = RetrievalDataLoader()
    loader.load_all()

    parser = QueryParser()
    planner = QueryPlanner()

    retriever = MetadataRetriever(loader)

    queries = [

        "find bicycle",

        "find red car",

        "find person with helmet",

        "show people carrying backpacks"

    ]

    for query in queries:

        print("=" * 80)

        print(query)

        parsed = parser.parse(query)

        plan = planner.create_plan(parsed)

        results = retriever.retrieve(plan)

        print(f"Matches : {len(results)}")

        for result in results[:5]:

            print(
                result.track_id,
                result.object_type,
                result.confidence,
                result.source
            )
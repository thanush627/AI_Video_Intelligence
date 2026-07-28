import json
from datetime import datetime
from typing import Dict, List


class ResponseGenerator:
    """
    Generates the final retrieval response returned by Phase 7.
    """

    def __init__(self):
        pass

    def generate(
        self,
        query: str,
        parsed_query: Dict,
        retrieval_plan: Dict,
        clips: List[Dict],
        statistics: Dict,
    ) -> Dict:

        response = {
            "query": query,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query_parser": parsed_query,
            "retrieval_plan": retrieval_plan,
            "statistics": statistics,
            "total_results": len(clips),
            "results": [],
        }

        for rank, clip in enumerate(clips, start=1):

            response["results"].append(
                {
                    "rank": rank,
                    "event_id": clip.get("event_id"),
                    "track_id": clip.get("track_id"),
                    "video_id": clip.get("video_id"),
                    "clip_id": clip.get("clip_id"),
                    "start_time": clip.get("start_time"),
                    "end_time": clip.get("end_time"),
                    "duration": clip.get("duration"),
                    "confidence": clip.get("confidence"),
                    "ranking_score": clip.get("ranking_score"),
                    "objects": clip.get("objects", []),
                    "colors": clip.get("colors", []),
                    "actions": clip.get("actions", []),
                    "attributes": clip.get("attributes", []),
                }
            )

        return response

    @staticmethod
    def save(response: Dict, output_path: str):

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(response, f, indent=4)


if __name__ == "__main__":

    generator = ResponseGenerator()

    clips = [
        {
            "event_id": 1,
            "track_id": 11,
            "video_id": "video001",
            "clip_id": "event_1",
            "start_time": "00:00:12",
            "end_time": "00:00:18",
            "duration": 6,
            "confidence": 0.96,
            "ranking_score": 0.94,
            "objects": ["person"],
            "colors": ["blue"],
            "actions": ["walking"],
            "attributes": ["backpack"],
        }
    ]

    parsed_query = {
        "object": "person",
        "color": "blue",
        "action": "walking",
        "attribute": "backpack",
        "query_type": "object",
    }

    retrieval_plan = {
        "strategy": "hybrid",
        "use_sql": True,
        "use_chromadb": True,
    }

    statistics = {
        "query_time_ms": 35,
        "sql_time_ms": 8,
        "vector_time_ms": 18,
        "total_time_ms": 61,
    }

    response = generator.generate(
        query="Show person wearing blue backpack",
        parsed_query=parsed_query,
        retrieval_plan=retrieval_plan,
        clips=clips,
        statistics=statistics,
    )

    generator.save(
        response,
        "../../outputs/phase7/retrieval_results.json",
    )

    print(json.dumps(response, indent=4))
import json
from typing import Dict, List


class ResultFormatter:
    """
    Formats retrieval results for console output
    and JSON export.
    """

    def __init__(self):
        pass

    def format_console(
        self,
        query: str,
        results: List[Dict],
        statistics: Dict,
    ) -> str:

        lines = []

        lines.append("=" * 80)
        lines.append("PHASE 7 - AGENTIC MULTIMODAL RETRIEVAL")
        lines.append("=" * 80)

        lines.append(f"Query            : {query}")
        lines.append(f"Results Returned : {len(results)}")

        lines.append(
            f"Total Time       : {statistics.get('total_time_ms',0):.2f} ms"
        )

        lines.append(
            f"SQL Time         : {statistics.get('sql_time_ms',0):.2f} ms"
        )

        lines.append(
            f"Vector Time      : {statistics.get('vector_time_ms',0):.2f} ms"
        )

        lines.append("=" * 80)

        if not results:

            lines.append("No matching events found.")
            lines.append("=" * 80)

            return "\n".join(lines)

        for index, result in enumerate(results, start=1):

            lines.append(f"Rank : {index}")
            lines.append("-" * 80)

            lines.append(f"Event ID       : {result.get('event_id')}")
            lines.append(f"Track ID       : {result.get('track_id')}")
            lines.append(f"Video ID       : {result.get('video_id')}")
            lines.append(f"Clip ID        : {result.get('clip_id')}")

            lines.append(
                f"Start Time     : {result.get('start_time')}"
            )

            lines.append(
                f"End Time       : {result.get('end_time')}"
            )

            lines.append(
                f"Duration       : {result.get('duration')}"
            )

            lines.append(
                f"Confidence     : {result.get('confidence')}"
            )

            lines.append(
                f"Ranking Score  : {result.get('ranking_score'):.4f}"
            )

            lines.append(
                f"Objects        : {', '.join(result.get('objects', []))}"
            )

            lines.append(
                f"Colors         : {', '.join(result.get('colors', []))}"
            )

            lines.append(
                f"Actions        : {', '.join(result.get('actions', []))}"
            )

            lines.append(
                f"Attributes     : {', '.join(result.get('attributes', []))}"
            )

            lines.append("=" * 80)

        return "\n".join(lines)

    @staticmethod
    def save_json(
        data: Dict,
        output_file: str
    ):

        with open(output_file, "w", encoding="utf-8") as file:

            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False
            )


if __name__ == "__main__":

    formatter = ResultFormatter()

    sample_results = [
        {
            "event_id": 5,
            "track_id": 18,
            "video_id": "video001",
            "clip_id": "event_5",
            "start_time": "00:01:32",
            "end_time": "00:01:38",
            "duration": 6,
            "confidence": 0.96,
            "ranking_score": 0.9435,
            "objects": ["person"],
            "colors": ["blue"],
            "actions": ["walking"],
            "attributes": ["backpack"]
        }
    ]

    stats = {
        "sql_time_ms": 12.6,
        "vector_time_ms": 41.8,
        "total_time_ms": 54.4
    }

    print(
        formatter.format_console(
            "Show person wearing blue backpack",
            sample_results,
            stats
        )
    )
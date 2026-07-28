from typing import Dict, List


class ClipSelector:
    """
    Selects the best clips from ranked retrieval results.

    This phase only returns clip references (timestamps).
    Actual clip extraction will be implemented in Phase 8.
    """

    def __init__(self, top_k: int = 10):
        self.top_k = top_k

    @staticmethod
    def _extract_clip_info(result: Dict) -> Dict:

        metadata = result.get("metadata", {})

        return {
            "event_id": result.get("event_id"),
            "track_id": metadata.get("track_id"),
            "video_id": metadata.get("video_id"),
            "clip_id": metadata.get(
                "clip_id",
                f"event_{result.get('event_id')}"
            ),
            "start_time": metadata.get(
                "start_time",
                metadata.get("start_timestamp")
            ),
            "end_time": metadata.get(
                "end_time",
                metadata.get("end_timestamp")
            ),
            "duration": metadata.get("duration"),
            "confidence": metadata.get(
                "confidence",
                result.get("ranking_score", 0.0)
            ),
            "ranking_score": result.get("ranking_score"),
            "objects": metadata.get("objects", []),
            "colors": metadata.get("colors", []),
            "actions": metadata.get("actions", []),
            "attributes": metadata.get("attributes", [])
        }

    def select(self, ranked_results: List[Dict]) -> List[Dict]:

        clips = []

        for result in ranked_results[: self.top_k]:
            clips.append(self._extract_clip_info(result))

        return clips


if __name__ == "__main__":

    sample_results = [
        {
            "event_id": 1,
            "ranking_score": 0.94,
            "metadata": {
                "track_id": 10,
                "video_id": "video001",
                "start_time": "00:01:12",
                "end_time": "00:01:18",
                "duration": 6,
                "confidence": 0.96,
                "objects": ["person"],
                "colors": ["blue"],
                "actions": ["walking"],
                "attributes": ["backpack"]
            }
        },
        {
            "event_id": 2,
            "ranking_score": 0.88,
            "metadata": {
                "track_id": 25,
                "video_id": "video001",
                "start_time": "00:02:40",
                "end_time": "00:02:48",
                "duration": 8,
                "confidence": 0.91,
                "objects": ["car"],
                "colors": ["red"],
                "actions": [],
                "attributes": []
            }
        }
    ]

    selector = ClipSelector(top_k=5)

    clips = selector.select(sample_results)

    print("=" * 80)

    for clip in clips:
        print(clip)
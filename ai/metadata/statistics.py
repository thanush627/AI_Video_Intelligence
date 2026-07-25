from pathlib import Path
from collections import Counter
import json


class StatisticsGenerator:

    def __init__(self):
        self.statistics = {}

    def generate(self, track_metadata):

        object_counter = Counter()
        action_counter = Counter()
        attribute_counter = Counter()
        color_counter = Counter()

        for track in track_metadata:

            object_counter.update([track["object_type"]])
            action_counter.update([track["action"]])

            for attribute in track.get("attributes", []):
                attribute_counter.update([attribute])

            for color in track.get("colors", {}).values():
                color_counter.update([color])

        self.statistics = {
            "total_tracks": len(track_metadata),
            "objects": dict(object_counter),
            "actions": dict(action_counter),
            "attributes": dict(attribute_counter),
            "colors": dict(color_counter)
        }

        return self.statistics

    def export(self, output_dir):

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "statistics.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                self.statistics,
                f,
                indent=4,
                ensure_ascii=False
            )

        return output_file
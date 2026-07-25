from pathlib import Path
import json


class TrackMetadataGenerator:

    def __init__(self):
        self.tracks = []

    def add(self, track_id, metadata):
        # metadata is already aggregated by MetadataAggregator
        self.tracks.append(metadata)

    def aggregate(self):
        # Nothing more to aggregate
        return self.tracks

    def export(self, output_dir):

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "track_metadata.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                self.tracks,
                f,
                indent=4,
                ensure_ascii=False
            )

        return output_file

    def clear(self):
        self.tracks.clear()
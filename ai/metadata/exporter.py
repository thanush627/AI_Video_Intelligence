from pathlib import Path
import json


class MetadataExporter:

    def __init__(self, output_dir):

        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("Metadata Output Folder")
        print(self.output_dir)
        print("=" * 70)

    def export_json(self, filename, data):

        output_file = self.output_dir / filename

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                indent=4,
                ensure_ascii=False
            )

        print("Saved ->", output_file)

        return output_file

    def export_object_metadata(self, data):
        return self.export_json("object_metadata.json", data)

    def export_track_metadata(self, data):
        return self.export_json("track_metadata.json", data)

    def export_event_metadata(self, data):
        return self.export_json("event_metadata.json", data)

    def export_semantic_metadata(self, data):
        return self.export_json("semantic_metadata.json", data)

    def export_statistics(self, data):
        return self.export_json("statistics.json", data)
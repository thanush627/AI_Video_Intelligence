from pathlib import Path
import json


class ObjectMetadataGenerator:

    def __init__(self):
        self.objects = []

    def add(
        self,
        track_id,
        image_name,
        metadata
    ):

        self.objects.append(
            {
                "track_id": track_id,
                "image": image_name,
                "object_type": metadata.get("object", "unknown"),
                "colors": metadata["colors"],
                "attributes": metadata["attributes"],
                "action": metadata["action"],
                "orientation": metadata["orientation"],
                "visibility": metadata["visibility"],
                "confidence": metadata["confidence"]
            }
        )

    def export(self, output_dir):

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "object_metadata.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                self.objects,
                f,
                indent=4,
                ensure_ascii=False
            )

        return output_file

    def get(self):
        return self.objects

    def clear(self):
        self.objects.clear()
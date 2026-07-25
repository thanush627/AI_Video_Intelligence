import json
from pathlib import Path


class MetadataFusion:

    def __init__(self):
        pass

    def load_json(self, path):

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_json(self, data, output):

        output = Path(output)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                indent=4,
            )

    def build_context(
        self,
        track_metadata,
        semantic_metadata,
        object_metadata,
        event_metadata,
    ):

        context = {

            "tracks": track_metadata,

            "semantic": semantic_metadata,

            "objects": object_metadata,

            "events": event_metadata,

        }

        return context
from pathlib import Path
from collections import defaultdict
import json


class EventMetadataGenerator:

    def __init__(self):
        self.events = defaultdict(list)

    def add(self, track_metadata):

        object_type = track_metadata["object_type"]
        action = track_metadata["action"]

        event_key = f"{object_type}_{action}"

        self.events[event_key].append(track_metadata)

    def generate(self):

        event_list = []

        for event_name, tracks in self.events.items():

            event_list.append({
                "event_name": event_name,
                "object_type": tracks[0]["object_type"],
                "action": tracks[0]["action"],
                "track_count": len(tracks),
                "track_ids": [
                    track["track_id"]
                    for track in tracks
                ],
                "tracks": tracks
            })

        return event_list

    def export(self, output_dir):

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "event_metadata.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                self.generate(),
                f,
                indent=4,
                ensure_ascii=False
            )

        return output_file

    def clear(self):
        self.events.clear()
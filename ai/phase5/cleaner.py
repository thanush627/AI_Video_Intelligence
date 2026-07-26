import json
from pathlib import Path
from typing import List


class MetadataCleaner:
    """
    Cleans Phase 4 metadata before embedding.
    """

    def __init__(self):

        self.ignore_attributes = {
            "",
            "unknown",
            "text",
            "none",
            "null",
            "n/a"
        }

        self.ignore_actions = {
            "",
            "unknown",
            "none"
        }

        self.ignore_colors = {
            "",
            "unknown"
        }

    def clean_attributes(self, attributes: List[str]) -> List[str]:

        cleaned = []

        for attr in attributes:

            if attr is None:
                continue

            attr = attr.strip().lower()

            if attr in self.ignore_attributes:
                continue

            if attr not in cleaned:
                cleaned.append(attr)

        return cleaned

    def clean_colors(self, colors: dict):

        cleaned = {}

        for key, value in colors.items():

            if value is None:
                continue

            value = value.strip().lower()

            if value in self.ignore_colors:
                continue

            cleaned[key] = value

        return cleaned

    def clean_action(self, action):

        if action is None:
            return ""

        action = action.strip().lower()

        if action in self.ignore_actions:
            return ""

        return action

    def clean_track(self, track):

        track["attributes"] = self.clean_attributes(
            track.get("attributes", [])
        )

        track["colors"] = self.clean_colors(
            track.get("colors", {})
        )

        track["action"] = self.clean_action(
            track.get("action", "")
        )

        return track

    def clean_file(self, input_json, output_json):

        with open(input_json, "r") as f:
            tracks = json.load(f)

        cleaned_tracks = []

        for track in tracks:
            cleaned_tracks.append(
                self.clean_track(track)
            )

        with open(output_json, "w") as f:
            json.dump(
                cleaned_tracks,
                f,
                indent=4
            )

        print("=" * 60)
        print("Metadata Cleaning Finished")
        print("Tracks :", len(cleaned_tracks))
        print("Saved :", output_json)
        print("=" * 60)


if __name__ == "__main__":

    root = Path(__file__).resolve().parents[2]

    input_file = root / "outputs" / "phase4" / "track_metadata.json"

    output_dir = root / "outputs" / "phase5"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "clean_track_metadata.json"

    cleaner = MetadataCleaner()

    cleaner.clean_file(
        input_file,
        output_file
    )
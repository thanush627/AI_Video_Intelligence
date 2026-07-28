import json
import os
from typing import Dict, List


class SQLRetriever:
    """
    Retrieves structured metadata from the Phase 6 event database.

    Current implementation uses JSON.
    Can later be replaced with PostgreSQL without changing the interface.
    """

    def __init__(self, event_database_path: str):

        self.event_database_path = event_database_path
        self.events = self._load_database()

    def _load_database(self) -> List[Dict]:

        if not os.path.exists(self.event_database_path):
            return []

        with open(self.event_database_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def retrieve(self, parsed_query: Dict) -> List[Dict]:
        print(f"Loaded events: {len(self.events)}")

        object_name = parsed_query.get("object")
        color = parsed_query.get("color")
        action = parsed_query.get("action")
        attribute = parsed_query.get("attribute")
        time_range = parsed_query.get("time_range")

        results = []

        for event in self.events:

            if object_name:
                objects = [
                    obj.lower()
                    for obj in event.get("objects", [])
                ]

                if object_name.lower() not in objects:
                    continue

            if color:

                colors = [
                    c.lower()
                    for c in event.get("colors", [])
                ]

                if color.lower() not in colors:
                    continue

            if action:

                actions = [
                    a.lower()
                    for a in event.get("actions", [])
                ]

                if action.lower() not in actions:
                    continue

            if attribute:

                attrs = [
                    a.lower()
                    for a in event.get("attributes", [])
                ]

                if attribute.lower() not in attrs:
                    continue

            if time_range:

                start = event.get("start_time", "")
                end = event.get("end_time", "")

                if not (time_range[0] <= start <= time_range[1]):
                    continue

                if not (time_range[0] <= end <= time_range[1]):
                    continue

            results.append(event)
        print(f"SQL Matches: {len(results)}")
        return results


if __name__ == "__main__":

    retriever = SQLRetriever(
        "../../outputs/phase6/event_database.json"
    )

    query = {
        "object": "person",
        "color": "blue",
        "action": None,
        "attribute": "backpack",
        "time_range": None
    }

    results = retriever.retrieve(query)

    print("=" * 80)
    print(f"Matches : {len(results)}")

    for result in results[:5]:
        print(result)
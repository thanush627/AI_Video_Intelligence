import re
from typing import Dict, List


class QueryParser:
    """
    Parses natural language queries into structured filters
    for SQL and ChromaDB retrieval.
    """

    OBJECTS = [
        "person",
        "car",
        "truck",
        "bus",
        "bicycle",
        "motorcycle",
        "bike",
        "helmet",
        "backpack",
        "bag",
        "dog",
        "cat"
    ]

    COLORS = [
        "red",
        "blue",
        "green",
        "yellow",
        "black",
        "white",
        "gray",
        "grey",
        "brown",
        "orange",
        "pink",
        "purple"
    ]

    ACTIONS = [
        "walking",
        "running",
        "standing",
        "sitting",
        "riding",
        "driving",
        "carrying",
        "entering",
        "leaving",
        "crossing"
    ]

    ATTRIBUTES = [
        "helmet",
        "backpack",
        "bag",
        "cap",
        "hat",
        "glasses",
        "mask"
    ]

    QUERY_TYPES = {
        "object": [
            "find",
            "show",
            "locate",
            "search"
        ],
        "event": [
            "walking",
            "running",
            "entering",
            "leaving"
        ],
        "time": [
            "before",
            "after",
            "between",
            "from"
        ]
    }

    def __init__(self):
        pass

    def parse(self, query: str) -> Dict:

        text = query.lower().strip()

        parsed = {
            "raw_query": query,
            "object": None,
            "color": None,
            "action": None,
            "attribute": None,
            "time_range": None,
            "query_type": "semantic",
            "keywords": []
        }

        # -------------------------
        # Object
        # -------------------------
        for obj in self.OBJECTS:
            if re.search(rf"\b{re.escape(obj)}\b", text):
                parsed["object"] = obj
                parsed["keywords"].append(obj)
                break

        # -------------------------
        # Color
        # -------------------------
        for color in self.COLORS:
            if re.search(rf"\b{re.escape(color)}\b", text):
                parsed["color"] = color
                parsed["keywords"].append(color)
                break

        # -------------------------
        # Action
        # -------------------------
        for action in self.ACTIONS:
            if re.search(rf"\b{re.escape(action)}\b", text):
                parsed["action"] = action
                parsed["keywords"].append(action)
                break

        # -------------------------
        # Attribute
        # -------------------------
        for attr in self.ATTRIBUTES:
            if re.search(rf"\b{re.escape(attr)}\b", text):
                parsed["attribute"] = attr
                parsed["keywords"].append(attr)
                break

        # -------------------------
        # Time Range
        # Example:
        # between 09:00 and 10:00
        # -------------------------
        time_match = re.search(
            r'between\s+(\d{1,2}:\d{2})\s+and\s+(\d{1,2}:\d{2})',
            text
        )

        if time_match:
            parsed["time_range"] = (
                time_match.group(1),
                time_match.group(2)
            )

        # -------------------------
        # Query Type
        # -------------------------

        if parsed["time_range"]:
            parsed["query_type"] = "time"

        elif parsed["action"]:
            parsed["query_type"] = "event"

        elif parsed["object"] or parsed["attribute"] or parsed["color"]:
            parsed["query_type"] = "object"

        else:
            parsed["query_type"] = "semantic"

        return parsed


if __name__ == "__main__":

    parser = QueryParser()

    queries = [
        "Show me the red car",
        "Find a person wearing helmet",
        "Walking person",
        "Blue backpack",
        "Find truck between 09:00 and 10:00",
        "Person carrying backpack"
    ]

    for q in queries:
        print("=" * 60)
        print(parser.parse(q))
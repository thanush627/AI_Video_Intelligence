from collections import Counter


class MetadataNormalizer:

    COLOR_MAP = {
        "dark blue": "blue",
        "light blue": "blue",
        "navy": "blue",
        "royal blue": "blue",
        "sky blue": "blue",

        "dark red": "red",
        "light red": "red",
        "maroon": "red",

        "dark green": "green",
        "light green": "green",
        "olive": "green",

        "grey": "gray",
        "dark grey": "gray",
        "light grey": "gray",

        "silver": "gray",

        "cream": "white",
        "ivory": "white"
    }

    ATTRIBUTE_MAP = {
        "bag": "backpack",
        "school bag": "backpack",
        "travel bag": "backpack",
        "rucksack": "backpack",

        "cap": "hat",
        "helmet cap": "helmet",

        "spectacles": "glasses",
        "sunglasses": "glasses"
    }

    @classmethod
    def normalize_color(cls, color):

        if not color:
            return color

        color = color.lower().strip()
        return cls.COLOR_MAP.get(color, color)

    @classmethod
    def normalize_colors(cls, colors):

        normalized = {}

        for part, color in colors.items():
            normalized[part] = cls.normalize_color(color)

        return normalized

    @classmethod
    def normalize_attributes(cls, attributes):

        normalized = []

        for attribute in attributes:

            attribute = attribute.lower().strip()
            attribute = cls.ATTRIBUTE_MAP.get(attribute, attribute)

            normalized.append(attribute)

        return sorted(list(set(normalized)))

    @staticmethod
    def normalize_action(actions):

        if isinstance(actions, str):
            return actions.lower()

        if not actions:
            return "unknown"

        actions = [a.lower() for a in actions]

        return Counter(actions).most_common(1)[0][0]

    @classmethod
    def normalize_metadata(cls, metadata):

        metadata["colors"] = cls.normalize_colors(
            metadata.get("colors", {})
        )

        metadata["attributes"] = cls.normalize_attributes(
            metadata.get("attributes", [])
        )

        metadata["action"] = cls.normalize_action(
            metadata.get("action", "")
        )

        return metadata
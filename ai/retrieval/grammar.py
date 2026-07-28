"""
Phase 7
Grammar Rules

Contains reusable parsing rules for natural language queries.
"""

import re

from .vocabulary import (
    OBJECT_ALIASES,
    COLORS,
    ACTIONS,
    ATTRIBUTES,
    ZONES
)


def extract_time_range(query: str):

    pattern = r"\d{1,2}:\d{2}"

    matches = re.findall(pattern, query)

    if len(matches) >= 2:
        return matches[0], matches[1]

    return None, None


def normalize_objects(words):

    objects = []

    for word in words:

        if word in OBJECT_ALIASES:

            objects.append(OBJECT_ALIASES[word])

    return objects


def extract_colors(words):

    return [w for w in words if w in COLORS]


def extract_actions(words):

    return [w for w in words if w in ACTIONS]


def extract_attributes(words):

    return [w for w in words if w in ATTRIBUTES]


def extract_zones(words):

    return [w for w in words if w in ZONES]


def determine_primary_object(objects, attributes):
    """
    Determine the primary object for retrieval.

    Priority:
    1. Person
    2. Vehicle
    3. Other detected object

    Wearable items (helmet, backpack, bag) remain attributes
    if a person is present.
    """

    if "person" in objects:
        return "person"

    wearable_items = {
        "helmet",
        "backpack",
        "bag"
    }

    filtered = [
        obj for obj in objects
        if obj not in wearable_items
    ]

    if filtered:
        return filtered[0]

    if objects:
        return objects[0]

    return None
from collections import Counter


class MetadataAggregator:

    @staticmethod
    def aggregate(track_id, metadata_list):

        if not metadata_list:
            return None

        objects = []
        actions = []
        orientations = []
        visibility = []

        attributes = set()
        colors = {}

        confidence = {
            "object": [],
            "colors": [],
            "attributes": [],
            "action": []
        }

        for item in metadata_list:

            objects.append(item["object"])
            actions.append(item["action"])
            orientations.append(item["orientation"])
            visibility.append(item["visibility"])

            for key, value in item["colors"].items():
                colors[key] = value

            attributes.update(item["attributes"])

            for key in confidence:
                confidence[key].append(
                    item["confidence"].get(key, 0)
                )

        return {
            "track_id": track_id,
            "object_type": Counter(objects).most_common(1)[0][0],
            "colors": colors,
            "attributes": sorted(attributes),
            "action": Counter(actions).most_common(1)[0][0],
            "orientation": Counter(orientations).most_common(1)[0][0],
            "visibility": Counter(visibility).most_common(1)[0][0],
            "confidence": {
                k: round(sum(v) / len(v), 3)
                for k, v in confidence.items()
            }
        }
from collections import Counter
from datetime import timedelta


class MetadataAggregator:

    @staticmethod
    def _format_timestamp(seconds):
        if seconds is None:
            return None

        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            return None

        if seconds < 0:
            return None

        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        milliseconds = int(round((seconds - total_seconds) * 1000))

        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"

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

        start_times = []
        end_times = []
        durations = []

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

            start_seconds = item.get("start_time_seconds")
            end_seconds = item.get("end_time_seconds")
            duration_seconds = item.get("duration_seconds")

            if start_seconds is not None:
                start_times.append(float(start_seconds))

            if end_seconds is not None:
                end_times.append(float(end_seconds))

            if duration_seconds is not None:
                durations.append(float(duration_seconds))

        aggregated = {
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

        if start_times:
            aggregated["start_time_seconds"] = round(min(start_times), 4)
            aggregated["start_timestamp"] = MetadataAggregator._format_timestamp(
                aggregated["start_time_seconds"]
            )

        if end_times:
            aggregated["end_time_seconds"] = round(max(end_times), 4)
            aggregated["end_timestamp"] = MetadataAggregator._format_timestamp(
                aggregated["end_time_seconds"]
            )

        if start_times and end_times:
            aggregated["duration_seconds"] = round(
                aggregated["end_time_seconds"] - aggregated["start_time_seconds"],
                4,
            )
        elif durations:
            aggregated["duration_seconds"] = round(sum(durations) / len(durations), 4)

        if aggregated.get("start_time_seconds") is not None:
            aggregated["timestamp"] = aggregated["start_timestamp"]

        return aggregated
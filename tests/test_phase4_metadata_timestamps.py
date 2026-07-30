from ai.metadata.aggregator import MetadataAggregator


def test_aggregate_preserves_timing_fields():
    metadata_items = [
        {
            "object": "car",
            "action": "moving",
            "orientation": "front",
            "visibility": "clear",
            "colors": {},
            "attributes": [],
            "confidence": {
                "object": 0.9,
                "colors": 0.8,
                "attributes": 0.1,
                "action": 0.7,
            },
            "start_time_seconds": 1.0,
            "end_time_seconds": 5.0,
            "duration_seconds": 4.0,
        },
        {
            "object": "car",
            "action": "moving",
            "orientation": "front",
            "visibility": "clear",
            "colors": {"upper_body": "red"},
            "attributes": ["large"],
            "confidence": {
                "object": 0.8,
                "colors": 0.7,
                "attributes": 0.2,
                "action": 0.6,
            },
            "start_time_seconds": 3.0,
            "end_time_seconds": 8.0,
            "duration_seconds": 5.0,
        },
    ]

    aggregated = MetadataAggregator.aggregate("car_track_1", metadata_items)

    assert aggregated["start_time_seconds"] == 1.0
    assert aggregated["end_time_seconds"] == 8.0
    assert aggregated["duration_seconds"] == 7.0
    assert aggregated["start_timestamp"] == "00:00:01.000"
    assert aggregated["end_timestamp"] == "00:00:08.000"

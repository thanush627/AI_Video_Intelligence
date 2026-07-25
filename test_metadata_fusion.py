from ai.events.metadata_fusion import MetadataFusion

fusion = MetadataFusion()

track = fusion.load_json(
    r"outputs/phase4/track_metadata.json"
)

semantic = fusion.load_json(
    r"outputs/phase4/semantic_metadata.json"
)

objects = fusion.load_json(
    r"outputs/phase4/object_metadata.json"
)

events = fusion.load_json(
    r"outputs/phase4/event_metadata.json"
)

context = fusion.build_context(
    track,
    semantic,
    objects,
    events,
)

fusion.save_json(
    context,
    r"outputs/phase6/track_context.json"
)

print("Fusion completed.")

print(context.keys())
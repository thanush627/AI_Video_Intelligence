import os
import logging
from pathlib import Path
from tqdm import tqdm

from ai.vlm.loader import ImageLoader
from ai.vlm.batch_processor import BatchProcessor
from ai.vlm.qwen_model import QwenVL
from ai.vlm.parser import MetadataParser
from ai.vlm.validator import MetadataValidator

from ai.metadata.normalizer import MetadataNormalizer
from ai.metadata.aggregator import MetadataAggregator
from ai.metadata.object_metadata import ObjectMetadataGenerator
from ai.metadata.track_metadata import TrackMetadataGenerator
from ai.metadata.event_metadata import EventMetadataGenerator
from ai.metadata.statistics import StatisticsGenerator
from ai.metadata.exporter import MetadataExporter

import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


class Phase4Pipeline:

    def __init__(
        self,
        representative_crop_dir,
        output_dir,
        batch_size=16
    ):

        self.representative_crop_dir = Path(representative_crop_dir)
        self.output_dir = Path(output_dir).resolve()

        self.loader = ImageLoader()
        self.batch_processor = BatchProcessor(batch_size=batch_size)

        logger.info("Loading Qwen2.5-VL...")
        self.vlm = QwenVL()

        self.object_generator = ObjectMetadataGenerator()
        self.track_generator = TrackMetadataGenerator()
        self.event_generator = EventMetadataGenerator()
        self.statistics_generator = StatisticsGenerator()
        self.exporter = MetadataExporter(self.output_dir)

        self.track_class_mapping = self.load_track_class_mapping()

    def run(self):

        logger.info("Starting Phase 4 Pipeline...")

        track_folders = sorted(os.listdir(self.representative_crop_dir))

        logger.info(f"Processing {len(track_folders)} tracks")

        for track_folder in tqdm(track_folders):

            try:

                logger.info(f"Processing {track_folder}")

                self.process_track(
                    self.representative_crop_dir / track_folder
                )

            except Exception as e:

                logger.exception(
                    f"Failed processing {track_folder}: {e}"
                )

        logger.info("Exporting metadata...")

        self.export_all()

        logger.info("Phase 4 Finished Successfully")

    def process_track(self, track_folder):

        track = self.loader.load_track(track_folder)

        if track["num_images"] == 0:
            logger.warning(f"No images found in {track_folder}")
            return

        track_id = Path(track_folder).name
        track_metadata = []

        batches = self.batch_processor.process_track(track)

        for batch in batches:

            responses = self.vlm.predict_batch(
                batch["images"]
            )

            print("\nRAW QWEN RESPONSE")
            print(responses)

            parsed_metadata = MetadataParser.parse_batch(
                responses
            )

            valid_metadata = MetadataValidator.filter_valid(
                parsed_metadata
            )

            print("\nVALID")
            print(valid_metadata)

            for image_path, metadata in zip(
                batch["image_paths"],
                valid_metadata
            ):

                metadata = MetadataNormalizer.normalize_metadata(
                    metadata
                )

                # -------------------------------------------------
                # Use YOLO detected class instead of Qwen object
                # -------------------------------------------------
                track_number = int(track_id.split("_")[-1])

                metadata["object"] = self.track_class_mapping.get(
                    track_number,
                    "unknown"
                )

                track_metadata.append(metadata)

                self.object_generator.add(
                    track_id=track_id,
                    image_name=Path(image_path).name,
                    metadata=metadata
                )

        if not track_metadata:
            logger.warning(f"No valid metadata for {track_id}")
            return

        print(f"Track: {track_id}")
        print(f"Collected metadata: {len(track_metadata)}")

        aggregated = MetadataAggregator.aggregate(
            track_id,
            track_metadata
        )

        print("Aggregated:")
        print(aggregated)

        print("=" * 60)
        print(track_id)
        print(aggregated)
        print("=" * 60)

        self.track_generator.add(
            track_id,
            aggregated
        )

        self.event_generator.add(
            aggregated
        )

    def export_all(self):

        logger.info("Generating Track Metadata...")

        track_metadata = self.track_generator.aggregate()

        logger.info("Generating Event Metadata...")

        self.event_generator.clear()

        for track in track_metadata:
            self.event_generator.add(track)

        event_metadata = self.event_generator.generate()

        logger.info("Generating Statistics...")

        statistics = self.statistics_generator.generate(
            track_metadata
        )

        semantic_metadata = {
            "tracks": track_metadata,
            "events": event_metadata
        }

        obj = self.exporter.export_object_metadata(
            self.object_generator.get()
        )

        track = self.exporter.export_track_metadata(
            track_metadata
        )

        event = self.exporter.export_event_metadata(
            event_metadata
        )

        semantic = self.exporter.export_semantic_metadata(
            semantic_metadata
        )

        stats = self.exporter.export_statistics(
            statistics
        )

        print("\n" + "=" * 70)
        print("FILES SAVED SUCCESSFULLY")
        print("=" * 70)
        print(obj)
        print(track)
        print(event)
        print(semantic)
        print(stats)
        print("=" * 70)
        print("Tracks stored:", len(self.track_generator.tracks))
        print("Objects stored:", len(self.object_generator.get()))

    def load_track_class_mapping(self):

        metadata_file = (
            Path("outputs")
            / "phase3"
            / "production_runs"
            / "test"
            / "03_object_tracking"
            / "track_metadata.json"
        )

        with open(metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        mapping = {}

        for track in data["tracks"].values():
            mapping[int(track["track_id"])] = track["class_name"]

        logger.info(f"Loaded {len(mapping)} track class mappings")

        return mapping


if __name__ == "__main__":

    BASE = Path("/kaggle/working/Phase4_Fix")

    pipeline = Phase4Pipeline(
        representative_crop_dir=BASE / "outputs/phase3/production_runs/test/04_representative_selection/crops",
        output_dir=BASE / "outputs/phase4",
        batch_size=16
    )

    pipeline.run()
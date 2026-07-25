from ai.pipeline.phase4_pipeline import Phase4Pipeline


def main():

    pipeline = Phase4Pipeline(
        representative_crop_dir="outputs/phase3/production_runs/test/04_representative_selection/crops",
        output_dir="outputs/phase4",
        batch_size=16
    )

    pipeline.run()


if __name__ == "__main__":
    main()
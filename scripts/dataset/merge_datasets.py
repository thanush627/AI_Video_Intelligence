from pathlib import Path
import shutil
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS = PROJECT_ROOT / "ai" / "datasets"

SOURCES = {
    "visdrone": DATASETS / "Combined",
    "detrac": DATASETS / "UA_DETRAC"
}

DEST = DATASETS / "Merged"

SPLITS = ["train", "valid", "test"]

for split in SPLITS:
    (DEST / split / "images").mkdir(parents=True, exist_ok=True)
    (DEST / split / "labels").mkdir(parents=True, exist_ok=True)

print("Project Root :", PROJECT_ROOT)
print("Datasets     :", DATASETS)
print("Destination  :", DEST)



def merge_dataset(prefix, source_root):
    for split in SPLITS:
        image_dir = source_root / split / "images"
        label_dir = source_root / split / "labels"

        images = sorted(image_dir.glob("*"))

        count = 1

        for image in tqdm(images, desc=f"{prefix} {split}"):

            if image.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            new_stem = f"{prefix}_{split}_{count:06d}"

            shutil.copy2(
                image,
                DEST / split / "images" / (new_stem + image.suffix.lower())
            )

            label = label_dir / (image.stem + ".txt")

            if label.exists():
                shutil.copy2(
                    label,
                    DEST / split / "labels" / (new_stem + ".txt")
                )

            count += 1
if __name__ == "__main__":

    print("\nStarting Dataset Merge...\n")

    print("Merging VisDrone...")
    merge_dataset(
        "visdrone",
        SOURCES["visdrone"]
    )

    print("\nVisDrone Merge Completed.")

    print(SOURCES["detrac"])
    print(SOURCES["detrac"].exists())

    print("\nMerging UA-DETRAC...")
    merge_dataset(
        "detrac",
        SOURCES["detrac"]
    )


    print("\nUA-DETRAC Merge Completed.")
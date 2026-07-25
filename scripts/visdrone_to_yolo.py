"""
=========================================================
VisDrone 2019 --> YOLO Dataset Converter

Project:
Agentic Multimodal Retrieval for Long-Horizon
Spatiotemporal Event Grounding

Author: Thanush
=========================================================
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# =====================================================
# PROJECT PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VISDRONE_ROOT = PROJECT_ROOT / "ai" / "datasets" / "VisDrone"

COMBINED_ROOT = PROJECT_ROOT / "ai" / "datasets" / "Combined"

TRAIN_ROOT = VISDRONE_ROOT / "VisDrone2019-VID-train"
VAL_ROOT = VISDRONE_ROOT / "VisDrone2019-VID-val"
TEST_ROOT = VISDRONE_ROOT / "VisDrone2019-VID-test-dev"

# =====================================================
# OUTPUT DIRECTORIES
# =====================================================

OUTPUTS = {
    "train": {
        "images": COMBINED_ROOT / "train" / "images",
        "labels": COMBINED_ROOT / "train" / "labels"
    },
    "valid": {
        "images": COMBINED_ROOT / "valid" / "images",
        "labels": COMBINED_ROOT / "valid" / "labels"
    },
    "test": {
        "images": COMBINED_ROOT / "test" / "images",
        "labels": COMBINED_ROOT / "test" / "labels"
    }
}

# =====================================================
# CREATE OUTPUT FOLDERS
# =====================================================

for split in OUTPUTS.values():

    split["images"].mkdir(parents=True, exist_ok=True)
    split["labels"].mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("OUTPUT FOLDERS VERIFIED")
print("=" * 60)

# =====================================================
# VISDRONE CLASS MAPPING
# =====================================================

"""
VisDrone Categories

0 ignored
1 pedestrian
2 people
3 bicycle
4 car
5 van
6 truck
7 tricycle
8 awning-tricycle
9 bus
10 motor
"""

VISDRONE_TO_YOLO = {

    1: 0,     # person

    2: 0,     # people -> person

    3: 5,     # bicycle

    4: 1,     # car

    5: 6,     # van

    6: 2,     # truck

    7: None,  # ignore

    8: None,  # ignore

    9: 3,     # bus

    10: 4     # motorcycle
}

YOLO_CLASSES = [
    "person",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "van",
    "dog",
    "bag",
    "helmet"
]

print("\nYOLO Classes\n")

for i, name in enumerate(YOLO_CLASSES):
    print(f"{i} -> {name}")


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def convert_bbox(x, y, w, h, img_width, img_height):
    """
    Convert VisDrone bounding box
    to YOLO normalized format.
    """

    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height

    width = w / img_width
    height = h / img_height

    return (
        round(x_center, 6),
        round(y_center, 6),
        round(width, 6),
        round(height, 6)
    )


def read_annotation_file(annotation_file):
    """
    Reads one VisDrone annotation file.

    Returns dictionary:

    {
        frame_number : [
            (class_id,x,y,w,h),
            ...
        ]
    }
    """

    frame_annotations = {}

    with open(annotation_file, "r") as f:

        for line in f:

            line = line.strip()

            if line == "":
                continue

            values = line.split(",")

            frame = int(values[0])

            x = float(values[2])
            y = float(values[3])
            w = float(values[4])
            h = float(values[5])

            category = int(values[7])

            if category not in VISDRONE_TO_YOLO:
                continue

            yolo_class = VISDRONE_TO_YOLO[category]

            if yolo_class is None:
                continue

            if frame not in frame_annotations:
                frame_annotations[frame] = []

            frame_annotations[frame].append(
                (
                    yolo_class,
                    x,
                    y,
                    w,
                    h
                )
            )

    return frame_annotations


print("\nHelper functions loaded successfully.")


# =====================================================
# PROCESS ONE DATASET SPLIT
# =====================================================

def process_split(split_name, split_root):

    print("\n" + "=" * 60)
    print(f"PROCESSING {split_name.upper()}")
    print("=" * 60)

    annotation_dir = split_root / "annotations"
    sequence_dir = split_root / "sequences"

    if not annotation_dir.exists():
        print(f"Missing annotation folder : {annotation_dir}")
        return

    if not sequence_dir.exists():
        print(f"Missing sequence folder : {sequence_dir}")
        return

    annotation_files = sorted(annotation_dir.glob("*.txt"))

    print(f"Annotation Files : {len(annotation_files)}")

    image_counter = 0
    object_counter = 0

    for annotation_file in tqdm(annotation_files):

        sequence_name = annotation_file.stem

        sequence_folder = sequence_dir / sequence_name

        if not sequence_folder.exists():
            print(f"Sequence Missing : {sequence_name}")
            continue

        frame_annotations = read_annotation_file(annotation_file)

        image_files = sorted(sequence_folder.glob("*.jpg"))

        for image_file in image_files:

            frame_number = int(image_file.stem)

            destination_image = (
                OUTPUTS[split_name]["images"] /
                f"{sequence_name}_{frame_number:07d}.jpg"
            )

            shutil.copy2(
                image_file,
                destination_image
            )

            image_counter += 1

            destination_label = (
                OUTPUTS[split_name]["labels"] /
                f"{sequence_name}_{frame_number:07d}.txt"
            )

            label_lines = []

            try:

                from PIL import Image

                img = Image.open(image_file)

                img_width, img_height = img.size

            except Exception:

                continue

            if frame_number in frame_annotations:

                for obj in frame_annotations[frame_number]:

                    cls, x, y, w, h = obj

                    xc, yc, bw, bh = convert_bbox(
                        x,
                        y,
                        w,
                        h,
                        img_width,
                        img_height
                    )

                    label_lines.append(
                        f"{cls} {xc} {yc} {bw} {bh}"
                    )

                    object_counter += 1

            with open(destination_label, "w") as f:

                for line in label_lines:
                    f.write(line + "\n")

    print()

    print(f"{split_name.upper()} COMPLETE")

    print(f"Images Copied : {image_counter}")

    print(f"Objects Converted : {object_counter}")


    # =====================================================
# DATASET STATISTICS
# =====================================================

def dataset_statistics():

    print("\n" + "=" * 60)
    print("FINAL DATASET STATISTICS")
    print("=" * 60)

    total_images = 0
    total_labels = 0

    for split in ["train", "valid", "test"]:

        image_dir = OUTPUTS[split]["images"]
        label_dir = OUTPUTS[split]["labels"]

        image_count = len(list(image_dir.glob("*.jpg")))
        label_count = len(list(label_dir.glob("*.txt")))

        total_images += image_count
        total_labels += label_count

        print(f"\n{split.upper()}")

        print(f"Images : {image_count}")
        print(f"Labels : {label_count}")

    print("\n" + "=" * 60)

    print(f"TOTAL IMAGES : {total_images}")
    print(f"TOTAL LABELS : {total_labels}")

    print("=" * 60)


# =====================================================
# VERIFY OUTPUT
# =====================================================

def verify_output():

    print("\nVerifying Output...\n")

    success = True

    for split in ["train", "valid", "test"]:

        image_dir = OUTPUTS[split]["images"]
        label_dir = OUTPUTS[split]["labels"]

        image_count = len(list(image_dir.glob("*.jpg")))
        label_count = len(list(label_dir.glob("*.txt")))

        if image_count != label_count:

            print(f"Mismatch detected in {split}")

            print(f"Images : {image_count}")

            print(f"Labels : {label_count}")

            success = False

    if success:

        print("Dataset verification successful.")

    else:

        print("Dataset verification failed.")


# =====================================================
# MAIN
# =====================================================

def main():

    print("\n")
    print("=" * 60)
    print("VISDRONE TO YOLO CONVERTER")
    print("=" * 60)

    process_split(
        "train",
        TRAIN_ROOT
    )

    process_split(
        "valid",
        VAL_ROOT
    )

    process_split(
        "test",
        TEST_ROOT
    )

    dataset_statistics()

    verify_output()

    print("\n")
    print("=" * 60)
    print("CONVERSION FINISHED")
    print("=" * 60)

    print("\nOutput Folder:")

    print(COMBINED_ROOT)

    print("\nYou can now create data.yaml and train YOLOv8.")


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":

    main()
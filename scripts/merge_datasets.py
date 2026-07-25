"""
==============================================================
MERGE DATASETS

Project:
Agentic Multimodal Retrieval for Long-Horizon
Spatiotemporal Event Grounding

Purpose:
Merge multiple YOLO datasets into one unified dataset.

Current Supported Datasets

✓ VisDrone
✓ UA-DETRAC

Future

✓ VIRAT
✓ MOT17
✓ Custom Datasets

Author: Thanush
==============================================================
"""

import os
import shutil
import logging
from pathlib import Path
from tqdm import tqdm

# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "ai" / "datasets"

VISDRONE_ROOT = DATASET_ROOT / "Combined"

DETRAC_ROOT = DATASET_ROOT / "UA_DETRAC"

MERGED_ROOT = DATASET_ROOT / "Merged"

# ==========================================================
# OUTPUT DIRECTORIES
# ==========================================================

OUTPUTS = {

    "train": {

        "images": MERGED_ROOT / "train" / "images",

        "labels": MERGED_ROOT / "train" / "labels"

    },

    "valid": {

        "images": MERGED_ROOT / "valid" / "images",

        "labels": MERGED_ROOT / "valid" / "labels"

    },

    "test": {

        "images": MERGED_ROOT / "test" / "images",

        "labels": MERGED_ROOT / "test" / "labels"

    }

}

# ==========================================================
# CREATE OUTPUT DIRECTORIES
# ==========================================================

for split in OUTPUTS.values():

    split["images"].mkdir(
        parents=True,
        exist_ok=True
    )

    split["labels"].mkdir(
        parents=True,
        exist_ok=True
    )

print("=" * 70)
print("MERGED DATASET FOLDERS VERIFIED")
print("=" * 70)

# ==========================================================
# LOGGER
# ==========================================================

LOG_FILE = MERGED_ROOT / "merge_log.txt"

logging.basicConfig(

    filename=LOG_FILE,

    level=logging.INFO,

    format="%(asctime)s - %(levelname)s - %(message)s"

)

logging.info("Dataset Merge Started")

# ==========================================================
# GLOBAL STATISTICS
# ==========================================================

stats = {

    "visdrone_images": 0,

    "visdrone_labels": 0,

    "detrac_images": 0,

    "detrac_labels": 0,

    "merged_images": 0,

    "merged_labels": 0

}

# ==========================================================
# PRINT HEADER
# ==========================================================

def print_header(title):

    print()

    print("=" * 70)

    print(title)

    print("=" * 70)

# ==========================================================
# LOGGER WRAPPER
# ==========================================================

def log(message):

    logging.info(message)

# ==========================================================
# COPY FILE
# ==========================================================

def copy_file(source, destination):

    shutil.copy2(source, destination)

# ==========================================================
# VERIFY IMAGE/LABEL PAIR
# ==========================================================

def verify_pair(image_file, label_file):

    return label_file.exists()

# ==========================================================
# COUNT FILES
# ==========================================================

def count_files(folder, extensions):

    total = 0

    for ext in extensions:

        total += len(list(folder.glob(f"*{ext}")))

    return total

# ==========================================================
# DATASET STATISTICS
# ==========================================================

def dataset_statistics():

    print_header("DATASET STATISTICS")

    print(f"VisDrone Images : {stats['visdrone_images']}")

    print(f"VisDrone Labels : {stats['visdrone_labels']}")

    print()

    print(f"UA-DETRAC Images : {stats['detrac_images']}")

    print(f"UA-DETRAC Labels : {stats['detrac_labels']}")

    print()

    print(f"TOTAL MERGED IMAGES : {stats['merged_images']}")

    print(f"TOTAL MERGED LABELS : {stats['merged_labels']}")

    print()

    log("Dataset Statistics Generated")

# ==========================================================
# VERIFY MERGED DATASET
# ==========================================================

def verify_merged():

    print_header("VERIFYING MERGED DATASET")

    success = True

    for split in ["train", "valid", "test"]:

        image_count = count_files(

            OUTPUTS[split]["images"],

            [".jpg", ".jpeg", ".png"]

        )

        label_count = count_files(

            OUTPUTS[split]["labels"],

            [".txt"]

        )

        print(f"{split.upper()}")

        print(f"Images : {image_count}")

        print(f"Labels : {label_count}")

        print()

        if image_count != label_count:

            success = False

    if success:

        print("✓ Dataset Verification Passed")

        log("Merged Dataset Verified Successfully")

    else:

        print("✗ Dataset Verification Failed")

        log("Merged Dataset Verification Failed")

# ==========================================================
# END OF PART 1
# ==========================================================
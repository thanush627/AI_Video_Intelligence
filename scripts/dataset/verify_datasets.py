"""
============================================================
DATASET VERIFICATION SCRIPT

Project:
Agentic Multimodal Retrieval for Long-Horizon
Spatiotemporal Event Grounding

Purpose:
Verify all datasets before YOLO training.

Datasets Supported:
    ✓ VisDrone
    ✓ UA-DETRAC
    ✓ UCF-Crime

Author: Thanush
============================================================
"""

from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATASET_ROOT = PROJECT_ROOT / "ai" / "datasets"

VISDRONE_ROOT = DATASET_ROOT / "VisDrone"
UA_DETRAC_ROOT = DATASET_ROOT / "UA_DETRAC"
UCF_ROOT = DATASET_ROOT / "UCF_Crime"

LINE = "=" * 70

overall_status = True


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def count_files(folder, extensions):
    """Count files with given extensions."""

    if not folder.exists():
        return 0

    count = 0

    for ext in extensions:
        count += len(list(folder.rglob(f"*{ext}")))

    return count


def header(title):

    print("\n")
    print(LINE)
    print(title)
    print(LINE)


# ============================================================
# VISDRONE
# ============================================================

def verify_visdrone():

    global overall_status

    header("VERIFYING VISDRONE")

    if not VISDRONE_ROOT.exists():

        print("❌ VisDrone folder missing")

        overall_status = False

        return

    splits = {

        "TRAIN": "VisDrone2019-VID-train",

        "VALID": "VisDrone2019-VID-val",

        "TEST": "VisDrone2019-VID-test-dev"

    }

    for split_name, folder_name in splits.items():

        folder = VISDRONE_ROOT / folder_name

        image_dir = folder / "sequences"

        label_dir = folder / "annotations"

        image_count = count_files(
            image_dir,
            [".jpg"]
        )

        label_count = count_files(
            label_dir,
            [".txt"]
        )

        print(f"\n{split_name}")

        print(f"Images      : {image_count}")

        print(f"Annotations : {label_count}")

        if image_count == 0:

            overall_status = False

            print("⚠ Images missing")

        if label_count == 0:

            overall_status = False

            print("⚠ Labels missing")


# ============================================================
# UA-DETRAC
# ============================================================

def verify_uadetrac():

    global overall_status

    header("VERIFYING UA-DETRAC")

    if not UA_DETRAC_ROOT.exists():

        print("❌ UA_DETRAC folder missing")

        overall_status = False

        return

    for split in ["train", "valid", "test"]:

        image_dir = UA_DETRAC_ROOT / split / "images"

        label_dir = UA_DETRAC_ROOT / split / "labels"

        image_count = count_files(
            image_dir,
            [".jpg", ".png", ".jpeg"]
        )

        label_count = count_files(
            label_dir,
            [".txt"]
        )

        print(f"\n{split.upper()}")

        print(f"Images : {image_count}")

        print(f"Labels : {label_count}")

        if image_count == 0:

            overall_status = False

            print("⚠ Images missing")

        if label_count == 0:

            overall_status = False

            print("⚠ Labels missing")


# ============================================================
# UCF CRIME
# ============================================================

def verify_ucfcrime():

    global overall_status

    header("VERIFYING UCF-CRIME")

    if not UCF_ROOT.exists():

        print("❌ UCF_Crime folder missing")

        overall_status = False

        return

    total_videos = 0

    event_folders = sorted(
        [x for x in UCF_ROOT.iterdir() if x.is_dir()]
    )

    if len(event_folders) == 0:

        print("⚠ No event folders found")

        overall_status = False

        return

    for folder in event_folders:

        videos = count_files(
            folder,
            [
                ".mp4",
                ".avi",
                ".mov",
                ".mkv"
            ]
        )

        total_videos += videos

        print(f"{folder.name:<25} : {videos}")

    print()

    print(f"Total Event Classes : {len(event_folders)}")

    print(f"Total Videos        : {total_videos}")

    if total_videos == 0:

        overall_status = False


# ============================================================
# SUMMARY
# ============================================================

def summary():

    header("SUMMARY")

    if overall_status:

        print("✅ ALL DATASETS VERIFIED SUCCESSFULLY")

    else:

        print("❌ SOME DATASETS HAVE ISSUES")

    print(LINE)


# ============================================================
# MAIN
# ============================================================

def main():

    print(LINE)
    print("AI VIDEO INTELLIGENCE DATASET VERIFICATION")
    print(LINE)

    verify_visdrone()

    verify_uadetrac()

    verify_ucfcrime()

    summary()


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()
import os

# ==============================
# VisDrone Dataset Verification
# ==============================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

VISDRONE_ROOT = os.path.join(
    PROJECT_ROOT,
    "ai",
    "datasets",
    "VisDrone"
)

print("=" * 60)
print("VISDRONE DATASET VERIFICATION")
print("=" * 60)

print("\nDataset Location:")
print(VISDRONE_ROOT)

print("\nChecking dataset...\n")

if not os.path.exists(VISDRONE_ROOT):
    print("❌ VisDrone folder not found!")
    exit()

print("✅ VisDrone folder found\n")

folders = [
    "VisDrone2019-VID-train",
    "VisDrone2019-VID-val",
    "VisDrone2019-VID-test-dev"
]

for folder in folders:

    folder_path = os.path.join(VISDRONE_ROOT, folder)

    print("=" * 60)
    print(folder)

    if not os.path.exists(folder_path):
        print("❌ Folder Missing")
        continue

    print("✅ Folder Exists")

    ann = os.path.join(folder_path, "annotations")
    seq = os.path.join(folder_path, "sequences")

    print()

    if os.path.exists(ann):
        print("✅ annotations folder found")
    else:
        print("⚠ annotations folder missing")

    if os.path.exists(seq):
        print("✅ sequences folder found")
    else:
        print("⚠ sequences folder missing")

    sequence_count = 0

    if os.path.exists(seq):
        sequence_count = len(
            [
                x for x in os.listdir(seq)
                if os.path.isdir(os.path.join(seq, x))
            ]
        )

    annotation_count = 0

    if os.path.exists(ann):
        annotation_count = len(
            [
                x for x in os.listdir(ann)
                if x.endswith(".txt")
            ]
        )

    print()

    print(f"Sequences : {sequence_count}")
    print(f"Annotations : {annotation_count}")

print("\n")
print("=" * 60)
print("Verification Complete")
print("=" * 60)

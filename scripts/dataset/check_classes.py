from pathlib import Path

print("Starting...")

ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = ROOT / "ai" / "datasets" / "Merged"

print("Label Directory:", LABEL_DIR)

count = 0
classes = set()

for split in ["train", "valid", "test"]:
    print(f"\nChecking {split}...")

    for label in (LABEL_DIR / split / "labels").glob("*.txt"):
        count += 1

        if count % 5000 == 0:
            print(f"Processed {count} label files...")

        with open(label, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    classes.add(int(line.split()[0]))

print("\nFinished!")
print("Total label files:", count)
print("Classes:", sorted(classes))
print("Total Classes:", len(classes))
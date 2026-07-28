from pathlib import Path

from ai.embeddings.embedding_pipeline import EmbeddingPipeline
from ai.embeddings.chroma_manager import ChromaManager

# -----------------------------
# Directories
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

CROPS_DIR = (
    BASE_DIR
    / "outputs"
    / "phase3"
    / "production_runs"
    / "test"
    / "04_representative_selection"
    / "crops"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "phase5"

print("=" * 60)
print("Phase 5 - Embedding Generation Test")
print("=" * 60)
print(f"Crops Directory : {CROPS_DIR}")
print(f"Output Directory: {OUTPUT_DIR}")

# -----------------------------
# Check directories
# -----------------------------
if not CROPS_DIR.exists():
    raise FileNotFoundError(
        f"\nRepresentative crops folder not found:\n{CROPS_DIR}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Count images
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

image_files = [
    f
    for f in CROPS_DIR.rglob("*")
    if f.is_file() and f.suffix.lower() in image_extensions
]

print(f"\nFound {len(image_files)} images.\n")

if len(image_files) == 0:
    raise RuntimeError("No representative crop images found.")

# -----------------------------
# Generate Embeddings
# -----------------------------
pipeline = EmbeddingPipeline()

pipeline.generate_embeddings(
    crops_directory=str(CROPS_DIR),
    output_directory=str(OUTPUT_DIR),
)

print("\nIndexing embeddings into ChromaDB...")

chroma = ChromaManager(
    db_path="database/chromadb",
    collection_name="image_embeddings",
)

chroma.store(
    embedding_file=OUTPUT_DIR / "image_embeddings.npy",
    metadata_file=OUTPUT_DIR / "embedding_metadata.json",
)

print("ChromaDB indexing completed.")
print("\n")
print("=" * 60)
print("Phase 5 Completed Successfully")
print("=" * 60)
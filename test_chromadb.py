from ai.embeddings.chroma_manager import ChromaManager

manager = ChromaManager()

manager.store(
    embedding_file="outputs/phase5/image_embeddings.npy",
    metadata_file="outputs/phase5/embedding_metadata.json",
)
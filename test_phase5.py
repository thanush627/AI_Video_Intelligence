from ai.embeddings.embedding_pipeline import EmbeddingPipeline

pipeline = EmbeddingPipeline()

pipeline.generate_embeddings(
    crops_directory=r"outputs\phase3\production_runs\test\04_representative_selection\crops",
    output_directory=r"outputs\phase5",
)
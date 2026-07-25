from ai.embeddings.embedding_generator import EmbeddingGenerator

generator = EmbeddingGenerator()

image_path = r"outputs\phase3\production_runs\test\04_representative_selection\crops\bicycle_track_18\rank_1_frame_000102.jpg"

embedding = generator.generate_image_embedding(image_path)

print("Embedding Shape:", embedding.shape)
print("First 10 Values:")
print(embedding[:10])
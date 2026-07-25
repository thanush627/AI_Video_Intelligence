from ai.embeddings.postgres_manager import PostgresManager

db = PostgresManager()

db.insert_embeddings(
    "outputs/phase5/embedding_metadata.json"
)
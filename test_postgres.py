from ai.embeddings.postgres_manager import engine, Base

# Import models so SQLAlchemy registers them
from ai.embeddings import models

print("Creating tables...")

Base.metadata.create_all(bind=engine)

print("Done!")
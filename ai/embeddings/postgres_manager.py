import json

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = (
    "postgresql+psycopg2://"
    "postgres:password@localhost:5432/video_ai"
)

engine = create_engine(DATABASE_URL, echo=False)

SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()


def get_session():
    return SessionLocal()


from ai.embeddings.models import Embedding


class PostgresManager:

    def __init__(self):
        self.session = get_session()

    def insert_embeddings(self, metadata_file):

        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        inserted = 0

        for item in metadata:

            exists = (
                self.session.query(Embedding)
                .filter_by(
                    embedding_id=item["embedding_id"]
                )
                .first()
            )

            if exists:
                continue

            row = Embedding(
                embedding_id=item["embedding_id"],
                track_id=item["track_id"],
                image_path=item["image_path"],
                rank=item["rank"],
            )

            self.session.add(row)
            inserted += 1

        self.session.commit()

        print(f"Inserted {inserted} rows.")
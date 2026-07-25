from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    ForeignKey,
    Text,
)

from ai.embeddings.postgres_manager import Base


class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True)

    video_id = Column(String, unique=True)
    filename = Column(String)
    duration = Column(Float)


class Track(Base):
    __tablename__ = "tracks"

    id = Column(Integer, primary_key=True)

    track_id = Column(String, unique=True)
    video_id = Column(
        String,
        ForeignKey("videos.video_id"),
    )

    object_type = Column(String)

    start_frame = Column(Integer)
    end_frame = Column(Integer)


class Object(Base):
    __tablename__ = "objects"

    id = Column(Integer, primary_key=True)

    object_id = Column(String, unique=True)

    track_id = Column(String)

    label = Column(String)

    confidence = Column(Float)

    color = Column(String)

    attributes = Column(Text)


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True)

    event_id = Column(String, unique=True)

    track_id = Column(String)

    action = Column(String)

    start_time = Column(Float)

    end_time = Column(Float)

    description = Column(Text)


class Embedding(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True)

    embedding_id = Column(String, unique=True)

    track_id = Column(String)

    image_path = Column(String)

    rank = Column(Integer)
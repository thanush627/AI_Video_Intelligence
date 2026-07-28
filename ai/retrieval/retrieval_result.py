"""
Phase 7

Standard Retrieval Result

Every retriever returns this object.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class RetrievalResult:

    # -------------------------------------------------
    # Required Fields
    # -------------------------------------------------

    track_id: str
    video_id: str
    object_type: str

    # -------------------------------------------------
    # Optional Identification
    # -------------------------------------------------

    event_id: Optional[str] = None

    # -------------------------------------------------
    # Image Information
    # -------------------------------------------------

    image_name: Optional[str] = None
    image_path: Optional[str] = None

    # -------------------------------------------------
    # Event Information
    # -------------------------------------------------

    event_type: Optional[str] = None

    timestamp: Optional[str] = None

    start_time: Optional[str] = None
    end_time: Optional[str] = None

    # -------------------------------------------------
    # Retrieval Scores
    # -------------------------------------------------

    similarity: float = 0.0

    confidence: float = 0.0

    quality_score: float = 0.0

    # -------------------------------------------------
    # Original Metadata
    # -------------------------------------------------

    metadata: Dict = field(default_factory=dict)

    source: str = ""

    # -------------------------------------------------
    # Convert to Dictionary
    # -------------------------------------------------

    def to_dict(self) -> Dict:

        return {

            "track_id": self.track_id,

            "video_id": self.video_id,

            "object_type": self.object_type,

            "event_id": self.event_id,

            "image_name": self.image_name,

            "image_path": self.image_path,

            "event_type": self.event_type,

            "timestamp": self.timestamp,

            "start_time": self.start_time,

            "end_time": self.end_time,

            "similarity": self.similarity,

            "confidence": self.confidence,

            "quality_score": self.quality_score,

            "metadata": self.metadata,

            "source": self.source

        }

    # -------------------------------------------------
    # Pretty Print
    # -------------------------------------------------

    def __repr__(self):

        return (

            f"RetrievalResult("
            f"track_id='{self.track_id}', "
            f"object_type='{self.object_type}', "
            f"confidence={self.confidence:.3f}, "
            f"similarity={self.similarity:.3f}, "
            f"source='{self.source}')"

        )


# -------------------------------------------------
# Testing
# -------------------------------------------------

if __name__ == "__main__":

    result = RetrievalResult(

        track_id="person_track_1",

        video_id="test_video",

        object_type="person",

        event_id="event_001",

        image_name="frame_000123.jpg",

        image_path="outputs/phase3/frame_000123.jpg",

        event_type="object_event",

        timestamp="00:00:12",

        start_time="00:00:10",

        end_time="00:00:15",

        similarity=0.95,

        confidence=0.91,

        quality_score=0.88,

        metadata={"helmet": True},

        source="metadata"

    )

    print(result)

    print(result.to_dict())
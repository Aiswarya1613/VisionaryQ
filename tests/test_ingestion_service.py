import pytest

from app.services.ingestion_service import IngestionService


class FakeVectorService:
    """
    Fake vector service used to verify what the
    ingestion pipeline attempts to store.
    """

    def __init__(self):
        self.documents = []

    def add_documents(self, documents):
        self.documents = documents

        return {
            "upserted_count": len(documents)
        }


def sample_transcription():
    """
    Deterministic fake Faster-Whisper output.
    """

    return {
        "text": (
            "Generative AI creates new content. "
            "It can generate text images music and video. "
            "Transformers process tokens and context."
        ),
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "text": (
                    "Generative AI creates new content."
                ),
            },
            {
                "start": 5.0,
                "end": 10.0,
                "text": (
                    "It can generate text images music "
                    "and video."
                ),
            },
            {
                "start": 10.0,
                "end": 15.0,
                "text": (
                    "Transformers process tokens "
                    "and context."
                ),
            },
        ],
        "language": "en",
        "language_probability": 1.0,
    }


# ---------------------------------------------------------
# Timestamp-aware chunking
# ---------------------------------------------------------

def test_build_timestamped_chunks_preserves_times():
    segments = sample_transcription()["segments"]

    chunks = (
        IngestionService
        ._build_timestamped_chunks(
            segments=segments,
            target_chars=100,
            overlap_segments=1,
        )
    )

    assert len(chunks) == 2

    assert chunks[0]["start_time"] == 0.0
    assert chunks[0]["end_time"] == 10.0

    assert chunks[1]["start_time"] == 5.0
    assert chunks[1]["end_time"] == 15.0


def test_build_timestamped_chunks_preserves_overlap():
    segments = sample_transcription()["segments"]

    chunks = (
        IngestionService
        ._build_timestamped_chunks(
            segments=segments,
            target_chars=100,
            overlap_segments=1,
        )
    )

    # Second Whisper segment should appear
    # in both neighbouring chunks.
    shared_text = (
        "It can generate text images music "
        "and video."
    )

    assert shared_text in chunks[0]["text"]
    assert shared_text in chunks[1]["text"]


def test_build_timestamped_chunks_cleans_text():
    segments = [
        {
            "start": 0.0,
            "end": 5.0,
            "text": "  Generative    AI   creates content.  ",
        }
    ]

    chunks = (
        IngestionService
        ._build_timestamped_chunks(
            segments=segments,
            target_chars=100,
            overlap_segments=0,
        )
    )

    assert chunks[0]["text"] == (
        "Generative AI creates content."
    )


def test_build_timestamped_chunks_empty_segments():
    chunks = (
        IngestionService
        ._build_timestamped_chunks(
            segments=[],
            target_chars=100,
            overlap_segments=1,
        )
    )

    assert chunks == []


def test_build_timestamped_chunks_rejects_invalid_target_chars():
    with pytest.raises(
        ValueError,
        match="target_chars must be greater than 0"
    ):
        (
            IngestionService
            ._build_timestamped_chunks(
                segments=[],
                target_chars=0,
                overlap_segments=1,
            )
        )


def test_build_timestamped_chunks_rejects_negative_overlap():
    with pytest.raises(
        ValueError,
        match="overlap_segments cannot be negative"
    ):
        (
            IngestionService
            ._build_timestamped_chunks(
                segments=[],
                target_chars=100,
                overlap_segments=-1,
            )
        )


# ---------------------------------------------------------
# Complete ingestion orchestration
# ---------------------------------------------------------

def test_ingest_video_builds_pinecone_documents(
    monkeypatch
):
    fake_vector_service = FakeVectorService()

    ingestion_service = IngestionService(
        vector_service=fake_vector_service
    )

    monkeypatch.setattr(
        "app.services.ingestion_service.transcribe_video",
        lambda video_path: sample_transcription(),
    )

    result = ingestion_service.ingest_video(
        video_path="fake_video.mp4",
        video_id="video-001",
        filename="fake_video.mp4",
    )

    assert result["status"] == "success"
    assert result["video_id"] == "video-001"
    assert result["filename"] == "fake_video.mp4"

    assert result["language"] == "en"
    assert result["language_probability"] == 1.0

    assert result["segment_count"] == 3

    assert (
        result["chunk_count"]
        == result["upserted_count"]
    )

    assert len(
        fake_vector_service.documents
    ) == result["chunk_count"]


def test_ingest_video_adds_timestamp_metadata(
    monkeypatch
):
    fake_vector_service = FakeVectorService()

    ingestion_service = IngestionService(
        vector_service=fake_vector_service
    )

    monkeypatch.setattr(
        "app.services.ingestion_service.transcribe_video",
        lambda video_path: sample_transcription(),
    )

    ingestion_service.ingest_video(
        video_path="fake_video.mp4",
        video_id="video-123",
        filename="fake_video.mp4",
    )

    document = (
        fake_vector_service.documents[0]
    )

    assert document["id"] == (
        "video-123-chunk-0"
    )

    assert document["metadata"][
        "video_id"
    ] == "video-123"

    assert document["metadata"][
        "filename"
    ] == "fake_video.mp4"

    assert document["metadata"][
        "chunk_index"
    ] == 0

    assert document["metadata"][
        "start_time"
    ] == 0.0

    assert document["metadata"][
        "end_time"
    ] == 15.0


# ---------------------------------------------------------
# Validation / failure handling
# ---------------------------------------------------------

def test_ingest_video_rejects_empty_video_id():
    ingestion_service = IngestionService(
        vector_service=FakeVectorService()
    )

    with pytest.raises(
        ValueError,
        match="video_id cannot be empty"
    ):
        ingestion_service.ingest_video(
            video_path="fake_video.mp4",
            video_id="",
            filename="fake_video.mp4",
        )


def test_ingest_video_rejects_empty_filename():
    ingestion_service = IngestionService(
        vector_service=FakeVectorService()
    )

    with pytest.raises(
        ValueError,
        match="filename cannot be empty"
    ):
        ingestion_service.ingest_video(
            video_path="fake_video.mp4",
            video_id="video-001",
            filename="",
        )


def test_ingest_video_rejects_empty_transcript(
    monkeypatch
):
    fake_vector_service = FakeVectorService()

    ingestion_service = IngestionService(
        vector_service=fake_vector_service
    )

    monkeypatch.setattr(
        "app.services.ingestion_service.transcribe_video",
        lambda video_path: {
            "text": "",
            "segments": [],
            "language": "en",
            "language_probability": 1.0,
        },
    )

    with pytest.raises(
        ValueError,
        match="No transcribable text was found"
    ):
        ingestion_service.ingest_video(
            video_path="fake_video.mp4",
            video_id="video-001",
            filename="fake_video.mp4",
        )

    # Failed ingestion must never write vectors.
    assert fake_vector_service.documents == []


def test_ingest_video_rejects_missing_segments(
    monkeypatch
):
    fake_vector_service = FakeVectorService()

    ingestion_service = IngestionService(
        vector_service=fake_vector_service
    )

    monkeypatch.setattr(
        "app.services.ingestion_service.transcribe_video",
        lambda video_path: {
            "text": "Valid transcript text.",
            "segments": [],
            "language": "en",
            "language_probability": 1.0,
        },
    )

    with pytest.raises(
        ValueError,
        match=(
            "No timestamped transcription "
            "segments were generated"
        )
    ):
        ingestion_service.ingest_video(
            video_path="fake_video.mp4",
            video_id="video-001",
            filename="fake_video.mp4",
        )

    assert fake_vector_service.documents == []
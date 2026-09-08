import os

from fastapi.testclient import TestClient

import app.api.routes as routes_module
from app.main import app


client = TestClient(app)


class FakeIngestionService:
    """
    Fake ingestion service for API tests.
    """

    def __init__(
        self,
        result=None,
        error=None
    ):
        self.result = result or {
            "video_id": "generated-video-id",
            "filename": "test_video.mp4",
            "status": "success",
            "language": "en",
            "language_probability": 1.0,
            "segment_count": 10,
            "chunk_count": 3,
            "upserted_count": 3,
        }

        self.error = error
        self.calls = []
        self.temporary_path = None
        self.uploaded_bytes = None

    def ingest_video(
        self,
        video_path,
        video_id,
        filename
    ):
        self.temporary_path = video_path

        # The temporary uploaded file should exist
        # while ingestion is running.
        assert os.path.exists(video_path)

        with open(
            video_path,
            "rb"
        ) as uploaded_file:
            self.uploaded_bytes = (
                uploaded_file.read()
            )

        self.calls.append(
            {
                "video_path": video_path,
                "video_id": video_id,
                "filename": filename,
            }
        )

        if self.error is not None:
            raise self.error

        result = dict(
            self.result
        )

        # The API generates the real UUID.
        result["video_id"] = video_id
        result["filename"] = filename

        return result


class FakeRAGService:
    """
    Fake RAG service for API tests.
    """

    def __init__(
        self,
        result=None,
        error=None
    ):
        self.result = result or {
            "query": "What is generative AI?",
            "video_id": "video-001",
            "answer": (
                "Generative AI creates "
                "new content."
            ),
            "context": (
                "Generative AI creates "
                "new content."
            ),
            "sources": [
                {
                    "id": "video-001-chunk-0",
                    "score": 0.82,
                    "text": (
                        "Generative AI creates "
                        "new content."
                    ),
                    "metadata": {
                        "video_id": "video-001",
                        "filename": "video.mp4",
                        "chunk_index": 0,
                        "start_time": 0.0,
                        "end_time": 10.0,
                    },
                }
            ],
            "retrieval_threshold": 0.30,
        }

        self.error = error
        self.calls = []

    def query(
        self,
        query,
        top_k=5,
        video_id=None
    ):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "video_id": video_id,
            }
        )

        if self.error is not None:
            raise self.error

        result = dict(
            self.result
        )

        result["query"] = query
        result["video_id"] = video_id

        return result


# ---------------------------------------------------------
# Basic endpoints
# ---------------------------------------------------------

def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")

    assert "VisionaryQ" in response.text
    assert "/static/styles.css" in response.text
    assert "/static/script.js" in response.text

def test_frontend_styles_are_served():
    response = client.get("/static/styles.css")

    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]
    assert ".container" in response.text


def test_frontend_script_is_served():
    response = client.get("/static/script.js")

    assert response.status_code == 200

    assert "/api/v1/video/ingest" in response.text
    assert "/api/v1/query" in response.text
    assert "currentVideoId" in response.text

def test_health_endpoint():
    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy"
    }


def test_api_status_endpoint():
    response = client.get(
        "/api/v1/status"
    )

    assert response.status_code == 200

    assert response.json() == {
        "service": "VisionaryQ API",
        "status": "operational",
        "version": "1.0.0",
    }


# ---------------------------------------------------------
# Video ingestion
# ---------------------------------------------------------

def test_video_ingest_success(
    monkeypatch
):
    fake_service = (
        FakeIngestionService()
    )

    monkeypatch.setattr(
        routes_module,
        "get_ingestion_service",
        lambda: fake_service,
    )

    video_bytes = (
        b"fake-video-binary-data"
    )

    response = client.post(
        "/api/v1/video/ingest",
        files={
            "file": (
                "test_video.mp4",
                video_bytes,
                "video/mp4",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["filename"] == (
        "test_video.mp4"
    )

    assert data["status"] == "success"
    assert data["language"] == "en"

    assert data[
        "language_probability"
    ] == 1.0

    assert data["segment_count"] == 10
    assert data["chunk_count"] == 3
    assert data["upserted_count"] == 3

    # UUID should have been generated.
    assert data["video_id"]

    assert len(
        fake_service.calls
    ) == 1

    call = fake_service.calls[0]

    assert call["filename"] == (
        "test_video.mp4"
    )

    assert call["video_id"] == (
        data["video_id"]
    )

    assert (
        fake_service.uploaded_bytes
        == video_bytes
    )

    # Temporary upload must be deleted
    # after the request completes.
    assert not os.path.exists(
        fake_service.temporary_path
    )


def test_video_ingest_rejects_unsupported_extension():
    response = client.post(
        "/api/v1/video/ingest",
        files={
            "file": (
                "document.txt",
                b"not-a-video",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400

    assert (
        "Unsupported video format"
        in response.json()["detail"]
    )


def test_video_ingest_converts_value_error_to_400(
    monkeypatch
):
    fake_service = FakeIngestionService(
        error=ValueError(
            "No transcribable text was found."
        )
    )

    monkeypatch.setattr(
        routes_module,
        "get_ingestion_service",
        lambda: fake_service,
    )

    response = client.post(
        "/api/v1/video/ingest",
        files={
            "file": (
                "silent.mp4",
                b"fake-video",
                "video/mp4",
            )
        },
    )

    assert response.status_code == 400

    assert response.json()["detail"] == (
        "No transcribable text was found."
    )

    # Cleanup should still occur after failure.
    assert not os.path.exists(
        fake_service.temporary_path
    )


# ---------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------

def test_query_endpoint_success(
    monkeypatch
):
    fake_service = FakeRAGService()

    monkeypatch.setattr(
        routes_module,
        "get_rag_service",
        lambda: fake_service,
    )

    response = client.post(
        "/api/v1/query",
        json={
            "video_id": "video-001",
            "query": (
                "What is generative AI?"
            ),
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["video_id"] == (
        "video-001"
    )

    assert data["answer"] == (
        "Generative AI creates "
        "new content."
    )

    assert len(data["sources"]) == 1

    assert data["sources"][0][
        "metadata"
    ]["start_time"] == 0.0

    assert data["sources"][0][
        "metadata"
    ]["end_time"] == 10.0

    assert fake_service.calls == [
        {
            "query": (
                "What is generative AI?"
            ),
            "top_k": 3,
            "video_id": "video-001",
        }
    ]


def test_query_rejects_top_k_zero():
    response = client.post(
        "/api/v1/query",
        json={
            "video_id": "video-001",
            "query": "Test question",
            "top_k": 0,
        },
    )

    # Pydantic/FastAPI validation error.
    assert response.status_code == 422


def test_query_rejects_missing_video_id():
    response = client.post(
        "/api/v1/query",
        json={
            "query": "Test question",
            "top_k": 3,
        },
    )

    assert response.status_code == 422


def test_query_converts_value_error_to_400(
    monkeypatch
):
    fake_service = FakeRAGService(
        error=ValueError(
            "Query cannot be empty."
        )
    )

    monkeypatch.setattr(
        routes_module,
        "get_rag_service",
        lambda: fake_service,
    )

    response = client.post(
        "/api/v1/query",
        json={
            "video_id": "video-001",
            "query": "   ",
            "top_k": 3,
        },
    )

    assert response.status_code == 400

    assert response.json()["detail"] == (
        "Query cannot be empty."
    )
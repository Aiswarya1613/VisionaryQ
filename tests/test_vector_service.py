from types import SimpleNamespace

import pytest

import app.services.vector_service as vector_module
from app.services.vector_service import VectorService


class FakeSettings:
    def __init__(
        self,
        api_key="fake-api-key",
        index_name="visionaryq"
    ):
        self.pinecone_api_key = api_key
        self.pinecone_index_name = index_name


class FakeEmbeddingService:
    def __init__(self):
        self.document_inputs = []
        self.query_inputs = []

    def get_dimension(self):
        return 384

    def embed_documents(self, texts):
        self.document_inputs.append(texts)

        return [
            [0.1, 0.2, 0.3]
            for _ in texts
        ]

    def embed_text(self, text):
        self.query_inputs.append(text)

        return [0.4, 0.5, 0.6]


class FakeIndex:
    def __init__(self):
        self.upsert_calls = []
        self.query_calls = []
        self.query_matches = []

    def upsert(self, vectors):
        self.upsert_calls.append(vectors)

    def query(self, **kwargs):
        self.query_calls.append(kwargs)

        return SimpleNamespace(
            matches=self.query_matches
        )


class FakePineconeClient:
    def __init__(
        self,
        existing_indexes=None
    ):
        self.existing_indexes = (
            existing_indexes or []
        )

        self.created_indexes = []
        self.fake_index = FakeIndex()
        self.requested_index_name = None

    def list_indexes(self):
        return [
            SimpleNamespace(name=name)
            for name in self.existing_indexes
        ]

    def create_index(self, **kwargs):
        self.created_indexes.append(kwargs)

    def Index(self, index_name):
        self.requested_index_name = index_name
        return self.fake_index


def configure_fake_settings(
    monkeypatch,
    api_key="fake-api-key",
    index_name="visionaryq"
):
    settings = FakeSettings(
        api_key=api_key,
        index_name=index_name
    )

    monkeypatch.setattr(
        vector_module,
        "get_settings",
        lambda: settings
    )

    return settings


def configure_fake_pinecone(
    monkeypatch,
    existing_indexes=None
):
    client = FakePineconeClient(
        existing_indexes=existing_indexes
    )

    monkeypatch.setattr(
        vector_module,
        "Pinecone",
        lambda api_key: client
    )

    return client


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

def test_vector_service_uses_central_configuration(
    monkeypatch
):
    configure_fake_settings(
        monkeypatch,
        api_key="test-key",
        index_name="test-index"
    )

    service = VectorService(
        embedding_service=FakeEmbeddingService()
    )

    assert service.api_key == "test-key"
    assert service.index_name == "test-index"


# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------

def test_initialize_rejects_missing_api_key(
    monkeypatch
):
    configure_fake_settings(
        monkeypatch,
        api_key=None
    )

    service = VectorService(
        embedding_service=FakeEmbeddingService()
    )

    with pytest.raises(
        RuntimeError,
        match="PINECONE_API_KEY is not configured"
    ):
        service.initialize()


def test_initialize_connects_to_existing_index(
    monkeypatch
):
    configure_fake_settings(
        monkeypatch
    )

    client = configure_fake_pinecone(
        monkeypatch,
        existing_indexes=["visionaryq"]
    )

    service = VectorService(
        embedding_service=FakeEmbeddingService()
    )

    service.initialize()

    assert service.initialized is True
    assert service.index is client.fake_index

    assert (
        client.requested_index_name
        == "visionaryq"
    )

    assert client.created_indexes == []


def test_initialize_creates_missing_index(
    monkeypatch
):
    configure_fake_settings(
        monkeypatch
    )

    client = configure_fake_pinecone(
        monkeypatch,
        existing_indexes=[]
    )

    embedding_service = (
        FakeEmbeddingService()
    )

    service = VectorService(
        embedding_service=embedding_service
    )

    service.initialize()

    assert len(
        client.created_indexes
    ) == 1

    create_call = (
        client.created_indexes[0]
    )

    assert create_call["name"] == "visionaryq"
    assert create_call["dimension"] == 384
    assert create_call["metric"] == "cosine"

    assert service.initialized is True


# ---------------------------------------------------------
# add_documents
# ---------------------------------------------------------

def test_add_documents_requires_initialization(
    monkeypatch
):
    configure_fake_settings(
        monkeypatch
    )

    service = VectorService(
        embedding_service=FakeEmbeddingService()
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Vector service has not been "
            "initialized"
        )
    ):
        service.add_documents(
            [
                {
                    "id": "chunk-1",
                    "text": "Some content.",
                    "metadata": {},
                }
            ]
        )


def test_add_documents_empty_list_returns_zero(
    monkeypatch
):
    configure_fake_settings(
        monkeypatch
    )

    service = VectorService(
        embedding_service=FakeEmbeddingService()
    )

    service.initialized = True
    service.index = FakeIndex()

    result = service.add_documents([])

    assert result == {
        "upserted_count": 0
    }


def test_add_documents_embeds_and_upserts_documents(
    monkeypatch
):
    configure_fake_settings(
        monkeypatch
    )

    embedding_service = (
        FakeEmbeddingService()
    )

    fake_index = FakeIndex()

    service = VectorService(
        embedding_service=embedding_service
    )

    service.initialized = True
    service.index = fake_index

    documents = [
        {
            "id": "video-001-chunk-0",
            "text": "Generative AI creates content.",
            "metadata": {
                "video_id": "video-001",
                "chunk_index": 0,
                "start_time": 0.0,
                "end_time": 10.0,
            },
        },
        {
            "id": "video-001-chunk-1",
            "text": "Transformers process tokens.",
            "metadata": {
                "video_id": "video-001",
                "chunk_index": 1,
                "start_time": 10.0,
                "end_time": 20.0,
            },
        },
    ]

    result = service.add_documents(
        documents
    )

    assert result["upserted_count"] == 2

    assert embedding_service.document_inputs == [
        [
            "Generative AI creates content.",
            "Transformers process tokens.",
        ]
    ]

    assert len(fake_index.upsert_calls) == 1

    vectors = fake_index.upsert_calls[0]

    assert len(vectors) == 2

    assert vectors[0]["id"] == (
        "video-001-chunk-0"
    )

    assert vectors[0]["values"] == [
        0.1,
        0.2,
        0.3,
    ]

    assert vectors[0]["metadata"][
        "video_id"
    ] == "video-001"

    assert vectors[0]["metadata"][
        "start_time"
    ] == 0.0

    assert vectors[0]["metadata"][
        "end_time"
    ] == 10.0

    assert vectors[0]["metadata"][
        "text"
    ] == (
        "Generative AI creates content."
    )


def test_add_documents_rejects_missing_id(
    monkeypatch
):
    configure_fake_settings(
        monkeypatch
    )

    service = VectorService(
        embedding_service=FakeEmbeddingService()
    )

    service.initialized = True
    service.index = FakeIndex()

    with pytest.raises(
        ValueError,
        match=(
            "Every document must contain "
            "an 'id'"
        )
    ):
        service.add_documents(
            [
                {
                    "text": "Missing ID.",
                    "metadata": {},
                }
            ]
        )


# ---------------------------------------------------------
# search
# ---------------------------------------------------------

def test_search_requires_initialization(
    monkeypatch
):
    configure_fake_settings(
        monkeypatch
    )

    service = VectorService(
        embedding_service=FakeEmbeddingService()
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Vector service has not been "
            "initialized"
        )
    ):
        service.search(
            "What is generative AI?"
        )


def test_search_rejects_empty_query(
    monkeypatch
):
    configure_fake_settings(
        monkeypatch
    )

    service = VectorService(
        embedding_service=FakeEmbeddingService()
    )

    service.initialized = True
    service.index = FakeIndex()

    with pytest.raises(
        ValueError,
        match="Query cannot be empty"
    ):
        service.search("   ")


@pytest.mark.parametrize(
    "top_k",
    [0, -1, -10]
)
def test_search_rejects_invalid_top_k(
    monkeypatch,
    top_k
):
    configure_fake_settings(
        monkeypatch
    )

    service = VectorService(
        embedding_service=FakeEmbeddingService()
    )

    service.initialized = True
    service.index = FakeIndex()

    with pytest.raises(
        ValueError,
        match=(
            "top_k must be greater than zero"
        )
    ):
        service.search(
            "valid query",
            top_k=top_k
        )


def test_search_uses_video_filter_and_converts_results(
    monkeypatch
):
    configure_fake_settings(
        monkeypatch
    )

    embedding_service = (
        FakeEmbeddingService()
    )

    fake_index = FakeIndex()

    fake_index.query_matches = [
        SimpleNamespace(
            id="video-001-chunk-0",
            score=0.82,
            metadata={
                "video_id": "video-001",
                "filename": "video.mp4",
                "chunk_index": 0,
                "start_time": 0.0,
                "end_time": 10.0,
                "text": (
                    "Generative AI creates content."
                ),
            }
        ),
        SimpleNamespace(
            id="video-001-chunk-1",
            score=0.65,
            metadata={
                "video_id": "video-001",
                "filename": "video.mp4",
                "chunk_index": 1,
                "start_time": 10.0,
                "end_time": 20.0,
                "text": (
                    "Transformers process tokens."
                ),
            }
        ),
    ]

    service = VectorService(
        embedding_service=embedding_service
    )

    service.initialized = True
    service.index = fake_index

    results = service.search(
        query="What is generative AI?",
        top_k=2,
        video_id="video-001",
    )

    assert embedding_service.query_inputs == [
        "What is generative AI?"
    ]

    assert len(
        fake_index.query_calls
    ) == 1

    query_call = (
        fake_index.query_calls[0]
    )

    assert query_call["vector"] == [
        0.4,
        0.5,
        0.6,
    ]

    assert query_call["top_k"] == 2

    assert (
        query_call["include_metadata"]
        is True
    )

    assert query_call["filter"] == {
        "video_id": "video-001"
    }

    assert len(results) == 2

    assert results[0]["id"] == (
        "video-001-chunk-0"
    )

    assert results[0]["score"] == 0.82

    assert results[0]["text"] == (
        "Generative AI creates content."
    )

    assert results[0]["metadata"][
        "start_time"
    ] == 0.0

    assert results[0]["metadata"][
        "end_time"
    ] == 10.0
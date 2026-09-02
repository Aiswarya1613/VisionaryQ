import pytest

from app.services.rag_service import RAGService


class FakeVectorService:
    """
    Fake vector database used for unit tests.
    """

    def __init__(self, documents=None):
        self.documents = documents or []
        self.last_query = None
        self.last_top_k = None
        self.last_video_id = None

    def search(
        self,
        query,
        top_k=5,
        video_id=None
    ):
        self.last_query = query
        self.last_top_k = top_k
        self.last_video_id = video_id

        return self.documents


class FakeLLMService:
    """
    Fake LLM used for unit tests.
    """

    def __init__(
        self,
        response="Generated answer"
    ):
        self.response = response
        self.calls = []

    def generate(
        self,
        query,
        context
    ):
        self.calls.append(
            {
                "query": query,
                "context": context
            }
        )

        return self.response


def make_document(
    document_id,
    text,
    score,
    video_id="video-001",
    start_time=0.0,
    end_time=10.0
):
    return {
        "id": document_id,
        "score": score,
        "text": text,
        "metadata": {
            "video_id": video_id,
            "filename": "test_video.mp4",
            "chunk_index": 0,
            "start_time": start_time,
            "end_time": end_time,
        }
    }


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------

def test_retrieve_forwards_query_parameters():
    vector_service = FakeVectorService()

    rag_service = RAGService(
        vector_service=vector_service,
        llm_service=FakeLLMService(),
        min_score=0.30,
    )

    rag_service.retrieve(
        query="What is generative AI?",
        top_k=3,
        video_id="video-123",
    )

    assert (
        vector_service.last_query
        == "What is generative AI?"
    )

    assert vector_service.last_top_k == 3
    assert vector_service.last_video_id == "video-123"


# ---------------------------------------------------------
# Context construction
# ---------------------------------------------------------

def test_build_context_combines_document_text():
    rag_service = RAGService(
        min_score=0.30
    )

    documents = [
        {
            "text": "Generative AI creates content."
        },
        {
            "text": "It can generate images."
        },
    ]

    context = rag_service.build_context(
        documents
    )

    assert context == (
        "Generative AI creates content."
        "\n\n"
        "It can generate images."
    )


def test_build_context_ignores_empty_text():
    rag_service = RAGService(
        min_score=0.30
    )

    documents = [
        {"text": "Useful information."},
        {"text": ""},
        {},
    ]

    context = rag_service.build_context(
        documents
    )

    assert context == "Useful information."


# ---------------------------------------------------------
# Relevance filtering
# ---------------------------------------------------------

def test_query_filters_documents_below_threshold():
    documents = [
        make_document(
            "chunk-1",
            "Generative AI creates new content.",
            0.80,
        ),
        make_document(
            "chunk-2",
            "Relevant additional information.",
            0.55,
        ),
        make_document(
            "chunk-3",
            "Unrelated information.",
            0.10,
        ),
    ]

    vector_service = FakeVectorService(
        documents
    )

    llm_service = FakeLLMService(
        response="Generative AI creates new content."
    )

    rag_service = RAGService(
        vector_service=vector_service,
        llm_service=llm_service,
        min_score=0.30,
    )

    result = rag_service.query(
        query="What is generative AI?",
        top_k=5,
        video_id="video-001",
    )

    assert len(result["sources"]) == 2

    assert [
        source["id"]
        for source in result["sources"]
    ] == [
        "chunk-1",
        "chunk-2",
    ]

    assert result[
        "retrieval_threshold"
    ] == 0.30

    assert len(llm_service.calls) == 1


def test_score_equal_to_threshold_is_relevant():
    documents = [
        make_document(
            "chunk-boundary",
            "Boundary score content.",
            0.30,
        )
    ]

    rag_service = RAGService(
        vector_service=FakeVectorService(
            documents
        ),
        llm_service=FakeLLMService(),
        min_score=0.30,
    )

    result = rag_service.query(
        query="Test question",
        video_id="video-001",
    )

    assert len(result["sources"]) == 1

    assert (
        result["sources"][0]["score"]
        == 0.30
    )


# ---------------------------------------------------------
# Guardrails
# ---------------------------------------------------------

def test_query_returns_fallback_when_no_documents():
    llm_service = FakeLLMService()

    rag_service = RAGService(
        vector_service=FakeVectorService([]),
        llm_service=llm_service,
        min_score=0.30,
    )

    result = rag_service.query(
        query="Who won the World Cup?",
        video_id="video-001",
    )

    assert result["answer"] == (
        "I could not find that information "
        "in the video."
    )

    assert result["context"] == ""
    assert result["sources"] == []

    # LLM must not be called when retrieval
    # contains no relevant information.
    assert llm_service.calls == []


def test_query_returns_fallback_when_all_scores_are_low():
    documents = [
        make_document(
            "chunk-1",
            "Some unrelated content.",
            0.15,
        ),
        make_document(
            "chunk-2",
            "More unrelated content.",
            0.08,
        ),
    ]

    llm_service = FakeLLMService()

    rag_service = RAGService(
        vector_service=FakeVectorService(
            documents
        ),
        llm_service=llm_service,
        min_score=0.30,
    )

    result = rag_service.query(
        query="Who won the World Cup?",
        video_id="video-001",
    )

    assert result["sources"] == []
    assert result["context"] == ""

    assert llm_service.calls == []


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def test_query_rejects_empty_query():
    rag_service = RAGService(
        vector_service=FakeVectorService(),
        llm_service=FakeLLMService(),
        min_score=0.30,
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty"
    ):
        rag_service.query(
            query="   ",
            video_id="video-001",
        )
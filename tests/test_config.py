import pytest

from app.core.config import Settings, get_settings


ENVIRONMENT_VARIABLES = [
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
    "WHISPER_MODEL",
    "RAG_MIN_SCORE",
    "OLLAMA_MODEL",
    "OLLAMA_HOST",
]


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """
    Ensure every test gets a fresh Settings object.
    """

    get_settings.cache_clear()

    yield

    get_settings.cache_clear()


def remove_config_environment(monkeypatch):
    """
    Remove VisionaryQ configuration variables so
    default-value behavior can be tested safely.

    This does not modify the user's .env file.
    """

    for variable in ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(
            variable,
            raising=False,
        )


def make_settings(
    pinecone_api_key="fake-key",
    pinecone_index_name="visionaryq",
    whisper_model="small.en",
    rag_min_score=0.30,
    ollama_model="llama3.2:3b",
    ollama_host="http://localhost:11434",
):
    return Settings(
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name,
        whisper_model=whisper_model,
        rag_min_score=rag_min_score,
        ollama_model=ollama_model,
        ollama_host=ollama_host,
    )


# ---------------------------------------------------------
# Default configuration
# ---------------------------------------------------------

def test_get_settings_uses_expected_defaults(
    monkeypatch
):
    remove_config_environment(
        monkeypatch
    )

    settings = get_settings()

    assert settings.pinecone_api_key is None

    assert settings.pinecone_index_name == (
        "visionaryq"
    )

    assert settings.whisper_model == (
        "small.en"
    )

    assert settings.rag_min_score == 0.30

    assert settings.ollama_model == (
        "llama3.2:3b"
    )

    assert settings.ollama_host == (
        "http://localhost:11434"
    )


# ---------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------

def test_get_settings_reads_environment_values(
    monkeypatch
):
    monkeypatch.setenv(
        "PINECONE_API_KEY",
        "unit-test-key",
    )

    monkeypatch.setenv(
        "PINECONE_INDEX_NAME",
        "unit-test-index",
    )

    monkeypatch.setenv(
        "WHISPER_MODEL",
        "base.en",
    )

    monkeypatch.setenv(
        "RAG_MIN_SCORE",
        "0.55",
    )

    monkeypatch.setenv(
        "OLLAMA_MODEL",
        "unit-test-model",
    )

    monkeypatch.setenv(
        "OLLAMA_HOST",
        "http://custom-ollama:11434"
    )

    settings = get_settings()

    assert settings.pinecone_api_key == (
        "unit-test-key"
    )

    assert settings.pinecone_index_name == (
        "unit-test-index"
    )

    assert settings.whisper_model == (
        "base.en"
    )

    assert settings.rag_min_score == 0.55

    assert settings.ollama_model == (
        "unit-test-model"
    )

    assert settings.ollama_host == (
        "http://custom-ollama:11434"
    )


# ---------------------------------------------------------
# Settings cache
# ---------------------------------------------------------

def test_get_settings_returns_cached_object(
    monkeypatch
):
    monkeypatch.setenv(
        "PINECONE_API_KEY",
        "cache-test-key",
    )

    first = get_settings()
    second = get_settings()

    assert first is second


# ---------------------------------------------------------
# RAG threshold validation
# ---------------------------------------------------------

def test_get_settings_rejects_non_numeric_rag_score(
    monkeypatch
):
    monkeypatch.setenv(
        "RAG_MIN_SCORE",
        "not-a-number",
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "RAG_MIN_SCORE must be "
            "a valid number"
        )
    ):
        get_settings()


@pytest.mark.parametrize(
    "invalid_score",
    [
        "-0.01",
        "1.01",
    ]
)
def test_get_settings_rejects_rag_score_outside_range(
    monkeypatch,
    invalid_score
):
    monkeypatch.setenv(
        "RAG_MIN_SCORE",
        invalid_score,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "RAG_MIN_SCORE must be "
            "between 0.0 and 1.0"
        )
    ):
        get_settings()


@pytest.mark.parametrize(
    "valid_score, expected",
    [
        ("0", 0.0),
        ("1", 1.0),
    ]
)
def test_get_settings_accepts_rag_score_boundaries(
    monkeypatch,
    valid_score,
    expected
):
    monkeypatch.setenv(
        "RAG_MIN_SCORE",
        valid_score,
    )

    settings = get_settings()

    assert settings.rag_min_score == expected


# ---------------------------------------------------------
# Pinecone validation
# ---------------------------------------------------------

def test_validate_pinecone_rejects_missing_api_key():
    settings = make_settings(
        pinecone_api_key=None
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "PINECONE_API_KEY is "
            "not configured"
        )
    ):
        settings.validate_pinecone()


def test_validate_pinecone_rejects_missing_index():
    settings = make_settings(
        pinecone_index_name=""
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "PINECONE_INDEX_NAME is "
            "not configured"
        )
    ):
        settings.validate_pinecone()


def test_validate_pinecone_accepts_valid_config():
    settings = make_settings()

    # Successful validation returns None
    # and raises no exception.
    result = settings.validate_pinecone()

    assert result is None
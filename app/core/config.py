import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


# Load local environment variables from .env
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """
    Central application configuration for VisionaryQ.

    Secrets are loaded from environment variables and are
    never hard-coded into the source code.
    """

    # Pinecone
    pinecone_api_key: str | None
    pinecone_index_name: str

    # Faster-Whisper
    whisper_model: str

    # RAG
    rag_min_score: float

    # Ollama
    ollama_model: str
    ollama_host: str

    def validate_pinecone(self) -> None:
        """
        Validate configuration required for Pinecone.
        """

        if not self.pinecone_api_key:
            raise RuntimeError(
                "PINECONE_API_KEY is not configured."
            )

        if not self.pinecone_index_name:
            raise RuntimeError(
                "PINECONE_INDEX_NAME is not configured."
            )


@lru_cache
def get_settings() -> Settings:
    """
    Load and cache VisionaryQ application settings.

    The settings object is created once per Python process.
    """

    try:
        rag_min_score = float(
            os.getenv(
                "RAG_MIN_SCORE",
                "0.30",
            )
        )

    except ValueError as exc:
        raise RuntimeError(
            "RAG_MIN_SCORE must be a valid number."
        ) from exc

    if not 0.0 <= rag_min_score <= 1.0:
        raise RuntimeError(
            "RAG_MIN_SCORE must be between 0.0 and 1.0."
        )

    return Settings(
        pinecone_api_key=os.getenv(
            "PINECONE_API_KEY"
        ),
        pinecone_index_name=os.getenv(
            "PINECONE_INDEX_NAME",
            "visionaryq",
        ),
        whisper_model=os.getenv(
            "WHISPER_MODEL",
            "small.en",
        ),
        rag_min_score=rag_min_score,
        ollama_model=os.getenv(
            "OLLAMA_MODEL",
            "llama3.2:3b",
        ),
        ollama_host=os.getenv(
            "OLLAMA_HOST",
            "http://localhost:11434",
        ),
    )
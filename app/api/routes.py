import os
import tempfile
from uuid import uuid4

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)
from pydantic import BaseModel, Field

from app.services.ingestion_service import IngestionService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.vector_service import VectorService


# ---------------------------------------------------------
# Router
# ---------------------------------------------------------

router = APIRouter(
    prefix="/api/v1",
    tags=["VisionaryQ"],
)


# ---------------------------------------------------------
# Supported video formats
# ---------------------------------------------------------

ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}


# ---------------------------------------------------------
# Lazy shared services
# ---------------------------------------------------------

_vector_service = None
_ingestion_service = None
_rag_service = None


def get_vector_service() -> VectorService:
    """
    Return one shared VectorService instance.

    This prevents Pinecone and the embedding model
    from being initialized again for every API request.
    """

    global _vector_service

    if _vector_service is None:
        _vector_service = VectorService()
        _vector_service.initialize()

    return _vector_service


def get_ingestion_service() -> IngestionService:
    """
    Return one shared IngestionService instance.
    """

    global _ingestion_service

    if _ingestion_service is None:
        _ingestion_service = IngestionService(
            vector_service=get_vector_service()
        )

    return _ingestion_service


def get_rag_service() -> RAGService:
    """
    Return one shared RAGService instance.
    """

    global _rag_service

    if _rag_service is None:
        _rag_service = RAGService(
            vector_service=get_vector_service(),
            llm_service=LLMService(),
        )

    return _rag_service


# ---------------------------------------------------------
# Request models
# ---------------------------------------------------------

class QueryRequest(BaseModel):
    video_id: str = Field(
        ...,
        min_length=1,
        description="Unique ID returned by the ingestion endpoint.",
    )

    query: str = Field(
        ...,
        min_length=1,
        description="Question to ask about the video.",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of retrieved chunks.",
    )


# ---------------------------------------------------------
# Status
# ---------------------------------------------------------

@router.get("/status")
def api_status():
    """
    API status endpoint.
    """

    return {
        "service": "VisionaryQ API",
        "status": "operational",
        "version": "1.0.0",
    }


# ---------------------------------------------------------
# Video ingestion
# ---------------------------------------------------------

@router.post("/video/ingest")
async def ingest_video(
    file: UploadFile = File(...)
):
    """
    Upload and index a video.

    Pipeline:

        Video upload
        -> Faster-Whisper
        -> Timestamp-aware chunks
        -> Embeddings
        -> Pinecone

    Returns a unique video_id that can later
    be supplied to /query.
    """

    filename = file.filename or ""

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded video must have a filename.",
        )

    extension = os.path.splitext(
        filename
    )[1].lower()

    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported video format: {extension}. "
                f"Allowed formats: "
                f"{sorted(ALLOWED_VIDEO_EXTENSIONS)}"
            ),
        )

    temporary_video_path = None

    # Every ingestion receives its own ID.
    video_id = str(
        uuid4()
    )

    try:

        # -------------------------------------------------
        # Save uploaded video temporarily
        # -------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temp_file:

            temporary_video_path = (
                temp_file.name
            )

            while True:

                data = await file.read(
                    1024 * 1024
                )

                if not data:
                    break

                temp_file.write(
                    data
                )

        # -------------------------------------------------
        # Run ingestion pipeline
        # -------------------------------------------------

        ingestion_service = (
            get_ingestion_service()
        )

        result = (
            ingestion_service.ingest_video(
                video_path=temporary_video_path,
                video_id=video_id,
                filename=filename,
            )
        )

        # -------------------------------------------------
        # Return concise API response
        # -------------------------------------------------

        return {
            "video_id": result["video_id"],
            "filename": result["filename"],
            "status": result["status"],
            "language": result.get(
                "language"
            ),
            "language_probability": result.get(
                "language_probability"
            ),
            "segment_count": result.get(
                "segment_count"
            ),
            "chunk_count": result[
                "chunk_count"
            ],
            "upserted_count": result[
                "upserted_count"
            ],
        }

    except HTTPException:
        raise

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "Video ingestion failed: "
                f"{str(e)}"
            ),
        )

    finally:

        # -------------------------------------------------
        # Delete temporary uploaded video
        # -------------------------------------------------

        if (
            temporary_video_path
            and os.path.exists(
                temporary_video_path
            )
        ):

            try:
                os.remove(
                    temporary_video_path
                )
            except OSError:
                pass

        await file.close()


# ---------------------------------------------------------
# RAG query
# ---------------------------------------------------------

@router.post("/query")
def query_video(
    request: QueryRequest
):
    """
    Ask a question about an indexed video.
    """

    try:

        rag_service = (
            get_rag_service()
        )

        result = rag_service.query(
            query=request.query,
            top_k=request.top_k,
            video_id=request.video_id,
        )

        return result

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "Query processing failed: "
                f"{str(e)}"
            ),
        )
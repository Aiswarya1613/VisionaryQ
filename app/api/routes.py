import os
import tempfile

from fastapi import APIRouter, File, UploadFile, HTTPException

from app.services.asr_service import video_to_text
from app.services.text_service import clean_text, chunk_text

from pydantic import BaseModel, Field

from app.services.vector_service import VectorService
from app.services.rag_service import RAGService
from app.services.llm_service import LLMService

class QueryRequest(BaseModel):
    video_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


router = APIRouter()


@router.get("/status")
def api_status():
    """
    API health/status endpoint.
    """
    return {
        "service": "VisionaryQ API",
        "status": "operational",
        "version": "1.0.0"
    }


@router.post("/video/process")
async def process_video(file: UploadFile = File(...)):
    """
    Process an uploaded video.

    Pipeline:
        Video
        -> Audio extraction
        -> Speech-to-text
        -> Text cleaning
        -> Text chunking
    """

    # Validate file type
    allowed_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm"
    }

    filename = file.filename or ""

    extension = os.path.splitext(filename)[1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported video format: {extension}. "
                f"Allowed formats: {sorted(allowed_extensions)}"
            )
        )

    temporary_video_path = None

    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temporary_video_path = temp_file.name

            # Read uploaded video in chunks
            while True:
                data = await file.read(1024 * 1024)

                if not data:
                    break

                temp_file.write(data)

        # -----------------------------
        # Step 1: Speech-to-text
        # -----------------------------

        raw_text = video_to_text(temporary_video_path)

        # Check whether ASR returned an error
        if raw_text.startswith("Error:"):
            raise HTTPException(
                status_code=500,
                detail=raw_text
            )

        # -----------------------------
        # Step 2: Clean text
        # -----------------------------

        cleaned_text = clean_text(raw_text)

        # -----------------------------
        # Step 3: Chunk text
        # -----------------------------

        chunks = chunk_text(
            cleaned_text,
            chunk_size=500,
            overlap=50
        )

        return {
            "filename": filename,
            "status": "success",
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "chunk_count": len(chunks),
            "chunks": chunks
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {str(e)}"
        )

    finally:

        # Delete temporary uploaded video
        if temporary_video_path and os.path.exists(
            temporary_video_path
        ):
            try:
                os.remove(temporary_video_path)
            except OSError:
                pass

        await file.close()

@router.post("/query")
def query_video(request: QueryRequest):
    """
    Query an indexed video using RAG.
    """

    try:
        vector_service = VectorService()
        vector_service.initialize()

        llm_service = LLMService()

        rag_service = RAGService(
            vector_service=vector_service,
            llm_service=llm_service
        )

        result = rag_service.query(
            query=request.query,
            top_k=request.top_k,
            video_id=request.video_id
        )

        return result

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )

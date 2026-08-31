from typing import Any, Dict

from app.services.asr_service import video_to_text
from app.services.text_service import clean_text, chunk_text
from app.services.vector_service import VectorService


class IngestionService:
    """
    Coordinates the complete video ingestion pipeline.

    Pipeline:
        Video
        -> ASR
        -> Text cleaning
        -> Text chunking
        -> Embeddings
        -> Pinecone
    """

    def __init__(
        self,
        vector_service: VectorService
    ):
        self.vector_service = vector_service

    def ingest_video(
        self,
        video_path: str,
        video_id: str,
        filename: str
    ) -> Dict[str, Any]:
        """
        Process and index a video.

        Parameters
        ----------
        video_path : str
            Path to the uploaded video.

        video_id : str
            Unique identifier for the video.

        filename : str
            Original filename.

        Returns
        -------
        Dict[str, Any]
            Ingestion result.
        """

        # Step 1: Transcribe video
        raw_text = video_to_text(video_path)

        if raw_text.startswith("Error:"):
            raise RuntimeError(raw_text)

        # Step 2: Clean transcript
        cleaned_text = clean_text(raw_text)

        if not cleaned_text:
            raise ValueError(
                "No transcribable text was found in the video."
            )

        # Step 3: Split transcript into chunks
        chunks = chunk_text(
            cleaned_text,
            chunk_size=500,
            overlap=50
        )

        if not chunks:
            raise ValueError(
                "No text chunks were generated."
            )

        # Step 4: Prepare Pinecone documents
        documents = []

        for index, chunk in enumerate(chunks):

            document = {
                "id": f"{video_id}-chunk-{index}",
                "text": chunk,
                "metadata": {
                    "video_id": video_id,
                    "filename": filename,
                    "chunk_index": index,
                    "text": chunk
                }
            }

            documents.append(document)

        # Step 5: Store vectors in Pinecone
        result = self.vector_service.add_documents(
            documents
        )

        return {
            "video_id": video_id,
            "filename": filename,
            "status": "success",
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "chunk_count": len(chunks),
            "upserted_count": result["upserted_count"]
        }
from typing import Any, Dict, List

from app.services.asr_service import transcribe_video
from app.services.text_service import clean_text
from app.services.vector_service import VectorService


class IngestionService:
    """
    Orchestrates the VisionaryQ video ingestion pipeline.

    Pipeline:
        Video
        -> Faster-Whisper transcription
        -> Timestamp-aware chunking
        -> Embeddings
        -> Pinecone
    """

    def __init__(
        self,
        vector_service: VectorService
    ):
        self.vector_service = vector_service

    @staticmethod
    def _build_timestamped_chunks(
        segments: List[Dict[str, Any]],
        target_chars: int = 500,
        overlap_segments: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Combine consecutive Whisper segments into larger
        chunks while preserving video timestamps.

        Parameters
        ----------
        segments:
            Timestamped Faster-Whisper segments.

        target_chars:
            Approximate maximum text size of each chunk.

        overlap_segments:
            Number of Whisper segments repeated between
            neighbouring chunks.
        """

        if target_chars <= 0:
            raise ValueError(
                "target_chars must be greater than 0."
            )

        if overlap_segments < 0:
            raise ValueError(
                "overlap_segments cannot be negative."
            )

        valid_segments = []

        # -------------------------------------------------
        # Normalize Whisper segments
        # -------------------------------------------------

        for segment in segments:

            text = clean_text(
                str(segment.get("text", ""))
            )

            if not text:
                continue

            valid_segments.append(
                {
                    "start": float(
                        segment.get("start", 0.0)
                    ),
                    "end": float(
                        segment.get("end", 0.0)
                    ),
                    "text": text
                }
            )

        if not valid_segments:
            return []

        # -------------------------------------------------
        # Build timestamp-aware chunks
        # -------------------------------------------------

        chunks = []

        start_index = 0

        while start_index < len(valid_segments):

            end_index = start_index

            text_parts = []

            character_count = 0

            while end_index < len(valid_segments):

                segment_text = (
                    valid_segments[end_index]["text"]
                )

                additional_length = (
                    len(segment_text)
                    + (1 if text_parts else 0)
                )

                # Keep at least one segment per chunk.
                if (
                    text_parts
                    and character_count
                    + additional_length
                    > target_chars
                ):
                    break

                text_parts.append(
                    segment_text
                )

                character_count += (
                    additional_length
                )

                end_index += 1

                if character_count >= target_chars:
                    break

            chunk_segments = (
                valid_segments[
                    start_index:end_index
                ]
            )

            chunk_text = clean_text(
                " ".join(text_parts)
            )

            chunks.append(
                {
                    "text": chunk_text,
                    "start_time": (
                        chunk_segments[0]["start"]
                    ),
                    "end_time": (
                        chunk_segments[-1]["end"]
                    )
                }
            )

            # Finished processing all segments.
            if end_index >= len(valid_segments):
                break

            # Preserve a small amount of semantic
            # overlap between neighbouring chunks.
            next_start = (
                end_index - overlap_segments
            )

            # Always guarantee forward progress.
            start_index = max(
                next_start,
                start_index + 1
            )

        return chunks

    def ingest_video(
        self,
        video_path: str,
        video_id: str,
        filename: str
    ) -> Dict[str, Any]:
        """
        Transcribe, chunk, embed, and index a video.
        """

        if not video_id or not video_id.strip():
            raise ValueError(
                "video_id cannot be empty."
            )

        if not filename or not filename.strip():
            raise ValueError(
                "filename cannot be empty."
            )

        # -------------------------------------------------
        # Step 1: Faster-Whisper transcription
        # -------------------------------------------------

        transcription = transcribe_video(
            video_path
        )

        raw_text = transcription.get(
            "text",
            ""
        ).strip()

        if not raw_text:
            raise ValueError(
                "No transcribable text was found "
                "in the video."
            )

        segments = transcription.get(
            "segments",
            []
        )

        if not segments:
            raise ValueError(
                "No timestamped transcription "
                "segments were generated."
            )

        # -------------------------------------------------
        # Step 2: Clean complete transcript
        # -------------------------------------------------

        cleaned_text = clean_text(
            raw_text
        )

        if not cleaned_text:
            raise ValueError(
                "Transcript is empty after cleaning."
            )

        # -------------------------------------------------
        # Step 3: Timestamp-aware chunking
        # -------------------------------------------------

        chunks = self._build_timestamped_chunks(
            segments=segments,
            target_chars=500,
            overlap_segments=1
        )

        if not chunks:
            raise ValueError(
                "No timestamped chunks were generated."
            )

        # -------------------------------------------------
        # Step 4: Prepare Pinecone documents
        # -------------------------------------------------

        documents = []

        for index, chunk in enumerate(chunks):

            documents.append(
                {
                    "id": (
                        f"{video_id}-chunk-{index}"
                    ),
                    "text": chunk["text"],
                    "metadata": {
                        "video_id": video_id,
                        "filename": filename,
                        "chunk_index": index,
                        "start_time": (
                            chunk["start_time"]
                        ),
                        "end_time": (
                            chunk["end_time"]
                        )
                    }
                }
            )

        # -------------------------------------------------
        # Step 5: Embed + store in Pinecone
        # -------------------------------------------------

        result = (
            self.vector_service
            .add_documents(documents)
        )

        return {
            "video_id": video_id,
            "filename": filename,
            "status": "success",
            "language": transcription.get(
                "language"
            ),
            "language_probability": (
                transcription.get(
                    "language_probability"
                )
            ),
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "segment_count": len(segments),
            "chunk_count": len(chunks),
            "upserted_count": result[
                "upserted_count"
            ]
        }
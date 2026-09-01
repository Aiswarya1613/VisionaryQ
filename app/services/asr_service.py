import os
from typing import Any, Dict, List

from faster_whisper import WhisperModel

from app.core.config import get_settings


# ---------------------------------------------------------
# Whisper configuration
# ---------------------------------------------------------

_whisper_model = None


def get_whisper_model() -> WhisperModel:
    """
    Lazily load and cache the Faster-Whisper model.
    """

    global _whisper_model

    if _whisper_model is None:

        settings = get_settings()

        print(
            f"Loading Faster-Whisper model: "
            f"{settings.whisper_model}"
        )

        _whisper_model = WhisperModel(
            settings.whisper_model,
            device="cpu",
            compute_type="int8"
        )

        print(
            "Faster-Whisper model loaded successfully."
        )

    return _whisper_model


def transcribe_video(
    video_path: str
) -> Dict[str, Any]:
    """
    Transcribe a video using Faster-Whisper.

    Returns both the complete transcript and
    timestamped transcription segments.
    """

    if not os.path.isfile(video_path):
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    model = get_whisper_model()

    print(
        f"Transcribing video: {video_path}"
    )

    segments, info = model.transcribe(
        video_path,
        language="en",
        beam_size=5,
        vad_filter=True
    )

    transcript_parts: List[str] = []
    segment_data: List[Dict[str, Any]] = []

    for segment in segments:

        text = segment.text.strip()

        if not text:
            continue

        transcript_parts.append(text)

        segment_data.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text
            }
        )

        print(
            f"[{segment.start:.2f}s -> "
            f"{segment.end:.2f}s] "
            f"{text}"
        )

    transcript = " ".join(
        transcript_parts
    ).strip()

    return {
        "text": transcript,
        "segments": segment_data,
        "language": info.language,
        "language_probability": float(
            info.language_probability
        )
    }


def video_to_text(
    video_path: str,
    chunk_length: int = 20,
    ffmpeg_path: str | None = None
) -> str:
    """
    Compatibility wrapper used by the existing
    VisionaryQ ingestion pipeline.

    chunk_length and ffmpeg_path are retained so
    existing callers do not break, but Faster-Whisper
    handles segmentation and media decoding internally.
    """

    try:

        result = transcribe_video(
            video_path
        )

        return result["text"]

    except Exception as e:

        print(
            f"ASR error: {e}"
        )

        return ""
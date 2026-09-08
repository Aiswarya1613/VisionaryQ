import re


def clean_text(text: str) -> str:
    """
    Clean transcribed text.

    Parameters
    ----------
    text : str
        Raw transcription text.

    Returns
    -------
    str
        Cleaned text.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def normalize_text(text: str) -> str:
    """
    Normalize text for downstream processing.
    """

    text = clean_text(text)

    # Normalize repeated punctuation
    text = re.sub(r"[.]{2,}", ".", text)
    text = re.sub(r"[,]{2,}", ",", text)

    return text


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> list[str]:
    """
    Split text into overlapping chunks without splitting words.

    Parameters
    ----------
    text : str
        Input text.

    chunk_size : int
        Maximum approximate number of characters per chunk.

    overlap : int
        Approximate number of overlapping characters.

    Returns
    -------
    list[str]
        List of text chunks.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero"
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative"
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    text = normalize_text(text)

    if not text:
        return []

    words = text.split()

    chunks = []
    current_words = []
    current_length = 0

    for word in words:

        word_length = len(word)

        # Account for spaces between words
        additional_length = (
            word_length
            if not current_words
            else word_length + 1
        )

        # If adding the word exceeds the target size,
        # finalize the current chunk.
        if (
            current_words
            and current_length + additional_length > chunk_size
        ):
            chunk = " ".join(current_words)
            chunks.append(chunk)

            # Build overlap using words from the end
            overlap_words = []
            overlap_length = 0

            for previous_word in reversed(current_words):

                additional_overlap = (
                    len(previous_word)
                    if not overlap_words
                    else len(previous_word) + 1
                )

                if (
                    overlap_length + additional_overlap
                    > overlap
                ):
                    break

                overlap_words.insert(
                    0,
                    previous_word
                )

                overlap_length += additional_overlap

            current_words = overlap_words
            current_length = len(
                " ".join(current_words)
            )

        current_words.append(word)

        current_length = len(
            " ".join(current_words)
        )

    # Add remaining words
    if current_words:
        chunks.append(
            " ".join(current_words)
        )

    return chunks
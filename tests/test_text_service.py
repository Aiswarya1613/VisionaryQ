import pytest

from app.services.text_service import (
    clean_text,
    normalize_text,
    chunk_text,
)


# ---------------------------------------------------------
# clean_text
# ---------------------------------------------------------

def test_clean_text_normalizes_whitespace():
    text = "  Hello     world \n this\tis VisionaryQ  "

    result = clean_text(text)

    assert result == "Hello world this is VisionaryQ"


def test_clean_text_empty_string():
    assert clean_text("") == ""


def test_clean_text_rejects_non_string():
    with pytest.raises(
        TypeError,
        match="text must be a string"
    ):
        clean_text(None)


# ---------------------------------------------------------
# normalize_text
# ---------------------------------------------------------

def test_normalize_text_repeated_punctuation():
    text = "Hello... world,,, this is VisionaryQ...."

    result = normalize_text(text)

    assert result == "Hello. world, this is VisionaryQ."


def test_normalize_text_also_normalizes_whitespace():
    text = "  Generative     AI...   creates   content. "

    result = normalize_text(text)

    assert result == "Generative AI. creates content."


# ---------------------------------------------------------
# chunk_text
# ---------------------------------------------------------

def test_chunk_text_empty_text_returns_empty_list():
    result = chunk_text(
        "",
        chunk_size=100,
        overlap=20
    )

    assert result == []


def test_chunk_text_short_text_returns_one_chunk():
    text = "Generative AI creates new content."

    result = chunk_text(
        text,
        chunk_size=100,
        overlap=20
    )

    assert result == [
        "Generative AI creates new content."
    ]


def test_chunk_text_creates_overlapping_chunks():
    text = (
        "one two three four five six seven eight "
        "nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen"
    )

    result = chunk_text(
        text,
        chunk_size=40,
        overlap=10
    )

    assert result == [
        "one two three four five six seven eight",
        "eight nine ten eleven twelve thirteen",
        "thirteen fourteen fifteen sixteen",
    ]


def test_chunk_text_preserves_complete_words():
    text = (
        "Generative artificial intelligence creates "
        "new original digital content from learned patterns"
    )

    result = chunk_text(
        text,
        chunk_size=35,
        overlap=8
    )

    original_words = set(
        normalize_text(text).split()
    )

    for chunk in result:
        for word in chunk.split():
            assert word in original_words


def test_chunk_text_does_not_split_long_word():
    text = "supercalifragilisticexpialidocious"

    result = chunk_text(
        text,
        chunk_size=10,
        overlap=2
    )

    assert result == [
        "supercalifragilisticexpialidocious"
    ]


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "chunk_size",
    [0, -1, -100]
)
def test_chunk_text_rejects_invalid_chunk_size(
    chunk_size
):
    with pytest.raises(
        ValueError,
        match="chunk_size must be greater than zero"
    ):
        chunk_text(
            "some text",
            chunk_size=chunk_size,
            overlap=0
        )


def test_chunk_text_rejects_negative_overlap():
    with pytest.raises(
        ValueError,
        match="overlap cannot be negative"
    ):
        chunk_text(
            "some text",
            chunk_size=100,
            overlap=-1
        )


def test_chunk_text_rejects_overlap_equal_to_chunk_size():
    with pytest.raises(
        ValueError,
        match="overlap must be smaller than chunk_size"
    ):
        chunk_text(
            "some text",
            chunk_size=100,
            overlap=100
        )


def test_chunk_text_rejects_overlap_larger_than_chunk_size():
    with pytest.raises(
        ValueError,
        match="overlap must be smaller than chunk_size"
    ):
        chunk_text(
            "some text",
            chunk_size=100,
            overlap=101
        )


def test_chunk_text_rejects_non_string():
    with pytest.raises(
        TypeError,
        match="text must be a string"
    ):
        chunk_text(
            None,
            chunk_size=100,
            overlap=20
        )
import pytest

from src.data.captions import (
    clean_caption,
    preprocess_captions,
    validate_captions
)


def test_clean_caption():

    caption = "   this   bird is   brown.   "

    result = clean_caption(caption)

    assert result == "this bird is brown."


def test_preprocess_captions():

    description = (
        "a brown bird.\n"
        "a bird with white wings.\n"
        "\n"
        "a small bird with a black beak."
    )

    result = preprocess_captions(description)

    assert len(result) == 3

    assert result[0] == "a brown bird."

    assert result[1] == "a bird with white wings."

    assert result[2] == "a small bird with a black beak."


def test_validate_captions():

    captions = [
        f"bird description number {i}."
        for i in range(10)
    ]

    assert validate_captions(captions) is True


def test_invalid_caption_count():

    captions = [
        "a brown bird.",
        "a white bird."
    ]

    assert validate_captions(captions) is False


def test_empty_caption():

    captions = ["a bird."] * 9 + [""]

    assert validate_captions(captions) is False


def test_invalid_description_type():

    with pytest.raises(TypeError):
        preprocess_captions(None)
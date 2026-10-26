import pytest

from src.data.captions import (
    MAX_LEN,
    MIN_FREQ,
    MIN_TOKENS,
    SPECIAL_TOKENS,
    build_vocabulary,
    clean_caption,
    clean_captions,
    preprocess_captions,
    validate_captions
)
from tests.conftest import DESCRIPTION


def test_clean_caption():

    caption = "   this   bird is   brown.   "

    result = clean_caption(caption)

    assert result == "this bird is brown"


def test_clean_caption_rules():

    caption = "This Bird's wing-tips are GREY, white & 2-toned!"

    assert clean_caption(caption) == (
        "this birds wing tips are grey white 2 toned"
    )


def test_clean_caption_repairs_replacement_characters():

    caption = "this��bird��has��red��wings"

    assert clean_caption(caption) == "this bird has red wings"


def test_clean_captions_drops_short_and_repeated():

    result = clean_captions(DESCRIPTION)

    # "this" is too short; the second caption repeats the first after cleaning
    assert result == [
        "this bird has a red crown and a white belly",
        "a small bird with brown wings",
        "the birds wing tips are grey",
    ]


def test_preprocess_captions_keeps_raw_text():

    result = preprocess_captions(DESCRIPTION)

    assert len(result) == 5
    assert result[0] == "This bird has a RED crown, and a white belly."
    assert result[2] == "this"


def test_caption_constants():

    assert MIN_TOKENS == 4
    assert MAX_LEN == 26
    assert MIN_FREQ == 5


def test_build_vocabulary():

    captions = ["red bird red wings", "red bird blue wings", "a bird"]

    vocab = build_vocabulary(captions, min_freq=2)

    assert vocab[:len(SPECIAL_TOKENS)] == list(SPECIAL_TOKENS)

    # By decreasing frequency, ties alphabetical; "blue" and "a" are rare
    assert vocab[len(SPECIAL_TOKENS):] == ["bird", "red", "wings"]


def test_build_vocabulary_invalid_min_freq():

    with pytest.raises(ValueError):
        build_vocabulary(["a bird"], min_freq=0)


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
import re
from typing import List


def clean_caption(caption: str) -> str:
    """
    Clean one caption while preserving useful information.

    Parameters
    ----------
    caption : str
        Original caption.

    Returns
    -------
    str
        Cleaned caption.
    """

    if not isinstance(caption, str):
        raise TypeError("Caption must be a string.")

    # Remove leading and trailing whitespace
    caption = caption.strip()

    # Replace repeated whitespace with one space
    caption = re.sub(r"\s+", " ", caption)

    return caption


def preprocess_captions(description: str) -> List[str]:
    """
    Split and clean the description column.

    Each CUB image is expected to have 10 captions,
    separated by newline characters.

    Parameters
    ----------
    description : str
        Original description containing captions.

    Returns
    -------
    List[str]
        List of cleaned, non-empty captions.
    """

    if not isinstance(description, str):
        raise TypeError("Description must be a string.")

    # Split description into separate captions
    raw_captions = description.splitlines()

    cleaned_captions = []

    for caption in raw_captions:

        cleaned = clean_caption(caption)

        # Ignore empty captions
        if cleaned:
            cleaned_captions.append(cleaned)

    return cleaned_captions


def validate_captions(
    captions: List[str],
    expected_count: int = 10
) -> bool:
    """
    Validate caption structure.

    Returns True if captions satisfy the expected format.
    """

    if not isinstance(captions, list):
        return False

    if len(captions) != expected_count:
        return False

    for caption in captions:

        if not isinstance(caption, str):
            return False

        if not caption.strip():
            return False

    return True
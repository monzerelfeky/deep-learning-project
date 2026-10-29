import re
from collections import Counter
from typing import Iterable, List

# Caption decisions from docs/eda_findings.md, section 5
MIN_TOKENS = 4  # "this", "the colorful bird" and similar fragments are dropped
MAX_LEN = 26  # 95th percentile of train caption length (24) + <start> and <end>
MIN_FREQ = 5
SPECIAL_TOKENS = ("<pad>", "<start>", "<end>", "<unk>")


def preprocess_captions(description: str) -> List[str]:
    """
    Split the description column into its raw captions.

    Each CUB image is expected to have 10 captions,
    separated by newline characters. Punctuation and
    casing are kept; use clean_captions() for the
    model-ready version.

    Parameters
    ----------
    description : str
        Original description containing captions.

    Returns
    -------
    List[str]
        List of stripped, non-empty captions.
    """

    if not isinstance(description, str):
        raise TypeError("Description must be a string.")

    captions = []

    for line in description.split("\n"):

        # Strip and replace repeated whitespace with one space
        line = re.sub(r"\s+", " ", line).strip()

        # Ignore empty lines (every description ends with "\n")
        if line:
            captions.append(line)

    return captions


def clean_caption(caption: str) -> str:
    """
    Normalize one caption for vocabulary building and training.

    Used by both the EDA and the pipeline.

    Parameters
    ----------
    caption : str
        One raw caption.

    Returns
    -------
    str
        Lowercase words separated by single spaces.
    """

    if not isinstance(caption, str):
        raise TypeError("Caption must be a string.")

    # 56 captions have every space replaced by a pair of U+FFFD characters
    caption = caption.replace("�", " ")

    caption = caption.lower()

    # bird's -> birds
    caption = re.sub(r"[’']", "", caption)

    # Punctuation and other symbols -> space
    caption = re.sub(r"[^a-z0-9 ]+", " ", caption)

    return re.sub(r"\s+", " ", caption).strip()


def clean_captions(
    description: str,
    min_tokens: int = MIN_TOKENS
) -> List[str]:
    """
    Model-ready captions of one image.

    Raw captions are cleaned, captions shorter than
    min_tokens are dropped, and exact repeats within
    the image are dropped (the first copy is kept).

    Parameters
    ----------
    description : str
        Original description containing captions.

    min_tokens : int
        Minimum number of words a caption must have.

    Returns
    -------
    List[str]
        Cleaned captions in their original order.
    """

    cleaned_captions = []

    for caption in preprocess_captions(description):

        cleaned = clean_caption(caption)

        if len(cleaned.split()) < min_tokens:
            continue

        if cleaned not in cleaned_captions:
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


def build_vocabulary(
    captions: Iterable[str],
    min_freq: int = MIN_FREQ
) -> List[str]:
    """
    Build the caption vocabulary.

    Pass captions of the training partition only, never
    validation or test captions.

    Parameters
    ----------
    captions : Iterable[str]
        Cleaned captions.

    min_freq : int
        Words seen fewer times become <unk>.

    Returns
    -------
    List[str]
        Index-to-word list: the special tokens first,
        then words by decreasing frequency.
    """

    if min_freq < 1:
        raise ValueError("min_freq must be at least 1.")

    counts = Counter(
        word
        for caption in captions
        for word in caption.split()
    )

    words = sorted(
        (word for word, count in counts.items()
         if count >= min_freq and word not in SPECIAL_TOKENS),
        key=lambda word: (-counts[word], word)
    )

    return list(SPECIAL_TOKENS) + words

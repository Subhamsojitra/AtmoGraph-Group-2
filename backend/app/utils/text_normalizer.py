"""Reusable text normalization utilities for AtmoGraph.

This module provides pure-text cleaning helpers used by the ingestion
pipeline. It does not perform NER, entity resolution, or any advanced NLP
processing; those belong to later modules.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normalize raw article text into a clean canonical form.

    Operations performed:
    - Unicode NFC normalization
    - Strip leading/trailing whitespace
    - Collapse repeated whitespace (including newlines) into single spaces
    - Remove carriage-return characters
    - Remove null bytes

    This function does NOT:
    - remove punctuation
    - perform stemming or lemmatization
    - strip characters that may be useful for NLP

    Args:
        text: Raw input text.

    Returns:
        Normalized text string.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # Unicode normalization (NFC)
    text = unicodedata.normalize("NFC", text)

    # Remove null bytes and carriage returns
    text = text.replace("\x00", "").replace("\r", "")

    # Collapse all whitespace (including newlines, tabs) into single spaces
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def compute_content_hash(text: str) -> str:
    """Compute a stable SHA-256 hash for a normalized text.

    Args:
        text: Normalized text to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest string.
    """
    import hashlib

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return hashlib.sha256(text.encode("utf-8")).hexdigest()

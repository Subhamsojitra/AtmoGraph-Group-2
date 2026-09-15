"""NLP text preprocessing for AtmoGraph (Module 7).

This module prepares raw text for the NER model. It deliberately **reuses**
the Module 6 normalization implementation (``app.utils.text_normalizer``)
instead of duplicating it, and adds only the NER-specific input guards:

- input type validation
- empty / whitespace-only rejection
- maximum text length enforcement

It does NOT tokenize, run NER, or perform any downstream analysis.
"""

from __future__ import annotations

from app.services.nlp.exceptions import TextPreprocessingError
from app.utils.text_normalizer import normalize_text


class TextPreprocessor:
    """Preprocesses raw text into a clean, NER-ready canonical form.

    The preprocessor is intentionally small and stateless (aside from its
    configured length limit). All heavy normalization is delegated to the
    shared Module 6 :func:`~app.utils.text_normalizer.normalize_text`.
    """

    def __init__(self, max_length: int) -> None:
        """Initialize the preprocessor with a maximum text length.

        Args:
            max_length: Maximum number of characters (after normalization) the
                NER pipeline will accept. Must be a positive integer.

        Raises:
            ValueError: If ``max_length`` is not a positive integer.
        """
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        self._max_length = max_length

    @property
    def max_length(self) -> int:
        """Maximum accepted character length (after normalization)."""
        return self._max_length

    def preprocess(self, text: str) -> str:
        """Normalize and validate text for NER processing.

        Args:
            text: Raw input text.

        Returns:
            Normalized text string (Unicode NFC, collapsed whitespace, no null
            bytes, stripped).

        Raises:
            TextPreprocessingError: If ``text`` is not a string, is empty or
                whitespace-only after normalization, or exceeds ``max_length``.
        """
        if not isinstance(text, str):
            raise TextPreprocessingError("text must be a string")

        # Reuse the shared Module 6 normalization (no duplicated logic).
        normalized = normalize_text(text)

        if not normalized:
            raise TextPreprocessingError("text must not be empty or whitespace-only")

        if len(normalized) > self._max_length:
            raise TextPreprocessingError(
                f"text exceeds the maximum allowed length of {self._max_length} characters"
            )

        return normalized

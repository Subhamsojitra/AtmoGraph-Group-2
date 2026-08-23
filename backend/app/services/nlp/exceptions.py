"""Domain exceptions for the NLP preprocessing & NER pipeline (Module 7).

The API layer maps these exceptions to HTTP status codes:

- ``TextPreprocessingError`` -> 422 (invalid input)
- ``ModelUnavailableError``  -> 503 (NLP model unavailable)
- ``NERProcessingError``     -> 500 (unexpected processing failure)
"""


class NERError(Exception):
    """Base class for all NER pipeline errors."""


class ModelUnavailableError(NERError):
    """Raised when the configured NER model cannot be loaded or is unavailable."""


class NERProcessingError(NERError):
    """Raised when the loaded NER model fails during inference."""


class TextPreprocessingError(NERError):
    """Raised when input text cannot be preprocessed.

    Covers non-string input, empty/whitespace-only text, and text exceeding
    the configured maximum length.
    """

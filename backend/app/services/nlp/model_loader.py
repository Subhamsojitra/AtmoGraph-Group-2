"""Reusable NER model loader for AtmoGraph (Module 7).

The :class:`NERModelManager` loads the configured Hugging Face Transformers
``token-classification`` pipeline **exactly once** and reuses it for every
subsequent request. Loading is:

* **lazy** -- the model is loaded on first use, never at application startup,
  so FastAPI keeps working even when the model package is not yet installed;
* **cached** -- the pipeline object lives for the lifetime of the process;
* **thread-safe** -- a lock guards the one-time lazy load, which is relevant
  because FastAPI runs synchronous endpoints in a thread pool.

No model paths are hardcoded here: the model name, device, and tokenizer
limits come from the centralized :mod:`app.core.config` settings.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from transformers import AutoTokenizer, pipeline

from app.core.config import settings
from app.core.logger import get_logger
from app.services.nlp.exceptions import ModelUnavailableError

logger = get_logger(__name__)


class NERModelManager:
    """Thread-safe, lazily-loaded wrapper around a Transformers NER pipeline."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: Optional[int] = None,
        use_fast_tokenizer: Optional[bool] = None,
    ) -> None:
        """Configure the manager from settings (or explicit overrides).

        Args:
            model_name: HuggingFace model identifier. Defaults to the
                ``NLP_MODEL_NAME`` setting.
            device: torch device string (e.g. ``"cpu"``, ``"cuda"``).
                Defaults to the ``NLP_DEVICE`` setting.
            max_length: Maximum number of tokens to keep per input.
                Defaults to the ``NLP_NER_MAX_LENGTH`` setting.
            use_fast_tokenizer: Whether to prefer the fast tokenizer.
                Defaults to the ``NLP_USE_FAST_TOKENIZER`` setting.
        """
        self._model_name = model_name or settings.nlp_model_name
        self._device = device or settings.nlp_device
        self._max_length = max_length or settings.nlp_ner_max_length
        if use_fast_tokenizer is None:
            use_fast_tokenizer = settings.nlp_use_fast_tokenizer
        self._use_fast_tokenizer = use_fast_tokenizer

        self._pipeline: Optional[Any] = None
        self._load_attempted = False
        self._load_error: Optional[ModelUnavailableError] = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Return ``True`` when the model pipeline has been loaded."""
        return self._pipeline is not None

    @property
    def model_name(self) -> str:
        """The configured HuggingFace model identifier."""
        return self._model_name

    def get_pipeline(self) -> Any:
        """Return the loaded NER pipeline, loading it on first use.

        The pipeline is constructed only once; subsequent calls return the
        cached instance. A failed load attempt is remembered and re-raised so
        the model is not repeatedly retried per request.

        Returns:
            The Transformers ``token-classification`` pipeline with
            ``aggregation_strategy="simple"``.

        Raises:
            ModelUnavailableError: If the model cannot be loaded (missing
                package, offline cache miss, OOM, etc.).
        """
        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            if self._load_attempted:
                # A previous load failed; fail fast instead of retrying the
                # expensive load for every request.
                assert self._load_error is not None
                raise self._load_error

            self._load_attempted = True
            try:
                logger.info(
                    "Loading NER model",
                    extra={"model_name": self._model_name, "device": self._device},
                )
                # Transformers v5 handles truncation automatically when the
                # tokenizer has a ``model_max_length``; the value is applied
                # here so the configured ``NLP_NER_MAX_LENGTH`` is honored.
                tokenizer = AutoTokenizer.from_pretrained(
                    self._model_name,
                    use_fast=self._use_fast_tokenizer,
                    model_max_length=self._max_length,
                )
                self._pipeline = pipeline(
                    task="token-classification",
                    model=self._model_name,
                    tokenizer=tokenizer,
                    device=self._device,
                    aggregation_strategy="simple",
                )
                logger.info(
                    "NER model loaded successfully",
                    extra={"model_name": self._model_name},
                )
            except Exception as exc:
                self._load_error = ModelUnavailableError(
                    f"Unable to load NLP model '{self._model_name}'. "
                    "Check that the model package is installed (see README)."
                )
                logger.error(
                    "NER model loading failed",
                    extra={"model_name": self._model_name},
                    exc_info=True,
                )
                raise self._load_error from exc

            return self._pipeline

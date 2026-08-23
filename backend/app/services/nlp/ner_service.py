"""Named Entity Recognition service for AtmoGraph (Module 7).

This service runs the configured NER model over normalized article text and
returns structured, occurrence-level entity results.

Architecture:

    FastAPI / caller
        ↓
    NERService (this module)
        ↓
    TextPreprocessor (reuses Module 6 normalization)
        ↓
    NERModelManager (lazily loaded, cached Transformers pipeline)
        ↓
    Hugging Face Transformers model

Boundaries (Module 7 scope):

* The service does NOT access Neo4j, update graph nodes, calculate risk, or
  create relationships. Those belong to Modules 8/9.
* The service does NOT map entity text to supply-chain concepts (ports,
  suppliers, warehouses). That is Module 8 entity resolution.
* The service only performs NER; no sentiment/classification/summarization.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from app.core.config import settings
from app.core.logger import get_logger
from app.schemas.ner import NERAnalysisRequest, NERAnalysisResponse, NEREntity
from app.services.nlp.exceptions import (
    ModelUnavailableError,
    NERProcessingError,
    TextPreprocessingError,
)
from app.services.nlp.model_loader import NERModelManager
from app.services.nlp.preprocessor import TextPreprocessor

logger = get_logger(__name__)


class NERService:
    """Runs Named Entity Recognition over normalized article text.

    The heavy NLP model is owned by :class:`NERModelManager`, which loads it
    exactly once (lazily) and reuses it across requests. The service itself
    holds no business data and performs no persistence.
    """

    def __init__(
        self,
        model_manager: Optional[NERModelManager] = None,
        preprocessor: Optional[TextPreprocessor] = None,
        max_text_length: Optional[int] = None,
    ) -> None:
        """Initialize the NER service with injectable dependencies.

        Args:
            model_manager: Model container. Defaults to a manager configured
                from :mod:`app.core.config` settings.
            preprocessor: Text preprocessor. Defaults to one built from the
                configured ``NLP_MAX_TEXT_LENGTH``.
            max_text_length: Convenience override for the preprocessor's
                maximum accepted character length (ignored when
                ``preprocessor`` is provided).
        """
        self._model_manager = model_manager or NERModelManager()
        if preprocessor is None:
            length = max_text_length or settings.nlp_max_text_length
            preprocessor = TextPreprocessor(max_length=length)
        self._preprocessor = preprocessor

    @property
    def model_loaded(self) -> bool:
        """Return ``True`` once the underlying NER model is loaded."""
        return self._model_manager.is_loaded

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def analyze(self, request: NERAnalysisRequest) -> NERAnalysisResponse:
        """Run the NER pipeline over an article and return structured results.

        Args:
            request: Validated request containing title, content, and an
                optional Module 6 ``article_id``.

        Returns:
            A ``NERAnalysisResponse`` with occurrence-level entities and their
            character offsets into the analyzed text.

        Raises:
            ValueError: If ``request`` is not a ``NERAnalysisRequest``.
            TextPreprocessingError: If the input is empty/whitespace-only or
                exceeds the configured maximum length.
            ModelUnavailableError: If the NER model cannot be loaded.
            NERProcessingError: If the model fails during inference.
        """
        if not isinstance(request, NERAnalysisRequest):
            raise ValueError("request must be a NERAnalysisRequest instance")

        normalized_title = self._preprocessor.preprocess(request.title)
        normalized_content = self._preprocessor.preprocess(request.content)

        analyzed_text = self._combine_text(normalized_title, normalized_content)

        entities = self.extract_entities(analyzed_text)

        article_id = self._resolve_article_id(request.article_id)

        logger.info(
            "NER analysis completed",
            extra={"article_id": article_id, "entity_count": len(entities)},
        )

        return NERAnalysisResponse(
            article_id=article_id,
            analyzed_text=analyzed_text,
            entities=entities,
            entity_count=len(entities),
        )

    def extract_entities(self, text: str) -> list[NEREntity]:
        """Extract named entities from a single text.

        Occurrences are preserved: if the same entity surface text appears
        multiple times, each occurrence is returned as its own
        :class:`NEREntity` with distinct offsets.

        Args:
            text: Raw text to analyze. It is preprocessed (normalized) before
                the model runs.

        Returns:
            A list of ``NEREntity`` objects (may be empty).

        Raises:
            TextPreprocessingError, ModelUnavailableError, NERProcessingError.
        """
        normalized = self._preprocessor.preprocess(text)

        # May raise ModelUnavailableError; load errors are cached & re-raised.
        ner_pipeline = self._model_manager.get_pipeline()

        try:
            raw_entities = ner_pipeline(normalized)
        except ModelUnavailableError:
            raise
        except Exception as exc:
            logger.error("NER model inference failed", exc_info=True)
            raise NERProcessingError("NER model inference failed") from exc

        return [self._to_entity(item) for item in raw_entities]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _combine_text(title: str, content: str) -> str:
        """Combine normalized title and content into one analyzed text.

        The title is followed by a period and the body, e.g.
        ``"Port strike... shipments. Workers at the Port of Rotterdam..."``.
        Character offsets in the response refer to this combined string.
        """
        return f"{title}. {content}"

    def _resolve_article_id(self, article_id: Optional[str]) -> str:
        """Use the caller-supplied Module 6 article id or generate a UUIDv4.

        The generation mechanism matches Module 6 ingestion exactly. No state
        is required because Module 7 persists nothing.
        """
        if article_id:
            return article_id
        return str(uuid.uuid4())

    @staticmethod
    def _to_entity(item: dict[str, Any]) -> NEREntity:
        """Map one raw pipeline span to a structured ``NEREntity``.

        Transformers ``token-classification`` with
        ``aggregation_strategy="simple"`` returns per-span dicts with
        ``entity_group`` (or ``entity`` on older versions), ``score``,
        ``word``, ``start`` and ``end``. Both key spellings are handled for
        robustness across transformers releases.
        """
        label = item.get("entity_group") or item.get("entity") or "UNKNOWN"
        start = int(item.get("start", 0))
        end = int(item.get("end", 0))
        score = item.get("score")
        # ``score`` is a numpy scalar (e.g. np.float32) in transformers v5 and
        # must be converted to a native float so the Pydantic response stays
        # JSON serializable (json.dumps cannot handle numpy scalars).
        confidence = float(score) if score is not None else None
        return NEREntity(
            text=str(item.get("word", "")).strip(),
            label=str(label),
            start=max(0, start),
            end=max(0, end),
            confidence=confidence,
        )

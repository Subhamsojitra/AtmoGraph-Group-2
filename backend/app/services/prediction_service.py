"""Prediction service for AtmoGraph (Module 14).

Serves node-level downstream-delay predictions from the trained GNN over the
current supply-chain graph. This service is the thin, testable seam between
the HTTP layer (:mod:`app.api.prediction`) and the ML layers, which are
reused unchanged::

    Prediction API (Module 14 route)
        |
    PredictionService (this module)
        |   lazy, REUSABLE GNNPredictor (Module 14, model loaded once)
        v
    GraphDatasetBuilder (Module 11, existing)  ->  GraphDataset
        |
    GNNPredictor.predict (Module 14)  ->  GNNPredictionResult
        |
    PredictionResponse (Pydantic schema)

Behaviour:
* The trained model is loaded AT MOST ONCE per service instance (lazily, on
  the first prediction) and reused across requests; ``reload_model()``
  clears it when a new checkpoint is deployed. The checkpoint itself is
  trained offline by Module 13 and never created, downloaded or committed
  here.
* The graph dataset is rebuilt per request via the EXISTING Module 11
  builder (fresh data, parameterized Cypher, no new repository/driver).
* When no checkpoint is configured the service raises
  :class:`ModelNotAvailableError`; the API layer maps it to HTTP 503. The
  prediction capability is deployment state — it is never faked.
* No filesystem paths, credentials or stack traces are included in errors
  that reach clients (detailed context goes to the logs only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from app.core.config import BACKEND_DIR, settings
from app.core.logger import get_logger
from app.ml.dataset import GraphDataset, GraphDatasetBuilder
from app.ml.prediction import GNNPredictor
from app.schemas.prediction import (
    NodePrediction,
    PredictionRequest,
    PredictionResponse,
)

logger = get_logger(__name__)

__all__ = ["ModelNotAvailableError", "PredictionService"]


def _resolve_checkpoint_path(raw: Union[str, Path]) -> Path:
    """Turn a configured checkpoint location into an absolute path.

    Absolute paths are used unchanged. Relative paths are resolved against
    the backend directory (the same ``__file__``-derived convention as
    :mod:`app.core.config`), so a setting like
    ``PREDICTION_CHECKPOINT_PATH=checkpoints/gnn_m13.pt`` works no matter
    which working directory the server was launched from.
    """
    path = Path(raw)
    if path.is_absolute():
        return path
    return BACKEND_DIR / path


class ModelNotAvailableError(Exception):
    """Raised when no trained GNN checkpoint is configured for serving.

    This is a controlled deployment state (no checkpoint has been trained /
    deployed yet), not a crash: the API maps it to HTTP 503 with a generic
    message.
    """


class PredictionService:
    """Application-level GNN prediction operations (Module 14).

    All external callers (the API layer) should use this service instead of
    touching :class:`~app.ml.prediction.GNNPredictor` or the Module 11
    builder directly.

    Args:
        checkpoint_path: Optional explicit checkpoint path override (mainly
            for tests). When omitted, ``settings.prediction_checkpoint_path``
            is used.
        device: Optional explicit torch device override (mainly for tests).
            When omitted, ``settings.prediction_device`` is used.
        dataset_builder: Optional Module 11 builder override (tests inject a
            synthetic-dataset stub; production uses the default builder over
            the existing ``GraphRepository``).
    """

    def __init__(
        self,
        *,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        dataset_builder: Optional[GraphDatasetBuilder] = None,
    ) -> None:
        self._checkpoint_override = checkpoint_path
        self._device_override = device
        self._builder = dataset_builder or GraphDatasetBuilder()
        self._predictor: Optional[GNNPredictor] = None
        self._predictor_source: Optional[Path] = None

    # ------------------------------------------------------------------ #
    # Configuration resolution
    # ------------------------------------------------------------------ #

    @property
    def checkpoint_path(self) -> Optional[Path]:
        """Configured checkpoint path, or ``None`` when serving is disabled.

        Relative configured paths are resolved against the backend directory
        (see :func:`_resolve_checkpoint_path`), so the value is always an
        absolute :class:`~pathlib.Path` when a checkpoint is configured.
        """
        raw = self._checkpoint_override or settings.prediction_checkpoint_path
        return _resolve_checkpoint_path(raw) if raw else None

    @property
    def device(self) -> str:
        """Configured inference device (``"cpu"`` by default)."""
        return self._device_override or settings.prediction_device

    # ------------------------------------------------------------------ #
    # Model lifecycle (loaded once, reused; explicit reload supported)
    # ------------------------------------------------------------------ #

    def _get_predictor(self) -> GNNPredictor:
        """Return the reusable predictor, loading it lazily on first use.

        Raises:
            ModelNotAvailableError: If no checkpoint is configured.
            GNNModelNotAvailableError: If the configured file is missing.
            GNNCheckpointInvalidError: If the checkpoint cannot be loaded.
            GNNModelIncompatibleError: If its architecture is not servable.
        """
        path = self.checkpoint_path
        if path is None:
            raise ModelNotAvailableError(
                "no trained GNN checkpoint is configured"
            )
        if self._predictor is None or self._predictor_source != path:
            logger.info("Loading GNN prediction model", extra={"device": self.device})
            self._predictor = GNNPredictor(path, device=self.device)
            self._predictor_source = path
        return self._predictor

    def reload_model(self) -> None:
        """Drop the cached predictor (e.g. after deploying a new checkpoint).

        The next prediction lazily reloads whatever checkpoint is configured
        at that point. Intentionally NOT called automatically per request:
        reloads are an explicit operational action.
        """
        self._predictor = None
        self._predictor_source = None

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def get_predictions(
        self, request: Optional[PredictionRequest] = None
    ) -> PredictionResponse:
        """Predict a downstream-delay value for every node in the graph.

        Args:
            request: Optional request (currently only
                ``relationship_types``); a missing body means defaults.

        Returns:
            A :class:`PredictionResponse` with one entry per graph node, in
            the dataset's node-id order.

        Raises:
            ModelNotAvailableError: No checkpoint is configured.
            GNNPredictionError: Any inference-layer failure (missing/invalid
                checkpoint, incompatible model, invalid graph, runtime
                failure) — the API layer maps these to HTTP responses.
            neo4j.ServiceUnavailable / neo4j.Neo4jError: Propagated from the
                Module 11 extraction; mapped by the API layer.
        """
        payload = request if request is not None else PredictionRequest()
        # Load/validate the model BEFORE touching the database so a missing
        # checkpoint fails fast without requiring Neo4j availability.
        predictor = self._get_predictor()

        # Module 11 (existing): fresh dataset from the live graph. The
        # builder raises EmptyGraphError / GraphDatasetValidationError for
        # unusable graphs — translated by the API layer, never papered over.
        dataset: GraphDataset = self._builder.build(
            relationship_types=payload.relationship_types
        )

        # Module 14 inference (eval mode, no grads, y never read).
        result = predictor.predict(dataset)

        response = PredictionResponse(
            predictions=[
                NodePrediction(node_id=node_id, prediction=value)
                for node_id, value in zip(result.node_ids, result.predictions)
            ],
            prediction_count=len(result.predictions),
        )
        logger.info(
            "Prediction response built",
            extra={
                "prediction_count": response.prediction_count,
                "checkpoint_epoch": result.checkpoint_epoch,
            },
        )
        return response

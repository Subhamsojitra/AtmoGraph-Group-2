"""Prediction API routes for AtmoGraph (Module 14).

Exposes one endpoint that returns the trained GNN's node-level downstream-
delay prediction for every entity in the current supply-chain graph. All
business logic lives in
:class:`app.services.prediction_service.PredictionService`, which reuses the
existing Module 11 dataset builder and the Module 14 inference layer — the
route itself stays thin and contains NO ML logic.

Architecture:

    FastAPI
        |
    Prediction API (this module)
        |
    PredictionService
        |
    GraphDatasetBuilder (Module 11, existing) | GNNPredictor (Module 14)
        |
    Neo4jDatabase (existing singleton)     |  trained checkpoint (state_dict)

Error mapping (without leaking internals — no paths, no stack traces):

| Outcome                                   | HTTP |
| ----------------------------------------- | ---- |
| Success                                   | 200  |
| Invalid request body                      | 422  |
| No trained checkpoint configured/missing  | 503  |
| Neo4j unavailable                         | 503  |
| Graph empty                               | 503  |
| Invalid/incompatible checkpoint or model  | 500  |
| Unexpected error                          | 500  |
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.core.logger import get_logger
from app.ml.exceptions import (
    EmptyGraphError,
    GraphDatasetError,
    GNNCheckpointInvalidError,
    GNNModelIncompatibleError,
    GNNModelNotAvailableError,
    GNNPredictionError,
    GNNPredictionInputError,
    GNNPredictionRuntimeError,
)
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.prediction_service import (
    ModelNotAvailableError,
    PredictionService,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/predictions", tags=["predictions"])


# --------------------------------------------------------------------------- #
# Dependency injection
# --------------------------------------------------------------------------- #

#: Module-level shared service instance. The GNN model is expensive to load,
#: so ONE service (and therefore ONE loaded model) is reused across requests
#: instead of reloading the checkpoint per prediction — the same lifecycle
#: pattern as the existing ``neo4j_db`` singleton. Tests override the whole
#: dependency, so this singleton never interferes with the test suite.
_prediction_service: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    """Provide the shared :class:`PredictionService` to route handlers."""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service


def reset_prediction_service() -> None:
    """Drop the shared service instance (used by tests and reload tooling)."""
    global _prediction_service
    _prediction_service = None


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@router.post(
    "",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict downstream delay for every entity in the graph",
    description=(
        "Build the GNN dataset from the current supply-chain graph (Module 11) "
        "and run the trained GNN (Modules 12/13 checkpoint) in eval mode to "
        "return one raw predicted downstream-delay value per node. The raw "
        "regression value is served as-is: no severity classification is "
        "derived because no thresholds are specified."
    ),
    responses={
        422: {"description": "Invalid request body"},
        503: {
            "description": (
                "Trained model not available (no checkpoint configured), "
                "graph database unavailable, or the graph is empty"
            )
        },
        500: {
            "description": (
                "Invalid/incompatible checkpoint or model, unusable graph "
                "data, or an unexpected internal error"
            )
        },
    },
)
def predict(
    payload: Optional[PredictionRequest] = None,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    """Serve one raw GNN prediction per graph node.

    The request body is optional; when present it may narrow the
    relationships used for dataset preparation (``relationship_types``).
    """
    try:
        return service.get_predictions(
            payload if payload is not None else PredictionRequest()
        )
    except ModelNotAvailableError:
        logger.warning("Prediction API: no trained model is configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Prediction service is not available: no trained GNN "
                "checkpoint is configured on the server."
            ),
        )
    except GNNModelNotAvailableError:
        logger.warning("Prediction API: configured checkpoint is missing")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Prediction service is not available: the configured "
                "trained GNN checkpoint could not be found."
            ),
        )
    except ServiceUnavailable:
        logger.error("Prediction API: Neo4j service unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Graph database service is currently unavailable. "
                "Please try again later."
            ),
        )
    except EmptyGraphError:
        logger.warning("Prediction API: graph is empty")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No graph data available for prediction: the graph "
                "database currently contains no entities."
            ),
        )
    except GNNPredictionInputError:
        logger.error("Prediction API: invalid inference input", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Prediction failed: the graph data is not usable by the model."
            ),
        )
    except GNNCheckpointInvalidError:
        logger.error("Prediction API: invalid checkpoint", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed: the stored model checkpoint is invalid.",
        )
    except GNNModelIncompatibleError:
        logger.error("Prediction API: model/data mismatch", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Prediction failed: the deployed model does not match the "
                "current graph features."
            ),
        )
    except GNNPredictionRuntimeError:
        logger.error("Prediction API: inference failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed while running the model.",
        )
    except GNNPredictionError:
        logger.error("Prediction API: prediction error", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed.",
        )
    except GraphDatasetError:
        logger.error(
            "Prediction API: graph data preparation failed", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed: the graph data could not be prepared.",
        )
    except Neo4jError as exc:
        logger.error("Prediction API: Neo4j error", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected database error occurred.",
        )
    except Exception:
        logger.error("Prediction API: unexpected error", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred.",
        )

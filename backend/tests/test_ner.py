"""Tests for Module 7 - NLP preprocessing & Named Entity Recognition.

These tests do NOT require a running Neo4j server. They cover:

* the text preprocessor (reusing Module 6 normalization),
* the lazy, fail-fast NER model manager,
* the NER service (deterministic fake + the real configured model),
* the ``POST /api/v1/news/analyze`` endpoint,
* JSON serializability and the 422/503/500 error mapping.

Real-model tests execute the *actual* configured Transformers model. They are
skipped (with an explicit message) only when the model package is not present
in the local HuggingFace cache, and they never trigger a model download
(``HF_HUB_OFFLINE=1``).
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.news import get_ner_service
from app.main import app
from app.schemas.ner import NERAnalysisRequest, NERAnalysisResponse, NEREntity
from app.services.nlp.exceptions import (
    ModelUnavailableError,
    TextPreprocessingError,
)
from app.services.nlp.model_loader import NERModelManager
from app.services.nlp.ner_service import NERService
from app.services.nlp.preprocessor import TextPreprocessor

# ---------------------------------------------------------------------------
# Test setup
# ---------------------------------------------------------------------------

# Fake Neo4j env vars before importing the app (required by pydantic-settings).
os.environ["NEO4J_URI"] = "neo4j://test-host:7687"
os.environ["NEO4J_USERNAME"] = "test_user"
os.environ["NEO4J_PASSWORD"] = "test_super_secret_123"
os.environ["NEO4J_DATABASE"] = "test_db"

client = TestClient(app)


# ---------------------------------------------------------------------------
# Deterministic fake model (never touches the real model or the network)
# ---------------------------------------------------------------------------

# Mimics the output of the transformers v5 TokenClassificationPipeline with
# aggregation_strategy="simple". np.float32 scores deliberately exercise the
# native-float conversion required for JSON serialization.
FAKE_ENTITIES: list[dict[str, Any]] = [
    {
        "entity_group": "ORG",
        "score": np.float32(0.991),
        "word": "Port of Rotterdam",
        "start": 15,
        "end": 32,
    },
    {
        "entity_group": "LOC",
        "score": np.float32(0.988),
        "word": "China",
        "start": 77,
        "end": 82,
    },
    {
        "entity_group": "LOC",
        "score": np.float32(0.974),
        "word": "Germany",
        "start": 86,
        "end": 93,
    },
]


class _FakePipeline:
    """Returns canned entities regardless of the input text."""

    def __init__(self, entities: list[dict[str, Any]]) -> None:
        self._entities = entities

    def __call__(self, text: str, **kwargs: Any) -> list[dict[str, Any]]:
        return self._entities


class _FakeModelManager:
    """In-memory stand-in for :class:`NERModelManager`.

    Mirrors the real semantics: the pipeline is built exactly once, then
    cached and reused (``load_count`` therefore counts actual model loads,
    matching what ``test_model_loaded_once_and_reused`` asserts). A configured
    load error is raised on every call, like the real fail-fast behavior.
    """

    def __init__(
        self,
        entities: list[dict[str, Any]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._entities = entities if entities is not None else FAKE_ENTITIES
        self._error = error
        self.load_count = 0
        self._pipeline: _FakePipeline | None = None

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def get_pipeline(self) -> _FakePipeline:
        if self._pipeline is None:
            self.load_count += 1
            if self._error is not None:
                raise self._error
            self._pipeline = _FakePipeline(self._entities)
        return self._pipeline


class _ExplodingPipeline(_FakePipeline):
    """Simulates an inference-time model failure."""

    def __call__(self, text: str, **kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("simulated inference failure")


class _ExplodingModelManager(_FakeModelManager):
    """Manager whose pipeline explodes during inference."""

    def get_pipeline(self) -> _FakePipeline:
        self.load_count += 1
        if self._error is not None:
            raise self._error
        return _ExplodingPipeline(self._entities)


@pytest.fixture()
def fake_model_manager() -> _FakeModelManager:
    return _FakeModelManager()


@pytest.fixture()
def fake_ner_service(fake_model_manager: _FakeModelManager) -> NERService:
    return NERService(
        model_manager=fake_model_manager,
        preprocessor=TextPreprocessor(max_length=10000),
    )


@pytest.fixture()
def api_client(fake_ner_service: NERService) -> Any:
    """TestClient with the NER dependency overridden by a fake service."""
    app.dependency_overrides[get_ner_service] = lambda: fake_ner_service
    yield client
    app.dependency_overrides.clear()


@pytest.fixture(scope="module")
def real_ner_service() -> Any:
    """A real NERService backed by the actual configured model.

    Loads the genuine Transformers model once per module. Skips (with an
    explicit message) when the model package is not present in the local
    HuggingFace cache; ``HF_HUB_OFFLINE=1`` guarantees this fixture never
    downloads the model on demand.
    """
    previous_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        manager = NERModelManager()
        service = NERService(model_manager=manager)
        manager.get_pipeline()  # force the real load now
    except Exception as exc:
        pytest.skip(f"Real NER model unavailable: {exc}")
    finally:
        if previous_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous_offline
    return service


# ---------------------------------------------------------------------------
# Preprocessor tests
# ---------------------------------------------------------------------------


def test_preprocessor_reuses_module6_normalization() -> None:
    pre = TextPreprocessor(max_length=10000)
    assert pre.preprocess("  hello\n\nworld\t\x00 ") == "hello world"


def test_preprocessor_rejects_empty_text() -> None:
    pre = TextPreprocessor(max_length=10000)
    with pytest.raises(TextPreprocessingError):
        pre.preprocess("   \n\t ")


def test_preprocessor_rejects_excessive_length() -> None:
    pre = TextPreprocessor(max_length=10)
    with pytest.raises(TextPreprocessingError):
        pre.preprocess("x" * 11)


def test_preprocessor_rejects_non_string() -> None:
    pre = TextPreprocessor(max_length=10)
    with pytest.raises(TextPreprocessingError):
        pre.preprocess(None)  # type: ignore[arg-type]


def test_preprocessor_rejects_invalid_max_length() -> None:
    with pytest.raises(ValueError):
        TextPreprocessor(max_length=0)


# ---------------------------------------------------------------------------
# Model manager tests (no network, no real model)
# ---------------------------------------------------------------------------


def test_model_manager_is_lazy_and_fail_fast(monkeypatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    manager = NERModelManager(model_name="nonexistent-model-xyz", max_length=128)
    assert manager.is_loaded is False
    with pytest.raises(ModelUnavailableError):
        manager.get_pipeline()
    # A second call must fail fast without re-attempting the load.
    with pytest.raises(ModelUnavailableError):
        manager.get_pipeline()
    assert manager.is_loaded is False


# ---------------------------------------------------------------------------
# NER service tests (deterministic fake model)
# ---------------------------------------------------------------------------


def test_service_initializes_without_loading_model() -> None:
    service = NERService()
    assert service.model_loaded is False


def test_extract_entities_returns_structured_entities(
    fake_ner_service: NERService,
) -> None:
    entities = fake_ner_service.extract_entities("any text")

    assert len(entities) == 3
    assert all(isinstance(e, NEREntity) for e in entities)
    assert entities[0].text == "Port of Rotterdam"
    assert entities[0].label == "ORG"
    assert entities[0].start == 15
    assert entities[0].end == 32
    assert entities[1].label == "LOC"


def test_confidence_is_native_float_and_json_serializable(
    fake_ner_service: NERService,
) -> None:
    entities = fake_ner_service.extract_entities("any text")

    assert all(isinstance(e.confidence, float) for e in entities)
    # np.float32 scores must have been converted to native floats, otherwise
    # json.dumps would raise.
    json.dumps([e.model_dump() for e in entities])


def test_extract_entities_handles_empty_text(fake_ner_service: NERService) -> None:
    with pytest.raises(TextPreprocessingError):
        fake_ner_service.extract_entities("   ")


def test_extract_entities_rejects_non_string(fake_ner_service: NERService) -> None:
    with pytest.raises(TextPreprocessingError):
        fake_ner_service.extract_entities(123)  # type: ignore[arg-type]


def test_analyze_rejects_invalid_payload(fake_ner_service: NERService) -> None:
    with pytest.raises(ValueError):
        fake_ner_service.analyze("not a request")  # type: ignore[arg-type]


def test_analyze_combines_title_and_content(fake_ner_service: NERService) -> None:
    response = fake_ner_service.analyze(
        NERAnalysisRequest(title="Title", content="Body text.")
    )

    assert response.analyzed_text == "Title. Body text."
    assert response.entity_count == 3
    assert isinstance(response, NERAnalysisResponse)


def test_analyze_uses_provided_article_id(fake_ner_service: NERService) -> None:
    response = fake_ner_service.analyze(
        NERAnalysisRequest(
            title="Title", content="Body text.", article_id="article-123"
        )
    )

    assert response.article_id == "article-123"


def test_analyze_generates_uuid_article_id(fake_ner_service: NERService) -> None:
    response = fake_ner_service.analyze(
        NERAnalysisRequest(title="Title", content="Body text.")
    )

    assert len(response.article_id) == 36
    assert response.article_id.count("-") == 4


def test_model_loaded_once_and_reused(
    fake_ner_service: NERService,
    fake_model_manager: _FakeModelManager,
) -> None:
    request = NERAnalysisRequest(title="Title", content="Body text.")
    fake_ner_service.analyze(request)
    fake_ner_service.analyze(request)

    # The model must not be (re)loaded per request.
    assert fake_model_manager.load_count == 1


def test_duplicate_occurrences_are_preserved(fake_ner_service: NERService) -> None:
    duplicate_entities = [
        {"entity_group": "LOC", "score": 0.99, "word": "China", "start": 0, "end": 5},
        {"entity_group": "LOC", "score": 0.98, "word": "Germany", "start": 23, "end": 30},
        {"entity_group": "LOC", "score": 0.97, "word": "China", "start": 32, "end": 37},
        {"entity_group": "LOC", "score": 0.96, "word": "Germany", "start": 52, "end": 59},
    ]
    service = NERService(
        model_manager=_FakeModelManager(entities=duplicate_entities),
        preprocessor=TextPreprocessor(max_length=10000),
    )

    entities = service.extract_entities(
        "China exports goods to Germany. China also supplies Germany."
    )

    assert len(entities) == 4
    chinas = [e for e in entities if e.text == "China"]
    assert len(chinas) == 2
    # Occurrences with different offsets must not be collapsed into one.
    assert len({(e.start, e.end) for e in chinas}) == 2
    assert chinas[0].start != chinas[1].start


# ---------------------------------------------------------------------------
# Real model tests (actual configured Transformers model, cached locally)
# ---------------------------------------------------------------------------


def test_real_model_simple_entity_extraction(real_ner_service: Any) -> None:
    entities = real_ner_service.extract_entities(
        "Apple Inc. is based in Cupertino, California."
    )

    by_text = {e.text: e.label for e in entities}
    assert by_text.get("Apple Inc") == "ORG"
    assert by_text.get("Cupertino") == "LOC"
    assert by_text.get("California") == "LOC"


def test_real_model_extracts_multiple_entities(real_ner_service: Any) -> None:
    entities = real_ner_service.extract_entities(
        "Workers at the Port of Rotterdam announced a strike affecting "
        "shipments from China to Germany."
    )

    assert len(entities) >= 3
    labels = {e.label for e in entities}
    assert "LOC" in labels
    texts = [e.text for e in entities]
    assert "China" in texts
    assert "Germany" in texts


def test_real_model_character_offsets_match_text(real_ner_service: Any) -> None:
    text = (
        "Port strike disrupts European shipments. Workers at the Port of "
        "Rotterdam announced a strike affecting shipments from China to Germany."
    )
    entities = real_ner_service.extract_entities(text)

    assert entities
    for entity in entities:
        assert text[entity.start : entity.end] == entity.text


def test_real_model_preserves_duplicate_occurrences(real_ner_service: Any) -> None:
    text = "China exports goods to Germany. China also supplies Germany."
    entities = real_ner_service.extract_entities(text)

    chinas = [e for e in entities if e.text == "China"]
    germanys = [e for e in entities if e.text == "Germany"]
    assert len(chinas) >= 2
    assert len(germanys) >= 2
    assert len({(e.start, e.end) for e in chinas}) == len(chinas)


def test_real_model_analyze_returns_structured_response(real_ner_service: Any) -> None:
    request = NERAnalysisRequest(
        title="Port strike disrupts European shipments",
        content=(
            "Workers at the Port of Rotterdam announced a strike affecting "
            "shipments from China to Germany."
        ),
        source="Example News",
    )

    response = real_ner_service.analyze(request)

    assert isinstance(response, NERAnalysisResponse)
    assert response.entity_count == len(response.entities)
    assert response.entity_count >= 3
    assert "LOC" in {e.label for e in response.entities}
    assert "China" in [e.text for e in response.entities]
    assert "Germany" in [e.text for e in response.entities]
    assert response.analyzed_text == (
        "Port strike disrupts European shipments. Workers at the Port of "
        "Rotterdam announced a strike affecting shipments from China to Germany."
    )
    assert len(response.article_id) == 36  # auto-generated UUIDv4


# ---------------------------------------------------------------------------
# API endpoint tests (deterministic fake service via dependency override)
# ---------------------------------------------------------------------------


def test_analyze_endpoint_returns_structured_ner_response(api_client: Any) -> None:
    payload = {
        "title": "Port strike disrupts European shipments",
        "content": (
            "Workers at the Port of Rotterdam announced a strike affecting "
            "shipments from China to Germany."
        ),
        "source": "Example News",
    }

    response = api_client.post("/api/v1/news/analyze", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["entity_count"] == 3
    assert len(data["entities"]) == 3
    assert data["entities"][0]["text"] == "Port of Rotterdam"
    assert data["entities"][0]["label"] == "ORG"
    assert data["entities"][0]["start"] == 15
    assert data["entities"][0]["end"] == 32
    assert isinstance(data["entities"][0]["confidence"], float)
    assert data["analyzed_text"].startswith("Port strike disrupts European shipments. ")
    assert data["article_id"]


def test_analyze_endpoint_response_is_json_serializable(api_client: Any) -> None:
    payload = {
        "title": "JSON test",
        "content": "Content for JSON serialization test.",
    }

    response = api_client.post("/api/v1/news/analyze", json=payload)

    assert response.status_code == 200
    response.json()  # must parse without error


def test_analyze_endpoint_missing_content_returns_422(api_client: Any) -> None:
    response = api_client.post("/api/v1/news/analyze", json={"title": "Title only"})

    assert response.status_code == 422


def test_analyze_endpoint_empty_content_returns_422(api_client: Any) -> None:
    response = api_client.post(
        "/api/v1/news/analyze", json={"title": "Valid title", "content": "   "}
    )

    assert response.status_code == 422


def test_analyze_endpoint_excessive_text_returns_422(api_client: Any) -> None:
    response = api_client.post(
        "/api/v1/news/analyze", json={"title": "Valid title", "content": "x" * 11000}
    )

    assert response.status_code == 422


def test_analyze_endpoint_invalid_article_id_returns_422(api_client: Any) -> None:
    response = api_client.post(
        "/api/v1/news/analyze",
        json={"title": "Valid title", "content": "Valid content.", "article_id": "   "},
    )

    assert response.status_code == 422


def test_analyze_endpoint_model_unavailable_returns_503() -> None:
    manager = _FakeModelManager(error=ModelUnavailableError("model unavailable"))
    service = NERService(
        model_manager=manager, preprocessor=TextPreprocessor(max_length=10000)
    )
    app.dependency_overrides[get_ner_service] = lambda: service
    try:
        response = client.post(
            "/api/v1/news/analyze", json={"title": "Title", "content": "Content."}
        )
        assert response.status_code == 503
    finally:
        app.dependency_overrides.clear()


def test_analyze_endpoint_processing_error_returns_500() -> None:
    service = NERService(
        model_manager=_ExplodingModelManager(),
        preprocessor=TextPreprocessor(max_length=10000),
    )
    app.dependency_overrides[get_ner_service] = lambda: service
    try:
        response = client.post(
            "/api/v1/news/analyze", json={"title": "Title", "content": "Content."}
        )
        assert response.status_code == 500
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Router registration & startup behavior
# ---------------------------------------------------------------------------


def test_analyze_route_is_registered() -> None:
    paths = {r.path for r in app.routes}
    assert "/api/v1/news/analyze" in paths


def test_ner_service_singleton_is_lazy() -> None:
    # Importing the app must not load the heavy NLP model.
    assert get_ner_service().model_loaded is False
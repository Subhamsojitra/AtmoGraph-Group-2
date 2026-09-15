"""Entity Resolution service for AtmoGraph (Module 8).

Maps NER entities extracted by Module 7 to candidate graph nodes in Neo4j
without creating or mutating any graph data. Reuses the existing
:class:`app.repositories.graph_repository.GraphRepository` and
:func:`app.utils.text_normalizer.normalize_text`.

Architecture:

    FastAPI
        ↓
    EntityResolutionService (this module)
        ↓
    GraphRepository
        ↓
    Neo4jDatabase (existing singleton)
"""

from __future__ import annotations

import difflib
import uuid
from typing import Any, Optional

from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.core.logger import get_logger
from app.repositories.graph_repository import GraphRepository
from app.schemas.entity_resolution import (
    EntityCandidate,
    EntityResolutionResponse,
    ResolvedEntity,
)
from app.schemas.ner import NEREntity
from app.services.entity_resolution.exceptions import EntityResolutionError
from app.utils.text_normalizer import normalize_text

logger = get_logger(__name__)


class EntityResolutionService:
    """Resolve NER entities against existing Neo4j graph nodes.

    Matching strategy:

    1. Normalize the entity text (whitespace, Unicode, case).
    2. Query :class:`GraphRepository.find_entity_candidates` for candidate
       nodes whose ``name``, ``aliases``, or ``id`` contain the normalized text.
    3. Classify each candidate as *exact*, *fuzzy*, or *alias*.
    4. Return a structured resolution result per entity.

    This service never writes to Neo4j.
    """

    def __init__(
        self,
        repository: Optional[GraphRepository] = None,
        fuzzy_threshold: float = 0.6,
    ) -> None:
        self._repository = repository or GraphRepository()
        self._fuzzy_threshold = fuzzy_threshold

    def resolve(
        self,
        entities: list[NEREntity],
        article_id: Optional[str] = None,
    ) -> EntityResolutionResponse:
        """Resolve a batch of NER entities against the graph.

        Args:
            entities: Occurrence-level NER entities from Module 7.
            article_id: Optional article identifier. Generated when omitted.

        Returns:
            ``EntityResolutionResponse`` with per-entity resolution results.

        Raises:
            ValueError: If ``entities`` is not a list.
            ServiceUnavailable: If Neo4j is unreachable.
            Neo4jError: If a database error occurs.
            EntityResolutionError: If an unexpected failure occurs.
        """
        if not isinstance(entities, list):
            raise ValueError("entities must be a list")

        if not entities:
            raise ValueError("entities must not be empty")

        article_id = article_id.strip() if article_id else str(uuid.uuid4())

        resolved_entities = [self._resolve_entity(entity) for entity in entities]

        resolved_count = sum(1 for r in resolved_entities if r.matched)
        unresolved_count = sum(
            1 for r in resolved_entities if not r.matched and r.match_method == "unresolved"
        )
        ambiguous_count = sum(
            1 for r in resolved_entities if r.match_method == "ambiguous"
        )

        return EntityResolutionResponse(
            article_id=article_id,
            resolved_entities=resolved_entities,
            total_entities=len(resolved_entities),
            resolved_count=resolved_count,
            unresolved_count=unresolved_count,
            ambiguous_count=ambiguous_count,
        )

    def _resolve_entity(self, entity: NEREntity) -> ResolvedEntity:
        """Resolve a single NER entity occurrence."""
        original_text = entity.text
        normalized_text = normalize_text(original_text).lower()

        if not normalized_text:
            return ResolvedEntity(
                original_text=original_text,
                normalized_text="",
                ner_label=entity.label,
                matched=False,
                match_method="unresolved",
                confidence=0.0,
            )

        try:
            candidates_raw = self._repository.find_entity_candidates(
                normalized_text, limit=20
            )
        except (ServiceUnavailable, Neo4jError):
            raise
        except Exception as exc:
            logger.error("Entity resolution repository failed", exc_info=True)
            raise EntityResolutionError("Entity resolution failed") from exc

        candidates = [self._extract_node(record) for record in candidates_raw]

        exact_matches = self._find_exact_matches(normalized_text, candidates)
        if len(exact_matches) == 1:
            return self._build_exact_result(original_text, normalized_text, entity, exact_matches[0])

        if len(exact_matches) > 1:
            return ResolvedEntity(
                original_text=original_text,
                normalized_text=normalized_text,
                ner_label=entity.label,
                matched=False,
                match_method="ambiguous",
                confidence=0.0,
                candidates=self._to_entity_candidates(exact_matches, "exact", 1.0),
            )

        fuzzy_matches = self._find_fuzzy_matches(normalized_text, candidates)
        if not fuzzy_matches:
            return ResolvedEntity(
                original_text=original_text,
                normalized_text=normalized_text,
                ner_label=entity.label,
                matched=False,
                match_method="unresolved",
                confidence=0.0,
            )

        fuzzy_matches.sort(key=lambda x: x[0], reverse=True)

        if len(fuzzy_matches) == 1:
            ratio, candidate = fuzzy_matches[0]
            return self._build_fuzzy_result(original_text, normalized_text, entity, candidate, ratio)

        best_ratio = fuzzy_matches[0][0]
        second_ratio = fuzzy_matches[1][0]

        if best_ratio - second_ratio > 0.1:
            ratio, candidate = fuzzy_matches[0]
            return self._build_fuzzy_result(original_text, normalized_text, entity, candidate, ratio)

        return ResolvedEntity(
            original_text=original_text,
            normalized_text=normalized_text,
            ner_label=entity.label,
            matched=False,
            match_method="ambiguous",
            confidence=0.0,
            candidates=self._to_entity_candidates(
                [c for _, c in fuzzy_matches[:5]], "fuzzy", best_ratio
            ),
        )

    def _find_exact_matches(
        self, normalized_text: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        matches = []
        for candidate in candidates:
            props = candidate.get("properties", {})
            name = (props.get("name") or candidate.get("id", "")).lower()
            aliases = [a.lower() for a in props.get("aliases", []) if isinstance(a, str)]
            if normalized_text == name or normalized_text in aliases:
                matches.append(candidate)
        return matches

    def _find_fuzzy_matches(
        self, normalized_text: str, candidates: list[dict[str, Any]]
    ) -> list[tuple[float, dict[str, Any]]]:
        matches = []
        for candidate in candidates:
            cname = (candidate.get("properties", {}).get("name") or candidate.get("id", "")).lower()
            ratio = difflib.SequenceMatcher(None, normalized_text, cname).ratio()
            if ratio >= self._fuzzy_threshold:
                matches.append((ratio, candidate))
        return matches

    def _build_exact_result(
        self,
        original_text: str,
        normalized_text: str,
        entity: NEREntity,
        candidate: dict[str, Any],
    ) -> ResolvedEntity:
        props = candidate.get("properties", {})
        name = props.get("name", candidate.get("id", ""))
        return ResolvedEntity(
            original_text=original_text,
            normalized_text=normalized_text,
            ner_label=entity.label,
            matched=True,
            node_id=candidate.get("id", ""),
            node_label=candidate.get("label", "Unknown"),
            node_name=name,
            match_method="exact",
            confidence=1.0,
        )

    def _build_fuzzy_result(
        self,
        original_text: str,
        normalized_text: str,
        entity: NEREntity,
        candidate: dict[str, Any],
        ratio: float,
    ) -> ResolvedEntity:
        props = candidate.get("properties", {})
        name = props.get("name", candidate.get("id", ""))
        return ResolvedEntity(
            original_text=original_text,
            normalized_text=normalized_text,
            ner_label=entity.label,
            matched=True,
            node_id=candidate.get("id", ""),
            node_label=candidate.get("label", "Unknown"),
            node_name=name,
            match_method="fuzzy",
            confidence=ratio,
        )

    def _to_entity_candidates(
        self,
        nodes: list[dict[str, Any]],
        match_type: str,
        score: float,
    ) -> list[EntityCandidate]:
        result = []
        for node in nodes:
            props = node.get("properties", {})
            name = props.get("name", node.get("id", ""))
            result.append(
                EntityCandidate(
                    node_id=node.get("id", ""),
                    node_name=name,
                    node_label=node.get("label", "Unknown"),
                    score=score,
                    match_type=match_type,
                    properties=props,
                )
            )
        return result

    def _extract_node(self, record: dict[str, Any]) -> dict[str, Any]:
        if isinstance(record, dict) and "id" in record and "properties" in record:
            return {
                "id": record.get("id", ""),
                "label": record.get("label", "Unknown"),
                "properties": record.get("properties", {}),
            }
        for value in record.values():
            if hasattr(value, "labels") and hasattr(value, "items"):
                node_props = dict(value.items())
                labels = list(value.labels)
                return {
                    "id": node_props.get("id", ""),
                    "label": labels[0] if labels else "Unknown",
                    "properties": node_props,
                }
        return {"id": "", "label": "Unknown", "properties": {}}

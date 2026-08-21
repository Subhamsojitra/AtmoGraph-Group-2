"""Graph service layer for AtmoGraph.

This module provides the business-level graph operations consumed by the
API layer. It depends on the repository for raw database access and does
not create or manage Neo4j drivers directly.

Architecture:

    Graph API
        ↓
    Graph Service
        ↓
    Graph Repository
        ↓
    Neo4jDatabase
        ↓
    Neo4j Driver
        ↓
    Neo4j Database
"""

from __future__ import annotations

from typing import Any, Optional

from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.core.logger import get_logger
from app.repositories.graph_repository import GraphRepository
from app.schemas.graph import (
    GraphNodeResponse,
    GraphQueryResponse,
    GraphRelationshipResponse,
    GraphResultResponse,
)

logger = get_logger(__name__)


class GraphService:
    """Application-level graph operations.

    All external callers should use this service instead of accessing the
    repository or database manager directly. This keeps API/business logic
    separate from raw database access.
    """

    def __init__(self, repository: Optional[GraphRepository] = None) -> None:
        self._repository = repository or GraphRepository()

    def get_node_by_id(self, node_id: str) -> Optional[GraphNodeResponse]:
        """Retrieve a single node by its identifier.

        Args:
            node_id: The node identifier to look up.

        Returns:
            A ``GraphNodeResponse`` if found, otherwise ``None``.

        Raises:
            ValueError: If ``node_id`` is empty or not a string.
            ServiceUnavailable: If Neo4j is unreachable.
        """
        if not node_id or not isinstance(node_id, str):
            raise ValueError("node_id must be a non-empty string")

        try:
            record = self._repository.get_node_by_id(node_id)
            if record is None:
                return None

            node = self._extract_node(record)
            return GraphNodeResponse(
                id=node.get("id", node_id),
                label=node.get("label", "Unknown"),
                properties=node.get("properties", {}),
            )
        except ServiceUnavailable:
            logger.error(
                "GraphService: Neo4j unavailable during node lookup",
                extra={"node_id": node_id},
            )
            raise
        except Neo4jError as e:
            logger.error(
                "GraphService: Neo4j error during node lookup",
                extra={"node_id": node_id, "error": str(e)},
            )
            raise
        except Exception as e:
            logger.error(
                "GraphService: unexpected error during node lookup",
                extra={"node_id": node_id, "error": str(e)},
                exc_info=True,
            )
            raise

    def get_nodes(self, limit: int = 100) -> list[GraphNodeResponse]:
        """Retrieve a list of generic graph nodes.

        Args:
            limit: Maximum number of nodes to return.

        Returns:
            List of ``GraphNodeResponse`` objects.
        """
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        try:
            records = self._repository.get_nodes(limit=limit)
            return [self._build_node_response(record) for record in records]
        except ServiceUnavailable:
            logger.error("GraphService: Neo4j unavailable during get_nodes")
            raise
        except Neo4jError as e:
            logger.error(
                "GraphService: Neo4j error during get_nodes",
                extra={"error": str(e)},
            )
            raise
        except Exception as e:
            logger.error(
                "GraphService: unexpected error during get_nodes",
                extra={"error": str(e)},
                exc_info=True,
            )
            raise

    def find_nodes(
        self,
        label: Optional[str] = None,
        properties: Optional[dict[str, Any]] = None,
        limit: int = 100,
    ) -> list[GraphNodeResponse]:
        """Find nodes matching optional label and property filters.

        Args:
            label: Optional node label to filter by.
            properties: Optional property key-value pairs to match.
            limit: Maximum number of results.

        Returns:
            List of ``GraphNodeResponse`` objects.
        """
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        properties = properties or {}

        try:
            records = self._repository.find_nodes(
                label=label, properties=properties, limit=limit
            )
            return [self._build_node_response(record) for record in records]
        except ServiceUnavailable:
            logger.error(
                "GraphService: Neo4j unavailable during find_nodes",
                extra={"label": label, "properties": properties},
            )
            raise
        except Neo4jError as e:
            logger.error(
                "GraphService: Neo4j error during find_nodes",
                extra={"label": label, "properties": properties, "error": str(e)},
            )
            raise
        except Exception as e:
            logger.error(
                "GraphService: unexpected error during find_nodes",
                extra={"label": label, "properties": properties, "error": str(e)},
                exc_info=True,
            )
            raise

    def find_neighbors(
        self,
        node_id: str,
        relationship_types: Optional[list[str]] = None,
        limit: int = 100,
    ) -> list[GraphNodeResponse]:
        """Find neighboring nodes for the given node.

        Args:
            node_id: The source node identifier.
            relationship_types: Optional relationship type filters.
            limit: Maximum number of neighbors.

        Returns:
            List of neighbor ``GraphNodeResponse`` objects.
        """
        if not node_id or not isinstance(node_id, str):
            raise ValueError("node_id must be a non-empty string")

        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        try:
            records = self._repository.find_neighbors(
                node_id=node_id,
                relationship_types=relationship_types,
                limit=limit,
            )
            return [self._build_node_response(record) for record in records]
        except ServiceUnavailable:
            logger.error(
                "GraphService: Neo4j unavailable during find_neighbors",
                extra={"node_id": node_id},
            )
            raise
        except Neo4jError as e:
            logger.error(
                "GraphService: Neo4j error during find_neighbors",
                extra={"node_id": node_id, "error": str(e)},
            )
            raise
        except Exception as e:
            logger.error(
                "GraphService: unexpected error during find_neighbors",
                extra={"node_id": node_id, "error": str(e)},
                exc_info=True,
            )
            raise

    def execute_read_query(
        self,
        query: str,
        parameters: Optional[dict[str, Any]] = None,
    ) -> GraphQueryResponse:
        """Execute a parameterized read query and return generic results.

        This internal method must NOT be exposed as an unrestricted public
        API endpoint.

        Args:
            query: Parameterized Cypher query.
            parameters: Query parameters.

        Returns:
            ``GraphQueryResponse`` containing the query results.
        """
        if not query or not isinstance(query, str):
            raise ValueError("query must be a non-empty string")

        try:
            records = self._repository.execute_read(query, parameters)
            results = [GraphResultResponse(data=record) for record in records]
            return GraphQueryResponse(results=results, count=len(results))
        except ServiceUnavailable:
            logger.error(
                "GraphService: Neo4j unavailable during read query",
            )
            raise
        except Neo4jError as e:
            logger.error(
                "GraphService: Neo4j error during read query",
                extra={"error": str(e)},
            )
            raise
        except Exception as e:
            logger.error(
                "GraphService: unexpected error during read query",
                extra={"error": str(e)},
                exc_info=True,
            )
            raise

    def execute_write_query(
        self,
        query: str,
        parameters: Optional[dict[str, Any]] = None,
    ) -> GraphQueryResponse:
        """Execute a parameterized write query and return generic results.

        This internal method must NOT be exposed as an unrestricted public
        API endpoint.

        Args:
            query: Parameterized Cypher query.
            parameters: Query parameters.

        Returns:
            ``GraphQueryResponse`` containing the query results.
        """
        if not query or not isinstance(query, str):
            raise ValueError("query must be a non-empty string")

        try:
            records = self._repository.execute_write(query, parameters)
            results = [GraphResultResponse(data=record) for record in records]
            return GraphQueryResponse(results=results, count=len(results))
        except ServiceUnavailable:
            logger.error(
                "GraphService: Neo4j unavailable during write query",
            )
            raise
        except Neo4jError as e:
            logger.error(
                "GraphService: Neo4j error during write query",
                extra={"error": str(e)},
            )
            raise
        except Exception as e:
            logger.error(
                "GraphService: unexpected error during write query",
                extra={"error": str(e)},
                exc_info=True,
            )
            raise

    def create_node(
        self,
        label: str,
        properties: dict[str, Any],
    ) -> GraphNodeResponse:
        """Create a generic node.

        This method is schema-neutral. The final supply-chain node types
        and required properties will be defined by the teammate and should
        be implemented as dedicated service methods once the schema is
        finalized.

        Args:
            label: Node label (e.g., ``Supplier``, ``Factory``).
            properties: Node properties.

        Returns:
            ``GraphNodeResponse`` for the created node.
        """
        if not label or not isinstance(label, str):
            raise ValueError("label must be a non-empty string")

        if not isinstance(properties, dict):
            raise ValueError("properties must be a dictionary")

        query = f"CREATE (n:`{label}` $properties) RETURN n"
        parameters = {"properties": properties}

        results = self._repository.execute_write(query, parameters)
        if not results:
            raise RuntimeError("Node creation returned no result")

        node = self._extract_node(results[0])
        return GraphNodeResponse(
            id=node.get("id", ""),
            label=label,
            properties=node.get("properties", properties),
        )

    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Optional[dict[str, Any]] = None,
    ) -> GraphRelationshipResponse:
        """Create a generic relationship between two nodes.

        This method is schema-neutral. Final supply-chain relationship
        types and required properties will be added once the schema is
        finalized.

        Args:
            source_id: Source node identifier.
            target_id: Target node identifier.
            rel_type: Relationship type (e.g., ``SUPPLIES``).
            properties: Optional relationship properties.

        Returns:
            ``GraphRelationshipResponse`` for the created relationship.
        """
        if not source_id or not isinstance(source_id, str):
            raise ValueError("source_id must be a non-empty string")

        if not target_id or not isinstance(target_id, str):
            raise ValueError("target_id must be a non-empty string")

        if not rel_type or not isinstance(rel_type, str):
            raise ValueError("rel_type must be a non-empty string")

        properties = properties or {}

        query = (
            "MATCH (a) WHERE a.id = $source_id "
            "MATCH (b) WHERE b.id = $target_id "
            f"CREATE (a)-[r:`{rel_type}` $properties]->(b) "
            "RETURN r"
        )
        parameters = {
            "source_id": source_id,
            "target_id": target_id,
            "properties": properties,
        }

        results = self._repository.execute_write(query, parameters)
        if not results:
            raise RuntimeError("Relationship creation returned no result")

        rel = self._extract_relationship(results[0])
        return GraphRelationshipResponse(
            id=rel.get("id", ""),
            type=rel_type,
            source_id=source_id,
            target_id=target_id,
            properties=rel.get("properties", properties),
        )

    def _build_node_response(self, record: dict[str, Any]) -> GraphNodeResponse:
        """Build a ``GraphNodeResponse`` from a raw record dictionary.

        Args:
            record: Raw record dictionary from the repository.

        Returns:
            ``GraphNodeResponse`` instance.
        """
        node = self._extract_node(record)
        return GraphNodeResponse(
            id=node.get("id", ""),
            label=node.get("label", "Unknown"),
            properties=node.get("properties", {}),
        )

    def _extract_node(self, record: dict[str, Any]) -> dict[str, Any]:
        """Extract node data from a raw repository record.

        Neo4j returns nodes under their variable names (e.g., ``n``).
        This helper normalizes that into a predictable structure. It also
        handles plain dictionaries returned by mocked or simplified
        repository implementations.

        Args:
            record: Raw dictionary from the repository.

        Returns:
            Dictionary with ``id``, ``label``, and ``properties`` keys.
        """
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

    def _extract_relationship(
        self, record: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract relationship data from a raw repository record.

        Args:
            record: Raw dictionary from the repository.

        Returns:
            Dictionary with ``id``, ``type``, and ``properties`` keys.
        """
        if isinstance(record, dict) and "type" in record:
            return {
                "id": record.get("id", ""),
                "type": record.get("type", "UNKNOWN"),
                "properties": record.get("properties", {}),
            }
        for value in record.values():
            if hasattr(value, "type") and hasattr(value, "items"):
                rel_props = dict(value.items())
                return {
                    "id": rel_props.get("id", ""),
                    "type": value.type,
                    "properties": rel_props,
                }
        return {"id": "", "type": "UNKNOWN", "properties": {}}

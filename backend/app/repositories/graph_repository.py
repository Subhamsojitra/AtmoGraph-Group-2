"""Generic graph repository for AtmoGraph.

This module provides a schema-neutral database access layer between the
application service layer and the Neo4j database manager. It is responsible
only for translating raw Neo4j driver results into plain Python structures
and does not contain business logic or FastAPI concerns.
"""

from __future__ import annotations

from typing import Any, Optional

from neo4j import Record
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.core.logger import get_logger
from app.database.neo4j import neo4j_db

logger = get_logger(__name__)


class GraphRepository:
    """Schema-neutral repository for graph database operations.

    All Cypher queries must use parameterized bindings to prevent injection.
    This class does not assume any final supply-chain node labels or
    relationship types; those will be provided by the teammate completing
    the Neo4j schema.
    """

    def __init__(self, db: Any = neo4j_db) -> None:
        self._db = db

    def execute_read(
        self,
        query: str,
        parameters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Execute a parameterized read-only Cypher query.

        Args:
            query: Parameterized Cypher query string.
            parameters: Query parameters for safe binding.

        Returns:
            List of records converted to dictionaries.

        Raises:
            ServiceUnavailable: If the Neo4j driver is not initialized.
            Neo4jError: If the query execution fails.
            ValueError: If query is empty or not a string.
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        parameters = parameters or {}

        try:
            raw_results = self._db.execute_read(query, parameters)
            return [self._record_to_dict(record) for record in raw_results]
        except ServiceUnavailable:
            logger.error("Neo4j driver not initialized during read query")
            raise
        except Neo4jError as e:
            logger.error(
                "Read query failed",
                extra={"query": query, "parameters": parameters, "error": str(e)},
                exc_info=True,
            )
            raise
        except Exception as e:
            logger.error(
                "Unexpected error during read query",
                extra={"query": query, "parameters": parameters, "error": str(e)},
                exc_info=True,
            )
            raise

    def execute_write(
        self,
        query: str,
        parameters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Execute a parameterized write Cypher query within a transaction.

        Args:
            query: Parameterized Cypher query string.
            parameters: Query parameters for safe binding.

        Returns:
            List of records converted to dictionaries.

        Raises:
            ServiceUnavailable: If the Neo4j driver is not initialized.
            Neo4jError: If the query execution fails.
            ValueError: If query is empty or not a string.
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        parameters = parameters or {}

        try:
            raw_results = self._db.execute_write(query, parameters)
            return [self._record_to_dict(record) for record in raw_results]
        except ServiceUnavailable:
            logger.error("Neo4j driver not initialized during write query")
            raise
        except Neo4jError as e:
            logger.error(
                "Write query failed",
                extra={"query": query, "parameters": parameters, "error": str(e)},
                exc_info=True,
            )
            raise
        except Exception as e:
            logger.error(
                "Unexpected error during write query",
                extra={"query": query, "parameters": parameters, "error": str(e)},
                exc_info=True,
            )
            raise

    def get_node_by_id(
        self,
        node_id: str,
    ) -> Optional[dict[str, Any]]:
        """Retrieve a single node by its identifier.

        This generic method does not assume any specific node label. The
        finalized schema from the teammate will dictate how node lookups
        are performed; this method provides a safe, parameterized default.

        Args:
            node_id: The node identifier to look up.

        Returns:
            A dictionary representing the node, or None if not found.

        Raises:
            ServiceUnavailable: If the Neo4j driver is not initialized.
            Neo4jError: If the query execution fails.
            ValueError: If node_id is empty or not a string.
        """
        if not node_id or not isinstance(node_id, str):
            raise ValueError("node_id must be a non-empty string")

        query = "MATCH (n) WHERE n.id = $node_id RETURN n LIMIT 1"
        parameters = {"node_id": node_id}

        results = self.execute_read(query, parameters)
        return results[0] if results else None

    def get_nodes(
        self,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Retrieve a list of nodes.

        Args:
            limit: Maximum number of nodes to return.

        Returns:
            List of node dictionaries.

        Raises:
            ValueError: If limit is not a positive integer.
        """
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        query = "MATCH (n) RETURN n LIMIT $limit"
        parameters = {"limit": limit}

        return self.execute_read(query, parameters)

    def find_nodes(
        self,
        label: Optional[str] = None,
        properties: Optional[dict[str, Any]] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Find nodes matching optional label and property filters.

        Args:
            label: Optional node label to filter by.
            properties: Optional property key-value pairs to match.
            limit: Maximum number of results.

        Returns:
            List of matching node dictionaries.
        """
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        properties = properties or {}

        if label:
            query_parts = ["MATCH (n:`{}`) ".format(label)]
        else:
            query_parts = ["MATCH (n) "]

        if properties:
            conditions = []
            for key in properties:
                conditions.append(f"n.`{key}` = $`{key}`")
            query_parts.append("WHERE " + " AND ".join(conditions))

        query_parts.append("RETURN n LIMIT $limit")
        query = "".join(query_parts)

        parameters: dict[str, Any] = dict(properties)
        parameters["limit"] = limit

        return self.execute_read(query, parameters)

    def find_neighbors(
        self,
        node_id: str,
        relationship_types: Optional[list[str]] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Find neighboring nodes connected to the given node.

        Args:
            node_id: The source node identifier.
            relationship_types: Optional list of relationship types to filter.
            limit: Maximum number of neighbors to return.

        Returns:
            List of neighbor dictionaries.
        """
        if not node_id or not isinstance(node_id, str):
            raise ValueError("node_id must be a non-empty string")

        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        if relationship_types:
            rel_pattern = "|".join(
                f"`{rel_type}`" for rel_type in relationship_types
            )
            query = (
                "MATCH (n) WHERE n.id = $node_id "
                "MATCH (n)-[r:`{}`]-(m) "
                "RETURN m LIMIT $limit"
            ).format(rel_pattern)
        else:
            query = (
                "MATCH (n) WHERE n.id = $node_id "
                "MATCH (n)-[r]-(m) "
                "RETURN m LIMIT $limit"
            )

        parameters = {"node_id": node_id, "limit": limit}

        return self.execute_read(query, parameters)

    def find_downstream_neighbors(
        self,
        node_ids: list[str],
        relationship_types: Optional[list[str]] = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Find nodes reachable via an *outgoing* relationship from any source.

        This is a single-hop, directional (downstream) traversal used by the
        risk propagation module (Module 10). A supply-chain risk flows from the
        source entity toward the entities that depend on it, i.e. along outgoing
        relationships (``(n)-[r]->(m)``).

        The ``node_ids`` are bound as a parameter (never interpolated) and the
        returned nodes are deduplicated. It is read-only and never creates data.

        Args:
            node_ids: Source node identifiers to expand one hop downstream.
            relationship_types: Optional list of relationship types to follow.
                When provided, only these relationship types are traversed.
            limit: Maximum number of downstream neighbors to return.

        Returns:
            List of record dictionaries of the shape
            ``{"node": <node>, "rel_type": str, "source_id": str}``.

        Raises:
            ValueError: If ``node_ids`` is empty/not a list of non-empty strings,
                or ``limit`` is not a positive integer.
            ServiceUnavailable: If Neo4j is unreachable.
            Neo4jError: If the query execution fails.
        """
        if not isinstance(node_ids, list) or not node_ids:
            raise ValueError("node_ids must be a non-empty list of node identifiers")
        cleaned_ids = []
        for node_id in node_ids:
            if not isinstance(node_id, str) or not node_id.strip():
                raise ValueError("node_ids must contain only non-empty strings")
            cleaned_ids.append(node_id.strip())

        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        rel_clause = ""
        if relationship_types:
            # Backtick-escape user-supplied type names, matching the existing
            # :meth:`find_neighbors` convention. These are schema identifiers,
            # used as literal relationship-type labels (Cypher does not allow
            # relationship types as bind parameters).
            valid_types = [
                t.strip().replace("`", "")
                for t in relationship_types
                if isinstance(t, str) and t.strip()
            ]
            if valid_types:
                rel_pattern = ":" + "|".join(f"`{t}`" for t in valid_types)
                rel_clause = f"[r{rel_pattern}]"
            else:
                rel_clause = "[r]"
        else:
            rel_clause = "[r]"

        query = (
            "MATCH (n) WHERE n.id IN $node_ids "
            f"MATCH (n)-{rel_clause}->(m) "
            "WHERE NOT m.id IN $node_ids "
            "RETURN DISTINCT m AS node, type(r) AS rel_type, n.id AS source_id "
            "LIMIT $limit"
        )

        parameters = {"node_ids": cleaned_ids, "limit": limit}

        return self.execute_read(query, parameters)

    def find_entity_candidates(
        self,
        search_text: str,
        labels: Optional[list[str]] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search Neo4j graph nodes for potential entity resolution candidates.

        This method executes parameterized Cypher queries matching node ``name``
        or ``aliases`` attributes case-insensitively.

        Args:
            search_text: Text string to search against node names and aliases.
            labels: Optional list of node labels to filter (e.g. ``["Port", "Supplier"]``).
            limit: Maximum candidate records to return.

        Returns:
            List of raw candidate dictionaries from the database.

        Raises:
            ValueError: If ``search_text`` is empty or invalid.
            ServiceUnavailable: If Neo4j is unreachable.
        """
        if not search_text or not isinstance(search_text, str):
            raise ValueError("search_text must be a non-empty string")

        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        cleaned_search = search_text.strip()
        if not cleaned_search:
            raise ValueError("search_text must not be whitespace-only")

        label_clause = ""
        if labels and isinstance(labels, list):
            valid_labels = [l.strip() for l in labels if l and isinstance(l, str) and l.strip()]
            if valid_labels:
                formatted_labels = " OR ".join(f"n:`{lbl}`" for lbl in valid_labels)
                label_clause = f"({formatted_labels}) AND "

        query = (
            f"MATCH (n) WHERE {label_clause}"
            "(toLower(n.name) CONTAINS toLower($search_text) "
            "OR ANY(alias IN coalesce(n.aliases, []) WHERE toLower(alias) CONTAINS toLower($search_text)) "
            "OR toLower(n.id) = toLower($search_text)) "
            "RETURN n LIMIT $limit"
        )

        parameters = {
            "search_text": cleaned_search,
            "limit": limit,
        }

        return self.execute_read(query, parameters)

    def run_graph_query(
        self,
        query: str,
        parameters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Execute an arbitrary read query against the graph.

        This is an internal utility; unrestricted Cypher must NOT be exposed
        as a public API endpoint.

        Args:
            query: Parameterized Cypher query.
            parameters: Query parameters.

        Returns:
            List of result dictionaries.
        """
        return self.execute_read(query, parameters)

    def _record_to_dict(self, record: Record) -> dict[str, Any]:
        """Convert a Neo4j record to a plain dictionary.

        Args:
            record: A raw Neo4j Record.

        Returns:
            Dictionary representation of the record values.
        """
        return dict(record.items())

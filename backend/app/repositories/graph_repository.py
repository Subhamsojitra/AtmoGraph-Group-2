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

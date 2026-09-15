"""Raw graph extraction for the GNN dataset pipeline (Module 11).

Converts the existing Neo4j graph into plain, typed Python structures using
the EXISTING :class:`~app.repositories.graph_repository.GraphRepository` and
the existing Neo4j connection. No second driver and no second repository are
created.

Extraction is strictly read-only. Every Cypher statement used here is
parameterized: node/relationship filtering values are bound as query
parameters (relationship *types* cannot be bind parameters in Cypher, so they
are backtick-quoted identifiers sanitized exactly like the existing
``find_downstream_neighbors`` convention).

Direction policy
----------------
Relationships are extracted exactly as stored in Neo4j:
``(source)-[rel]->(target)`` means "target depends on / consumes from source",
matching the Module 10 risk-propagation convention where risk flows along
OUTGOING relationships. Nothing is reversed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from app.core.logger import get_logger
from app.ml.exceptions import GraphDatasetValidationError
from app.repositories.graph_repository import GraphRepository

logger = get_logger(__name__)

#: Upper bound on extracted nodes; protects against accidentally materializing
#: an unbounded graph in memory. Generous by design and configurable per build.
DEFAULT_NODE_LIMIT = 10_000

#: Upper bound on extracted relationships (see ``DEFAULT_NODE_LIMIT``).
DEFAULT_EDGE_LIMIT = 50_000


@dataclass(frozen=True)
class RawNode:
    """A single graph node as needed for feature preparation."""

    id: str
    labels: tuple[str, ...]
    properties: Mapping[str, Any]

    @property
    def primary_label(self) -> str:
        """Deterministically chosen first label (empty when unlabeled)."""
        return sorted(self.labels)[0] if self.labels else ""


@dataclass(frozen=True)
class RawEdge:
    """A single directed relationship ``(source)-[rel_type]->(target)``."""

    source_id: str
    target_id: str
    rel_type: str


@dataclass(frozen=True)
class RawGraph:
    """Validated raw snapshot of the supply-chain graph.

    ``node_id_to_index`` assigns deterministic contiguous integer indices to
    nodes: node ids are sorted lexicographically so two extractions of the
    same logical graph always produce identical mappings regardless of Neo4j
    row order. Original Neo4j ids are preserved as the mapping keys.
    """

    nodes: tuple[RawNode, ...]
    edges: tuple[RawEdge, ...]
    node_id_to_index: Mapping[str, int]


def _unwrap_record_value(record: Mapping[str, Any]) -> Any:
    """Return the inner node object from a repository record row.

    ``GraphRepository.get_nodes`` returns rows shaped ``{"n": <node>}``. Mocked
    or simplified repositories may return other shapes; this helper finds the
    single meaningful value while leaving already-unwrapped data untouched
    (same approach as ``RiskPropagationService._unwrap_node``).
    """
    if isinstance(record, dict) and len(record) == 1:
        only_value = next(iter(record.values()))
        return only_value
    # Prefer conventional keys before falling back to the record itself.
    for key in ("n", "node"):
        if key in record and record[key] is not None:
            return record[key]
    return record


def _record_to_node(record: Any) -> RawNode:
    """Normalize one repository record into a :class:`RawNode`.

    Supports both real Neo4j ``Node`` objects (``items()``/``labels``) and
    plain dictionaries as returned by unit-test mocks.
    """
    value = _unwrap_record_value(record)

    if hasattr(value, "labels") and hasattr(value, "items"):
        props = dict(value.items())
        labels = tuple(sorted(str(lbl) for lbl in value.labels))
        node_id = props.get("id", "")
        return RawNode(id=str(node_id), labels=labels, properties=props)

    if isinstance(value, dict):
        if "id" in value and isinstance(value.get("properties"), Mapping):
            # Already normalized shape (id/labels/properties).
            return RawNode(
                id=str(value.get("id", "")),
                labels=tuple(sorted(str(l) for l in value.get("labels", ()))),
                properties=dict(value["properties"]),
            )
        # Bare property dictionary (common with mocked repositories).
        node_id = str(value.get("id", ""))
        labels = value.get("labels") or ()
        if isinstance(labels, str):
            labels = (labels,)
        return RawNode(
            id=node_id,
            labels=tuple(sorted(str(l) for l in labels)),
            properties=dict(value),
        )

    raise GraphDatasetValidationError(
        f"Unsupported node record type '{type(value).__name__}': "
        "cannot extract id/properties"
    )


def extract_graph(
    repository: GraphRepository,
    node_limit: int = DEFAULT_NODE_LIMIT,
    edge_limit: int = DEFAULT_EDGE_LIMIT,
    relationship_types: Optional[list[str]] = None,
) -> RawGraph:
    """Extract and validate a full raw snapshot of the graph.

    Args:
        repository: The existing :class:`GraphRepository` (reused, not wrapped).
        node_limit: Maximum number of nodes to extract.
        edge_limit: Maximum number of relationships to extract.
        relationship_types: Optional relationship-type filter forwarded to the
            parameterized Cypher query. When omitted, ALL outgoing
            relationship types are traversed, mirroring Module 10 semantics.

    Returns:
        A validated :class:`RawGraph`.

    Raises:
        GraphDatasetValidationError: When nodes/edges violate integrity rules:
            blank ids, duplicate ids, edges referencing unknown endpoints, or
            unusable record shapes. Invalid graphs must fail loudly instead of
            silently producing corrupt datasets.
    """
    if not isinstance(node_limit, int) or node_limit <= 0:
        raise ValueError("node_limit must be a positive integer")
    if not isinstance(edge_limit, int) or edge_limit <= 0:
        raise ValueError("edge_limit must be a positive integer")

    # --- Nodes ---------------------------------------------------------- #
    node_records = repository.get_nodes(limit=node_limit)
    nodes_by_id: dict[str, RawNode] = {}
    duplicates: list[str] = []
    blank_count = 0

    for record in node_records:
        try:
            node = _record_to_node(record)
        except GraphDatasetValidationError:
            logger.warning("Skipping malformed node record during extraction")
            blank_count += 1
            continue
        if not node.id.strip():
            blank_count += 1
            continue
        if node.id in nodes_by_id:
            if node.id not in duplicates:
                duplicates.append(node.id)
            continue
        nodes_by_id[node.id] = node

    if blank_count:
        raise GraphDatasetValidationError(
            f"{blank_count} node record(s) had a missing/blank 'id' property; "
            "every GNN dataset node requires a stable Neo4j identifier"
        )
    if duplicates:
        sample = ", ".join(repr(d) for d in sorted(duplicates)[:5])
        raise GraphDatasetValidationError(
            f"{len(duplicates)} duplicate node id(s) detected "
            f"(sample: {sample}); node ids must be unique to build an "
            "index mapping"
        )

    ordered_ids = sorted(nodes_by_id)
    node_id_to_index = {node_id: idx for idx, node_id in enumerate(ordered_ids)}
    nodes = tuple(nodes_by_id[node_id] for node_id in ordered_ids)

    # --- Edges ---------------------------------------------------------- #
    edge_records = repository.find_all_relationships(
        relationship_types=(
            list(relationship_types) if relationship_types else None
        ),
        limit=edge_limit,
    )

    edges: list[RawEdge] = []
    unknown_endpoints: set[str] = set()
    self_loop_count = 0

    for record in edge_records:
        source_id = str(record.get("source_id", ""))
        target_id = str(record.get("target_id", ""))
        rel_type = str(record.get("rel_type", ""))

        if not source_id or not target_id:
            raise GraphDatasetValidationError(
                "Relationship record with missing endpoint id encountered; "
                "edges without both endpoints cannot be indexed"
            )
        unknown = [
            eid for eid in (source_id, target_id) if eid not in node_id_to_index
        ]
        if unknown:
            unknown_endpoints.update(unknown)
            continue
        if source_id == target_id:
            self_loop_count += 1
        edges.append(RawEdge(source_id, target_id, rel_type))

    if unknown_endpoints:
        sample = ", ".join(repr(e) for e in sorted(unknown_endpoints)[:5])
        raise GraphDatasetValidationError(
            f"{len(unknown_endpoints)} edge endpoint(s) reference unknown nodes "
            f"(sample: {sample}); this indicates a truncated/inconsistent read - "
            "raise node_limit/edge_limit rather than dropping references"
        )

    # Deterministic ordering: index-space sort keeps the dataset reproducible.
    edges.sort(key=lambda e: (
        node_id_to_index[e.source_id],
        node_id_to_index[e.target_id],
        e.rel_type,
    ))

    logger.info(
        "Raw graph extraction complete",
        extra={
            "nodes": len(nodes),
            "edges": len(edges),
            "self_loops": self_loop_count,
            "relationship_filter": relationship_types or None,
        },
    )

    return RawGraph(
        nodes=nodes,
        edges=tuple(edges),
        node_id_to_index=node_id_to_index,
    )


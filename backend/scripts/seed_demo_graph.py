"""Seed a clearly-marked [DEMO/SYNTHETIC] supply-chain graph into Neo4j.

Offline OPERATIONAL tooling for the GNN prediction pipeline (Modules 13/14/16).
This script is NOT part of the production serving path: it only creates demo
data so the existing Module 11 -> 13 -> 14 pipeline can be trained and the
Module 16 WebSocket smoke test can be exercised end to end.

What it creates (idempotent; only touches its own ``DemoEntity`` nodes):

* 12 ``(:DemoEntity)`` nodes in 3 supply-chain tiers (``id``, ``name``,
  ``tier``, ``risk_score``, ``risk_level``, ``downstream_delay_days`` and a
  ``demo_synthetic: true`` marker).
* 14 ``[:SUPPLIES]`` edges following the project-wide direction convention:
  ``(source)-[:SUPPLIES]->(target)`` means *target consumes from source*.

``downstream_delay_days`` values are SYNTHETIC (deterministic, correlated with
``risk_score`` and in-degree so the GNN has a learnable pattern). They are NOT
real-world measurements and no predictive accuracy is claimed anywhere.

Every node carries a finite non-negative numeric target, which the Module 11
builder requires for labeled (trainable) datasets — the builder itself never
fabricates labels. Deleting nodes: only ``(:DemoEntity)`` nodes are removed,
never user data (use ``--wipe``).

Usage (from ``backend/``)::

    python scripts/seed_demo_graph.py           # (re)create the demo graph
    python scripts/seed_demo_graph.py --wipe    # remove only the demo nodes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow direct execution (``python scripts/seed_demo_graph.py`` from backend/)
# by putting the backend directory on sys.path so ``app`` is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logger import get_logger  # noqa: E402
from app.database.neo4j import neo4j_db  # noqa: E402
from app.repositories.graph_repository import GraphRepository  # noqa: E402
from app.services.risk.risk_service import RiskService  # noqa: E402

logger = get_logger("scripts.seed_demo_graph")

DEMO_LABEL = "DemoEntity"
DEMO_REL_TYPE = "SUPPLIES"
DEMO_TARGET_PROPERTY = "downstream_delay_days"

# ---------------------------------------------------------------------------
# Deterministic demo content (tier 1 -> tier 2 -> tier 3).
# Edges follow the project convention: (source)-[:SUPPLIES]->(target) means
# the TARGET consumes from the SOURCE.
# ---------------------------------------------------------------------------
_SUPPLIERS = [
    ("demo-sup-001", "Pacific Raw Metals", 1, 12.0),
    ("demo-sup-002", "Baltic Polymer Works", 1, 47.0),
    ("demo-sup-003", "Atlas Electronics Co.", 1, 78.0),
]
_MANUFACTURERS = [
    ("demo-man-001", "Nordwind Assembly", 2, 33.0),
    ("demo-man-002", "Meridian Components", 2, 61.0),
    ("demo-man-003", "Solstice Fabrication", 2, 24.0),
    ("demo-man-004", "Vertex Systems", 2, 88.0),
]
_DISTRIBUTORS = [
    ("demo-dis-001", "Harbor Logistics", 3, 18.0),
    ("demo-dis-002", "Continent Retail Group", 3, 55.0),
    ("demo-dis-003", "Highway Distribution", 3, 71.0),
    ("demo-dis-004", "Lakeside Wholesale", 3, 39.0),
    ("demo-dis-005", "Gateway Export", 3, 92.0),
]
_EDGES = [
    # tier 1 -> tier 2
    ("demo-sup-001", "demo-man-001"),
    ("demo-sup-002", "demo-man-001"),
    ("demo-sup-002", "demo-man-002"),
    ("demo-sup-003", "demo-man-002"),
    ("demo-sup-001", "demo-man-003"),
    ("demo-sup-003", "demo-man-004"),
    # tier 2 -> tier 3
    ("demo-man-001", "demo-dis-001"),
    ("demo-man-003", "demo-dis-001"),
    ("demo-man-001", "demo-dis-002"),
    ("demo-man-002", "demo-dis-002"),
    ("demo-man-002", "demo-dis-003"),
    ("demo-man-004", "demo-dis-003"),
    ("demo-man-003", "demo-dis-004"),
    ("demo-man-004", "demo-dis-005"),
]

_DELETE_QUERY = f"MATCH (n:`{DEMO_LABEL}`) DETACH DELETE n"

_CREATE_NODES_QUERY = f"""
UNWIND $nodes AS row
CREATE (n:`{DEMO_LABEL}` {{
    id: row.id,
    name: row.name,
    tier: row.tier,
    risk_score: row.risk_score,
    risk_level: row.risk_level,
    {DEMO_TARGET_PROPERTY}: row.delay,
    demo_synthetic: true
}})
"""

_CREATE_EDGES_QUERY = f"""
UNWIND $edges AS row
MATCH (a:`{DEMO_LABEL}` {{id: row.source_id}})
MATCH (b:`{DEMO_LABEL}` {{id: row.target_id}})
CREATE (a)-[:`{DEMO_REL_TYPE}`]->(b)
"""

_COUNT_QUERY = f"MATCH (n:`{DEMO_LABEL}`) RETURN count(n) AS nodes"


def _build_demo_rows() -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    """Return the demo node rows and edge rows (deterministic, no randomness).

    The synthetic delay target is correlated with ``risk_score`` and the
    node's in-degree so the GNN has a real, learnable pattern. ``risk_level``
    is derived with the EXISTING Module 9 engine (no duplicated thresholds).
    """
    risk_service = RiskService()
    in_degree: dict[str, int] = {}
    for _, target_id in _EDGES:
        in_degree[target_id] = in_degree.get(target_id, 0) + 1

    nodes: list[dict[str, object]] = []
    for node_id, name, tier, score in (
        _SUPPLIERS + _MANUFACTURERS + _DISTRIBUTORS
    ):
        delay = round(0.5 + 0.05 * score + 0.4 * in_degree.get(node_id, 0), 2)
        nodes.append(
            {
                "id": node_id,
                "name": name,
                "tier": tier,
                "risk_score": score,
                "risk_level": risk_service.calculate_risk_level(score).upper(),
                DEMO_TARGET_PROPERTY: delay,
            }
        )

    edges = [
        {"source_id": source_id, "target_id": target_id}
        for source_id, target_id in _EDGES
    ]
    return nodes, edges


def wipe_demo_graph(repository: GraphRepository) -> int:
    """Delete ONLY the ``DemoEntity`` nodes created by this script."""
    repository.execute_write(_DELETE_QUERY)
    remaining = repository.execute_read(_COUNT_QUERY)
    count = int(remaining[0]["nodes"]) if remaining else 0
    print(f"Demo graph wiped; remaining {DEMO_LABEL} nodes: {count}")
    return count


def seed_demo_graph(repository: GraphRepository) -> int:
    """(Re)create the demo graph; returns the number of demo nodes."""
    repository.execute_write(_DELETE_QUERY)  # idempotent re-seed
    nodes, edges = _build_demo_rows()
    repository.execute_write(_CREATE_NODES_QUERY, {"nodes": nodes})
    repository.execute_write(_CREATE_EDGES_QUERY, {"edges": edges})
    remaining = repository.execute_read(_COUNT_QUERY)
    count = int(remaining[0]["nodes"]) if remaining else 0
    if count != len(nodes):
        raise RuntimeError(
            f"Demo seed verification failed: expected {len(nodes)} nodes, "
            f"found {count}"
        )
    print(
        f"Demo graph created: {count} '{DEMO_LABEL}' nodes, "
        f"{len(edges)} ':{DEMO_REL_TYPE}' edges, synthetic target property "
        f"'{DEMO_TARGET_PROPERTY}' on every node."
    )
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Seed (or wipe) the clearly-marked synthetic demo supply-chain "
            "graph used for GNN training and the WebSocket smoke test."
        )
    )
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="Delete only the DemoEntity nodes, then exit.",
    )
    args = parser.parse_args()

    neo4j_db.initialize()
    try:
        if not neo4j_db.verify_connectivity():
            print(
                "ERROR: Neo4j is not reachable. Start your Neo4j instance and "
                "make sure the NEO4J_* settings in the root .env are correct."
            )
            return 2
        repository = GraphRepository(neo4j_db)
        if args.wipe:
            wipe_demo_graph(repository)
        else:
            seed_demo_graph(repository)
            print(
                "Next step: python scripts/train_gnn.py   "
                f"(trains on this graph via '{DEMO_TARGET_PROPERTY}')"
            )
    finally:
        neo4j_db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

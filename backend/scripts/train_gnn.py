"""Offline Module 13 GNN training tooling for AtmoGraph.

Operational script (NOT part of the production serving path): it reuses the
EXISTING Modules 11-13 pipeline end to end and saves a checkpoint in the exact
format the Module 14 serving layer (``GNNPredictor`` /
``PredictionService``) loads::

    GraphDatasetBuilder.build(target_property=...)   (Module 11, live Neo4j)
        or a clearly-marked synthetic GraphDataset (--synthetic, no Neo4j)
        v
    GNNModel  (Module 12 architecture, unchanged)
        v
    GNNTrainer.train()  (Module 13: real regression training, val/test split)
        v
    GNNTrainer.save_checkpoint  ->  checkpoints/gnn_m13.pt  (gitignored)

Modes
-----
* default            — trains on the LIVE Neo4j graph. Requires a finite,
                       non-negative numeric ``--target`` property (default
                       ``downstream_delay_days``) on EVERY node; the Module 11
                       builder fails loudly otherwise and never fabricates
                       labels. Seed demo data with
                       ``python scripts/seed_demo_graph.py`` if your database
                       has no targets yet.
* ``--synthetic``    — trains on a deterministic [TEST/SYNTHETIC] dataset
                       built in-memory with the same GraphDataset schema. No
                       Neo4j needed. Exists so the serving path can be
                       exercised before real labels exist; NO predictive
                       accuracy is claimed for models trained this way.

Both modes run the SAME real training loop — only the data source differs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow direct execution (``python scripts/train_gnn.py`` from backend/) by
# putting the backend directory on sys.path so ``app`` is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from app.core.config import BACKEND_DIR  # noqa: E402
from app.ml.dataset import GraphDataset, GraphDatasetBuilder  # noqa: E402
from app.ml.features import NodeFeatureEncoder  # noqa: E402
from app.ml.model import GNNModel  # noqa: E402
from app.ml.training import GNNTrainingConfig, GNNTrainer  # noqa: E402

DEFAULT_CHECKPOINT = Path("checkpoints/gnn_m13.pt")
DEFAULT_TARGET = "downstream_delay_days"
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 300

_SYNTHETIC_NUM_NODES = 24
_SYNTHETIC_NOTE = (
    "*** [TEST/SYNTHETIC] TRAINING DATA ***\n"
    "Training on a deterministic in-memory synthetic dataset (no Neo4j).\n"
    "The checkpoint is a REAL trained Module 13 artifact, but because the\n"
    "project has no real downstream-delay labels, NO predictive accuracy is\n"
    "claimed. Retrain on the live graph (python scripts/train_gnn.py) once\n"
    "labeled data exists."
)


def build_synthetic_dataset(
    num_nodes: int = _SYNTHETIC_NUM_NODES,
) -> GraphDataset:
    """Build a clearly-marked synthetic labeled dataset (no Neo4j required).

    Same schema and validation as every Module 11 build: float32 feature
    matrix of the fixed encoder width, int64 edge index, finite non-negative
    regression target. The target is a deterministic linear function of the
    risk features and the node in-degree, so the GNN has a learnable signal.
    """
    feature_dim = len(NodeFeatureEncoder.FEATURE_NAMES)
    rng = np.random.default_rng(DEFAULT_SEED)

    x = rng.random((num_nodes, feature_dim)).astype(np.float32)

    # Layered supply-chain-like DAG: every node consumes from earlier nodes.
    sources: list[int] = []
    targets: list[int] = []
    for dst in range(1, num_nodes):
        sources.append(int(rng.integers(0, dst)))
        targets.append(dst)
        if rng.random() < 0.35:
            sources.append(int(rng.integers(0, dst)))
            targets.append(dst)
    edge_index = np.asarray([sources, targets], dtype=np.int64)
    rel_types = tuple(["SUPPLIES"] * len(sources))

    in_degree = np.bincount(edge_index[1], minlength=num_nodes).astype(
        np.float32
    )
    y = (0.5 + 4.0 * x[:, 0] + 2.0 * x[:, 1] + 0.75 * in_degree).astype(
        np.float32
    )

    return GraphDataset(
        node_ids=tuple(f"synthetic-node-{i:03d}" for i in range(num_nodes)),
        x=x,
        edge_index=edge_index,
        edge_rel_types=rel_types,
        y=y,
        target_name=DEFAULT_TARGET,
        metadata=None,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the existing AtmoGraph GNN (Modules 12/13) and save a "
            "serving checkpoint for the prediction service (Module 14/16)."
        )
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help=(
            "Train on the deterministic [TEST/SYNTHETIC] in-memory dataset "
            "instead of the live Neo4j graph (no Neo4j required)."
        ),
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help=(
            "Node property holding the regression target when training on "
            "the live graph (every node must carry it). "
            f"Default: {DEFAULT_TARGET}"
        ),
    )
    parser.add_argument(
        "--relationship-types",
        nargs="*",
        default=None,
        help=(
            "Optional relationship types representing supply-chain flow "
            "(Module 11 builder, e.g. SUPPLIES). Omitted = all relationships."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Training epochs (default: {DEFAULT_EPOCHS}).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="GNN hidden width (default: 64).",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of GCN layers (default: 2).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability (default: 0.1).",
    )
    parser.add_argument(
        "--loss",
        default="mse",
        help="Regression loss: mse, mae or huber (default: mse).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed for reproducible training (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_CHECKPOINT),
        help=(
            "Checkpoint output path. Relative paths resolve against the "
            f"backend directory ({BACKEND_DIR}). "
            f"Default: {DEFAULT_CHECKPOINT}"
        ),
    )
    return parser.parse_args()


def _resolve_output_path(raw: str) -> Path:
    """Make the checkpoint output path absolute (backend-dir relative)."""
    path = Path(raw)
    if not path.is_absolute():
        path = BACKEND_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_dataset(args: argparse.Namespace) -> GraphDataset:
    """Build the training dataset from Neo4j or the synthetic fallback."""
    if args.synthetic:
        print(_SYNTHETIC_NOTE)
        return build_synthetic_dataset()

    from app.database.neo4j import neo4j_db  # imported lazily: not needed for --synthetic

    neo4j_db.initialize()
    if not neo4j_db.verify_connectivity():
        print(
            "ERROR: Neo4j is not reachable. Start your Neo4j instance and "
            "check the NEO4J_* settings in the root .env, or use --synthetic."
        )
        raise SystemExit(2)
    try:
        dataset = GraphDatasetBuilder().build(
            target_property=args.target,
            relationship_types=(
                args.relationship_types if args.relationship_types else None
            ),
        )
    finally:
        neo4j_db.close()
    if dataset.y is None:
        raise SystemExit(
            f"ERROR: the live graph produced an UNLABELED dataset; training "
            f"needs a finite non-negative numeric '{args.target}' property on "
            "EVERY node (the builder never fabricates labels). Seed demo data "
            "with: python scripts/seed_demo_graph.py"
        )
    return dataset


def main() -> int:
    args = _parse_args()
    dataset = _load_dataset(args)
    output_path = _resolve_output_path(args.output)

    # Weight init happens at construction (before train() seeds its own RNG),
    # so seed torch here too for fully reproducible checkpoints.
    torch.manual_seed(args.seed)
    model = GNNModel(
        input_dim=dataset.num_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    config = GNNTrainingConfig(
        epochs=args.epochs,
        seed=args.seed,
        device="cpu",
        loss=args.loss,
        checkpoint_path=output_path,  # auto-saved after train() (Module 13)
    )

    print(
        f"Training GNNModel(input_dim={dataset.num_features}, "
        f"hidden_dim={args.hidden_dim}, num_layers={args.num_layers}) on "
        f"{dataset.num_nodes} nodes / {dataset.num_edges} edges "
        f"for {args.epochs} epochs on cpu..."
    )
    trainer = GNNTrainer(model, dataset, config)
    history = trainer.train()

    if trainer.split.test_indices.size > 0:
        test_loss, metrics = trainer.evaluate_test()
        formatted = ", ".join(
            f"{name}={value}" if value is not None else f"{name}=n/a"
            for name, value in metrics.items()
        )
        print(f"Test loss: {test_loss:.6f}")
        print(f"Test metrics: {formatted}")

    best = history.best_val_loss if history.best_val_loss is not None else "n/a"
    print(f"Best validation loss: {best}")
    print(f"Checkpoint saved: {output_path}")
    print(
        "Enable serving (root .env, relative paths resolve against backend/):\n"
        f"    PREDICTION_CHECKPOINT_PATH={output_path.relative_to(BACKEND_DIR).as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

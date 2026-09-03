# AtmoGraph-Group-2

## Modules

### Module 7 — NLP Preprocessing & Named Entity Recognition (NER)

FastAPI endpoint:

```
POST /api/v1/news/analyze
```

Accepts the same body shape as `POST /api/v1/news/ingest` (`title`, `content`,
optional `source`/`url`/`published_at`) plus an optional `article_id` from a
prior ingestion. The service:

1. Reuses the Module 6 `normalize_text` pipeline (no duplicated logic).
2. Runs the configured Hugging Face Transformers NER model.
3. Returns structured, occurrence-level entities with text, label
   (CoNLL-03: PER/ORG/LOC/MISC), character offsets, and model confidence.
4. Never touches Neo4j; it works even when `localhost:7687` is down.

Example:

```json
{
  "title": "Port strike disrupts European shipments",
  "content": "Workers at the Port of Rotterdam announced a strike affecting shipments from China to Germany.",
  "source": "Example News"
}
```

#### One-time model installation (required before using NER)

The NER model is **not** downloaded automatically by the app or by pytest.
Install `dslim/bert-base-NER` into the local HuggingFace cache once with:

```bash
python -c "from transformers import pipeline; \
  pipeline('token-classification', model='dslim/bert-base-NER', \
           aggregation_strategy='simple', device='cpu')"
```

After the first run the model is cached offline. Real-model tests skip (they
never download) when the cache is empty.

#### NLP settings (all optional, see `.env.example`)

| Variable | Default | Purpose |
| --- | --- | --- |
| `NLP_MODEL_NAME` | `dslim/bert-base-NER` | HuggingFace model id |
| `NLP_NER_MAX_LENGTH` | `512` | Max tokens per input (truncation) |
| `NLP_MAX_TEXT_LENGTH` | `10000` | Max characters accepted per text |
| `NLP_DEVICE` | `cpu` | torch device (`cpu`/`cuda`) |
| `NLP_USE_FAST_TOKENIZER` | `true` | Prefer the fast tokenizer |

### Module 9 — Risk State Update

Takes an entity already resolved by Module 8 and updates its **risk state** in
Neo4j. Module 9 never performs entity recognition or entity resolution again and
never creates new nodes.

FastAPI endpoint:

```
POST /api/v1/news/risk-update
```

Request body (built from a Module 8 `node_id`):

```json
{
  "entity_id": "entity-001",
  "entity_name": "Port of Rotterdam",
  "risk_score": 65,
  "reason": "Port disruption"
}
```

Architecture:

```
Resolved Entity (Module 8)
        ↓
Risk Update Request
        ↓
RiskService
        ↓
GraphRepository (existing)
        ↓
Neo4j Database (existing)
        ↓
Update existing entity node (MATCH ... SET n.risk_score, n.risk_level)
```

Behaviour:

- Risk score scale is **0–100** (0 = lowest, 100 = highest). Out-of-range values
  are rejected with a `422`; they are never silently clamped.
- Risk levels are derived from the score: **LOW / MEDIUM / HIGH / CRITICAL**.
- The update is **atomic** and **parameterized** (a single transactional Cypher
  statement reads the previous state and writes the new one, so there is no
  read-then-write race and no string interpolation of user input).
- Unresolved entities (no `node_id`/blank id) return a controlled
  `updated: false` response without touching the database.
- A nonexistent node id returns `404`; Module 9 never creates nodes.
- Neo4j being unavailable returns `503`.

Error mapping (all without leaking internals):

| Outcome | HTTP |
| --- | --- |
| Success | `200` + `updated: true` |
| Unresolved entity | `200` + `updated: false` |
| Invalid risk score | `422` |
| Entity node not found | `404` |
| Neo4j unavailable | `503` |
| Unexpected error | `500` |

Example response:

```json
{
  "entity_id": "entity-001",
  "entity_name": "Port of Rotterdam",
  "updated": true,
  "previous_risk_score": 30,
  "new_risk_score": 65,
  "previous_risk_level": "LOW",
  "new_risk_level": "MEDIUM",
  "reason": "Port disruption",
  "timestamp": "2026-01-01T00:00:00Z"
}
```

#### Risk thresholds (configurable assumption)

The project does not yet specify official risk-level thresholds, so they are a
**configurable assumption** (see `backend/app/core/config.py` / `.env.example`):

```
LOW      [0, 30]
MEDIUM   [31, 70]
HIGH     [71, 90]
CRITICAL [91, 100]
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `RISK_SCORE_MIN` | `0` | Lowest accepted score |
| `RISK_SCORE_MAX` | `100` | Highest accepted score |
| `RISK_LEVEL_MEDIUM` | `31` | Score at which the level becomes MEDIUM |
| `RISK_LEVEL_HIGH` | `71` | Score at which the level becomes HIGH |
| `RISK_LEVEL_CRITICAL` | `91` | Score at which the level becomes CRITICAL |

These are *not* official specification values; they are a clearly-documented
assumption that can be tuned without code changes.

#### Tests

```
python -m pytest tests/test_risk_service.py -v   # unit (mocked repository)
python -m pytest tests/test_risk_api.py -v       # API (mocked service)
python -m pytest tests/test_risk_neo4j.py -v     # real Neo4j integration (skipped when down)
```

### Module 10 — Risk Propagation / Ripple Effect

Starts from a resolved entity that already carries a risk score (set by
Module 9) and propagates that risk **downstream** through the supply-chain
graph to the entities that depend on it. Module 10 never performs entity
recognition, entity resolution, or risk state updates, and it never creates or
modifies nodes (it only reads the graph).

FastAPI endpoint:

```
POST /api/v1/risk-propagation/propagate
```

Request body (built from a Module 8 `node_id` + its Module 9 `risk_score`):

```json
{
  "entity_id": "entity-001",
  "entity_name": "Port of Rotterdam",
  "risk_score": 80,
  "max_depth": 3,
  "attenuation": 0.5,
  "relationship_types": ["SUPPLIES"]
}
```

Only `entity_id` (blank = unresolved) and `risk_score` are meaningful
requirements; the rest are optional and fall back to configuration defaults.

Architecture:

```
Resolved Entity (Module 8)
        ↓
Risk State Update (Module 9) -> risk_score stored on the node
        ↓
Risk Propagation Request
        ↓
RiskPropagationService
        ↓
GraphRepository (existing) | RiskService.calculate_risk_level (existing)
        ↓
Neo4j Database (existing)
        ↓
Affected entities + propagated risk/impact
```

Behaviour:

- Traversal is **directed downstream** (outgoing relationships `(n)-[r]->(m)`)
  and **breadth-first with a configurable depth limit**.
- **Cycle prevention**: a visited set ensures each node is reported once, at
  its shallowest depth, and the depth bound prevents infinite traversal.
- The propagated risk of an affected entity at depth `d` is
  `source_score * attenuation ** d`, clamped to the 0-100 scale. If the source
  node already carries a persisted `risk_score` (from Module 9) it is used in
  preference to the request value.
- Risk levels are derived exactly as in Module 9 by reusing
  `RiskService.calculate_risk_level` — no second risk engine is introduced.
- Unresolved entities (blank `node_id`) return a controlled
  `propagated: false` response without touching the database.
- A nonexistent source node returns `404`; Module 10 never creates nodes.
- Neo4j being unavailable returns `503`.
- All Cypher is **parameterized** (user-supplied values are bound, never
  interpolated).

Error mapping (all without leaking internals):

| Outcome | HTTP |
| --- | --- |
| Success | `200` + `propagated: true` |
| Unresolved entity | `200` + `propagated: false` |
| Invalid risk score / depth / attenuation | `422` |
| Source entity node not found | `404` |
| Neo4j unavailable | `503` |
| Unexpected error | `500` |

Example response:

```json
{
  "source_entity_id": "entity-001",
  "source_entity_name": "Port of Rotterdam",
  "source_risk_score": 80.0,
  "propagated": true,
  "affected_entities": [
    {
      "entity_id": "entity-002",
      "entity_name": "Gigafactory Assembly",
      "depth": 1,
      "propagated_risk_score": 40.0,
      "propagated_risk_level": "MEDIUM"
    }
  ],
  "affected_count": 1,
  "max_depth_reached": 1,
  "timestamp": "2026-01-01T00:00:00Z",
  "error": null
}
```

#### Propagation settings (all optional, see `.env.example`)

| Variable | Default | Purpose |
| --- | --- | --- |
| `RISK_PROPAGATION_MAX_DEPTH` | `5` | Default maximum traversal depth (hops) |
| `RISK_PROPAGATION_ATTENUATION` | `0.5` | Default per-hop risk attenuation factor |
| `RISK_PROPAGATION_MAX_AFFECTED` | `500` | Upper bound on reported affected entities |

These are *not* official specification values; they are a clearly-documented
assumption (like the Module 9 thresholds) that can be tuned without code
changes.

#### Tests

```
python -m pytest tests/test_risk_propagation_service.py -v  # unit (mocked repository/service)
python -m pytest tests/test_risk_propagation_api.py -v      # API (mocked service)
python -m pytest tests/test_risk_propagation_neo4j.py -v    # real Neo4j integration (skipped when down)
```

### Module 11 — Graph Data Preparation for GNN

Module 11 turns the existing Neo4j supply-chain graph into a numerical,
GNN-ready dataset. It implements **data preparation only** — no model,
no training, no evaluation (those belong to Module 12+).

Architecture (reuses every existing layer; no second driver/repository):

```
Existing Neo4j graph
        ↓  GraphRepository.get_nodes / find_all_relationships (parameterized Cypher)
app.ml.extraction       RawGraph + deterministic node-id → index mapping
        ↓
app.ml.features         NodeFeatureEncoder (fit/transform, stored parameters)
        ↓
app.ml.dataset          GraphDatasetBuilder → GraphDataset
        ↓              (x float32 [N,F], edge_index int64 [2,E], y float32 [N])
Module 12 — GNN model   (GraphDataset.to_pyg_data() when PyG is installed)
```

Discovered graph schema used by Module 11:

| Item | Value |
| --- | --- |
| Node identifier | string property `id` (unique; validated, never guessed) |
| Node properties | `id`, `name`, `aliases`, `risk_score`, `risk_level` |
| Relationships | schema-neutral; supply-chain flow = outgoing `(a)-[r]->(b)` |
| Canonical example type | `SUPPLIES` (same convention as Module 10) |

INPUT FEATURES vs TARGET (leakage policy):

| feature | source / encoding |
| --- | --- |
| `risk_score_norm` | Module 9 `risk_score`, min-max rescaled by config bounds |
| `risk_level_code` | ordinal LOW=0 < MEDIUM=1 < HIGH=2 < CRITICAL=3 |
| `label_code` | primary Neo4j label integer-encoded over a sorted vocabulary (index 0 = unseen) |
| `out_degree_norm` / `in_degree_norm` | node degree scaled by fitted maximum |
| **target** (`y`) | OPTIONAL, supplied via `build(target_property=...)`; **the database currently contains NO real target values**, so builds default to an unlabeled dataset (`y is None`). Synthetic labels exist ONLY in test fixtures. A labeled build requires a finite non-negative numeric property on EVERY node and hard-rejects using any feature name as the target. |

Validation is strict (fail loudly, never corrupt): blank/duplicate node ids,
dangling edges, out-of-range scores, NaN/inf values, missing targets,
inconsistent shapes and empty graphs all raise `GraphDatasetError` subclasses.

Dataset persistence uses `.npz` with `allow_pickle=False` both ways
(`GraphDataset.to_npz` / `GraphDataset.load_npz`), carrying arrays plus the
fitted `FeatureMetadata` JSON so inference reuses identical encoder parameters.
Node indices are assigned over lexicographically sorted ids, making repeated
builds byte-for-byte reproducible regardless of Neo4j row order.

PyTorch Geometric is deliberately NOT in requirements.txt yet (see comments
there); `GraphDataset.to_pyg_data()` raises a helpful `ImportError` until the
Module 12 developer installs matching wheels for the installed torch build.

Tests:

```
python -m pytest tests/test_gnn_dataset.py -q        # unit (mocked repository) - no Neo4j needed
python -m pytest tests/test_gnn_dataset_neo4j.py -q  # real Neo4j integration (skipped when down)
```
### Module 12 - GNN Model Architecture (node-level delay regression)

Module 12 adds the GNN architecture that consumes the Module 11 dataset. It
implements the MODEL ONLY - no training loop, no optimizer, no evaluation and
no prediction API (Modules 13+). The model is UNTRAINED: it must not be
presented as predicting real delays yet.

Architecture (node-level regression, one prediction per graph node):

```
node features x [N, F]                    (Module 11 NodeFeatureEncoder)
        v
GCNConv(F -> hidden_dim) + ReLU + Dropout
        v   (repeated num_layers times; message passing along the
        v    Module 11 edge direction: upstream -> downstream)
node embeddings [N, hidden_dim]
        v
Linear regression head (hidden_dim -> output_dim)
        v
predicted downstream delay per node [N]   (output_dim == 1, default)
```

Why GCN (Kipf & Welling 2017):

- Simple, well-supported and adequate for a first node-regression model; runs
  on CPU with the pure-Python `torch_geometric` core (the compiled
  `torch-scatter`/`torch-sparse` companion wheels are NOT required).
- Message passing flows source -> target along `edge_index` exactly as stored
  by Module 11 (`(source)-[rel]->(target)` = "target consumes from source"),
  so upstream disruption features propagate toward downstream nodes - the
  supply-chain ripple-effect task. Edges are never reversed; Neo4j
  relationship semantics are untouched.
- GCNConv adds self-loops internally, so isolated nodes keep their own
  features instead of receiving an all-zero aggregate.

Usage:

```python
from app.ml.model import GNNModel

data = dataset.to_pyg_data()            # Module 11 export (PyG now required)
model = GNNModel(input_dim=dataset.num_features)  # derive dim from Module 11
predictions = model(data.x, data.edge_index)      # shape [num_nodes]
```

Contract:

| Aspect | Behaviour |
| --- | --- |
| Output | `[num_nodes]` when `output_dim == 1` (matches Module 11 target layout `y: float32 [num_nodes]`), otherwise `[num_nodes, output_dim]`. Exactly one prediction per node - no graph-level pooling. |
| Target leakage | Impossible by construction: `forward(x, edge_index)` has no target parameter; `y` stays reserved for the Module 13 loss. |
| Configuration | `GNNModel(input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.1)` or `GNNModel.from_config(GNNConfig(...))`. All values are validated; violations raise `GNNModelConfigError`. Forward-pass violations (wrong width/dtype/NaN/out-of-bounds indices) raise `GNNModelInputError`. |
| Persistence | `model.save_state(path)` / `GNNModel.load_state(path)` - a plain `{config, state_dict}` checkpoint (tensors + primitives only, loaded with `weights_only=True`; no arbitrary object deserialization). |
| Device | Plain `nn.Module`: CPU required and default; `model.to("cuda")` works when a GPU exists. No GPU, no downloads and no internet needed. |
| Determinism | Construction is seeded by the caller (`torch.manual_seed`); eval-mode forwards are deterministic. Dropout applies in train mode only. |

STATUS - UNTRAINED: weights are randomly initialized at construction. Module 12
proves the architecture, not accuracy: no accuracy numbers exist yet, and none
may be claimed until Module 13 trains and evaluates the model.

Install note: `pip install torch_geometric` (pure-Python wheel). It must be
compatible with the installed torch build; `app.ml` now imports PyG, so
Module 11's `GraphDataset.to_pyg_data()` export works out of the box.

Tests:

```
python -m pytest tests/test_gnn_model.py -q   # architecture unit tests (synthetic, no Neo4j)
```

### Module 13 — GNN Training & Evaluation

Module 13 adds the training/evaluation layer AROUND the existing modules. It
trains the Module 12 `GNNModel` to predict downstream delays
(**node-level regression**) from upstream disruption features, using the
Module 11 `GraphDataset` unchanged. It contains NO HTTP endpoint, NO model
serving and NO real-time inference — those belong to later modules, which
will consume `GNNTrainer.model` or a saved checkpoint.

Architecture (reuses Modules 11 & 12 unchanged; no duplicated model/dataset code):

```
Module 11  GraphDataset (x float32 [N,F], edge_index int64 [2,E], y float32 [N])
        |   app.ml.dataset (unchanged)  ->  to_pyg_data() tensors
        v
Module 12  GNNModel(x, edge_index)  ->  [N] predictions (unchanged)
        v
Module 13  GNNTrainer                    (app.ml.training)
        |   split_nodes() -> NodeSplit   (app.ml.splitting, seeded)
        |   Adam + regression loss on TRAIN nodes only
        v
Validation (eval mode, torch.no_grad, per eval_interval)
        v
TrainingHistory + metrics                (app.ml.evaluation)
        v
state_dict checkpoints (GNNCheckpoint)   ->  later prediction module
```

New files: `app/ml/splitting.py` (deterministic node split),
`app/ml/evaluation.py` (regression metrics), `app/ml/training.py`
(`GNNTrainingConfig`, `TrainingHistory`, `GNNCheckpoint`, `GNNTrainer`,
`set_seed`). No new pip dependencies; no Neo4j schema change; no API change.

Usage:

```python
from app.ml.training import GNNTrainer, GNNTrainingConfig
from app.ml.model import GNNModel

dataset = GraphDatasetBuilder(repository).build(target_property="downstream_delay_days")
model = GNNModel(input_dim=dataset.num_features, hidden_dim=64, num_layers=2, dropout=0.1)
trainer = GNNTrainer(model, dataset, GNNTrainingConfig(epochs=200, seed=42))
history = trainer.train()            # structured per-epoch history
loss, metrics = trainer.evaluate_test()
trainer.save_checkpoint("checkpoints/m13.pt")
```


Training objective — node-level regression only
------------------------------------------------

For N graph nodes the model produces N predictions and the loss compares them
against ground-truth per-node delay targets on the TRAIN nodes only. There is
no classification, no HIGH/MEDIUM/LOW classes, and no graph-level pooling.
Targets are NEVER model inputs: `GNNModel.forward(x, edge_index)` has no
target parameter (Module 12 anti-leakage contract) and the trainer only ever
calls `model(x, edge_index)`.

Target validation is strict (fail loudly, never fabricate): an unlabeled
dataset (`y is None`), an empty graph, a non-finite target or a model with
`output_dim != 1` aborts trainer construction with `GNNTrainingError`
subclasses. The database currently contains NO real delay labels, so default
builds are unlabeled and the trainer rejects them — synthetic labels exist
only in clearly-marked test fixtures.

Train/validation/test strategy
------------------------------

`split_nodes(num_nodes, validation_split=0.2, test_split=0.1, seed=...)`
partitions NODE indices (transductive setup — the standard Kipf & Welling
scheme for one graph). Why this is safe here:

* Module 11 yields ONE connected supply-chain graph and Module 12 consumes
  one `(x, edge_index)` pair for the whole graph.
* Validation/test targets are used ONLY by their own loss/metrics — never by
  training and never as features.
* Module 11 guarantees targets cannot be features
  (`_assert_no_target_leakage`), so messages passed along edges carry
  current-state features only; validation/test labels cannot leak into
  training through message passing.
* An inductive subgraph split (dropping edges) is intentionally NOT used: it
  would change the very message-passing structure the model must learn on.

The split is deterministic for a given seed (a locally seeded
`numpy.random.Generator`; the global `random`/`numpy.random` state is never
touched), the three counts always sum to `num_nodes` (no silent data loss),
and a dataset too small for the requested split is rejected explicitly.
`validation_split` must be > 0 (training always validates); `test_split`
may be 0.0 to skip the held-out test evaluation.


Loss, optimizer, metrics, history
---------------------------------

| Aspect | Choice | Rationale |
| --- | --- | --- |
| Loss | `MSELoss` (default; configurable `loss="mse"`, `"mae"` or `"huber"`) | Standard differentiable regression loss; no custom loss invented because no specification requires one. |
| Optimizer | `Adam` (`learning_rate=0.01`, `weight_decay=5e-4` defaults) | Standard, well-supported; both values configurable (defaults are the conventional GCN values of Kipf & Welling 2017). |
| Metrics | `{"mse", "rmse", "mae", "r2"}` per split | `r2` is `None` (not NaN) when undefined — single sample or zero target variance. Empty inputs, shape mismatches and NaN/inf values raise instead of producing silent garbage. |
| History | `TrainingHistory` (`train_loss`, `val_loss`, `train_metrics`, `val_metrics`, `stopped_early`, `best_epoch`, `best_val_loss`) | Structured and JSON-safe; `val_*[i]` is `None` for epochs where validation was skipped (`eval_interval`). Training stops loudly if the loss becomes non-finite. |

Validation runs after every epoch (or every `eval_interval`-th epoch; the
final epoch is always validated) in `model.eval()` + `torch.no_grad()` —
parameters are never updated during validation.

Early stopping, checkpointing, reproducibility
----------------------------------------------

* **Early stopping** (optional): `patience` (in validation rounds, `None` =
  disabled) monitors the validation loss and stops when it stops improving;
  `restore_best=True` (default) reloads the best-validation-loss weights
  afterwards.
* **Checkpoints**: `GNNTrainer.save_checkpoint(path)` writes a plain
  `torch.save` mapping — `model_state_dict`, `optimizer_state_dict`, `epoch`,
  `best_val_loss`, both configs, last validation metrics and the history —
  and `GNNTrainer.load_checkpoint(path, model=..., optimizer=...)` restores
  it with `torch.load(..., weights_only=True)`. No arbitrary object pickling
  (same policy as Module 12). Trained `.pt` binaries must NOT be committed
  to Git. `GNNTrainingConfig(checkpoint_path=...)` auto-saves after `train()`.
* **Reproducibility**: with `GNNTrainingConfig(seed=...)` the trainer seeds
  torch's global RNG at the start of `train()` (dropout draws from it) and
  the node split uses its own locally seeded numpy Generator. Same seed +
  same dataset => identical loss histories on the same machine/build
  (verified by tests). No application-global RNG state is modified otherwise.

CPU-first: the development environment is CPU-only and CPU training is fully
supported and tested; `device="cuda"` works when a GPU exists and fails
loudly on CPU-only machines. No downloads and no internet are needed.


Limitations (read before trusting any numbers)
----------------------------------------------

* **No real training data exists yet.** The Neo4j database contains no
  downstream-delay/lead-time labels, so no real-data training has been run.
  All verification uses small SYNTHETIC datasets; none of the numbers they
  produce may be presented as real-world predictive accuracy.
* **Model performance depends entirely on the quality and size of the
  training dataset.** With few labeled nodes (a handful per split) metrics
  are noisy and the model cannot generalize; meaningful evaluation needs an
  appropriately large, correctly labeled dataset covering realistic
  disruption scenarios.
* The transductive node split assumes the whole graph is available at
  prediction time; adding new nodes requires re-encoding (Module 11
  `transform`) and a fresh forward pass.
* GCN with 2 layers propagates at most 2 hops; longer ripple paths need more
  layers (or a different architecture) and more data.

Tests (all synthetic, CPU-only; no Neo4j / internet / GPU required):

```
python -m pytest tests/test_gnn_training.py -q   # Module 13: config, metrics, split, training, checkpoints
```

### Module 14 — GNN Prediction / Inference

Module 14 adds the prediction/serving layer AROUND the existing modules. It
safely loads a trained Module 13 checkpoint, rebuilds the supply-chain graph
dataset with the Module 11 builder, runs the Module 12 GNN in eval mode with
`torch.no_grad()`, and serves one **raw predicted downstream-delay value per
node** through a FastAPI endpoint. Modules 11-13 are reused unchanged: no
model, dataset or training code is duplicated, the model is never retrained
at serving time, and the regression target `y` is never an inference input.

FastAPI endpoint:

```
POST /api/v1/predictions
```

Optional request body (empty body = defaults; `relationship_types` mirrors
the Module 11 dataset builder — omitted means all outgoing relationships):

```json
{
  "relationship_types": ["SUPPLIES"]
}
```

Example response:

```json
{
  "predictions": [
    { "node_id": "entity-001", "prediction": 1.5 },
    { "node_id": "entity-002", "prediction": 0.0 }
  ],
  "prediction_count": 2,
  "timestamp": "2026-01-01T00:00:00Z"
}
```

Architecture:

```
Trained Module 13 checkpoint (state_dict, torch.load(weights_only=True))
        |   GNNTrainer.load_checkpoint (existing, reused)
        v
GNNPredictor (app.ml.prediction)   eval mode + torch.no_grad, device-validated
        v
GraphDatasetBuilder (Module 11, existing)  ->  GraphDataset (x, edge_index)
        |   (y is NEVER read; unlabeled graphs predict fine)
        v
GNNPredictionResult  (node_id -> scalar prediction, positional mapping)
        v
PredictionService (app.services.prediction_service, model loaded once & reused)
        v
POST /api/v1/predictions  ->  PredictionResponse (Pydantic)
```

Behaviour / contract:

| Aspect | Behaviour |
| --- | --- |
| Output | Exactly one prediction per graph node: `predictions[i] = {node_id, prediction}`, plus `prediction_count` and `timestamp`. `node_id` is the Neo4j entity identifier (the frontend maps it to its graph node id). |
| Raw values only | `prediction` is the raw model output (node-level regression, e.g. predicted downstream delay). **No severity classification (HIGH/MEDIUM/LOW) is derived** because the specification defines no thresholds; raw value and any future classification stay strictly separate. |
| Checkpoint behaviour | Served from `PREDICTION_CHECKPOINT_PATH` (default: unset -> endpoint returns 503). Checkpoints are produced offline by Module 13 (`GNNTrainer.save_checkpoint`), loaded with `torch.load(..., weights_only=True)` (no arbitrary object deserialization), and are never committed to Git or downloaded. Missing file -> 503; corrupt payload, weight/config mismatch or `output_dim != 1` -> loud domain errors (500), never silent garbage. |
| Model lifecycle | Loaded lazily ONCE and reused across requests (same singleton pattern as the Neo4j database); `reload_model()` exists for post-deployment refresh. No per-request reload. |
| Target leakage | Impossible by construction: inference reads only `x`/`edge_index`; `y` is never an input and unlabeled graphs predict fine. |
| Input validation | Empty graphs, non-dataset objects, feature-width mismatches, NaN/infinite features and non-finite outputs all raise dedicated `app.ml.exceptions` errors instead of being repaired. |
| Device | `PREDICTION_DEVICE` (default `cpu`); `cuda` only honoured when a GPU exists, otherwise fails loudly. CPU is fully supported and tested. |
| Neo4j dependency | The dataset is built per request from the live graph via the existing parameterized `GraphRepository` (read-only, no schema change, no new driver). Neo4j down -> 503; empty graph -> 503. |

Error mapping (responses never contain filesystem paths, stack traces or
credentials):

| Outcome | HTTP |
| --- | --- |
| Success | `200` |
| Invalid request body | `422` |
| No checkpoint configured / checkpoint file missing | `503` |
| Neo4j unavailable / empty graph | `503` |
| Invalid or incompatible checkpoint/model, unusable graph data | `500` |
| Unexpected error | `500` |

Configuration (see `.env.example`; both are documented assumptions, not
official specification values):

| Variable | Default | Purpose |
| --- | --- | --- |
| `PREDICTION_CHECKPOINT_PATH` | *(unset)* | Path to a Module 13 checkpoint (relative paths resolve against `backend/`, independent of the launch directory); unset = prediction disabled (HTTP 503 / WebSocket `MODEL_UNAVAILABLE`) |
| `PREDICTION_DEVICE` | `cpu` | Torch device used for inference |

Training and deploying a checkpoint
-----------------------------------

The prediction endpoint and the Module 16 WebSocket `prediction_request` stay
disabled until a trained Module 13 checkpoint is configured. Two offline
scripts under `backend/scripts/` produce one using the EXISTING pipeline
(no new model code, no serving changes):

1. *(optional, for graph-based training)* Seed a clearly-marked synthetic demo
   supply-chain graph (label `DemoEntity`, `:SUPPLIES` edges, a finite
   non-negative `downstream_delay_days` on every node) into a RUNNING Neo4j:

   ```bash
   cd backend
   python scripts/seed_demo_graph.py          # add --wipe to remove it again
   ```

2. Train and save the checkpoint (auto-saved by
   `GNNTrainingConfig.checkpoint_path` after `GNNTrainer.train()`):

   ```bash
   cd backend
   python scripts/train_gnn.py --synthetic    # no Neo4j needed (synthetic data)
   python scripts/train_gnn.py                # trains on the LIVE Neo4j graph
   ```

   Both modes run the same real training loop — only the data source differs.
   `--synthetic` is clearly marked and exists so the serving path can be
   exercised before real labels exist; live mode requires a numeric target
   property on EVERY node (the Module 11 builder fails loudly otherwise and
   never fabricates labels).

3. Point the server at the artifact (in the root `.env`, which is never
   committed):

   ```bash
   PREDICTION_CHECKPOINT_PATH=checkpoints/gnn_m13.pt
   ```

   The default output `checkpoints/gnn_m13.pt` is gitignored — trained `.pt`
   binaries must NOT be committed (repo policy).

4. Restart the backend and verify over a real WebSocket (see the Module 16
   smoke-test client): `uvicorn app.main:app --port 8765` then
   `python _smoke_ws_client.py`.

Tests (synthetic + mocked, CPU-only; no Neo4j / internet / GPU required):

```
python -m pytest tests/test_gnn_prediction.py -q     # Module 14: inference layer (checkpoint load, validation, determinism)
python -m pytest tests/test_prediction_service.py -q # Module 14: service orchestration (lazy model, error translation)
python -m pytest tests/test_prediction_api.py -q     # Module 14: API contract (mocked service, status-code mapping)
```

Limitations (read before trusting any numbers):

* **No real labeled data exists.** Predictions come from whatever checkpoint
  is deployed via `PREDICTION_CHECKPOINT_PATH`. The project database contains
  no real downstream-delay labels, so **no real-world predictive accuracy is
  claimed anywhere**; all verification uses clearly-marked synthetic data.
* Without a deployed checkpoint the endpoint answers `503` by design — the
  prediction capability is deployment state and is never faked.
* The transductive model assumes the whole graph is available at prediction
  time (see Module 13 limitations); GCN propagates at most `num_layers` hops.
* The prediction contract is intentionally minimal (`node_id` + raw scalar).
  Frontend-facing fields such as `predictedRisk`/`predictedLevel` in the
  frontend mock service are explicitly NOT part of this backend contract.

### Module 15 — WebSocket Foundation

Module 15 establishes a production-quality FastAPI WebSocket transport that
Modules 16/17 use for real-time ML / ripple-effect prediction streaming.
It implements connection lifecycle, JSON message transport, strict message
validation and structured errors **only** — connecting and pinging involve no
ML inference, no GNN model and no database access. Module 16 (below) adds GNN
prediction streaming on top of this same transport.

WebSocket endpoint:

```
ws://localhost:8000/api/v1/ws
```

(HTTP API runs under `/api/v1`, so the WebSocket route lives at
`/api/v1/ws`. FastAPI does not advertise WebSocket routes in the OpenAPI
`/docs` schema — verify the route with the test suite or
`app.routes` instead.)

How to start the backend:

```bash
cd backend
uvicorn app.main:app --reload
```

How to connect (Python `websockets` library):

```python
import asyncio, json
import websockets

async def main():
    async with websockets.connect("ws://localhost:8000/api/v1/ws") as ws:
        print(await ws.recv())                      # {"type": "connected", ...}
        await ws.send(json.dumps({"type": "ping"}))
        print(await ws.recv())                      # {"type": "pong", ...}

asyncio.run(main())
```

Message contract (Module 15 foundation / Module 16 prediction extension):

| Direction | Type | Payload | Notes |
| --- | --- | --- | --- |
| Client → Server | `ping` | optional `data` object | heartbeat / liveness check |
| Client → Server | `prediction_request` | optional `data` object with `node_id` and/or `relationship_types` | Module 16: run the trained GNN over the current graph |
| Server → Client | `connected` | `client_id`, `protocol`, `supported_client_messages` | sent once, immediately after accept |
| Server → Client | `pong` | `data.echo` = the ping's `data` (if any) | reply to a validated `ping` |
| Server → Client | `prediction_result` | `data.predictions`, `data.prediction_count`, `data.timestamp` (the Module 14 `PredictionResponse`) | Module 16: reply to a validated `prediction_request` |
| Server → Client | `error` | `error.code` + `error.message` | structured failure; connection stays usable |

Error codes (stable, machine-readable):

| Code | Meaning |
| --- | --- |
| `INVALID_JSON` | The text frame is not valid JSON |
| `INVALID_MESSAGE` | Not a JSON object, missing/invalid `type`, invalid prediction payload, or unknown fields |
| `UNSUPPORTED_MESSAGE_TYPE` | Unknown message type |
| `NOT_SUPPORTED_YET` | Recognized message reserved for the future Module 17 (`ripple_prediction`) |
| `MODEL_UNAVAILABLE` | No trained GNN model configured, or the configured checkpoint file is missing |
| `PREDICTION_FAILED` | Empty/unusable graph, invalid/incompatible checkpoint or model, inference failure, Neo4j unavailable, or unexpected server error |
| `NODE_NOT_FOUND` | The requested `node_id` has no prediction in the current graph |
| `INTERNAL_ERROR` | Unexpected server-side failure |

Example session:

```
CLIENT CONNECT
  ⇣
SERVER: {"type":"connected","data":{"client_id":"…","protocol":1,
         "supported_client_messages":["ping"]},"timestamp":"…"}
CLIENT: {"type":"ping"}
  ⇣
SERVER: {"type":"pong","timestamp":"…"}
CLIENT: {"type":"nonsense"}
  ⇣
SERVER: {"type":"error","error":{"code":"UNSUPPORTED_MESSAGE_TYPE",
         "message":"Unsupported message type 'nonsense'. Supported types: ping."},
         "timestamp":"…"}          # connection remains open
CLIENT DISCONNECT                   # server stays healthy
```

Behaviour / contract:

* The transport validates every inbound message with a strict Pydantic schema
  (unknown top-level fields are rejected, never silently ignored).
* Errors never contain stack traces, filesystem paths or credentials.
* One faulty client (bad JSON, unknown type, invalid prediction request,
  failing inference, abrupt disconnect) never crashes the server or affects
  other connections.
* `ConnectionManager` (`app/services/websocket_manager.py`) keeps an
  in-process registry of live clients and offers targeted `send_json` and
  `broadcast_json`; later modules reuse it to push prediction streams.
* Connecting, pinging and validating messages require **no Neo4j and no GNN
  model**. Only a validated `prediction_request` touches the Module 14
  prediction pipeline (once per request, off the event loop).

Tests:

```bash
cd backend
python -m pytest tests/test_websocket.py -q   # Module 15 transport, no Neo4j / GNN needed
```

### Module 16 — WebSocket → GNN Prediction Integration

Module 16 connects the Module 15 transport to the existing Module 14 GNN
prediction pipeline. The transport itself is unchanged; the only new message is
`prediction_request` (client → server) with its `prediction_result` reply.

Architecture:

```text
Frontend
   |  WebSocket request (prediction_request)
   v
FastAPI WebSocket  /api/v1/ws   (app.api.websocket — transport only)
   |  validated InboundWebSocketMessage
   v
app.services.websocket_prediction  (Module 16 dispatcher — async wrapper)
   |  blocking Module 14 inference moved to a worker thread
   v
PredictionService.get_predictions  (Module 14, existing model loaded once)
   |  GraphDatasetBuilder (Module 11) -> GNNPredictor (existing checkpoint)
   v
PredictionResponse  ->  prediction_result envelope | structured error
```

Request (client → server):

```json
{
  "type": "prediction_request",
  "data": {
    "node_id": "supplier-001",
    "relationship_types": ["SUPPLIES"]
  }
}
```

Both `data` fields are optional:

* `node_id` — return exactly one graph node's prediction. The server always
  runs the existing whole-graph inference and then selects ONLY the requested
  node from the real model output. A node id that has no prediction in the
  current graph yields a structured `NODE_NOT_FOUND` error. Field naming on
  the wire is `node_id` (the backend REST contract); the frontend maps it to
  its graph node id (`prediction.nodeId`).
* `relationship_types` — passed through unchanged to the Module 11 dataset
  builder (same semantics as `POST /api/v1/predictions`).

Response (server → client) for a `prediction_request`:

```json
{
  "type": "prediction_result",
  "timestamp": "…",
  "data": {
    "predictions": [
      { "node_id": "supplier-001", "prediction": 72.4 },
      { "node_id": "supplier-002", "prediction": 5.1 }
    ],
    "prediction_count": 2,
    "timestamp": "…"
  }
}
```

`data` is exactly the Module 14 `PredictionResponse` (the same shape as
`POST /api/v1/predictions`), so the WebSocket and REST contracts never
diverge. When a single node was requested, its id is echoed as
`data.requested_node_id` and `predictions` contains exactly one entry.

Node identification: the backend contract field is **`node_id`** on every
prediction entry; the frontend maps `prediction.nodeId` →
`data.predictions[i].node_id` → graph node `id`. The `nodeId` camelCase name
is **not** accepted on the wire — the strict payload schema rejects it with an
`INVALID_MESSAGE` error so typos are never silently ignored.

Error codes added by Module 16 (same `error` envelope): `MODEL_UNAVAILABLE`
(no checkpoint configured or file missing), `PREDICTION_FAILED` (unusable
graph, invalid/incompatible model, inference failure, Neo4j unavailable,
unexpected server error) and `NODE_NOT_FOUND` (requested node has no
prediction). All errors are client-safe (no paths, tracebacks or credentials)
and never terminate the connection. Within `PREDICTION_FAILED` the message
distinguishes a Neo4j outage ("the graph database is currently unavailable")
from a model/inference failure so infrastructure problems are never mistaken
for model problems.

#### Are 30/60/90-day horizons supported?

**No.** The Module 14 pipeline returns exactly one raw scalar per graph node
with no time dimension, so no horizon fields exist in the request or response
contract. Module 16 never fabricates horizon values; horizon-aware prediction
remains a future integration requirement and will be added behind this same
transport when the model supports it.

Behaviour / contract:

* A validated `prediction_request` reuses the existing `PredictionService`
  (shared process-wide singleton with the REST API — the model is loaded at
  most once). No new model, no training and no downloads happen per request.
* Inference is synchronous and blocking (Neo4j graph extraction + torch
  forward pass), so it is executed through `run_in_threadpool` — the event
  loop is never blocked by prediction work.
* An invalid or failing prediction request returns a structured error and the
  connection remains fully usable; other clients are never affected.
* Consuming `connected`, `ping`/`pong`, `disconnect` and the REST predictions
  API are unchanged by Module 16.

Tests (mocked PredictionService — no Neo4j / checkpoint / GPU required):

```bash
cd backend
python -m pytest tests/test_websocket.py tests/test_websocket_prediction.py -q
```

Real-checkpoint integration tests (real Module 13 checkpoint -> real
`PredictionService` -> WebSocket `prediction_result`; only the Neo4j dataset
builder is stubbed, exactly like `tests/test_prediction_service.py`):

```bash
cd backend
python -m pytest tests/test_real_checkpoint_serving.py -q
```

#### Live smoke test (real checkpoint + real graph)

With a trained checkpoint configured (see Module 14 "Training and deploying a
checkpoint") and Neo4j running with at least one entity, verify the full
WebSocket flow against a live server:

```bash
cd backend
uvicorn app.main:app --port 8765        # terminal 1
python _smoke_ws_client.py              # terminal 2
```

Expected: `CONNECT -> PING -> PREDICTION_REQUEST -> prediction_result (REAL
model output) -> single-node result -> PING -> DISCONNECT` and
`SMOKE TEST PASSED`. The client exits non-zero and prints the exact remediation
when it instead receives:

* `MODEL_UNAVAILABLE` — no checkpoint configured / file missing
  (train one: `python scripts/train_gnn.py`, then set
  `PREDICTION_CHECKPOINT_PATH`);
* `PREDICTION_FAILED` + "graph database is currently unavailable" — Neo4j is
  down (start it; seed demo data with `python scripts/seed_demo_graph.py` if
  the graph is empty);
* any other `PREDICTION_FAILED` — graph/model inference failure (check the
  server logs).



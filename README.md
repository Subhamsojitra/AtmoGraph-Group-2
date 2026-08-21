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
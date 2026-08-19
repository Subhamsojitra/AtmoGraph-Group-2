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

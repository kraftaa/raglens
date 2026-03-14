# rag-audit

RAG retrieval readiness diagnostics CLI. Focuses on corpus quality, chunking, metadata, and retrieval behavior (including explanations), not model answers. **WIP:** actively evolving; this note will be removed once the MVP stabilizes.

## Quick start
```bash
cargo install --path .
rag-audit readiness docs/
rag-audit simulate docs/ --queries queries.yaml
rag-audit explain docs/ --query "refund after 90 days"
```

## Commands
- `readiness <path>`: full audit (stats, chunk quality, metadata, dominant docs).
- `simulate <path> [--queries queries.yaml]`: synthetic + user queries, expectation checks, dominant docs.
- `chunks <path>`: chunk size/overlap/coherence diagnostics.
- `coverage <path>`: topic imbalance heuristic.
- `explain <path> --query "...": explain why top docs ranked.
- `compare <path> --query "...": contrast top docs/chunks for a query.

Add `--json` for CI-friendly output.

## Configuration
`rag-audit.toml` (auto-loaded from CWD) or `--config path`:
```toml
chunk_size = 400
chunk_overlap = 40
max_tokens = 1200
min_tokens = 60
top_k = 5
dominant_threshold = 0.2
required_metadata = ["product", "region", "version"]

# Embeddings
provider = "null" # or "openai"
model = "text-embedding-3-small"
base_url = "https://api.openai.com/v1"
cache_dir = ".rag-audit-cache"
```
Set `OPENAI_API_KEY` when using `provider = "openai"`. Default uses deterministic bag-of-words embeddings (offline).

## Query expectations
`queries.yaml`:
```yaml
queries:
  - id: refund_window
    query: "refund after 90 days"
    expect_docs: ["refund.md"]
```
Simulation reports PASS/FAIL counts, per-query outcomes, and emits `EXPECTATION_MISS` findings when expectations are not in top_k.

## Example output
```
RAG Retrieval Simulation
========================
Queries: 2
Documents: 3
Chunks: 3
PASS: 2  FAIL: 0

Expectations:
- PASS refund_window (expected ["refund.md"], rank Some(1))
- PASS shipping_options (expected ["shipping.md"], rank Some(1))

- HIGH: faq.md retrieved in 67%
- LOW: 3 docs missing product
```

## Project status
MVP scaffold with offline embedder, config, caching, expectation checks, and basic heuristics. Next steps: richer coherence checks, topic clustering, real embedder adapters with batching, HTML report.

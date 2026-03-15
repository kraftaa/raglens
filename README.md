# rag-audit

A CLI to diagnose **retrieval** problems in RAG systems. It analyzes your corpus before you build a full pipeline, answering:

- Are my chunks too large or too small?
- Do some documents dominate retrieval?
- Why did the retriever pick a specific document?
- Which queries are poorly covered?

Scope guard: retrieval diagnostics only. No answer grading, hallucination detection, prompt eval, or framework integrations in v1.

## Quick demo
Example corpus:
```
docs/
  refund_policy.md
  refund_old.md
  faq.md
  shipping.md
```
Example queries (`queries.txt`, one per line):
```
refund after 90 days
late shipment refund
how long does shipping take
```
Run:
```bash
rag-audit readiness docs --queries queries.txt
```
Sample (abridged):
```
RAG readiness: NEEDS WORK
Documents: 4
Chunks: 82
Chunk size avg 910 | max 1820
Issues
- large chunks (>800): 31
- duplicate chunks: 6
Retrieval simulation
Queries tested: 50
Dominant documents
- faq.md: top-1 in 41% of queries
Coverage
- weak matches: 6
- no matches: 2
```

## Install
```bash
cargo install rag-audit
```
or build locally:
```bash
cargo build --release
```

## Core commands (v1)
- `readiness <docs> [--queries queries.txt] [--json-out file]`: one-shot health check (chunk stats, duplicates, dominance, weak coverage).
- `chunks <docs> [--json-out file]`: chunk quality (avg/min/max/p50/p95 tokens, large/small counts, duplicates).
- `simulate <docs> --queries queries.txt [--json-out file]`: run queries, report dominant docs, low/no-match counts, similarity stats.
- `explain <docs> --query "..." [--json-out file]`: EXPLAIN-style breakdown of why top docs ranked.
- `coverage <docs> [--queries queries.txt] [--json-out file]`: with queries → Good/Weak/None coverage; without queries → topic imbalance.

Add `--artifacts-dir artifacts/` to save standard JSON files per command (e.g., `artifacts/readiness.json`, `simulation.json`, `chunks.json`, `explain.json`).

## How it works (deterministic pipeline)
1) Load docs (`.md`, `.txt`, `.json`) and optional frontmatter metadata.  
2) Normalize and chunk with fixed rules (default: size 400, overlap 50; demo config uses 200/30). Sequential, so runs are repeatable.  
3) Embed chunks: default deterministic “Null” bag-of-words embedder (offline). Optional OpenAI embeddings with retry/backoff/timeout and on-disk cache keyed by text+model.  
4) Retrieve: embed queries, cosine similarity, top-k (default 5), deterministic ordering.  
5) Diagnose: chunk stats (avg/min/max/p50/p95, large/small, duplicates), dominance (top1/top3 freq), coverage (low/no-match vs thresholds), EXPLAIN per query (semantic, keyword overlap, length penalty placeholder, tokens).  
6) Report: human output + JSON artifacts (`--json-out` or `--artifacts-dir`) for CI/diffs.

## Inputs
Documents (v1): `.md`, `.txt`, `.json`. (PDF optional later.)

Queries: 
- YAML with `queries:` list (supports `expect_docs`), or
- Plain text, one query per line.

## Defaults (deterministic)
- Chunk size 400 tokens, overlap 50.
- Retrieval: cosine, top_k=5.
- Embedding: single configured backend (Null bag-of-words by default; OpenAI available via config/env). Sequential processing for repeatability.

## JSON artifacts (per command)
`--json-out file` writes that command’s report; `--artifacts-dir dir` writes standard filenames:
- `readiness.json`: findings + chunk_stats
- `chunks.json`: chunk stats + findings
- `simulation.json`: dominant docs, top1/top3 freq, avg_top1_similarity, low/no-match counts, expectations
- `coverage.json`: Good/Weak/None (if queries) or topic imbalance
- `explain.json` / `compare.json`: ranked results with score components

## Non-goals (v1)
- No hallucination/answer grading
- No prompt eval
- No vector DB or framework adapters (LangChain/LlamaIndex/etc.)
- No dashboards/UI

## Why use this
- Early detection of chunking mistakes, duplication, dominance, and weak coverage.
- Deterministic, numeric outputs engineers can trust.
- CI-friendly JSON artifacts for gating changes.

## Config/overrides
`rag-audit.toml` (optional):
```
chunk_size = 400
chunk_overlap = 50
top_k = 5
dominant_threshold = 0.2
provider = "null" # or "openai"
model = "text-embedding-3-small"
cache_dir = ".rag-audit-cache"
```
CLI overrides: `--embedder null|openai`, `--cache-dir <dir>`, `--json-out <file>`, `--artifacts-dir <dir>`.

## Project status
Early, but end-to-end runnable: load docs, chunk, embed, retrieve, diagnose, emit JSON. Focus remains on retrieval diagnostics; broader RAG features intentionally out of scope for v1.

## License
MIT

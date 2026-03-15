# rag-audit

“A CLI to debug retrieval behavior in RAG systems.”
Scope: retrieval diagnostics only (no answer grading, hallucination detection, prompt eval, or framework integrations).

It analyzes your corpus before you build a full pipeline, answering:

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

Add `--artifacts-dir artifacts/` to save standard JSON files per command (e.g., `artifacts/readiness.json`, `simulation.json`, `chunks.json`, `explain.json`, `compare_runs.json`).

## How it works (deterministic pipeline)
1) Load & chunk docs  
2) Generate embeddings (default offline deterministic; optional OpenAI)  
3) Retrieve with cosine top-k  
4) Analyze chunk stats & retrieval behavior (dominance, low/no-match, EXPLAIN)  
5) Output human report + JSON artifacts (`--json-out` or `--artifacts-dir`)

## Inputs
Documents (v1): `.md`, `.txt`, `.json`. (PDF optional later.)

Queries: 
- YAML with `queries:` list (supports `expect_docs`), or
- Plain text, one query per line.

## Defaults (deterministic)
- Chunk size 400 tokens, overlap 50.
- Retrieval: cosine, top_k=5.
- Embedding: single configured backend (Null bag-of-words by default; OpenAI available via config/env). Sequential processing for repeatability.

## When this is useful
- Debugging unexpected retrieval results
- Validating chunking strategy
- Detecting dominant documents
- Testing query coverage of a knowledge base

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

# RAGLens

RAGLens is a CLI to analyze and fix retrieval behavior across your RAG system.
Use it to understand patterns across many queries (coverage, dominance, chunking).

Scope: retrieval diagnostics only (no answer grading, hallucination detection, prompt eval, or framework integrations).

Command name: `raglens` (`rag-audit` is kept as a compatibility alias).

It answers:

- Are my chunks too large or too small?
- Do some documents dominate retrieval?
- Why did the retriever pick a specific document?
- Which queries are poorly covered?
- What likely causes a dominant document (oversized chunks, duplicate content, missing metadata)?

## 5-second demo
```bash
raglens --config examples/rag-audit.toml explain examples/docs --query "refund after 90 days"
```
Output (abridged):
```text
Top result: refund_policy.md#5 (doc: refund_policy.md)
score: 0.622
why: semantic 0.472 + keyword boost 0.150
```

## Before / After
Before:
- Wrong retrieval result
- Scroll logs and guess chunking problems
- Tweak settings blindly

After:
- `raglens explain ...`
- `raglens readiness ...`
- Concrete ranking evidence + dominant-document and chunking signals

## What this is NOT
- Not prompt evaluation
- Not hallucination detection
- Not LLM tracing
- Not a full RAG framework

## Quick demo
Examples below use `cargo run -- ...` so they work before install.

Example corpus:
```
docs/
  refund_policy.md
  refund_old.md
  faq.md
  shipping.md
  returns.md
  billing.md
  support.md
  pricing.md
```
Example plain queries (`examples/queries.txt`, one per line):
```
refund after 90 days
late shipment refund
how long does shipping take
return damaged item
billing dispute window
lost package claim
express shipping speed
```
Structured queries with expected docs for simulation/readiness live in `examples/queries_structured.txt`.
Run:
```bash
cargo run -- --config examples/rag-audit.toml readiness examples/docs --queries examples/queries_structured.txt
cargo run -- --config examples/rag-audit.toml simulate examples/docs --queries examples/queries_structured.txt
cargo run -- --config examples/rag-audit.toml explain examples/docs --query "refund after 90 days"
```
Sample (current output, abridged):
```
RAG Retrieval Readiness Report
==============================
Documents: 8
Chunks: 11
Chunk config: size 80 overlap 20 | top_k 5 | thresholds low 0.35 no-match 0.25 | embedder Null
Chunk size avg 55.4 | min 23 | max 80 | p50 54 | p95 80
- LOW: 1 chunks below min_tokens 30
- HIGH: refund_policy.md appears as top-1 in 50% of queries (top-k 83%)
- FAIL: Query late_shipping expected ["shipping.md", "faq.md"], not found in top 5
- MEDIUM: refund_policy.md dominance likely driven by: broad lexical coverage across many queries
```

## Quick Start (30 seconds)
```bash
# Install locally
cargo install --path .

# Verify install
raglens --version
raglens --help
raglens self-test --json

# Run readiness audit
raglens --config ./examples/rag-audit.toml readiness ./examples/docs --queries ./examples/queries_structured.txt

# Debug one query
raglens --config ./examples/rag-audit.toml explain ./examples/docs --query "refund after 90 days"
```

## Real-world quickstart (recommended)
Use your real docs folder and a plain query file.

`queries.txt` (one real user question per line):
```text
how do refunds work after 90 days
how long does international shipping take
how to dispute an incorrect charge
```

Run:
```bash
raglens readiness ./docs --queries ./queries.txt --artifacts-dir ./artifacts
raglens simulate ./docs --queries ./queries.txt --artifacts-dir ./artifacts
raglens explain ./docs --query "how do refunds work after 90 days"
raglens explain ./docs --query "how do refunds work after 90 days" --html-out ./artifacts/explain.html
```

Or use the helper script:
```bash
scripts/run-audit.sh ./docs ./queries.txt ./artifacts
```

HTML reports:
- `explain` and `compare-query` support `--html-out <file>`.
- If you pass `--artifacts-dir`, those commands also write `explain.html` / `compare.html` automatically.

## Install
Local install from this repo (recommended while in development):
```bash
cargo install --path .
```
This installs two executable names:
- `raglens` (primary CLI name)
- `rag-audit` (compatibility alias)

If you do not want to install yet, run directly:
```bash
cargo run -- --config examples/rag-audit.toml readiness examples/docs --queries examples/queries_structured.txt
```

## Core commands (v1)
- `readiness <docs> [--queries queries.txt] [--json-out file]`: one-shot health check (chunk stats, duplicates, dominance, weak coverage).
- `chunks <docs> [--json-out file]`: chunk quality (avg/min/max/p50/p95 tokens, large/small counts, duplicates).
- `simulate <docs> --queries queries.txt [--json-out file]`: run queries, report dominant docs, low/no-match counts, similarity stats.
- `explain <docs> --query "..." [--json-out file] [--html-out file]`: EXPLAIN-style breakdown of why top docs ranked.
- `compare-query <docs> --query "..." [--json-out file] [--html-out file]`: side-by-side top result comparison for a single query.
- `optimize <docs> --queries queries.txt [--chunk-sizes 200,300,400,600] [--chunk-overlaps 20,40,80] [--top-n 5] [--write-config best.toml] [--json-out file]`: search chunking candidates and suggest best retrieval metrics.
- `coverage <docs> [--queries queries.txt] [--json-out file]`: with queries → Good/Weak/None coverage; without queries → topic imbalance.
- `compare-runs baseline.json improved.json` (alias: `compare`) `[--format summary|table] [--fail-if-weak-increases] [--fail-if-no-match-increases] [--fail-if-similarity-drops] [--fail-if-regressed] [--fail-if-top1-dominant-rate-exceeds 0.60] [--fail-if-top1-dominant-rate-increases] [--fail-if-query-count-mismatch] [--json-out file]`: compare before/after simulation artifacts with IMPROVED/REGRESSED/NEUTRAL verdict and optional CI gating.
- `self-test [--docs examples/docs] [--queries examples/queries_structured.txt] [--json]`: built-in smoke check for install/CI sanity.

Add `--artifacts-dir artifacts/` to save standard report files per command (JSON + HTML where supported).

Note: global `--fail-on-*` flags apply only to `readiness` and `simulate`.

## Exit codes
- `0`: success
- `1`: runtime/usage/config error
- `2`: readiness/simulate fail gate triggered
- `3`: compare-runs gate triggered (regression/delta mismatch checks)

## How it works (deterministic pipeline)
1) Load & chunk docs  
2) Generate embeddings (default offline deterministic; optional OpenAI)  
3) Retrieve with cosine top-k  
4) Analyze chunk stats & retrieval behavior (dominance, low/no-match, EXPLAIN)  
5) Output human report + artifacts (`--json-out` / `--html-out` / `--artifacts-dir`)

### Sample output (readiness on `examples/docs`)
```
RAG Retrieval Readiness Report
==============================

Documents: 8
Chunks: 11

Chunk config: size 80 overlap 20 | top_k 5 | thresholds low 0.35 no-match 0.25 | embedder Null
Chunk size avg 55.4 | min 23 | max 80 | p50 54 | p95 80
- LOW: 1 chunks below min_tokens 30
- HIGH: refund_policy.md appears as top-1 in 50% of queries (top-k 83%)
- FAIL: Query late_shipping expected ["shipping.md", "faq.md"], not found in top 5
- MEDIUM: refund_policy.md dominance likely driven by: broad lexical coverage across many queries
```

## Inputs
Documents (v1): `.md`, `.txt`, `.html`, `.json`. (PDF optional later.)

Queries: 
- YAML with `queries:` list (supports `expect_docs`), or
- Plain text, one query per line.
- Optional tab-separated text lines: `id<TAB>query<TAB>expect_doc1,expect_doc2`.
  - Example: `examples/queries_structured.txt`

## Defaults (deterministic)
- Chunk size 400 tokens, overlap 40.
- Retrieval: cosine, top_k=5.
- Dominant document threshold: 0.30 (top-1 share).
- Embedding: single configured backend (Null bag-of-words by default; OpenAI available via config/env). Sequential processing for repeatability.

## When this is useful
- Debugging unexpected retrieval results
- Validating chunking strategy
- Detecting dominant documents
- Testing query coverage of a knowledge base

## CI gating example
```
raglens simulate docs --queries queries.txt --json-out report.json
jq 'select(.sim_summary.low_similarity_queries==0 and .sim_summary.no_match_queries==0)' report.json >/dev/null \
  || { echo "Retrieval coverage failed"; exit 1; }

raglens compare-runs baseline.json improved.json --fail-if-regressed --fail-if-no-match-increases
```

## Compare-runs example
```bash
raglens simulate examples/docs --queries examples/queries_structured.txt --json-out baseline.json
# ...make corpus/chunking changes...
raglens simulate examples/docs --queries examples/queries_structured.txt --json-out improved.json

raglens compare-runs baseline.json improved.json --format table
raglens compare-runs baseline.json improved.json \
  --fail-if-regressed \
  --fail-if-top1-dominant-rate-exceeds 0.60 \
  --fail-if-top1-dominant-rate-increases
```

## Auto-tune chunking example
```bash
raglens optimize ./docs \
  --queries ./queries_structured.txt \
  --chunk-sizes 200,300,400,600 \
  --chunk-overlaps 20,40,80 \
  --top-n 5 \
  --write-config ./best-chunking.toml
```

## Dominance root-cause hints
When dominance is detected, readiness/simulate can emit cause hints:
```text
- MEDIUM: faq.md dominance likely driven by: oversized chunks, duplicate chunk content, missing metadata fields
```

## JSON artifacts (per command)
`--json-out file` writes that command’s report; `--artifacts-dir dir` writes standard filenames:
- Most artifacts include `meta` (`schema_version`, `tool_version`, `command`) for CI traceability.
- Artifact shape: `{ "meta": {...}, ...command-specific fields... }`.
- `readiness.json`: findings + chunk_stats
- `chunks.json`: chunk stats + findings
- `simulation.json`: dominant docs, top1/top3 freq, avg_top1_similarity, low/no-match counts, expectations
- `coverage.json`: Good/Weak/None (if queries) or topic imbalance
- `explain.json` / `compare.json`: ranked results with score components
- `optimize.json`: tested chunking candidates and best recommendation

## HTML artifacts
- `explain` and `compare-query` support `--html-out file`.
- With `--artifacts-dir`, they also emit:
  - `explain.html`
  - `compare.html`
- You can write both JSON and HTML in one run:
  - `raglens explain docs --query "refund after 90 days" --json-out artifacts/explain.json --html-out artifacts/explain.html`

## Non-goals (v1)
- No hallucination/answer grading
- No prompt eval
- No vector DB or framework adapters (LangChain/LlamaIndex/etc.)
- No dashboards/UI

## Why use this
- Before: “why did production retrieval fail on this query?” often requires manual log spelunking.
- After: run `raglens explain ...` / `raglens simulate ...` to see concrete retrieval behavior and failure signals.
- Early detection of chunking mistakes, duplication, dominance, and weak coverage.
- Deterministic, numeric outputs engineers can trust.
- CI-friendly JSON artifacts for gating changes.

## Config/overrides
`rag-audit.toml` (optional):
```
chunk_size = 400
chunk_overlap = 40
top_k = 5
dominant_threshold = 0.3
provider = "null" # or "openai"
model = "text-embedding-3-small"
cache_dir = ".rag-audit-cache"
```
CLI overrides: `--embedder null|openai`, `--cache-dir <dir>`, `--json-out <file>`, `--html-out <file>`, `--artifacts-dir <dir>`.
See `rag-audit.toml.example` for a fuller template.

## Roadmap
- [ ] PDF loader (with deterministic extraction path)
- [ ] Additional embedding providers
- [ ] Hybrid retrieval diagnostics (BM25 + semantic)

## Project status
Early, but end-to-end runnable: load docs, chunk, embed, retrieve, diagnose, emit JSON. Focus remains on retrieval diagnostics; broader RAG features intentionally out of scope for v1.

## License
MIT

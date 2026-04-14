# RAGLens

RAGLens is a CLI to debug retrieval behavior in RAG systems.

MVP loop:
1. `explain` one bad query
2. `simulate` many queries
3. `fix` suggests the first change to try

Scope: retrieval diagnostics only.  
Not answer grading, hallucination detection, prompt eval, or agent tracing.

## Quick Start

```bash
# from repo root
cargo run -- explain inputs/docs --query "refund after 90 days"
cargo run -- simulate inputs/docs --queries inputs/queries.txt
cargo run -- fix inputs/docs --queries inputs/queries.txt
```

Use a richer sample corpus:

```bash
cargo run -- explain inputs/examples/ecommerce/docs --query "refund after 90 days"
cargo run -- simulate inputs/examples/ecommerce/docs --queries inputs/examples/ecommerce/queries.txt
cargo run -- fix inputs/examples/ecommerce/docs --queries inputs/examples/ecommerce/queries.txt
```

Install locally:

```bash
cargo install --path .
raglens --help
```

Install with pip (no Rust toolchain required once wheels are published):

```bash
pip install raglens-cli
raglens --help
```

## Primary Commands

### `explain`

Explain why top documents/chunks ranked for a single query.

```bash
raglens explain ./docs --query "refund after 90 days"
```

Outputs:
- top-ranked chunks/docs
- score breakdown (semantic + lexical components)
- quick signal for why rank #1 won

Optional artifacts:

```bash
raglens explain ./docs --query "refund after 90 days" \
  --json-out artifacts/explain.json \
  --html-out artifacts/explain.html
```

### `simulate`

Simulate retrieval over a query set.

```bash
raglens simulate ./docs --queries ./queries.txt
```

Outputs:
- top-1 document frequency
- low-similarity query count
- no-match query count
- dominant-document warning

### `fix`

Rules-based diagnostic advisor.  
It does not mutate files or auto-run agents.

```bash
raglens fix ./docs --queries ./queries.txt
```

Outputs:
- detected issue
- likely causes
- first fix to try
- rerun command

Example:

```text
Issue: refund_policy.md dominates 48% of top-1 results

Likely causes:
- chunk size too large for mixed-topic content
- duplicate/repeated chunk language boosts one document

Try first: reduce chunk_size from 400 to 200
Then rerun: raglens simulate <docs> --queries queries.txt
```

### `diff`

Compare two run artifacts and explain why behavior changed.

```bash
raglens diff --baseline run_a.json --current run_b.json
```

Optional machine output:

```bash
raglens diff --baseline run_a.json --current run_b.json --format json
```

Outputs:
- whether answer changed
- whether retrieval changed (added/removed/common docs)
- likely root cause classification
- confidence level

Example artifacts:

```bash
raglens diff \
  --baseline inputs/examples/run_diff/robocall_run_a.json \
  --current inputs/examples/run_diff/robocall_run_b.json
```

More scenarios (equivalent, contradictory, score-shift-only, more-specific):
`inputs/examples/run_diff/README.md`

### `save-run`

Save one run artifact JSON from your app outputs so it can be diffed later.

```bash
raglens save-run \
  --out artifacts/runs/2026-04-13T10-20-00_run.json \
  --question "Why did revenue increase?" \
  --answer "Revenue increased due to US growth" \
  --retrieved-docs artifacts/runs/retrieved_docs.json \
  --model gpt-4.1 \
  --top-k 5
```

`--retrieved-docs` accepts:
- a JSON array of retrieved docs
- or an object containing a `retrieved_docs` array

### `mcp-import`

Convert an MCP/agent trace JSON into a valid run artifact for `diff`.
This avoids hand-writing schema-matching JSON.

```bash
raglens mcp-import \
  --in ./trace.json \
  --out artifacts/runs/run_a.json
```

If your trace uses custom field paths, set JSON pointers:

```bash
raglens mcp-import \
  --in ./trace.json \
  --out artifacts/runs/run_a.json \
  --question-pointer /payload/q \
  --answer-pointer /payload/final \
  --docs-pointer /payload/ctx/hits
```

Then compare runs:

```bash
raglens diff --baseline artifacts/runs/run_a.json --current artifacts/runs/run_b.json
```

## Inputs

Recommended MVP inputs:
- docs: `.md`, `.txt`
- queries: plain text, one query per line

Supported (advanced) query formats:
- YAML with `queries:`
- tab-separated: `id<TAB>query<TAB>expect_doc1,expect_doc2`
- plain text query files can include blank lines and `# comment` lines (ignored)

## Deterministic by Default

- default embedder: local deterministic null embedder
- deterministic chunking and ranking pipeline
- consistent outputs for same corpus + queries + config

## Artifacts

All commands support `--json-out`.
`explain` also supports `--html-out`.

You can also use `--artifacts-dir` to write standard report files.

## Real-World Use

Run on your own corpus:

```bash
raglens simulate ./docs --queries ./queries.txt --artifacts-dir ./artifacts
raglens fix ./docs --queries ./queries.txt
```

If you want a simple wrapper:

```bash
scripts/run-audit.sh ./docs ./queries.txt ./artifacts
```

Use real web docs as input (optional):

```bash
scripts/import-web-docs.sh ./inputs/public_urls.txt ./inputs/docs_web
cargo run -- simulate ./inputs/docs_web --queries ./inputs/queries.txt
```

Notes:
- imported files are saved as plain `.txt` with a `Source:` header
- imported pages that are mostly one long line are still split safely (sentence/token-based) during chunking
- keep only pages you are allowed to store/use in your environment

## Advanced / Experimental

RAGLens includes additional advanced commands for deeper workflows (comparison, optimization, etc.).
They are intentionally hidden from default help to keep the MVP interface focused.

Experimental deterministic answer checker (CSV truth layer):

```bash
raglens answer-audit \
  --data ./inputs/examples/answer_audit/sales.csv \
  --group-by region,channel \
  --metric revenue \
  --period-col period \
  --baseline old \
  --current new \
  --question "Why did revenue increase?" \
  --answer "Revenue increased due to EU growth"
```

Unknown dataset quick start (auto infer schema):

```bash
raglens answer-audit \
  --data ./my_data.csv \
  --auto \
  --answer "Revenue increased because EU grew"
```

`--auto` infers:
- metric column
- period column
- baseline/current period values
- group-by columns

Optional period bucketing:

```bash
raglens answer-audit \
  --data ./my_data.csv \
  --auto \
  --period-granularity month \
  --answer "Revenue increased because EU grew"
```

- `--period-granularity raw|month|week` (default `raw`)
- `month`/`week` require parseable date-like period values

More answer-audit examples:

```bash
# expected verdict: SUPPORTED
raglens answer-audit \
  --data ./inputs/examples/answer_audit/sales_supported.csv \
  --group-by region,channel \
  --metric revenue \
  --period-col period \
  --baseline old \
  --current new \
  --answer "Revenue increased due to strong US Direct growth"

# expected verdict: RISKY (mentions weak contributor)
raglens answer-audit \
  --data ./inputs/examples/answer_audit/sales_risky.csv \
  --group-by region,channel \
  --metric revenue \
  --period-col period \
  --baseline old \
  --current new \
  --answer "Revenue increased due to US Direct and LATAM growth"
```

## Non-Goals

- Full RAG framework
- Answer quality evaluator
- Hallucination detector
- Autonomous tuning agent

## License

MIT

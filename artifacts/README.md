# rag-audit artifacts

Standard JSON outputs when using `--artifacts-dir` or `--json-out`:

Most command artifacts now include a `meta` block with:
- `schema_version`
- `tool_version`
- `command`

General shape:
```json
{
  "meta": { "...": "..." },
  "...": "command-specific payload"
}
```

- readiness.json: findings, chunk_stats, optional sim_summary/coverage when queries provided.
- simulation.json: sim_summary (avg_top1_similarity, low/no-match counts, dominant docs), findings.
- chunks.json: chunk_stats (avg/min/max/p50/p95 tokens, large/small, duplicates) and findings.
- coverage.json: coverage summary (good/weak/none) or topic imbalance findings.
- explain.json: ranked chunks with score components for a single query.
- compare_runs.json: diff between two simulation reports (before/after metrics, verdict, improved/regressed metric lists, and top-1 doc deltas).

Fields are human-readable and suitable for CI gating or diffing between runs.

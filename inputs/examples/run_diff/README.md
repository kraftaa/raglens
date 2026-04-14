# Run Diff Examples (Public Web Sources)

These files show how to demo `raglens diff` with realistic run artifacts.

## Files

Robocall examples:
- `robocall_run_a.json` — baseline run
- `robocall_run_b.json` — retrieval + answer changed
- `robocall_run_b_same_answer.json` — retrieval changed, answer stable
- `robocall_run_b_alt_answer.json` — answer changed, retrieval stable
- `robocall_run_b_equivalent.json` — equivalent answer (case/whitespace variation)
- `robocall_run_b_score_shift.json` — same docs, significant score shifts

Revenue examples:
- `revenue_run_a.json` — baseline run
- `revenue_run_b_contradictory.json` — contradictory answer with stable retrieval
- `revenue_run_b_more_specific.json` — more specific answer with retrieval change

## Demo Commands

### 1) Answer + retrieval changed

```bash
raglens diff \
  --baseline inputs/examples/run_diff/robocall_run_a.json \
  --current inputs/examples/run_diff/robocall_run_b.json
```

Expected root cause: `retrieval_changed`

### 2) Retrieval changed, answer stable

```bash
raglens diff \
  --baseline inputs/examples/run_diff/robocall_run_a.json \
  --current inputs/examples/run_diff/robocall_run_b_same_answer.json --format json
```

Expected root cause: `retrieval_changed_but_answer_stable`

### 3) Answer changed, retrieval stable

```bash
raglens diff \
  --baseline inputs/examples/run_diff/robocall_run_a.json \
  --current inputs/examples/run_diff/robocall_run_b_alt_answer.json --format json
```

Expected root cause: `answer_generation_changed`

### 4) Equivalent answer (no meaningful change)

```bash
raglens diff \
  --baseline inputs/examples/run_diff/robocall_run_a.json \
  --current inputs/examples/run_diff/robocall_run_b_equivalent.json --format json
```

Expected: `answer_diff.change_type = equivalent`, root cause `no_meaningful_change`

### 5) Score-shift-only retrieval change

```bash
raglens diff \
  --baseline inputs/examples/run_diff/robocall_run_a.json \
  --current inputs/examples/run_diff/robocall_run_b_score_shift.json --format json
```

Expected root cause: `retrieval_changed_but_answer_stable` (via score delta threshold)

### 6) Contradictory answer shift

```bash
raglens diff \
  --baseline inputs/examples/run_diff/revenue_run_a.json \
  --current inputs/examples/run_diff/revenue_run_b_contradictory.json --format json
```

Expected: `answer_diff.change_type = contradictory`, root cause `answer_generation_changed`

### 7) More-specific answer shift

```bash
raglens diff \
  --baseline inputs/examples/run_diff/revenue_run_a.json \
  --current inputs/examples/run_diff/revenue_run_b_more_specific.json --format json
```

Expected: `answer_diff.change_type = more_specific`, root cause often `retrieval_changed`

## Public source links used for doc snippets

- https://www.usa.gov/telemarketer-scam-call-complaints
- https://www.usa.gov/consumer-complaints
- https://consumer.ftc.gov/articles/0259-robocalls

All snippets in these artifacts are short paraphrases of public pages.

## Real-life workflow

Create your own run artifacts from app logs/output:

```bash
raglens save-run \
  --out artifacts/runs/2026-04-13T10-20-00_run.json \
  --question "How can I reduce unwanted robocalls and where should I report scam calls?" \
  --answer "Use FCC robocall guidance to reduce calls." \
  --retrieved-docs artifacts/runs/retrieved_docs_a.json \
  --model gpt-4.1 \
  --top-k 5
```

Then compare two runs:

```bash
raglens diff \
  --baseline artifacts/runs/2026-04-13T10-20-00_run.json \
  --current artifacts/runs/2026-04-13T10-45-00_run.json
```

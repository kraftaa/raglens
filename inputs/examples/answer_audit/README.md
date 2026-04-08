# Answer Audit Examples

1) Contradiction + missing main driver (expected `INCORRECT`)

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

2) Main driver mentioned, no contradiction (expected `SUPPORTED`)

```bash
raglens answer-audit \
  --data ./inputs/examples/answer_audit/sales_supported.csv \
  --group-by region,channel \
  --metric revenue \
  --period-col period \
  --baseline old \
  --current new \
  --answer "Revenue increased due to strong US Direct growth"
```

3) Mentions weak minor contributor (expected `RISKY`)

```bash
raglens answer-audit \
  --data ./inputs/examples/answer_audit/sales_risky.csv \
  --group-by region,channel \
  --metric revenue \
  --period-col period \
  --baseline old \
  --current new \
  --answer "Revenue increased due to US Direct and LATAM growth"
```

4) Auto infer mode (minimal flags)

```bash
raglens answer-audit \
  --data ./inputs/examples/answer_audit/sales.csv \
  --auto \
  --answer "Revenue increased due to EU growth"
```

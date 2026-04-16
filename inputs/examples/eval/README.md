# Eval Example

Use the existing run-diff artifacts with deterministic eval rules:

```bash
raglens eval \
  --run inputs/examples/run_diff/robocall_run_b.json \
  --rules inputs/examples/eval/rules_robocall.yaml
```

Generate a markdown report:

```bash
raglens report \
  --run inputs/examples/run_diff/robocall_run_b.json \
  --rules inputs/examples/eval/rules_robocall.yaml
```

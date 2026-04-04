# Input Starter Pack

Use these files to run a first audit quickly.

Run:

```bash
raglens --config ./inputs/rag-audit.toml readiness ./inputs/docs --queries ./inputs/queries_structured.txt
raglens --config ./inputs/rag-audit.toml simulate ./inputs/docs --queries ./inputs/queries_structured.txt
raglens --config ./inputs/rag-audit.toml explain ./inputs/docs --query "refund after 90 days"
```

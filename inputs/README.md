# Input Starter Pack

Use these files to run a first audit quickly.

Run:

```bash
raglens --config ./inputs/rag-audit.toml readiness ./inputs/docs --queries ./inputs/queries_structured.txt
raglens --config ./inputs/rag-audit.toml simulate ./inputs/docs --queries ./inputs/queries_structured.txt
raglens --config ./inputs/rag-audit.toml explain ./inputs/docs --query "refund after 90 days"
```

Richer example corpus:

```bash
raglens explain ./inputs/examples/ecommerce/docs --query "refund after 90 days"
raglens simulate ./inputs/examples/ecommerce/docs --queries ./inputs/examples/ecommerce/queries.txt
raglens fix ./inputs/examples/ecommerce/docs --queries ./inputs/examples/ecommerce/queries.txt
```

Optional: import public web pages as docs:

```bash
scripts/import-web-docs.sh ./inputs/public_urls.txt ./inputs/docs_web
raglens simulate ./inputs/docs_web --queries ./inputs/queries.txt
```

Query file tips:
- one query per line
- blank lines are ignored
- lines starting with `#` are treated as comments

# Ecommerce Example Corpus

Run MVP workflow:

```bash
raglens explain ./inputs/examples/ecommerce/docs --query "refund after 90 days"
raglens simulate ./inputs/examples/ecommerce/docs --queries ./inputs/examples/ecommerce/queries.txt
raglens fix ./inputs/examples/ecommerce/docs --queries ./inputs/examples/ecommerce/queries.txt
```

Structured query run:

```bash
raglens simulate ./inputs/examples/ecommerce/docs \
  --queries ./inputs/examples/ecommerce/queries_structured.txt
```

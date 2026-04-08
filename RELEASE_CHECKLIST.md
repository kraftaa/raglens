# Release Checklist

Use this checklist before cutting a new `rag-audit` release.

## 1) Local quality checks

```bash
cargo fmt
cargo clippy -q
cargo test -q
```

## 2) Smoke test CLI behavior

```bash
cargo run -- readiness examples/docs --queries examples/queries.txt
cargo run -- simulate examples/docs --queries examples/queries_structured.txt
cargo run -- explain examples/docs --query "refund after 90 days"
cargo run -- self-test --docs examples/docs --queries examples/queries_structured.txt
```

## 3) Smoke test compare + CI gates

```bash
cargo run -- simulate examples/docs --queries examples/queries_structured.txt --json-out baseline.json
cargo run -- simulate examples/docs --queries examples/queries_structured.txt --json-out improved.json
cargo run -- compare-runs baseline.json improved.json --format table
cargo run -- compare-runs baseline.json improved.json --fail-if-regressed
```

## 4) Verify docs and metadata

- `README.md` examples match current CLI output/options.
- `artifacts/README.md` matches artifact schema.
- `Cargo.toml` has correct package metadata/version.
- `pyproject.toml` version matches `Cargo.toml` version.

## 5) Packaging sanity check

```bash
cargo package --allow-dirty
```

Review package output and ensure only intended files are included.

## 6) Tag/release prep

- Bump version in `Cargo.toml`.
- Bump version in `pyproject.toml`.
- Update changelog/release notes.
- Re-run steps 1–5.
- Create git tag for the version.

## 7) Publishing secrets/config

- `CARGO_REGISTRY_TOKEN` is set (if publishing crate).
- `PYPI_API_TOKEN` is set (if publishing PyPI wheel).

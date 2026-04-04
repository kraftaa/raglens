#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: scripts/run-audit.sh <docs_dir> <queries_file> [artifacts_dir]"
  echo "Example: scripts/run-audit.sh ./docs ./queries.txt ./artifacts"
  exit 1
fi

DOCS_DIR="$1"
QUERIES_FILE="$2"
ARTIFACTS_DIR="${3:-artifacts}"

run_raglens() {
  if command -v raglens >/dev/null 2>&1; then
    raglens "$@"
  else
    cargo run -- "$@"
  fi
}

echo "Running readiness..."
run_raglens readiness "$DOCS_DIR" --queries "$QUERIES_FILE" --artifacts-dir "$ARTIFACTS_DIR"

echo
echo "Running simulate..."
run_raglens simulate "$DOCS_DIR" --queries "$QUERIES_FILE" --artifacts-dir "$ARTIFACTS_DIR"

echo
echo "Done. Artifacts written to: $ARTIFACTS_DIR"

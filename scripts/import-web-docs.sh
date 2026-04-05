#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: scripts/import-web-docs.sh <urls_file> <output_docs_dir> [timeout_seconds]"
  echo "Example: scripts/import-web-docs.sh ./inputs/public_urls.txt ./inputs/docs_web 20"
  exit 1
fi

URLS_FILE="$1"
OUT_DIR="$2"
TIMEOUT="${3:-20}"

mkdir -p "$OUT_DIR"

python3 - "$URLS_FILE" "$OUT_DIR" "$TIMEOUT" <<'PY'
import html
import os
import re
import sys
import urllib.request
import urllib.parse

urls_file, out_dir, timeout = sys.argv[1], sys.argv[2], int(sys.argv[3])

with open(urls_file, 'r', encoding='utf-8') as f:
    urls = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

def slug(u: str) -> str:
    p = urllib.parse.urlparse(u)
    host = p.netloc.replace(':', '_')
    path = p.path.strip('/').replace('/', '_') or 'index'
    base = f"{host}_{path}"
    base = re.sub(r'[^A-Za-z0-9._-]+', '_', base)
    return base[:180] + '.txt'

def html_to_text(raw: str) -> str:
    raw = re.sub(r'(?is)<script.*?>.*?</script>', ' ', raw)
    raw = re.sub(r'(?is)<style.*?>.*?</style>', ' ', raw)
    raw = re.sub(r'(?is)<[^>]+>', ' ', raw)
    raw = html.unescape(raw)
    raw = re.sub(r'\s+', ' ', raw).strip()
    return raw

for u in urls:
    try:
        req = urllib.request.Request(
            u,
            headers={"User-Agent": "raglens-import/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode('utf-8', errors='ignore')
        txt = html_to_text(body)
        if len(txt) < 80:
            print(f"skip short/empty: {u}")
            continue
        out_path = os.path.join(out_dir, slug(u))
        with open(out_path, 'w', encoding='utf-8') as out:
            out.write(f"Source: {u}\n\n")
            out.write(txt)
        print(f"saved: {out_path}")
    except Exception as e:
        print(f"error: {u} :: {e}")
PY

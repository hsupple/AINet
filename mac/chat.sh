#!/usr/bin/env bash
# Run OAC chat from the mac/ edition (resolves mac/db).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m ollama chat "$@"

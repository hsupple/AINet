#!/usr/bin/env bash
# Local UI that proxies all AI calls to the Windows tunnel (pathroom.org).
# Local Ollama: AINET_REMOTE_URL=local ./web.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export AINET_REMOTE_URL="${AINET_REMOTE_URL:-https://pathroom.org}"
exec python3 -m ollama web "$@"

#!/usr/bin/env bash
# Smoke-test Brave web_search (+ optional web_fetch) for the Mac AINet edition.
# Usage:
#   ./test_search.sh
#   ./test_search.sh "ATP synthase"
#   ./test_search.sh --fetch "brave search api"
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

DO_FETCH=0
QUERY="qwen3 ollama"
ARGS=()
for arg in "$@"; do
  case "$arg" in
    --fetch) DO_FETCH=1 ;;
    -h|--help)
      cat <<'EOF'
Usage: ./test_search.sh [--fetch] [query]

Checks BRAVE_API_KEY / AINET_BRAVE_API_KEY, runs web_search via ainet tools,
and optionally web_fetch on the first result URL.

Exit 0 on success, non-zero on failure.
EOF
      exit 0
      ;;
    *) ARGS+=("$arg") ;;
  esac
done
if ((${#ARGS[@]})); then
  QUERY="${ARGS[*]}"
fi

if [[ -z "${BRAVE_API_KEY:-}${AINET_BRAVE_API_KEY:-}" ]]; then
  echo "FAIL: set BRAVE_API_KEY or AINET_BRAVE_API_KEY first" >&2
  echo "  export BRAVE_API_KEY=\"your_token_here\"" >&2
  exit 2
fi

echo "→ web_search query=$(printf %q "$QUERY")"
python3 - "$QUERY" "$DO_FETCH" <<'PY'
import json
import sys

from ainet.tools.web import web_fetch, web_search

query = sys.argv[1]
do_fetch = sys.argv[2] == "1"

try:
    result = web_search(query, count=3)
except Exception as exc:
    print(f"FAIL: web_search: {exc}", file=sys.stderr)
    sys.exit(1)

ok = bool(result.get("ok"))
results = result.get("results") or []
print(json.dumps(
    {
        "ok": ok,
        "provider": result.get("provider"),
        "query": result.get("query"),
        "count": len(results),
        "results": [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "snippet": (r.get("snippet") or "")[:160],
            }
            for r in results
            if isinstance(r, dict)
        ],
        "error": result.get("error"),
    },
    indent=2,
    ensure_ascii=False,
))

if not ok or not results:
    print("FAIL: no search hits", file=sys.stderr)
    sys.exit(1)

print("PASS: web_search")

if do_fetch:
    url = results[0].get("url") or ""
    if not url:
        print("FAIL: first hit has no url", file=sys.stderr)
        sys.exit(1)
    print(f"→ web_fetch url={url}")
    try:
        fetched = web_fetch(url, max_chars=800)
    except Exception as exc:
        print(f"FAIL: web_fetch: {exc}", file=sys.stderr)
        sys.exit(1)
    text = (fetched.get("text") or "")[:400]
    print(json.dumps(
        {
            "ok": fetched.get("ok"),
            "url": fetched.get("url") or url,
            "chars": len(fetched.get("text") or ""),
            "preview": text,
            "error": fetched.get("error"),
        },
        indent=2,
        ensure_ascii=False,
    ))
    if not fetched.get("ok"):
        print("FAIL: web_fetch", file=sys.stderr)
        sys.exit(1)
    print("PASS: web_fetch")

print("OK search stack")
PY

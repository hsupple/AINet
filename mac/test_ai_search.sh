#!/usr/bin/env bash
# Test whether OAC actually *calls* web_search (not just that Brave works).
# Usage:
#   ./test_ai_search.sh
#   ./test_ai_search.sh "What is the Brave Search API rate limit for free plans?"
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

QUERY="${*:-Look up the official Ollama library page for qwen3 and tell me the model family name from the search results.}"

if [[ -z "${BRAVE_API_KEY:-}${AINET_BRAVE_API_KEY:-}" ]]; then
  echo "FAIL: set BRAVE_API_KEY or AINET_BRAVE_API_KEY first" >&2
  exit 2
fi

echo "→ ping Ollama"
python3 -m ollama ping >/dev/null

echo "→ OAC research ask (must call web_search)"
echo "  query: $QUERY"
python3 - "$QUERY" <<'PY'
import json
import sys

from ollama.client import OllamaClient, OllamaError
from ollama.config import OllamaConfig
from ollama.modes import get_mode
from ollama.session import ChatSession

query = sys.argv[1]
prompt = (
    "You must use the web_search tool for this request. "
    "Do not answer from memory. Call web_search first, then answer briefly from the results.\n\n"
    f"User request: {query}"
)

cfg = OllamaConfig.from_env()
# Keep this test ephemeral — no SOI / no conversation store noise
session = ChatSession(
    mode=get_mode("research"),
    config=cfg,
    auto_mode=False,
    persist_conversation=False,
    resume_session=False,
)

try:
    reply = session.ask(prompt)
except OllamaError as exc:
    print(f"FAIL: Ollama error: {exc}", file=sys.stderr)
    sys.exit(1)

calls: list[dict] = []
for msg in session.messages:
    if msg.get("role") != "assistant":
        continue
    for call in msg.get("tool_calls") or []:
        fn = call.get("function") or {}
        name = fn.get("name") or ""
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {"_raw": args}
        calls.append({"name": name, "arguments": args})

search_calls = [c for c in calls if c["name"] == "web_search"]
fetch_calls = [c for c in calls if c["name"] == "web_fetch"]

print(json.dumps(
    {
        "model": cfg.model,
        "mode": session.mode.id,
        "tool_calls": calls,
        "web_search_called": bool(search_calls),
        "web_fetch_called": bool(fetch_calls),
        "reply_preview": (reply or "")[:500],
    },
    indent=2,
    ensure_ascii=False,
))

if not search_calls:
    print(
        "FAIL: model did not call web_search "
        f"(tools used: {[c['name'] for c in calls] or 'none'})",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"PASS: OAC called web_search ({len(search_calls)}x)")
if fetch_calls:
    print(f"note: also called web_fetch ({len(fetch_calls)}x)")
print("OK AI search-tool use")
PY

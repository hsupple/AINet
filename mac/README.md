# AINet (macOS)

Mac edition of AINet — same OAC/SOI + personal DB system, hosted on **macOS** instead of Windows.

```
Mic → ESP32 → Mac (Ollama + AINet tools) → ESP32 → Speaker
```

This folder is self-contained: `ainet/`, `ollama/`, and `db/` live here. The repo root remains the Windows-oriented tree.

## Setup

1. Install [Ollama](https://ollama.com) for Mac (or `brew install ollama` + `brew services start ollama`) and pull a model (default: `ollama pull qwen3:8b`).
2. From this directory:

```bash
cd mac
export BRAVE_API_KEY="your_token_here"   # optional; for web_search
# or put that line in ~/.zshrc
./chat.sh
```

## Commands

```bash
cd mac

# OAC live chat (streams tokens; shows tool calls; SOI wakes after idle)
./chat.sh
./chat.sh --mode research --topic "ATP Synthase"

# DB tools
./ainet.sh list-tools
./ainet.sh call list_dir '{"path":"."}'

# Brave search smoke test (needs BRAVE_API_KEY)
./test_search.sh
./test_search.sh --fetch "ATP synthase"
# Does OAC actually call web_search?
./test_ai_search.sh


# SOI
./chat.sh  # includes idle SOI + live SOI logs in the chat
python3 -m ollama soi-status
python3 -m ollama soi-run --phase auto
# Persistent log: mac/db/runtime/soi/events.jsonl
```

Always run with `mac/` as the working directory (or `PYTHONPATH` including this folder) so `db/` resolves to `mac/db`.

## Brave Search

```bash
export BRAVE_API_KEY="your_token_here"
```

## Notes

- DB paths still use forward slashes: `Hayden/Read.json`.
- Personal/runtime data under `mac/db/runtime/` stays local (see `.gitignore`).
- Keep root (Windows) and `mac/` DBs separate unless you intentionally sync them.

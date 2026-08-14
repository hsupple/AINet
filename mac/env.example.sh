#!/usr/bin/env bash
# Optional: load Brave key for this shell session before chatting.
# Usage: source ./env.example.sh
#        export BRAVE_API_KEY="..."
export AINET_REMOTE_URL="${AINET_REMOTE_URL:-https://pathroom.org}"
# Local Ollama only if you explicitly disable the tunnel:
# export AINET_REMOTE_URL=local
export AINET_OLLAMA_HOST="${AINET_OLLAMA_HOST:-http://127.0.0.1:11434}"
export AINET_OLLAMA_MODEL="${AINET_OLLAMA_MODEL:-qwen3:8b}"
export AINET_OAC_THINK="${AINET_OAC_THINK:-0}"   # always off unless you flip to 1
export AINET_SOI_THINK="${AINET_SOI_THINK:-0}"
export AINET_SOI_TIMEOUT="${AINET_SOI_TIMEOUT:-600}"  # seconds; tools need headroom
# Cloudflare Access service token (needed once email restriction is actually on):
# export CF_ACCESS_CLIENT_ID="..."
# export CF_ACCESS_CLIENT_SECRET="..."
# export BRAVE_API_KEY="your_token_here"
# export AINET_BRAVE_API_KEY="your_token_here"

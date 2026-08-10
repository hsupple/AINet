#!/usr/bin/env bash
# Optional: load Brave key for this shell session before chatting.
# Usage: source ./env.example.sh
#        export BRAVE_API_KEY="..."
export AINET_OLLAMA_HOST="${AINET_OLLAMA_HOST:-http://127.0.0.1:11434}"
export AINET_OLLAMA_MODEL="${AINET_OLLAMA_MODEL:-llama3.2}"
# export BRAVE_API_KEY="your_token_here"
# export AINET_BRAVE_API_KEY="your_token_here"

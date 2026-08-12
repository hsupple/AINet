#!/usr/bin/env python3
"""Launch the SOI test harness web UI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ollama.config import default_db_root
from ollama.soi_test_server import build_config, serve


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SOI test harness web UI (ephemeral, never writes db/)")
    p.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=1112, help="Port (default: 1112)")
    p.add_argument("--db", type=Path, default=None, help="Source db/ to read (default: repo db/)")
    p.add_argument("--model", default=None, help="Ollama model override")
    p.add_argument("--ollama-host", default=None, help="Ollama host override")
    args = p.parse_args(argv)

    config = build_config(
        host=args.ollama_host,
        model=args.model,
        db=(args.db or default_db_root()).resolve(),
    )
    serve(host=args.host, port=args.port, config=config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

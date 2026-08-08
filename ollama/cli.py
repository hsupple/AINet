"""CLI for AINet Ollama runtime (OAC live + SOI idle).

Examples (from repo root):
  python -m ollama list-modes
  python -m ollama chat --mode companion
  python -m ollama soi-run
  python -m ollama soi-status
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from ollama.client import OllamaClient, OllamaError
from ollama.config import OllamaConfig
from ollama.idle import IdleSOIWatcher
from ollama.modes import DEFAULT_MODE_ID, get_mode, list_modes
from ollama.router import suggest_mode
from ollama.session import ChatSession
from ollama.soi_worker import SOIWorker


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    parser = argparse.ArgumentParser(prog="ollama-ainet", description="AINet OAC/SOI runtime")
    parser.add_argument("--host", default=None, help="Ollama base URL")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--db", type=Path, default=None, help="Database root")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-modes", help="List available modes")
    show = sub.add_parser("show-prompt", help="Print the system prompt for a mode")
    show.add_argument("mode", help="Mode id")

    chat = sub.add_parser("chat", help="OAC interactive or one-shot chat (SOI runs after idle)")
    chat.add_argument("--mode", default=DEFAULT_MODE_ID, help="OAC flavor mode id")
    chat.add_argument("--topic", default=None, help="Research topic title")
    chat.add_argument("--auto-mode", dest="auto_mode", action="store_true", default=None)
    chat.add_argument("--no-auto-mode", dest="auto_mode", action="store_false")
    chat.add_argument("--no-soi", action="store_true", help="Disable idle SOI watcher")
    chat.add_argument(
        "--soi-idle",
        type=float,
        default=None,
        help="Seconds of OAC silence before SOI filing (default 45)",
    )
    chat.add_argument(
        "--soi-read-idle",
        type=float,
        default=None,
        help="Seconds of OAC silence before SOI Read refresh (default 600)",
    )
    chat.add_argument("message", nargs="?", default=None, help="Optional one-shot message")

    route = sub.add_parser("route", help="Preview OAC flavor auto-route")
    route.add_argument("--current", default=DEFAULT_MODE_ID)
    route.add_argument("message")

    sub.add_parser("ping", help="Check Ollama connectivity")
    soi_run = sub.add_parser("soi-run", help="Run SOI now (filing and/or Read refresh)")
    soi_run.add_argument(
        "--phase",
        choices=("filing", "read_refresh", "auto"),
        default="auto",
        help="filing=phase1, read_refresh=phase2, auto=filing if pending else read_refresh if needed",
    )
    sub.add_parser("soi-status", help="Show SOI pending work / last state")

    args = parser.parse_args(argv)
    config = OllamaConfig.from_env()
    updates = {}
    if args.host:
        updates["host"] = args.host.rstrip("/")
    if args.model:
        updates["model"] = args.model
    if args.db:
        updates["db_root"] = args.db
    if getattr(args, "auto_mode", None) is not None:
        updates["auto_mode"] = args.auto_mode
    if getattr(args, "no_soi", False):
        updates["soi_enabled"] = False
    if getattr(args, "soi_idle", None) is not None:
        updates["soi_idle_seconds"] = args.soi_idle
    if getattr(args, "soi_read_idle", None) is not None:
        updates["soi_read_refresh_idle_seconds"] = args.soi_read_idle
    if updates:
        config = replace(config, **updates)

    if args.command == "list-modes":
        for mode in list_modes():
            print(f"{mode.id}\t{mode.role}\t{mode.name}\t{mode.description}")
        return 0

    if args.command == "show-prompt":
        try:
            mode = get_mode(args.mode)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        sys.stdout.write(mode.prompt if mode.prompt.endswith("\n") else mode.prompt + "\n")
        return 0

    if args.command == "ping":
        client = OllamaClient(config)
        try:
            models = client.list_models()
        except OllamaError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}))
            return 1
        print(
            json.dumps(
                {"ok": True, "host": config.host, "models": models, "configured_model": config.model},
                indent=2,
            )
        )
        return 0

    if args.command == "route":
        decision = suggest_mode(args.message, args.current)
        print(
            json.dumps(
                {
                    "current": args.current,
                    "suggested": decision.mode_id,
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                    "would_switch": (
                        decision.mode_id != args.current
                        and decision.confidence >= config.auto_mode_min_confidence
                    ),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "soi-status":
        worker = SOIWorker(config)
        state = None
        if worker.state_path.exists():
            try:
                state = json.loads(worker.state_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                state = None
        print(
            json.dumps(
                {
                    "soi_idle_seconds": config.soi_idle_seconds,
                    "soi_read_refresh_idle_seconds": config.soi_read_refresh_idle_seconds,
                    "pending_changelog": len(worker.pending_changelog()),
                    "pending_inbox": len(worker.pending_inbox()),
                    "has_filing_work": worker.has_filing_work(),
                    "needs_read_refresh": worker.needs_read_refresh(),
                    "read_json_count": len(worker.list_read_json_paths()),
                    "state_file": str(worker.state_path),
                    "state": state,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "soi-run":
        worker = SOIWorker(config)
        phase = getattr(args, "phase", "auto")
        if phase == "filing":
            result = worker.run_filing()
        elif phase == "read_refresh":
            result = worker.run_read_refresh()
        else:
            if worker.has_filing_work():
                result = worker.run_filing()
            elif worker.needs_read_refresh():
                result = worker.run_read_refresh()
            else:
                result = {"ok": True, "ran": False, "reason": "no filing or read-refresh work"}
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("ok") else 1

    if args.command == "chat":
        try:
            mode = get_mode(args.mode)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if mode.role != "oac":
            print("chat is for OAC flavors (companion/conversation/research/planner). Use soi-run for SOI.")
            return 2
        session = ChatSession(
            mode=mode,
            config=config,
            topic_title=args.topic,
            auto_mode=config.auto_mode,
        )
        if args.topic:
            print(f"(topic bound: {session.topic['path']})")
        if args.message is not None:
            try:
                reply = session.ask(args.message)
            except OllamaError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            print(reply)
            return 0

        watcher = IdleSOIWatcher(
            session,
            config,
            on_status=lambda msg: print(msg, flush=True),
        )
        watcher.start()
        print(
            f"AINet OAC  mode={session.mode.id}  model={config.model}  "
            f"soi_file={config.soi_idle_seconds:.0f}s  "
            f"soi_read={config.soi_read_refresh_idle_seconds:.0f}s  "
            f"db={config.db_root}"
        )
        if session.session_id:
            print(f"oac_session={session.session_id}")
        if session.topic:
            print(f"topic={session.topic['title']}  path={session.topic['path']}")
        print("Commands: /exit  /reset  /mode <id>  /auto  /topic <title>  /soi")
        try:
            while True:
                try:
                    line = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not line:
                    continue
                session.touch()
                if line in {"/exit", "/quit"}:
                    break
                if line == "/reset":
                    session.reset()
                    print("(new OAC session; short-term memory cleared)")
                    continue
                if line == "/auto":
                    session.auto_mode = True
                    session.mode_locked = False
                    print("(auto flavor on)")
                    continue
                if line == "/soi":
                    result = SOIWorker(config).run_once()
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                    session.touch()
                    continue
                if line.startswith("/mode "):
                    new_id = line.split(None, 1)[1].strip()
                    try:
                        session.set_mode(new_id, lock=True)
                    except KeyError as exc:
                        print(str(exc))
                        continue
                    if session.mode.role != "oac":
                        print("Stay on OAC flavors in chat; use /soi or soi-run for SOI.")
                        session.set_mode(DEFAULT_MODE_ID, lock=False)
                        continue
                    print(f"(locked OAC mode={session.mode.id}; /auto to unlock)")
                    continue
                if line.startswith("/topic "):
                    title = line.split(None, 1)[1].strip()
                    if not title:
                        print("usage: /topic <title>")
                        continue
                    info = session.bind_topic(title)
                    print(f"(topic bound: {info['path']}; mode={session.mode.id})")
                    continue
                try:
                    reply = session.ask(line)
                except OllamaError as exc:
                    print(f"error: {exc}")
                    continue
                print(reply)
        finally:
            watcher.stop()
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

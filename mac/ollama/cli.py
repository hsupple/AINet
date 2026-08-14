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
import os
import queue
import sys
import tempfile
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

from ollama.client import OllamaClient, OllamaError
from ollama.config import OllamaConfig
from ollama.modes import DEFAULT_MODE_ID, get_mode, list_modes
from ollama.remote import RemoteAinetClient, RemoteError
from ollama.router import suggest_mode


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass


def _play_wav_bytes(data: bytes) -> None:
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(data)
            path = tmp.name
        if sys.platform == "win32":
            import winsound

            winsound.PlaySound(path, winsound.SND_FILENAME)
        else:
            import subprocess

            for cmd in (
                ["afplay", path],
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
                ["aplay", path],
            ):
                try:
                    subprocess.run(cmd, check=False, capture_output=True)
                    break
                except OSError:
                    continue
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


def _chat_turn(session: Any, config: OllamaConfig, line: str, *, voice: bool) -> str:
    """Stream tokens to stdout; optionally speak sentence chunks via TTS."""
    from ollama.tts import SpeechPipeline, TtsClient, TtsConfig
    pipe: SpeechPipeline | None = None
    play_q: queue.Queue[bytes | None] | None = None
    player: threading.Thread | None = None

    if voice and config.tts_enabled:
        client = TtsClient(
            TtsConfig(
                url=config.tts_url,
                enabled=True,
                language=config.tts_language,
                timeout_s=config.tts_timeout_s,
            )
        )
        if not client.healthy():
            print(f"(voice offline @ {config.tts_url})", file=sys.stderr, flush=True)
        else:
            play_q = queue.Queue()

            def _player() -> None:
                assert play_q is not None
                while True:
                    item = play_q.get()
                    if item is None:
                        return
                    _play_wav_bytes(item)

            player = threading.Thread(target=_player, name="ainet-tts-play", daemon=True)
            player.start()

            def _on_audio(wav: bytes, _seq: int, _text: str) -> None:
                assert play_q is not None
                play_q.put(wav)

            def _on_err(err: str) -> None:
                print(f"\n(voice error: {err})", file=sys.stderr, flush=True)

            pipe = SpeechPipeline(client, on_audio=_on_audio, on_error=_on_err, enabled=True)

    def _on_token(delta: str) -> None:
        sys.stdout.write(delta)
        sys.stdout.flush()
        if pipe is not None:
            pipe.feed(delta)

    try:
        reply = session.ask(line, stream=True, on_token=_on_token)
    finally:
        if pipe is not None:
            pipe.close()
        if play_q is not None:
            play_q.put(None)
        if player is not None:
            player.join(timeout=300.0)
        sys.stdout.write("\n")
        sys.stdout.flush()
    return reply


def _remote_chat_turn(client: RemoteAinetClient, line: str) -> str:
    reply_parts: list[str] = []
    try:
        for event in client.chat_stream(line):
            et = str(event.get("type") or "")
            if et == "token":
                delta = str(event.get("text") or "")
                if delta:
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                    reply_parts.append(delta)
            elif et == "error":
                raise RemoteError(str(event.get("error") or "remote chat error"))
            elif et == "done":
                blob = str(event.get("reply") or "")
                if blob and not reply_parts:
                    sys.stdout.write(blob)
                    sys.stdout.flush()
                    reply_parts.append(blob)
    finally:
        sys.stdout.write("\n")
        sys.stdout.flush()
    return "".join(reply_parts)


def _run_remote_chat(client: RemoteAinetClient, args: argparse.Namespace) -> int:
    if args.message is not None:
        try:
            _remote_chat_turn(client, args.message)
        except RemoteError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    try:
        status = client.status()
    except RemoteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"AINet OAC  remote={client.base}  "
        f"model={status.get('model') or ''}  mode={status.get('mode') or ''}  "
        f"session={status.get('session_id') or ''}"
    )
    print("Commands: /exit  /reset  /mode <id>  /soi")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in {"/exit", "/quit"}:
            break
        try:
            if line == "/reset":
                client.reset()
                print("(new OAC session on Windows host)")
                continue
            if line == "/soi":
                print(json.dumps(client.run_soi(), indent=2, ensure_ascii=False))
                continue
            if line.startswith("/mode "):
                payload = client.set_mode(line.split(None, 1)[1].strip())
                if not payload.get("ok"):
                    print(payload.get("error") or payload)
                    continue
                print(f"(mode={payload.get('mode')})")
                continue
            _remote_chat_turn(client, line)
        except RemoteError as exc:
            print(f"error: {exc}")
            continue
    return 0


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    parser = argparse.ArgumentParser(prog="ollama-ainet", description="AINet OAC/SOI runtime")
    parser.add_argument("--host", default=None, help="Ollama base URL (local mode only)")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--db", type=Path, default=None, help="Database root")
    parser.add_argument(
        "--remote",
        default=None,
        help="Windows AINet URL (default https://pathroom.org). Empty/local disables.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local Ollama instead of the Windows tunnel",
    )
    parser.add_argument(
        "--oac-think",
        dest="oac_think",
        action="store_true",
        default=None,
        help="Enable Qwen3 thinking for OAC (default off)",
    )
    parser.add_argument(
        "--no-oac-think",
        dest="oac_think",
        action="store_false",
        help="Disable Qwen3 thinking for OAC",
    )
    parser.add_argument(
        "--soi-think",
        dest="soi_think",
        action="store_true",
        default=None,
        help="Enable Qwen3 thinking for SOI (default off)",
    )
    parser.add_argument(
        "--no-soi-think",
        dest="soi_think",
        action="store_false",
        help="Disable Qwen3 thinking for SOI",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-modes", help="List available modes")
    show = sub.add_parser("show-prompt", help="Print the system prompt for a mode")
    show.add_argument("mode", help="Mode id")

    chat = sub.add_parser("chat", help="OAC interactive or one-shot chat (SOI runs after idle)")
    chat.add_argument("--mode", default=DEFAULT_MODE_ID, help="OAC flavor mode id")
    chat.add_argument("--auto-mode", dest="auto_mode", action="store_true", default=None)
    chat.add_argument("--no-auto-mode", dest="auto_mode", action="store_false")
    chat.add_argument("--no-soi", action="store_true", help="Disable idle SOI watcher")
    chat.add_argument(
        "--voice",
        dest="voice",
        action="store_true",
        default=None,
        help="Speak replies via Qwen3-TTS (default: on when TTS enabled)",
    )
    chat.add_argument(
        "--no-voice",
        dest="voice",
        action="store_false",
        help="Disable spoken replies",
    )
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

    web = sub.add_parser("web", help="LAN web chat UI (default 0.0.0.0:1111)")
    web.add_argument("--bind", default="0.0.0.0", help="Bind address (0.0.0.0 = all interfaces)")
    web.add_argument("--port", type=int, default=1111, help="TCP port")
    web.add_argument("--mode", default=DEFAULT_MODE_ID, help="Initial OAC mode")
    web.add_argument("--no-soi", action="store_true", help="Disable idle SOI watcher")
    web.add_argument("--no-tts", action="store_true", help="Disable Qwen3-TTS voice")

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
    if getattr(args, "local", False):
        updates["remote_url"] = ""
    elif getattr(args, "remote", None) is not None:
        raw = str(args.remote).strip().rstrip("/")
        updates["remote_url"] = "" if raw.lower() in {"", "0", "false", "off", "local", "none"} else raw
    if getattr(args, "oac_think", None) is not None:
        updates["oac_think"] = args.oac_think
    if getattr(args, "soi_think", None) is not None:
        updates["soi_think"] = args.soi_think
    if getattr(args, "auto_mode", None) is not None:
        updates["auto_mode"] = args.auto_mode
    if getattr(args, "no_soi", False):
        updates["soi_enabled"] = False
    if getattr(args, "no_tts", False):
        updates["tts_enabled"] = False
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

    if args.command == "web":
        from ollama.webserver import serve

        if getattr(args, "no_soi", False):
            config = replace(config, soi_enabled=False)
        try:
            get_mode(args.mode)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        serve(host=args.bind, port=args.port, config=config, mode_id=args.mode)
        return 0

    if args.command == "ping":
        if config.remote_url:
            client = RemoteAinetClient(config)
            try:
                status = client.status()
            except RemoteError as exc:
                print(json.dumps({"ok": False, "error": str(exc)}))
                return 1
            print(
                json.dumps(
                    {
                        "ok": True,
                        "remote": client.base,
                        "model": status.get("model"),
                        "mode": status.get("mode"),
                        "session_id": status.get("session_id"),
                    },
                    indent=2,
                )
            )
            return 0
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
        if config.remote_url:
            client = RemoteAinetClient(config)
            try:
                print(json.dumps(client.status(), indent=2, ensure_ascii=False))
            except RemoteError as exc:
                print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
                return 1
            return 0
        from ollama.soi_worker import SOIWorker

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
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "soi-run":
        if config.remote_url:
            client = RemoteAinetClient(config)
            try:
                result = client.run_soi()
            except RemoteError as exc:
                print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
                return 1
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result.get("ok") else 1
        from ollama.soi_worker import SOIWorker

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
        if config.remote_url:
            return _run_remote_chat(RemoteAinetClient(config), args)
        try:
            mode = get_mode(args.mode)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if mode.role != "oac":
            print("chat is for OAC flavors (companion/conversation/planner). Use soi-run for SOI.")
            return 2
        from ollama.idle import IdleSOIWatcher
        from ollama.session import ChatSession
        from ollama.soi_log import status_line
        from ollama.soi_worker import SOIWorker

        voice = True if getattr(args, "voice", None) is None else bool(args.voice)
        session = ChatSession(
            mode=mode,
            config=config,
            auto_mode=config.auto_mode,
        )
        if args.message is not None:
            try:
                _chat_turn(session, config, args.message, voice=voice)
            except OllamaError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            return 0

        watcher = IdleSOIWatcher(
            session,
            config,
            on_status=lambda msg: print(status_line(msg), flush=True),
        )
        watcher.start()
        print(
            f"AINet OAC  mode={session.mode.id}  model={config.model}  "
            f"soi_file={config.soi_idle_seconds:.0f}s  "
            f"soi_read={config.soi_read_refresh_idle_seconds:.0f}s  "
            f"soi_timeout={config.soi_timeout_s:.0f}s  "
            f"think oac={int(config.oac_think)} soi={int(config.soi_think)}  "
            f"voice={'on' if voice and config.tts_enabled else 'off'}  "
            f"db={config.db_root}"
        )
        if session.session_id:
            print(f"oac_session={session.session_id}")
        print("Commands: /exit  /reset  /mode <id>  /auto  /soi  /voice")
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
                if line == "/voice":
                    voice = not voice
                    print(f"(voice={'on' if voice else 'off'})")
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
                try:
                    _chat_turn(session, config, line, voice=voice)
                except OllamaError as exc:
                    print(f"error: {exc}")
                    continue
        finally:
            watcher.stop()
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

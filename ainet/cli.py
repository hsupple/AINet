"""CLI entrypoint for AI tool calls.

Windows (PowerShell) examples:
  python -m ainet list-tools
  python -m ainet call list_dir '{\"path\":\".\"}'
  python -m ainet call create_folder --args-file args.json
  Get-Content args.json -Raw | python -m ainet call create_folder --args-stdin

macOS/Linux examples:
  python -m ainet call list_dir '{"path":"."}'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _configure_stdio() -> None:
    """Prefer UTF-8 on Windows consoles so JSON output does not crash on unicode."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass


def default_db_root() -> Path:
    return Path(__file__).resolve().parent.parent / "db"


def _load_arguments(raw: str | None, args_file: Path | None, args_stdin: bool) -> dict:
    if sum(bool(x) for x in (raw and raw != "{}", args_file, args_stdin)) > 1:
        raise SystemExit("Use only one of: positional arguments, --args-file, or --args-stdin")

    if args_stdin:
        text = sys.stdin.read()
    elif args_file is not None:
        text = args_file.read_text(encoding="utf-8")
    else:
        text = raw if raw is not None else "{}"

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        print(json.dumps({"ok": False, "error": f"Invalid JSON arguments: {exc}"}))
        raise SystemExit(2) from exc
    if not isinstance(payload, dict):
        print(json.dumps({"ok": False, "error": "Arguments must be a JSON object"}))
        raise SystemExit(2)
    return payload


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()

    # Import after stdio config so import errors still print cleanly.
    from ainet.tools.ops import DatabaseTools
    from ainet.tools.registry import TOOL_DEFINITIONS, dispatch

    parser = argparse.ArgumentParser(prog="ainet", description="AINet database tools (Windows/macOS/Linux)")
    parser.add_argument(
        "--db",
        type=Path,
        default=default_db_root(),
        help="Path to database root (default: <repo>/db)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-tools", help="Print Ollama tool definitions as JSON")

    defaults_parser = sub.add_parser(
        "defaults",
        help="List or show editable default JSON templates under ainet/defaults/",
    )
    defaults_parser.add_argument(
        "name",
        nargs="?",
        help="Optional template filename to print (e.g. Profile.json)",
    )

    call = sub.add_parser("call", help="Call a tool by name with a JSON arguments object")
    call.add_argument("name", help="Tool name")
    call.add_argument(
        "arguments",
        nargs="?",
        default="{}",
        help='JSON object of arguments. On PowerShell prefer --args-file or --args-stdin.',
    )
    call.add_argument(
        "--args-file",
        type=Path,
        help="Read JSON arguments from a UTF-8 file (best for Windows).",
    )
    call.add_argument(
        "--args-stdin",
        action="store_true",
        help="Read JSON arguments from stdin.",
    )

    args = parser.parse_args(argv)
    db = DatabaseTools(args.db)

    if args.command == "list-tools":
        json.dump(TOOL_DEFINITIONS, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0

    if args.command == "defaults":
        from ainet.defaults import defaults_dir, list_templates, package_template_text

        if args.name:
            try:
                text = package_template_text(args.name)
            except FileNotFoundError as exc:
                print(json.dumps({"ok": False, "error": str(exc)}))
                return 1
            sys.stdout.write(text if text.endswith("\n") else text + "\n")
            return 0
        json.dump(
            {
                "ok": True,
                "templates": list_templates(),
                "edit_dir": str(defaults_dir()),
                "hint": "Edit files in edit_dir. _ainet blocks are stripped when seeding DB files.",
            },
            sys.stdout,
            indent=2,
            ensure_ascii=False,
        )
        sys.stdout.write("\n")
        return 0

    if args.command == "call":
        payload = _load_arguments(args.arguments, args.args_file, args.args_stdin)
        result = dispatch(db, args.name, payload)
        json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0 if result.get("ok") else 1

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

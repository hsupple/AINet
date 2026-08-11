"""Parse tool calls out of SOI prose and map junk paths onto Folderrules domains."""

from __future__ import annotations

import json
import re
from typing import Any

_TOOL_XML = re.compile(r"<tool_call>\s*", re.S)
_FENCE = re.compile(r"```(?:json)?\s*", re.I)
_PLANS_COURSES = re.compile(r"(?i)^Hayden/(?:Plans|Planner)/Courses/")
_PLANS_ROOT = re.compile(r"(?i)^Hayden/(?:Plans|Planner)(?=/|$)")


def remap_folderrules_path(path: str) -> str:
    p = (path or "").replace("\\", "/").strip().strip('"')
    p = _PLANS_COURSES.sub("School/Courses/", p)
    p = _PLANS_ROOT.sub("School", p)
    if re.search(r"(?i)^School/Schedule\.json$", p):
        return "School/Plan.json"
    return p


def normalize_soi_tool(name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    args = dict(args or {})
    for key in ("path", "folder_path", "src", "dest", "file"):
        if key in args and isinstance(args[key], str):
            args[key] = remap_folderrules_path(args[key])
    if "path" not in args and args.get("folder_path"):
        args["path"] = args["folder_path"]
    path = str(args.get("path") or "")
    if name == "create_folder" and "/Courses/" in path:
        return "create_cop", {"path": path, "kind": "course", "summary": args.get("summary")}
    if name == "create_folder" and "/Projects/" in path:
        return "create_cop", {"path": path, "kind": "project", "summary": args.get("summary")}
    if name == "create_cop":
        kind = str(args.get("kind") or args.get("cop_type") or "").strip()
        if not kind:
            kind = "course" if "/Courses/" in path or path.startswith("School/") else "project"
        if path.endswith(".json"):
            return "write_json", {
                "path": path,
                "data": args.get("data") if isinstance(args.get("data"), dict) else {},
                "create": True,
                "summary": args.get("summary"),
            }
        return "create_cop", {"path": path, "kind": kind, "summary": args.get("summary")}
    return name, args


def _decode_objects_after(marker: re.Pattern[str], text: str) -> list[Any]:
    out: list[Any] = []
    idx = 0
    decoder = json.JSONDecoder()
    while True:
        match = marker.search(text, idx)
        if not match:
            break
        start = match.end()
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            idx = start + 1
            continue
        out.append(obj)
        idx = start + end
    return out


def parse_content_tool_calls(text: str) -> list[dict[str, Any]]:
    """Turn narrated JSON / <tool_call> blobs into Ollama-shaped tool_calls."""
    calls: list[dict[str, Any]] = []
    text = text or ""
    for obj in _decode_objects_after(_TOOL_XML, text):
        calls.extend(_calls_from_obj(obj))
    for obj in _decode_objects_after(_FENCE, text):
        calls.extend(_calls_from_obj(obj))
    if not calls:
        calls.extend(_calls_from_loose_json(text))
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for call in calls:
        fn = call.get("function") or {}
        name = str(fn.get("name") or "")
        args = fn.get("arguments") if isinstance(fn.get("arguments"), dict) else {}
        name, args = normalize_soi_tool(name, args)
        if not name:
            continue
        key = (name, json.dumps(args, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        out.append({"function": {"name": name, "arguments": args}})
    return out[:24]


def _wrap(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"function": {"name": name, "arguments": args}}


def _calls_from_obj(obj: Any) -> list[dict[str, Any]]:
    if not isinstance(obj, dict):
        return []
    if obj.get("name") and (obj.get("arguments") is not None or obj.get("args") is not None):
        args = obj.get("arguments") or obj.get("args") or {}
        if isinstance(args, dict):
            return [_wrap(str(obj.get("name")), args)]
    out: list[dict[str, Any]] = []
    folders = obj.get("create_folder")
    if isinstance(folders, dict):
        for path in folders:
            if isinstance(path, str) and path.strip():
                out.append(_wrap("create_folder", {"path": path}))
    elif isinstance(folders, list):
        for path in folders:
            if isinstance(path, str) and path.strip():
                out.append(_wrap("create_folder", {"path": path}))
    cops = obj.get("create_cop")
    if isinstance(cops, dict) and ("path" in cops or "folder_path" in cops):
        out.append(_wrap("create_cop", cops))
    elif isinstance(cops, dict):
        for path, spec in cops.items():
            if not isinstance(path, str):
                continue
            args: dict[str, Any] = {"path": path}
            if path.endswith(".json"):
                continue
            if isinstance(spec, dict) and (spec.get("kind") or spec.get("cop_type")):
                args["kind"] = spec.get("kind") or spec.get("cop_type")
            out.append(_wrap("create_cop", args))
    writes = obj.get("write_json")
    if isinstance(writes, dict) and "path" in writes:
        out.append(_wrap("write_json", writes))
    elif isinstance(writes, dict):
        for path, data in writes.items():
            if isinstance(path, str):
                out.append(_wrap("write_json", {"path": path, "data": data, "create": True}))
    return out


def _calls_from_loose_json(text: str) -> list[dict[str, Any]]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start or end - start > 8000:
        return []
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    if any(k in obj for k in ("create_folder", "create_cop", "write_json", "name")):
        return _calls_from_obj(obj)
    return []

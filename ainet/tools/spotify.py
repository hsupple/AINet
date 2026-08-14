"""Spotify Web API — OAuth + playback control for AINet OAC.

What this does: after Hayden authorizes once in the browser, OAC can check
what's playing, search the catalog, and control playback (play/pause/skip/
volume/queue) on an active Spotify device (desktop or phone app must be open).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

_API = "https://api.spotify.com/v1"
_AUTH = "https://accounts.spotify.com"
_SCOPES = " ".join(
    [
        "user-read-playback-state",
        "user-modify-playback-state",
        "user-read-currently-playing",
        "user-read-recently-played",
    ]
)
_LOCK = threading.RLock()
_PENDING: dict[str, dict[str, Any]] = {}

_DEFAULT_REDIRECT = "http://127.0.0.1:1111/auth/spotify/callback"


def _db_root() -> Path:
    raw = (os.environ.get("AINET_DB") or "").strip()
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parents[2] / "db"


def _runtime_dir() -> Path:
    d = _db_root() / "runtime" / "spotify"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _app_path() -> Path:
    return _runtime_dir() / "app.json"


def _token_path() -> Path:
    return _runtime_dir() / "tokens.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def get_app_config() -> dict[str, str]:
    file_cfg = _load_json(_app_path())
    client_id = (
        os.environ.get("AINET_SPOTIFY_CLIENT_ID")
        or str(file_cfg.get("client_id") or "")
    ).strip()
    client_secret = (
        os.environ.get("AINET_SPOTIFY_CLIENT_SECRET")
        or str(file_cfg.get("client_secret") or "")
    ).strip()
    redirect_uri = (
        os.environ.get("AINET_SPOTIFY_REDIRECT_URI")
        or str(file_cfg.get("redirect_uri") or "")
        or _DEFAULT_REDIRECT
    ).strip()
    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
    }


def save_app_config(
    *,
    client_id: str = "",
    client_secret: str = "",
    redirect_uri: str = "",
) -> dict[str, Any]:
    cfg = get_app_config()
    if client_id.strip():
        cfg["client_id"] = client_id.strip()
    if client_secret.strip():
        cfg["client_secret"] = client_secret.strip()
    if redirect_uri.strip():
        cfg["redirect_uri"] = redirect_uri.strip()
    if not cfg["client_id"] or not cfg["client_secret"]:
        return {"ok": False, "error": "client_id and client_secret are required"}
    cfg.setdefault("redirect_uri", _DEFAULT_REDIRECT)
    _save_json(
        _app_path(),
        {
            "client_id": cfg["client_id"],
            "client_secret": cfg["client_secret"],
            "redirect_uri": cfg["redirect_uri"],
        },
    )
    return {
        "ok": True,
        "client_id": cfg["client_id"],
        "redirect_uri": cfg["redirect_uri"],
        "configured": True,
    }


def _tokens() -> dict[str, Any]:
    return _load_json(_token_path())


def _save_tokens(data: dict[str, Any]) -> None:
    _save_json(_token_path(), data)


def connection_status() -> dict[str, Any]:
    cfg = get_app_config()
    tok = _tokens()
    connected = bool(tok.get("access_token") or tok.get("refresh_token"))
    return {
        "ok": True,
        "configured": bool(cfg["client_id"] and cfg["client_secret"]),
        "connected": connected,
        "redirect_uri": cfg["redirect_uri"],
        "has_refresh": bool(tok.get("refresh_token")),
        "expires_at": tok.get("expires_at"),
        "auth_url": "/auth/spotify" if cfg["client_id"] and cfg["client_secret"] else None,
        "summary": (
            "Spotify connected"
            if connected
            else (
                "Spotify app configured — visit /auth/spotify to link account"
                if cfg["client_id"] and cfg["client_secret"]
                else "Spotify not configured (need Client ID + Secret)"
            )
        ),
    }


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def begin_auth() -> dict[str, Any]:
    cfg = get_app_config()
    if not cfg["client_id"] or not cfg["client_secret"]:
        return {
            "ok": False,
            "error": (
                "Spotify Client ID/Secret missing. Set AINET_SPOTIFY_CLIENT_ID and "
                "AINET_SPOTIFY_CLIENT_SECRET, or save them via the credentials endpoint."
            ),
        }
    verifier = _b64url(secrets.token_bytes(64))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    state = _b64url(secrets.token_bytes(24))
    with _LOCK:
        _PENDING[state] = {
            "verifier": verifier,
            "created": time.time(),
        }
        # Drop stale pending states
        dead = [k for k, v in _PENDING.items() if time.time() - float(v.get("created") or 0) > 900]
        for k in dead:
            _PENDING.pop(k, None)
    params = {
        "client_id": cfg["client_id"],
        "response_type": "code",
        "redirect_uri": cfg["redirect_uri"],
        "scope": _SCOPES,
        "state": state,
        "code_challenge_method": "S256",
        "code_challenge": challenge,
        "show_dialog": "false",
    }
    url = f"{_AUTH}/authorize?" + urllib.parse.urlencode(params)
    return {"ok": True, "url": url, "state": state, "redirect_uri": cfg["redirect_uri"]}


def _http_form(url: str, data: dict[str, str], *, headers: dict[str, str] | None = None) -> dict[str, Any]:
    body = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            **(headers or {}),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(err_body) if err_body else {}
        except json.JSONDecodeError:
            payload = {"error": err_body or str(exc)}
        raise RuntimeError(
            str(payload.get("error_description") or payload.get("error") or err_body or exc)
        ) from exc


def _http_json(
    method: str,
    url: str,
    *,
    token: str,
    payload: dict[str, Any] | None = None,
    params: dict[str, str] | None = None,
) -> Any:
    if params:
        url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    data = None
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method.upper(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            if not raw:
                return {"ok": True, "status": resp.status}
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(err_body) if err_body else {}
        except json.JSONDecodeError:
            parsed = {}
        msg = (
            (parsed.get("error") or {}).get("message")
            if isinstance(parsed.get("error"), dict)
            else parsed.get("error")
        )
        raise RuntimeError(str(msg or err_body or f"HTTP {exc.code}")) from exc


def finish_auth(*, code: str, state: str) -> dict[str, Any]:
    code = (code or "").strip()
    state = (state or "").strip()
    if not code or not state:
        return {"ok": False, "error": "Missing code or state from Spotify"}
    with _LOCK:
        pending = _PENDING.pop(state, None)
    if not pending:
        return {"ok": False, "error": "Auth state expired or unknown — start again from /auth/spotify"}
    cfg = get_app_config()
    basic = base64.b64encode(
        f"{cfg['client_id']}:{cfg['client_secret']}".encode("utf-8")
    ).decode("ascii")
    try:
        data = _http_form(
            f"{_AUTH}/api/token",
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": cfg["redirect_uri"],
                "code_verifier": str(pending["verifier"]),
            },
            headers={"Authorization": f"Basic {basic}"},
        )
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    access = str(data.get("access_token") or "")
    if not access:
        return {"ok": False, "error": "Spotify did not return an access token"}
    expires_in = int(data.get("expires_in") or 3600)
    stored = {
        "access_token": access,
        "refresh_token": str(data.get("refresh_token") or _tokens().get("refresh_token") or ""),
        "token_type": str(data.get("token_type") or "Bearer"),
        "scope": str(data.get("scope") or _SCOPES),
        "expires_at": time.time() + expires_in - 30,
    }
    _save_tokens(stored)
    return {"ok": True, "connected": True, "summary": "Spotify account linked"}


def _refresh_access() -> str:
    cfg = get_app_config()
    tok = _tokens()
    refresh = str(tok.get("refresh_token") or "")
    if not refresh:
        raise RuntimeError("Spotify not connected — open /auth/spotify first")
    basic = base64.b64encode(
        f"{cfg['client_id']}:{cfg['client_secret']}".encode("utf-8")
    ).decode("ascii")
    data = _http_form(
        f"{_AUTH}/api/token",
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh,
        },
        headers={"Authorization": f"Basic {basic}"},
    )
    access = str(data.get("access_token") or "")
    if not access:
        raise RuntimeError("Failed to refresh Spotify token")
    expires_in = int(data.get("expires_in") or 3600)
    tok["access_token"] = access
    tok["expires_at"] = time.time() + expires_in - 30
    if data.get("refresh_token"):
        tok["refresh_token"] = str(data["refresh_token"])
    _save_tokens(tok)
    return access


def _access_token() -> str:
    with _LOCK:
        tok = _tokens()
        access = str(tok.get("access_token") or "")
        expires_at = float(tok.get("expires_at") or 0)
        if access and time.time() < expires_at:
            return access
        return _refresh_access()


def _api(method: str, path: str, *, payload: dict[str, Any] | None = None, params: dict[str, str] | None = None) -> Any:
    token = _access_token()
    url = path if path.startswith("http") else f"{_API}{path}"
    try:
        return _http_json(method, url, token=token, payload=payload, params=params)
    except RuntimeError as exc:
        # One retry on auth failure
        if "token" in str(exc).lower() or "unauthorized" in str(exc).lower():
            token = _refresh_access()
            return _http_json(method, url, token=token, payload=payload, params=params)
        raise


def _track_summary(item: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    artists = ", ".join(
        str(a.get("name") or "") for a in (item.get("artists") or []) if isinstance(a, dict)
    )
    album = item.get("album") if isinstance(item.get("album"), dict) else {}
    return {
        "id": item.get("id"),
        "uri": item.get("uri"),
        "name": item.get("name"),
        "artists": artists,
        "album": album.get("name"),
        "duration_ms": item.get("duration_ms"),
    }


def now_playing() -> dict[str, Any]:
    try:
        data = _api("GET", "/me/player/currently-playing")
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    if not data or data is True:
        return {"ok": True, "playing": False, "summary": "Nothing is currently playing"}
    item = data.get("item") if isinstance(data, dict) else None
    track = _track_summary(item if isinstance(item, dict) else None)
    is_playing = bool(isinstance(data, dict) and data.get("is_playing"))
    device = None
    try:
        playback = _api("GET", "/me/player")
        if isinstance(playback, dict) and isinstance(playback.get("device"), dict):
            device = {
                "id": playback["device"].get("id"),
                "name": playback["device"].get("name"),
                "type": playback["device"].get("type"),
                "volume_percent": playback["device"].get("volume_percent"),
            }
    except RuntimeError:
        pass
    summary = (
        f"{'Playing' if is_playing else 'Paused'}: {track.get('name') or 'Unknown'} — {track.get('artists') or '?'}"
        if track
        else ("Playing" if is_playing else "Nothing playing")
    )
    return {
        "ok": True,
        "playing": is_playing,
        "track": track or None,
        "progress_ms": data.get("progress_ms") if isinstance(data, dict) else None,
        "device": device,
        "summary": summary,
    }


def list_devices() -> dict[str, Any]:
    try:
        data = _api("GET", "/me/player/devices")
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    devices = []
    for d in (data.get("devices") if isinstance(data, dict) else None) or []:
        if not isinstance(d, dict):
            continue
        devices.append(
            {
                "id": d.get("id"),
                "name": d.get("name"),
                "type": d.get("type"),
                "is_active": bool(d.get("is_active")),
                "volume_percent": d.get("volume_percent"),
            }
        )
    active = next((d for d in devices if d.get("is_active")), None)
    return {
        "ok": True,
        "devices": devices,
        "active": active,
        "summary": (
            f"{len(devices)} device(s)"
            + (f"; active: {active.get('name')}" if active else "; none active — open Spotify on a device")
        ),
    }


def search_catalog(query: str, *, limit: int = 5, types: str = "track") -> dict[str, Any]:
    q = (query or "").strip()
    if not q:
        return {"ok": False, "error": "query is required"}
    limit = max(1, min(10, int(limit)))
    type_list = ",".join(t.strip() for t in (types or "track").split(",") if t.strip()) or "track"
    try:
        data = _api(
            "GET",
            "/search",
            params={"q": q, "type": type_list, "limit": str(limit)},
        )
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    tracks = []
    for item in ((data.get("tracks") or {}).get("items") if isinstance(data, dict) else None) or []:
        if isinstance(item, dict):
            tracks.append(_track_summary(item))
    return {
        "ok": True,
        "query": q,
        "tracks": tracks,
        "summary": (
            f"Found {len(tracks)} track(s)"
            + (f"; top: {tracks[0].get('name')} — {tracks[0].get('artists')}" if tracks else "")
        ),
    }


def play(
    *,
    uri: str = "",
    query: str = "",
    device_id: str = "",
) -> dict[str, Any]:
    track_uri = (uri or "").strip()
    if not track_uri and (query or "").strip():
        found = search_catalog(query, limit=1)
        if not found.get("ok"):
            return found
        tracks = found.get("tracks") or []
        if not tracks:
            return {"ok": False, "error": f"No tracks found for {query!r}"}
        track_uri = str(tracks[0].get("uri") or "")
    params: dict[str, str] = {}
    if device_id.strip():
        params["device_id"] = device_id.strip()
    payload: dict[str, Any] | None = None
    if track_uri:
        if track_uri.startswith("spotify:playlist:") or track_uri.startswith("spotify:album:"):
            payload = {"context_uri": track_uri}
        else:
            payload = {"uris": [track_uri]}
    try:
        _api("PUT", "/me/player/play", payload=payload if payload else {}, params=params or None)
    except RuntimeError as exc:
        msg = str(exc)
        if "NO_ACTIVE_DEVICE" in msg.upper() or "active device" in msg.lower():
            return {
                "ok": False,
                "error": "No active Spotify device. Open Spotify on your PC or phone, play anything once, then retry.",
            }
        return {"ok": False, "error": msg}
    np = now_playing()
    return {
        "ok": True,
        "uri": track_uri or None,
        "summary": np.get("summary") or "Playback started",
        "track": np.get("track"),
    }


def pause() -> dict[str, Any]:
    try:
        _api("PUT", "/me/player/pause", payload={})
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "summary": "Paused"}


def next_track() -> dict[str, Any]:
    try:
        _api("POST", "/me/player/next", payload={})
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    time.sleep(0.35)
    np = now_playing()
    return {"ok": True, "summary": np.get("summary") or "Skipped to next", "track": np.get("track")}


def previous_track() -> dict[str, Any]:
    try:
        _api("POST", "/me/player/previous", payload={})
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    time.sleep(0.35)
    np = now_playing()
    return {"ok": True, "summary": np.get("summary") or "Went to previous", "track": np.get("track")}


def set_volume(percent: int) -> dict[str, Any]:
    vol = max(0, min(100, int(percent)))
    try:
        _api("PUT", "/me/player/volume", params={"volume_percent": str(vol)})
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "volume_percent": vol, "summary": f"Volume set to {vol}%"}


def queue_track(*, uri: str = "", query: str = "") -> dict[str, Any]:
    track_uri = (uri or "").strip()
    if not track_uri and (query or "").strip():
        found = search_catalog(query, limit=1)
        if not found.get("ok"):
            return found
        tracks = found.get("tracks") or []
        if not tracks:
            return {"ok": False, "error": f"No tracks found for {query!r}"}
        track_uri = str(tracks[0].get("uri") or "")
        label = f"{tracks[0].get('name')} — {tracks[0].get('artists')}"
    else:
        label = track_uri
    if not track_uri:
        return {"ok": False, "error": "uri or query is required"}
    try:
        _api("POST", "/me/player/queue", params={"uri": track_uri})
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "uri": track_uri, "summary": f"Queued {label}"}


def transfer_playback(device_id: str, *, play: bool = True) -> dict[str, Any]:
    did = (device_id or "").strip()
    if not did:
        return {"ok": False, "error": "device_id is required"}
    try:
        _api("PUT", "/me/player", payload={"device_ids": [did], "play": bool(play)})
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "device_id": did, "summary": f"Transferred playback to {did}"}


def spotify(
    action: str = "status",
    *,
    query: str = "",
    uri: str = "",
    device_id: str = "",
    volume: int | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Unified Spotify tool for OAC."""
    act = (action or "status").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "now": "now_playing",
        "current": "now_playing",
        "playing": "now_playing",
        "what": "now_playing",
        "auth": "connect",
        "login": "connect",
        "link": "connect",
        "skip": "next",
        "fwd": "next",
        "back": "previous",
        "prev": "previous",
        "stop": "pause",
        "vol": "volume",
        "device": "devices",
        "find": "search",
    }
    act = aliases.get(act, act)

    if act in {"status", "state"}:
        return connection_status()
    if act == "connect":
        started = begin_auth()
        if not started.get("ok"):
            return started
        # Open browser to authorize
        try:
            from ainet.tools.browser import open_chrome

            open_chrome(str(started["url"]), new_tab=True)
            opened = True
        except Exception:
            opened = False
        return {
            "ok": True,
            "opened_browser": opened,
            "auth_url": started["url"],
            "summary": (
                "Opened Spotify login in Chrome — approve access, then ask again."
                if opened
                else f"Open this URL to link Spotify: {started['url']}"
            ),
        }
    if act == "now_playing":
        return now_playing()
    if act == "devices":
        return list_devices()
    if act == "search":
        return search_catalog(query, limit=limit)
    if act == "play":
        return play(uri=uri, query=query, device_id=device_id)
    if act == "pause":
        return pause()
    if act == "next":
        return next_track()
    if act == "previous":
        return previous_track()
    if act == "volume":
        if volume is None:
            return {"ok": False, "error": "volume (0-100) is required"}
        return set_volume(int(volume))
    if act == "queue":
        return queue_track(uri=uri, query=query)
    if act == "transfer":
        return transfer_playback(device_id, play=True)
    return {
        "ok": False,
        "error": (
            f"Unknown action '{action}'. Use: status, connect, now_playing, search, "
            "play, pause, next, previous, volume, queue, devices, transfer"
        ),
    }


def auth_success_html() -> bytes:
    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><title>Spotify linked</title>
<style>
  body{font-family:Figtree,system-ui,sans-serif;background:#f4f6f2;color:#121412;
  display:grid;place-items:center;min-height:100vh;margin:0}
  .card{background:#fff;padding:2rem 2.2rem;border-radius:18px;box-shadow:0 12px 40px rgba(0,0,0,.08);
  max-width:28rem}
  h1{font-size:1.35rem;margin:0 0 .5rem;letter-spacing:-.02em}
  p{margin:0;color:#5a635c;line-height:1.45}
  a{color:#0c8f55}
</style></head>
<body><div class="card">
  <h1>Spotify linked</h1>
  <p>You can close this tab and go back to <a href="/">AINet chat</a>. Ask to play something.</p>
</div></body></html>"""
    return html.encode("utf-8")


def auth_error_html(message: str) -> bytes:
    safe = (
        (message or "Unknown error")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><title>Spotify link failed</title>
<style>
  body{{font-family:Figtree,system-ui,sans-serif;background:#f4f6f2;color:#121412;
  display:grid;place-items:center;min-height:100vh;margin:0}}
  .card{{background:#fff;padding:2rem 2.2rem;border-radius:18px;box-shadow:0 12px 40px rgba(0,0,0,.08);
  max-width:28rem}}
  h1{{font-size:1.35rem;margin:0 0 .5rem}}
  p{{margin:0;color:#5a635c;line-height:1.45}}
  a{{color:#0c8f55}}
</style></head>
<body><div class="card">
  <h1>Spotify link failed</h1>
  <p>{safe}</p>
  <p style="margin-top:1rem"><a href="/auth/spotify">Try again</a> · <a href="/">Chat</a></p>
</div></body></html>"""
    return html.encode("utf-8")

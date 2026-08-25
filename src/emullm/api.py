"""Simulated LLM backend: relay chat-completion requests to a human or
agent connected over a WebSocket, instead of a real model API.

Exposes an OpenAI-compatible surface so it can be registered as an
ordinary local, **keyless** backend (baseUrl "http://127.0.0.1:8801/v1",
adapter "openai_chat_completions"). Clients never need an API key or
bearer token: nothing on /v1/* inspects Authorization. If an SDK requires
api_key, any dummy value is fine.

Implemented routes, and how each is emulated:
  - GET  /v1/models                 -- list personas (see list_models())
  - GET  /v1/models/{model_id}       -- fetch one persona's metadata
  - GET  /emullm/caps/{worker_id}  -- lightweight per-worker lookup:
                                         connected?, its models, and its
                                         declared "pretend" capabilities
  - GET  /emullm/docs/{rel_path}   -- serves this feature's own design
                                         docs (docs/**) straight off disk,
                                         e.g.
                                         /emullm/docs/EMULLM_RELAY.md
  - GET  /emullm/tokens/new        -- optional bookkeeping only: mint a
                                         token/public key. NOT required for
                                         any client or worker route; never
                                         enforced as auth on /v1/*
  - POST /emullm/tokens            -- the JSON API behind that page
                                         (same: issuance only, no gating)
  - /emullm/storage/{path}         -- GET/PUT/DELETE generic durable
                                         blobs (a worker "borrowing" this
                                         server's disk for its own scratch
                                         space), plus GET /emullm/storage
                                         to list everything stored
  - /emullm/specific_worker/{worker_id}/v1/*
                                     -- the SAME /v1/* surface (models,
                                         chat/completions, completions,
                                         responses, embeddings,
                                         moderations, images, audio),
                                         but with worker_id pinned from
                                         the URL instead of parsed out of
                                         "model" -- for a client that can
                                         only configure a fixed baseUrl
  - POST /v1/chat/completions        -- relayed to the connected worker (real)
                                         with normal JSON or SSE output
  - POST /v1/completions             -- legacy text-completion; wraps the
                                         prompt as a single user message and
                                         relays it the same way
  - POST /v1/responses               -- newer "Responses API" shape; also
                                         relayed the same way, response
                                         reshaped to the Responses schema
  - POST /v1/embeddings              -- NOT relayed (there's no sensible way
                                         for a text reply to become a real
                                         embedding vector). Returns a
                                         deterministic pseudo-random vector
                                         hashed from the input text, so
                                         repeated calls with the same text
                                         are stable. Not semantically
                                         meaningful -- for wiring/testing
                                         only.
  - POST /v1/moderations             -- NOT relayed. Always reports the
                                         input as not flagged (stub).
  - POST /v1/images/generations      -- NOT relayed. Returns a tiny stub
                                         placeholder image (data: URL).
  - POST /v1/audio/transcriptions    -- NOT relayed (no audio understanding
                                         is available here). Validates a real
                                         multipart upload and returns a fixed
                                         stub transcript string.
  - POST /v1/audio/speech            -- NOT relayed. Returns a valid synthetic
                                         silent WAV payload.
  - /v1/files                       -- local filesystem-backed emulation of
                                         multipart upload, list, metadata,
                                         content download, and deletion
  - /v1/assistants, /v1/threads,
    /v1/fine_tuning/jobs            -- heavier platform CRUD surfaces.
                                         Atomic JSON persistence, retrieval,
                                         modification/deletion and stable
                                         cursor pagination. Fine-tune input
                                         JSONL is validated, but training is
                                         unavailable and ends in a structured
                                         failed state.
  - /admin/emullm/* (alias: /emullm/admin/*)
                                     -- NOT part of the OpenAI-compatible
                                         surface. A small test-controller
                                         API so tests (or an operator) can
                                         drive this server over plain
                                         HTTP: GET state (worker
                                         connected?, pending requests,
                                         record counts), POST runtime_dir
                                         to repoint the durable stores at a
                                         different directory (e.g. a
                                         test's tmp_path), POST reset to
                                         wipe all persisted records,
                                         and DELETE records/{kind}/{id} to
                                         remove one. Both URL forms hit
                                         the exact same handlers.

Any request that is genuinely relayed (chat/completions, completions,
responses) is queued and forwarded to whichever worker is currently
connected at WebSocket /emullm/{worker_id}/ws for that model's
worker_id prefix (e.g. model "alice/same" routes to whoever is connected
at /emullm/alice/ws). The worker reads the forwarded prompt, composes a
reply, and sends it back tagged with the same request id; that reply
becomes the HTTP response. If no
worker happens to be connected right now (e.g. a request lands during
one of the worker's idle "rest" windows), the call does NOT fail fast --
it just waits for a worker to (re)connect, like a slow API server. Only
if no worker ever connects/replies within the overall timeout does the
HTTP call fail, with 504, instead of hanging forever.

The intended worker-side pattern (see scripts/emullm_worker.py) is: an
agent connects, waits up to ~10s for one request; if one arrives, it
answers it (however long that takes) and immediately reconnects to wait
for the next one; if nothing arrives within ~10s, it disconnects, goes
back to its other duties, and reconnects again after a randomized rest
of up to ~30s -- so it isn't permanently tied up polling an idle socket.

IMPORTANT: that ~30s rest is a randomized MAX, not a fixed cadence, and
each connect/idle/rest cycle runs independently of any external clock.
Real request traffic shifts the timing of subsequent cycles (answering a
request delays the start of the next idle window by however long the
answer took), so the worker's connect/disconnect pattern naturally drifts
in and out of phase over time. This is expected -- do not "fix" it into a
synchronized fixed-interval heartbeat.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import mimetypes
import os
import secrets
import struct
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.datastructures import UploadFile

from . import supervisor as _sup

router = APIRouter()

_REQUEST_TIMEOUT_SECONDS = 900  # generous -- a human/agent may take a while to reply

# Human-facing server "mode", surfaced on the status page and in the admin
# state JSON, and used by _relay to decide how to answer when a worker isn't
# available. Override with EMULLM_MODE (default "relay").
#
# Behaviors that affect the relay path:
#   mock              -- synthesize a fake reply, never wait for a worker
#   error-when-empty  -- if no worker is connected, fail fast with 503
#   wait / wait-then-serve / recruit / self / auto / relay (default)
#                     -- wait up to _REQUEST_TIMEOUT_SECONDS for a worker
_SERVER_MODE = os.environ.get("EMULLM_MODE", "relay")
_SERVER_STARTED_AT = time.time()

# Sentinel: a relay step couldn't handle the request -> try the next mode.
_PASS: Any = object()


def _current_modes() -> list[str]:
    """The ordered fallback chain of run modes. ``_SERVER_MODE`` may be a
    single mode, a comma-separated string ("recruit,proxy,mock"), or a list;
    each is tried in order until one produces a reply."""
    raw = _SERVER_MODE
    if isinstance(raw, (list, tuple)):
        modes = [str(m).strip() for m in raw if str(m).strip()]
    else:
        modes = [part.strip() for part in str(raw).split(",") if part.strip()]
    return modes or ["relay"]


def _mock_reply(model: str, prompt_text: str, instruction: str | None = None) -> str:
    """The deterministic *success content* used when the system pretends a
    worker answered (see `mock` mode). Controllable so a test can assert an
    exact response:

      - ``EMULLM_MOCK_REPLY`` (env) or config ``{"mock": {"reply": "..."}}``
        -> return exactly that string.
      - ``EMULLM_MOCK_TEMPLATE`` (env) or config ``{"mock": {"template": ...}}``
        -> ``str.format`` with {prompt}, {model}, {persona}.
      - otherwise -> ``"mock: <prompt>"`` (deterministic echo).
    """
    mock_cfg = _read_config().get("mock")
    mock_cfg = mock_cfg if isinstance(mock_cfg, dict) else {}

    fixed = os.environ.get("EMULLM_MOCK_REPLY")
    if fixed is None and mock_cfg.get("reply") is not None:
        fixed = str(mock_cfg["reply"])
    if fixed is not None:
        return fixed

    template = os.environ.get("EMULLM_MOCK_TEMPLATE") or mock_cfg.get("template") or "mock: {prompt}"
    try:
        return str(template).format(prompt=prompt_text, model=model, persona=instruction or "")
    except (KeyError, IndexError, ValueError):
        return f"mock: {prompt_text}"


class _MockWorker:
    """A pretend websocket peer for `mock` mode.

    It is NOT a simulated agent/persona -- it just makes the relay behave as
    if a worker were present at the websocket and answered successfully: its
    ``send_json`` immediately fulfills the pending request future. Used both
    for ephemeral per-request success (the `mock` step) and for a set of
    pre-registered pretend peers (``register_mock_workers``).
    """

    def __init__(self, worker_id: str, reply: str | None = None, template: str | None = None) -> None:
        self.worker_id = worker_id
        self.reply = reply
        self.template = template

    async def send_json(self, payload: dict[str, Any]) -> None:
        future = _pending.get(payload.get("id"))
        if future is None or future.done():
            return
        future.set_result(self._answer(payload))

    def _answer(self, payload: dict[str, Any]) -> str:
        if self.reply is not None:
            return str(self.reply)
        template = self.template or "mock[{worker}]: {prompt}"
        try:
            return str(template).format(
                worker=self.worker_id, prompt=payload.get("prompt", ""), model=payload.get("model", "")
            )
        except (KeyError, IndexError, ValueError):
            return f"mock[{self.worker_id}]: {payload.get('prompt', '')}"


def register_mock_workers(specs: list[dict[str, Any]]) -> list[str]:
    """Register a set of pretend copilots as connected in-process workers.

    Each spec: ``{ "id": "alice", "reply": "...", "template": "...",
    "capabilities": ["images"], "role": "trusted", "models": {...} }``.
    Returns the ids registered. Used by the app lifespan (from config
    ``mock_workers``) and directly by tests.
    """
    registered: list[str] = []
    for spec in specs or []:
        if not isinstance(spec, dict):
            continue
        worker_id = str(spec.get("id") or spec.get("worker_id") or "").strip()
        if not worker_id:
            continue
        _connected_workers[worker_id] = _MockWorker(worker_id, spec.get("reply"), spec.get("template"))
        caps = spec.get("capabilities")
        if isinstance(caps, list):
            _worker_capabilities[worker_id] = {str(c): True for c in caps}
        elif isinstance(caps, dict):
            _worker_capabilities[worker_id] = {str(k): bool(v) for k, v in caps.items()}
        role = spec.get("role")
        _worker_roles[worker_id] = str(role) if role else "mock"
        models = spec.get("models")
        if isinstance(models, dict) and models:
            _worker_models[worker_id] = models
        registered.append(worker_id)
    return registered


def unregister_mock_workers(worker_ids: list[str]) -> None:
    """Remove previously registered mock copilots (used on shutdown)."""
    for worker_id in worker_ids:
        worker = _connected_workers.get(worker_id)
        if isinstance(worker, _MockWorker):
            _connected_workers.pop(worker_id, None)
        _worker_capabilities.pop(worker_id, None)
        _worker_roles.pop(worker_id, None)
        _worker_models.pop(worker_id, None)


# --- Proxy backends (proxy / proxy-observe modes) --------------------------
# In `proxy` mode a request is forwarded to a real OpenAI-compatible backend
# instead of a worker; in `proxy-observe` the real answer is returned to the
# client AND mirrored to any connected worker so an agent can learn to
# emulate it. Backends come from config.json ("backends": [...]) or env.
def _all_backends() -> list[dict[str, Any]]:
    """Every configured proxy backend (config ``backends`` / agent
    ``launch: proxy``), else the env ``EMULLM_PROXY_BASE_URL`` fallback."""
    config = _sup.expand_agents(_read_config())
    backends = [
        entry
        for entry in (config.get("backends") or [])
        if isinstance(entry, dict) and entry.get("base_url")
    ]
    if backends:
        return backends
    base_url = os.environ.get("EMULLM_PROXY_BASE_URL")
    if base_url:
        return [
            {
                "name": "env",
                "base_url": base_url,
                "api_key_env": os.environ.get("EMULLM_PROXY_API_KEY_ENV"),
                "model": os.environ.get("EMULLM_PROXY_MODEL"),
            }
        ]
    return []


def _select_backend() -> dict[str, Any] | None:
    """The backend to proxy to: the one marked ``default`` (else the first
    configured), else the env fallback."""
    backends = _all_backends()
    for entry in backends:
        if entry.get("default"):
            return entry
    return backends[0] if backends else None


def _backend_api_key(backend: dict[str, Any]) -> str | None:
    if backend.get("api_key"):
        return str(backend["api_key"])
    env_name = backend.get("api_key_env")
    if env_name:
        return os.environ.get(str(env_name))
    return None


def _http_post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
    """Blocking JSON POST (run in an executor). Split out so tests can stub it."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310 -- URL comes from operator config
        url, data=data, method="POST", headers={"Content-Type": "application/json", **headers}
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def _http_get_json(url: str, headers: dict[str, str], timeout: float = 15.0) -> dict[str, Any]:
    """Blocking JSON GET (run in an executor). Split out so tests can stub it."""
    request = urllib.request.Request(url, method="GET", headers=headers)  # noqa: S310
    with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def _http_post_raw(
    url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float = 60.0
) -> tuple[int, str, int]:
    """Blocking POST returning ``(status, content_type, byte_count)`` without
    parsing -- for binary endpoints like audio speech. Split out so tests can
    stub it."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310 -- URL comes from operator config
        url, data=data, method="POST", headers={"Content-Type": "application/json", **headers}
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310
        body = resp.read()
        return resp.status, resp.headers.get("Content-Type", ""), len(body)


async def _proxy_chat(
    backend: dict[str, Any], model: str, prompt_text: str, instruction: str | None = None
) -> str:
    base = str(backend["base_url"]).rstrip("/")
    url = f"{base}/chat/completions"
    headers: dict[str, str] = {}
    key = _backend_api_key(backend)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    messages: list[dict[str, str]] = []
    if instruction:
        # persona capability dial (percentNN): shape apparent capability via a
        # system message, so the same real model can look more/less capable.
        messages.append({"role": "system", "content": instruction})
    messages.append({"role": "user", "content": prompt_text})
    payload = {"model": _backend_model_for(model, backend), "messages": messages}
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, lambda: _http_post_json(url, headers, payload))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"proxy backend error: {exc}") from exc
    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="proxy backend returned an unexpected response") from exc
    return content or ""


async def _observe(worker_id: str, model: str, prompt_text: str, reply_text: str) -> None:
    """Best-effort: mirror a proxied exchange to a connected worker so it can
    learn to emulate the backend. Never fails the client's request."""
    worker = _connected_workers.get(worker_id)
    if worker is None:
        return
    try:
        await worker.send_json(
            {"type": "observe", "model": model, "prompt": prompt_text, "reply": reply_text}
        )
    except Exception:  # noqa: BLE001
        pass


async def _probe_model(
    base: str, headers: dict[str, str], model_id: str, *, retries: int = 2, backoff: float = 2.0
) -> dict[str, Any]:
    """Give one claimed model the "IQ test" (text identity, then vision) off
    the event loop. Thin async wrapper over the shared, blocking
    :func:`_probe_modalities_sync` -- the same battery we use to validate
    recruits -- so the probe endpoint reports chat/embeddings/vision/identity
    and a ``live`` / ``reachable`` / ``embeddings-only`` / ``not_loaded`` /
    ``rate_limited`` / ``error`` status."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: _probe_modalities_sync(base, headers, model_id, retries=retries, backoff=backoff)
    )


async def probe_backend(
    backend: dict[str, Any], *, verify: bool = False, limit: int | None = None
) -> dict[str, Any]:
    """Ask a proxy backend what it can do by calling its ``/v1/models``.

    The advertised model list is only a claim -- an endpoint may list a model
    it never actually loaded. With ``verify=True`` each model (up to ``limit``)
    is actually exercised, splitting the list into ``live`` and
    ``falsely_advertised``. Best-effort: an unreachable/broken backend yields
    ``ok: false`` and never raises."""
    base = str(backend.get("base_url") or "").rstrip("/")
    result: dict[str, Any] = {"name": backend.get("name") or "backend", "base_url": base, "ok": False}
    if not base:
        result["error"] = "no base_url"
        return result
    headers: dict[str, str] = {}
    key = _backend_api_key(backend)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    url = f"{base}/models"
    loop = asyncio.get_running_loop()
    try:
        data = await loop.run_in_executor(None, lambda: _http_get_json(url, headers))
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
        return result
    models = data.get("data") if isinstance(data, dict) else None
    ids = (
        [str(m["id"]) for m in models if isinstance(m, dict) and m.get("id")]
        if isinstance(models, list)
        else []
    )
    result["ok"] = True
    result["models"] = ids
    if verify:
        subset = ids[:limit] if limit else ids
        verified = []
        for model_id in subset:
            verified.append(await _probe_model(base, headers, model_id))
            await asyncio.sleep(0.5)  # gentle spacing so we don't self-trip rate limits
        result["verified"] = verified
        result["live"] = [v["id"] for v in verified if v.get("status") == "live"]
        result["serving"] = [
            v["id"] for v in verified
            if str(v.get("status") or "").endswith("-only") or v.get("status") == "serving"
        ]
        result["reachable"] = [v["id"] for v in verified if v.get("status") == "reachable"]
        result["falsely_advertised"] = [v["id"] for v in verified if v.get("status") == "not_loaded"]
        result["inconclusive"] = [v["id"] for v in verified if v.get("status") in ("rate_limited", "error")]
    return result


async def probe_backends(*, verify: bool = False, limit: int | None = None) -> list[dict[str, Any]]:
    """Probe every configured proxy backend (see :func:`_all_backends`)."""
    return [await probe_backend(entry, verify=verify, limit=limit) for entry in _all_backends()]


# ---------------------------------------------------------------------------
# Usage tracking / rate limiting, so a worker (e.g. a human/agent
# emulator) doesn't get overused. Requests are counted per worker_id in a
# rolling window; once a worker hits the limit within that window, any
# FURTHER relayed request for that worker_id is rejected with 429 (not
# queued/waited-on -- overload protection should fail fast, unlike the
# "slow API" wait-for-a-worker-to-connect behavior in _relay).
# ---------------------------------------------------------------------------
_USAGE_WINDOW_SECONDS = float(os.environ.get("EMULLM_RATE_LIMIT_WINDOW_SECONDS", "60"))
_USAGE_MAX_PER_WINDOW = int(os.environ.get("EMULLM_RATE_LIMIT_PER_WINDOW", "20"))
_worker_usage: dict[str, dict[str, Any]] = {}  # worker_id -> {"total": int, "recent": [timestamps], "last_used_at": float}


def _check_and_record_usage(worker_id: str) -> None:
    now = time.monotonic()
    usage = _worker_usage.setdefault(worker_id, {"total": 0, "recent": []})
    recent: list[float] = usage["recent"]
    cutoff = now - _USAGE_WINDOW_SECONDS
    while recent and recent[0] < cutoff:
        recent.pop(0)
    if len(recent) >= _USAGE_MAX_PER_WINDOW:
        retry_after_seconds = max(1.0, recent[0] + _USAGE_WINDOW_SECONDS - now)
        retry_after_display = (
            f"{retry_after_seconds / 60:.1f} minutes" if retry_after_seconds >= 90 else f"{retry_after_seconds:.0f}s"
        )
        raise HTTPException(
            status_code=429,
            detail=(
                f"worker '{worker_id}' rate-limited: already handled {_USAGE_MAX_PER_WINDOW} "
                f"requests in the last {_USAGE_WINDOW_SECONDS:.0f}s -- come back in about "
                f"{retry_after_display}, so it doesn't get overused"
            ),
            headers={"Retry-After": str(int(retry_after_seconds) + 1)},
        )
    recent.append(now)
    usage["total"] += 1
    usage["last_used_at"] = time.time()


# Multiple workers ("a small pool of emulators") can be connected at once,
# each under its own worker_id (e.g. "yourself", "alice", "bob"). A model
# id is "<worker_id>/<persona-suffix>" (see _PERSONA_SUFFIXES below), and a
# request for that model is routed to whichever worker is currently
# registered under that worker_id -- not to "whoever happens to be
# connected" like a single-worker design would.
_connected_workers: dict[str, WebSocket] = {}
_worker_lock = asyncio.Lock()
_pending: dict[str, "asyncio.Future[str]"] = {}

# Each worker declares its OWN model list on connect (see the websocket
# handshake below): a dict of suffix -> {"display_name", "instruction"}.
# /v1/models aggregates these across every currently connected worker. A
# worker_id with no declared list yet falls back to _PERSONA_SUFFIXES
# below, so a bare-bones/older worker still gets a sensible default menu
# without having to declare anything itself.
_worker_models: dict[str, dict[str, dict[str, Any]]] = {}

# A worker can ALSO opt in, at register time, to "pretending" for the
# non-text stub surfaces below (embeddings/moderations/images/audio) --
# i.e. actually answering (in character, via the normal text relay) as if
# it could see/hear/produce that modality, instead of the server's fixed
# static stub. Declared as {"embeddings": true, "images": true, ...}; any
# capability not declared true just uses the ordinary static stub.
_worker_capabilities: dict[str, dict[str, bool]] = {}

# A worker's self-declared role/phase, shown on the status pages. Free-form,
# but common values are "trusted" (serving clients normally) and "training"
# (a subagent still learning to use the websocket). Purely informational for
# now -- it does NOT gate routing yet. Declared in the register message as
# {"role": "..."}; defaults to "trusted".
_DEFAULT_WORKER_ROLE = "trusted"
_worker_roles: dict[str, str] = {}

_DEFAULT_WORKER_ID = "yourself"

# ---------------------------------------------------------------------------
# Default/fallback persona suffixes, used for any worker_id that hasn't
# declared its own model list: "<id>/same" answers normally; the
# "<id>/percentNN" variants ask that worker to deliberately answer as if
# only NN% as capable (dumber, terser, more error-prone -- possibly
# emulating a weaker model's style), surfaced to the worker via the
# persona's `instruction`.
# ---------------------------------------------------------------------------
_PERSONA_SUFFIXES: dict[str, dict[str, Any]] = {
    "same": {
        "display_name": "(unmodified)",
        "instruction": "Answer normally, at your full/actual capability.",
    },
    "percent125": {
        "display_name": "(~125% -- extra thorough)",
        "instruction": "Answer as if boosted beyond your normal capability: be extra thorough, careful, and complete.",
    },
    "percent100": {
        "display_name": "(100% -- normal)",
        "instruction": "Answer normally, at your full/actual capability.",
    },
    "percent75": {
        "display_name": "(~75% capable)",
        "instruction": "Answer as if only about 75% as capable as usual: slightly less careful/thorough, occasional minor omissions.",
    },
    "percent25": {
        "display_name": "(~25% capable)",
        "instruction": "Answer as if only about 25% as capable as usual: noticeably weaker, terser, more likely to miss nuance -- emulate a much smaller/weaker model's style.",
    },
    "percent10": {
        "display_name": "(~10% capable)",
        "instruction": "Answer as if only about 10% as capable as usual: very weak, minimal, simplistic -- emulate a small/weak model's style, possibly with mistakes.",
    },
}
_DEFAULT_MODEL_ID = f"{_DEFAULT_WORKER_ID}/same"


def _split_model_id(model: str) -> tuple[str, str]:
    """"<worker_id>/<suffix>" -> (worker_id, suffix); a bare id with no
    "/" is treated as that worker_id with the "same" persona."""
    worker_id, sep, suffix = model.partition("/")
    if not worker_id:
        worker_id = _DEFAULT_WORKER_ID
    if not sep:
        suffix = "same"
    return worker_id, suffix


def _models_for(worker_id: str) -> dict[str, dict[str, Any]]:
    """The persona/model menu for worker_id: whatever it declared on
    connect (see the websocket handshake), or _PERSONA_SUFFIXES as a
    fallback for a worker_id that hasn't declared one (yet)."""
    return _worker_models.get(worker_id, _PERSONA_SUFFIXES)


def _worker_can_pretend(worker_id: str, capability: str) -> bool:
    return bool(_worker_capabilities.get(worker_id, {}).get(capability))


def _worker_capability_state(worker_id: str, capability: str) -> bool | None:
    """True/False if this worker_id EXPLICITLY declared the capability
    (opted in or out) at register time; None if it never said either way
    (unknown -> caller should fall back to the generic static stub)."""
    declared = _worker_capabilities.get(worker_id)
    if declared is None:
        return None
    return declared.get(capability)


def _raise_if_capability_declined(worker_id: str, capability: str) -> None:
    """If this worker_id explicitly declared it will NOT emulate
    `capability`, stop the request right here with a clear 501 -- don't
    silently fall back to the generic stub (that would blur "no worker
    opinion" with "this worker said no"), and don't bother relaying
    anything to the worker for a capability it already declined."""
    if _worker_capability_state(worker_id, capability) is False:
        raise HTTPException(
            status_code=501,
            detail=f"worker '{worker_id}' has declared it will not emulate '{capability}' -- not asking it",
        )


# --- Non-text capability fallback policy ----------------------------------
# When a non-text /v1 endpoint (embeddings/moderations/images/audio) is asked
# for a capability that no worker has opted into emulating, this decides what
# happens. Mirrors _SERVER_MODE: override with EMULLM_CAPABILITY_FALLBACK or
# config.json "capability_fallback".
#   stub  (default) -- answer immediately with the deterministic local fake
#   wait            -- hold the request ("slow, not broken") until a worker
#                      connects and opts in; 504 on timeout
#   error           -- fail fast with 503
# An EXPLICIT worker opt-out (capabilities.X = false) is always a hard 501,
# regardless of this policy: a deliberate "no" is not "nobody available".
_CAPABILITY_FALLBACK = os.environ.get("EMULLM_CAPABILITY_FALLBACK", "stub")

# Map an internal capability name to its user-facing service name (the keys
# used in a config `services` map).
_CAPABILITY_SERVICE = {
    "embeddings": "embeddings",
    "moderations": "moderations",
    "images": "images",
    "audio_transcription": "audio",
    "audio_speech": "audio",
    "fine_tuning": "fine_tuning",
}

# Policy derived from config `agents` / server-level `services` (populated by
# apply_agent_policies at startup; see app lifespan).
_worker_service_behavior: dict[str, dict[str, str]] = {}  # worker_id -> service -> behavior
_service_fallback: dict[str, list[str]] = {}             # service -> fallback chain (server-level)
_observers: dict[str, Any] = {}                           # worker_id -> True | set[str] (services)
_agent_descriptions: dict[str, str] = {}                  # worker_id -> user-facing description
_service_descriptions: dict[str, str] = {}                # service -> user-facing description
_model_routes: dict[str, str] = {}                        # full model id -> worker_id that serves it
_server_description: str | None = None                    # user-facing server description
_advertised_base: list[str] = []                          # services.models (manual base)
_advertised_default: str | None = None                    # services.model (default advertised)
_advertised_agents: list[dict[str, Any]] = []             # agents flagged to contribute their models
_model_fetch_cache: dict[str, dict[str, Any]] = {}        # base_url -> {fetched_at, models}
_MODEL_FETCH_TTL_SECONDS = 86400.0                        # fallback TTL when nothing set
_DEFAULT_VALIDATION_INTERVAL = "1week"                    # server-level default cadence
_VALIDATION_TIMEOUT_SECONDS = 120.0                       # per-model probe budget ("never returned")
_validation_interval: Any = _DEFAULT_VALIDATION_INTERVAL  # config default_validation_interval (inherited)
_validation_interval_override: Any = None                 # config validation_interval_override (forces all)


def _capability_fallback() -> str:
    value = str(_CAPABILITY_FALLBACK or "stub").strip().lower()
    return value if value in ("stub", "wait", "error") else "stub"


# Server-level fallback tokens that mean "distribute to a volunteering agent"
# (agents just volunteer/reject; the *system* picks how to spread across them).
_STRATEGY_TOKENS = {"round-robin", "random", "failover", "aggregate", "serve", "distribute"}


def _parse_chain(value: Any) -> list[str]:
    """A server-level ``fallback`` may be a single token, a comma string, or a
    list -- a chain tried left to right (like the run-mode chain). Returns the
    lowercased tokens."""
    if isinstance(value, (list, tuple)):
        return [str(v).strip().lower() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [part.strip().lower() for part in value.split(",") if part.strip()]
    return []


def _service_entry(value: Any) -> tuple[str | None, str | None]:
    """Normalize a `services` map value (a bare behavior string or a
    ``{behavior/fallback, description}`` object) to ``(behavior, description)``."""
    if isinstance(value, str):
        return (value.strip().lower() or None), None
    if isinstance(value, dict):
        raw = value.get("behavior") or value.get("fallback")
        behavior = str(raw).strip().lower() if raw else None
        description = value.get("description")
        return behavior, (str(description) if description else None)
    return None, None


def _model_id(entry: Any) -> str | None:
    """A model-list entry may be a bare id string or a node object
    (``{"id": ..., "validation_startedAt": ..., ...}``); return its id
    either way, so results can "appear anywhere" a plain id can."""
    if isinstance(entry, str):
        return entry or None
    if isinstance(entry, dict):
        mid = entry.get("id") or entry.get("model")
        return str(mid) if mid else None
    return None


def _agent_model_list(agent: dict[str, Any]) -> list[Any]:
    """The agent's model list/cache. Preferred home is the aggregate service
    node's ``catalog`` (``services.models.catalog`` -- avoids the awkward
    ``models.models``); falls back to ``services.models.models`` then the
    agent-level ``models``. Entries may be ids or result nodes."""
    services = agent.get("services") if isinstance(agent.get("services"), dict) else {}
    models_cfg = services.get("models") if isinstance(services.get("models"), dict) else {}
    for key in ("catalog", "models"):
        if isinstance(models_cfg.get(key), list):
            return models_cfg[key]
    if isinstance(agent.get("models"), list):
        return agent["models"]
    return []


def clear_agent_policies() -> None:
    """Reset all config-derived per-agent/service policy (used on shutdown)."""
    global _server_description, _advertised_default, _validation_interval, _validation_interval_override
    _worker_service_behavior.clear()
    _service_fallback.clear()
    _observers.clear()
    _agent_descriptions.clear()
    _service_descriptions.clear()
    _model_routes.clear()
    _server_description = None
    _advertised_base.clear()
    _advertised_default = None
    _advertised_agents.clear()
    _validation_interval = _DEFAULT_VALIDATION_INTERVAL
    _validation_interval_override = None
    # note: _model_fetch_cache intentionally NOT cleared -- it's a daily cache
    # that should survive config reloads.


def apply_agent_policies(config: dict[str, Any]) -> None:
    """Populate per-agent service behaviors, observers, and user-facing
    descriptions from a config's ``agents`` list and server-level
    ``services`` / ``description``. Clears any prior policy first."""
    global _server_description, _advertised_default, _validation_interval, _validation_interval_override
    clear_agent_policies()
    if not isinstance(config, dict):
        return
    for _k in ("validation_interval_default", "validation_interval"):
        if config.get(_k) is not None:
            _validation_interval = config[_k]
            break
    _validation_interval_override = config.get("validation_interval_override")
    if isinstance(config.get("description"), str):
        _server_description = config["description"]
    services = config.get("services")
    if isinstance(services, dict):
        if isinstance(services.get("models"), list):
            _advertised_base[:] = [mid for m in services["models"] if (mid := _model_id(m))]
        if isinstance(services.get("model"), str):
            _advertised_default = services["model"]
        for service, value in services.items():
            if service in ("model", "default", "models"):
                continue  # server-level catalog fields, not per-service behavior
            behavior, description = _service_entry(value)
            raw_fallback = value.get("fallback") if isinstance(value, dict) else value
            chain = _parse_chain(raw_fallback)
            if chain:
                _service_fallback[str(service)] = chain
            if description:
                _service_descriptions[str(service)] = description
    agents = config.get("agents")
    if isinstance(agents, list):
        for agent in agents:
            if not isinstance(agent, dict):
                continue
            worker_id = str(agent.get("id") or agent.get("worker_id") or "").strip()
            if not worker_id:
                continue
            if isinstance(agent.get("description"), str):
                _agent_descriptions[worker_id] = agent["description"]
            # An agent may declare the catalog model ids it serves (so callers
            # of those ids route to this worker) and/or replace another agent's
            # whole catalog.
            serves = agent.get("serves")
            if isinstance(serves, (list, tuple)):
                for mid in serves:
                    if isinstance(mid, str) and mid:
                        _model_routes[mid] = worker_id
            observe = agent.get("observe")
            if observe:
                if observe is True or observe == "all":
                    _observers[worker_id] = True
                elif isinstance(observe, (list, tuple)):
                    _observers[worker_id] = {str(x) for x in observe}
                else:
                    _observers[worker_id] = {str(observe)}
            agent_services = agent.get("services") if isinstance(agent.get("services"), dict) else {}
            models_cfg = agent_services.get("models") if isinstance(agent_services.get("models"), dict) else {}
            # An agent publishes its models into the user-facing catalog when
            # its reserved services.models catalog entry has behavior "aggregate".
            if models_cfg.get("behavior") == "aggregate":
                _advertised_agents.append(agent)
                if not _advertised_default and isinstance(agent.get("model"), str):
                    _advertised_default = agent["model"]
            behaviors = {}
            for service, value in agent_services.items():
                if service == "models":
                    continue  # reserved: catalog config, not a routable service
                behavior, _description = _service_entry(value)
                if behavior:
                    behaviors[str(service)] = behavior
            if behaviors:
                _worker_service_behavior[worker_id] = behaviors
        # Second pass: an agent can `replaces: "<other_id>"` to take over that
        # agent's whole advertised catalog (map each of its ids to this worker).
        by_id = {str(a.get("id") or a.get("worker_id") or ""): a for a in agents if isinstance(a, dict)}
        for agent in agents:
            if not isinstance(agent, dict):
                continue
            target = agent.get("replaces")
            worker_id = str(agent.get("id") or agent.get("worker_id") or "").strip()
            if isinstance(target, str) and target in by_id and worker_id:
                for entry in _agent_model_list(by_id[target]):
                    mid = _model_id(entry)
                    if mid:
                        _model_routes[mid] = worker_id
    # Top-level explicit routes win (operator override).
    routes = config.get("model_routes")
    if isinstance(routes, dict):
        for mid, wid in routes.items():
            if isinstance(mid, str) and isinstance(wid, str) and mid and wid:
                _model_routes[mid] = wid


_INTERVAL_UNITS = {
    "": 86400.0, "d": 86400.0, "day": 86400.0, "days": 86400.0,
    "h": 3600.0, "hr": 3600.0, "hour": 3600.0, "hours": 3600.0,
    "m": 60.0, "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
    "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
    "w": 604800.0, "week": 604800.0, "weeks": 604800.0,
}


def _parse_interval(value: Any) -> float | None:
    """Parse an ``update_interval`` to seconds. ``None``/``"null"``/``"never"``
    -> None (don't refresh); a number -> seconds; a duration string like
    ``"1day"``/``"12h"``/``"30m"`` -> its seconds; ``"always"``/0 -> 0."""
    if value is None:
        return None
    if isinstance(value, bool):
        return _MODEL_FETCH_TTL_SECONDS if value else None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else 0.0
    text = str(value).strip().lower()
    if not text or text in ("null", "none", "never"):
        return None
    if text in ("always", "0"):
        return 0.0
    number = ""
    index = 0
    while index < len(text) and (text[index].isdigit() or text[index] == "."):
        number += text[index]
        index += 1
    unit = text[index:].strip()
    try:
        magnitude = float(number) if number else 1.0
    except ValueError:
        return _MODEL_FETCH_TTL_SECONDS
    return magnitude * _INTERVAL_UNITS.get(unit, 86400.0)


_GREEN_SQUARE_DATA_URL: str | None = None


def _green_square_data_url(size: int = 10) -> str:
    """A tiny solid-green PNG (data URL) used to validate a model's vision:
    ask what color the square is and check it answers 'green'."""
    global _GREEN_SQUARE_DATA_URL
    if _GREEN_SQUARE_DATA_URL is None:
        import base64
        import zlib

        def _chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + tag
                + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # RGB, 8-bit
        raw = (b"\x00" + b"\x00\xff\x00" * size) * size  # each row: filter 0 + green pixels
        png = (
            b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(raw))
            + _chunk(b"IEND", b"")
        )
        _GREEN_SQUARE_DATA_URL = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    return _GREEN_SQUARE_DATA_URL


def _probe_modalities_sync(
    base: str, headers: dict[str, str], model_id: str, *, retries: int = 2, backoff: float = 2.0
) -> dict[str, Any]:
    """Blocking: validate a claimed model across model kinds. Text first ("what
    model are you" -- also captures identity); if text passes, vision (a 10x10
    green square -> expect "green"). If text fails (a non-chat model kind), try
    embeddings, image generation, and audio speech (TTS). Retries on 429.
    ``status`` is live (chats back) / reachable (answers but no usable chat) /
    <kind>-only (serves a non-chat kind) / not_loaded (definite 4xx) /
    rate_limited / error."""

    def _post(path: str, payload: dict[str, Any]) -> tuple[Any, Any]:
        for attempt in range(retries + 1):
            try:
                return _http_post_json(f"{base}{path}", headers, payload, 30.0), None
            except Exception as exc:  # noqa: BLE001
                code = getattr(exc, "code", None)
                if code == 429 and attempt < retries:
                    time.sleep(backoff * (attempt + 1))
                    continue
                return None, code
        return None, 429

    def _post_raw(path: str, payload: dict[str, Any]) -> tuple[Any, Any]:
        for attempt in range(retries + 1):
            try:
                status, ctype, nbytes = _http_post_raw(f"{base}{path}", headers, payload, 30.0)
                return (status, ctype, nbytes), None
            except Exception as exc:  # noqa: BLE001
                code = getattr(exc, "code", None)
                if code == 429 and attempt < retries:
                    time.sleep(backoff * (attempt + 1))
                    continue
                return None, code
        return None, 429

    result: dict[str, Any] = {
        "id": model_id,
        "chat": False,
        "embeddings": False,
        "vision": False,
        "image": False,
        "audio": False,
        "identity": None,
    }
    codes: list[Any] = []
    data, code = _post(
        "/chat/completions",
        {
            "model": model_id,
            "messages": [{"role": "user", "content": "What model are you? Reply briefly."}],
            "max_tokens": 256,  # headroom: reasoning models spend tokens before the answer
        },
    )
    if isinstance(data, dict) and data.get("choices"):
        result["chat"] = True  # the endpoint served a well-formed completion
        choice = data["choices"][0] or {}
        content = (choice.get("message", {}) or {}).get("content", "") or ""
        result["identity"] = str(content).strip()[:200] or None
        if not result["identity"]:
            # live, but said nothing -- usually token budget on a reasoning model
            fr = choice.get("finish_reason")
            result["empty_reason"] = (
                f"responded but empty content (finish_reason={fr})"
                + ("; raise max_tokens" if fr == "length" else "")
            )
        # Vision is only worth checking once text works.
        vdata, _vcode = _post(
            "/chat/completions",
            {
                "model": model_id,
                "max_tokens": 5,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What color is this square? Reply with one word."},
                            {"type": "image_url", "image_url": {"url": _green_square_data_url()}},
                        ],
                    }
                ],
            },
        )
        if isinstance(vdata, dict) and vdata.get("choices"):
            vcontent = ((vdata["choices"][0] or {}).get("message", {}) or {}).get("content", "") or ""
            if "green" in str(vcontent).lower():
                result["vision"] = True
    else:
        # Text failed -> a non-chat model kind. Note the code; embeddings below.
        codes.append(code)
        edata, ecode = _post("/embeddings", {"model": model_id, "input": "ping"})
        if isinstance(edata, dict) and edata.get("data"):
            result["embeddings"] = True
        else:
            codes.append(ecode)
    # Image generation and audio (TTS) are independent kinds -- a model may do
    # them alongside (or instead of) chat -- so probe them regardless, with
    # simple jobs. Only counted as a failure code when chat also failed, so a
    # plain chat model isn't marked not_loaded just for lacking image/audio.
    idata, icode = _post("/images/generations", {"model": model_id, "prompt": "a small green square", "n": 1})
    if isinstance(idata, dict) and idata.get("data"):
        result["image"] = True
    elif not result["chat"]:
        codes.append(icode)
    araw, acode = _post_raw("/audio/speech", {"model": model_id, "input": "ping", "voice": "alloy"})
    if araw is not None and araw[0] == 200 and (araw[1].startswith("audio/") or araw[2] > 0):
        result["audio"] = True
    elif not result["chat"]:
        codes.append(acode)
    # "live" is reserved for models that actually chat back. A model that only
    # serves a non-chat kind (embeddings/image/audio) is usable but not "live".
    # A model whose chat endpoint answers but returns no usable content (empty)
    # and serves nothing else is "reachable" -- not dead, but it won't chat.
    chat_usable = bool(result["chat"] and (result["identity"] or result["vision"]))
    served_nonchat = [k for k in ("embeddings", "image", "audio") if result[k]]
    result["live"] = bool(chat_usable or served_nonchat)
    if chat_usable:
        result["status"] = "live"
    elif served_nonchat:
        result["status"] = f"{served_nonchat[0]}-only" if len(served_nonchat) == 1 else "serving"
    elif result["chat"]:
        # endpoint answered, but no usable chat content and nothing else served
        result["status"] = "reachable"
    elif 429 in codes:
        result["status"] = "rate_limited"
    elif codes and all(c in (400, 404) for c in codes):
        result["status"] = "not_loaded"
    else:
        result["status"] = "error"
    # Human-readable note on what the backend actually returned, so an operator
    # can judge whether a non-live result made sense -- confirming the exact 4xx.
    seen = [c for c in codes if c is not None]
    codes_str = ", ".join(dict.fromkeys(str(c) for c in seen))  # dedupe, keep order
    if chat_usable or served_nonchat:
        kinds = [k for k in ("chat", "vision", "embeddings", "image", "audio") if result[k]]
        note = "served: " + ", ".join(kinds)
        if not chat_usable:
            note += " (no chat)"
        if result.get("empty_reason"):
            note += f" ({result['empty_reason']})"
        result["notes"] = note
    elif result["status"] == "reachable":
        er = result.get("empty_reason") or "returned no usable chat content"
        result["notes"] = f"reachable but won't chat: {er}"
    elif result["status"] == "not_loaded":
        result["notes"] = f"confirmed not loaded (HTTP {codes_str})"
    elif seen:
        result["notes"] = f"returned error immediately; HTTP code(s): {codes_str}"
    else:
        result["notes"] = "no response / unrecognized reply shape"
    return result


def _validate_models(base: str, headers: dict[str, str], model_ids: list[str]) -> list[str]:
    """Keep only claimed models that aren't definitively dead (a real 404 on
    every surface). Live / rate-limited / transient are kept (don't drop over a
    blip). Sequential with gentle spacing to avoid self-tripping rate limits."""
    kept: list[str] = []
    for model_id in model_ids:
        if _probe_modalities_sync(base, headers, model_id).get("status") != "not_loaded":
            kept.append(model_id)
        time.sleep(0.3)
    return kept


def _resolve_interval(sources: list[dict[str, Any]]) -> float | None:
    """Resolve a validation cadence in seconds (or None = never) by walking
    ``sources`` most-specific first (e.g. [node, services.models, agent]), then
    the server default, then the hard default. ``validation_interval_override``
    wins over everything (draconian). A concrete value / null / "never" stops
    the walk; "default"/absent inherits from the next parent up."""
    if _validation_interval_override is not None:
        return _parse_interval(_validation_interval_override)
    for src in list(sources) + [{"validation_interval": _validation_interval}]:
        if not isinstance(src, dict):
            continue
        if "validation_interval" in src:
            val = src["validation_interval"]
        elif "update_interval" in src:
            val = src["update_interval"]
        else:
            continue
        if isinstance(val, str) and val.strip().lower() in ("default", "system", "inherit"):
            continue  # inherit from the next parent up
        return _parse_interval(val)
    return _parse_interval(_DEFAULT_VALIDATION_INTERVAL)


def _fetch_models_cached(agent: dict[str, Any]) -> list[str]:
    """Models for an advertised agent, honoring ``update_interval``: refresh
    from the backend at most once per interval and cache; ``null`` (or no
    interval) uses the agent's configured ``models`` list. Offline-safe: a
    failed fetch caches the fallback so it isn't retried until the interval."""
    configured = [mid for m in _agent_model_list(agent) if (mid := _model_id(m))]
    services = agent.get("services") if isinstance(agent.get("services"), dict) else {}
    models_cfg = services.get("models") if isinstance(services.get("models"), dict) else {}
    # agent-level list-refresh cadence: services.models -> agent -> server
    # default -> hard default (validation_interval_override forces all).
    interval = _resolve_interval([models_cfg, agent])
    validate = bool(models_cfg.get("validate"))
    if interval is None and not validate:
        return configured
    base = str(agent.get("base_url") or "").rstrip("/")
    if not base:
        return configured
    ttl = interval if interval is not None else _MODEL_FETCH_TTL_SECONDS
    now = time.time()
    cached = _model_fetch_cache.get(base)
    if cached and (now - cached["fetched_at"]) < ttl:
        return cached["models"]
    headers: dict[str, str] = {}
    key = _backend_api_key(agent)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        if interval is not None:
            data = _http_get_json(f"{base}/models", headers)
            entries = data.get("data") if isinstance(data, dict) else None
            claimed = (
                [str(m["id"]) for m in entries if isinstance(m, dict) and m.get("id")]
                if isinstance(entries, list)
                else []
            ) or configured
        else:
            claimed = configured
        models = _validate_models(base, headers, claimed) if validate else claimed
        _model_fetch_cache[base] = {"fetched_at": now, "models": models, "source": "live"}
        return models
    except Exception:  # noqa: BLE001 -- offline/unreachable: use cache, else config models
        fallback = cached["models"] if cached else configured
        _model_fetch_cache[base] = {"fetched_at": now, "models": fallback, "source": "config"}
        return fallback


def advertised_catalog() -> dict[str, Any]:
    """The effective user-facing model catalog ("the one we make in the end"):
    the server-level ``services.models`` plus every advertised agent's models
    (refreshed per ``update_interval``), deduped in order."""
    models = list(_advertised_base)
    for agent in _advertised_agents:
        models.extend(_fetch_models_cached(agent))
    seen: set[str] = set()
    deduped: list[str] = []
    for m in models:
        mid = _model_id(m)
        if mid and mid not in seen:
            seen.add(mid)
            deduped.append(mid)
    return {"model": _advertised_default, "models": deduped}


def agents_for_model(model_id: str) -> list[str]:
    """Agents (by id) whose catalog offers ``model_id`` and isn't known-dead,
    in config order -- the cross-agent failover order. The validated cache is
    the authority: a node with ``status == "not_loaded"`` (or ``live: false``)
    is skipped, so if one agent's model is down another that offers it wins."""
    out: list[str] = []
    for agent in _advertised_agents:
        agent_id = str(agent.get("id") or agent.get("name") or "")
        for entry in _agent_model_list(agent):
            if _model_id(entry) != model_id:
                continue
            dead = isinstance(entry, dict) and (
                entry.get("live") is False or entry.get("status") == "not_loaded"
            )
            if not dead and agent_id and agent_id not in out:
                out.append(agent_id)
            break
    return out


def model_failover_map() -> dict[str, list[str]]:
    """Every advertised model -> the ordered list of live agents that can serve
    it (primary first). Models with >1 entry have cross-agent failover."""
    models: list[str] = []
    seen: set[str] = set()
    for agent in _advertised_agents:
        for entry in _agent_model_list(agent):
            mid = _model_id(entry)
            if mid and mid not in seen:
                seen.add(mid)
                models.append(mid)
    return {mid: agents_for_model(mid) for mid in models}


_round_robin_state: dict[str, int] = {}


def _live_catalog(entries: list[Any]) -> list[str]:
    """Model ids from a catalog, skipping validated-dead nodes (status
    not_loaded / live:false). The validated cache is the authority."""
    live: list[str] = []
    for entry in entries or []:
        mid = _model_id(entry)
        if not mid:
            continue
        dead = isinstance(entry, dict) and (entry.get("live") is False or entry.get("status") == "not_loaded")
        if not dead:
            live.append(mid)
    return live


def select_from_catalog(
    entries: list[Any], strategy: str = "failover", *, key: str = "default"
) -> tuple[str | None, list[str]]:
    """Pick a model id from a service catalog, honoring liveness + strategy.
    Returns ``(chosen, ordered)`` where ``ordered`` is the failover order after
    the chosen one. Strategies: ``failover`` (config order, first live),
    ``round-robin`` (rotate, keyed by ``key``), ``random``. Dead entries are
    filtered out first, so no strategy ever routes to a known-dead model."""
    live = _live_catalog(entries)
    if not live:
        return None, []
    strat = (strategy or "failover").strip().lower()
    if strat == "round-robin":
        index = _round_robin_state.get(key, 0) % len(live)
        _round_robin_state[key] = index + 1
        return live[index], live[index:] + live[:index]
    if strat == "random":
        import random

        index = random.randrange(len(live))
        return live[index], live[index:] + live[:index]
    return live[0], live  # failover (default)


def resolve_service_route(agent: dict[str, Any], service: str) -> tuple[str | None, list[str]]:
    """For an ``aggregate`` service on an agent, pick a model from its catalog
    per its ``strategy`` (live-filtered). Returns ``(chosen, ordered)``; the
    ``models`` service falls back to the agent's model list when it has no
    explicit catalog."""
    services = agent.get("services") if isinstance(agent.get("services"), dict) else {}
    cfg = services.get(service) if isinstance(services.get(service), dict) else {}
    if cfg.get("behavior") != "aggregate":
        return None, []
    catalog = cfg.get("catalog")
    if not isinstance(catalog, list):
        catalog = _agent_model_list(agent) if service == "models" else []
    key = f"{agent.get('id') or agent.get('name') or ''}/{service}"
    return select_from_catalog(catalog, cfg.get("strategy") or "failover", key=key)


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _iso_age(timestamp: Any) -> float:
    """Seconds since an ISO-8601 timestamp; inf if unparseable/missing."""
    from datetime import datetime, timezone

    try:
        dt = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds()
    except (ValueError, TypeError):
        return float("inf")


def _probe_with_timeout(
    base: str, headers: dict[str, str], model_id: str, timeout: float
) -> dict[str, Any]:
    """Run the per-model IQ test but give up if it never returns within
    ``timeout`` seconds -> a ``status: "timeout"`` node with a description that
    tells the operator to raise the timeout. The stuck worker thread is
    abandoned (shutdown wait=False), never blocking the run."""
    import concurrent.futures

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_probe_modalities_sync, base, headers, model_id)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        return {
            "id": model_id,
            "status": "timeout",
            "live": False,
            "chat": False,
            "embeddings": False,
            "vision": False,
            "image": False,
            "audio": False,
            "identity": None,
            "description": (
                f"timeout after {int(timeout)} seconds "
                "(please raise the timeout in this section if you think it is needed)"
            ),
        }
    finally:
        executor.shutdown(wait=False)


def validate_agent_models(agent: dict[str, Any]) -> list[dict[str, Any]]:
    """Run the "IQ test" on each of an agent's claimed models and return result
    nodes: ``{id, validation_startedAt, validation_doneAt, status, chat,
    embeddings, vision, identity}``. These nodes can be written back into the
    config's ``models`` list (which reads ids or nodes interchangeably) so the
    validation state persists."""
    services = agent.get("services") if isinstance(agent.get("services"), dict) else {}
    models_cfg = services.get("models") if isinstance(services.get("models"), dict) else {}
    entries = _agent_model_list(agent)
    base = str(agent.get("base_url") or "").rstrip("/")
    if not base:
        return [{"id": mid} for m in entries if (mid := _model_id(m))]
    headers: dict[str, str] = {}
    key = _backend_api_key(agent)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    nodes: list[dict[str, Any]] = []
    for entry in entries:
        mid = _model_id(entry)
        if not mid:
            continue
        node_cfg = entry if isinstance(entry, dict) else {}
        # a node overrides its parent: node -> services.models -> agent -> server.
        node_interval = _resolve_interval([node_cfg, models_cfg, agent])
        done = node_cfg.get("validation_doneAt")
        # skip re-testing when this node is set to never, or was validated
        # within its (node-overridden) interval; otherwise run the IQ test.
        if node_interval is None or (done is not None and _iso_age(done) < node_interval):
            nodes.append(entry)  # keep the existing entry as-is
            continue
        started = _now_iso()
        # per-model budget: node -> services.models -> agent -> default 120s.
        timeout = _VALIDATION_TIMEOUT_SECONDS
        for _src in (node_cfg, models_cfg, agent):
            if _src.get("validation_timeout") is not None:
                timeout = _parse_interval(_src["validation_timeout"]) or _VALIDATION_TIMEOUT_SECONDS
                break
        result = _probe_with_timeout(base, headers, mid, timeout)
        node = {
            "id": mid,
            "validation_startedAt": started,
            "validation_doneAt": _now_iso(),
            "status": result["status"],
            "chat": result["chat"],
            "embeddings": result["embeddings"],
            "vision": result["vision"],
            "image": result["image"],
            "audio": result["audio"],
            "identity": result["identity"],
        }
        if result.get("notes"):
            node["notes"] = result["notes"]
        if result.get("description"):
            node["description"] = result["description"]  # e.g. the timeout note
        nodes.append(node)
        time.sleep(0.3)
    return nodes


async def _wait_for_capability(worker_id: str, capability: str) -> bool:
    """Poll until a worker (re)connects and opts into ``capability`` (returns
    True), or 504 when _REQUEST_TIMEOUT_SECONDS elapses. Honors a live
    opt-out (501) that appears while waiting."""
    deadline = time.monotonic() + _REQUEST_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        await asyncio.sleep(0.5)
        _raise_if_capability_declined(worker_id, capability)
        if _worker_can_pretend(worker_id, capability):
            return True
    raise HTTPException(
        status_code=504,
        detail=f"no worker opted into '{capability}' for '{worker_id}' (wait timed out)",
    )


async def _capable_or_policy(worker_id: str, capability: str) -> bool:
    """Decide how a non-text capability request is served.

    Order of precedence: a live worker opt-out is a hard 501; then this
    agent's per-service ``behavior`` from config (serve/stub/wait/error/
    decline); then an opted-in worker shapes it; then the server-level
    per-service ``fallback`` (else the global ``capability_fallback``):
    stub answers with the local fake, wait holds until a worker opts in
    (504 on timeout), error fails fast (503). Returns True to shape via a
    worker, False to use the local stub."""
    _raise_if_capability_declined(worker_id, capability)
    service = _CAPABILITY_SERVICE.get(capability, capability)
    behavior = _worker_service_behavior.get(worker_id, {}).get(service)
    if behavior == "decline":
        raise HTTPException(
            status_code=501,
            detail=f"agent '{worker_id}' does not offer '{service}' (services.{service}=decline)",
        )
    if behavior in ("serve", "aggregate"):
        return True  # the agent volunteers for this service
    if behavior == "stub":
        return False
    if behavior == "error":
        raise HTTPException(
            status_code=503,
            detail=f"agent '{worker_id}' cannot serve '{service}' now (services.{service}=error)",
        )
    if behavior == "wait":
        return await _wait_for_capability(worker_id, capability)

    if _worker_can_pretend(worker_id, capability):
        return True
    # Server-level fallback CHAIN: agents just volunteered/rejected above; the
    # system now decides. Strategy tokens (round-robin/failover/...) mean "use a
    # volunteering agent" -- none did here, so pass to the next token. stub /
    # error / wait are terminals.
    chain = _service_fallback.get(service) or [_capability_fallback()]
    for token in chain:
        if token in _STRATEGY_TOKENS:
            continue  # no agent volunteered for this service -> next in chain
        if token == "stub":
            return False
        if token == "error":
            raise HTTPException(
                status_code=503,
                detail=f"no agent volunteered for '{capability}' ('{worker_id}'; fallback={','.join(chain)})",
            )
        if token == "wait":
            return await _wait_for_capability(worker_id, capability)
    raise HTTPException(
        status_code=503,
        detail=f"no agent volunteered for '{capability}' ('{worker_id}'; fallback={','.join(chain)})",
    )


async def _mirror_to_observers(
    answering_worker_id: str, model: str, prompt_text: str, reply_text: str, service: str = "chat"
) -> None:
    """Best-effort: mirror an exchange to any config-declared observer whose
    scope covers ``service`` (as in proxy-observe). Never fails the request."""
    for observer_id, scope in list(_observers.items()):
        if observer_id == answering_worker_id:
            continue
        if scope is not True and service not in scope:
            continue
        await _observe(observer_id, model, prompt_text, reply_text)


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def _extract_images(content: Any) -> list[str]:
    """Pull image URLs / data-URLs out of an OpenAI-style message content list
    (``{"type": "image_url", "image_url": {"url": ...}}`` or ``input_image``)."""
    urls: list[str] = []
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") in ("image_url", "input_image"):
                iu = item.get("image_url")
                url = iu.get("url") if isinstance(iu, dict) else iu
                if isinstance(url, str) and url:
                    urls.append(url)
    return urls


def _reply_content(result: Any) -> str:
    """The text of a relay reply, whether the worker answered with a bare
    string (legacy/mock/proxy) or a structured dict (real two-way worker)."""
    if isinstance(result, dict):
        return str(result.get("content") or "")
    if result is None or result is _PASS:
        return ""
    return str(result)


def _reply_image(result: Any) -> tuple[str | None, str | None, str | None]:
    """Any real image a worker returned two-way: (image_b64, image_url, mime)."""
    if isinstance(result, dict):
        return result.get("image_b64"), result.get("image_url"), result.get("mime")
    return None, None, None


def _reply_audio(result: Any) -> tuple[str | None, str | None, str | None]:
    """Any real audio a worker returned two-way: (audio_b64, audio_url, mime)."""
    if isinstance(result, dict):
        return result.get("audio_b64"), result.get("audio_url"), result.get("mime")
    return None, None, None


def _dataurl_to_b64(url: str | None) -> str | None:
    if isinstance(url, str) and url.startswith("data:") and "," in url:
        return url.split(",", 1)[1]
    return None



def _model_entry(worker_id: str, suffix: str, persona: dict[str, Any]) -> dict[str, Any]:
    model_id = f"{worker_id}/{suffix}"
    return {
        "id": model_id,
        "object": "model",
        "display_name": f"{worker_id} {persona['display_name']}",
        "context_length": 200000,
        "supported_parameters": [],
        "owned_by": worker_id,
        "connected": worker_id in _connected_workers,
    }


class ChatMessage(BaseModel):
    role: str
    content: Any = ""


class ChatRequest(BaseModel):
    model: str = _DEFAULT_MODEL_ID
    messages: list[ChatMessage] = Field(default_factory=list)
    temperature: float | None = Field(default=None, ge=0, le=2)
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str = _DEFAULT_MODEL_ID
    prompt: Any = ""
    temperature: float | None = Field(default=None, ge=0, le=2)
    stream: bool = False


class ResponsesRequest(BaseModel):
    model: str = _DEFAULT_MODEL_ID
    input: Any = ""
    temperature: float | None = Field(default=None, ge=0, le=2)
    stream: bool = False


class EmbeddingsRequest(BaseModel):
    model: str = _DEFAULT_MODEL_ID
    input: Any = ""
    dimensions: int = Field(default=8, ge=1, le=3072)
    encoding_format: Literal["float"] = "float"


class ModerationsRequest(BaseModel):
    model: str = _DEFAULT_MODEL_ID
    input: Any = ""


class ImagesRequest(BaseModel):
    model: str = _DEFAULT_MODEL_ID
    prompt: str = Field(min_length=1)
    n: int = Field(default=1, ge=1, le=10)
    size: str = "256x256"
    response_format: Literal["url", "b64_json"] = "url"


class AudioSpeechRequest(BaseModel):
    model: str = _DEFAULT_MODEL_ID
    input: str = Field(min_length=1, max_length=4096)
    voice: str = "stub"
    response_format: Literal["wav"] = "wav"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)


def _proxy_available() -> bool:
    """True when a proxy mode is active and a backend is configured -- i.e. we
    can forward an otherwise-unknown model id straight to a real backend."""
    modes = _current_modes()
    return ("proxy" in modes or "proxy-observe" in modes) and _select_backend() is not None


def _require_model(model: str) -> tuple[str, str, dict[str, Any]]:
    # A configured/runtime route sends this exact catalog id to a serving worker,
    # forwarding the original id so the worker knows which model to emulate.
    route = _model_routes.get(model)
    if route:
        return route, "", {
            "instruction": f"You are serving model id '{model}'. Emulate that model faithfully.",
            "passthrough": True,
            "served_model": model,
        }
    worker_id, suffix = _split_model_id(model)
    persona = _models_for(worker_id).get(suffix)
    if persona is None:
        # In proxy mode, accept the backend's own model ids (e.g. a real
        # multi-model upstream) instead of 404ing -- they're forwarded as-is.
        if _proxy_available():
            return model, "", {"instruction": None, "passthrough": True}
        raise HTTPException(status_code=404, detail=f"unknown model '{model}'")
    return worker_id, suffix, persona


def _backend_model_for(model: str, backend: dict[str, Any]) -> str:
    """The model id to send upstream: forward a real backend model id as-is;
    for a local persona/default, use the backend's configured model."""
    worker_id, suffix = _split_model_id(model)
    if model == _DEFAULT_MODEL_ID or _models_for(worker_id).get(suffix) is not None:
        return str(backend.get("model") or model)
    return model


def _token_count(value: Any) -> int:
    text = _flatten_content(value)
    return len(text.split()) if text else 0


async def _relay_full(
    model: str,
    prompt_text: str,
    *,
    images: list[str] | None = None,
    audio: str | None = None,
    files: dict[str, Any] | None = None,
    kind: str | None = None,
) -> Any:
    """Forward prompt_text (and optionally real media -- ``images`` for a
    vision/image job, ``audio`` for a speech/transcription job, ``files`` for a
    fine-tune job) to the worker registered for this model's worker_id, and
    await its reply. Returns the raw reply -- a structured dict for a two-way
    worker (``content`` plus optional ``image_b64``/``image_url``/``audio_b64``/
    ``audio_url``/``file_id``/``mime``) or a bare string for legacy/mock/proxy
    peers. Use :func:`_relay` for text-only.

    If that specific worker isn't connected right now (e.g. it's between
    connect cycles, or simply never showed up), this does NOT fail fast --
    it just waits, like a slow API server, polling for that worker to
    (re)connect and retrying the send, until _REQUEST_TIMEOUT_SECONDS has
    elapsed overall. Only then does it give up with a 504. This matches
    the intended behavior of an occasionally-away human/agent worker: a
    caller should experience "slow", not "broken".

    Overload protection is the opposite: if this worker_id has already
    been used _USAGE_MAX_PER_WINDOW times in the last _USAGE_WINDOW_SECONDS,
    this fails FAST with 429 (and a Retry-After telling the caller when to
    come back -- possibly minutes away), rather than queuing yet more work
    onto an already-busy worker."""
    worker_id, _, persona = _require_model(model)
    _check_and_record_usage(worker_id)
    instruction = persona.get("instruction")

    extra: dict[str, Any] = {}
    if images:
        extra["images"] = images
    if audio:
        extra["audio"] = audio
    if files:
        extra["files"] = files
    if kind:
        extra["kind"] = kind

    modes = _current_modes()
    for mode in modes:
        result = await _relay_step(mode, worker_id, model, prompt_text, instruction, extra or None)
        if result is not _PASS:
            await _mirror_to_observers(worker_id, model, prompt_text, _reply_content(result))
            return result
    # Every step passed (e.g. `recruit` with nobody connected and no fallback).
    raise HTTPException(
        status_code=504,
        detail=f"no strategy produced a reply for '{worker_id}' (modes={','.join(modes)})",
    )


async def _relay(model: str, prompt_text: str) -> str:
    """Text-only convenience over :func:`_relay_full` -- returns just the
    reply's text content, so all the plain-text endpoints stay unchanged."""
    return _reply_content(await _relay_full(model, prompt_text))


async def _relay_step(
    mode: str, worker_id: str, model: str, prompt_text: str, instruction: str | None, extra: dict[str, Any] | None = None
) -> Any:
    """Run one mode in the fallback chain. Returns a reply (string, or a
    structured dict for a two-way worker), or _PASS to fall through to the next
    mode. May raise HTTPException to stop the chain (e.g. error-when-empty, or a
    misconfigured proxy)."""
    if mode == "mock":
        # `mock` = pretend a worker was present at the websocket and the
        # exchange succeeded. We run the real relay plumbing (request payload,
        # pending future, reply) against a pretend peer, rather than
        # fabricating a response outside the transport. A registered peer
        # (from register_mock_workers) is used if present; otherwise an
        # ephemeral pretend peer answers with a deterministic success.
        if worker_id in _connected_workers:
            result = await _relay_to_worker(worker_id, model, prompt_text, instruction, wait=False, extra=extra)
            if result is not _PASS:
                return result
        canned = _mock_reply(model, prompt_text, instruction)
        pretend_peer = _MockWorker(worker_id, reply=canned)
        return await _relay_to_worker(
            worker_id, model, prompt_text, instruction, wait=False, peer_override=pretend_peer, extra=extra
        )

    if mode in ("proxy", "proxy-observe"):
        backend = _select_backend()
        if backend is None:
            raise HTTPException(
                status_code=502,
                detail="proxy mode: no backend configured (set config.json 'backends' or EMULLM_PROXY_BASE_URL)",
            )
        reply = await _proxy_chat(backend, model, prompt_text, instruction)
        if mode == "proxy-observe":
            await _observe(worker_id, model, prompt_text, reply)
        return reply

    if mode == "error-when-empty":
        if worker_id not in _connected_workers:
            raise HTTPException(
                status_code=503,
                detail=f"no emullm worker connected for '{worker_id}' (mode=error-when-empty)",
            )
        return await _relay_to_worker(worker_id, model, prompt_text, instruction, wait=False, extra=extra)

    if mode in ("recruit", "self", "auto"):
        # Use a connected worker if one is present right now; don't wait.
        return await _relay_to_worker(worker_id, model, prompt_text, instruction, wait=False, extra=extra)

    # wait / wait-then-serve / relay / unknown -> the classic wait-for-a-worker path.
    return await _relay_to_worker(worker_id, model, prompt_text, instruction, wait=True, extra=extra)


async def _relay_to_worker(
    worker_id: str,
    model: str,
    prompt_text: str,
    instruction: str | None,
    *,
    wait: bool,
    peer_override: Any = None,
    extra: dict[str, Any] | None = None,
) -> Any:
    """Send the request to the worker for worker_id and await its reply.

    If ``peer_override`` is given (e.g. an ephemeral pretend peer for `mock`
    mode), it is used as the websocket peer instead of a registered worker,
    so the full request/reply plumbing still runs -- the system behaves as
    if a worker were present at the websocket and the exchange succeeded.

    If ``wait`` is True this polls for the worker to (re)connect up to
    _REQUEST_TIMEOUT_SECONDS (the classic "slow API" behavior, 504 on
    timeout). If ``wait`` is False and no peer is available, it returns
    _PASS so the caller can fall through to the next mode in the chain."""
    request_id = uuid.uuid4().hex
    loop = asyncio.get_running_loop()
    future: "asyncio.Future[Any]" = loop.create_future()
    _pending[request_id] = future

    payload = {
        "type": "request",
        "id": request_id,
        "model": model,
        "worker_id": worker_id,
        "prompt": prompt_text,
    }
    if instruction:
        payload["persona_instruction"] = instruction
    if extra:
        # real two-way extras: ``images`` (list of urls/data-urls) and ``kind``
        for key, value in extra.items():
            if value is not None:
                payload[key] = value

    deadline = time.monotonic() + _REQUEST_TIMEOUT_SECONDS
    try:
        while True:
            peer = peer_override if peer_override is not None else _connected_workers.get(worker_id)
            if peer is None:
                if not wait:
                    return _PASS
                if time.monotonic() >= deadline:
                    raise HTTPException(
                        status_code=504,
                        detail=f"no emullm worker registered as '{worker_id}' (timed out waiting)",
                    )
                await asyncio.sleep(0.5)
                continue
            try:
                await peer.send_json(payload)
                break
            except Exception:
                if not wait:
                    return _PASS
                # That worker may have just disconnected; keep waiting/retrying.
                await asyncio.sleep(0.5)
                continue

        remaining = max(1.0, deadline - time.monotonic())
        try:
            return await asyncio.wait_for(future, timeout=remaining)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="emullm worker did not reply in time")
    finally:
        _pending.pop(request_id, None)


@router.get("/v1/models")
def list_models() -> dict[str, Any]:
    """Aggregates the model/persona menu across every currently connected
    worker, plus the default worker_id's fallback menu even if it isn't
    connected right now (so the primary identity is always discoverable)."""
    worker_ids = sorted(set(_connected_workers) | {_DEFAULT_WORKER_ID})
    data = [
        _model_entry(worker_id, suffix, persona)
        for worker_id in worker_ids
        for suffix, persona in _models_for(worker_id).items()
    ]
    return {"object": "list", "data": data}


@router.get("/v1/models/{model_id:path}")
def get_model(model_id: str) -> dict[str, Any]:
    worker_id, suffix = _split_model_id(model_id)
    persona = _models_for(worker_id).get(suffix)
    if persona is None:
        raise HTTPException(status_code=404, detail=f"unknown model '{model_id}'")
    return _model_entry(worker_id, suffix, persona)


@router.get("/emullm/caps/{worker_id}")
def worker_caps(worker_id: str) -> dict[str, Any]:
    """Quick per-worker lookup: is it connected, what models does it
    offer (its own declared list, or the _PERSONA_SUFFIXES fallback), and
    which non-text "pretend" capabilities has it opted into. A lighter,
    read-only companion to /admin/emullm/state (which lists every
    worker at once)."""
    return {
        "worker_id": worker_id,
        "connected": worker_id in _connected_workers,
        "models": sorted(_models_for(worker_id).keys()),
        "capabilities": _worker_capabilities.get(worker_id, {}),
    }


# ---------------------------------------------------------------------------
# Serves this feature's own design docs (docs/**) straight off
# disk, so e.g. /emullm/docs/EMULLM_RELAY.md always reflects
# whatever is currently checked out -- no separate copy to keep in sync.
#
# Docs can ALSO be "registered" from another directory: register_doc_alias()
# maps a virtual path (as requested under /emullm/docs/) to a real file
# or directory that physically lives ELSEWHERE (outside _DOCS_ROOT -- e.g.
# a .copilotignore'd folder, or another package). This lets a doc kept in
# one directory appear as if it were part of the already-registered docs
# tree, without moving or copying it. Registration is in-process only (no
# HTTP endpoint accepts arbitrary filesystem paths), so it can't be abused
# as a read-anything vector from the network.
# ---------------------------------------------------------------------------
_DOCS_ROOT = next(
    (
        candidate
        for candidate in (
            # src layout (src/emullm/api.py -> repo root /docs)
            Path(__file__).resolve().parent.parent.parent / "docs",
            # flat layout (emullm/api.py -> repo root /docs)
            Path(__file__).resolve().parent.parent / "docs",
        )
        if candidate.is_dir()
    ),
    Path(__file__).resolve().parent.parent / "docs",
)

# virtual rel_path (no leading/trailing slash) -> real file or directory.
_DOC_ALIASES: dict[str, Path] = {}


def register_doc_alias(virtual_rel_path: str, real_path: Path | str) -> None:
    """Register a file or directory that lives outside _DOCS_ROOT so it's
    served under /emullm/docs/<virtual_rel_path>. If real_path is a
    file, it aliases exactly that one virtual path; if it's a directory,
    it aliases the whole subtree beneath that virtual prefix."""
    _DOC_ALIASES[virtual_rel_path.strip("/")] = Path(real_path)


def _resolve_doc_alias(rel_path: str) -> Path | None:
    rel_path = rel_path.strip("/")
    exact = _DOC_ALIASES.get(rel_path)
    if exact is not None and exact.is_file():
        return exact
    # Longest matching directory prefix wins, so nested aliases behave.
    for prefix in sorted(_DOC_ALIASES, key=len, reverse=True):
        target = _DOC_ALIASES[prefix]
        if not target.is_dir():
            continue
        if rel_path == prefix or rel_path.startswith(prefix + "/"):
            suffix = rel_path[len(prefix):].lstrip("/")
            if not suffix or ".." in Path(suffix).parts:
                return None
            candidate = target / suffix
            if candidate.is_file():
                return candidate
    return None


def _substitute_doc_placeholders(text: str, request: Request) -> str:
    """Docs describe example URLs for "this same server" using
    placeholder tokens instead of a hardcoded host/port, so what's served
    always matches wherever this request actually arrived -- whether
    mounted on the main workbench server (:8000) or run standalone on a
    different port (see run_emullm_standalone.py)."""
    host = request.url.hostname or "127.0.0.1"
    port = request.url.port
    scheme = request.url.scheme or "http"
    ws_scheme = "wss" if scheme == "https" else "ws"
    host_port = f"{host}:{port}" if port else host
    return (
        text.replace("{{EMULLM_BASE_URL}}", f"{scheme}://{host_port}")
        .replace("{{EMULLM_WS_HOST}}", host_port)
        .replace("{{EMULLM_WS_BASE_URL}}", f"{ws_scheme}://{host_port}")
    )


@router.get("/emullm/docs/{rel_path:path}")
def serve_doc(rel_path: str, request: Request) -> Response:
    # A registered alias (a doc living outside _DOCS_ROOT) takes priority;
    # otherwise fall back to the real file under _DOCS_ROOT.
    path = _resolve_doc_alias(rel_path)
    if path is None:
        if not rel_path or ".." in Path(rel_path).parts or Path(rel_path).is_absolute():
            raise HTTPException(status_code=400, detail=f"invalid doc path '{rel_path}'")
        path = _DOCS_ROOT / rel_path
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"no such doc '{rel_path}'")
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".md":
        text = _substitute_doc_placeholders(text, request)
    media_type = "text/markdown; charset=utf-8" if path.suffix == ".md" else "text/plain; charset=utf-8"
    return Response(content=text, media_type=media_type)


# ---------------------------------------------------------------------------
# Static HTML (and other assets) under workbench/server/emullm/static/,
# served both under a namespaced /emullm/static/ path and as bare
# top-level /{name}.html files (so e.g. a landing page at
# static/index.html is reachable as /index.html). The bare route only
# matches a single path segment ending in .html, so it can't shadow the
# other /v1, /admin, or /emullm routes.
# ---------------------------------------------------------------------------
_STATIC_ROOT = Path(__file__).resolve().parent / "static"

_STATIC_MEDIA_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


def _serve_static(rel_path: str) -> Response:
    if not rel_path or ".." in Path(rel_path).parts or Path(rel_path).is_absolute():
        raise HTTPException(status_code=400, detail=f"invalid static path '{rel_path}'")
    path = _STATIC_ROOT / rel_path
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"no such static file '{rel_path}'")
    media_type = _STATIC_MEDIA_TYPES.get(path.suffix, "application/octet-stream")
    if media_type.startswith(("text/", "application/json")):
        return Response(content=path.read_text(encoding="utf-8"), media_type=media_type)
    return Response(content=path.read_bytes(), media_type=media_type)


@router.get("/emullm/static/{rel_path:path}")
def serve_static_asset(rel_path: str) -> Response:
    return _serve_static(rel_path)


@router.get("/{filename}.html")
def serve_root_html(filename: str) -> Response:
    """Bare top-level *.html convenience, e.g. /index.html -> static/index.html."""
    return _serve_static(f"{filename}.html")

# ---------------------------------------------------------------------------
# Generic durable storage -- lets a worker "borrow" this server's disk as
# a scratch space (notes, drafts, anything it wants to persist between
# its own connect/rest cycles), independent of the OpenAI /v1/files API.
# Plain path-addressed blobs under runtime/emullm/storage/,
# with no ownership/ACL model -- any worker_id can read/write any path.
# ---------------------------------------------------------------------------
def _storage_root() -> Path:
    return _RUNTIME_DIR / "storage"


def _safe_storage_path(rel_path: str) -> Path:
    if not rel_path or ".." in Path(rel_path).parts or Path(rel_path).is_absolute():
        raise HTTPException(status_code=400, detail=f"invalid storage path '{rel_path}'")
    return _storage_root() / rel_path


@router.get("/emullm/storage")
def storage_list() -> dict[str, Any]:
    root = _storage_root()
    if not root.exists():
        return {"files": []}
    files = sorted(str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*") if p.is_file())
    return {"files": files}


@router.get("/emullm/storage/{rel_path:path}")
def storage_get(rel_path: str) -> Response:
    path = _safe_storage_path(rel_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"no such storage file '{rel_path}'")
    return Response(content=path.read_bytes(), media_type="application/octet-stream")


@router.put("/emullm/storage/{rel_path:path}")
async def storage_put(rel_path: str, request: Request) -> dict[str, Any]:
    path = _safe_storage_path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = await request.body()
    path.write_bytes(body)
    return {"path": rel_path, "bytes": len(body)}


@router.delete("/emullm/storage/{rel_path:path}")
def storage_delete(rel_path: str) -> dict[str, Any]:
    path = _safe_storage_path(rel_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"no such storage file '{rel_path}'")
    path.unlink()
    return {"deleted": rel_path}


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatRequest) -> Any:
    if not body.messages:
        raise HTTPException(status_code=400, detail="messages is required")
    prompt_text = "\n\n".join(f"[{m.role}] {_flatten_content(m.content)}" for m in body.messages)
    images: list[str] = []
    for m in body.messages:
        images.extend(_extract_images(m.content))
    result = await _relay_full(body.model, prompt_text, images=images or None, kind="vision" if images else "chat")
    reply_text = _reply_content(result)
    completion_id = _new_resource_id("chatcmpl")
    created = int(time.time())
    usage = {
        "prompt_tokens": _token_count(prompt_text),
        "completion_tokens": _token_count(reply_text),
    }
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    result = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply_text},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }
    if not body.stream:
        return result

    async def events() -> Any:
        chunks = [
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": body.model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            },
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": body.model,
                "choices": [{"index": 0, "delta": {"content": reply_text}, "finish_reason": None}],
            },
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": body.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]
        for chunk in chunks:
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@router.post("/v1/completions")
async def completions(body: CompletionRequest) -> Any:
    prompt_text = _flatten_content(body.prompt)
    if not prompt_text:
        raise HTTPException(status_code=400, detail="prompt is required")
    reply_text = await _relay(body.model, prompt_text)
    completion_id = _new_resource_id("cmpl")
    created = int(time.time())
    usage = {
        "prompt_tokens": _token_count(prompt_text),
        "completion_tokens": _token_count(reply_text),
    }
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    result = {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": body.model,
        "choices": [{"index": 0, "text": reply_text, "finish_reason": "stop"}],
        "usage": usage,
    }
    if not body.stream:
        return result

    async def events() -> Any:
        chunk = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": body.model,
            "choices": [{"index": 0, "text": reply_text, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@router.post("/v1/responses")
async def responses(body: ResponsesRequest) -> Any:
    prompt_text = _flatten_content(body.input)
    if not prompt_text:
        raise HTTPException(status_code=400, detail="input is required")
    reply_text = await _relay(body.model, prompt_text)
    response_id = _new_resource_id("resp")
    created_at = int(time.time())
    output_id = _new_resource_id("msg")
    usage = {
        "input_tokens": _token_count(prompt_text),
        "output_tokens": _token_count(reply_text),
    }
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    result = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "model": body.model,
        "status": "completed",
        "output_text": reply_text,
        "output": [
            {
                "id": output_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": reply_text}],
            }
        ],
        "usage": usage,
    }
    if not body.stream:
        return result

    async def events() -> Any:
        started = {**result, "status": "in_progress", "output": [], "output_text": ""}
        frames = [
            ("response.created", {"type": "response.created", "response": started}),
            (
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": output_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": reply_text,
                },
            ),
            ("response.completed", {"type": "response.completed", "response": result}),
        ]
        for event_name, data in frames:
            yield f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@router.post("/v1/embeddings")
async def embeddings(body: EmbeddingsRequest) -> dict[str, Any]:
    """NOT a real embedding. If the target worker declared it's willing
    to "pretend" at embeddings (capabilities.embeddings=true at register
    time), it's asked -- via the normal text relay, so this routes to
    the right worker for its worker_id -- to describe the text's key
    semantic features, and THAT description is hashed into the vector.
    Otherwise the raw input text is hashed directly. Either way the
    result is a deterministic (given the same wording) pseudo-random
    vector, never a real embedding. If the worker EXPLICITLY declared it
    won't do embeddings (capabilities.embeddings=false), this stops with
    501 instead of falling back to the stub -- and the worker is never
    even asked."""
    worker_id, _, _ = _require_model(body.model)
    can_pretend = await _capable_or_policy(worker_id, "embeddings")
    inputs = body.input if isinstance(body.input, list) else [body.input]
    if not inputs:
        raise HTTPException(status_code=400, detail="input must not be empty")
    dimension = body.dimensions
    data = []
    input_tokens = 0
    for index, item in enumerate(inputs):
        text = _flatten_content(item)
        if not text:
            raise HTTPException(status_code=400, detail=f"input at index {index} must not be empty")
        input_tokens += _token_count(text)
        if can_pretend:
            text = await _relay(
                body.model,
                "(pretend-embeddings) In one short sentence, describe the key "
                f"semantic features of this text as if about to embed it: {text}",
            )
        vector: list[float] = []
        block = 0
        while len(vector) < dimension:
            digest = hashlib.sha256(f"{text}\0{block}".encode("utf-8")).digest()
            vector.extend(((byte / 255.0) * 2.0 - 1.0) for byte in digest)
            block += 1
        vector = vector[:dimension]
        data.append({"index": index, "object": "embedding", "embedding": vector})
    return {
        "object": "list",
        "model": body.model,
        "data": data,
        "usage": {"prompt_tokens": input_tokens, "total_tokens": input_tokens},
    }


@router.post("/v1/moderations")
async def moderations(body: ModerationsRequest) -> dict[str, Any]:
    """Stub. If the target worker declared moderations capability, ask it
    (via the normal relay, routed to that worker) whether the input
    should be flagged, and use its verdict; otherwise always reports the
    input as not flagged. If the worker EXPLICITLY declared it won't do
    moderations, this stops with 501 instead -- the worker is never
    asked."""
    worker_id, _, _ = _require_model(body.model)
    can_pretend = await _capable_or_policy(worker_id, "moderations")
    inputs = body.input if isinstance(body.input, list) else [body.input]
    if not inputs:
        raise HTTPException(status_code=400, detail="input must not be empty")
    category_names = (
        "harassment",
        "harassment/threatening",
        "hate",
        "hate/threatening",
        "illicit",
        "illicit/violent",
        "self-harm",
        "self-harm/intent",
        "self-harm/instructions",
        "sexual",
        "sexual/minors",
        "violence",
        "violence/graphic",
    )
    results = []
    for item in inputs:
        if not _flatten_content(item):
            raise HTTPException(status_code=400, detail="input must not be empty")
        flagged = False
        if can_pretend:
            verdict = await _relay(
                body.model,
                "(pretend-moderation) Reply with exactly one word, FLAG or OK, for "
                f"whether this content should be moderation-flagged: {_flatten_content(item)}",
            )
            flagged = "flag" in verdict.strip().lower()
        results.append(
            {
                "flagged": flagged,
                "categories": {name: flagged for name in category_names},
                "category_scores": {name: 1.0 if flagged else 0.0 for name in category_names},
                "category_applied_input_types": {name: ["text"] for name in category_names},
            }
        )
    return {"id": _new_resource_id("modr"), "model": body.model, "results": results}


_STUB_PIXEL_PNG_DATA_URL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


@router.post("/v1/images/generations")
async def images_generations(body: ImagesRequest) -> dict[str, Any]:
    """Two-way image pass-through. If the target worker declared images
    capability, it's routed the prompt and given the chance to return a REAL
    image (``image_b64``/``image_url`` in its reply) -- which we hand straight
    back to the caller. If it only replies with text (an older/text worker),
    that goes in ``pretend_description`` alongside a 1x1 placeholder. If the
    worker EXPLICITLY declared it won't do images, this stops with 501 instead
    -- the worker is never asked."""
    worker_id, _, _ = _require_model(body.model)
    can_pretend = await _capable_or_policy(worker_id, "images")
    pretend_description = None
    worker_b64 = worker_url = worker_mime = None
    if can_pretend:
        result = await _relay_full(
            body.model,
            "(image-generation) Produce an image for this prompt. If you can, return it "
            "as base64 PNG in an 'image_b64' field; otherwise describe, in one or two "
            f"sentences, the image you would generate: {body.prompt}",
            kind="image",
        )
        worker_b64, worker_url, worker_mime = _reply_image(result)
        if not (worker_b64 or worker_url):
            pretend_description = _reply_content(result) or None
    entry: dict[str, Any] = {"revised_prompt": body.prompt}
    if worker_b64 or worker_url:
        # A real image came back from the worker. Persist it to the shared cloud
        # files store and hand back a stable URL (plus b64 if the caller asked).
        data, mime = _decode_media(worker_b64, worker_url)
        if data is not None:
            record = _store_cloud_bytes(data, f"image{_ext_for_mime(mime or worker_mime)}", purpose="output")
            entry["file_id"] = record["id"]
            entry["source"] = "worker"
            if body.response_format == "b64_json":
                entry["b64_json"] = worker_b64 or base64.b64encode(data).decode("ascii")
            else:
                entry["url"] = _cloud_file_url(record["id"])
        else:
            # A non-data URL (already hosted somewhere) -- pass it straight through.
            entry["url"] = worker_url
            entry["source"] = "worker"
    else:
        if body.response_format == "b64_json":
            entry["b64_json"] = _STUB_PIXEL_PNG_DATA_URL.split(",", 1)[1]
        else:
            entry["url"] = _STUB_PIXEL_PNG_DATA_URL
        if pretend_description:
            entry["pretend_description"] = pretend_description
    return {"created": int(time.time()), "data": [dict(entry) for _ in range(body.n)]}


_MAX_AUDIO_BYTES = max(1, int(os.environ.get("EMULLM_MAX_AUDIO_BYTES", str(25 * 1024 * 1024))))


async def _audio_transcription_from_request(
    request: Request,
    forced_worker_id: str | None = None,
) -> dict[str, Any]:
    """Two-way transcription. The uploaded audio is persisted to the shared
    cloud files store and, if the target worker declared audio_transcription
    capability, the worker is routed a reference to the REAL clip (a cloud file
    URL) so it can actually work from the bytes; its reply is the transcript.
    Otherwise returns a fixed "not implemented" notice. If the worker EXPLICITLY
    declared it won't do audio_transcription, this stops with 501 instead."""
    if not request.headers.get("content-type", "").startswith("multipart/form-data"):
        raise HTTPException(status_code=415, detail="use multipart/form-data with a file field")
    form = await request.form()
    upload = form.get("file")
    if not isinstance(upload, UploadFile):
        raise HTTPException(status_code=400, detail="file is required")
    model = str(form.get("model") or _DEFAULT_MODEL_ID)
    if forced_worker_id is not None:
        model = _force_worker_id(model, forced_worker_id)
    worker_id, _, _ = _require_model(model)
    audio_bytes = bytearray()
    try:
        while chunk := await upload.read(1024 * 1024):
            audio_bytes += chunk
            if len(audio_bytes) > _MAX_AUDIO_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"audio file exceeds the {_MAX_AUDIO_BYTES}-byte upload limit",
                )
    finally:
        await upload.close()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="audio file must not be empty")
    total_bytes = len(audio_bytes)

    if await _capable_or_policy(worker_id, "audio_transcription"):
        # Persist the real clip so the worker can be handed a reference to the
        # actual bytes (a shared cloud file URL), not just a byte count.
        record = _store_cloud_bytes(bytes(audio_bytes), _safe_filename(upload.filename or "audio"), purpose="output")
        cloud_url = _cloud_file_url(record["id"])
        result = await _relay_full(
            model,
            f"(audio-transcription) Transcribe this audio clip {upload.filename!r} "
            f"({total_bytes} bytes), available at {cloud_url}. Reply with the transcript text.",
            audio=cloud_url,
            files={"audio_file": record["id"], "url": cloud_url, "bytes": total_bytes},
            kind="audio_transcription",
        )
        return {"text": _reply_content(result), "audio_file": record["id"]}
    return {"text": "[emullm stub: audio transcription is not implemented]"}


@router.post("/v1/audio/transcriptions")
async def audio_transcriptions(request: Request) -> dict[str, Any]:
    return await _audio_transcription_from_request(request)


def _silent_wav(text: str, speed: float) -> bytes:
    """Return a valid mono PCM WAV whose bounded duration tracks input size."""
    sample_rate = 8000
    duration = min(2.0, max(0.2, len(text) / (40.0 * speed)))
    samples = int(sample_rate * duration)
    pcm = b"\x00\x00" * samples
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(pcm),
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        len(pcm),
    ) + pcm


@router.post("/v1/audio/speech")
async def audio_speech(body: AudioSpeechRequest) -> Response:
    """Two-way speech synthesis. If the target worker declared audio_speech
    capability and returns REAL audio (``audio_b64``/``audio_url``), those bytes
    are persisted to the shared cloud files store and returned as the actual
    audio response (with an ``X-EMULLM-File`` reference). If it only replies
    with text, that describes how it would speak (``X-EMULLM-Description``)
    alongside a synthetic silent WAV. If the worker EXPLICITLY declared it won't
    do audio_speech, this stops with 501 instead -- the worker is never asked."""
    worker_id, _, _ = _require_model(body.model)
    headers = {"X-EMULLM-Synthetic": "true"}
    if await _capable_or_policy(worker_id, "audio_speech"):
        result = await _relay_full(
            body.model,
            "(audio-speech) Speak this text. If you can synthesize audio, return it as "
            "base64 in an 'audio_b64' field (with 'mime'); otherwise, in one short "
            f"sentence, describe how you would say it out loud: {body.input}",
            kind="audio_speech",
        )
        audio_b64, audio_url, audio_mime = _reply_audio(result)
        data, mime = _decode_media(audio_b64, audio_url)
        if data is not None:
            # Real audio came back -- persist to cloud files and return the bytes.
            mime = mime or audio_mime or "audio/wav"
            record = _store_cloud_bytes(data, f"speech{_ext_for_mime(mime)}", purpose="output")
            headers = {"X-EMULLM-Synthetic": "false", "X-EMULLM-File": _cloud_file_url(record["id"])}
            return Response(content=data, media_type=mime, headers=headers)
        description = _reply_content(result)
        safe_description = "".join(character for character in description if 32 <= ord(character) < 127)[:512]
        if safe_description:
            headers["X-EMULLM-Description"] = safe_description
    return Response(content=_silent_wav(body.input, body.speed), media_type="audio/wav", headers=headers)


# ---------------------------------------------------------------------------
# /emullm/specific_worker/{worker_id}/v1/* -- the SAME OpenAI-compatible
# surface as /v1/* above, but with worker_id forced from the URL path
# instead of parsed out of the request's "model" field. This is for
# clients that can only configure a fixed baseUrl (no per-request model
# override): point one at
# "http://<host>/emullm/specific_worker/alice/v1" and every request it
# sends -- regardless of what "model" it fills in -- is pinned to alice's
# worker_id (only the persona SUFFIX from its "model" field is kept).
# ---------------------------------------------------------------------------
def _force_worker_id(model: str, worker_id: str) -> str:
    _, suffix = _split_model_id(model)
    return f"{worker_id}/{suffix}"


@router.get("/emullm/specific_worker/{worker_id}/v1/models")
def specific_worker_list_models(worker_id: str) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [_model_entry(worker_id, suffix, persona) for suffix, persona in _models_for(worker_id).items()]
    }


@router.get("/emullm/specific_worker/{worker_id}/v1/models/{model_id:path}")
def specific_worker_get_model(worker_id: str, model_id: str) -> dict[str, Any]:
    _, suffix = _split_model_id(model_id)
    persona = _models_for(worker_id).get(suffix)
    if persona is None:
        raise HTTPException(status_code=404, detail=f"unknown model '{model_id}' for worker '{worker_id}'")
    return _model_entry(worker_id, suffix, persona)


@router.post("/emullm/specific_worker/{worker_id}/v1/chat/completions")
async def specific_worker_chat_completions(worker_id: str, body: ChatRequest) -> dict[str, Any]:
    body.model = _force_worker_id(body.model, worker_id)
    return await chat_completions(body)


@router.post("/emullm/specific_worker/{worker_id}/v1/completions")
async def specific_worker_completions(worker_id: str, body: CompletionRequest) -> dict[str, Any]:
    body.model = _force_worker_id(body.model, worker_id)
    return await completions(body)


@router.post("/emullm/specific_worker/{worker_id}/v1/responses")
async def specific_worker_responses(worker_id: str, body: ResponsesRequest) -> dict[str, Any]:
    body.model = _force_worker_id(body.model, worker_id)
    return await responses(body)


@router.post("/emullm/specific_worker/{worker_id}/v1/embeddings")
async def specific_worker_embeddings(worker_id: str, body: EmbeddingsRequest) -> dict[str, Any]:
    body.model = _force_worker_id(body.model, worker_id)
    return await embeddings(body)


@router.post("/emullm/specific_worker/{worker_id}/v1/moderations")
async def specific_worker_moderations(worker_id: str, body: ModerationsRequest) -> dict[str, Any]:
    body.model = _force_worker_id(body.model, worker_id)
    return await moderations(body)


@router.post("/emullm/specific_worker/{worker_id}/v1/images/generations")
async def specific_worker_images_generations(worker_id: str, body: ImagesRequest) -> dict[str, Any]:
    body.model = _force_worker_id(body.model, worker_id)
    return await images_generations(body)


@router.post("/emullm/specific_worker/{worker_id}/v1/audio/transcriptions")
async def specific_worker_audio_transcriptions(worker_id: str, request: Request) -> dict[str, Any]:
    return await _audio_transcription_from_request(request, worker_id)


@router.post("/emullm/specific_worker/{worker_id}/v1/audio/speech")
async def specific_worker_audio_speech(worker_id: str, body: AudioSpeechRequest) -> Response:
    body.model = _force_worker_id(body.model, worker_id)
    return await audio_speech(body)


# ---------------------------------------------------------------------------
# Filesystem-backed emulation for files and the heavier OpenAI platform
# resource surfaces (assistants/threads/fine-tuning jobs). Records are
# parked as individual JSON files under runtime/<kind>/ so they
# survive a server restart instead of vanishing like a pure in-memory
# dict would -- matching this repo's "no mocks, real filesystem" rule.
# Uploaded file bytes are real and retrievable. Assistant/thread resource
# lifecycle is durable, but there is no hosted agent runtime. Fine-tune input
# is validated and jobs are recorded, but no training is performed.
# ---------------------------------------------------------------------------
_RUNTIME_DIR = Path(
    os.environ.get("EMULLM_RUNTIME_DIR") or (Path(__file__).resolve().parent.parent / "runtime")
)
_MAX_FILE_BYTES = max(1, int(os.environ.get("EMULLM_MAX_FILE_BYTES", str(512 * 1024 * 1024))))
_ALLOWED_FILE_PURPOSES = {"assistants", "batch", "fine-tune", "vision", "user_data", "evals", "output"}


def _new_resource_id(prefix: str) -> str:
    """Return an opaque ID whose lexical order also preserves creation order."""
    return f"{prefix}-{time.time_ns():x}{secrets.token_hex(4)}"


def _safe_filename(value: Any) -> str:
    filename = Path(str(value or "upload")).name or "upload"
    filename = "".join(character for character in filename if character >= " " and character != "\x7f")[:255]
    return filename or "upload"


def _file_purpose(value: Any) -> str:
    purpose = str(value or "")
    if purpose not in _ALLOWED_FILE_PURPOSES:
        raise HTTPException(status_code=400, detail=f"purpose must be one of {sorted(_ALLOWED_FILE_PURPOSES)}")
    return purpose


class _JsonRecordStore:
    """One JSON file per record, under _RUNTIME_DIR/<kind>/<id>.json."""

    def __init__(self, kind: str) -> None:
        self._dir = _RUNTIME_DIR / kind
        self._lock = threading.RLock()

    def _path(self, record_id: str) -> Path:
        if (
            not isinstance(record_id, str)
            or not record_id
            or len(record_id) > 255
            or any(not (character.isascii() and (character.isalnum() or character in "-_.")) for character in record_id)
            or record_id in {".", ".."}
        ):
            raise ValueError("record ID contains unsafe filename characters")
        return self._dir / f"{record_id}.json"

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self._dir.exists():
                return []
            records: list[dict[str, Any]] = []
            for path in sorted(self._dir.glob("*.json")):
                try:
                    records.append(json.loads(path.read_text(encoding="utf-8-sig")))
                except (OSError, json.JSONDecodeError):
                    continue
            return records

    def save(self, record: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._dir.mkdir(parents=True, exist_ok=True)
            path = self._path(record["id"])
            temporary = self._dir / f".{record['id']}.{uuid.uuid4().hex}.json.tmp"
            try:
                temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return record

    def get(self, record_id: str) -> dict[str, Any] | None:
        with self._lock:
            try:
                path = self._path(record_id)
            except ValueError:
                return None
            if not path.is_file():
                return None
            try:
                return json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                return None

    def delete(self, record_id: str) -> bool:
        with self._lock:
            try:
                path = self._path(record_id)
            except ValueError:
                return False
            if not path.exists():
                return False
            path.unlink()
            return True

    def clear(self) -> int:
        with self._lock:
            if not self._dir.exists():
                return 0
            removed = 0
            for path in self._dir.glob("*.json"):
                path.unlink()
                removed += 1
            return removed


class _FileRecordStore(_JsonRecordStore):
    """File metadata plus a sibling binary blob for each `/v1/files` ID."""

    def content_path(self, record_id: str) -> Path:
        self._path(record_id)  # Validate before constructing the sibling blob path.
        return self._dir / f"{record_id}.content"

    def temporary_content_path(self, record_id: str) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        return self._dir / f".{record_id}.{uuid.uuid4().hex}.content.tmp"

    def commit(self, record: dict[str, Any], temporary_content: Path) -> dict[str, Any]:
        """Atomically publish a completed blob, then its atomic metadata."""
        with self._lock:
            content_path = self.content_path(record["id"])
            temporary_content.replace(content_path)
            try:
                return self.save(record)
            except Exception:
                content_path.unlink(missing_ok=True)
                raise

    def active_records(self) -> list[dict[str, Any]]:
        now = int(time.time())
        active: list[dict[str, Any]] = []
        for record in self.list():
            expires_at = record.get("expires_at")
            if isinstance(expires_at, int) and expires_at <= now:
                self.delete(str(record.get("id", "")))
            else:
                active.append(record)
        return active

    def delete(self, record_id: str) -> bool:
        with self._lock:
            deleted = super().delete(record_id)
            try:
                content_path = self.content_path(record_id)
            except ValueError:
                return deleted
            if content_path.is_file():
                content_path.unlink()
                deleted = True
            return deleted

    def clear(self) -> int:
        with self._lock:
            removed = super().clear()
            if self._dir.exists():
                for pattern in ("*.content", ".*.content.tmp"):
                    for path in self._dir.glob(pattern):
                        path.unlink()
            return removed


_files_store = _FileRecordStore("files")
_assistants_store = _JsonRecordStore("assistants")
_threads_store = _JsonRecordStore("threads")
_fine_tuning_jobs_store = _JsonRecordStore("fine_tuning_jobs")
_fine_tuning_events_store = _JsonRecordStore("fine_tuning_events")
_tokens_store = _JsonRecordStore("tokens")


# ---------------------------------------------------------------------------
# Optional token-issuance website -- GET /emullm/tokens/new is a minimal
# HTML page that can mint/store a token or SSH public key for deployments
# that want credentials later. IMPORTANT: this repo does NOT gate any
# client (/v1/*) or worker route on these records. OpenAI-compatible
# clients never need an API key; workers never need a token. is_valid_token
# / is_registered_public_key exist only so callers could opt into checks
# later -- nothing in this module calls them for access control.
# ---------------------------------------------------------------------------
class TokenRequest(BaseModel):
    email: str
    token: str | None = None  # bring-your-own; a random one is generated if omitted
    public_key: str | None = None  # SSH-style public key to register instead/as well


def _issue_token(email: str, token: str | None, public_key: str | None) -> dict[str, Any]:
    token = token or secrets.token_urlsafe(32)
    record: dict[str, Any] = {"id": token, "email": email, "created_at": int(time.time())}
    if public_key:
        record["public_key"] = public_key
    return _tokens_store.save(record)


def is_valid_token(token: str) -> bool:
    """True if `token` is one this server actually issued (and hasn't
    been revoked/deleted). Lookup helper only -- no route in this module
    requires a token. Client /v1/* access stays keyless."""
    try:
        return _tokens_store._path(token).is_file()
    except ValueError:
        return False


def is_registered_public_key(public_key: str) -> bool:
    """True if `public_key` was registered on some token record (no
    signature is verified -- this only checks it was accepted before)."""
    needle = public_key.strip()
    return any(record.get("public_key") == needle for record in _tokens_store.list())


@router.post("/emullm/tokens")
def create_token(body: TokenRequest) -> dict[str, Any]:
    if not body.email or not body.email.strip():
        raise HTTPException(status_code=400, detail="email is required")
    try:
        return _issue_token(
            body.email.strip(),
            body.token.strip() if body.token else None,
            body.public_key.strip() if body.public_key else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="token contains unsafe characters") from exc


@router.get("/emullm/tokens/new", response_class=HTMLResponse)
def tokens_new_page() -> str:
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>emullm -- optional token (not required)</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 32rem; margin: 3rem auto; padding: 0 1rem; }
  label { display: block; margin-top: 1rem; font-size: 0.9rem; }
  input, textarea { width: 100%; box-sizing: border-box; font-size: 1rem; padding: 0.4rem; margin-top: 0.25rem; font-family: inherit; }
  textarea { font-family: monospace; font-size: 0.85rem; min-height: 4rem; }
  button { font-size: 1rem; padding: 0.5rem 1rem; cursor: pointer; margin-top: 1.25rem; }
  code { background: #f0f0f0; padding: 0.6rem; display: block; word-break: break-all; margin-top: 1rem; border-radius: 4px; }
  .hint { color: #666; font-size: 0.9rem; }
  .error { color: #b00020; }
  .banner { background: #eef6ff; border: 1px solid #b6d4fe; padding: 0.75rem 1rem; border-radius: 6px; }
</style>
</head>
<body>
<h1>Optional token</h1>
<p class="banner"><strong>You do not need a token to use this server.</strong>
OpenAI-compatible clients call <code>/v1/*</code> with no API key, and
workers connect without credentials. This page only mints optional
bookkeeping records; they are not checked for access.</p>
<p>If you still want one recorded: enter your email, then pick one -- leave
the token field blank to generate one, paste a token you already plan to
use, or paste an SSH-style public key (no signature challenge -- it's
just accepted and remembered).</p>
<label for="email">Email</label>
<input id="email" type="email" placeholder="you@example.com" required>
<label for="token">Token (optional -- leave blank to generate one)</label>
<input id="token" type="text" placeholder="leave blank to generate">
<label for="publicKey">SSH public key (optional)</label>
<textarea id="publicKey" placeholder="ssh-ed25519 AAAA... you@host"></textarea>
<button id="go">Get token</button>
<code id="out" style="display:none"></code>
<p class="hint" id="hint" style="display:none">
  Copy this now -- it isn't shown again, but it stays valid until revoked.
</p>
<p class="error" id="error" style="display:none"></p>
<script>
document.getElementById('go').addEventListener('click', async () => {
  const email = document.getElementById('email').value;
  const token = document.getElementById('token').value;
  const publicKey = document.getElementById('publicKey').value;
  const errorEl = document.getElementById('error');
  errorEl.style.display = 'none';
  const response = await fetch('/emullm/tokens', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email: email, token: token || null, public_key: publicKey || null }),
  });
  const data = await response.json();
  if (!response.ok) {
    errorEl.textContent = data.detail || 'something went wrong';
    errorEl.style.display = 'block';
    return;
  }
  const out = document.getElementById('out');
  out.textContent = data.id;
  out.style.display = 'block';
  document.getElementById('hint').style.display = 'block';
});
</script>
</body>
</html>"""


@router.get("/v1/files")
def list_files(
    purpose: str | None = None,
    after: str | None = None,
    limit: int = 10000,
    order: str = "desc",
) -> dict[str, Any]:
    if not 1 <= limit <= 10000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 10000")
    if order not in {"asc", "desc"}:
        raise HTTPException(status_code=400, detail="order must be 'asc' or 'desc'")

    records = _files_store.active_records()
    if purpose is not None:
        records = [record for record in records if record.get("purpose") == purpose]
    records.sort(key=lambda record: (int(record.get("created_at", 0)), str(record.get("id", ""))))
    if order == "desc":
        records.reverse()
    if after is not None:
        cursor_index = next((index for index, record in enumerate(records) if record.get("id") == after), None)
        if cursor_index is None:
            raise HTTPException(status_code=400, detail=f"Invalid pagination cursor: {after}")
        records = records[cursor_index + 1 :]

    page = records[:limit]
    return {
        "object": "list",
        "data": page,
        "first_id": page[0]["id"] if page else None,
        "last_id": page[-1]["id"] if page else None,
        "has_more": len(records) > limit,
    }


def _expires_at_from_form(form: Any, purpose: str, created_at: int) -> int | None:
    anchor = form.get("expires_after[anchor]")
    seconds_value = form.get("expires_after[seconds]")
    raw_policy = form.get("expires_after")
    if raw_policy:
        try:
            policy = json.loads(str(raw_policy))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="expires_after must be valid JSON") from exc
        if not isinstance(policy, dict):
            raise HTTPException(status_code=400, detail="expires_after must be an object")
        anchor = policy.get("anchor")
        seconds_value = policy.get("seconds")

    if anchor is None and seconds_value is None:
        return created_at + 30 * 24 * 60 * 60 if purpose == "batch" else None
    if anchor != "created_at":
        raise HTTPException(status_code=400, detail="expires_after.anchor must be 'created_at'")
    try:
        seconds = int(seconds_value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="expires_after.seconds must be a positive integer") from exc
    if seconds <= 0:
        raise HTTPException(status_code=400, detail="expires_after.seconds must be a positive integer")
    return created_at + seconds


@router.post("/v1/files")
async def create_file(request: Request) -> dict[str, Any]:
    file_id = _new_resource_id("file")
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        purpose = _file_purpose(form.get("purpose"))
        if not isinstance(upload, UploadFile):
            raise HTTPException(status_code=400, detail="file is required")
        filename = _safe_filename(upload.filename)
        temporary_content = _files_store.temporary_content_path(file_id)
        bytes_written = 0
        try:
            with temporary_content.open("wb") as destination:
                while chunk := await upload.read(1024 * 1024):
                    bytes_written += len(chunk)
                    if bytes_written > _MAX_FILE_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"file exceeds the {_MAX_FILE_BYTES}-byte upload limit",
                        )
                    destination.write(chunk)
        except BaseException:
            temporary_content.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()

        created_at = int(time.time())
        record = {
            "id": file_id,
            "object": "file",
            "bytes": bytes_written,
            "created_at": created_at,
            "filename": filename,
            "purpose": purpose,
            "status": "processed",
        }
        expires_at = _expires_at_from_form(form, purpose, created_at)
        if expires_at is not None:
            record["expires_at"] = expires_at
        return _files_store.commit(record, temporary_content)
    elif content_type.startswith("application/json"):
        # Preserve the old lightweight JSON probe behavior while giving the
        # resulting record a real, empty content blob.
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object")
        reserved = {"id", "object", "created_at", "bytes", "status", "expires_at"}
        extra = {key: value for key, value in body.items() if key not in reserved}
        record = {
            "id": file_id,
            "object": "file",
            "bytes": 0,
            "created_at": int(time.time()),
            "filename": _safe_filename(extra.pop("filename", f"{file_id}.bin")),
            "purpose": _file_purpose(extra.pop("purpose", "assistants")),
            "status": "processed",
            **extra,
        }
        temporary_content = _files_store.temporary_content_path(file_id)
        temporary_content.write_bytes(b"")
        return _files_store.commit(record, temporary_content)
    else:
        raise HTTPException(status_code=415, detail="use multipart/form-data with file and purpose fields")


def _get_file_or_404(file_id: str) -> dict[str, Any]:
    record = _files_store.get(file_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No such file: {file_id}")
    expires_at = record.get("expires_at")
    if isinstance(expires_at, int) and expires_at <= int(time.time()):
        _files_store.delete(file_id)
        raise HTTPException(status_code=404, detail=f"No such file: {file_id}")
    return record


@router.get("/v1/files/{file_id}")
def retrieve_file(file_id: str) -> dict[str, Any]:
    return _get_file_or_404(file_id)


@router.get("/v1/files/{file_id}/content")
def retrieve_file_content(file_id: str) -> FileResponse:
    record = _get_file_or_404(file_id)
    content_path = _files_store.content_path(file_id)
    if not content_path.is_file():
        raise HTTPException(status_code=404, detail=f"No content stored for file: {file_id}")
    media_type = mimetypes.guess_type(str(record.get("filename", "")))[0] or "application/octet-stream"
    return FileResponse(
        path=content_path,
        media_type=media_type,
        filename=str(record.get("filename") or file_id),
    )


@router.delete("/v1/files/{file_id}")
def delete_file(file_id: str) -> dict[str, Any]:
    _get_file_or_404(file_id)
    _files_store.delete(file_id)
    return {"id": file_id, "object": "file", "deleted": True}


# ---------------------------------------------------------------------------
# Shared cloud files -- one durable blob store the relay AND connected workers
# both use. Media flowing two-way (a generated image, synthesized audio, an
# uploaded clip to transcribe, a fine-tune result) is persisted here once and
# then passed around by a stable URL (/emullm/cloud/files/<id>) instead of
# shipping base64 everywhere. Backed by the same _files_store as /v1/files.
# ---------------------------------------------------------------------------
_CLOUD_FILES_PREFIX = "/emullm/cloud/files"
_MIME_EXT = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp",
             "image/gif": ".gif", "audio/wav": ".wav", "audio/mpeg": ".mp3",
             "audio/mp3": ".mp3", "audio/ogg": ".ogg", "application/json": ".json"}


def _ext_for_mime(mime: str | None) -> str:
    return _MIME_EXT.get((mime or "").split(";", 1)[0].strip(), ".bin")


def _store_cloud_bytes(data: bytes, filename: str, purpose: str = "output") -> dict[str, Any]:
    """Persist bytes into the shared cloud files store; return the file record.
    The blob is then retrievable at :func:`_cloud_file_url` by relay or worker."""
    file_id = _new_resource_id("file")
    temporary_content = _files_store.temporary_content_path(file_id)
    temporary_content.write_bytes(data)
    record = {
        "id": file_id,
        "object": "file",
        "bytes": len(data),
        "created_at": int(time.time()),
        "filename": _safe_filename(filename),
        "purpose": purpose,
        "status": "processed",
    }
    return _files_store.commit(record, temporary_content)


def _cloud_file_url(file_id: str) -> str:
    return f"{_CLOUD_FILES_PREFIX}/{file_id}"


def _decode_media(b64: str | None, url: str | None) -> tuple[bytes | None, str | None]:
    """Bytes + mime from a base64 string or a ``data:`` URL, else (None, None)."""
    mime: str | None = None
    if not b64 and isinstance(url, str) and url.startswith("data:") and "," in url:
        head, b64 = url.split(",", 1)
        mime = head[5:].split(";", 1)[0] or None
    if b64:
        try:
            return base64.b64decode(b64), mime
        except (binascii.Error, ValueError):
            return None, None
    return None, None


def _serve_cloud_file(file_id: str) -> FileResponse:
    record = _get_file_or_404(file_id)
    content_path = _files_store.content_path(file_id)
    if not content_path.is_file():
        raise HTTPException(status_code=404, detail=f"No content stored for file: {file_id}")
    media_type = mimetypes.guess_type(str(record.get("filename", "")))[0] or "application/octet-stream"
    return FileResponse(
        path=content_path,
        media_type=media_type,
        filename=str(record.get("filename") or file_id),
    )


@router.get("/emullm/cloud/files/{file_id}")
def cloud_file(file_id: str) -> FileResponse:
    return _serve_cloud_file(file_id)


@router.get("/emullm/cloud/files/{file_id}")
def cloud_file_alias(file_id: str) -> FileResponse:
    # tolerate the alternate spelling the operator might use
    return _serve_cloud_file(file_id)


def _paginate_records(
    records: list[dict[str, Any]],
    *,
    after: str | None,
    before: str | None,
    limit: int,
    order: str,
    maximum_limit: int = 100,
) -> dict[str, Any]:
    """Apply stable OpenAI-style cursor pagination to resource records."""
    if not 1 <= limit <= maximum_limit:
        raise HTTPException(status_code=400, detail=f"limit must be between 1 and {maximum_limit}")
    if order not in {"asc", "desc"}:
        raise HTTPException(status_code=400, detail="order must be 'asc' or 'desc'")
    if after is not None and before is not None:
        raise HTTPException(status_code=400, detail="after and before cannot be used together")

    records.sort(key=lambda record: (int(record.get("created_at", 0)), str(record.get("id", ""))))
    if order == "desc":
        records.reverse()

    cursor = after if after is not None else before
    if cursor is not None:
        cursor_index = next((index for index, record in enumerate(records) if record.get("id") == cursor), None)
        if cursor_index is None:
            raise HTTPException(status_code=400, detail=f"Invalid pagination cursor: {cursor}")
        records = records[cursor_index + 1 :] if after is not None else records[:cursor_index]

    page = records[:limit]
    return {
        "object": "list",
        "data": page,
        "first_id": page[0]["id"] if page else None,
        "last_id": page[-1]["id"] if page else None,
        "has_more": len(records) > limit,
    }


def _resource_or_404(store: _JsonRecordStore, resource_id: str, resource_name: str) -> dict[str, Any]:
    record = store.get(resource_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No such {resource_name}: {resource_id}")
    return record


def _metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="metadata must be an object")
    if len(value) > 16:
        raise HTTPException(status_code=400, detail="metadata may contain at most 16 keys")
    return value


@router.get("/v1/assistants")
def list_assistants(
    after: str | None = None,
    before: str | None = None,
    limit: int = 20,
    order: str = "desc",
) -> dict[str, Any]:
    return _paginate_records(
        _assistants_store.list(), after=after, before=before, limit=limit, order=order
    )


@router.post("/v1/assistants")
def create_assistant(body: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(body or {})
    model = payload.get("model", _DEFAULT_MODEL_ID)
    if not isinstance(model, str) or not model.strip():
        raise HTTPException(status_code=400, detail="model must be a non-empty string")
    payload["metadata"] = _metadata(payload.get("metadata"))
    record = {
        **payload,
        "id": _new_resource_id("asst"),
        "object": "assistant",
        "created_at": int(time.time()),
        "model": model,
        "name": payload.get("name"),
        "description": payload.get("description"),
        "instructions": payload.get("instructions"),
        "tools": payload.get("tools", []),
        "tool_resources": payload.get("tool_resources", {}),
        "temperature": payload.get("temperature", 1.0),
        "top_p": payload.get("top_p", 1.0),
        "response_format": payload.get("response_format", "auto"),
    }
    return _assistants_store.save(record)


@router.get("/v1/assistants/{assistant_id}")
def retrieve_assistant(assistant_id: str) -> dict[str, Any]:
    return _resource_or_404(_assistants_store, assistant_id, "assistant")


@router.post("/v1/assistants/{assistant_id}")
def modify_assistant(assistant_id: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    record = _resource_or_404(_assistants_store, assistant_id, "assistant")
    updates = dict(body or {})
    if "metadata" in updates:
        updates["metadata"] = _metadata(updates["metadata"])
    for protected in ("id", "object", "created_at"):
        updates.pop(protected, None)
    record.update(updates)
    return _assistants_store.save(record)


@router.delete("/v1/assistants/{assistant_id}")
def delete_assistant(assistant_id: str) -> dict[str, Any]:
    _resource_or_404(_assistants_store, assistant_id, "assistant")
    _assistants_store.delete(assistant_id)
    return {"id": assistant_id, "object": "assistant.deleted", "deleted": True}


@router.get("/v1/threads")
def list_threads(
    after: str | None = None,
    before: str | None = None,
    limit: int = 20,
    order: str = "desc",
) -> dict[str, Any]:
    return _paginate_records(_threads_store.list(), after=after, before=before, limit=limit, order=order)


@router.post("/v1/threads")
def create_thread(body: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(body or {})
    payload["metadata"] = _metadata(payload.get("metadata"))
    record = {
        **payload,
        "id": _new_resource_id("thread"),
        "object": "thread",
        "created_at": int(time.time()),
        "tool_resources": payload.get("tool_resources", {}),
    }
    return _threads_store.save(record)


@router.get("/v1/threads/{thread_id}")
def retrieve_thread(thread_id: str) -> dict[str, Any]:
    return _resource_or_404(_threads_store, thread_id, "thread")


@router.post("/v1/threads/{thread_id}")
def modify_thread(thread_id: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    record = _resource_or_404(_threads_store, thread_id, "thread")
    updates = dict(body or {})
    if "metadata" in updates:
        updates["metadata"] = _metadata(updates["metadata"])
    for protected in ("id", "object", "created_at"):
        updates.pop(protected, None)
    record.update(updates)
    return _threads_store.save(record)


@router.delete("/v1/threads/{thread_id}")
def delete_thread(thread_id: str) -> dict[str, Any]:
    _resource_or_404(_threads_store, thread_id, "thread")
    _threads_store.delete(thread_id)
    return {"id": thread_id, "object": "thread.deleted", "deleted": True}


@router.get("/v1/fine_tuning/jobs")
def list_fine_tuning_jobs(after: str | None = None, limit: int = 20) -> dict[str, Any]:
    return _paginate_records(
        _fine_tuning_jobs_store.list(), after=after, before=None, limit=limit, order="desc"
    )


def _validate_fine_tuning_file(file_id: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(file_id, str) or not file_id:
        raise HTTPException(status_code=400, detail=f"{field_name} is required")
    record = _get_file_or_404(file_id)
    if record.get("purpose") != "fine-tune":
        raise HTTPException(status_code=400, detail=f"{field_name} must have purpose 'fine-tune'")
    content_path = _files_store.content_path(file_id)
    try:
        saw_record = False
        with content_path.open("r", encoding="utf-8-sig") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                saw_record = True
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("each line must contain a JSON object")
        if not saw_record:
            raise ValueError("the file contains no JSONL records")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} is not valid UTF-8 JSONL: {exc}") from exc
    return record


@router.post("/v1/fine_tuning/jobs")
async def create_fine_tuning_job(body: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(body or {})
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise HTTPException(status_code=400, detail="model is required")
    training_file = payload.get("training_file")
    _validate_fine_tuning_file(training_file, "training_file")
    validation_file = payload.get("validation_file")
    if validation_file is not None:
        _validate_fine_tuning_file(validation_file, "validation_file")

    worker_id, _ = _split_model_id(model)
    volunteers = await _capable_or_policy(worker_id, "fine_tuning")

    job_id = _new_resource_id("ftjob")
    created_at = int(time.time())
    hyperparameters = payload.get(
        "hyperparameters",
        {"n_epochs": "auto", "batch_size": "auto", "learning_rate_multiplier": "auto"},
    )

    if volunteers:
        # A worker volunteered to "train". Route it a reference to the real
        # training data (a shared cloud file URL) and let it acknowledge; then
        # publish a result file to the cloud store and a fine_tuned_model id.
        ack = _reply_content(
            await _relay_full(
                model,
                f"(fine-tuning) A fine-tune job for base model {model!r} using training file "
                f"{training_file} (at {_cloud_file_url(str(training_file))}). Acknowledge and, in "
                "one short line, summarize how you'd train it.",
                files={
                    "training_file": training_file,
                    "validation_file": validation_file,
                    "url": _cloud_file_url(str(training_file)),
                },
                kind="fine_tuning",
            )
        )
        fine_tuned_model = f"ft:{model}:emul-{job_id[-8:]}"
        manifest = json.dumps(
            {
                "job_id": job_id,
                "base_model": model,
                "fine_tuned_model": fine_tuned_model,
                "training_file": training_file,
                "validation_file": validation_file,
                "worker": worker_id,
                "note": ack or "trained by worker",
            },
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")
        result_file = _store_cloud_bytes(manifest, f"{job_id}-result.json", purpose="output")
        finished_at = int(time.time())
        record = {
            **payload,
            "id": job_id,
            "object": "fine_tuning.job",
            "created_at": created_at,
            "finished_at": finished_at,
            "model": model,
            "training_file": training_file,
            "validation_file": validation_file,
            "status": "succeeded",
            "fine_tuned_model": fine_tuned_model,
            "estimated_finish": finished_at,
            "error": None,
            "result_files": [result_file["id"]],
            "trained_tokens": max(1, len(manifest)),
            "hyperparameters": hyperparameters,
            "integrations": payload.get("integrations", []),
        }
        saved = _fine_tuning_jobs_store.save(record)
        _fine_tuning_events_store.save(
            {
                "id": _new_resource_id("ftevent"),
                "object": "fine_tuning.job.event",
                "created_at": finished_at,
                "job_id": job_id,
                "level": "info",
                "message": f"fine-tune completed by worker '{worker_id}': {fine_tuned_model}",
                "data": {"fine_tuned_model": fine_tuned_model, "result_file": result_file["id"]},
            }
        )
        return saved

    record = {
        **payload,
        "id": job_id,
        "object": "fine_tuning.job",
        "created_at": created_at,
        "finished_at": created_at,
        "model": model,
        "training_file": training_file,
        "validation_file": validation_file,
        "status": "failed",
        "fine_tuned_model": None,
        "estimated_finish": None,
        "error": {
            "code": "training_not_available",
            "message": "emullm validates and persists fine-tuning jobs, but no worker volunteered to train",
            "param": None,
        },
        "result_files": [],
        "trained_tokens": None,
        "hyperparameters": hyperparameters,
        "integrations": payload.get("integrations", []),
    }
    saved = _fine_tuning_jobs_store.save(record)
    _fine_tuning_events_store.save(
        {
            "id": _new_resource_id("ftevent"),
            "object": "fine_tuning.job.event",
            "created_at": created_at,
            "job_id": job_id,
            "level": "error",
            "message": record["error"]["message"],
            "data": {"code": record["error"]["code"]},
        }
    )
    return saved


@router.get("/v1/fine_tuning/jobs/{job_id}")
def retrieve_fine_tuning_job(job_id: str) -> dict[str, Any]:
    return _resource_or_404(_fine_tuning_jobs_store, job_id, "fine-tuning job")


@router.post("/v1/fine_tuning/jobs/{job_id}/cancel")
def cancel_fine_tuning_job(job_id: str) -> dict[str, Any]:
    record = retrieve_fine_tuning_job(job_id)
    if record.get("status") in {"succeeded", "failed", "cancelled"}:
        raise HTTPException(status_code=409, detail=f"fine-tuning job is already {record['status']}")
    record["status"] = "cancelled"
    record["finished_at"] = int(time.time())
    return _fine_tuning_jobs_store.save(record)


@router.get("/v1/fine_tuning/jobs/{job_id}/events")
def list_fine_tuning_events(
    job_id: str,
    after: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    retrieve_fine_tuning_job(job_id)
    records = [record for record in _fine_tuning_events_store.list() if record.get("job_id") == job_id]
    return _paginate_records(records, after=after, before=None, limit=limit, order="desc")


@router.get("/v1/fine_tuning/jobs/{job_id}/checkpoints")
def list_fine_tuning_checkpoints(job_id: str, after: str | None = None, limit: int = 20) -> dict[str, Any]:
    retrieve_fine_tuning_job(job_id)
    if after is not None:
        raise HTTPException(status_code=400, detail=f"Invalid pagination cursor: {after}")
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
    return {"object": "list", "data": [], "first_id": None, "last_id": None, "has_more": False}


# ---------------------------------------------------------------------------
# Admin/test-controller surface -- NOT part of the OpenAI-compatible API.
# Lets a test (or an operator) drive this running server over plain HTTP:
# repoint where the durable resources persist to, delete individual records, wipe
# them all, or inspect current relay/worker state -- without needing
# direct access to this module's Python objects. Namespaced under
# /admin/emullm so it can never collide with a real /v1/... path. Also
# reachable as /emullm/admin/... (an alias, registered on the exact
# same handlers below) -- pick whichever reads better for a given caller,
# both act identically.
# ---------------------------------------------------------------------------
_KIND_STORES: dict[str, "_JsonRecordStore"] = {}  # populated once the stores below are constructed


class SetRuntimeDirRequest(BaseModel):
    path: str


@router.get("/admin/emullm/state")
@router.get("/emullm/admin/state")
def admin_state() -> dict[str, Any]:
    return {
        "mode": ",".join(_current_modes()),
        "modes": _current_modes(),
        "started_at": _SERVER_STARTED_AT,
        "uptime_seconds": round(time.time() - _SERVER_STARTED_AT, 1),
        "runtime_dir": str(_RUNTIME_DIR),
        "connected_worker_ids": sorted(_connected_workers.keys()),
        "worker_models": {worker_id: sorted(models.keys()) for worker_id, models in _worker_models.items()},
        "worker_capabilities": dict(_worker_capabilities),
        "worker_roles": {
            worker_id: _worker_roles.get(worker_id, _DEFAULT_WORKER_ROLE)
            for worker_id in sorted(set(_connected_workers) | set(_worker_roles))
        },
        "worker_usage": {
            worker_id: {
                "total_requests": usage["total"],
                "requests_in_window": len(usage["recent"]),
                "window_seconds": _USAGE_WINDOW_SECONDS,
                "max_per_window": _USAGE_MAX_PER_WINDOW,
                "last_used_at": usage.get("last_used_at"),
            }
            for worker_id, usage in _worker_usage.items()
        },
        "pending_request_ids": sorted(_pending.keys()),
        "record_counts": {kind: len(store.list()) for kind, store in _KIND_STORES.items()},
        "managed_workers": _sup.get_supervisor().status() if _sup.get_supervisor() else [],
        "backend": (
            {"name": _b.get("name"), "base_url": _b.get("base_url"), "model": _b.get("model")}
            if (_b := _select_backend())
            else None
        ),
        "backends": [
            {"name": entry.get("name"), "base_url": entry.get("base_url"), "model": entry.get("model")}
            for entry in _all_backends()
        ],
        "server_description": _server_description,
        "service_behaviors": {wid: dict(m) for wid, m in _worker_service_behavior.items()},
        "service_fallback": dict(_service_fallback),
        "observers": {
            wid: ("all" if scope is True else sorted(scope)) for wid, scope in _observers.items()
        },
        "agent_descriptions": dict(_agent_descriptions),
        "service_descriptions": dict(_service_descriptions),
        "advertised_models": (_cat := advertised_catalog())["models"],
        "advertised_model": _cat["model"],
        "model_agents": model_failover_map(),
        "model_routes": dict(_model_routes),
    }


# Human-readable status dashboards over the same data as
# /emullm/admin/state. Two views share one template (a DETAIL flag):
#   /emullm/status         -- concise operator overview
#   /emullm/status/detail  -- everything (model instructions, usage
#                               timing, pending request ids, record counts)
# Both are also reachable under the /admin/emullm/... alias.
_STATUS_PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>emullm -- status</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; max-width: 60rem; margin: 2rem auto; padding: 0 1rem; }
  h1 { margin-bottom: 0.25rem; }
  .sub { color: #888; font-size: 0.9rem; margin-top: 0; }
  .cards { display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 1rem 0; }
  .card { border: 1px solid #8883; border-radius: 8px; padding: 0.6rem 0.9rem; min-width: 8rem; }
  .card .k { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.03em; }
  .card .v { font-size: 1.4rem; font-weight: 600; }
  .mode { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 999px; background: #4a90d922; border: 1px solid #4a90d977; font-weight: 600; }
  table { border-collapse: collapse; width: 100%; margin-top: 0.5rem; }
  th, td { text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid #8882; font-size: 0.9rem; vertical-align: top; }
  th { color: #888; font-weight: 600; }
  .pill { display: inline-block; padding: 0.05rem 0.4rem; margin: 0 0.15rem 0.15rem 0; border-radius: 999px; background: #8882; font-size: 0.8rem; }
  .role { display: inline-block; padding: 0.05rem 0.5rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600; background: #8882; }
  .role-trusted { background: #2ecc7133; border: 1px solid #2ecc7188; }
  .role-training { background: #f39c1233; border: 1px solid #f39c1288; }
  .muted { color: #888; }
  .dot { height: 0.6rem; width: 0.6rem; border-radius: 50%; display: inline-block; margin-right: 0.35rem; }
  .on { background: #2ecc71; } .off { background: #bbb; }
  code { background: #8881; padding: 0.05rem 0.3rem; border-radius: 4px; }
  footer { margin-top: 1.5rem; color: #888; font-size: 0.8rem; }
</style>
</head>
<body>
<h1>emullm status <span id="view-tag" class="muted" style="font-size:1rem"></span></h1>
<p class="sub">mode <span id="mode" class="mode">-</span> &middot;
  auto-refreshing every 3s &middot; <span id="updated" class="muted">-</span> &middot;
  <a id="view-toggle" href="/emullm/status/detail">detailed view</a></p>

<div class="cards">
  <div class="card"><div class="k">Workers</div><div class="v" id="worker-count">-</div></div>
  <div class="card"><div class="k">Pending requests</div><div class="v" id="pending-count">-</div></div>
  <div class="card"><div class="k">Uptime</div><div class="v" id="uptime">-</div></div>
</div>

<h2>Connected workers</h2>
<table>
  <thead><tr><th>Worker</th><th>Role</th><th>Models</th><th>Capabilities</th><th>Usage (window / total)</th></tr></thead>
  <tbody id="workers"><tr><td colspan="5" class="muted">loading...</td></tr></tbody>
</table>

<div id="detail-sections"></div>

<h2>Runtime</h2>
<p class="muted">runtime dir: <code id="runtime-dir">-</code></p>
<p class="muted">records: <span id="records">-</span></p>

<footer>Raw JSON: <a href="/emullm/admin/state">/emullm/admin/state</a></footer>

<script>
const DETAIL = __DETAIL__;
function fmtUptime(s) {
  s = Math.floor(s || 0);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (h) return h + "h " + m + "m";
  if (m) return m + "m " + sec + "s";
  return sec + "s";
}
function pills(obj, onlyTrue) {
  const keys = Array.isArray(obj) ? obj : Object.keys(obj || {}).filter(k => !onlyTrue || obj[k]);
  if (!keys.length) return '<span class="muted">--</span>';
  return keys.map(k => '<span class="pill">' + k + '</span>').join(' ');
}
function roleBadge(role) {
  role = role || 'trusted';
  const cls = (role === 'trusted' || role === 'training') ? ' role-' + role : '';
  return '<span class="role' + cls + '">' + role + '</span>';
}
function fmtTime(t) { return t ? new Date(t * 1000).toLocaleTimeString() : '--'; }
function esc(x) { return String(x).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

// Header/toggle reflect which view we're in.
document.getElementById('view-tag').textContent = DETAIL ? '(detailed)' : '(overview)';
const toggle = document.getElementById('view-toggle');
toggle.textContent = DETAIL ? 'overview' : 'detailed view';
toggle.href = DETAIL ? '/emullm/status' : '/emullm/status/detail';

async function refresh() {
  try {
    const r = await fetch('/emullm/admin/state', { cache: 'no-store' });
    const s = await r.json();
    document.getElementById('mode').textContent = s.mode || 'relay';
    document.getElementById('uptime').textContent = fmtUptime(s.uptime_seconds);
    const workers = s.connected_worker_ids || [];
    document.getElementById('worker-count').textContent = workers.length;
    document.getElementById('pending-count').textContent = (s.pending_request_ids || []).length;
    document.getElementById('runtime-dir').textContent = s.runtime_dir || '-';
    const rc = s.record_counts || {};
    document.getElementById('records').textContent =
      Object.keys(rc).length ? Object.entries(rc).map(([k, v]) => k + ': ' + v).join(', ') : 'none';

    const roles = s.worker_roles || {};
    const tbody = document.getElementById('workers');
    if (!workers.length) {
      tbody.innerHTML = '<tr><td colspan="5" class="muted">no workers connected</td></tr>';
    } else {
      tbody.innerHTML = workers.map(id => {
        const models = (s.worker_models || {})[id] || [];
        const caps = (s.worker_capabilities || {})[id] || {};
        const u = (s.worker_usage || {})[id] || {};
        const usage = (u.requests_in_window ?? 0) + ' / ' + (u.max_per_window ?? '-') +
          ' &middot; ' + (u.total_requests ?? 0) + ' total';
        return '<tr><td><span class="dot on"></span><b>' + esc(id) + '</b></td>' +
          '<td>' + roleBadge(roles[id]) + '</td>' +
          '<td>' + pills(models) + '</td>' +
          '<td>' + pills(caps, true) + '</td>' +
          '<td>' + usage + '</td></tr>';
      }).join('');
    }

    // Detailed view adds per-worker usage timing and the pending-request list.
    const det = document.getElementById('detail-sections');
    if (DETAIL) {
      let html = '<h2>Per-worker detail</h2>';
      if (!workers.length) {
        html += '<p class="muted">no workers connected</p>';
      } else {
        html += workers.map(id => {
          const u = (s.worker_usage || {})[id] || {};
          const roleLine = 'role: ' + roleBadge(roles[id]);
          const usageLine = 'window: ' + (u.requests_in_window ?? 0) + '/' + (u.max_per_window ?? '-') +
            ' over ' + (u.window_seconds ?? '-') + 's &middot; total: ' + (u.total_requests ?? 0) +
            ' &middot; last used: ' + fmtTime(u.last_used_at);
          const models = (s.worker_models || {})[id] || [];
          return '<div class="card" style="display:block;margin-bottom:0.5rem">' +
            '<b>' + esc(id) + '</b> &middot; ' + roleLine + '<br>' +
            '<span class="muted">models:</span> ' + pills(models) + '<br>' +
            '<span class="muted">' + usageLine + '</span></div>';
        }).join('');
      }
      const pend = s.pending_request_ids || [];
      html += '<h2>Pending requests (' + pend.length + ')</h2>';
      html += pend.length
        ? '<p>' + pend.map(x => '<span class="pill">' + esc(x) + '</span>').join(' ') + '</p>'
        : '<p class="muted">none waiting</p>';
      html += '<h2>Server</h2><p class="muted">started: ' + fmtTime(s.started_at) + '</p>';
      det.innerHTML = html;
    }
    document.getElementById('updated').textContent = 'updated ' + new Date().toLocaleTimeString();
  } catch (e) {
    document.getElementById('updated').textContent = 'error fetching state';
  }
}
refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>"""


def _render_status_page(detail: bool) -> str:
    return _STATUS_PAGE_HTML.replace("__DETAIL__", "true" if detail else "false")


@router.get("/admin/emullm/status", response_class=HTMLResponse)
@router.get("/emullm/status", response_class=HTMLResponse)
def status_page() -> str:
    """Concise operator overview: mode, worker roles, models, usage."""
    return _render_status_page(False)


@router.get("/admin/emullm/status/detail", response_class=HTMLResponse)
@router.get("/emullm/status/detail", response_class=HTMLResponse)
def status_page_detail() -> str:
    """Detailed view: adds per-worker usage timing and the pending-request list."""
    return _render_status_page(True)


# --- Managed worker control (the `auto`-mode supervisor) -------------------
# When the server was started in `auto` mode it spawned its own worker
# subprocesses (see emullm.supervisor / app.py). These endpoints let an
# operator list them and start/stop individual ones without restarting the
# server. If no supervisor is active (e.g. not in `auto` mode), the list is
# empty and start/stop report that.
@router.get("/admin/emullm/workers")
@router.get("/emullm/admin/workers")
def admin_list_workers() -> dict[str, Any]:
    supervisor = _sup.get_supervisor()
    return {
        "supervisor_active": supervisor is not None,
        "workers": supervisor.status() if supervisor else [],
    }


@router.post("/admin/emullm/workers/{worker_id}/start")
@router.post("/emullm/admin/workers/{worker_id}/start")
def admin_start_worker(worker_id: str) -> dict[str, Any]:
    supervisor = _sup.get_supervisor()
    if supervisor is None:
        raise HTTPException(status_code=409, detail="no supervisor active (server not in auto mode)")
    if worker_id not in {s.worker_id for s in supervisor.specs()}:
        raise HTTPException(status_code=404, detail=f"no managed worker '{worker_id}'")
    started = supervisor.start(worker_id)
    return {"worker_id": worker_id, "started": started, "workers": supervisor.status()}


@router.post("/admin/emullm/workers/{worker_id}/stop")
@router.post("/emullm/admin/workers/{worker_id}/stop")
def admin_stop_worker(worker_id: str) -> dict[str, Any]:
    supervisor = _sup.get_supervisor()
    if supervisor is None:
        raise HTTPException(status_code=409, detail="no supervisor active (server not in auto mode)")
    if worker_id not in {s.worker_id for s in supervisor.specs()}:
        raise HTTPException(status_code=404, detail=f"no managed worker '{worker_id}'")
    stopped = supervisor.stop(worker_id)
    return {"worker_id": worker_id, "stopped": stopped, "workers": supervisor.status()}


# --- config.json (typed schema + read/write) ------------------------------
# A small persisted config document the admin page edits, validated against
# the schema below on write (unknown top-level keys are rejected so a
# hand-edited file catches typos). Location: EMULLM_CONFIG_FILE, else
# <repo root>/config.json.
_CONFIG_PATH = Path(os.environ.get("EMULLM_CONFIG_FILE") or (Path(__file__).resolve().parent.parent.parent / "config.json"))


class WorkerConfig(BaseModel):
    """One managed (auto-mode) worker. ``id`` (or ``worker_id``) names it;
    the optional ``launch`` overrides the worker type for this entry."""

    model_config = ConfigDict(extra="allow")
    id: str | None = None
    worker_id: str | None = None
    role: str | None = None
    cwd: str | None = None
    launch: str | list[str] | None = None
    model: str | None = None  # AI model for a Copilot-agent launch (ignored for the plain worker loop)


class MockWorkerConfig(BaseModel):
    """A pretend peer registered at startup (see register_mock_workers)."""

    model_config = ConfigDict(extra="allow")
    id: str | None = None
    worker_id: str | None = None
    reply: str | None = None
    template: str | None = None
    capabilities: dict[str, bool] | list[str] | None = None
    role: str | None = None
    models: Any | None = None


class BackendConfig(BaseModel):
    """A real OpenAI-compatible upstream for the proxy modes."""

    model_config = ConfigDict(extra="allow")
    name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    model: str | None = None
    default: bool | None = None


class MockConfig(BaseModel):
    """Global mock reply for ``mock`` mode (when no mock_worker matches)."""

    model_config = ConfigDict(extra="allow")
    reply: str | None = None
    template: str | None = None


class ServiceConfig(BaseModel):
    """What happens at one /v1 service (chat, embeddings, images, ...).

    For a request the server first tries to *find an agent that serves it*
    (in ``mode`` order); ``fallback`` is *what we do when we can't*.
    ``behavior`` pins how a specific agent handles the service; both are
    optional. ``description`` is user-facing (surfaced on the status page)."""

    model_config = ConfigDict(extra="allow")
    behavior: Literal["serve", "stub", "wait", "error", "decline", "aggregate"] | None = None
    fallback: Literal["stub", "wait", "error"] | None = None
    strategy: Literal["failover", "round-robin", "random", "weighted", "broadcast"] | None = None
    description: str | None = None
    # NOTE: an "aggregate" models entry may also carry `validate: true` (ping
    # each claimed model, keep only live ones) -- accepted as an extra key
    # rather than a typed field, since `validate` shadows a BaseModel method.


class AgentConfig(BaseModel):
    """One answerer. ``launch`` is its type: ``recruit`` (interactive, connects
    itself), ``subagent`` (auto-configured, we spawn it), ``proxy`` (a real
    upstream we route to), or ``mock`` (a deliberate fake). ``services`` maps a
    /v1 service to how this agent handles it (string shorthand or a
    ServiceConfig). Launch-specific fields ride alongside."""

    model_config = ConfigDict(extra="allow")
    kind: Literal["agent"] = "agent"
    id: str | None = None
    worker_id: str | None = None
    launch: Literal["recruit", "subagent", "proxy", "mock"] | None = None
    description: str | None = None  # user-facing
    role: str | None = None
    # subagent-only:
    command: str | list[str] | None = None
    cwd: str | None = None
    # a worker/agent can serve specific catalog model ids, or take over another
    # agent's whole catalog (both drive the model -> worker route map):
    serves: list[str] | None = None
    replaces: str | None = None
    # proxy-only:
    name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    model: str | None = None
    default: bool | None = None
    # mock-only:
    reply: str | None = None
    template: str | None = None
    capabilities: dict[str, bool] | list[str] | None = None
    models: Any | None = None
    # what exchanges this agent wants mirrored to it (as in proxy-observe):
    # true/"all" for everything, or a list of service names to scope it.
    observe: bool | str | list[str] | None = None
    # a (proxy) agent publishes its models into the user-facing catalog via a
    # `services.models` entry with behavior "aggregate". The refresh cadence is
    # `validation_interval` (`update_interval` is a legacy alias): null = never
    # (use the config `models`), or a duration like "1day"/"12h"/"30m". May be
    # set here (agent-level) or inside the services.models entry.
    validation_interval: str | int | float | None = None
    update_interval: str | int | float | None = None
    # per-service behavior/description for this agent:
    services: dict[str, str | ServiceConfig] | None = None


class ServicesConfig(BaseModel):
    """Server-level service catalog. ``model`` is the default model we
    advertise, ``models`` the full advertised list, and ``default`` marks this
    catalog as the default. Any *other* key is a service name (chat,
    embeddings, images, ...) mapped to its fallback behavior -- a bare string
    or a ServiceConfig -- kept as an extra field."""

    model_config = ConfigDict(extra="allow")
    model: str | None = None
    default: bool | None = None
    models: list[str | dict[str, Any]] | None = None  # ids or nodes (with validation results)


class EmullmConfig(BaseModel):
    """The full config.json schema. Every field is optional (omit to fall
    back to the matching env var / default); unknown top-level keys are
    rejected so a hand-edited file catches typos.

    ``agents`` is the unified answerer list (kind/launch/services); the flat
    ``workers``/``mock_workers``/``backends``/``mock`` keys are the earlier
    per-type forms and are still accepted."""

    model_config = ConfigDict(extra="forbid")
    description: str | None = None  # user-facing server/service description
    mode: str | list[str] | None = None
    subagent_launch: str | list[str] | None = None
    subagent_model: str | None = None  # default AI model for spawned Copilot workers
    model_routes: dict[str, str] | None = None  # catalog model id -> serving worker_id
    capability_fallback: Literal["stub", "wait", "error"] | None = None
    validation_interval_default: str | int | float | None = None  # inherited default cadence
    validation_interval_override: str | int | float | None = None  # forces cadence on ALL agents
    validation_interval: str | int | float | None = None  # alias for validation_interval_default
    services: ServicesConfig | None = None  # server-level catalog + per-service fallback
    agents: list[AgentConfig] | None = None
    # legacy flat forms (still honored by the runtime today):
    workers: list[WorkerConfig] | None = None
    mock_workers: list[MockWorkerConfig] | None = None
    backends: list[BackendConfig] | None = None
    mock: MockConfig | None = None


class SaveConfigRequest(BaseModel):
    config: dict[str, Any]


def _read_config() -> dict[str, Any]:
    try:
        if _CONFIG_PATH.is_file():
            data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except (OSError, ValueError):
        pass
    return {}


@router.get("/admin/emullm/config")
@router.get("/emullm/admin/config")
def admin_get_config() -> dict[str, Any]:
    return {"path": str(_CONFIG_PATH), "config": _read_config()}


@router.get("/admin/emullm/config/schema")
@router.get("/emullm/admin/config/schema")
def admin_get_config_schema() -> dict[str, Any]:
    """The JSON Schema for config.json (what a PUT is validated against)."""
    return EmullmConfig.model_json_schema()


@router.put("/admin/emullm/config")
@router.put("/emullm/admin/config")
def admin_put_config(body: SaveConfigRequest) -> dict[str, Any]:
    try:
        EmullmConfig.model_validate(body.config)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=[
                {"loc": list(err.get("loc", ())), "msg": err.get("msg"), "type": err.get("type")}
                for err in exc.errors()
            ],
        )
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(body.config, indent=2) + "\n", encoding="utf-8")
    return {"path": str(_CONFIG_PATH), "config": body.config, "saved": True}


@router.get("/admin/emullm/backends/probe")
@router.get("/emullm/admin/backends/probe")
async def admin_probe_backends(verify: bool = False, limit: int | None = None) -> dict[str, Any]:
    """On-demand: probe each configured proxy backend's ``/v1/models`` (its
    reference capability set). With ``?verify=true`` each model is actually
    called (chat, then embeddings) to catch *falsely advertised* models --
    listed but never loaded -- splitting them into ``live`` /
    ``falsely_advertised``. ``?limit=N`` caps how many models are exercised.
    Not on any hot path or startup; an unreachable backend reports
    ``ok: false`` and never breaks the server."""
    return {"backends": await probe_backends(verify=verify, limit=limit)}


# --- admin control page ----------------------------------------------------
# A static-ish operator page: edit config.json and start/stop managed
# workers, on top of the endpoints above. This surface MUTATES state and can
# spawn processes, so keep it bound to localhost (or otherwise gated) -- do
# not expose it publicly like the keyless /v1 surface.
_ADMIN_PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>emullm -- admin</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; max-width: 60rem; margin: 2rem auto; padding: 0 1rem; }
  h1 { margin-bottom: 0.25rem; }
  .sub { color: #888; font-size: 0.9rem; margin-top: 0; }
  .warn { background: #f39c1222; border: 1px solid #f39c1288; padding: 0.5rem 0.75rem; border-radius: 6px; font-size: 0.85rem; }
  table { border-collapse: collapse; width: 100%; margin-top: 0.5rem; }
  th, td { text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid #8882; font-size: 0.9rem; }
  th { color: #888; }
  button { font-size: 0.85rem; padding: 0.25rem 0.7rem; cursor: pointer; margin-right: 0.3rem; }
  textarea { width: 100%; min-height: 16rem; font-family: monospace; font-size: 0.85rem; padding: 0.5rem; box-sizing: border-box; }
  .dot { height: 0.6rem; width: 0.6rem; border-radius: 50%; display: inline-block; margin-right: 0.35rem; }
  .on { background: #2ecc71; } .off { background: #bbb; }
  .muted { color: #888; }
  .msg { font-size: 0.85rem; margin-left: 0.5rem; }
  .ok { color: #2ecc71; } .err { color: #e74c3c; }
</style>
</head>
<body>
<h1>emullm admin</h1>
<p class="sub">mode <b id="mode">-</b> &middot; <a href="/emullm/status">status dashboard</a> &middot;
  <span id="updated" class="muted">-</span></p>
<p class="warn">This page can edit config and start/stop worker processes. Keep it bound to localhost.</p>

<h2>Managed workers <span id="sup-note" class="muted"></span></h2>
<table>
  <thead><tr><th>Worker</th><th>Role</th><th>PID</th><th>State</th><th>Actions</th></tr></thead>
  <tbody id="workers"><tr><td colspan="5" class="muted">loading...</td></tr></tbody>
</table>

<h2>config.json <span id="cfg-path" class="muted"></span></h2>
<textarea id="config" spellcheck="false"></textarea>
<div style="margin-top:0.5rem">
  <button id="reload">Reload</button>
  <button id="save">Save</button>
  <span id="cfg-msg" class="msg"></span>
</div>

<script>
// Resolve REST calls relative to wherever this page is served, so it works
// under either admin prefix (/emullm/admin or /admin/emullm) -- and would
// survive being mounted under a sub-path. Both alias trees exist server-side.
const ADMIN = location.pathname.replace(/\/+$/, '') || '/emullm/admin';
async function getJSON(u, opts) { const r = await fetch(u, opts); return { ok: r.ok, status: r.status, body: await r.json().catch(() => ({})) }; }
function esc(x) { return String(x).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

async function refreshWorkers() {
  const r = await getJSON(ADMIN + '/workers', { cache: 'no-store' });
  const note = document.getElementById('sup-note');
  const tbody = document.getElementById('workers');
  const workers = (r.body && r.body.workers) || [];
  note.textContent = r.body.supervisor_active ? '' : '(no supervisor -- server not in auto mode)';
  if (!workers.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="muted">none</td></tr>';
    return;
  }
  tbody.innerHTML = workers.map(w =>
    '<tr><td><span class="dot ' + (w.running ? 'on' : 'off') + '"></span><b>' + esc(w.worker_id) + '</b></td>' +
    '<td>' + esc(w.role || '') + '</td>' +
    '<td>' + (w.pid ?? '--') + '</td>' +
    '<td>' + (w.running ? 'running' : 'stopped') + '</td>' +
    '<td><button data-act="start" data-id="' + esc(w.worker_id) + '">Start</button>' +
    '<button data-act="stop" data-id="' + esc(w.worker_id) + '">Stop</button></td></tr>'
  ).join('');
}
document.getElementById('workers').addEventListener('click', async (e) => {
  const btn = e.target.closest('button'); if (!btn) return;
  await fetch(ADMIN + '/workers/' + encodeURIComponent(btn.dataset.id) + '/' + btn.dataset.act, { method: 'POST' });
  refreshWorkers();
});

async function loadConfig() {
  const r = await getJSON(ADMIN + '/config', { cache: 'no-store' });
  document.getElementById('cfg-path').textContent = r.body.path || '';
  document.getElementById('config').value = JSON.stringify(r.body.config || {}, null, 2);
  setMsg('', '');
}
function setMsg(text, cls) { const m = document.getElementById('cfg-msg'); m.textContent = text; m.className = 'msg ' + cls; }
document.getElementById('reload').addEventListener('click', loadConfig);
document.getElementById('save').addEventListener('click', async () => {
  let parsed;
  try { parsed = JSON.parse(document.getElementById('config').value); }
  catch (err) { setMsg('invalid JSON: ' + err.message, 'err'); return; }
  const r = await getJSON(ADMIN + '/config', {
    method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ config: parsed }),
  });
  setMsg(r.ok ? 'saved' : ('save failed (' + r.status + ')'), r.ok ? 'ok' : 'err');
});

async function tick() {
  try {
    const s = await getJSON(ADMIN + '/state', { cache: 'no-store' });
    document.getElementById('mode').textContent = (s.body && s.body.mode) || 'relay';
    await refreshWorkers();
    document.getElementById('updated').textContent = 'updated ' + new Date().toLocaleTimeString();
  } catch (e) { document.getElementById('updated').textContent = 'error'; }
}
loadConfig();
tick();
setInterval(tick, 3000);
</script>
</body>
</html>"""


@router.get("/admin/emullm", response_class=HTMLResponse)
@router.get("/emullm/admin", response_class=HTMLResponse)
def admin_page() -> str:
    """Operator control page: edit config.json and start/stop managed workers."""
    return _ADMIN_PAGE_HTML


@router.post("/admin/emullm/runtime_dir")
@router.post("/emullm/admin/runtime_dir")
def admin_set_runtime_dir(body: SetRuntimeDirRequest) -> dict[str, Any]:
    """Repoint every durable record store at a new root directory (e.g. a test's
    tmp_path), so tests can isolate themselves from the real
    workbench/server/runtime/emullm/ directory over plain HTTP."""
    global _RUNTIME_DIR
    _RUNTIME_DIR = Path(body.path)
    for kind, store in _KIND_STORES.items():
        store._dir = _RUNTIME_DIR / kind
    return admin_state()


class ModelRoutesRequest(BaseModel):
    routes: dict[str, str] = Field(default_factory=dict)
    replace: bool = False


@router.get("/admin/emullm/model_routes")
@router.get("/emullm/admin/model_routes")
def admin_get_model_routes() -> dict[str, Any]:
    return {"model_routes": dict(_model_routes)}


@router.post("/admin/emullm/model_routes")
@router.post("/emullm/admin/model_routes")
def admin_set_model_routes(body: ModelRoutesRequest) -> dict[str, Any]:
    """Set catalog-model-id -> serving-worker-id routes at runtime (for ad-hoc
    workers not declared in config). Merges by default; ``replace: true`` clears
    first; an empty worker_id removes that route."""
    if body.replace:
        _model_routes.clear()
    for mid, wid in body.routes.items():
        if not isinstance(mid, str) or not mid:
            continue
        if wid:
            _model_routes[mid] = str(wid)
        else:
            _model_routes.pop(mid, None)
    return {"model_routes": dict(_model_routes)}


@router.post("/admin/emullm/reset")
@router.post("/emullm/admin/reset")
def admin_reset() -> dict[str, Any]:
    """Deletes every persisted record (files/assistants/threads/fine-tuning
    jobs/events) under the current runtime dir. Does not touch a
    connected worker or in-flight relayed requests."""
    removed = {kind: store.clear() for kind, store in _KIND_STORES.items()}
    return {"removed": removed, **admin_state()}


@router.post("/admin/emullm/usage/reset")
@router.post("/emullm/admin/usage/reset")
def admin_reset_usage(worker_id: str | None = None) -> dict[str, Any]:
    """Clears rate-limit usage counters -- for one worker_id, or every
    worker_id if none is given. Does not affect connections/records."""
    if worker_id is None:
        _worker_usage.clear()
    else:
        _worker_usage.pop(worker_id, None)
    return admin_state()


@router.delete("/admin/emullm/records/{kind}/{record_id}")
@router.delete("/emullm/admin/records/{kind}/{record_id}")
def admin_delete_record(kind: str, record_id: str) -> dict[str, Any]:
    store = _KIND_STORES.get(kind)
    if store is None:
        raise HTTPException(status_code=404, detail=f"unknown kind '{kind}' (expected one of {sorted(_KIND_STORES)})")
    if not store.delete(record_id):
        raise HTTPException(status_code=404, detail=f"no such record '{record_id}' in '{kind}'")
    return {"deleted": record_id, "kind": kind}


_KIND_STORES.update(
    {
        "files": _files_store,
        "assistants": _assistants_store,
        "threads": _threads_store,
        "fine_tuning_jobs": _fine_tuning_jobs_store,
        "fine_tuning_events": _fine_tuning_events_store,
    }
)


@router.websocket("/emullm/{worker_id}/ws")
async def emullm_socket(websocket: WebSocket, worker_id: str) -> None:
    """A small pool of workers can be connected at once, one per
    worker_id (taken directly from the URL path -- e.g. connect to
    /emullm/yourself/ws, /emullm/alice/ws, .../bob/ws). A new
    connection under the SAME worker_id replaces the previous one for
    that id, but a different worker_id is tracked independently and
    routed to separately.

    On connect, the server optionally asks the worker to declare its own
    model list / capabilities: it sends {"type":"hello","worker_id":
    worker_id} and waits (briefly) for an optional
    {"type":"register", "models": {suffix: {display_name, instruction},
    ...}, "capabilities": {embeddings: bool, moderations: bool, images:
    bool, audio_transcription: bool, audio_speech: bool, fine_tuning:
    bool}}. A worker that
    skips this (or never sends anything at all -- e.g. an older/simpler
    client) just falls back to _PERSONA_SUFFIXES and no extra
    capabilities, under whatever worker_id the URL gave it.

    Job protocol: the server sends {"type":"request","id",...,"prompt",
    optional "persona_instruction", and -- for two-way media -- "images"
    (urls/data-urls from a vision request), "audio" (a cloud file URL of a
    clip to transcribe), "files" (e.g. a fine-tune training_file + cloud
    url), and "kind" (chat/vision/image/audio_speech/audio_transcription/
    fine_tuning). The worker replies {"type":"reply","id","content"} and MAY
    additionally return real media: "image_b64"/"image_url"/"mime" (image
    gen) or "audio_b64"/"audio_url"/"mime" (speech). Those are persisted to
    the shared cloud files store and passed back through the matching
    /v1/... endpoint."""
    await websocket.accept()
    first: dict[str, Any] | None = None
    try:
        await websocket.send_json({"type": "hello", "worker_id": worker_id})
        first = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
    except (asyncio.TimeoutError, WebSocketDisconnect):
        first = None
    except Exception:
        first = None

    if isinstance(first, dict) and first.get("type") == "register":
        models = first.get("models")
        if isinstance(models, dict) and models:
            _worker_models[worker_id] = models
        capabilities = first.get("capabilities")
        if isinstance(capabilities, dict):
            _worker_capabilities[worker_id] = {str(k): bool(v) for k, v in capabilities.items()}
        role = first.get("role")
        if isinstance(role, str) and role.strip():
            _worker_roles[worker_id] = role.strip()
    elif isinstance(first, dict):
        # Not a register message -- an older/simpler worker that ignored
        # the "hello" and just started talking. Don't drop it; handle it
        # as normal traffic under this instance_id.
        await _handle_worker_message(worker_id, first)

    async with _worker_lock:
        _connected_workers[worker_id] = websocket
    try:
        while True:
            data = await websocket.receive_json()
            await _handle_worker_message(worker_id, data)
    except WebSocketDisconnect:
        pass
    finally:
        async with _worker_lock:
            if _connected_workers.get(worker_id) is websocket:
                del _connected_workers[worker_id]


async def _handle_worker_message(worker_id: str, data: dict[str, Any]) -> None:
    if data.get("type") == "reply":
        request_id = str(data.get("id") or "")
        future = _pending.pop(request_id, None)
        if future and not future.done():
            # A two-way worker may return real media alongside (or instead of)
            # text: image_b64 / image_url / mime. Keep the reply structured so
            # image-gen and vision can hand back actual bytes.
            reply: dict[str, Any] = {"content": str(data.get("content") or "")}
            for key in ("image_b64", "image_url", "audio_b64", "audio_url", "mime", "images", "file_id", "file_url"):
                if data.get(key) is not None:
                    reply[key] = data[key]
            future.set_result(reply)

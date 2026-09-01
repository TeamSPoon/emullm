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
  - /ws_collab/v1/*                  -- mailbox_chat-compatible service API
                                         (also /mailbox_chat/v1, /api,
                                         /emullm, and bare aliases):
                                         workers are durable mailboxes,
                                         with config/mailboxes.json and
                                         events_logs/<worker>.jsonl
  - WS   /ws_collab/ws                -- mailbox_chat stream adapter
                                         (also /mailbox_chat/ws,
                                         /mailbox/ws, /emullm/mailbox/ws)
  - /emullm/specific_worker/{worker_id}/v1/*
                                     -- the SAME /v1/* surface (models,
                                         chat/completions, messages,
                                         completions, responses, embeddings,
                                         moderations, images, audio),
                                         but with worker_id pinned from
                                         the URL instead of parsed out of
                                         "model" -- for a client that can
                                         only configure a fixed baseUrl
  - POST /v1/chat/completions        -- relayed to the connected worker (real)
                                         with normal JSON or SSE output
  - POST /v1/messages                -- Anthropic Messages API-compatible
                                         surface (Claude SDK / Claude Code
                                         style). Relayed exactly like
                                         chat/completions; the reply is
                                         reshaped into Anthropic content
                                         blocks, with Anthropic-style SSE
                                         events when stream=true
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

Any request that is genuinely relayed (chat/completions, messages,
completions, responses) is queued and forwarded to whichever worker is currently
connected at WebSocket /emullm/ws. A worker identifies with the ``worker_id``
query parameter and can declare ``modelmasks`` query patterns; a worker without
masks is eligible for every model. The worker reads the forwarded prompt, composes a
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
from contextvars import ContextVar
from fnmatch import fnmatchcase
import hashlib
import ipaddress
import json
import logging
import math
import mimetypes
import os
import re
import secrets
import struct
import tempfile
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.background import BackgroundTask, BackgroundTasks
from starlette.datastructures import UploadFile

from . import copilot_api as _copilot_api
from . import process_control as _process_control
from .test_media import test_media_samples
from . import supervisor as _sup

_request_affinity: ContextVar[dict[str, Any] | None] = ContextVar(
    "emullm_request_affinity",
    default=None,
)
_request_assigned_worker: ContextVar[str | None] = ContextVar(
    "emullm_request_assigned_worker",
    default=None,
)


class _ClientTrackingRoute(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()

        async def tracked(request: Request) -> Response:
            path = request.url.path.rstrip("/")
            if (
                path == "/emullm/admin"
                or path.startswith("/emullm/admin/")
                or path == "/admin/emullm"
                or path.startswith("/admin/emullm/")
            ):
                _require_local_process_control(request)
            if not request.url.path.startswith("/v1/"):
                return await original(request)
            client = request.client
            host = client.host if client is not None else "unknown"
            port = client.port if client is not None else 0
            port_start = (
                port // _COPILOT_AFFINITY_PORT_RANGE
            ) * _COPILOT_AFFINITY_PORT_RANGE
            affinity_token = _request_affinity.set(
                {
                    "host": host,
                    "port": port,
                    "port_start": port_start,
                    "port_end": (
                        port_start + _COPILOT_AFFINITY_PORT_RANGE - 1
                    ),
                }
            )
            assigned_token = _request_assigned_worker.set(None)
            await _wait_for_client_worker_capacity(request)
            request_token = _begin_openai_client_request(request)
            try:
                try:
                    response = await original(request)
                except BaseException as error:
                    _finish_openai_client_request(
                        request_token,
                        int(getattr(error, "status_code", 500)),
                    )
                    raise
                assigned_worker = _request_assigned_worker.get()
                if assigned_worker:
                    response.headers["X-EmuLLM-Worker-ID"] = assigned_worker
                finalize = BackgroundTask(
                    _finish_openai_client_request,
                    request_token,
                    response.status_code,
                )
                response.background = (
                    BackgroundTasks([finalize, response.background])
                    if response.background is not None
                    else finalize
                )
                return response
            finally:
                _request_assigned_worker.reset(assigned_token)
                _request_affinity.reset(affinity_token)

        return tracked


router = APIRouter(route_class=_ClientTrackingRoute)
_LOGGER = logging.getLogger(__name__)

_REQUEST_TIMEOUT_SECONDS = 900  # generous -- a human/agent may take a while to reply


def _websocket_catalog_parameters(path: str) -> list[dict[str, Any]]:
    if path == "/emullm/ws":
        return [
            {
                "name": "worker_id",
                "in": "query",
                "required": False,
                "description": "Stable worker/mailbox identity; omitted values receive a generated servant ID.",
                "schema": {"type": "string"},
            },
            {
                "name": "modelmasks",
                "in": "query",
                "required": False,
                "description": "Comma-separated or repeated model glob patterns; omit to accept all models.",
                "schema": {"type": "array", "items": {"type": "string"}},
            },
        ]
    if path in {"/emullm/websock_to_llm_user/ws", "/websock_to_llm_user/ws"}:
        return [
            {
                "name": name,
                "in": "query",
                "required": False,
                "description": description,
                "schema": {"type": "string"},
            }
            for name, description in (
                ("worker_id", "Return interactions for one worker identity."),
                ("model", "Return interactions for one exact model ID."),
                ("modelmask", "Return interactions whose model matches this glob."),
                ("type", "Return one or more comma-separated event types."),
                ("after", "Resume after this event cursor."),
            )
        ]
    return []


@router.get("/endpoints")
@router.get("/emullm/endpoints")
def service_catalog() -> dict[str, Any]:
    """Advertise EMULLM services with comments, parameters, and schemas."""
    def walk_routes(routes: list[Any]) -> list[Any]:
        flattened: list[Any] = []
        for route in routes:
            included = getattr(route, "original_router", None)
            if included is not None:
                flattened.extend(walk_routes(list(getattr(included, "routes", []))))
            else:
                flattened.append(route)
        return flattened

    catalog_routes = walk_routes(list(router.routes))
    openapi = get_openapi(title="EMULLM", version="1.0.0", routes=catalog_routes)
    openapi_paths = openapi.get("paths", {})
    endpoints: list[dict[str, Any]] = []
    for route in catalog_routes:
        path = getattr(route, "path", None)
        if not isinstance(path, str):
            continue
        methods = sorted(method for method in (getattr(route, "methods", None) or []) if method != "HEAD")
        endpoint = getattr(route, "endpoint", None)
        doc = (getattr(endpoint, "__doc__", None) or "").strip()
        name = str(getattr(route, "name", "") or "")
        if not methods:
            comment = doc or name.replace("_", " ").strip().capitalize() or "WebSocket endpoint"
            endpoints.append(
                {
                    "path": path,
                    "methods": ["WS"],
                    "name": name,
                    "summary": comment.splitlines()[0],
                    "description": comment,
                    "comment": comment,
                    "transport": "websocket",
                    "parameters": _websocket_catalog_parameters(path),
                    "requestBody": None,
                    "responses": {"101": {"description": "WebSocket protocol upgrade"}},
                }
            )
            continue

        operations: dict[str, dict[str, Any]] = {}
        for method in methods:
            operation = dict(openapi_paths.get(path, {}).get(method.lower(), {}))
            operations[method] = {
                key: operation[key]
                for key in (
                    "summary",
                    "description",
                    "parameters",
                    "requestBody",
                    "responses",
                    "tags",
                    "deprecated",
                    "security",
                )
                if key in operation
            }
        primary = operations.get(methods[0], {})
        comment = str(primary.get("description") or doc or primary.get("summary") or name.replace("_", " ")).strip()
        endpoints.append(
            {
                "path": path,
                "methods": methods,
                "name": name,
                "summary": str(primary.get("summary") or comment.splitlines()[0]),
                "description": comment,
                "comment": comment,
                "transport": "http",
                "parameters": primary.get("parameters", []),
                "requestBody": primary.get("requestBody"),
                "responses": primary.get("responses", {}),
                "operations": operations,
            }
        )
    endpoints.sort(key=lambda entry: (entry["path"], entry["methods"]))
    return {
        "id": "emullm",
        "prefix": "/emullm",
        "count": len(endpoints),
        "endpoints": endpoints,
        "schemas": openapi.get("components", {}).get("schemas", {}),
    }


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


class _WorkerRejected(Exception):
    """A worker declined an offered request before producing a reply."""

    def __init__(self, worker_id: str, reason: str) -> None:
        self.worker_id = worker_id
        self.reason = reason
        super().__init__(f"worker '{worker_id}' rejected the request: {reason}")


class _WorkerNotReady(Exception):
    """A worker encountered a transient failure and asked to be retried later."""

    def __init__(self, worker_id: str, reason: str, retry_after: float) -> None:
        self.worker_id = worker_id
        self.reason = reason
        self.retry_after = retry_after
        super().__init__(
            f"worker '{worker_id}' is not ready for {retry_after:g}s: {reason}"
        )


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
    "capabilities": ["images"], "role": "trusted", "models": {...},
    "modelmasks": ["vendor/*"] }``.
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
        _worker_kinds[worker_id] = "mock"
        _worker_descriptions[worker_id] = str(
            spec.get("description") or f"Configured mock worker {worker_id}."
        )
        models = spec.get("models")
        if isinstance(models, dict) and models:
            _worker_models[worker_id] = models
        if "modelmasks" in spec:
            _set_worker_model_masks(worker_id, _normalise_model_masks(spec["modelmasks"]))
        else:
            _worker_model_masks.pop(worker_id, None)
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
        _worker_model_masks.pop(worker_id, None)
        _worker_kinds.pop(worker_id, None)
        _worker_runtime_models.pop(worker_id, None)
        _worker_descriptions.pop(worker_id, None)


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
        await _send_worker_json(
            worker_id,
            worker,
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
_native_worker_ids: set[str] = set()
_worker_lock = asyncio.Lock()
_pending: dict[str, "asyncio.Future[Any]"] = {}
_pending_models: dict[str, str] = {}
_pending_worker_controls: dict[str, "asyncio.Future[dict[str, Any]]"] = {}
_worker_not_ready_until: dict[str, float] = {}
_worker_inflight: dict[str, int] = {}
_worker_reservations: dict[str, int] = {}
_model_inflight: dict[str, int] = {}
_worker_last_busy_at: dict[str, float] = {}
_worker_load_lock = threading.RLock()
_worker_service_stats: dict[str, dict[str, dict[str, float | int]]] = {}
_model_service_stats: dict[str, dict[str, dict[str, float | int]]] = {}
_active_service_requests: dict[str, dict[str, Any]] = {}
_waiting_for_worker: dict[str, dict[str, Any]] = {}
_waiting_for_worker_lock = threading.RLock()
_admin_test_tasks: dict[str, "asyncio.Task[Any]"] = {}
_active_websockets: dict[str, dict[str, Any]] = {}
_active_websockets_lock = threading.RLock()
_worker_connection_ids: dict[str, str] = {}
_socket_worker_log_dir = (
    Path(tempfile.gettempdir()) / "emullm" / "socket-worker-logs"
)
_socket_worker_log_segment_bytes = 2 * 1024 * 1024
_socket_worker_log_lock = threading.RLock()
_socket_worker_media_file_bytes = 25 * 1024 * 1024
_socket_worker_media_total_bytes = 64 * 1024 * 1024
_openai_clients: dict[str, dict[str, Any]] = {}
_openai_requests: dict[str, dict[str, Any]] = {}
_client_capacity_waiters: dict[str, int] = {}
_openai_clients_lock = threading.RLock()
_MAX_OPENAI_CLIENTS = 500
_MAX_OPENAI_REQUESTS = 500
_CLIENT_WORKER_RESERVE_MIN = 5
_CLIENT_WORKER_RESERVE_FRACTION = 0.30
_STUCK_WORKER_SECONDS = 120.0
_BULK_WORKER_ACTION_BATCH_SIZE = 7
_WORKER_RECONNECT_TIMEOUT_SECONDS = 60.0
_WORKER_CAPACITY_PATHS = {
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/messages",
    "/v1/responses",
    "/v1/images/generations",
    "/v1/images/edits",
    "/v1/audio/transcriptions",
    "/v1/audio/speech",
}
_ON_DEMAND_COPILOT_PREFIX = "worker-copilot-"
_ON_DEMAND_COPILOT_START = 5
_BASELINE_COPILOT_WORKERS = 4
_MAX_CONCURRENT_CALLS_LIMIT = 50
_DEFAULT_MAX_CONCURRENT_CALLS = max(
    _BASELINE_COPILOT_WORKERS,
    min(
        _MAX_CONCURRENT_CALLS_LIMIT,
        int(os.environ.get("EMULLM_MAX_CONCURRENT_CALLS", "50")),
    ),
)
_max_concurrent_calls = _DEFAULT_MAX_CONCURRENT_CALLS
_DEFAULT_IDLE_WORKER_TARGET = 5
_idle_worker_target = _DEFAULT_IDLE_WORKER_TARGET
_DEFAULT_IDLE_GRACE_SECONDS = 30.0
_idle_grace_seconds = _DEFAULT_IDLE_GRACE_SECONDS
_DEFAULT_BACKEND_FALLBACK_DELAY_SECONDS = 5.0
_backend_fallback_delay_seconds = _DEFAULT_BACKEND_FALLBACK_DELAY_SECONDS
_idle_maintenance_paused = False
_on_demand_copilot_lock = threading.RLock()


def _on_demand_copilot_limit() -> int:
    return max(0, _max_concurrent_calls - _BASELINE_COPILOT_WORKERS)


def _on_demand_copilot_ids() -> list[str]:
    return [
        f"{_ON_DEMAND_COPILOT_PREFIX}{index}"
        for index in range(
            _ON_DEMAND_COPILOT_START,
            _ON_DEMAND_COPILOT_START + _on_demand_copilot_limit(),
        )
    ]


def _is_on_demand_copilot(worker_id: str) -> bool:
    return worker_id in _on_demand_copilot_ids()


def _is_elastic_copilot(worker_id: str) -> bool:
    if not worker_id.startswith(_ON_DEMAND_COPILOT_PREFIX):
        return False
    suffix = worker_id[len(_ON_DEMAND_COPILOT_PREFIX) :]
    return suffix.isdigit() and int(suffix) >= _ON_DEMAND_COPILOT_START


def _register_active_websocket(
    websocket: WebSocket, kind: str, **metadata: Any
) -> str:
    connection_id = f"ws-{uuid.uuid4().hex[:16]}"
    client = websocket.client
    with _active_websockets_lock:
        _active_websockets[connection_id] = {
            "connection_id": connection_id,
            "kind": kind,
            "endpoint": str(websocket.scope.get("path") or ""),
            "client": (
                f"{client.host}:{client.port}"
                if client is not None
                else None
            ),
            "connected_at": _now_iso(),
            "connected_at_epoch": time.time(),
            "last_satisfied_at": None,
            "last_satisfied_at_epoch": None,
            "last_satisfied_kind": None,
            "last_client_work_at": None,
            "last_client_work_at_epoch": None,
            "messages_in": 0,
            "messages_out": 0,
            **metadata,
        }
    return connection_id


def _socket_log_paths(worker_id: str) -> tuple[Path, Path, Path]:
    current = _socket_worker_log_dir / f"{worker_id}.jsonl"
    return (
        current.with_name(f"{worker_id}.first.jsonl"),
        current,
        current.with_name(f"{worker_id}.1.jsonl"),
    )


def _worker_start_prompt(worker_id: str, first: Path) -> tuple[str | None, str]:
    manager = _copilot_api.get_manager()
    if manager is not None:
        try:
            instance = manager.get(worker_id)
        except _copilot_api.CopilotInstanceError:
            instance = None
        config = instance.get("config") if isinstance(instance, dict) else None
        prompt = config.get("system_prompt") if isinstance(config, dict) else None
        if isinstance(prompt, str) and prompt:
            return prompt, "managed-config"
    if first.is_file():
        for raw_line in first.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines():
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            frame = record.get("frame") if isinstance(record, dict) else None
            prompt = (
                frame.get("startup_prompt")
                if isinstance(frame, dict)
                and frame.get("type") == "register"
                else None
            )
            if isinstance(prompt, str) and prompt:
                return prompt, "worker-registration"
    return None, "unavailable"


def _nanosecond_decimal(value: int) -> str:
    return f"{value // 1_000_000_000}.{value % 1_000_000_000:09d}"


def _socket_log_clock_fields() -> dict[str, Any]:
    from datetime import datetime, timezone

    wall_ns = time.time_ns()
    precision_ns = time.perf_counter_ns()
    return {
        "timestamp": datetime.fromtimestamp(
            wall_ns / 1_000_000_000,
            tz=timezone.utc,
        ).isoformat(timespec="microseconds"),
        "timestamp_epoch_decimal": _nanosecond_decimal(wall_ns),
        "precision_clock_decimal": _nanosecond_decimal(precision_ns),
        "precision_clock_ns": precision_ns,
    }


def _socket_media_extension(mime_type: str) -> str:
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/ogg": ".ogg",
        "audio/flac": ".flac",
        "audio/mp4": ".m4a",
        "audio/webm": ".webm",
    }.get(mime_type.lower(), ".bin")


def _socket_worker_media_dir(worker_id: str) -> Path:
    return _socket_worker_log_dir / "media" / worker_id


def _prune_socket_worker_media(directory: Path) -> None:
    files = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and not path.name.endswith(".tmp")
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    retained_bytes = 0
    for path in files:
        size = path.stat().st_size
        if retained_bytes + size <= _socket_worker_media_total_bytes:
            retained_bytes += size
            continue
        path.unlink(missing_ok=True)


def _store_socket_media(
    worker_id: str,
    encoded: str,
    kind: str,
    mime_type: str,
    source_field: str,
) -> dict[str, Any]:
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return {
            "kind": kind,
            "mime_type": mime_type,
            "source_field": source_field,
            "available": False,
            "error": "invalid base64",
        }
    if not data:
        return {
            "kind": kind,
            "mime_type": mime_type,
            "source_field": source_field,
            "available": False,
            "error": "empty media",
        }
    if len(data) > _socket_worker_media_file_bytes:
        return {
            "kind": kind,
            "mime_type": mime_type,
            "source_field": source_field,
            "available": False,
            "bytes": len(data),
            "error": (
                f"media exceeds {_socket_worker_media_file_bytes}-byte "
                "preview limit"
            ),
        }
    artifact_id = hashlib.sha256(data).hexdigest()
    extension = _socket_media_extension(mime_type)
    filename = f"{artifact_id}{extension}"
    directory = _socket_worker_media_dir(worker_id)
    path = directory / filename
    with _socket_worker_log_lock:
        directory.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            temporary = directory / f".{filename}.{uuid.uuid4().hex}.tmp"
            try:
                temporary.write_bytes(data)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            os.utime(path, None)
        _prune_socket_worker_media(directory)
    return {
        "artifact_id": artifact_id,
        "kind": kind,
        "mime_type": mime_type,
        "bytes": len(data),
        "source_field": source_field,
        "available": path.is_file(),
        "url": (
            f"/emullm/admin/websockets/{worker_id}/media/{filename}"
            if path.is_file()
            else None
        ),
    }


def _socket_log_media(
    worker_id: str,
    value: Any,
    *,
    path: str = "",
    inherited_mime: str | None = None,
    depth: int = 0,
) -> list[dict[str, Any]]:
    if depth > 6:
        return []
    if isinstance(value, dict):
        mime_type = str(
            value.get("mime")
            or value.get("mime_type")
            or inherited_mime
            or ""
        ).lower()
        media: list[dict[str, Any]] = []
        for key, child in list(value.items())[:128]:
            child_path = f"{path}.{key}" if path else str(key)
            kind = (
                "image"
                if key in {"image_b64", "b64_json"}
                else ("audio" if key == "audio_b64" else None)
            )
            if kind is not None and isinstance(child, str):
                resolved_mime = mime_type or (
                    "image/png" if kind == "image" else "audio/wav"
                )
                if resolved_mime.startswith(f"{kind}/"):
                    media.append(
                        _store_socket_media(
                            worker_id,
                            child,
                            kind,
                            resolved_mime,
                            child_path,
                        )
                    )
                continue
            media.extend(
                _socket_log_media(
                    worker_id,
                    child,
                    path=child_path,
                    inherited_mime=mime_type or inherited_mime,
                    depth=depth + 1,
                )
            )
        return media
    if isinstance(value, (list, tuple)):
        media = []
        for index, child in enumerate(value[:64]):
            media.extend(
                _socket_log_media(
                    worker_id,
                    child,
                    path=f"{path}[{index}]",
                    inherited_mime=inherited_mime,
                    depth=depth + 1,
                )
            )
        return media
    return []


def _socket_log_value(
    value: Any,
    *,
    key: str = "",
    depth: int = 0,
) -> Any:
    lowered = key.lower()
    if lowered in {
        "api_key",
        "authorization",
        "token",
        "access_token",
        "refresh_token",
    }:
        return "[redacted]"
    if isinstance(value, str):
        if lowered.endswith("_b64") or lowered in {"data_b64"}:
            return {"omitted_base64_characters": len(value)}
        if len(value) > 16_384:
            return value[:16_384] + f"...[truncated {len(value) - 16_384} chars]"
        return value
    if depth >= 6:
        return "[maximum depth]"
    if isinstance(value, dict):
        items = list(value.items())
        result = {
            str(child_key): _socket_log_value(
                child_value,
                key=str(child_key),
                depth=depth + 1,
            )
            for child_key, child_value in items[:128]
        }
        if len(items) > 128:
            result["_omitted_fields"] = len(items) - 128
        return result
    if isinstance(value, (list, tuple)):
        result = [
            _socket_log_value(item, depth=depth + 1)
            for item in value[:64]
        ]
        if len(value) > 64:
            result.append({"omitted_items": len(value) - 64})
        return result
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def _append_socket_worker_log(
    worker_id: str,
    connection_id: str,
    direction: str,
    payload: Any,
) -> tuple[Path, int]:
    source, target = (
        ("EMULLM", worker_id)
        if direction == "outbound"
        else (
            (worker_id, "EMULLM")
            if direction == "inbound"
            else ("SYSTEM", worker_id)
        )
    )
    record = {
        **_socket_log_clock_fields(),
        "worker_id": worker_id,
        "connection_id": connection_id,
        "direction": direction,
        "from": source,
        "sender": source,
        "frame": _socket_log_value(payload),
    }
    media = _socket_log_media(worker_id, payload)
    if media:
        record["media"] = media
    line = (
        json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if len(line) > _socket_worker_log_segment_bytes:
        record["frame"] = {
            "type": payload.get("type") if isinstance(payload, dict) else None,
            "id": payload.get("id") if isinstance(payload, dict) else None,
            "omitted_serialized_bytes": len(line),
        }
        line = (
            json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
    first, current, previous = _socket_log_paths(worker_id)
    with _socket_worker_log_lock:
        current.parent.mkdir(parents=True, exist_ok=True)
        first_bytes = first.stat().st_size if first.is_file() else 0
        tail_started = current.is_file() or previous.is_file()
        if (
            not tail_started
            and first_bytes + len(line) <= _socket_worker_log_segment_bytes
        ):
            with first.open("ab") as stream:
                stream.write(line)
        else:
            current_bytes = current.stat().st_size if current.is_file() else 0
            if current_bytes + len(line) > _socket_worker_log_segment_bytes:
                previous.unlink(missing_ok=True)
                if current.is_file():
                    os.replace(current, previous)
            with current.open("ab") as stream:
                stream.write(line)
        total_bytes = sum(
            path.stat().st_size
            for path in (first, previous, current)
            if path.is_file()
        )
    return current, total_bytes


def _track_worker_socket_frame(
    connection_id: str,
    direction: str,
    payload: Any,
) -> None:
    with _active_websockets_lock:
        connection = _active_websockets.get(connection_id)
        worker_id = (
            str(connection.get("worker_id") or "")
            if connection is not None and connection.get("kind") == "worker"
            else ""
        )
    if not worker_id:
        return
    try:
        current, total_bytes = _append_socket_worker_log(
            worker_id,
            connection_id,
            direction,
            payload,
        )
    except OSError as error:
        _LOGGER.warning("could not write worker socket log for %s: %s", worker_id, error)
        _update_active_websocket(connection_id, log_error=str(error))
        return
    _update_active_websocket(
        connection_id,
        log_path=str(current),
        log_url=(
            "/emullm/admin/websockets/"
            f"{worker_id}/log"
        ),
        log_bytes=total_bytes,
        log_limit_bytes=3 * _socket_worker_log_segment_bytes,
        log_error=None,
    )


def _update_active_websocket(connection_id: str, **metadata: Any) -> None:
    with _active_websockets_lock:
        if connection_id in _active_websockets:
            _active_websockets[connection_id].update(metadata)


def _remove_active_websocket(connection_id: str) -> None:
    with _active_websockets_lock:
        _active_websockets.pop(connection_id, None)


def _increment_active_websocket(connection_id: str | None, field: str) -> None:
    if connection_id is None:
        return
    with _active_websockets_lock:
        if connection_id in _active_websockets:
            _active_websockets[connection_id][field] += 1


def _mark_active_websocket_satisfied(
    connection_id: str | None,
    *,
    kind: str,
    client_work: bool = False,
) -> None:
    if connection_id is None:
        return
    now = time.time()
    updates = {
        "last_satisfied_at": _now_iso(),
        "last_satisfied_at_epoch": now,
        "last_satisfied_kind": kind,
    }
    if client_work:
        updates.update(
            last_client_work_at=updates["last_satisfied_at"],
            last_client_work_at_epoch=now,
        )
    _update_active_websocket(
        connection_id,
        **updates,
    )


async def _tracked_ws_send_json(
    websocket: WebSocket, connection_id: str, payload: Any
) -> None:
    await websocket.send_json(payload)
    _increment_active_websocket(connection_id, "messages_out")
    _track_worker_socket_frame(connection_id, "outbound", payload)


async def _tracked_ws_receive_json(
    websocket: WebSocket, connection_id: str
) -> Any:
    payload = await websocket.receive_json()
    _increment_active_websocket(connection_id, "messages_in")
    _track_worker_socket_frame(connection_id, "inbound", payload)
    return payload


async def _send_worker_json(
    worker_id: str,
    websocket: Any,
    payload: dict[str, Any],
) -> None:
    connection_id = _worker_connection_ids.get(worker_id)
    if connection_id is None:
        await websocket.send_json(payload)
        return
    await _tracked_ws_send_json(websocket, connection_id, payload)


def _active_websocket_rows() -> list[dict[str, Any]]:
    now = time.time()
    with _active_websockets_lock:
        rows = []
        for value in _active_websockets.values():
            row = {
                key: item
                for key, item in value.items()
                if not key.endswith("_epoch")
            }
            row["connected_seconds"] = round(now - float(value["connected_at_epoch"]), 1)
            satisfied_at = value.get("last_satisfied_at_epoch")
            row["last_satisfied_seconds"] = (
                round(max(0.0, now - float(satisfied_at)), 1)
                if satisfied_at is not None
                else None
            )
            client_work_at = value.get("last_client_work_at_epoch")
            row["last_client_work_seconds"] = (
                round(max(0.0, now - float(client_work_at)), 1)
                if client_work_at is not None
                else None
            )
            rows.append(row)
    return sorted(rows, key=lambda row: (row["kind"], row["endpoint"], row["connection_id"]))


def _openai_client_key(request: Request) -> tuple[str, str, str | None, str]:
    client = request.client
    host = client.host if client is not None else "unknown"
    declared_id = (
        request.headers.get("x-emullm-client-id")
        or request.headers.get("x-client-id")
        or request.headers.get("x-session-id")
    )
    declared_id = declared_id.strip()[:200] if declared_id else None
    user_agent = (request.headers.get("user-agent") or "unknown").strip()[:500]
    identity = f"id:{declared_id}" if declared_id else f"agent:{user_agent}"
    digest = hashlib.sha256(f"{host}\0{identity}".encode("utf-8")).hexdigest()[:16]
    return f"client-{digest}", host, declared_id, user_agent


def _client_worker_capacity() -> tuple[int, int]:
    connected = len(_connected_workers)
    reserve = min(
        connected,
        max(
            _CLIENT_WORKER_RESERVE_MIN,
            math.ceil(connected * _CLIENT_WORKER_RESERVE_FRACTION),
        ),
    )
    return max(1, min(_max_concurrent_calls, connected - reserve)), reserve


async def _wait_for_client_worker_capacity(request: Request) -> None:
    if (
        request.method != "POST"
        or request.url.path.rstrip("/") not in _WORKER_CAPACITY_PATHS
    ):
        return
    client_id, *_ = _openai_client_key(request)
    deadline = time.monotonic() + _REQUEST_TIMEOUT_SECONDS
    waiting = False
    try:
        while True:
            limit, reserve = _client_worker_capacity()
            with _openai_clients_lock:
                active = int(
                    (_openai_clients.get(client_id) or {}).get(
                        "active_requests",
                        0,
                    )
                )
                if active < limit:
                    return
                if not waiting:
                    _client_capacity_waiters[client_id] = (
                        _client_capacity_waiters.get(client_id, 0) + 1
                    )
                    waiting = True
            if time.monotonic() >= deadline:
                raise HTTPException(
                    status_code=504,
                    detail=(
                        f"client capacity wait timed out; maximum {limit} active "
                        f"request(s), {reserve} worker(s) reserved"
                    ),
                )
            await asyncio.sleep(0.1)
    finally:
        if waiting:
            with _openai_clients_lock:
                remaining = _client_capacity_waiters.get(client_id, 0) - 1
                if remaining > 0:
                    _client_capacity_waiters[client_id] = remaining
                else:
                    _client_capacity_waiters.pop(client_id, None)


def _begin_openai_client_request(
    request: Request,
) -> tuple[str | None, str | None]:
    now = time.time()
    timestamp = _now_iso()
    client_id, host, declared_id, user_agent = _openai_client_key(request)
    client = request.client
    with _openai_clients_lock:
        if client_id not in _openai_clients and len(_openai_clients) >= _MAX_OPENAI_CLIENTS:
            inactive = [
                (key, value)
                for key, value in _openai_clients.items()
                if not value["active_requests"]
            ]
            if inactive:
                oldest, _ = min(
                    inactive,
                    key=lambda item: float(item[1]["last_seen_at_epoch"]),
                )
                _openai_clients.pop(oldest, None)
            else:
                return None, None
        row = _openai_clients.setdefault(
            client_id,
            {
                "client_id": client_id,
                "host": host,
                "declared_id": declared_id,
                "user_agent": user_agent,
                "first_seen_at": timestamp,
                "first_seen_at_epoch": now,
                "last_seen_at": timestamp,
                "last_seen_at_epoch": now,
                "last_completed_at": None,
                "last_completed_at_epoch": None,
                "active_requests": 0,
                "requests": 0,
                "last_status": None,
                "last_endpoint": request.url.path,
                "last_method": request.method,
                "last_port": client.port if client is not None else None,
            },
        )
        row.update(
            host=host,
            declared_id=declared_id,
            user_agent=user_agent,
            last_seen_at=timestamp,
            last_seen_at_epoch=now,
            last_endpoint=request.url.path,
            last_method=request.method,
            last_port=client.port if client is not None else None,
        )
        row["active_requests"] += 1
        row["requests"] += 1
        request_id: str | None = None
        if len(_openai_requests) >= _MAX_OPENAI_REQUESTS:
            completed = [
                (key, value)
                for key, value in _openai_requests.items()
                if not value["active"]
            ]
            if completed:
                oldest, _ = min(
                    completed,
                    key=lambda item: float(item[1]["started_at_epoch"]),
                )
                _openai_requests.pop(oldest, None)
        if len(_openai_requests) < _MAX_OPENAI_REQUESTS:
            request_id = f"http-{uuid.uuid4().hex[:16]}"
            _openai_requests[request_id] = {
                "request_id": request_id,
                "external_request_id": (
                    request.headers.get("x-request-id")
                    or request.headers.get("openai-request-id")
                ),
                "client_id": client_id,
                "declared_client_id": declared_id,
                "host": host,
                "user_agent": user_agent,
                "method": request.method,
                "endpoint": request.url.path,
                "started_at": timestamp,
                "started_at_epoch": now,
                "completed_at": None,
                "completed_at_epoch": None,
                "duration_seconds": None,
                "status": None,
                "active": True,
            }
    return client_id, request_id


def _finish_openai_client_request(
    request_token: tuple[str | None, str | None],
    status_code: int,
) -> None:
    client_id, request_id = request_token
    if client_id is None:
        return
    now = time.time()
    with _openai_clients_lock:
        row = _openai_clients.get(client_id)
        if row is None:
            return
        row["active_requests"] = max(0, int(row["active_requests"]) - 1)
        row["last_completed_at"] = _now_iso()
        row["last_completed_at_epoch"] = now
        row["last_status"] = status_code
        request_row = (
            _openai_requests.get(request_id)
            if request_id is not None
            else None
        )
        if request_row is not None:
            request_row["completed_at"] = _now_iso()
            request_row["completed_at_epoch"] = now
            request_row["duration_seconds"] = round(
                max(0.0, now - float(request_row["started_at_epoch"])),
                3,
            )
            request_row["status"] = status_code
            request_row["active"] = False


def _openai_client_rows() -> list[dict[str, Any]]:
    now = time.time()
    with _openai_clients_lock:
        rows = []
        for value in _openai_clients.values():
            row = {
                key: item
                for key, item in value.items()
                if not key.endswith("_epoch")
            }
            row["first_seen_seconds"] = round(
                max(0.0, now - float(value["first_seen_at_epoch"])),
                1,
            )
            row["last_seen_seconds"] = round(
                max(0.0, now - float(value["last_seen_at_epoch"])),
                1,
            )
            completed_at = value.get("last_completed_at_epoch")
            row["last_completed_seconds"] = (
                round(max(0.0, now - float(completed_at)), 1)
                if completed_at is not None
                else None
            )
            row["connected"] = bool(row["active_requests"])
            rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            not row["connected"],
            float(row["last_seen_seconds"]),
            row["client_id"],
        ),
    )


def _openai_request_rows() -> list[dict[str, Any]]:
    now = time.time()
    with _openai_clients_lock:
        sortable_rows = []
        for value in _openai_requests.values():
            row = {
                key: item
                for key, item in value.items()
                if not key.endswith("_epoch")
            }
            row["age_seconds"] = round(
                max(0.0, now - float(value["started_at_epoch"])),
                1,
            )
            if row["active"]:
                row["duration_seconds"] = row["age_seconds"]
            sortable_rows.append((float(value["started_at_epoch"]), row))
    sortable_rows.sort(
        key=lambda item: (
            not item[1]["active"],
            -item[0],
            item[1]["request_id"],
        )
    )
    return [row for _, row in sortable_rows]

# Each worker declares its OWN model list on connect (see the websocket
# handshake below): a dict of suffix -> {"display_name", "instruction"}.
# /v1/models aggregates these across every currently connected worker. A
# worker_id with no declared list yet falls back to _PERSONA_SUFFIXES
# below, so a bare-bones/older worker still gets a sensible default menu
# without having to declare anything itself.
_worker_models: dict[str, dict[str, dict[str, Any]]] = {}
_worker_kinds: dict[str, str] = {}
_worker_runtime_models: dict[str, str] = {}
_worker_descriptions: dict[str, str] = {}
_worker_model_switch_stats: dict[str, dict[str, Any]] = {}

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
_COPILOT_POOL_WORKER_ID = "worker-copilot-n"
_COPILOT_AFFINITY_PORT_RANGE = 1024
_copilot_client_affinity: dict[str, dict[str, Any]] = {}
_copilot_client_affinity_lock = threading.RLock()
_MAX_COPILOT_CLIENT_AFFINITIES = 5_000

# A missing entry means the worker accepts all models. A present empty tuple
# means it intentionally accepts none; non-empty entries are glob patterns.
_worker_model_masks: dict[str, tuple[str, ...]] = {}

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
_EXPORTED_PERSONA_SUFFIXES = {"percent125", "percent100", "percent25"}
_HIDDEN_PERSONA_SUFFIXES = {"same", "percent75", "percent10"}
_DEFAULT_MODEL_ID = f"{_COPILOT_POOL_WORKER_ID}/percent100"
_EMULLM_DEFAULT_MODEL_ID = "emullm/default"
_CAPABILITY_MODEL_ALIASES: dict[str, set[str]] = {
    "audio": {"audio_input"},
    "video": {"vision_input", "file_input"},
    "vision": {"vision_input"},
    "file": {"file_input"},
    "code": {"code"},
    "summarization": {"summarization"},
    "image-generation": {"image_generation"},
    "image-output": {"image_output"},
}


def _catalog_model_is_visible(model_id: str) -> bool:
    worker_id, _, suffix = model_id.partition("/")
    return (
        suffix not in _HIDDEN_PERSONA_SUFFIXES
        and not _numeric_copilot_worker_id(worker_id)
    )


def _numeric_copilot_worker_id(worker_id: str) -> bool:
    return bool(re.fullmatch(r"worker-copilot-[1-9][0-9]*", worker_id))


def _capability_alias_spec(
    model_id: str,
) -> tuple[str, str, set[str]] | None:
    prefix = "router/"
    if not model_id.startswith(prefix):
        return None
    alias = model_id[len(prefix) :]
    selector = next(
        (
            candidate
            for candidate in ("best", "worse")
            if alias.endswith(f"-{candidate}")
        ),
        None,
    )
    if selector is None:
        return None
    capability = alias[: -(len(selector) + 1)]
    if capability not in _CAPABILITY_MODEL_ALIASES:
        return None
    return selector, capability, set(_CAPABILITY_MODEL_ALIASES[capability])


def _worker_quality_rank(worker_id: str) -> int:
    backing_model = _worker_runtime_models.get(worker_id)
    metadata = (
        _copilot_model_metadata(backing_model)
        if backing_model
        else None
    )
    rank = metadata.get("quality_rank") if isinstance(metadata, dict) else None
    return int(rank) if isinstance(rank, int) and rank > 0 else 10_000


def _capability_alias_worker(model_id: str) -> str:
    spec = _capability_alias_spec(model_id)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"unknown capability alias '{model_id}'")
    selector, capability, required = spec
    candidates = [
        worker_id
        for worker_id in _connected_workers
        if _worker_ready_for_offer(worker_id)
        if all(
            _worker_capabilities.get(worker_id, {}).get(name) is True
            for name in required
        )
    ]
    if not candidates:
        raise HTTPException(
            status_code=503,
            detail=(
                f"no connected worker explicitly advertises capability "
                f"'{capability}'"
            ),
        )
    candidates.sort(
        key=lambda worker_id: (
            _worker_quality_rank(worker_id)
            if selector == "best"
            else -_worker_quality_rank(worker_id),
            _worker_load(worker_id),
            worker_id,
        )
    )
    worker_id = candidates[0]
    _request_assigned_worker.set(worker_id)
    return worker_id


def _copilot_pool_workers() -> list[str]:
    connected = sorted(
        worker_id
        for worker_id in _connected_workers
        if _numeric_copilot_worker_id(worker_id)
    )
    if connected:
        return connected
    manager = _copilot_api.get_manager()
    if manager is not None:
        configured = sorted(
            str(instance.get("worker_id") or "")
            for instance in manager.list()
            if _numeric_copilot_worker_id(
                str(instance.get("worker_id") or "")
            )
        )
        if configured:
            return configured
    return ["worker-copilot-1"]


def _copilot_pool_model_entry(
    suffix: str,
    persona: dict[str, Any],
) -> dict[str, Any]:
    entry = _model_entry(_COPILOT_POOL_WORKER_ID, suffix, persona)
    workers = _copilot_pool_workers()
    entry.update(
        display_name=f"Copilot pool N · {persona.get('display_name') or suffix}",
        description=(
            "Client-affine Copilot pool alias. Literal n is assigned by client "
            "IP and 1024-port source range; replace n with any positive integer "
            "to address or create that exact worker."
        ),
        owned_by="emullm-copilot-pool",
        connected=any(worker in _connected_workers for worker in workers),
        active_workers=[
            worker for worker in workers if worker in _connected_workers
        ],
        routing_mode="client_ip_port_range_affinity",
        affinity_port_range_size=_COPILOT_AFFINITY_PORT_RANGE,
        assigned_worker_header="X-EmuLLM-Worker-ID",
    )
    return entry


def _capability_alias_model_entry(
    selector: str,
    capability: str,
    required: set[str],
) -> dict[str, Any]:
    capable_workers = [
        worker_id
        for worker_id in _connected_workers
        if all(
            _worker_capabilities.get(worker_id, {}).get(name) is True
            for name in required
        )
    ]
    return {
        "id": f"router/{capability}-{selector}",
        "object": "model",
        "display_name": f"{selector.title()} · {capability}",
        "description": (
            f"Routes to the {selector} quality-ranked connected worker that "
            f"explicitly advertises: {', '.join(sorted(required))}."
        ),
        "owned_by": "emullm-capability-router",
        "connected": bool(capable_workers),
        "active_workers": sorted(capable_workers),
        "selection": selector,
        "capability_alias": capability,
        "required_capabilities": sorted(required),
        "assigned_worker_header": "X-EmuLLM-Worker-ID",
        "simulated": False,
    }


def _assign_copilot_pool_worker() -> str:
    candidates = _copilot_pool_workers()
    affinity = _request_affinity.get()
    if affinity is None:
        index = _round_robin_state.get(_COPILOT_POOL_WORKER_ID, 0)
        worker_id = candidates[index % len(candidates)]
        _round_robin_state[_COPILOT_POOL_WORKER_ID] = index + 1
    else:
        key = (
            f"{affinity['host']}:{affinity['port_start']}-"
            f"{affinity['port_end']}"
        )
        with _copilot_client_affinity_lock:
            existing = _copilot_client_affinity.get(key)
            if existing and existing.get("worker_id") in candidates:
                worker_id = str(existing["worker_id"])
            else:
                digest = hashlib.sha256(key.encode("utf-8")).digest()
                worker_id = candidates[
                    int.from_bytes(digest[:8], "big") % len(candidates)
                ]
                if len(_copilot_client_affinity) >= _MAX_COPILOT_CLIENT_AFFINITIES:
                    oldest = min(
                        _copilot_client_affinity,
                        key=lambda item: float(
                            _copilot_client_affinity[item]["last_used_at"]
                        ),
                    )
                    _copilot_client_affinity.pop(oldest, None)
            _copilot_client_affinity[key] = {
                "worker_id": worker_id,
                "host": affinity["host"],
                "port_start": affinity["port_start"],
                "port_end": affinity["port_end"],
                "last_used_at": time.time(),
            }
    _request_assigned_worker.set(worker_id)
    return worker_id


def _split_model_id(model: str) -> tuple[str, str]:
    """"<worker_id>/<suffix>" -> (worker_id, suffix); a bare id with no
    "/" retains the legacy "same" parsing so opaque model IDs pass through."""
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


def _normalise_model_masks(value: Any) -> tuple[str, ...] | None:
    """Normalize comma-delimited or JSON-list model glob patterns."""
    if value is None:
        return None
    raw_values = value if isinstance(value, (list, tuple, set)) else [value]
    masks: list[str] = []
    for raw_value in raw_values:
        if not isinstance(raw_value, str):
            continue
        masks.extend(mask.strip() for mask in raw_value.split(",") if mask.strip())
    return tuple(dict.fromkeys(masks))


def _set_worker_model_masks(worker_id: str, masks: tuple[str, ...] | None) -> None:
    if masks is None:
        _worker_model_masks.pop(worker_id, None)
    else:
        _worker_model_masks[worker_id] = masks


def _new_automatic_worker_id() -> str:
    """Create a collision-resistant mailbox-safe identity for an anonymous socket."""
    while True:
        worker_id = f"worker-unknown-{uuid.uuid4().hex[:16]}"
        if worker_id not in _connected_workers:
            return worker_id


def _worker_load(worker_id: str) -> int:
    with _worker_load_lock:
        return (
            _worker_inflight.get(worker_id, 0)
            + _worker_reservations.get(worker_id, 0)
        )


def _reserve_worker(worker_id: str) -> None:
    with _worker_load_lock:
        total = sum(_worker_inflight.values()) + sum(_worker_reservations.values())
        if total >= _max_concurrent_calls:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"maximum simultaneous call limit "
                    f"({_max_concurrent_calls}) reached"
                ),
                headers={"Retry-After": "1"},
            )
        _worker_reservations[worker_id] = (
            _worker_reservations.get(worker_id, 0) + 1
        )


def _begin_worker_request(worker_id: str, model: str) -> None:
    with _worker_load_lock:
        reservations = _worker_reservations.get(worker_id, 0)
        if reservations > 0:
            if reservations == 1:
                _worker_reservations.pop(worker_id, None)
            else:
                _worker_reservations[worker_id] = reservations - 1
        else:
            total = sum(_worker_inflight.values()) + sum(
                _worker_reservations.values()
            )
            if total >= _max_concurrent_calls:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"maximum simultaneous call limit "
                        f"({_max_concurrent_calls}) reached"
                    ),
                    headers={"Retry-After": "1"},
                )
        _worker_inflight[worker_id] = _worker_inflight.get(worker_id, 0) + 1
        _model_inflight[model] = _model_inflight.get(model, 0) + 1


def _end_worker_request(worker_id: str, model: str) -> None:
    with _worker_load_lock:
        inflight = _worker_inflight.get(worker_id, 0)
        if inflight <= 1:
            _worker_inflight.pop(worker_id, None)
        else:
            _worker_inflight[worker_id] = inflight - 1
        model_inflight = _model_inflight.get(model, 0)
        if model_inflight <= 1:
            _model_inflight.pop(model, None)
        else:
            _model_inflight[model] = model_inflight - 1
        _worker_last_busy_at[worker_id] = time.monotonic()


def _release_worker_reservation(worker_id: str) -> None:
    with _worker_load_lock:
        reservations = _worker_reservations.get(worker_id, 0)
        if reservations <= 1:
            _worker_reservations.pop(worker_id, None)
        else:
            _worker_reservations[worker_id] = reservations - 1


def _worker_is_idle(worker_id: str) -> bool:
    if _worker_load(worker_id) > 0:
        return False
    last_busy = _worker_last_busy_at.get(worker_id)
    return (
        last_busy is None
        or time.monotonic() - last_busy >= _idle_grace_seconds
    )


def _mark_waiting_for_worker(
    wait_id: str,
    worker_id: str,
    model: str,
    reason: str,
) -> None:
    with _waiting_for_worker_lock:
        _waiting_for_worker.setdefault(
            wait_id,
            {
                "id": wait_id,
                "worker_id": worker_id,
                "model": model,
                "reason": reason,
                "started_at": time.time(),
            },
        )


def _clear_waiting_for_worker(wait_id: str) -> None:
    with _waiting_for_worker_lock:
        _waiting_for_worker.pop(wait_id, None)


def _waiting_for_worker_snapshot() -> list[dict[str, Any]]:
    now = time.time()
    with _waiting_for_worker_lock:
        return [
            {
                **entry,
                "waiting_seconds": round(
                    max(0.0, now - float(entry["started_at"])),
                    3,
                ),
            }
            for entry in _waiting_for_worker.values()
        ]


def _record_worker_service(
    worker_id: str,
    model: str,
    service_kind: str,
    duration_seconds: float,
    outcome: str,
) -> None:
    with _worker_load_lock:
        worker = _worker_service_stats.setdefault(worker_id, {})
        stats = worker.setdefault(
            service_kind,
            {
                "attempts": 0,
                "served": 0,
                "failed": 0,
                "rejected": 0,
                "deferred": 0,
                "cancelled": 0,
                "total_seconds": 0.0,
            },
        )
        stats["attempts"] = int(stats["attempts"]) + 1
        key = outcome if outcome in stats else "failed"
        stats[key] = int(stats[key]) + 1
        stats["total_seconds"] = (
            float(stats["total_seconds"]) + duration_seconds
        )
        model_services = _model_service_stats.setdefault(model, {})
        model_stats = model_services.setdefault(
            service_kind,
            {
                "attempts": 0,
                "served": 0,
                "failed": 0,
                "rejected": 0,
                "deferred": 0,
                "cancelled": 0,
                "total_seconds": 0.0,
            },
        )
        model_stats["attempts"] = int(model_stats["attempts"]) + 1
        model_stats[key] = int(model_stats[key]) + 1
        model_stats["total_seconds"] = (
            float(model_stats["total_seconds"]) + duration_seconds
        )


def _service_stats_snapshot() -> tuple[dict[str, Any], dict[str, Any]]:
    workers: dict[str, Any] = {}
    team_services: dict[str, dict[str, float | int]] = {}
    with _worker_load_lock:
        for worker_id, service_map in _worker_service_stats.items():
            worker_services: dict[str, Any] = {}
            for service_kind, values in service_map.items():
                stats = dict(values)
                attempts = int(stats["attempts"])
                stats["average_seconds"] = round(
                    float(stats["total_seconds"]) / attempts,
                    3,
                ) if attempts else 0.0
                stats["total_seconds"] = round(
                    float(stats["total_seconds"]),
                    3,
                )
                worker_services[service_kind] = stats
                team = team_services.setdefault(
                    service_kind,
                    {
                        "attempts": 0,
                        "served": 0,
                        "failed": 0,
                        "rejected": 0,
                        "deferred": 0,
                        "cancelled": 0,
                        "total_seconds": 0.0,
                    },
                )
                for key in (
                    "attempts",
                    "served",
                    "failed",
                    "rejected",
                    "deferred",
                    "cancelled",
                ):
                    team[key] = int(team[key]) + int(stats[key])
                team["total_seconds"] = (
                    float(team["total_seconds"])
                    + float(stats["total_seconds"])
                )
            workers[worker_id] = {
                "kind": (
                    "backend"
                    if worker_id.startswith("backend-")
                    else _worker_kinds.get(worker_id, "worker")
                ),
                "active": _worker_inflight.get(worker_id, 0),
                "reserved": _worker_reservations.get(worker_id, 0),
                "services": worker_services,
            }
        for worker_id in (
            set(_worker_inflight)
            | set(_worker_reservations)
            | set(_worker_model_switch_stats)
        ):
            workers.setdefault(
                worker_id,
                {
                    "kind": (
                        "backend"
                        if worker_id.startswith("backend-")
                        else _worker_kinds.get(worker_id, "worker")
                    ),
                    "active": _worker_inflight.get(worker_id, 0),
                    "reserved": _worker_reservations.get(worker_id, 0),
                    "services": {},
                },
            )
        for service_kind, stats in team_services.items():
            attempts = int(stats["attempts"])
            stats["average_seconds"] = round(
                float(stats["total_seconds"]) / attempts,
                3,
            ) if attempts else 0.0
            stats["total_seconds"] = round(
                float(stats["total_seconds"]),
                3,
            )
        for metadata in _active_service_requests.values():
            service_kind = metadata["service_kind"]
            stats = team_services.setdefault(
                service_kind,
                {
                    "attempts": 0,
                    "served": 0,
                    "failed": 0,
                    "rejected": 0,
                    "deferred": 0,
                    "cancelled": 0,
                    "total_seconds": 0.0,
                    "average_seconds": 0.0,
                },
            )
            stats["active"] = int(stats.get("active", 0)) + 1
        for stats in team_services.values():
            stats.setdefault("active", 0)
    models: dict[str, Any] = {}
    with _worker_load_lock:
        for model, service_map in _model_service_stats.items():
            model_services: dict[str, Any] = {}
            totals = {
                key: 0
                for key in (
                    "attempts",
                    "served",
                    "failed",
                    "rejected",
                    "deferred",
                    "cancelled",
                )
            }
            total_seconds = 0.0
            for service_kind, values in service_map.items():
                stats = dict(values)
                attempts = int(stats["attempts"])
                stats["average_seconds"] = round(
                    float(stats["total_seconds"]) / attempts,
                    3,
                ) if attempts else 0.0
                stats["total_seconds"] = round(
                    float(stats["total_seconds"]),
                    3,
                )
                model_services[service_kind] = stats
                for key in totals:
                    totals[key] += int(stats[key])
                total_seconds += float(stats["total_seconds"])
            totals["total_seconds"] = round(total_seconds, 3)
            totals["average_seconds"] = round(
                total_seconds / totals["attempts"],
                3,
            ) if totals["attempts"] else 0.0
            models[model] = {
                "active": _model_inflight.get(model, 0),
                "totals": totals,
                "services": model_services,
            }
        for model, active in _model_inflight.items():
            models.setdefault(
                model,
                {
                    "active": active,
                    "totals": {
                        "attempts": 0,
                        "served": 0,
                        "failed": 0,
                        "rejected": 0,
                        "deferred": 0,
                        "cancelled": 0,
                        "total_seconds": 0.0,
                        "average_seconds": 0.0,
                    },
                    "services": {},
                },
            )

    team_totals = {
        key: sum(int(stats[key]) for stats in team_services.values())
        for key in (
            "attempts",
            "served",
            "failed",
            "rejected",
            "deferred",
            "cancelled",
        )
    }
    team_total_seconds = sum(
        float(stats["total_seconds"]) for stats in team_services.values()
    )
    team_totals["total_seconds"] = round(team_total_seconds, 3)
    team_totals["average_seconds"] = round(
        team_total_seconds / team_totals["attempts"],
        3,
    ) if team_totals["attempts"] else 0.0
    return workers, {
        "totals": team_totals,
        "services": team_services,
        "models": models,
    }


def _worker_retry_delay(worker_id: str) -> float:
    del worker_id
    return 0.0


def _worker_ready_for_offer(worker_id: str) -> bool:
    return _worker_retry_delay(worker_id) <= 0


def _capability_ordered_workers(
    worker_ids: list[str],
    required_capabilities: set[str] | None,
) -> list[str]:
    """Prefer declared-capable workers, retain unknowns, and skip opt-outs."""
    worker_ids = [
        worker_id
        for worker_id in worker_ids
        if _worker_ready_for_offer(worker_id)
    ]
    if not required_capabilities:
        return sorted(worker_ids, key=_worker_load)
    capable: list[str] = []
    unknown: list[str] = []
    for worker_id in worker_ids:
        declared = _worker_capabilities.get(worker_id, {})
        states = [declared.get(capability) for capability in required_capabilities]
        if any(state is False for state in states):
            continue
        if all(state is True for state in states):
            capable.append(worker_id)
        else:
            unknown.append(worker_id)
    capable.sort(key=_worker_load)
    unknown.sort(key=_worker_load)
    return capable + unknown


def _worker_candidates_for_model(
    model: str,
    resolved_worker_id: str,
    required_capabilities: set[str] | None = None,
) -> list[str]:
    """Return live workers eligible to accept a model offer, in try order."""
    if isinstance(_model_routes.get(model), str):
        return _capability_ordered_workers(
            [resolved_worker_id],
            required_capabilities,
        )  # Explicit operator route always wins unless temporarily not ready.

    requested_worker_id, _ = _split_model_id(model)
    if (
        requested_worker_id == _COPILOT_POOL_WORKER_ID
        or _capability_alias_spec(model) is not None
    ):
        return _capability_ordered_workers(
            [resolved_worker_id],
            required_capabilities,
        )
    if requested_worker_id != _DEFAULT_WORKER_ID and requested_worker_id in _connected_workers:
        return _capability_ordered_workers(
            [requested_worker_id],
            required_capabilities,
        )  # Preserve explicit named-worker routing.

    matching = sorted(
        worker_id
        for worker_id in _native_worker_ids
        if worker_id in _connected_workers
        if (masks := _worker_model_masks.get(worker_id)) and any(fnmatchcase(model, mask) for mask in masks)
    )
    unmasked = sorted(
        worker_id
        for worker_id in _native_worker_ids
        if worker_id in _connected_workers and worker_id not in _worker_model_masks
    )
    ordered: list[str] = []
    for candidates, key in ((matching, "websocket-modelmasks:matching"), (unmasked, "websocket-modelmasks:all")):
        if candidates:
            _, rotated = select_from_catalog(candidates, "round-robin", key=key)
            ordered.extend(rotated)
    return _capability_ordered_workers(
        ordered or [resolved_worker_id],
        required_capabilities,
    )


def _is_known_worker_id(worker_id: str) -> bool:
    """Whether a model prefix explicitly addresses an available worker."""
    return (
        worker_id == _DEFAULT_WORKER_ID
        or worker_id in _connected_workers
        or worker_id in _worker_models
        or worker_id in _worker_capabilities
        or worker_id in _worker_roles
        or worker_id in _agent_descriptions
        or worker_id in _worker_service_behavior
        or any(
            target == worker_id
            for route in _model_routes.values()
            for target in ([route] if isinstance(route, str) else route)
        )
    )


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
_model_routes: dict[str, str | list[str]] = {}            # full model id -> worker or ordered target chain
_server_description: str | None = None                    # user-facing server description
_advertised_base: list[str] = []                          # services.models (manual base)
_advertised_default: str | None = None                    # services.model (default advertised)
_advertised_agents: list[dict[str, Any]] = []             # agents flagged to contribute their models
_model_catalog_overrides: dict[str, dict[str, Any]] = {}  # exported id -> {hidden, patch}
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


def _normalise_model_route(value: Any) -> str | list[str] | None:
    """Normalize an exact worker target or ordered worker-glob/backend chain."""
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else parts
    if isinstance(value, (list, tuple)):
        targets = list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
        return targets or None
    return None


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
    global _server_description, _advertised_default, _validation_interval
    global _validation_interval_override, _max_concurrent_calls
    global _idle_worker_target, _idle_grace_seconds
    global _backend_fallback_delay_seconds
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
    _model_catalog_overrides.clear()
    _validation_interval = _DEFAULT_VALIDATION_INTERVAL
    _validation_interval_override = None
    _max_concurrent_calls = _DEFAULT_MAX_CONCURRENT_CALLS
    _idle_worker_target = _DEFAULT_IDLE_WORKER_TARGET
    _idle_grace_seconds = _DEFAULT_IDLE_GRACE_SECONDS
    _backend_fallback_delay_seconds = _DEFAULT_BACKEND_FALLBACK_DELAY_SECONDS
    # note: _model_fetch_cache intentionally NOT cleared -- it's a daily cache
    # that should survive config reloads.


def apply_agent_policies(config: dict[str, Any]) -> None:
    """Populate per-agent service behaviors, observers, and user-facing
    descriptions from a config's ``agents`` list and server-level
    ``services`` / ``description``. Clears any prior policy first."""
    global _server_description, _advertised_default, _validation_interval
    global _validation_interval_override, _max_concurrent_calls
    global _idle_worker_target, _idle_grace_seconds
    global _backend_fallback_delay_seconds
    clear_agent_policies()
    if not isinstance(config, dict):
        return
    for _k in ("validation_interval_default", "validation_interval"):
        if config.get(_k) is not None:
            _validation_interval = config[_k]
            break
    _validation_interval_override = config.get("validation_interval_override")
    if config.get("max_concurrent_calls") is not None:
        _max_concurrent_calls = max(
            _BASELINE_COPILOT_WORKERS,
            min(
                _MAX_CONCURRENT_CALLS_LIMIT,
                int(config["max_concurrent_calls"]),
            ),
        )
    if config.get("idle_worker_target") is not None:
        _idle_worker_target = max(
            0,
            min(_max_concurrent_calls, int(config["idle_worker_target"])),
        )
    if config.get("idle_grace_seconds") is not None:
        _idle_grace_seconds = max(
            0.0,
            min(3600.0, float(config["idle_grace_seconds"])),
        )
    if config.get("backend_fallback_delay_seconds") is not None:
        _backend_fallback_delay_seconds = max(
            0.0,
            min(300.0, float(config["backend_fallback_delay_seconds"])),
        )
    if isinstance(config.get("description"), str):
        _server_description = config["description"]
    model_overrides = config.get("model_catalog_overrides")
    if isinstance(model_overrides, dict):
        for model_id, override in model_overrides.items():
            if isinstance(model_id, str) and isinstance(override, dict):
                _model_catalog_overrides[model_id] = dict(override)
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
            if agent.get("enabled") is False:
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
        by_id = {
            str(a.get("id") or a.get("worker_id") or ""): a
            for a in agents
            if isinstance(a, dict) and a.get("enabled") is not False
        }
        for agent in agents:
            if not isinstance(agent, dict):
                continue
            if agent.get("enabled") is False:
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
        for mid, route in routes.items():
            normalized = _normalise_model_route(route)
            if isinstance(mid, str) and mid and normalized:
                _model_routes[mid] = normalized


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


_MAX_V1_INLINE_IMAGES = 12
_MAX_V1_INLINE_IMAGE_BYTES = max(
    1,
    int(os.environ.get("EMULLM_V1_INLINE_IMAGE_BYTES", str(25 * 1024 * 1024))),
)
_MAX_V1_INLINE_IMAGES_TOTAL_BYTES = max(
    _MAX_V1_INLINE_IMAGE_BYTES,
    int(
        os.environ.get(
            "EMULLM_V1_INLINE_IMAGES_TOTAL_BYTES",
            str(50 * 1024 * 1024),
        )
    ),
)


def _store_relay_attachment(
    data: bytes,
    filename: str,
    mime_type: str,
    purpose: str,
) -> dict[str, Any]:
    record = _store_cloud_bytes(data, filename, purpose=purpose)
    return {
        "file_id": record["id"],
        "name": record["filename"],
        "mime_type": mime_type,
        "bytes": len(data),
        "url": _cloud_file_url(record["id"]),
    }


def _prepare_inline_image_attachments(
    image_urls: list[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Persist OpenAI/Anthropic base64 data URLs and return compact worker input."""
    if len(image_urls) > _MAX_V1_INLINE_IMAGES:
        raise HTTPException(
            status_code=413,
            detail=f"at most {_MAX_V1_INLINE_IMAGES} inline images are allowed",
        )
    normalized_urls: list[str] = []
    attachments: list[dict[str, Any]] = []
    total_bytes = 0
    for index, image_url in enumerate(image_urls, start=1):
        if not image_url.startswith("data:"):
            normalized_urls.append(image_url)
            continue
        if "," not in image_url:
            raise HTTPException(status_code=422, detail=f"inline image {index} is malformed")
        header, encoded = image_url.split(",", 1)
        parts = header[5:].split(";")
        mime_type = (parts[0] or "application/octet-stream").strip().lower()
        if mime_type == "image/jpg":
            mime_type = "image/jpeg"
        if not mime_type.startswith("image/") or "base64" not in parts[1:]:
            raise HTTPException(
                status_code=422,
                detail=f"inline image {index} must be a base64 image data URL",
            )
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as error:
            raise HTTPException(
                status_code=422,
                detail=f"inline image {index} is not valid base64",
            ) from error
        if not data:
            raise HTTPException(status_code=422, detail=f"inline image {index} is empty")
        if len(data) > _MAX_V1_INLINE_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"inline image {index} exceeds the "
                    f"{_MAX_V1_INLINE_IMAGE_BYTES}-byte limit"
                ),
            )
        total_bytes += len(data)
        if total_bytes > _MAX_V1_INLINE_IMAGES_TOTAL_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    "inline images exceed the "
                    f"{_MAX_V1_INLINE_IMAGES_TOTAL_BYTES}-byte total limit"
                ),
            )
        filename = f"input-image-{index}{_ext_for_mime(mime_type)}"
        attachment = _store_relay_attachment(
            data,
            filename,
            mime_type,
            "vision",
        )
        normalized_urls.append(attachment["url"])
        attachments.append(attachment)
    return normalized_urls, attachments


_V1_AUDIO_FORMATS = {
    "flac": ("audio/flac", ".flac"),
    "m4a": ("audio/mp4", ".m4a"),
    "mp3": ("audio/mpeg", ".mp3"),
    "ogg": ("audio/ogg", ".ogg"),
    "wav": ("audio/wav", ".wav"),
    "webm": ("audio/webm", ".webm"),
}
_MAX_V1_INLINE_AUDIO_FILES = 12
_MAX_V1_INLINE_AUDIO_BYTES = max(
    1,
    int(os.environ.get("EMULLM_V1_INLINE_AUDIO_BYTES", str(25 * 1024 * 1024))),
)
_MAX_V1_INLINE_AUDIO_TOTAL_BYTES = max(
    _MAX_V1_INLINE_AUDIO_BYTES,
    int(
        os.environ.get(
            "EMULLM_V1_INLINE_AUDIO_TOTAL_BYTES",
            str(50 * 1024 * 1024),
        )
    ),
)


def _extract_audio_inputs(content: Any) -> list[tuple[str, str]]:
    """Pull standard OpenAI input_audio base64 payloads from message content."""
    inputs: list[tuple[str, str]] = []
    if not isinstance(content, list):
        return inputs
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "input_audio":
            continue
        audio = item.get("input_audio")
        if not isinstance(audio, dict) or not isinstance(audio.get("data"), str):
            continue
        inputs.append((audio["data"], str(audio.get("format") or "").lower()))
    return inputs


def _prepare_inline_audio_attachments(
    audio_inputs: list[tuple[str, str]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Persist OpenAI input_audio payloads and return compact worker input."""
    if len(audio_inputs) > _MAX_V1_INLINE_AUDIO_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"at most {_MAX_V1_INLINE_AUDIO_FILES} inline audio files are allowed",
        )
    audio_urls: list[str] = []
    attachments: list[dict[str, Any]] = []
    total_bytes = 0
    for index, (encoded, audio_format) in enumerate(audio_inputs, start=1):
        format_entry = _V1_AUDIO_FORMATS.get(audio_format)
        if format_entry is None:
            supported = ", ".join(sorted(_V1_AUDIO_FORMATS))
            raise HTTPException(
                status_code=422,
                detail=(
                    f"inline audio {index} format must be one of: {supported}"
                ),
            )
        mime_type, extension = format_entry
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as error:
            raise HTTPException(
                status_code=422,
                detail=f"inline audio {index} is not valid base64",
            ) from error
        if not data:
            raise HTTPException(status_code=422, detail=f"inline audio {index} is empty")
        if len(data) > _MAX_V1_INLINE_AUDIO_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"inline audio {index} exceeds the "
                    f"{_MAX_V1_INLINE_AUDIO_BYTES}-byte limit"
                ),
            )
        total_bytes += len(data)
        if total_bytes > _MAX_V1_INLINE_AUDIO_TOTAL_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    "inline audio files exceed the "
                    f"{_MAX_V1_INLINE_AUDIO_TOTAL_BYTES}-byte total limit"
                ),
            )
        attachment = _store_relay_attachment(
            data,
            f"input-audio-{index}{extension}",
            mime_type,
            "audio",
        )
        audio_urls.append(attachment["url"])
        attachments.append(attachment)
    return audio_urls, attachments


def _flatten_anthropic_content(content: Any) -> str:
    """Like :func:`_flatten_content`, but also descends into Anthropic
    ``tool_result`` blocks (whose ``content`` may itself be a string or a
    nested block list) so tool output text isn't silently dropped from the
    relayed prompt."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") == "tool_result":
                    inner = _flatten_anthropic_content(item.get("content"))
                    if inner:
                        parts.append(inner)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return "" if content is None else str(content)


def _extract_anthropic_images(content: Any) -> list[str]:
    """Pull images out of Anthropic-style content blocks:
    ``{"type": "image", "source": {"type": "base64", "media_type", "data"}}``
    (converted to a data: URL the relay already understands) or
    ``{"type": "image", "source": {"type": "url", "url": ...}}``."""
    urls: list[str] = []
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            source = item.get("source")
            if not isinstance(source, dict):
                continue
            if source.get("type") == "base64" and source.get("data"):
                media_type = source.get("media_type") or "image/png"
                urls.append(f"data:{media_type};base64,{source['data']}")
            elif isinstance(source.get("url"), str) and source["url"]:
                urls.append(source["url"])
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



def _copilot_public_model_id(backing_model: str) -> str:
    return f"copilot/{backing_model}"


def _resolved_emullm_default_model() -> str:
    candidate = (
        _advertised_default
        if _advertised_default
        and _advertised_default != _EMULLM_DEFAULT_MODEL_ID
        else _DEFAULT_MODEL_ID
    )
    worker_id, suffix = _split_model_id(candidate)
    if worker_id == _DEFAULT_WORKER_ID:
        return f"{_COPILOT_POOL_WORKER_ID}/" + (
            suffix if suffix in _EXPORTED_PERSONA_SUFFIXES else "percent100"
        )
    if suffix in _HIDDEN_PERSONA_SUFFIXES:
        return f"{worker_id}/percent100"
    return candidate


def _emullm_default_model_entry() -> dict[str, Any]:
    resolved_model = _resolved_emullm_default_model()
    route = _model_routes.get(_EMULLM_DEFAULT_MODEL_ID)
    if route is None:
        route = _model_routes.get(resolved_model)
    route_targets = [route] if isinstance(route, str) else list(route or [])
    return {
        "id": _EMULLM_DEFAULT_MODEL_ID,
        "object": "model",
        "display_name": "EMULLM · Default",
        "description": (
            f"Stable alias for the configured default model '{resolved_model}'. "
            "An explicit route_targets override on emullm/default takes precedence."
        ),
        "context_length": 200000,
        "supported_parameters": ["messages", "temperature", "stream"],
        "owned_by": "emullm",
        "connected": bool(_connected_workers),
        "default": True,
        "resolved_model": resolved_model,
        "route_targets": route_targets,
        "routing_mode": "configured" if route_targets else "default_model",
        "simulated": True,
        "input_modalities": {
            "attachment_transport": {
                "supported": True,
                "media_types": ["*/*"],
                "source": "resolved model route",
            },
            "image": {"enabled": True, "status": "resolved_model_dependent"},
            "audio": {"enabled": True, "status": "resolved_model_dependent"},
            "general_file": {
                "enabled": True,
                "status": "transport_supported",
                "model_comprehension": "resolved-model-dependent",
            },
        },
    }


def _merge_model_patch(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_model_patch(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_model_catalog_override(
    entry: dict[str, Any],
    *,
    include_hidden: bool = False,
) -> dict[str, Any] | None:
    model_id = str(entry["id"])
    override = _model_catalog_overrides.get(model_id)
    if not isinstance(override, dict):
        if not include_hidden:
            return entry
        visible = dict(entry)
        visible["hidden"] = False
        visible["exported"] = True
        return visible
    is_hidden = override.get("hidden") is True
    if is_hidden and not include_hidden:
        return None
    patch = override.get("patch")
    merged = _merge_model_patch(entry, patch) if isinstance(patch, dict) else dict(entry)
    merged["id"] = model_id
    if include_hidden:
        merged["hidden"] = is_hidden
        merged["exported"] = not is_hidden
    return merged


def _copilot_model_metadata(backing_model: str) -> dict[str, Any] | None:
    return next(
        (
            model
            for model in _copilot_api.copilot_models()["models"]
            if model.get("id") == backing_model
        ),
        None,
    )


def _copilot_backing_model(public_model_id: str) -> str | None:
    prefix = "copilot/"
    if not public_model_id.startswith(prefix):
        return None
    backing_model = public_model_id[len(prefix) :]
    return backing_model if _copilot_model_metadata(backing_model) is not None else None


def _copilot_input_modalities(
    backing_model: str,
    metadata: dict[str, Any],
    active_workers: list[str],
) -> dict[str, Any]:
    capabilities = metadata.get("capabilities")
    limits = capabilities.get("limits") if isinstance(capabilities, dict) else {}
    vision = limits.get("vision") if isinstance(limits, dict) else {}
    media_types = (
        [str(value) for value in vision.get("supported_media_types", [])]
        if isinstance(vision, dict)
        else []
    )
    audio_media_types = [
        media_type for media_type in media_types if media_type.startswith("audio/")
    ]
    declared_audio = any(
        _worker_capabilities.get(worker_id, {}).get("audio_input") is True
        for worker_id in active_workers
    )
    if audio_media_types:
        audio_status = "sdk_advertised"
        audio_source = "copilot_model_catalog"
    elif declared_audio:
        audio_status = "operator_declared"
        audio_source = "connected_servant capability declaration"
    elif backing_model.startswith("gemini-"):
        audio_status = "family_implied"
        audio_source = "native_model_family; Copilot SDK does not advertise audio"
    else:
        audio_status = "not_advertised"
        audio_source = "Copilot SDK schema has no audio capability field"
    return {
        "attachment_transport": {
            "supported": True,
            "source": "Copilot SDK blob attachments via EMULLM cloud-file transport",
            "media_types": ["*/*"],
        },
        "image": {
            "status": "sdk_advertised" if media_types else "not_advertised",
            "media_types": media_types,
            "enabled": bool(media_types),
        },
        "audio": {
            "status": audio_status,
            "source": audio_source,
            "media_types": audio_media_types,
            "transport_supported": True,
            "enabled": audio_status != "not_advertised",
        },
        "general_file": {
            "status": "transport_supported",
            "model_comprehension": "model-dependent",
            "enabled": True,
        },
    }


def _copilot_task_capabilities(
    backing_model: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    normalized = backing_model.lower()
    code_implied = any(token in normalized for token in ("code", "codex"))
    capabilities = metadata.get("capabilities")
    model_type = (
        capabilities.get("type")
        if isinstance(capabilities, dict)
        else None
    )
    return {
        "code": {
            "enabled": code_implied,
            "status": "model_name_implied" if code_implied else "not_advertised",
        },
        "image_generation": {
            "enabled": code_implied,
            "status": "tool_generated" if code_implied else "not_advertised",
        },
        "image_output": {
            "enabled": code_implied,
            "status": "tool_generated" if code_implied else "not_advertised",
        },
        "summarization": {
            "enabled": model_type in (None, "chat"),
            "status": "general_chat_capability",
        },
    }


def _copilot_catalog_model_entry(metadata: dict[str, Any]) -> dict[str, Any]:
    backing_model = str(metadata["id"])
    task_capabilities = _copilot_task_capabilities(
        backing_model,
        metadata,
    )
    codex_supplier = _codex_supplier_for_model(backing_model)
    image_output = bool(
        task_capabilities["image_output"]["enabled"]
    )
    public_model_id = _copilot_public_model_id(backing_model)
    route = _model_routes.get(public_model_id)
    route_targets = [route] if isinstance(route, str) else list(route or [])
    elastic_limit = _on_demand_copilot_limit()
    elastic_end = _ON_DEMAND_COPILOT_START + elastic_limit - 1
    active_workers = sorted(
        worker_id
        for worker_id, runtime_model in _worker_runtime_models.items()
        if runtime_model == backing_model and worker_id in _connected_workers
    )
    capabilities = metadata.get("capabilities")
    limits = capabilities.get("limits") if isinstance(capabilities, dict) else {}
    return {
        "id": public_model_id,
        "object": "model",
        "display_name": f"GitHub Copilot · {metadata.get('name') or backing_model}",
        "description": (
            f"Authenticated GitHub Copilot backing model '{backing_model}'. "
            f"Requesting this ID lazily starts or reuses one of "
            f"{elastic_limit} elastic workers "
            f"({_ON_DEMAND_COPILOT_PREFIX}{_ON_DEMAND_COPILOT_START}.."
            f"{elastic_end})."
        ),
        "context_length": (
            limits.get("max_context_window_tokens", 200000)
            if isinstance(limits, dict)
            else 200000
        ),
        "supported_parameters": ["messages", "temperature", "stream"],
        "owned_by": "github-copilot",
        "provider": "github-copilot",
        "codex_supplier": (
            codex_supplier.get("id") if codex_supplier is not None else None
        ),
        "backing_model": backing_model,
        "connected": bool(active_workers),
        "active_workers": active_workers,
        "route_targets": route_targets,
        "routing_mode": "configured" if route_targets else "on_demand",
        "on_demand": True,
        "on_demand_worker_prefix": _ON_DEMAND_COPILOT_PREFIX,
        "on_demand_worker_start": _ON_DEMAND_COPILOT_START,
        "on_demand_worker_limit": elastic_limit,
        "max_concurrent_calls": _max_concurrent_calls,
        "simulated": False,
        "capabilities": capabilities or {},
        "billing": metadata.get("billing", {}),
        "policy": metadata.get("policy", {}),
        "quality_rank": metadata.get("quality_rank"),
        "quality_tier": metadata.get("quality_tier"),
        "input_modalities": _copilot_input_modalities(
            backing_model,
            metadata,
            active_workers,
        ),
        "task_capabilities": task_capabilities,
        "output_modalities": {
            "image": {
                "enabled": image_output,
                "capability": "image_output",
                "status": "tool_generated" if image_output else "not_advertised",
                "media_types": ["image/png"] if image_output else [],
            }
        },
    }


def _model_entry(worker_id: str, suffix: str, persona: dict[str, Any]) -> dict[str, Any]:
    model_id = f"{worker_id}/{suffix}"
    worker_kind = _worker_kinds.get(worker_id)
    runtime_model = _worker_runtime_models.get(worker_id)
    worker_description = _worker_descriptions.get(worker_id)
    worker_label = worker_id
    if worker_kind:
        worker_label += f" · {worker_kind}"
    if runtime_model:
        worker_label += f" ({runtime_model})"
    persona_name = str(persona.get("display_name") or suffix)
    description_parts = [
        value
        for value in (
            worker_description,
            f"Stable worker ID: {worker_id}.",
            f"Backing model: {runtime_model}." if runtime_model else None,
            str(persona.get("instruction") or "") or None,
        )
        if value
    ]
    return {
        "id": model_id,
        "object": "model",
        "display_name": f"{worker_label} · {persona_name}",
        "description": " ".join(description_parts),
        "context_length": 200000,
        "supported_parameters": [],
        "owned_by": worker_id,
        "connected": worker_id in _connected_workers,
        "worker_id": worker_id,
        "worker_kind": worker_kind,
        "backing_model": runtime_model,
        "simulated": True,
    }


def _active_workers_for_route(
    route: str | list[str] | None,
    model_id: str | None = None,
) -> list[str]:
    targets = [route] if isinstance(route, str) else (route or [])
    active: list[str] = []
    for target in targets:
        if target.startswith(("http://", "https://", "backend-")):
            continue
        if target == "worker-in-name" and model_id:
            worker_id, _ = _split_model_id(model_id)
            if worker_id in _connected_workers and worker_id not in active:
                active.append(worker_id)
            continue
        for worker_id in sorted(_connected_workers):
            if fnmatchcase(worker_id, target) and worker_id not in active:
                active.append(worker_id)
    return active


def _configured_model_entry(model_id: str) -> dict[str, Any]:
    route = _model_routes.get(model_id)
    targets = [route] if isinstance(route, str) else list(route or [])
    active_workers = _active_workers_for_route(route, model_id)
    backing_models = {
        worker_id: _worker_runtime_models.get(worker_id)
        for worker_id in active_workers
        if _worker_runtime_models.get(worker_id)
    }
    return {
        "id": model_id,
        "object": "model",
        "display_name": f"{model_id} · simulated by EMULLM",
        "description": (
            "Configured simulated model. "
            + (
                f"Ordered route: {' -> '.join(targets)}."
                if targets
                else "Uses the default EMULLM worker-selection policy."
            )
        ),
        "context_length": 200000,
        "supported_parameters": ["messages", "temperature", "stream"],
        "owned_by": "emullm",
        "connected": bool(
            active_workers
            or any(target.startswith(("http://", "https://")) for target in targets)
        ),
        "simulated": True,
        "route_targets": targets,
        "active_workers": active_workers,
        "backing_models": backing_models,
    }


def _backing_model_alias_entry(worker_id: str) -> dict[str, Any] | None:
    backing_model = _worker_runtime_models.get(worker_id)
    if not backing_model:
        return None
    metadata = next(
        (
            model
            for model in _copilot_api.copilot_models()["models"]
            if model.get("id") == backing_model
        ),
        {},
    )
    alias_id = f"{worker_id}/{backing_model}"
    worker_kind = _worker_kinds.get(worker_id) or "worker"
    return {
        "id": alias_id,
        "object": "model",
        "display_name": (
            f"{worker_id} · {worker_kind} backed by "
            f"{metadata.get('name') or backing_model} ({backing_model})"
        ),
        "description": (
            f"Direct alias for stable worker '{worker_id}' using its active "
            f"backing model '{backing_model}'."
        ),
        "context_length": (
            metadata.get("capabilities", {})
            .get("limits", {})
            .get("max_context_window_tokens", 200000)
        ),
        "supported_parameters": ["messages", "temperature", "stream"],
        "owned_by": worker_id,
        "connected": worker_id in _connected_workers,
        "worker_id": worker_id,
        "worker_kind": worker_kind,
        "backing_model": backing_model,
        "simulated": True,
        "capabilities": metadata.get("capabilities", {}),
        "billing": metadata.get("billing", {}),
        "quality_rank": metadata.get("quality_rank"),
        "quality_tier": metadata.get("quality_tier"),
        "input_modalities": _copilot_input_modalities(
            backing_model,
            metadata,
            [worker_id],
        ),
    }
class ChatMessage(BaseModel):
    role: str
    content: Any = ""


class ChatRequest(BaseModel):
    model: str = _DEFAULT_MODEL_ID
    messages: list[ChatMessage] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list, max_length=32)
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


class MessagesRequest(BaseModel):
    """Anthropic Messages API request shape (the surface Claude SDKs and
    Claude Code speak). ``system`` is a top-level string or block list rather
    than a message role; tool fields are accepted for wire compatibility but
    tools are not executed by the relay -- the worker sees their text."""

    model: str = _DEFAULT_MODEL_ID
    messages: list[ChatMessage] = Field(default_factory=list)
    max_tokens: int | None = Field(default=None, ge=1)
    system: Any = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    stream: bool = False
    stop_sequences: list[str] | None = None
    metadata: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None


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
    requested_worker_id, requested_suffix = _split_model_id(model)
    if requested_worker_id == _COPILOT_POOL_WORKER_ID:
        persona = _PERSONA_SUFFIXES.get(requested_suffix)
        if requested_suffix not in _EXPORTED_PERSONA_SUFFIXES or persona is None:
            raise HTTPException(
                status_code=404,
                detail=f"unknown Copilot pool persona '{model}'",
            )
        return _assign_copilot_pool_worker(), requested_suffix, persona
    if _numeric_copilot_worker_id(requested_worker_id):
        persona = _models_for(requested_worker_id).get(requested_suffix)
        if persona is not None:
            return requested_worker_id, requested_suffix, persona
        return requested_worker_id, "", {
            "instruction": None,
            "passthrough": True,
            "served_model": model,
        }
    if _capability_alias_spec(model) is not None:
        return _capability_alias_worker(model), "", {
            "instruction": (
                f"Serve the capability-selected model alias '{model}' faithfully."
            ),
            "passthrough": True,
            "served_model": model,
        }
    # A configured/runtime route sends this exact catalog id to a serving worker,
    # forwarding the original id so the worker knows which model to emulate.
    route = _model_routes.get(model)
    if isinstance(route, str):
        return route, "", {
            "instruction": f"You are serving model id '{model}'. Emulate that model faithfully.",
            "passthrough": True,
            "served_model": model,
        }
    if isinstance(route, list):
        for target in route:
            if target.startswith(("http://", "https://")):
                continue
            matches = sorted(
                worker_id for worker_id in _connected_workers if fnmatchcase(worker_id, target)
            )
            if matches:
                return matches[0], "", {
                    "instruction": f"You are serving model id '{model}'. Emulate that model faithfully.",
                    "passthrough": True,
                    "served_model": model,
                }
        return _DEFAULT_WORKER_ID, "", {
            "instruction": f"You are serving model id '{model}'. Emulate that model faithfully.",
            "passthrough": True,
            "served_model": model,
        }
    worker_id, suffix = requested_worker_id, requested_suffix
    persona = _models_for(worker_id).get(suffix)
    if persona is not None and suffix != "same":
        # Percent dials are virtual personas even when their prefix is not a
        # registered worker (for example ``smart/percent10`` in proxy mode).
        return worker_id, suffix, persona
    if _is_known_worker_id(worker_id):
        if persona is not None:
            return worker_id, suffix, persona
        # A named worker can receive a model it did not advertise. Keep the
        # caller's model intact instead of treating its suffix as a persona.
        return worker_id, "", {"instruction": None, "passthrough": True, "served_model": model}

    # Arbitrary model IDs are valid relay input. They are opaque to the
    # generic servant, which receives the original model value unchanged.
    return _DEFAULT_WORKER_ID, "", {"instruction": None, "passthrough": True, "served_model": model}


def _backend_model_for(model: str, backend: dict[str, Any]) -> str:
    """The model id to send upstream: forward a real backend model id as-is;
    for a local persona/default, use the backend's configured model."""
    worker_id, suffix = _split_model_id(model)
    if model == _DEFAULT_MODEL_ID or _models_for(worker_id).get(suffix) is not None:
        return str(backend.get("model") or model)
    return model


def _route_backend(target: str) -> dict[str, Any]:
    normalized = target.rstrip("/")
    for backend in _all_backends():
        if str(backend.get("base_url") or "").rstrip("/") == normalized:
            return backend
    return {
        "name": normalized,
        "base_url": normalized,
        "api_key_env": os.environ.get("EMULLM_PROXY_API_KEY_ENV"),
    }


def _route_backend_candidates(target: str) -> list[dict[str, Any]]:
    if not target.startswith("backend-"):
        return []
    pattern = target[len("backend-") :]
    matches = [
        backend
        for backend in _all_backends()
        if fnmatchcase(str(backend.get("name") or ""), pattern)
    ]
    if len(matches) < 2:
        return matches
    names = [str(backend.get("name") or "") for backend in matches]
    _, rotated_names = select_from_catalog(
        names,
        "round-robin",
        key=f"backend-route:{target}",
    )
    by_name = {
        str(backend.get("name") or ""): backend
        for backend in matches
    }
    return [by_name[name] for name in rotated_names]


async def _proxy_chat_with_stats(
    backend: dict[str, Any],
    model: str,
    prompt_text: str,
    instruction: str | None,
    service_kind: str,
) -> str:
    backend_id = f"backend-{backend.get('name') or 'proxy'}"
    _begin_worker_request(backend_id, model)
    started = time.monotonic()
    outcome = "failed"
    try:
        reply = await _proxy_chat(
            backend,
            model,
            prompt_text,
            instruction,
        )
        outcome = "served"
        return reply
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    finally:
        _record_worker_service(
            backend_id,
            model,
            service_kind,
            max(0.0, time.monotonic() - started),
            outcome,
        )
        _end_worker_request(backend_id, model)


def _required_input_capabilities(extra: dict[str, Any] | None) -> set[str]:
    required: set[str] = set()
    explicit = extra.get("required_capabilities") if extra else None
    if isinstance(explicit, list):
        required.update(
            capability
            for capability in explicit
            if isinstance(capability, str) and capability
        )
    if extra and extra.get("images"):
        required.add("vision_input")
    if extra and extra.get("audio"):
        required.add("audio_input")
    return required


def _route_worker_candidates(
    model: str,
    target: str,
    required_capabilities: set[str] | None = None,
) -> list[str]:
    if target == "worker-in-name":
        worker_id, _ = _split_model_id(model)
        if worker_id not in _connected_workers:
            return []
        return _capability_ordered_workers(
            [worker_id],
            required_capabilities,
        )
    requested_backing = _copilot_backing_model(model)
    matches = sorted(
        worker_id
        for worker_id in _connected_workers
        if fnmatchcase(worker_id, target)
        if _worker_ready_for_offer(worker_id)
        if not _is_elastic_copilot(worker_id)
        or requested_backing is None
        or _worker_runtime_models.get(worker_id) == requested_backing
    )
    if not matches:
        return []
    _, rotated = select_from_catalog(
        matches, "round-robin", key=f"model-route:{model}:{target}"
    )
    return _capability_ordered_workers(rotated, required_capabilities)


async def _wait_before_backend_fallback(model: str, target: str) -> None:
    if _backend_fallback_delay_seconds <= 0:
        return
    wait_id = f"backend-delay-{uuid.uuid4().hex}"
    _mark_waiting_for_worker(
        wait_id,
        target,
        model,
        "waiting before last-resort backend fallback",
    )
    try:
        await asyncio.sleep(_backend_fallback_delay_seconds)
    finally:
        _clear_waiting_for_worker(wait_id)


async def _relay_model_route_chain(
    model: str,
    targets: list[str],
    prompt_text: str,
    extra: dict[str, Any] | None,
) -> Any:
    """Try worker-ID globs and OpenAI-compatible backend URLs in order."""
    instruction = f"You are serving model id '{model}'. Emulate that model faithfully."
    required_capabilities = _required_input_capabilities(extra)
    service_kind = str((extra or {}).get("kind") or "chat")
    failures: list[str] = []
    backend_delay_complete = False
    for target in targets:
        is_backend = target.startswith(
            ("http://", "https://", "backend-")
        )
        if is_backend and not backend_delay_complete:
            await _wait_before_backend_fallback(model, target)
            backend_delay_complete = True
        if target.startswith("backend-"):
            backend_candidates = _route_backend_candidates(target)
            if not backend_candidates:
                failures.append(f"{target}: no configured backend matched")
                continue
            for backend in backend_candidates:
                backend_name = str(backend.get("name") or target)
                try:
                    reply = await _proxy_chat_with_stats(
                        backend,
                        model,
                        prompt_text,
                        instruction,
                        service_kind,
                    )
                except HTTPException as error:
                    failures.append(f"backend-{backend_name}: {error.detail}")
                    continue
                await _mirror_to_observers(
                    f"backend-{backend_name}",
                    model,
                    prompt_text,
                    reply,
                )
                return reply
            continue
        if target.startswith(("http://", "https://")):
            try:
                reply = await _proxy_chat_with_stats(
                    _route_backend(target),
                    model,
                    prompt_text,
                    instruction,
                    service_kind,
                )
            except HTTPException as error:
                failures.append(f"{target}: {error.detail}")
                continue
            await _mirror_to_observers(target, model, prompt_text, reply)
            return reply

        candidates = _route_worker_candidates(model, target, required_capabilities)
        if target == "worker-copilot-*":
            replica = await _elastic_replica_for_busy_workers(
                model,
                candidates,
                required_capabilities,
            )
            if replica is not None:
                candidates = [replica, *candidates]
        if not candidates:
            failures.append(f"{target}: no connected worker matched")
            continue
        for worker_id in candidates:
            try:
                _check_and_record_usage(worker_id)
                result = await _relay_to_worker(
                    worker_id,
                    model,
                    prompt_text,
                    instruction,
                    wait=False,
                    extra=extra,
                )
            except _WorkerNotReady as error:
                failures.append(
                    f"{worker_id}: not ready ({error.reason})"
                )
                continue
            except _WorkerRejected as error:
                failures.append(f"{worker_id}: {error.reason}")
                continue
            except HTTPException as error:
                _release_worker_reservation(worker_id)
                failures.append(f"{worker_id}: {error.detail}")
                continue
            if result is _PASS:
                failures.append(f"{worker_id}: disconnected before accepting")
                continue
            await _mirror_to_observers(worker_id, model, prompt_text, _reply_content(result))
            return result

    detail = "; ".join(failures) if failures else "the route chain is empty"
    raise HTTPException(
        status_code=503,
        detail=f"all configured routes failed for model '{model}' ({detail})",
    )


def _token_count(value: Any) -> int:
    text = _flatten_content(value)
    return len(text.split()) if text else 0


def _on_demand_copilot_capabilities(model_entry: dict[str, Any]) -> list[str]:
    modalities = model_entry.get("input_modalities")
    if not isinstance(modalities, dict):
        return []
    configured: list[str] = []
    for modality, capability in (
        ("image", "vision_input"),
        ("audio", "audio_input"),
        ("general_file", "file_input"),
    ):
        value = modalities.get(modality)
        if not isinstance(value, dict) or not isinstance(value.get("enabled"), bool):
            continue
        configured.append(capability if value["enabled"] else f"!{capability}")
    tasks = model_entry.get("task_capabilities")
    if isinstance(tasks, dict):
        for capability, value in tasks.items():
            if not isinstance(value, dict) or not isinstance(value.get("enabled"), bool):
                continue
            configured.append(
                str(capability)
                if value["enabled"]
                else f"!{capability}"
            )
    return configured


def _new_on_demand_copilot_config(
    worker_id: str,
    backing_model: str,
    model_entry: dict[str, Any],
    *,
    warmup: bool = False,
) -> _copilot_api.HeadlessCopilotConfig:
    return _copilot_api.HeadlessCopilotConfig(
        worker_id=worker_id,
        model=backing_model,
        modelmasks=[_copilot_public_model_id(backing_model)],
        role="on-demand-copilot",
        capabilities=_on_demand_copilot_capabilities(model_entry),
        autostart=False,
        warmup=warmup,
    )


def _provision_on_demand_copilot(
    backing_model: str,
    model_entry: dict[str, Any],
    *,
    require_new: bool = False,
    warmup: bool = False,
) -> tuple[str, Any, bool]:
    if model_entry.get("on_demand") is False:
        raise HTTPException(
            status_code=409,
            detail=f"on-demand loading is disabled for '{model_entry['id']}'",
        )
    manager = _copilot_api.get_manager()
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="headless Copilot manager is unavailable",
        )
    with _on_demand_copilot_lock:
        instances = manager.list()
        dynamic = {
            str(instance["worker_id"]): instance
            for instance in instances
            if _is_on_demand_copilot(str(instance.get("worker_id") or ""))
        }
        matching = sorted(
            (
                instance
                for instance in dynamic.values()
                if (
                    _worker_runtime_models.get(str(instance["worker_id"]))
                    or instance.get("selected_model")
                    or (instance.get("config") or {}).get("model")
                ) == backing_model
            ),
            key=lambda instance: _worker_load(str(instance["worker_id"])),
        )
        available_matching = next(
            (
                instance
                for instance in matching
                if _worker_load(str(instance["worker_id"])) == 0
            ),
            None,
        )
        if available_matching is not None and not require_new:
            matching = available_matching
            worker_id = str(matching["worker_id"])
            if not matching.get("running"):
                manager.start(worker_id)
            _reserve_worker(worker_id)
            return worker_id, manager, False

        reusable = next(
            (
                instance
                for instance in sorted(
                    dynamic.values(),
                    key=lambda item: str(item.get("worker_id") or ""),
                )
                if not instance.get("running")
            ),
            None,
        )
        if reusable is not None:
            worker_id = str(reusable["worker_id"])
            manager.update(
                worker_id,
                _new_on_demand_copilot_config(
                    worker_id,
                    backing_model,
                    model_entry,
                    warmup=warmup,
                ),
                restart=False,
            )
            manager.start(worker_id)
            _reserve_worker(worker_id)
            return worker_id, manager, False

        for worker_id in _on_demand_copilot_ids():
            if worker_id in dynamic:
                continue
            manager.create(
                _new_on_demand_copilot_config(
                    worker_id,
                    backing_model,
                    model_entry,
                    warmup=warmup,
                ),
                start=True,
            )
            _reserve_worker(worker_id)
            return worker_id, manager, False

        switchable = next(
            (
                instance
                for instance in sorted(
                    dynamic.values(),
                    key=lambda item: str(item.get("worker_id") or ""),
                )
                if instance.get("running")
                and instance.get("connected")
                and _worker_is_idle(str(instance["worker_id"]))
            ),
            None,
        )
        if switchable is not None and not require_new:
            worker_id = str(switchable["worker_id"])
            _reserve_worker(worker_id)
            return worker_id, manager, True

    raise HTTPException(
        status_code=503,
        detail=(
            f"all {_on_demand_copilot_limit()} elastic Copilot workers are busy; "
            "stop one in the admin UI before loading another model"
        ),
    )


async def _ensure_on_demand_copilot(
    public_model_id: str,
    backing_model: str,
    *,
    require_new: bool = False,
    warmup: bool = False,
) -> str:
    metadata = _copilot_model_metadata(backing_model)
    if metadata is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown Copilot backing model '{backing_model}'",
        )
    entry = _apply_model_catalog_override(
        _copilot_catalog_model_entry(metadata)
    )
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"model '{public_model_id}' is hidden",
        )
    worker_id, manager, switch_model = await asyncio.to_thread(
        _provision_on_demand_copilot,
        backing_model,
        entry,
        require_new=require_new,
        warmup=warmup,
    )
    wait_id = f"provision-{uuid.uuid4().hex}"
    try:
        deadline = time.monotonic() + min(90.0, _REQUEST_TIMEOUT_SECONDS)
        while time.monotonic() < deadline:
            instance = await asyncio.to_thread(manager.get, worker_id)
            if instance.get("connected"):
                if switch_model:
                    await _set_worker_runtime_model(
                        worker_id,
                        backing_model,
                        model_entry,
                    )
                return worker_id
            _mark_waiting_for_worker(
                wait_id,
                worker_id,
                public_model_id,
                "waiting for elastic Copilot worker to connect",
            )
            if not instance.get("running") and instance.get("returncode") is not None:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"on-demand worker '{worker_id}' exited with "
                        f"code {instance['returncode']}"
                    ),
                )
            await asyncio.sleep(0.2)
        raise HTTPException(
            status_code=504,
            detail=f"on-demand worker '{worker_id}' did not connect in time",
        )
    except BaseException:
        _release_worker_reservation(worker_id)
        raise
    finally:
        _clear_waiting_for_worker(wait_id)


def _provision_explicit_copilot_worker(
    worker_id: str,
) -> Any:
    if not _numeric_copilot_worker_id(worker_id) or len(worker_id) > 128:
        raise HTTPException(
            status_code=422,
            detail=(
                "explicit Copilot worker IDs must be "
                "worker-copilot-<positive integer>"
            ),
        )
    manager = _copilot_api.get_manager()
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="headless Copilot manager is unavailable",
        )
    with _on_demand_copilot_lock:
        try:
            instance = manager.get(worker_id)
        except _copilot_api.CopilotInstanceMissing:
            client_requested = [
                item
                for item in manager.list()
                if item.get("role") == "client-requested-copilot"
            ]
            if len(client_requested) >= _max_concurrent_calls:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "client-requested Copilot worker quota reached "
                        f"({_max_concurrent_calls}); remove an unused worker "
                        "from the local admin UI before creating another"
                    ),
                )
            manager.create(
                _copilot_api.HeadlessCopilotConfig(
                    worker_id=worker_id,
                    model=None,
                    model_selector="random",
                    modelmasks=[f"{worker_id}/*"],
                    role="client-requested-copilot",
                    autostart=True,
                    warmup=True,
                    use_shared_anti_idle=True,
                ),
                start=True,
            )
        else:
            if not instance.get("running") and not instance.get("connected"):
                manager.start(worker_id)
    return manager


async def _ensure_explicit_copilot_worker(worker_id: str) -> None:
    if worker_id in _connected_workers:
        return
    manager = await asyncio.to_thread(
        _provision_explicit_copilot_worker,
        worker_id,
    )
    wait_id = f"explicit-provision-{uuid.uuid4().hex}"
    try:
        deadline = time.monotonic() + min(90.0, _REQUEST_TIMEOUT_SECONDS)
        while time.monotonic() < deadline:
            instance = await asyncio.to_thread(manager.get, worker_id)
            if instance.get("connected"):
                return
            _mark_waiting_for_worker(
                wait_id,
                worker_id,
                f"{worker_id}/percent100",
                "waiting for explicitly numbered Copilot worker to connect",
            )
            if not instance.get("running") and instance.get("returncode") is not None:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"explicit worker '{worker_id}' exited with "
                        f"code {instance['returncode']}"
                    ),
                )
            await asyncio.sleep(0.2)
        raise HTTPException(
            status_code=504,
            detail=f"explicit worker '{worker_id}' did not connect in time",
        )
    finally:
        _clear_waiting_for_worker(wait_id)


def _idle_copilot_worker_count() -> int:
    return sum(
        1
        for worker_id in _connected_workers
        if worker_id.startswith(_ON_DEMAND_COPILOT_PREFIX)
        if not _is_elastic_copilot(worker_id)
        or _is_on_demand_copilot(worker_id)
        if _worker_is_idle(worker_id)
        if _worker_ready_for_offer(worker_id)
    )


async def maintain_idle_copilot_workers_once() -> list[str]:
    """Start enough elastic workers to restore the configured idle reserve."""
    started: list[str] = []
    if _idle_maintenance_paused:
        return started
    manager = _copilot_api.get_manager()
    if manager is not None:
        instances = await asyncio.to_thread(_copilot_api.manager_status)
        if any(
            instance.get("running") and not instance.get("connected")
            for instance in instances
        ):
            return started
        for instance in instances:
            worker_id = str(instance.get("worker_id") or "")
            if (
                _is_elastic_copilot(worker_id)
                and not _is_on_demand_copilot(worker_id)
                and instance.get("role") != "client-requested-copilot"
                and instance.get("running")
                and _worker_load(worker_id) == 0
            ):
                if worker_id in _connected_workers:
                    await _shutdown_connected_worker(
                        worker_id,
                        "outside configured elastic capacity",
                    )
                else:
                    await asyncio.to_thread(manager.stop, worker_id)
        idle_workers = [
            worker_id
            for worker_id in _connected_workers
            if _is_on_demand_copilot(worker_id)
            and _worker_is_idle(worker_id)
            and _worker_ready_for_offer(worker_id)
        ]
        total_idle = _idle_copilot_worker_count()
        excess = max(0, total_idle - min(_idle_worker_target, _max_concurrent_calls))
        for worker_id in sorted(
            idle_workers,
            key=lambda value: int(value.rsplit("-", 1)[1]),
            reverse=True,
        )[:excess]:
            if worker_id in _connected_workers:
                await _shutdown_connected_worker(
                    worker_id,
                    "idle reserve scale-down",
                )
            else:
                await asyncio.to_thread(manager.stop, worker_id)
    target = min(_idle_worker_target, _max_concurrent_calls)
    if _idle_copilot_worker_count() < target:
        worker_id = await _ensure_on_demand_copilot(
            _copilot_public_model_id("auto"),
            "auto",
            require_new=True,
            warmup=True,
        )
        _release_worker_reservation(worker_id)
        started.append(worker_id)
    return started


async def maintain_idle_copilot_workers(
    initial_delay_seconds: float = 1,
) -> None:
    """Keep the configured number of zero-load Copilot workers connected."""
    await asyncio.sleep(initial_delay_seconds)
    while True:
        try:
            await maintain_idle_copilot_workers_once()
        except HTTPException:
            pass
        except Exception as error:
            print(f"[emullm] idle worker maintainer: {error}", flush=True)
        await asyncio.sleep(1)


async def _elastic_replica_for_busy_workers(
    model: str,
    worker_ids: list[str],
    required_capabilities: set[str] | None,
) -> str | None:
    candidates = [
        worker_id
        for worker_id in worker_ids
        if worker_id.startswith(_ON_DEMAND_COPILOT_PREFIX)
        and _worker_runtime_models.get(worker_id)
    ]
    if not candidates or any(_worker_load(worker_id) == 0 for worker_id in candidates):
        return None
    source = min(candidates, key=_worker_load)
    backing_model = _worker_runtime_models[source]
    metadata = _copilot_model_metadata(backing_model)
    if metadata is None:
        return None
    entry = _apply_model_catalog_override(
        _copilot_catalog_model_entry(metadata)
    )
    if entry is None:
        return None
    modalities = entry.setdefault("input_modalities", {})
    if required_capabilities:
        if "vision_input" in required_capabilities:
            modalities.setdefault("image", {})["enabled"] = True
        if "audio_input" in required_capabilities:
            modalities.setdefault("audio", {})["enabled"] = True
    try:
        return await _ensure_on_demand_copilot(
            _copilot_public_model_id(backing_model),
            backing_model,
            require_new=True,
        )
    except HTTPException:
        return None


async def _relay_full(
    model: str,
    prompt_text: str,
    *,
    images: list[str] | None = None,
    audio: str | None = None,
    files: dict[str, Any] | None = None,
    attachments: list[dict[str, Any]] | None = None,
    kind: str | None = None,
    required_capabilities: set[str] | None = None,
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
    if capability_alias := _capability_alias_spec(model):
        required_capabilities = set(required_capabilities or set())
        required_capabilities.update(capability_alias[2])
    extra: dict[str, Any] = {}
    if images:
        extra["images"] = images
    if audio:
        extra["audio"] = audio
    if files:
        extra["files"] = files
    if attachments:
        extra["attachments"] = attachments
    if kind:
        extra["kind"] = kind
    if required_capabilities:
        extra["required_capabilities"] = sorted(required_capabilities)
    resolved_capabilities = _required_input_capabilities(extra)
    if resolved_capabilities:
        extra["required_capabilities"] = sorted(resolved_capabilities)

    configured_route = _model_routes.get(model)
    if isinstance(configured_route, list):
        return await _relay_model_route_chain(
            model,
            configured_route,
            prompt_text,
            extra or None,
        )

    if model == _EMULLM_DEFAULT_MODEL_ID and configured_route is None:
        return await _relay_full(
            _resolved_emullm_default_model(),
            prompt_text,
            images=images,
            audio=audio,
            files=files,
            attachments=attachments,
            kind=kind,
            required_capabilities=resolved_capabilities,
        )

    if configured_route is None and (
        backing_model := _copilot_backing_model(model)
    ):
        primary_worker_id = await _ensure_on_demand_copilot(model, backing_model)
        worker_ids = list(dict.fromkeys([
            primary_worker_id,
            *_route_worker_candidates(
                model,
                "worker-copilot-*",
                resolved_capabilities,
            ),
        ]))
        failures: list[str] = []
        for worker_id in worker_ids:
            try:
                _check_and_record_usage(worker_id)
            except HTTPException as error:
                if worker_id == primary_worker_id:
                    _release_worker_reservation(worker_id)
                failures.append(f"{worker_id}: {error.detail}")
                continue
            try:
                result = await _relay_to_worker(
                    worker_id,
                    model,
                    prompt_text,
                    (
                        f"You are the GitHub Copilot backing model '{backing_model}'. "
                        "Answer directly at your native capability."
                    ),
                    wait=len(worker_ids) == 1,
                    extra=extra or None,
                )
            except _WorkerNotReady as error:
                failures.append(f"{worker_id}: not ready ({error.reason})")
                continue
            except _WorkerRejected as error:
                failures.append(f"{worker_id}: rejected ({error.reason})")
                continue
            except HTTPException as error:
                failures.append(f"{worker_id}: {error.detail}")
                continue
            if result is _PASS:
                failures.append(f"{worker_id}: disconnected before accepting")
                continue
            await _mirror_to_observers(
                worker_id,
                model,
                prompt_text,
                _reply_content(result),
            )
            return result
        raise HTTPException(
            status_code=503,
            detail=(
                f"no eligible servant completed model '{model}' "
                f"({'; '.join(failures) or 'no candidates'})"
            ),
        )

    resolved_worker_id, _, persona = _require_model(model)
    if _numeric_copilot_worker_id(resolved_worker_id):
        await _ensure_explicit_copilot_worker(resolved_worker_id)
    worker_ids = _worker_candidates_for_model(
        model,
        resolved_worker_id,
        resolved_capabilities,
    )
    requested_worker_id, _ = _split_model_id(model)
    if requested_worker_id not in _connected_workers:
        replica = await _elastic_replica_for_busy_workers(
            model,
            worker_ids,
            resolved_capabilities,
        )
        if replica is not None:
            worker_ids = [replica, *worker_ids]
    instruction = persona.get("instruction")

    modes = _current_modes()
    rejections: list[_WorkerRejected] = []
    not_ready: list[_WorkerNotReady] = []
    for worker_id in worker_ids:
        try:
            _check_and_record_usage(worker_id)
        except BaseException:
            _release_worker_reservation(worker_id)
            raise
        for mode in modes:
            try:
                result = await _relay_step(mode, worker_id, model, prompt_text, instruction, extra or None)
            except _WorkerNotReady as deferred:
                not_ready.append(deferred)
                break
            except _WorkerRejected as rejection:
                rejections.append(rejection)
                break
            if result is not _PASS:
                await _mirror_to_observers(worker_id, model, prompt_text, _reply_content(result))
                return result
        else:
            continue
        # A rejection is an explicit refusal, so offer the request to the next
        # matching/all-model worker instead of manufacturing a reply.
        continue
    if rejections or not_ready:
        reasons = "; ".join(
            [
                *(f"{item.worker_id}: rejected ({item.reason})" for item in rejections),
                *(
                    f"{item.worker_id}: not ready ({item.reason})"
                    for item in not_ready
                ),
            ]
        )
        raise HTTPException(
            status_code=503,
            detail=f"no eligible worker completed model '{model}' ({reasons})",
        )
    # Every step passed (e.g. `recruit` with nobody connected and no fallback).
    raise HTTPException(
        status_code=504,
        detail=f"no strategy produced a reply for '{resolved_worker_id}' (modes={','.join(modes)})",
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
        await _wait_before_backend_fallback(
            model,
            f"backend-{backend.get('name') or 'proxy'}",
        )
        reply = await _proxy_chat_with_stats(
            backend,
            model,
            prompt_text,
            instruction,
            str((extra or {}).get("kind") or "chat"),
        )
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


async def _set_worker_runtime_model(
    worker_id: str,
    backing_model: str,
    model_entry: dict[str, Any],
) -> None:
    peer = _connected_workers.get(worker_id)
    if peer is None:
        raise HTTPException(
            status_code=503,
            detail=f"worker '{worker_id}' disconnected before model switch",
        )
    control_id = f"model-switch-{uuid.uuid4().hex}"
    future: "asyncio.Future[dict[str, Any]]" = (
        asyncio.get_running_loop().create_future()
    )
    _pending_worker_controls[control_id] = future
    capabilities = model_entry.get("capabilities")
    limits = capabilities.get("limits") if isinstance(capabilities, dict) else {}
    vision = limits.get("vision") if isinstance(limits, dict) else {}
    media_types = (
        vision.get("supported_media_types", [])
        if isinstance(vision, dict)
        else []
    )
    try:
        await _send_worker_json(
            worker_id,
            peer,
            {
                "type": "set_model",
                "id": control_id,
                "model": backing_model,
                "modelmasks": [_copilot_public_model_id(backing_model)],
                "capabilities": _on_demand_copilot_capabilities(model_entry),
                "supported_media_types": media_types,
            }
        )
        response = await asyncio.wait_for(future, timeout=60)
        if response.get("type") == "model_change_error":
            raise HTTPException(
                status_code=503,
                detail=(
                    f"worker '{worker_id}' could not switch to "
                    f"'{backing_model}': {response.get('error')}"
                ),
            )
    except asyncio.TimeoutError as error:
        raise HTTPException(
            status_code=504,
            detail=f"worker '{worker_id}' model switch timed out",
        ) from error
    finally:
        _pending_worker_controls.pop(control_id, None)


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
    _begin_worker_request(worker_id, model)
    request_id = uuid.uuid4().hex
    loop = asyncio.get_running_loop()
    future: "asyncio.Future[Any]" = loop.create_future()
    _pending[request_id] = future
    _pending_models[request_id] = model

    payload = {
        "type": "request",
        "id": request_id,
        "model": model,
        "worker_id": worker_id,
        "prompt": prompt_text,
        "acceptance_requested": True,
    }
    if instruction:
        payload["persona_instruction"] = instruction
    if extra:
        # real two-way extras: ``images`` (list of urls/data-urls) and ``kind``
        for key, value in extra.items():
            if value is not None:
                payload[key] = value

    deadline = time.monotonic() + _REQUEST_TIMEOUT_SECONDS
    sent_peer: Any = None
    service_started: float | None = None
    service_kind = str((extra or {}).get("kind") or "chat")
    outcome = "failed"
    try:
        while True:
            peer = peer_override if peer_override is not None else _connected_workers.get(worker_id)
            if peer is None:
                if not wait:
                    return _PASS
                _mark_waiting_for_worker(
                    request_id,
                    worker_id,
                    model,
                    "no worker is connected and ready",
                )
                if time.monotonic() >= deadline:
                    raise HTTPException(
                        status_code=504,
                        detail=f"no emullm worker registered as '{worker_id}' (timed out waiting)",
                    )
                await asyncio.sleep(0.5)
                continue
            _clear_waiting_for_worker(request_id)
            try:
                if peer_override is None:
                    await _send_worker_json(worker_id, peer, payload)
                else:
                    await peer.send_json(payload)
                sent_peer = peer
                service_started = time.monotonic()
                _request_assigned_worker.set(worker_id)
                with _worker_load_lock:
                    _active_service_requests[request_id] = {
                        "worker_id": worker_id,
                        "model": model,
                        "service_kind": service_kind,
                        "started_at": _now_iso(),
                        "started_monotonic": service_started,
                    }
            except Exception:
                if not wait:
                    return _PASS
                # That worker may have just disconnected; keep waiting/retrying.
                await asyncio.sleep(0.5)
                continue
            _record_relay_request(worker_id, request_id, payload)
            break

        remaining = max(1.0, deadline - time.monotonic())
        try:
            result = await asyncio.wait_for(future, timeout=remaining)
            # Real WebSocket replies are recorded by _handle_worker_message;
            # this idempotent call also covers test/mock peers that resolve
            # the Future directly without passing through that handler.
            _record_relay_reply(worker_id, request_id, result, model)
            outcome = "served"
            return result
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="emullm worker did not reply in time")
    except _WorkerNotReady:
        outcome = "deferred"
        raise
    except _WorkerRejected:
        outcome = "rejected"
        raise
    except asyncio.CancelledError:
        outcome = "cancelled"
        if sent_peer is not None and peer_override is None:
            try:
                await asyncio.shield(
                    _send_worker_json(
                        worker_id,
                        sent_peer,
                        {"type": "cancel", "id": request_id},
                    )
                )
            except Exception:
                pass
        raise
    finally:
        _pending.pop(request_id, None)
        _pending_models.pop(request_id, None)
        _clear_waiting_for_worker(request_id)
        with _worker_load_lock:
            _active_service_requests.pop(request_id, None)
        if service_started is not None:
            _record_worker_service(
                worker_id,
                model,
                service_kind,
                max(0.0, time.monotonic() - service_started),
                outcome,
            )
        _end_worker_request(worker_id, model)


@router.get("/v1/models")
def list_models(hidden: bool = False) -> dict[str, Any]:
    """Aggregates the model/persona menu across every currently connected
    worker, plus the default worker_id's fallback menu even if it isn't
    connected right now (so the primary identity is always discoverable)."""
    worker_ids = sorted(
        worker_id
        for worker_id in _connected_workers
        if worker_id != _DEFAULT_WORKER_ID
        and not _numeric_copilot_worker_id(worker_id)
    )
    worker_data = [
        _model_entry(worker_id, suffix, persona)
        for worker_id in worker_ids
        for suffix, persona in _models_for(worker_id).items()
        if suffix not in _HIDDEN_PERSONA_SUFFIXES
    ]
    pool_data = [
        _copilot_pool_model_entry(suffix, _PERSONA_SUFFIXES[suffix])
        for suffix in ("percent125", "percent100", "percent25")
    ]
    capability_data = [
        _capability_alias_model_entry(selector, capability, required)
        for capability, required in _CAPABILITY_MODEL_ALIASES.items()
        for selector in ("best", "worse")
    ]
    backing_aliases = [
        alias
        for worker_id in sorted(_connected_workers)
        if not _numeric_copilot_worker_id(worker_id)
        if (alias := _backing_model_alias_entry(worker_id)) is not None
    ]
    seen = {
        entry["id"]
        for entry in [
            *pool_data,
            *capability_data,
            *worker_data,
            *backing_aliases,
        ]
    }
    seen.add(_EMULLM_DEFAULT_MODEL_ID)
    configured_ids = [
        model_id
        for model_id in [*advertised_catalog()["models"], *_model_routes.keys()]
        if (
            model_id
            and _catalog_model_is_visible(model_id)
            and model_id not in seen
            and not seen.add(model_id)
        )
    ]
    configured_data = [
        _configured_model_entry(model_id) for model_id in configured_ids
    ]
    seen.update(entry["id"] for entry in configured_data)
    copilot_data = [
        _copilot_catalog_model_entry(metadata)
        for metadata in _copilot_api.copilot_models()["models"]
        if _copilot_public_model_id(str(metadata["id"])) not in seen
    ]
    base_data = [
        _emullm_default_model_entry(),
        *pool_data,
        *capability_data,
        *worker_data,
        *backing_aliases,
        *configured_data,
        *copilot_data,
    ]
    data = [
        configured
        for entry in base_data
        if (
            configured := _apply_model_catalog_override(
                entry,
                include_hidden=hidden,
            )
        )
        is not None
    ]
    base_ids = {entry["id"] for entry in base_data}
    for model_id, override in _model_catalog_overrides.items():
        if model_id in base_ids or not _catalog_model_is_visible(model_id):
            continue
        if override.get("hidden") is True and not hidden:
            continue
        patch = override.get("patch")
        if not isinstance(patch, dict):
            continue
        custom = _apply_model_catalog_override(
            {
                "id": model_id,
                "object": "model",
                "owned_by": "emullm-operator",
                "connected": False,
                "simulated": True,
            },
            include_hidden=hidden,
        )
        if custom is not None:
            data.append(custom)
    return {"object": "list", "data": data}


@router.get("/v1/models/{model_id:path}")
def get_model(model_id: str) -> dict[str, Any]:
    if not _catalog_model_is_visible(model_id):
        raise HTTPException(
            status_code=404,
            detail=f"legacy persona model '{model_id}' is not exported",
        )
    if model_id == _EMULLM_DEFAULT_MODEL_ID:
        entry = _apply_model_catalog_override(_emullm_default_model_entry())
        if entry is None:
            raise HTTPException(status_code=404, detail=f"model '{model_id}' is hidden")
        return entry
    worker_id, suffix = _split_model_id(model_id)
    if (
        worker_id == _COPILOT_POOL_WORKER_ID
        and suffix in _EXPORTED_PERSONA_SUFFIXES
    ):
        entry = _apply_model_catalog_override(
            _copilot_pool_model_entry(
                suffix,
                _PERSONA_SUFFIXES[suffix],
            )
        )
        if entry is None:
            raise HTTPException(status_code=404, detail=f"model '{model_id}' is hidden")
        return entry
    if spec := _capability_alias_spec(model_id):
        entry = _apply_model_catalog_override(
            _capability_alias_model_entry(*spec)
        )
        if entry is None:
            raise HTTPException(status_code=404, detail=f"model '{model_id}' is hidden")
        return entry
    if model_id in _model_routes or model_id in advertised_catalog()["models"]:
        entry = _apply_model_catalog_override(_configured_model_entry(model_id))
        if entry is None:
            raise HTTPException(status_code=404, detail=f"model '{model_id}' is hidden")
        return entry
    if backing_model := _copilot_backing_model(model_id):
        metadata = _copilot_model_metadata(backing_model)
        if metadata is not None:
            entry = _apply_model_catalog_override(
                _copilot_catalog_model_entry(metadata)
            )
            if entry is None:
                raise HTTPException(status_code=404, detail=f"model '{model_id}' is hidden")
            return entry
    if suffix == _worker_runtime_models.get(worker_id):
        alias = _backing_model_alias_entry(worker_id)
        if alias is not None:
            entry = _apply_model_catalog_override(alias)
            if entry is None:
                raise HTTPException(status_code=404, detail=f"model '{model_id}' is hidden")
            return entry
    persona = _models_for(worker_id).get(suffix)
    if persona is None:
        override = _model_catalog_overrides.get(model_id)
        if isinstance(override, dict) and isinstance(override.get("patch"), dict):
            entry = _apply_model_catalog_override(
                {
                    "id": model_id,
                    "object": "model",
                    "owned_by": "emullm-operator",
                    "connected": False,
                    "simulated": True,
                }
            )
            if entry is not None:
                return entry
        raise HTTPException(status_code=404, detail=f"unknown model '{model_id}'")
    entry = _apply_model_catalog_override(
        _model_entry(worker_id, suffix, persona)
    )
    if entry is None:
        raise HTTPException(status_code=404, detail=f"model '{model_id}' is hidden")
    return entry


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
        "modelmasks": list(_worker_model_masks[worker_id]) if worker_id in _worker_model_masks else None,
        "worker_kind": _worker_kinds.get(worker_id),
        "backing_model": _worker_runtime_models.get(worker_id),
        "description": _worker_descriptions.get(worker_id),
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
    audio_inputs: list[tuple[str, str]] = []
    for m in body.messages:
        images.extend(_extract_images(m.content))
        audio_inputs.extend(_extract_audio_inputs(m.content))
    images, image_attachments = _prepare_inline_image_attachments(images)
    audio_urls, audio_attachments = _prepare_inline_audio_attachments(audio_inputs)
    if images and audio_urls:
        request_kind = "multimodal"
    elif images:
        request_kind = "vision"
    elif audio_urls:
        request_kind = "audio_attachment"
    else:
        request_kind = "chat"
    result = await _relay_full(
        body.model,
        prompt_text,
        images=images or None,
        audio=audio_urls[0] if audio_urls else None,
        attachments=[*image_attachments, *audio_attachments] or None,
        kind=request_kind,
        required_capabilities={
            capability.strip()
            for capability in body.required_capabilities
            if capability.strip()
        },
    )
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


@router.post("/v1/messages")
async def anthropic_messages(body: MessagesRequest) -> Any:
    """Anthropic Messages API-compatible endpoint. Relays exactly like
    /v1/chat/completions (same worker routing, personas, model_routes, and
    modes), then reshapes the reply into an Anthropic ``message`` object --
    or, when ``stream`` is true, the Anthropic SSE event sequence
    (message_start / content_block_* / message_delta / message_stop) -- so
    Anthropic SDKs and Claude Code can point at this relay unchanged."""
    if not body.messages:
        raise HTTPException(status_code=400, detail="messages is required")
    parts: list[str] = []
    system_text = _flatten_anthropic_content(body.system) if body.system is not None else ""
    if system_text:
        parts.append(f"[system] {system_text}")
    images: list[str] = []
    for m in body.messages:
        parts.append(f"[{m.role}] {_flatten_anthropic_content(m.content)}")
        images.extend(_extract_anthropic_images(m.content))
        images.extend(_extract_images(m.content))  # tolerate OpenAI-style blocks too
    prompt_text = "\n\n".join(parts)
    images, image_attachments = _prepare_inline_image_attachments(images)
    result = await _relay_full(
        body.model,
        prompt_text,
        images=images or None,
        attachments=image_attachments or None,
        kind="vision" if images else "chat",
    )
    reply_text = _reply_content(result)
    message_id = _new_resource_id("msg")
    usage = {
        "input_tokens": _token_count(prompt_text),
        "output_tokens": _token_count(reply_text),
    }
    message = {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": body.model,
        "content": [{"type": "text", "text": reply_text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": usage,
    }
    if not body.stream:
        return message

    async def events() -> Any:
        frames = [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        **message,
                        "content": [],
                        "stop_reason": None,
                        "usage": {"input_tokens": usage["input_tokens"], "output_tokens": 0},
                    },
                },
            ),
            (
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            ),
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": reply_text}},
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": usage["output_tokens"]},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
        for event_name, data in frames:
            yield f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

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


def _materialize_image_result(
    result: Any,
    *,
    prompt: str,
    model: str,
    operation: str,
    response_format: Literal["url", "b64_json"],
    inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    worker_b64, worker_url, worker_mime = _reply_image(result)
    entry: dict[str, Any] = {
        "revised_prompt": prompt,
        "model": model,
        "operation": operation,
    }
    if inputs:
        entry["inputs"] = inputs
    if worker_b64 or worker_url:
        data, mime = _decode_media(worker_b64, worker_url)
        if data is not None:
            resolved_mime = mime or worker_mime or "image/png"
            record = _store_cloud_bytes(
                data,
                f"image{_ext_for_mime(resolved_mime)}",
                purpose="output",
                mime_type=resolved_mime,
            )
            artifact_url = _cloud_file_url(record["id"])
            entry.update(
                {
                    "file_id": record["id"],
                    "source": "worker",
                    "mime_type": resolved_mime,
                    "artifact": {
                        "source": "worker",
                        "file_id": record["id"],
                        "url": artifact_url,
                        "mime_type": resolved_mime,
                        "bytes": len(data),
                    },
                }
            )
            if response_format == "b64_json":
                entry["b64_json"] = (
                    worker_b64
                    or base64.b64encode(data).decode("ascii")
                )
            else:
                entry["url"] = artifact_url
            return entry
        entry.update(
            {
                "url": worker_url,
                "source": "worker",
                "mime_type": worker_mime,
                "artifact": {
                    "source": "worker",
                    "url": worker_url,
                    "mime_type": worker_mime,
                },
            }
        )
        return entry

    if response_format == "b64_json":
        entry["b64_json"] = _STUB_PIXEL_PNG_DATA_URL.split(",", 1)[1]
    else:
        entry["url"] = _STUB_PIXEL_PNG_DATA_URL
    description = _reply_content(result) if result is not None else ""
    if description:
        entry["pretend_description"] = description
    stub_data = base64.b64decode(
        _STUB_PIXEL_PNG_DATA_URL.split(",", 1)[1]
    )
    entry.update(
        {
            "source": "simulated",
            "mime_type": "image/png",
            "artifact": {
                "source": "simulated",
                "url": _STUB_PIXEL_PNG_DATA_URL,
                "mime_type": "image/png",
                "bytes": len(stub_data),
            },
        }
    )
    return entry


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
    copilot_backing = _copilot_backing_model(body.model)
    copilot_metadata = (
        _copilot_model_metadata(copilot_backing)
        if copilot_backing
        else None
    )
    copilot_entry = (
        _copilot_catalog_model_entry(copilot_metadata)
        if copilot_metadata
        else None
    )
    can_pretend = (
        True
        if copilot_entry
        and (
            (copilot_entry.get("task_capabilities") or {})
            .get("image_output", {})
            .get("enabled")
        )
        else await _capable_or_policy(worker_id, "images")
    )
    result: Any = None
    if can_pretend:
        result = await _relay_full(
            body.model,
            "(image-generation) Create a real PNG image for the prompt below. Use your "
            "enabled file/terminal tools and write it as 'emullm-generated-image.png' "
            "inside the current workspace. You may generate PNG bytes with Python's "
            "standard library. When the file is complete, reply only: "
            "EMULLM_IMAGE_FILE: emullm-generated-image.png\n\n"
            f"Image prompt: {body.prompt}",
            kind="image",
            required_capabilities={"image_output"},
        )
    entry = _materialize_image_result(
        result,
        prompt=body.prompt,
        model=body.model,
        operation="generation",
        response_format=body.response_format,
    )
    return {"created": int(time.time()), "data": [dict(entry) for _ in range(body.n)]}


async def _store_image_edit_upload(
    upload: UploadFile,
    index: int,
) -> dict[str, Any]:
    data = bytearray()
    try:
        while chunk := await upload.read(1024 * 1024):
            data += chunk
            if len(data) > _MAX_ADMIN_TEST_ATTACHMENT_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"attachment-{index} exceeds the image upload limit",
                )
    finally:
        await upload.close()
    if not data:
        raise HTTPException(
            status_code=400,
            detail=f"attachment-{index} is empty",
        )
    mime_type = (
        str(upload.content_type or "").split(";", 1)[0].strip().lower()
        or "application/octet-stream"
    )
    if not mime_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"attachment-{index} must be an image",
        )
    name = _anonymous_attachment_name(index)
    record = _store_cloud_bytes(
        bytes(data),
        name,
        purpose="user_data",
        mime_type=mime_type,
    )
    return {
        "file_id": record["id"],
        "name": name,
        "url": _cloud_file_url(record["id"]),
        "mime_type": mime_type,
        "bytes": len(data),
    }


@router.post("/v1/images/edits")
async def images_edits(request: Request) -> dict[str, Any]:
    if not request.headers.get("content-type", "").startswith(
        "multipart/form-data"
    ):
        raise HTTPException(
            status_code=415,
            detail="use multipart/form-data with image and optional mask fields",
        )
    form = await request.form()
    source_upload = form.get("image")
    mask_upload = form.get("mask")
    if not isinstance(source_upload, UploadFile):
        raise HTTPException(status_code=400, detail="image is required")
    if mask_upload is not None and not isinstance(mask_upload, UploadFile):
        raise HTTPException(status_code=400, detail="mask must be an image file")
    prompt = str(form.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    model = str(form.get("model") or "copilot/gpt-5.3-codex")
    response_format = str(form.get("response_format") or "url")
    if response_format not in {"url", "b64_json"}:
        raise HTTPException(
            status_code=422,
            detail="response_format must be url or b64_json",
        )
    response_format_value: Literal["url", "b64_json"] = (
        "b64_json" if response_format == "b64_json" else "url"
    )
    try:
        count = int(str(form.get("n") or "1"))
    except ValueError as error:
        raise HTTPException(status_code=422, detail="n must be an integer") from error
    if not 1 <= count <= 10:
        raise HTTPException(status_code=422, detail="n must be between 1 and 10")
    size = str(form.get("size") or "256x256")

    source = await _store_image_edit_upload(source_upload, 1)
    mask = (
        await _store_image_edit_upload(mask_upload, 2)
        if isinstance(mask_upload, UploadFile)
        else None
    )
    inputs = {"image": source, "mask": mask, "size": size}
    attachments = [source, *([mask] if mask else [])]

    worker_id, _, _ = _require_model(model)
    copilot_backing = _copilot_backing_model(model)
    metadata = (
        _copilot_model_metadata(copilot_backing)
        if copilot_backing
        else None
    )
    copilot_entry = (
        _copilot_catalog_model_entry(metadata)
        if metadata
        else None
    )
    can_generate = (
        True
        if copilot_entry
        and (
            (copilot_entry.get("task_capabilities") or {})
            .get("image_output", {})
            .get("enabled")
        )
        else await _capable_or_policy(worker_id, "images")
    )
    result: Any = None
    if can_generate:
        mask_instruction = (
            "Attachment-2 is the mask; modify the masked/transparent region only."
            if mask
            else "No mask was supplied; edit the complete source image."
        )
        result = await _relay_full(
            model,
            "(image-edit) Attachment-1 is the source image. "
            f"{mask_instruction} Create a {size} PNG satisfying the prompt. "
            "Use enabled tools and write 'emullm-generated-image.png' in the "
            "current workspace. Reply only: "
            "EMULLM_IMAGE_FILE: emullm-generated-image.png\n\n"
            f"Edit prompt: {prompt}",
            images=[item["url"] for item in attachments],
            files={"image_edit": inputs},
            attachments=attachments,
            kind="image_edit",
            required_capabilities={"vision_input", "image_output"},
        )
    entry = _materialize_image_result(
        result,
        prompt=prompt,
        model=model,
        operation="edit",
        response_format=response_format_value,
        inputs=inputs,
    )
    return {
        "created": int(time.time()),
        "data": [dict(entry) for _ in range(count)],
    }


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
        "data": [
            _model_entry(worker_id, suffix, persona)
            for suffix, persona in _models_for(worker_id).items()
            if suffix not in _HIDDEN_PERSONA_SUFFIXES
        ]
    }


@router.get("/emullm/specific_worker/{worker_id}/v1/models/{model_id:path}")
def specific_worker_get_model(worker_id: str, model_id: str) -> dict[str, Any]:
    _, suffix = _split_model_id(model_id)
    if suffix in _HIDDEN_PERSONA_SUFFIXES:
        raise HTTPException(
            status_code=404,
            detail=f"legacy persona model '{model_id}' is not exported",
        )
    persona = _models_for(worker_id).get(suffix)
    if persona is None:
        raise HTTPException(status_code=404, detail=f"unknown model '{model_id}' for worker '{worker_id}'")
    return _model_entry(worker_id, suffix, persona)


@router.post("/emullm/specific_worker/{worker_id}/v1/chat/completions")
async def specific_worker_chat_completions(worker_id: str, body: ChatRequest) -> dict[str, Any]:
    body.model = _force_worker_id(body.model, worker_id)
    return await chat_completions(body)


@router.post("/emullm/specific_worker/{worker_id}/v1/messages")
async def specific_worker_messages(worker_id: str, body: MessagesRequest) -> Any:
    body.model = _force_worker_id(body.model, worker_id)
    return await anthropic_messages(body)


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

    def count(self) -> int:
        with self._lock:
            try:
                return sum(
                    1
                    for entry in os.scandir(self._dir)
                    if entry.is_file() and entry.name.endswith(".json")
                )
            except OSError:
                return 0

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


# ---------------------------------------------------------------------------
# Worker mailboxes are durable service channels, not transient WebSocket
# bookkeeping. Their definitions/cursors live in config/mailboxes.json and
# each ordered stream lives in events_logs/<mailbox>.jsonl. This mirrors the
# on-disk layout used by the collaboration services while keeping the worker's
# existing WebSocket request/reply protocol unchanged.
# ---------------------------------------------------------------------------
_MAILBOX_CONFIG_SCHEMA_VERSION = 2
_MAILBOX_MAX_ID_LENGTH = 64
_mailbox_lock = threading.RLock()
_mailbox_event_summary_cache: dict[Path, tuple[int, int, int, str | None]] = {}
_LLM_USER_ID = "LLM_USER"
_LLM_USER_MAILBOX = "websock_to_llm_user"


def _mailbox_id(value: Any) -> str:
    """Validate the portable mailbox identifier shared by REST, files, and WS."""
    mailbox = str(value or "").strip()
    if (
        not mailbox
        or len(mailbox) > _MAILBOX_MAX_ID_LENGTH
        or not mailbox[0].isalnum()
        or any(not (character.isascii() and (character.isalnum() or character in "._-")) for character in mailbox)
    ):
        raise ValueError(
            "mailbox id must match ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
        )
    return mailbox


def _mailbox_agent_id(value: Any) -> str:
    agent = str(value or "").strip()
    if not agent or len(agent) > 255 or "\x00" in agent:
        raise ValueError("agent id must be a non-empty string no longer than 255 characters")
    return agent


def _mailbox_config_path() -> Path:
    return _RUNTIME_DIR / "config" / "mailboxes.json"


def _mailbox_events_dir() -> Path:
    return _RUNTIME_DIR / "events_logs"


def _empty_mailbox_config() -> dict[str, Any]:
    return {
        "schema_version": _MAILBOX_CONFIG_SCHEMA_VERSION,
        "mailboxes": {},
        "agents": {},
        "cursors": {},
    }


def _load_mailbox_config() -> dict[str, Any]:
    """Read the durable mailbox configuration, failing loudly if it is corrupt."""
    path = _mailbox_config_path()
    if not path.is_file():
        return _empty_mailbox_config()
    try:
        config = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read mailbox configuration at {path}") from exc
    if not isinstance(config, dict):
        raise RuntimeError(f"mailbox configuration at {path} must be a JSON object")
    for key in ("mailboxes", "agents", "cursors"):
        value = config.get(key)
        if value is None:
            config[key] = {}
        elif not isinstance(value, dict):
            raise RuntimeError(f"mailbox configuration field '{key}' must be an object")
    try:
        schema_version = int(config.get("schema_version") or 1)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"mailbox configuration at {path} has an invalid schema_version") from exc
    if schema_version < 2:
        for record in config["mailboxes"].values():
            if not isinstance(record, dict):
                continue
            if record.get("source") == "events_logs":
                record["source"] = "jsonl"
            if record.get("transports") == ["events_logs", "websocket"]:
                record["transports"] = ["jsonl", "ws"]
        config["schema_version"] = _MAILBOX_CONFIG_SCHEMA_VERSION
        _save_mailbox_config(config)
    else:
        config["schema_version"] = schema_version
    return config


def _save_mailbox_config(config: dict[str, Any]) -> None:
    path = _mailbox_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".mailboxes.{uuid.uuid4().hex}.json.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(config, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _mailbox_event_log_path(mailbox: str) -> Path:
    return _mailbox_events_dir() / f"{_mailbox_id(mailbox)}.jsonl"


def _mailbox_events(mailbox: str) -> list[dict[str, Any]]:
    """Read one mailbox's append-only JSONL event log in sequence order."""
    path = _mailbox_event_log_path(mailbox)
    with _mailbox_lock:
        if not path.is_file():
            return []
        events: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8-sig") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise RuntimeError(
                            f"mailbox event log {path} has a non-object event on line {line_number}"
                        )
                    events.append(event)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read mailbox event log at {path}") from exc
    return events


def _mailbox_event_summary(mailbox: str) -> tuple[int, str | None]:
    """Count a JSONL stream without deserializing its complete history."""
    path = _mailbox_event_log_path(mailbox)
    with _mailbox_lock:
        if not path.is_file():
            _mailbox_event_summary_cache.pop(path, None)
            return 0, None
        metadata = path.stat()
        cached = _mailbox_event_summary_cache.get(path)
        if cached and cached[:2] == (metadata.st_mtime_ns, metadata.st_size):
            return cached[2], cached[3]
        count = 0
        last_line: bytes | None = None
        with path.open("rb") as handle:
            for line in handle:
                if line.strip():
                    count += 1
                    last_line = line
        last_activity = None
        if last_line is not None:
            try:
                last_event = json.loads(last_line.decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"cannot read mailbox event log at {path}") from exc
            if isinstance(last_event, dict):
                last_activity = last_event.get("ts")
        _mailbox_event_summary_cache[path] = (
            metadata.st_mtime_ns,
            metadata.st_size,
            count,
            last_activity,
        )
        return count, last_activity


def _mailbox_json_value(value: Any) -> Any:
    """Make an observed worker value safely representable in an event log."""
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except (TypeError, ValueError):
        try:
            return json.loads(json.dumps(value, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            return str(value)


def _ensure_mailbox(
    mailbox: str,
    *,
    purpose: str | None = None,
    hidden: bool | None = None,
    writable: bool | None = None,
    source: str | None = None,
    transports: list[str] | None = None,
) -> dict[str, Any]:
    """Create or update one durable mailbox descriptor."""
    mailbox = _mailbox_id(mailbox)
    with _mailbox_lock:
        _mailbox_events_dir().mkdir(parents=True, exist_ok=True)
        config = _load_mailbox_config()
        existing = config["mailboxes"].get(mailbox)
        record = dict(existing) if isinstance(existing, dict) else {}
        created = not record
        now = _now_iso()
        if created:
            record = {
                "id": mailbox,
                "name": mailbox,
                "global_name": mailbox,
                "purpose": purpose or f"LLM relay mailbox for worker '{mailbox}'",
                "kind": "mailbox",
                "source": source or "jsonl",
                "transports": transports or ["jsonl", "ws"],
                "hidden": bool(hidden) if hidden is not None else False,
                "writable": bool(writable) if writable is not None else True,
                "created_at": now,
                "updated_at": now,
            }
        else:
            changed = False
            for key, value in (
                ("purpose", purpose),
                ("hidden", hidden),
                ("writable", writable),
                ("source", source),
                ("transports", transports),
            ):
                if value is not None and record.get(key) != value:
                    record[key] = value
                    changed = True
            for key, value in (
                ("id", mailbox),
                ("name", mailbox),
                ("global_name", mailbox),
                ("kind", "mailbox"),
            ):
                if record.get(key) != value:
                    record[key] = value
                    changed = True
            if changed:
                record["updated_at"] = now
        config["mailboxes"][mailbox] = record
        if created or record != existing:
            _save_mailbox_config(config)
        return dict(record)


def _ensure_llm_user_mailbox() -> dict[str, Any]:
    return _ensure_mailbox(
        _LLM_USER_MAILBOX,
        purpose="Aggregate LLM_USER to worker request, acceptance, rejection, and reply log",
        writable=False,
        source="jsonl",
        transports=["jsonl", "ws"],
    )


def _upsert_mailbox_agent(agent_id: str, properties: dict[str, Any] | None = None) -> dict[str, Any]:
    agent_id = _mailbox_agent_id(agent_id)
    with _mailbox_lock:
        config = _load_mailbox_config()
        existing = config["agents"].get(agent_id)
        record = dict(existing) if isinstance(existing, dict) else {}
        changed = record.get("id") != agent_id
        record["id"] = agent_id
        if properties:
            safe_properties = _mailbox_json_value(properties)
            for key, value in safe_properties.items():
                if record.get(key) != value:
                    record[key] = value
                    changed = True
        if changed:
            record["updated_at"] = _now_iso()
            config["agents"][agent_id] = record
            _save_mailbox_config(config)
        return dict(record)


def _ensure_worker_mailbox(worker_id: str) -> dict[str, Any]:
    """Expose a connected servant as its identically named mailbox and agent."""
    mailbox = _ensure_mailbox(
        worker_id,
        writable=True,
        source="jsonl",
        transports=["jsonl", "ws"],
    )
    _upsert_mailbox_agent(
        worker_id,
        {
            "kind": "worker",
            "worker_id": worker_id,
            "role": _worker_roles.get(worker_id, _DEFAULT_WORKER_ROLE),
        },
    )
    return mailbox


def _mailbox_is_known(mailbox: str) -> bool:
    mailbox = _mailbox_id(mailbox)
    with _mailbox_lock:
        config = _load_mailbox_config()
        return mailbox in config["mailboxes"] or mailbox in _connected_workers


def _require_mailbox(mailbox: str) -> str:
    try:
        mailbox = _mailbox_id(mailbox)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if mailbox in _connected_workers:
        _ensure_worker_mailbox(mailbox)
    if not _mailbox_is_known(mailbox):
        raise HTTPException(status_code=404, detail=f"no mailbox named '{mailbox}'")
    return mailbox


def _append_mailbox_event(
    stream: str,
    event_type: str,
    data: dict[str, Any],
    *,
    source_id: str,
    source_kind: str,
    correlation_id: str | None = None,
    idempotency_key: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """Append one event atomically enough for a local durable JSONL stream."""
    try:
        stream = _mailbox_id(stream)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    event_type = str(event_type or "").strip()
    if not event_type:
        raise HTTPException(status_code=400, detail="event type is required")
    source_id = _mailbox_agent_id(source_id)
    source_kind = str(source_kind or "service").strip() or "service"
    idempotency_key = str(idempotency_key).strip() if idempotency_key else None

    with _mailbox_lock:
        _ensure_mailbox(stream)
        events = _mailbox_events(stream)
        if idempotency_key:
            for event in reversed(events):
                if event.get("idempotency_key") == idempotency_key:
                    return event, True
        sequence = max((int(event.get("seq") or 0) for event in events), default=0) + 1
        event: dict[str, Any] = {
            "id": _new_resource_id("evt"),
            "stream": stream,
            "seq": sequence,
            "type": event_type,
            "ts": _now_iso(),
            "schema_version": 1,
            "source_id": source_id,
            "source_kind": source_kind,
            "data": _mailbox_json_value(data),
        }
        if correlation_id:
            event["correlation_id"] = str(correlation_id)
        if idempotency_key:
            event["idempotency_key"] = idempotency_key
        path = _mailbox_event_log_path(stream)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise RuntimeError(f"cannot append mailbox event to {path}") from exc
        return event, False


def _mailbox_event_to_message(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    text = data.get("text", data.get("prompt", ""))
    if isinstance(text, (dict, list)):
        text = json.dumps(text, ensure_ascii=False)
    return {
        "id": str(event.get("id") or ""),
        "timestamp": event.get("ts"),
        "from": data.get("from") or event.get("source_id"),
        "to": data.get("to"),
        "send_to": data.get("send_to"),
        "text": str(text or ""),
        "type": event.get("type"),
        "mailboxId": event.get("stream"),
        "mailboxName": event.get("stream"),
        "author": data.get("author") or event.get("source_id"),
        "authorName": data.get("authorName") or data.get("author") or event.get("source_id"),
        "raw": event,
    }


def _mailbox_endpoint_paths(mailbox: str) -> dict[str, str]:
    mailbox = _mailbox_id(mailbox)
    base = "/ws_collab/v1"
    if mailbox == _LLM_USER_MAILBOX:
        return {
            "events": "/emullm/websock_to_llm_user/events",
            "ws": "/emullm/websock_to_llm_user/ws",
            "tail": f"{base}/streams/{mailbox}/tail",
        }
    return {
        "read": f"{base}/mailbox/messages?mailbox={mailbox}",
        "send": f"{base}/mailbox/send",
        "tail": f"{base}/streams/{mailbox}/tail",
        "events": f"{base}/events?stream={mailbox}",
        "ws": "/ws_collab/ws",
        "worker_ws": f"/emullm/ws?worker_id={mailbox}",
    }


def _mailbox_cursor_payload(mailbox: str, agent: str) -> dict[str, Any]:
    mailbox = _require_mailbox(mailbox)
    try:
        agent = _mailbox_agent_id(agent)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with _mailbox_lock:
        config = _load_mailbox_config()
        events = _mailbox_events(mailbox)
        record = config["cursors"].get(f"{agent}\x1f{mailbox}")
        initialized = isinstance(record, dict) and bool(record.get("initialized"))
        offset = int(record.get("offset") or 0) if initialized else 0
        offset = max(0, min(offset, len(events)))
    return {
        "mailbox": mailbox,
        "agent": agent,
        "initialized": initialized,
        "offset": offset,
        "size": len(events),
        "behind": len(events) - offset,
        "entries_consumed": offset,
        "entry_next": offset,
        "entries_total": len(events),
        "last_read_id": events[offset - 1].get("id") if offset else None,
        "next_unread_id": events[offset].get("id") if offset < len(events) else None,
    }


def _set_mailbox_cursor(mailbox: str, agent: str, start: str) -> dict[str, Any]:
    mailbox = _require_mailbox(mailbox)
    try:
        agent = _mailbox_agent_id(agent)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if start not in {"now", "beginning"}:
        raise HTTPException(status_code=400, detail="cursor start must be 'now' or 'beginning'")
    with _mailbox_lock:
        config = _load_mailbox_config()
        offset = len(_mailbox_events(mailbox)) if start == "now" else 0
        config["cursors"][f"{agent}\x1f{mailbox}"] = {
            "agent": agent,
            "mailbox": mailbox,
            "initialized": True,
            "offset": offset,
            "updated_at": _now_iso(),
        }
        _save_mailbox_config(config)
    return _mailbox_cursor_payload(mailbox, agent)


def _clear_mailbox_cursor(mailbox: str, agent: str) -> dict[str, Any]:
    mailbox = _require_mailbox(mailbox)
    try:
        agent = _mailbox_agent_id(agent)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with _mailbox_lock:
        config = _load_mailbox_config()
        config["cursors"].pop(f"{agent}\x1f{mailbox}", None)
        _save_mailbox_config(config)
    return _mailbox_cursor_payload(mailbox, agent)


def _mailbox_event_count() -> int:
    directory = _mailbox_events_dir()
    with _mailbox_lock:
        if not directory.is_dir():
            return 0
        paths = list(directory.glob("*.jsonl"))
    return sum(_mailbox_event_summary(path.stem)[0] for path in paths)


def _mailbox_count() -> int:
    with _mailbox_lock:
        config = _load_mailbox_config()
        return sum(
            1 for record in config["mailboxes"].values() if isinstance(record, dict)
        )


def _clear_mailbox_storage() -> dict[str, int]:
    """Remove mailbox configuration and every stream log for admin/test reset."""
    with _mailbox_lock:
        removed = {"mailbox_config": 0, "mailbox_events": 0}
        config = _mailbox_config_path()
        if config.is_file():
            config.unlink()
            removed["mailbox_config"] = 1
        directory = _mailbox_events_dir()
        if directory.is_dir():
            for path in directory.glob("*.jsonl"):
                path.unlink()
                removed["mailbox_events"] += 1
        return removed


def _record_relay_request(worker_id: str, request_id: str, payload: dict[str, Any]) -> None:
    data = {
        "text": payload.get("prompt", ""),
        "prompt": payload.get("prompt", ""),
        "model": payload.get("model"),
        "worker_id": worker_id,
        "from": _LLM_USER_ID,
        "to": worker_id,
        "send_to": worker_id,
        "direction": "LLM_USER->worker",
        "request": _mailbox_json_value(payload),
    }
    _append_mailbox_event(
        worker_id,
        "LLM_REQUEST",
        data,
        source_id=_LLM_USER_ID,
        source_kind="user",
        correlation_id=request_id,
        idempotency_key=f"request:{request_id}",
    )
    _append_mailbox_event(
        _LLM_USER_MAILBOX,
        "LLM_REQUEST",
        data,
        source_id=_LLM_USER_ID,
        source_kind="user",
        correlation_id=request_id,
        idempotency_key=f"request:{request_id}",
    )


def _record_relay_reply(worker_id: str, request_id: str, reply: Any, model: str | None = None) -> None:
    text = reply.get("content", "") if isinstance(reply, dict) else reply
    data = {
        "text": str(text or ""),
        "model": model,
        "worker_id": worker_id,
        "from": worker_id,
        "to": _LLM_USER_ID,
        "send_to": None,
        "direction": "worker->LLM_USER",
        "reply": _mailbox_json_value(reply),
    }
    _append_mailbox_event(
        worker_id,
        "LLM_REPLY",
        data,
        source_id=worker_id,
        source_kind="worker",
        correlation_id=request_id or None,
        idempotency_key=f"reply:{request_id}" if request_id else None,
    )
    _append_mailbox_event(
        _LLM_USER_MAILBOX,
        "LLM_REPLY",
        data,
        source_id=worker_id,
        source_kind="worker",
        correlation_id=request_id or None,
        idempotency_key=f"reply:{request_id}" if request_id else None,
    )


def _record_relay_accept(worker_id: str, request_id: str, model: str | None = None) -> None:
    data = {
        "model": model,
        "worker_id": worker_id,
        "from": worker_id,
        "to": _LLM_USER_ID,
        "send_to": None,
        "direction": "worker->LLM_USER",
    }
    _append_mailbox_event(
        worker_id,
        "LLM_ACCEPT",
        data,
        source_id=worker_id,
        source_kind="worker",
        correlation_id=request_id,
        idempotency_key=f"accept:{request_id}",
    )
    _append_mailbox_event(
        _LLM_USER_MAILBOX,
        "LLM_ACCEPT",
        data,
        source_id=worker_id,
        source_kind="worker",
        correlation_id=request_id,
        idempotency_key=f"accept:{request_id}",
    )


def _record_relay_rejection(worker_id: str, request_id: str, reason: str, model: str | None = None) -> None:
    data = {
        "text": reason,
        "reason": reason,
        "model": model,
        "worker_id": worker_id,
        "from": worker_id,
        "to": _LLM_USER_ID,
        "send_to": None,
        "direction": "worker->LLM_USER",
    }
    _append_mailbox_event(
        worker_id,
        "LLM_REJECT",
        data,
        source_id=worker_id,
        source_kind="worker",
        correlation_id=request_id,
        idempotency_key=f"reject:{request_id}",
    )
    _append_mailbox_event(
        _LLM_USER_MAILBOX,
        "LLM_REJECT",
        data,
        source_id=worker_id,
        source_kind="worker",
        correlation_id=request_id,
        idempotency_key=f"reject:{request_id}",
    )


def _record_relay_not_ready(
    worker_id: str,
    request_id: str,
    reason: str,
    retry_after: float,
    model: str | None = None,
) -> None:
    data = {
        "text": reason,
        "reason": reason,
        "reported_retry_after": retry_after,
        "cooldown_applied": False,
        "model": model,
        "worker_id": worker_id,
        "from": worker_id,
        "to": _LLM_USER_ID,
        "send_to": None,
        "direction": "worker->LLM_USER",
    }
    for stream in (worker_id, _LLM_USER_MAILBOX):
        _append_mailbox_event(
            stream,
            "LLM_NOT_READY",
            data,
            source_id=worker_id,
            source_kind="worker",
            correlation_id=request_id,
            idempotency_key=f"not-ready:{request_id}",
        )


def _record_worker_connection(worker_id: str, model_masks: tuple[str, ...] | None) -> None:
    _append_mailbox_event(
        worker_id,
        "WORKER_CONNECTED",
        {
            "worker_id": worker_id,
            "modelmasks": list(model_masks) if model_masks is not None else None,
        },
        source_id="emullm",
        source_kind="service",
    )


def _record_worker_disconnection(worker_id: str) -> None:
    _append_mailbox_event(
        worker_id,
        "WORKER_DISCONNECTED",
        {"worker_id": worker_id},
        source_id="emullm",
        source_kind="service",
    )


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


def _mailbox_descriptor(record: dict[str, Any], agent: str | None = None) -> dict[str, Any]:
    mailbox = _mailbox_id(record.get("id"))
    message_count, last_activity = _mailbox_event_summary(mailbox)
    descriptor: dict[str, Any] = {
        "id": mailbox,
        "name": str(record.get("name") or mailbox),
        "global_name": str(record.get("global_name") or mailbox),
        "purpose": str(record.get("purpose") or f"LLM relay mailbox for worker '{mailbox}'"),
        "kind": str(record.get("kind") or "mailbox"),
        "source": str(record.get("source") or "jsonl"),
        "transports": list(record.get("transports") or ["jsonl", "ws"]),
        "storage": "events_logs",
        "hidden": bool(record.get("hidden")),
        "writable": bool(record.get("writable", True)),
        "messages": message_count,
        "connected": mailbox in _connected_workers,
        "last_activity": last_activity,
        "filename": str(_mailbox_event_log_path(mailbox)),
        "endpoints": _mailbox_endpoint_paths(mailbox),
    }
    if agent:
        cursor = _mailbox_cursor_payload(mailbox, agent)
        descriptor.update(
            {
                "unread": cursor["behind"],
                "cursorOffset": cursor["offset"],
                "cursorInitialized": cursor["initialized"],
                "lastReadMessageId": cursor["last_read_id"],
                "nextUnreadMessageId": cursor["next_unread_id"],
            }
        )
    return descriptor


def _mailbox_directory(agent: str | None = None, include_activity: bool = False) -> list[dict[str, Any]]:
    for worker_id in tuple(_connected_workers):
        _ensure_worker_mailbox(worker_id)
    if agent:
        try:
            agent = _mailbox_agent_id(agent)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    with _mailbox_lock:
        config = _load_mailbox_config()
        records = [
            dict(record)
            for record in config["mailboxes"].values()
            if isinstance(record, dict) and (include_activity or not bool(record.get("hidden")))
        ]
    return sorted((_mailbox_descriptor(record, agent) for record in records), key=lambda item: item["id"])


def _mailbox_event_page(
    stream: str,
    *,
    after: str | None,
    limit: int,
    event_type: str | None = None,
    source_id: str | None = None,
    correlation_id: str | None = None,
) -> tuple[list[dict[str, Any]], str | None, bool]:
    stream = _require_mailbox(stream)
    if not 1 <= limit <= 1000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 1000")
    events = _mailbox_events(stream)
    start = 0
    if after:
        matching_index = next((index for index, event in enumerate(events) if event.get("id") == after), None)
        if matching_index is None:
            raise HTTPException(status_code=409, detail="cursor_invalid")
        start = matching_index + 1
    candidates = events[start:]
    if event_type:
        candidates = [event for event in candidates if event.get("type") == event_type]
    if source_id:
        candidates = [event for event in candidates if event.get("source_id") == source_id]
    if correlation_id:
        candidates = [event for event in candidates if event.get("correlation_id") == correlation_id]
    selected = candidates[:limit]
    return selected, (selected[-1].get("id") if selected else after), len(candidates) > len(selected)


def _llm_user_event_matches(
    event: dict[str, Any],
    *,
    worker_id: str | None = None,
    model: str | None = None,
    modelmask: str | None = None,
    event_types: set[str] | None = None,
) -> bool:
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    if worker_id and data.get("worker_id") != worker_id:
        return False
    if model and data.get("model") != model:
        return False
    if modelmask and not fnmatchcase(str(data.get("model") or ""), modelmask):
        return False
    return not event_types or event.get("type") in event_types


def _llm_user_event_page(
    *,
    after: str | None,
    limit: int,
    worker_id: str | None = None,
    model: str | None = None,
    modelmask: str | None = None,
    event_types: set[str] | None = None,
) -> tuple[list[dict[str, Any]], str | None, bool]:
    """Read a filtered page while advancing past filtered-out JSONL events."""
    if not 1 <= limit <= 1000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 1000")
    _ensure_llm_user_mailbox()
    events = _mailbox_events(_LLM_USER_MAILBOX)
    start = 0
    if after:
        index = next((index for index, event in enumerate(events) if event.get("id") == after), None)
        if index is None:
            raise HTTPException(status_code=409, detail="cursor_invalid")
        start = index + 1

    selected: list[dict[str, Any]] = []
    cursor = after
    index = start
    while index < len(events) and len(selected) < limit:
        event = events[index]
        cursor = str(event.get("id") or "") or cursor
        if _llm_user_event_matches(
            event,
            worker_id=worker_id,
            model=model,
            modelmask=modelmask,
            event_types=event_types,
        ):
            selected.append(event)
        index += 1
    return selected, cursor, index < len(events)


def _parse_event_types(values: list[str]) -> set[str] | None:
    types = {item.strip() for value in values for item in value.split(",") if item.strip()}
    return types or None


class MailboxCreateRequest(BaseModel):
    id: str | None = None
    name: str | None = None
    purpose: str | None = None
    hidden: bool | None = None
    writable: bool | None = None
    source: str | None = None


class MailboxSendRequest(BaseModel):
    to: str | None = None
    text: str
    sender: str = "operator"
    send_to: str | None = None
    correlation_id: str | None = None
    idempotency_key: str | None = None


class MailboxAgentRequest(BaseModel):
    id: str
    properties: dict[str, Any] = Field(default_factory=dict)


class MailboxCursorRequest(BaseModel):
    mailbox: str
    agent: str
    start: Literal["now", "beginning"] = "now"


class MailboxEventRequest(BaseModel):
    stream: str
    type: str
    data: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = None
    idempotency_key: str | None = None
    source_id: str = "external"
    source_kind: str = "service"


@router.get("/capabilities")
@router.get("/emullm/capabilities")
@router.get("/ws_collab/v1/capabilities")
@router.get("/mailbox_chat/v1/capabilities")
def mailbox_capabilities() -> dict[str, Any]:
    mailboxes = _mailbox_directory(include_activity=True)
    return {
        "service": "emullm",
        "mailboxes": mailboxes,
        "streams": [mailbox["id"] for mailbox in mailboxes],
        "rest_base": "/ws_collab/v1",
        "transports": ["jsonl", "ws"],
        "server_time": _now_iso(),
    }


@router.get("/mailbox/mailboxes")
@router.get("/api/mailbox/mailboxes")
@router.get("/emullm/mailbox/mailboxes")
@router.get("/ws_collab/v1/mailbox/mailboxes")
@router.get("/mailbox_chat/v1/mailbox/mailboxes")
def mailbox_mailboxes(agent: str | None = None, include_activity: bool = False) -> dict[str, Any]:
    return {
        "place": "emullm",
        "global_name": "emullm",
        "mailboxes": _mailbox_directory(agent=agent, include_activity=include_activity),
        "server_time": _now_iso(),
    }


@router.post("/mailbox/create")
@router.post("/api/mailbox/create")
@router.post("/emullm/mailbox/create")
@router.post("/ws_collab/v1/mailbox/create")
@router.post("/mailbox_chat/v1/mailbox/create")
@router.post("/mailbox/mailboxes")
@router.post("/api/mailbox/mailboxes")
@router.post("/emullm/mailbox/mailboxes")
@router.post("/ws_collab/v1/mailbox/mailboxes")
@router.post("/mailbox_chat/v1/mailbox/mailboxes")
def mailbox_create(body: MailboxCreateRequest) -> dict[str, Any]:
    proposed = body.id or body.name
    if body.id and body.name and body.id != body.name:
        raise HTTPException(status_code=400, detail="id and name must match when both are provided")
    try:
        mailbox = _mailbox_id(proposed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    created = not _mailbox_is_known(mailbox)
    record = _ensure_mailbox(
        mailbox,
        purpose=body.purpose.strip() if body.purpose else None,
        hidden=body.hidden,
        writable=body.writable,
        source=body.source.strip() if body.source else None,
    )
    return {"created": created, "mailbox": _mailbox_descriptor(record)}


@router.get("/mailbox/agents")
@router.get("/api/mailbox/agents")
@router.get("/emullm/mailbox/agents")
@router.get("/ws_collab/v1/mailbox/agents")
@router.get("/mailbox_chat/v1/mailbox/agents")
def mailbox_agents() -> dict[str, Any]:
    for worker_id in tuple(_connected_workers):
        _ensure_worker_mailbox(worker_id)
    with _mailbox_lock:
        config = _load_mailbox_config()
        agents = [dict(agent) for agent in config["agents"].values() if isinstance(agent, dict)]
    return {"agents": sorted(agents, key=lambda agent: str(agent.get("id") or ""))}


@router.post("/mailbox/agents")
@router.post("/api/mailbox/agents")
@router.post("/emullm/mailbox/agents")
@router.post("/ws_collab/v1/mailbox/agents")
@router.post("/mailbox_chat/v1/mailbox/agents")
def mailbox_register_agent(body: MailboxAgentRequest) -> dict[str, Any]:
    try:
        return _upsert_mailbox_agent(body.id, body.properties)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/mailbox/messages")
@router.get("/api/mailbox/messages")
@router.get("/emullm/mailbox/messages")
@router.get("/ws_collab/v1/mailbox/messages")
@router.get("/mailbox_chat/v1/mailbox/messages")
def mailbox_messages(
    mailbox: str,
    from_: str | None = Query(default=None, alias="from"),
    to: str | None = None,
    send_to: str | None = None,
    text: str | None = None,
    filter: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    mailbox = _require_mailbox(mailbox)
    if not 1 <= limit <= 1000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 1000")
    messages = [_mailbox_event_to_message(event) for event in _mailbox_events(mailbox)]
    if from_:
        messages = [message for message in messages if message["from"] == from_]
    if to:
        messages = [message for message in messages if message["to"] == to]
    if send_to:
        messages = [message for message in messages if message["send_to"] == send_to]
    match_text = text or filter
    if match_text:
        needle = match_text.casefold()
        messages = [message for message in messages if needle in message["text"].casefold()]
    messages = messages[-limit:]
    return {"messages": messages, "user": from_, "peer": to}


@router.post("/mailbox/send")
@router.post("/api/mailbox/send")
@router.post("/emullm/mailbox/send")
@router.post("/ws_collab/v1/mailbox/send")
@router.post("/mailbox_chat/v1/mailbox/send")
def mailbox_send(
    body: MailboxSendRequest,
    idempotency_header: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, Any]:
    target = body.send_to or body.to
    if not target:
        raise HTTPException(status_code=400, detail="send_to or to must name a mailbox")
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    mailbox = _require_mailbox(target)
    try:
        sender = _mailbox_agent_id(body.sender)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _upsert_mailbox_agent(sender, {"kind": "external"})
    event, _ = _append_mailbox_event(
        mailbox,
        "CONVERSATION_MESSAGE",
        {
            "text": body.text,
            "from": sender,
            "to": body.to,
            "send_to": body.send_to or mailbox,
            "author": sender,
            "authorName": sender,
        },
        source_id=sender,
        source_kind="agent",
        correlation_id=body.correlation_id,
        idempotency_key=idempotency_header or body.idempotency_key,
    )
    return {"message": _mailbox_event_to_message(event)}


@router.get("/mailbox/cursor")
@router.get("/api/mailbox/cursor")
@router.get("/emullm/mailbox/cursor")
@router.get("/ws_collab/v1/mailbox/cursor")
@router.get("/mailbox_chat/v1/mailbox/cursor")
def mailbox_cursor(mailbox: str, agent: str) -> dict[str, Any]:
    return _mailbox_cursor_payload(mailbox, agent)


@router.post("/mailbox/cursor")
@router.post("/api/mailbox/cursor")
@router.post("/emullm/mailbox/cursor")
@router.post("/ws_collab/v1/mailbox/cursor")
@router.post("/mailbox_chat/v1/mailbox/cursor")
def mailbox_set_cursor(body: MailboxCursorRequest) -> dict[str, Any]:
    return _set_mailbox_cursor(body.mailbox, body.agent, body.start)


@router.delete("/mailbox/cursor")
@router.delete("/api/mailbox/cursor")
@router.delete("/emullm/mailbox/cursor")
@router.delete("/ws_collab/v1/mailbox/cursor")
@router.delete("/mailbox_chat/v1/mailbox/cursor")
def mailbox_delete_cursor(mailbox: str, agent: str) -> dict[str, Any]:
    return _clear_mailbox_cursor(mailbox, agent)


@router.get("/events")
@router.get("/api/events")
@router.get("/emullm/events")
@router.get("/ws_collab/v1/events")
@router.get("/mailbox_chat/v1/events")
async def mailbox_events(
    stream: str,
    after: str | None = None,
    limit: int = 100,
    wait_ms: int = 0,
    type: str | None = None,
    source_id: str | None = None,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    if not 0 <= wait_ms <= 30000:
        raise HTTPException(status_code=400, detail="wait_ms must be between 0 and 30000")
    deadline = time.monotonic() + (wait_ms / 1000)
    while True:
        events, next_cursor, has_more = _mailbox_event_page(
            stream,
            after=after,
            limit=limit,
            event_type=type,
            source_id=source_id,
            correlation_id=correlation_id,
        )
        if events or wait_ms == 0 or time.monotonic() >= deadline:
            return {
                "stream": _require_mailbox(stream),
                "events": events,
                "next_cursor": next_cursor,
                "has_more": has_more,
                "server_time": _now_iso(),
            }
        await asyncio.sleep(min(0.1, max(0.01, deadline - time.monotonic())))


@router.get("/emullm/websock_to_llm_user/events")
@router.get("/websock_to_llm_user/events")
def websock_to_llm_user_events(
    worker_id: str | None = None,
    model: str | None = None,
    modelmask: str | None = None,
    type: str | None = None,
    after: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Read the aggregate LLM_USER-to-worker JSONL log with server-side filters."""
    try:
        normalized_worker_id = _mailbox_id(worker_id) if worker_id else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    events, next_cursor, has_more = _llm_user_event_page(
        after=after,
        limit=limit,
        worker_id=normalized_worker_id,
        model=model,
        modelmask=modelmask,
        event_types=_parse_event_types([type] if type else []),
    )
    return {
        "stream": _LLM_USER_MAILBOX,
        "events": events,
        "next_cursor": next_cursor,
        "has_more": has_more,
        "filters": {
            "worker_id": normalized_worker_id,
            "model": model,
            "modelmask": modelmask,
            "type": type,
        },
    }


@router.post("/events")
@router.post("/api/events")
@router.post("/emullm/events")
@router.post("/ws_collab/v1/events")
@router.post("/mailbox_chat/v1/events")
def mailbox_post_event(
    body: MailboxEventRequest,
    idempotency_header: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, Any]:
    try:
        _upsert_mailbox_agent(body.source_id, {"kind": body.source_kind})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    event, duplicate = _append_mailbox_event(
        body.stream,
        body.type,
        body.data,
        source_id=body.source_id,
        source_kind=body.source_kind,
        correlation_id=body.correlation_id,
        idempotency_key=idempotency_header or body.idempotency_key,
    )
    return {
        "id": event["id"],
        "seq": event["seq"],
        "cursor": event["id"],
        "duplicate": duplicate,
        "server_time": _now_iso(),
    }


@router.get("/streams/{mailbox}/tail")
@router.get("/api/streams/{mailbox}/tail")
@router.get("/emullm/streams/{mailbox}/tail")
@router.get("/ws_collab/v1/streams/{mailbox}/tail")
@router.get("/mailbox_chat/v1/streams/{mailbox}/tail")
def mailbox_stream_tail(mailbox: str, count: int = 100) -> dict[str, Any]:
    mailbox = _require_mailbox(mailbox)
    if not 1 <= count <= 2000:
        raise HTTPException(status_code=400, detail="count must be between 1 and 2000")
    return {"stream": mailbox, "events": _mailbox_events(mailbox)[-count:]}


def _mailbox_ws_cursor(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("cursor") or value.get("after")
    return str(value) if value else None


async def _mailbox_ws_flush(
    websocket: WebSocket,
    connection_id: str,
    subscriptions: dict[str, str | None],
) -> None:
    """Push available events for each subscribed mailbox, preserving cursors."""
    for stream, after in tuple(subscriptions.items()):
        try:
            events, next_cursor, _ = _mailbox_event_page(stream, after=after, limit=1000)
        except HTTPException as exc:
            subscriptions.pop(stream, None)
            await _tracked_ws_send_json(
                websocket,
                connection_id,
                {"type": "error", "stream": stream, "detail": exc.detail},
            )
            continue
        for event in events:
            await _tracked_ws_send_json(
                websocket, connection_id, {"type": "event", "event": event}
            )
        if events:
            subscriptions[stream] = next_cursor


@router.websocket("/ws_collab/ws")
@router.websocket("/mailbox_chat/ws")
@router.websocket("/mailbox/ws")
@router.websocket("/emullm/mailbox/ws")
async def mailbox_service_socket(websocket: WebSocket) -> None:
    """ws_collab/mailbox_chat stream transport for adapter clients.

    Clients send ``{"type":"subscribe","streams":[...],"cursors":{...}}``
    and receive ``{"type":"event","event":{...}}`` frames. ``publish`` (or
    ``event``) accepts a typed event using the same fields as POST /events.
    Polling the durable JSONL streams avoids a second, lossy in-memory
    fan-out path and works across the worker relay's reconnect cycles.
    """
    await websocket.accept()
    subscriptions: dict[str, str | None] = {}
    connection_id = _register_active_websocket(
        websocket, "mailbox", subscriptions=[]
    )
    try:
        while True:
            try:
                frame = await asyncio.wait_for(
                    _tracked_ws_receive_json(websocket, connection_id), timeout=0.1
                )
            except asyncio.TimeoutError:
                await _mailbox_ws_flush(websocket, connection_id, subscriptions)
                continue
            if not isinstance(frame, dict):
                await _tracked_ws_send_json(
                    websocket,
                    connection_id,
                    {"type": "error", "detail": "WebSocket frame must be an object"},
                )
                continue
            frame_type = str(frame.get("type") or "")
            if frame_type == "subscribe":
                streams = frame.get("streams")
                if not isinstance(streams, list) or not streams:
                    await _tracked_ws_send_json(
                        websocket,
                        connection_id,
                        {"type": "error", "detail": "subscribe requires a non-empty streams list"},
                    )
                    continue
                cursors = frame.get("cursors")
                cursors = cursors if isinstance(cursors, dict) else {}
                try:
                    for raw_stream in streams:
                        stream = _require_mailbox(str(raw_stream))
                        subscriptions[stream] = _mailbox_ws_cursor(cursors.get(stream))
                except HTTPException as exc:
                    await _tracked_ws_send_json(
                        websocket, connection_id, {"type": "error", "detail": exc.detail}
                    )
                    continue
                await _tracked_ws_send_json(
                    websocket,
                    connection_id,
                    {"type": "subscribed", "streams": sorted(subscriptions)},
                )
                _update_active_websocket(
                    connection_id, subscriptions=sorted(subscriptions)
                )
            elif frame_type == "unsubscribe":
                streams = frame.get("streams")
                if not isinstance(streams, list):
                    await _tracked_ws_send_json(
                        websocket,
                        connection_id,
                        {"type": "error", "detail": "unsubscribe requires a streams list"},
                    )
                    continue
                for raw_stream in streams:
                    subscriptions.pop(str(raw_stream), None)
                await _tracked_ws_send_json(
                    websocket,
                    connection_id,
                    {"type": "unsubscribed", "streams": sorted(subscriptions)},
                )
                _update_active_websocket(
                    connection_id, subscriptions=sorted(subscriptions)
                )
            elif frame_type in {"publish", "event"}:
                raw_event = frame.get("event") if isinstance(frame.get("event"), dict) else frame
                stream = raw_event.get("stream")
                event_type = raw_event.get("event_type") or raw_event.get("type")
                data = raw_event.get("data")
                if not isinstance(data, dict):
                    await _tracked_ws_send_json(
                        websocket,
                        connection_id,
                        {"type": "error", "detail": "published event data must be an object"},
                    )
                    continue
                try:
                    source_id = str(raw_event.get("source_id") or "external")
                    source_kind = str(raw_event.get("source_kind") or "service")
                    _upsert_mailbox_agent(source_id, {"kind": source_kind})
                    event, duplicate = _append_mailbox_event(
                        str(stream or ""),
                        str(event_type or ""),
                        data,
                        source_id=source_id,
                        source_kind=source_kind,
                        correlation_id=raw_event.get("correlation_id"),
                        idempotency_key=raw_event.get("idempotency_key"),
                    )
                except (HTTPException, ValueError) as exc:
                    detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
                    await _tracked_ws_send_json(
                        websocket, connection_id, {"type": "error", "detail": detail}
                    )
                    continue
                await _tracked_ws_send_json(
                    websocket,
                    connection_id,
                    {
                        "type": "published",
                        "id": event["id"],
                        "seq": event["seq"],
                        "cursor": event["id"],
                        "duplicate": duplicate,
                    }
                )
            elif frame_type == "ping":
                await _tracked_ws_send_json(
                    websocket,
                    connection_id,
                    {"type": "pong", "server_time": _now_iso()},
                )
            else:
                await _tracked_ws_send_json(
                    websocket,
                    connection_id,
                    {"type": "error", "detail": f"unsupported frame type '{frame_type}'"},
                )
                continue
            await _mailbox_ws_flush(websocket, connection_id, subscriptions)
    except WebSocketDisconnect:
        pass
    except RuntimeError as exc:
        if "disconnect message" not in str(exc):
            raise
    finally:
        _remove_active_websocket(connection_id)


@router.websocket("/emullm/websock_to_llm_user/ws")
@router.websocket("/websock_to_llm_user/ws")
async def websock_to_llm_user_socket(websocket: WebSocket) -> None:
    """Stream the aggregate LLM_USER interaction mailbox with query filters."""
    await websocket.accept()
    try:
        worker_id_value = websocket.query_params.get("worker_id")
        worker_id = _mailbox_id(worker_id_value) if worker_id_value else None
        event_types = _parse_event_types(websocket.query_params.getlist("type") + websocket.query_params.getlist("types"))
        model = websocket.query_params.get("model")
        modelmask = websocket.query_params.get("modelmask")
        after = websocket.query_params.get("after")
        _ensure_llm_user_mailbox()
    except ValueError as exc:
        await websocket.send_json({"type": "error", "detail": str(exc)})
        await websocket.close(code=1008)
        return

    filters = {
        "worker_id": worker_id,
        "model": model,
        "modelmask": modelmask,
        "types": sorted(event_types) if event_types else None,
    }
    connection_id = _register_active_websocket(
        websocket, "interaction-log", filters=filters
    )
    await _tracked_ws_send_json(
        websocket,
        connection_id,
        {"type": "subscribed", "stream": _LLM_USER_MAILBOX, "filters": filters, "cursor": after},
    )
    try:
        while True:
            events, next_cursor, has_more = _llm_user_event_page(
                after=after,
                limit=1000,
                worker_id=worker_id,
                model=model,
                modelmask=modelmask,
                event_types=event_types,
            )
            for event in events:
                await _tracked_ws_send_json(
                    websocket, connection_id, {"type": "event", "event": event}
                )
            after = next_cursor
            if has_more:
                await asyncio.sleep(0)
                continue
            try:
                frame = await asyncio.wait_for(
                    _tracked_ws_receive_json(websocket, connection_id), timeout=0.25
                )
            except asyncio.TimeoutError:
                continue
            if isinstance(frame, dict) and frame.get("type") == "ping":
                await _tracked_ws_send_json(
                    websocket,
                    connection_id,
                    {"type": "pong", "server_time": _now_iso()},
                )
            else:
                await _tracked_ws_send_json(
                    websocket,
                    connection_id,
                    {"type": "error", "detail": "only ping is supported on this read-only stream"},
                )
    except WebSocketDisconnect:
        pass
    except RuntimeError as exc:
        if "disconnect message" not in str(exc):
            raise
    finally:
        _remove_active_websocket(connection_id)


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
    normalized = (mime or "").split(";", 1)[0].strip().lower()
    return _MIME_EXT.get(normalized) or mimetypes.guess_extension(normalized) or ".bin"


def _anonymous_attachment_name(index: int) -> str:
    return f"attachment-{index}"


def _store_cloud_bytes(
    data: bytes,
    filename: str,
    purpose: str = "output",
    *,
    mime_type: str | None = None,
) -> dict[str, Any]:
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
        "mime_type": mime_type,
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
    media_type = (
        str(record.get("mime_type") or "").strip()
        or mimetypes.guess_type(str(record.get("filename", "")))[0]
        or "application/octet-stream"
    )
    return FileResponse(
        path=content_path,
        media_type=media_type,
        filename=str(record.get("filename") or file_id),
    )


@router.get("/emullm/cloud/files/{file_id}")
def cloud_file(file_id: str) -> FileResponse:
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
    worker_service_stats, team_service_stats = _service_stats_snapshot()
    for backend in _all_backends():
        backend_id = f"backend-{backend.get('name') or 'proxy'}"
        worker_service_stats.setdefault(
            backend_id,
            {
                "kind": "backend",
                "active": _worker_inflight.get(backend_id, 0),
                "reserved": _worker_reservations.get(backend_id, 0),
                "services": {},
            },
        )
    waiting = _waiting_for_worker_snapshot()
    now_monotonic = time.monotonic()
    with _worker_load_lock:
        active_service_requests = [
            {
                "request_id": request_id,
                "worker_id": str(metadata.get("worker_id") or ""),
                "model": str(metadata.get("model") or ""),
                "service_kind": str(metadata.get("service_kind") or ""),
                "started_at": metadata.get("started_at"),
                "age_seconds": round(
                    max(
                        0.0,
                        now_monotonic
                        - float(metadata.get("started_monotonic") or now_monotonic),
                    ),
                    1,
                ),
            }
            for request_id, metadata in _active_service_requests.items()
        ]
    stuck_workers = [
        request
        for request in active_service_requests
        if request["age_seconds"] >= _STUCK_WORKER_SECONDS
    ]
    headless_copilots = _copilot_api.manager_status()
    connection_errors = []
    for instance in headless_copilots:
        runtime = instance.get("runtime")
        runtime = runtime if isinstance(runtime, dict) else {}
        error_count = max(0, int(runtime.get("connection_errors") or 0))
        last_error = str(runtime.get("last_connection_error") or "").strip()
        if not error_count and not last_error:
            continue
        connection_errors.append(
            {
                "worker_id": str(instance.get("worker_id") or ""),
                "connected": bool(instance.get("connected")),
                "running": bool(instance.get("running")),
                "connection_errors": error_count,
                "last_connection_error": last_error or None,
                "last_disconnected_at": runtime.get("last_disconnected_at"),
            }
        )
    client_worker_limit, client_worker_reserve = _client_worker_capacity()
    client_capacity_waiting = sum(_client_capacity_waiters.values())
    connected_workers = sorted(_connected_workers.keys())
    busy_workers = sum(
        1 for worker_id in connected_workers if _worker_load(worker_id) > 0
    )
    idle_workers = sum(
        1 for worker_id in connected_workers if _worker_is_idle(worker_id)
    )
    return {
        "mode": ",".join(_current_modes()),
        "modes": _current_modes(),
        "started_at": _SERVER_STARTED_AT,
        "uptime_seconds": round(time.time() - _SERVER_STARTED_AT, 1),
        "runtime_dir": str(_RUNTIME_DIR),
        "connected_worker_ids": connected_workers,
        "worker_models": {worker_id: sorted(models.keys()) for worker_id, models in _worker_models.items()},
        "worker_kinds": dict(_worker_kinds),
        "worker_runtime_models": dict(_worker_runtime_models),
        "worker_descriptions": dict(_worker_descriptions),
        "worker_model_switches": dict(_worker_model_switch_stats),
        "team_model_switches": sum(
            int(stats.get("count", 0))
            for stats in _worker_model_switch_stats.values()
        ),
        "worker_capabilities": dict(_worker_capabilities),
        "worker_model_masks": {
            worker_id: list(_worker_model_masks[worker_id]) if worker_id in _worker_model_masks else None
            for worker_id in sorted(_connected_workers)
        },
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
        "worker_service_stats": worker_service_stats,
        "team_service_stats": team_service_stats,
        "concurrency": {
            "max_calls": _max_concurrent_calls,
            "active_calls": sum(_worker_inflight.values()),
            "reserved_calls": sum(_worker_reservations.values()),
            "waiting_for_worker": len(waiting),
            "connected_workers": len(connected_workers),
            "busy_workers": busy_workers,
            "idle_workers": idle_workers,
            "recently_busy_workers": (
                len(connected_workers) - busy_workers - idle_workers
            ),
            "idle_worker_target": _idle_worker_target,
            "idle_grace_seconds": _idle_grace_seconds,
            "idle_maintenance_paused": _idle_maintenance_paused,
            "backend_fallback_delay_seconds": _backend_fallback_delay_seconds,
            "client_worker_limit": client_worker_limit,
            "client_worker_reserve": client_worker_reserve,
            "client_capacity_waiting": client_capacity_waiting,
            "stuck_worker_seconds": _STUCK_WORKER_SECONDS,
            "stuck_workers": len(stuck_workers),
        },
        "waiting_for_worker": waiting,
        "active_service_requests": active_service_requests,
        "stuck_workers": stuck_workers,
        "connection_errors": connection_errors,
        "worker_not_ready": {
            worker_id: {"retry_after": round(delay, 3)}
            for worker_id in list(_worker_not_ready_until)
            if (delay := _worker_retry_delay(worker_id)) > 0
        },
        "pending_request_ids": sorted(_pending.keys()),
        "active_test_request_ids": sorted(_admin_test_tasks.keys()),
        "process": {
            "pid": os.getpid(),
            "host": os.environ.get("EMULLM_HOST", "127.0.0.1"),
            "port": int(os.environ.get("EMULLM_HTTP_PORT", "8801")),
            "shutdown_available": _process_control.shutdown_available(),
        },
        "record_counts": {kind: store.count() for kind, store in _KIND_STORES.items()},
        "mailboxes": {
            "config_path": str(_mailbox_config_path()),
            "events_dir": str(_mailbox_events_dir()),
            "count": _mailbox_count(),
            "event_count": _mailbox_event_count(),
        },
        "managed_workers": _sup.get_supervisor().status() if _sup.get_supervisor() else [],
        "headless_copilots": headless_copilots,
        "active_websockets": _active_websocket_rows(),
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
<meta name="description" content="EMULLM live status dashboard for model workers, routing, usage, and persistent stores.">
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
  .poll-controls { display: flex; flex-wrap: wrap; gap: 0.5rem 0.8rem; align-items: center; }
  .poll-controls label { display: inline-flex; gap: 0.35rem; align-items: center; }
  .poll-controls select { width: auto; }
  code { background: #8881; padding: 0.05rem 0.3rem; border-radius: 4px; }
  footer { margin-top: 1.5rem; color: #888; font-size: 0.8rem; }
</style>
</head>
<body>
<h1>emullm status <span id="view-tag" class="muted" style="font-size:1rem"></span></h1>
<p class="sub">mode <span id="mode" class="mode">-</span> &middot;
  <span id="poll-status">polling</span> &middot; <span id="updated" class="muted">-</span> &middot;
  <a id="view-toggle" href="/emullm/status/detail">detailed view</a></p>
<div class="poll-controls">
  <label>Polling
    <select id="poll-window">
      <option value="0">continuous</option>
      <option value="60000">stop after 1 minute</option>
      <option value="120000" selected>stop after 2 minutes</option>
      <option value="300000">stop after 5 minutes</option>
    </select>
  </label>
  <label><input id="poll-hidden" type="checkbox"> poll hidden page every 2 minutes</label>
  <button id="poll-wake" type="button">Wake / refresh now</button>
</div>

<div class="cards">
  <div class="card"><div class="k">Workers</div><div class="v" id="worker-count">-</div></div>
  <div class="card"><div class="k">Pending requests</div><div class="v" id="pending-count">-</div></div>
  <div class="card"><div class="k">Uptime</div><div class="v" id="uptime">-</div></div>
</div>

<h2>Connected workers</h2>
<table>
  <thead><tr><th>Worker</th><th>Role</th><th>Models</th><th>Model masks</th><th>Capabilities</th><th>Usage (window / total)</th></tr></thead>
  <tbody id="workers"><tr><td colspan="6" class="muted">loading...</td></tr></tbody>
</table>

<div id="detail-sections"></div>

<h2>Runtime</h2>
<p class="muted">runtime dir: <code id="runtime-dir">-</code></p>
<p class="muted">records: <span id="records">-</span></p>

<footer>Raw JSON: <a href="/emullm/admin/state">/emullm/admin/state</a></footer>

<script>
const DETAIL = __DETAIL__;
const POLL_VISIBLE_MS = 3000;
const POLL_HIDDEN_MS = 120000;
const POLL_WINDOW_KEY = 'emullm.poll.window';
const POLL_HIDDEN_KEY = 'emullm.poll.hidden';
let pollTimer = null;
let pollWindowStartedAt = Date.now();
let pollInFlight = false;
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
function esc(x) { return String(x).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

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
      tbody.innerHTML = '<tr><td colspan="6" class="muted">no workers connected</td></tr>';
    } else {
      tbody.innerHTML = workers.map(id => {
        const models = (s.worker_models || {})[id] || [];
        const masks = (s.worker_model_masks || {})[id];
        const caps = (s.worker_capabilities || {})[id] || {};
        const u = (s.worker_usage || {})[id] || {};
        const usage = (u.requests_in_window ?? 0) + ' / ' + (u.max_per_window ?? '-') +
          ' &middot; ' + (u.total_requests ?? 0) + ' total';
        return '<tr><td><span class="dot on"></span><b>' + esc(id) + '</b></td>' +
          '<td>' + roleBadge(roles[id]) + '</td>' +
          '<td>' + pills(models) + '</td>' +
          '<td>' + (masks === null ? '<span class="muted">all models</span>' : pills(masks || [])) + '</td>' +
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
          const masks = (s.worker_model_masks || {})[id];
          return '<div class="card" style="display:block;margin-bottom:0.5rem">' +
            '<b>' + esc(id) + '</b> &middot; ' + roleLine + '<br>' +
            '<span class="muted">models:</span> ' + pills(models) + '<br>' +
            '<span class="muted">model masks:</span> ' +
            (masks === null ? 'all models' : pills(masks || [])) + '<br>' +
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
function storedPollValue(key, fallback) {
  try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; }
}
function savePollValue(key, value) {
  try { localStorage.setItem(key, value); } catch {}
}
function schedulePoll() {
  clearTimeout(pollTimer);
  const hiddenPolling = document.getElementById('poll-hidden').checked;
  if (document.hidden && !hiddenPolling) {
    document.getElementById('poll-status').textContent = 'paused while hidden';
    return;
  }
  const windowMs = Number(document.getElementById('poll-window').value);
  if (!document.hidden && windowMs > 0 && Date.now() - pollWindowStartedAt >= windowMs) {
    document.getElementById('poll-status').textContent = 'paused · wake to resume';
    return;
  }
  const delay = document.hidden ? POLL_HIDDEN_MS : POLL_VISIBLE_MS;
  document.getElementById('poll-status').textContent = document.hidden
    ? 'hidden · every 2 minutes'
    : 'every 3 seconds';
  pollTimer = setTimeout(runPoll, delay);
}
async function runPoll(wake = false) {
  if (wake) pollWindowStartedAt = Date.now();
  if (pollInFlight) return;
  pollInFlight = true;
  try { await refresh(); } finally {
    pollInFlight = false;
    schedulePoll();
  }
}
const pollWindow = document.getElementById('poll-window');
const savedPollWindow = storedPollValue(POLL_WINDOW_KEY, '120000');
pollWindow.value = Array.from(pollWindow.options).some(option => option.value === savedPollWindow)
  ? savedPollWindow : '120000';
const pollHidden = document.getElementById('poll-hidden');
pollHidden.checked = storedPollValue(POLL_HIDDEN_KEY, 'false') === 'true';
pollWindow.addEventListener('change', () => {
  savePollValue(POLL_WINDOW_KEY, pollWindow.value);
  runPoll(true);
});
pollHidden.addEventListener('change', () => {
  savePollValue(POLL_HIDDEN_KEY, String(pollHidden.checked));
  runPoll(true);
});
document.getElementById('poll-wake').addEventListener('click', () => runPoll(true));
document.addEventListener('visibilitychange', () => {
  if (document.hidden) schedulePoll();
  else runPoll(true);
});
runPoll(true);
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
    modelmasks: str | list[str] | None = None


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
    modelmasks: str | list[str] | None = None


class BackendConfig(BaseModel):
    """A real OpenAI-compatible upstream for the proxy modes."""

    model_config = ConfigDict(extra="allow")
    name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    model: str | None = None
    default: bool | None = None


class AdminBackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
    )
    base_url: str = Field(min_length=1, max_length=2_000)
    description: str | None = Field(default=None, max_length=2_000)
    api_key: str | None = Field(default=None, max_length=10_000)
    api_key_env: str | None = Field(default=None, max_length=200)
    clear_api_key: bool = False
    model: str | None = Field(default=None, max_length=300)
    default: bool = False
    validation_interval: str | int | float | None = None
    expected_name: str | None = Field(default=None, max_length=100)
    expected_revision: str | None = Field(default=None, max_length=64)

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return value.rstrip("/")


class CodexSupplierConfig(BaseModel):
    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    id: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
    )
    name: str = Field(min_length=1, max_length=200)
    kind: Literal["copilot", "codex-cli", "openai-compatible", "custom"] = "custom"
    enabled: bool = True
    priority: int = Field(default=0, ge=-10_000, le=10_000)
    description: str | None = Field(default=None, max_length=2_000)
    worker_pattern: str | None = Field(default=None, max_length=300)
    model_prefix: str | None = Field(default=None, max_length=300)
    model_patterns: list[str] = Field(default_factory=list, max_length=100)
    command: str | None = Field(default=None, max_length=2_000)
    base_url: str | None = Field(default=None, max_length=2_000)
    api_key_env: str | None = Field(default=None, max_length=200)

    @field_validator("model_patterns")
    @classmethod
    def _clean_model_patterns(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))


_DEFAULT_CODEX_SUPPLIER = CodexSupplierConfig(
    id="copilot",
    name="GitHub Copilot",
    kind="copilot",
    enabled=True,
    priority=0,
    description=(
        "Current Codex supplier: resident Copilot SDK workers serving Codex-family models."
    ),
    worker_pattern="worker-copilot-*",
    model_prefix="copilot/",
    model_patterns=["*codex*"],
    command="copilot",
).model_dump(mode="json")


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
    enabled: bool = True
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
    modelmasks: str | list[str] | None = None
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


class ModelCatalogOverrideConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hidden: bool = False
    patch: dict[str, Any] = Field(default_factory=dict)


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
    max_concurrent_calls: int | None = Field(default=None, ge=4, le=50)
    idle_worker_target: int | None = Field(default=None, ge=0, le=50)
    idle_grace_seconds: float | None = Field(default=None, ge=0, le=3600)
    backend_fallback_delay_seconds: float | None = Field(
        default=None,
        ge=0,
        le=300,
    )
    model_routes: dict[str, str | list[str]] | None = None  # model id -> worker or ordered targets
    model_catalog_overrides: dict[str, ModelCatalogOverrideConfig] | None = None
    capability_fallback: Literal["stub", "wait", "error"] | None = None
    validation_interval_default: str | int | float | None = None  # inherited default cadence
    validation_interval_override: str | int | float | None = None  # forces cadence on ALL agents
    validation_interval: str | int | float | None = None  # alias for validation_interval_default
    services: ServicesConfig | None = None  # server-level catalog + per-service fallback
    agents: list[AgentConfig] | None = None
    # legacy flat forms (still honored by the runtime today):
    workers: list[WorkerConfig] | None = None
    headless_copilots: list[_copilot_api.HeadlessCopilotConfig] | None = None
    anti_idle: _copilot_api.AntiIdleConfig | None = None
    mock_workers: list[MockWorkerConfig] | None = None
    backends: list[BackendConfig] | None = None
    codex_suppliers: list[CodexSupplierConfig] | None = None
    mock: MockConfig | None = None


class SaveConfigRequest(BaseModel):
    config: dict[str, Any]
    expected_revision: str


class ConfigSectionRequest(BaseModel):
    value: Any = None
    delete: bool = False
    expected_revision: str


class AdminServerSettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str | None = Field(default=None, max_length=2_000)
    mode: str | list[str] | None = None
    capability_fallback: Literal["stub", "wait", "error"] = "stub"
    subagent_model: str | None = Field(default=None, max_length=300)
    max_concurrent_calls: int = Field(default=50, ge=4, le=50)
    idle_worker_target: int = Field(default=5, ge=0, le=50)
    idle_grace_seconds: float = Field(default=30, ge=0, le=3_600)
    backend_fallback_delay_seconds: float = Field(default=5, ge=0, le=300)
    validation_interval_default: str | int | float | None = None
    validation_interval_override: str | int | float | None = None
    expected_revision: str


class AdminAntiIdleConfigRequest(BaseModel):
    config: _copilot_api.AntiIdleConfig
    expected_revision: str


class AdminAntiIdleEnabledRequest(BaseModel):
    enabled: bool
    expected_revision: str


class AdminModelConfigRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=300)
    hidden: bool = False
    patch: dict[str, Any] = Field(default_factory=dict)
    set_route: bool = False
    route: str | list[str] | None = None
    reset: bool = False
    expected_revision: str


class AgentEnabledRequest(BaseModel):
    enabled: bool


class AdminTestChatRequest(BaseModel):
    request_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    model: str = Field(min_length=1, max_length=300)
    prompt: str = Field(min_length=1, max_length=100_000)
    required_capabilities: list[str] = Field(default_factory=list, max_length=32)
    attachments: list["AdminTestAttachment"] = Field(default_factory=list, max_length=12)


class AdminTestAttachment(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(default="application/octet-stream", max_length=255)
    data_b64: str = Field(min_length=1, max_length=48_000_000)


_MAX_ADMIN_TEST_ATTACHMENT_BYTES = max(
    1,
    int(
        os.environ.get(
            "EMULLM_ADMIN_TEST_ATTACHMENT_BYTES",
            str(25 * 1024 * 1024),
        )
    ),
)
_MAX_ADMIN_TEST_ATTACHMENTS_TOTAL_BYTES = max(
    _MAX_ADMIN_TEST_ATTACHMENT_BYTES,
    int(
        os.environ.get(
            "EMULLM_ADMIN_TEST_ATTACHMENTS_TOTAL_BYTES",
            str(50 * 1024 * 1024),
        )
    ),
)
AdminTestChatRequest.model_rebuild()


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
    try:
        config = _copilot_api.read_config_document(_CONFIG_PATH)
    except _copilot_api.CopilotInstanceError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {
        "path": str(_CONFIG_PATH),
        "config": config,
        "revision": _copilot_api.config_document_revision(config),
    }


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
    def replace(config: dict[str, Any]) -> dict[str, Any]:
        config.clear()
        config.update(body.config)
        return {
            "path": str(_CONFIG_PATH),
            "config": config,
            "saved": True,
        }

    return _atomic_config_update(
        replace,
        expected_revision=body.expected_revision,
    )


_EDITABLE_CONFIG_SECTIONS = {
    "services",
    "agents",
    "model_routes",
    "model_catalog_overrides",
    "workers",
    "mock_workers",
    "backends",
    "codex_suppliers",
    "anti_idle",
    "mock",
}


@router.put("/admin/emullm/config/section/{section}")
@router.put("/emullm/admin/config/section/{section}")
def admin_put_config_section(section: str, body: ConfigSectionRequest) -> dict[str, Any]:
    """Validate and atomically replace or delete one configuration section."""
    if section not in _EDITABLE_CONFIG_SECTIONS:
        raise HTTPException(
            status_code=404,
            detail=f"unknown editable config section '{section}'",
        )
    def mutate(config: dict[str, Any]) -> dict[str, Any]:
        if body.delete:
            config.pop(section, None)
        else:
            config[section] = body.value
        return {
            "path": str(_CONFIG_PATH),
            "section": section,
            "deleted": body.delete,
            "value": config.get(section),
            "config": config,
            "saved": True,
            "restart_required": True,
        }

    return _atomic_config_update(
        mutate,
        expected_revision=body.expected_revision,
    )


def _atomic_config_update(
    mutator: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    def validated(config: dict[str, Any]) -> dict[str, Any]:
        if (
            expected_revision is not None
            and _copilot_api.config_document_revision(config)
            != expected_revision
        ):
            raise HTTPException(
                status_code=409,
                detail="configuration changed; reload before saving",
            )
        result = mutator(config)
        try:
            EmullmConfig.model_validate(config)
        except ValidationError as error:
            raise HTTPException(status_code=422, detail=error.errors()) from error
        result["revision"] = _copilot_api.config_document_revision(config)
        return result

    try:
        return _copilot_api.update_config_document(_CONFIG_PATH, validated)
    except _copilot_api.CopilotInstanceError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.put("/admin/emullm/config/server-settings")
@router.put("/emullm/admin/config/server-settings")
def admin_put_server_settings(
    body: AdminServerSettingsRequest,
) -> dict[str, Any]:
    def mutate(config: dict[str, Any]) -> dict[str, Any]:
        for key in (
            "description",
            "mode",
            "subagent_model",
            "validation_interval_default",
            "validation_interval_override",
        ):
            value = getattr(body, key)
            if value in (None, ""):
                config.pop(key, None)
            else:
                config[key] = value
        config.update(
            capability_fallback=body.capability_fallback,
            max_concurrent_calls=body.max_concurrent_calls,
            idle_worker_target=body.idle_worker_target,
            idle_grace_seconds=body.idle_grace_seconds,
            backend_fallback_delay_seconds=body.backend_fallback_delay_seconds,
        )
        return {
            "path": str(_CONFIG_PATH),
            "config": config,
            "saved": True,
            "restart_required": True,
        }

    return _atomic_config_update(
        mutate,
        expected_revision=body.expected_revision,
    )


def _anti_idle_config(
    config: dict[str, Any] | None = None,
) -> _copilot_api.AntiIdleConfig:
    document = config if config is not None else _read_config()
    return _copilot_api.AntiIdleConfig.model_validate(
        document.get("anti_idle") or {}
    )


def _anti_idle_prompt_rows(
    anti_idle: _copilot_api.AntiIdleConfig,
) -> list[dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {
        prompt.id: {
            "attempts": 0,
            "completed": 0,
            "slow": 0,
            "timeouts": 0,
            "total_duration_ms": 0,
            "min_duration_ms": None,
            "max_duration_ms": 0,
            "shortest_worker_id": None,
            "longest_worker_id": None,
            "retired_workers": 0,
        }
        for prompt in anti_idle.prompts
    }
    manager = _copilot_api.get_manager()
    instances = _copilot_api.manager_status()
    for instance in instances:
        worker_id = str(instance.get("worker_id") or "")
        runtime = instance.get("runtime")
        task_stats = (
            runtime.get("keepalive_task_stats")
            if isinstance(runtime, dict)
            else None
        )
        retired = set(
            runtime.get("retired_keepalive_tasks") or []
            if isinstance(runtime, dict)
            else []
        )
        if not isinstance(task_stats, dict):
            continue
        for prompt_id, totals in aggregated.items():
            stats = task_stats.get(prompt_id)
            if not isinstance(stats, dict):
                continue
            for key in (
                "attempts",
                "completed",
                "slow",
                "timeouts",
                "total_duration_ms",
            ):
                totals[key] += int(stats.get(key) or 0)
            minimum = stats.get("min_duration_ms")
            if minimum is not None and (
                totals["min_duration_ms"] is None
                or int(minimum) < int(totals["min_duration_ms"])
            ):
                totals["min_duration_ms"] = int(minimum)
                totals["shortest_worker_id"] = worker_id
            maximum = int(stats.get("max_duration_ms") or 0)
            if maximum > int(totals["max_duration_ms"]):
                totals["max_duration_ms"] = maximum
                totals["longest_worker_id"] = worker_id
            if prompt_id in retired:
                totals["retired_workers"] += 1
    rows = []
    for number, prompt in enumerate(anti_idle.prompts, start=1):
        totals = aggregated[prompt.id]
        attempts = int(totals["attempts"])
        rows.append(
            {
                "number": number,
                **prompt.model_dump(mode="json"),
                **totals,
                "average_duration_ms": round(
                    int(totals["total_duration_ms"]) / attempts,
                    1,
                )
                if attempts
                else 0.0,
            }
        )
    return rows


@router.get("/admin/emullm/anti-idle")
@router.get("/emullm/admin/anti-idle")
def admin_get_anti_idle() -> dict[str, Any]:
    try:
        document = _copilot_api.read_config_document(_CONFIG_PATH)
    except _copilot_api.CopilotInstanceError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    anti_idle = _anti_idle_config(document)
    return {
        "config": anti_idle.model_dump(mode="json"),
        "prompts": _anti_idle_prompt_rows(anti_idle),
        "revision": _copilot_api.config_document_revision(document),
        "worker_count": (
            len(_copilot_api.manager_status())
            if _copilot_api.get_manager() is not None
            else 0
        ),
    }


@router.put("/admin/emullm/anti-idle")
@router.put("/emullm/admin/anti-idle")
def admin_put_anti_idle(
    body: AdminAntiIdleConfigRequest,
) -> dict[str, Any]:
    def mutate(config: dict[str, Any]) -> dict[str, Any]:
        config["anti_idle"] = body.config.model_dump(mode="json")
        return {
            "saved": True,
            "restart_required": True,
            "config": config["anti_idle"],
            "prompts": _anti_idle_prompt_rows(body.config),
        }

    return _atomic_config_update(
        mutate,
        expected_revision=body.expected_revision,
    )


async def _set_connected_anti_idle(
    worker_id: str,
    enabled: bool,
) -> dict[str, Any]:
    peer = _connected_workers.get(worker_id)
    if peer is None:
        return {"worker_id": worker_id, "connected": False, "updated": False}
    control_id = f"anti-idle-{uuid.uuid4().hex}"
    future: asyncio.Future[dict[str, Any]] = (
        asyncio.get_running_loop().create_future()
    )
    _pending_worker_controls[control_id] = future
    try:
        await _send_worker_json(
            worker_id,
            peer,
            {
                "type": "set_anti_idle",
                "id": control_id,
                "enabled": enabled,
            },
        )
        result = await asyncio.wait_for(future, timeout=5)
        return {
            "worker_id": worker_id,
            "connected": True,
            "updated": True,
            "enabled": result.get("enabled") is True,
        }
    except Exception as error:  # noqa: BLE001 - isolate each worker update
        return {
            "worker_id": worker_id,
            "connected": True,
            "updated": False,
            "error": str(error),
        }
    finally:
        _pending_worker_controls.pop(control_id, None)


@router.put("/admin/emullm/anti-idle/enabled")
@router.put("/emullm/admin/anti-idle/enabled")
async def admin_put_anti_idle_enabled(
    body: AdminAntiIdleEnabledRequest,
) -> dict[str, Any]:
    def mutate(config: dict[str, Any]) -> dict[str, Any]:
        current = _anti_idle_config(config)
        config["anti_idle"] = current.model_copy(
            update={"enabled": body.enabled}
        ).model_dump(mode="json")
        return {
            "saved": True,
            "enabled": body.enabled,
            "config": config["anti_idle"],
        }

    saved = _atomic_config_update(
        mutate,
        expected_revision=body.expected_revision,
    )
    results = await asyncio.gather(
        *(
            _set_connected_anti_idle(worker_id, body.enabled)
            for worker_id in sorted(_connected_workers)
        )
    )
    saved["results"] = results
    saved["updated_workers"] = sum(
        1 for result in results if result.get("updated")
    )
    saved["failed_workers"] = sum(
        1 for result in results if not result.get("updated")
    )
    saved["restart_required"] = False
    return saved


async def _reset_connected_keepalive_stats(worker_id: str) -> dict[str, Any]:
    peer = _connected_workers.get(worker_id)
    if peer is None:
        return {"worker_id": worker_id, "connected": False, "reset": False}
    control_id = f"keepalive-reset-{uuid.uuid4().hex}"
    future: asyncio.Future[dict[str, Any]] = (
        asyncio.get_running_loop().create_future()
    )
    _pending_worker_controls[control_id] = future
    try:
        await _send_worker_json(
            worker_id,
            peer,
            {"type": "reset_keepalive_stats", "id": control_id},
        )
        await asyncio.wait_for(future, timeout=5)
        return {"worker_id": worker_id, "connected": True, "reset": True}
    except Exception as error:  # noqa: BLE001 - isolate each worker reset
        return {
            "worker_id": worker_id,
            "connected": True,
            "reset": False,
            "error": (
                "worker did not acknowledge reset within 5 seconds"
                if isinstance(error, TimeoutError)
                else str(error)
            ),
        }
    finally:
        _pending_worker_controls.pop(control_id, None)


@router.post("/admin/emullm/anti-idle/reset-stats")
@router.post("/emullm/admin/anti-idle/reset-stats")
async def admin_reset_anti_idle_stats() -> dict[str, Any]:
    manager = _copilot_api.get_manager()
    if manager is None:
        return {"reset": 0, "results": []}
    instances = await asyncio.to_thread(manager.list)
    connected = [
        str(instance["worker_id"])
        for instance in instances
        if instance.get("connected")
    ]
    results = list(
        await asyncio.gather(
            *(
                _reset_connected_keepalive_stats(worker_id)
                for worker_id in connected
            )
        )
    )
    connected_ids = set(connected)
    for instance in instances:
        worker_id = str(instance["worker_id"])
        if worker_id in connected_ids:
            continue
        cleared = await asyncio.to_thread(
            manager.clear_keepalive_stats,
            worker_id,
        )
        results.append(
            {
                "worker_id": worker_id,
                "connected": False,
                "reset": cleared,
            }
        )
    return {
        "reset": sum(bool(result.get("reset")) for result in results),
        "results": results,
    }


def _configured_backend_records(
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    document = config if config is not None else _read_config()
    records: list[dict[str, Any]] = []
    for source, entries in (
        ("backends", document.get("backends")),
        ("agents", document.get("agents")),
    ):
        if not isinstance(entries, list):
            continue
        for index, raw_entry in enumerate(entries):
            if not isinstance(raw_entry, dict):
                continue
            if source == "agents" and raw_entry.get("launch") != "proxy":
                continue
            entry = dict(raw_entry)
            revision = hashlib.sha256(
                json.dumps(
                    raw_entry,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
            ).hexdigest()[:24]
            has_api_key = bool(entry.pop("api_key", None))
            records.append(
                {
                    **entry,
                    "source": source,
                    "index": index,
                    "record_id": f"{source}:{index}",
                    "revision": revision,
                    "has_api_key": has_api_key,
                }
            )
    return records


def _backend_entry(
    config: dict[str, Any],
    source: str,
    index: int,
    expected_name: str | None = None,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    if source not in {"backends", "agents"}:
        raise HTTPException(status_code=404, detail=f"unknown backend source '{source}'")
    entries = config.get(source)
    if not isinstance(entries, list) or not 0 <= index < len(entries):
        raise HTTPException(status_code=404, detail="backend record not found")
    entry = entries[index]
    if not isinstance(entry, dict) or (
        source == "agents" and entry.get("launch") != "proxy"
    ):
        raise HTTPException(status_code=404, detail="backend record not found")
    if expected_name is not None and entry.get("name") != expected_name:
        raise HTTPException(
            status_code=409,
            detail="backend record changed; refresh before editing",
        )
    if expected_revision is not None:
        revision = hashlib.sha256(
            json.dumps(
                entry,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()[:24]
        if revision != expected_revision:
            raise HTTPException(
                status_code=409,
                detail="backend record changed; refresh before editing",
            )
    return entry


def _clear_backend_defaults(config: dict[str, Any]) -> None:
    for source in ("backends", "agents"):
        entries = config.get(source)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict) and (
                source == "backends" or entry.get("launch") == "proxy"
            ):
                entry["default"] = False


def _apply_backend_config(
    entry: dict[str, Any],
    body: AdminBackendConfig,
) -> None:
    entry.update(
        name=body.name,
        base_url=body.base_url,
        default=body.default,
    )
    for key in ("description", "api_key_env", "model", "validation_interval"):
        value = getattr(body, key)
        if value in (None, ""):
            entry.pop(key, None)
        else:
            entry[key] = value
    if body.clear_api_key:
        entry.pop("api_key", None)
    elif body.api_key:
        entry["api_key"] = body.api_key


def _ensure_unique_backend_names(config: dict[str, Any]) -> None:
    names = [
        str(record.get("name") or "").lower()
        for record in _configured_backend_records(config)
    ]
    duplicates = sorted({name for name in names if name and names.count(name) > 1})
    if duplicates:
        raise HTTPException(
            status_code=409,
            detail=f"duplicate backend name(s): {', '.join(duplicates)}",
        )


@router.get("/admin/emullm/backends/configured")
@router.get("/emullm/admin/backends/configured")
def admin_configured_backends() -> dict[str, Any]:
    records = _configured_backend_records()
    return {"count": len(records), "backends": records}


@router.post("/admin/emullm/backends/configured", status_code=201)
@router.post("/emullm/admin/backends/configured", status_code=201)
def admin_create_backend(body: AdminBackendConfig) -> dict[str, Any]:
    def mutate(config: dict[str, Any]) -> dict[str, Any]:
        entries = config.setdefault("backends", [])
        if not isinstance(entries, list):
            raise HTTPException(status_code=409, detail="backends config is not a list")
        if body.default:
            _clear_backend_defaults(config)
        entry: dict[str, Any] = {}
        _apply_backend_config(entry, body)
        entries.append(entry)
        _ensure_unique_backend_names(config)
        records = _configured_backend_records(config)
        created = next(
            record
            for record in records
            if record["source"] == "backends"
            and record["index"] == len(entries) - 1
        )
        return {
            "saved": True,
            "restart_required": True,
            "backend": created,
            "backends": records,
        }

    return _atomic_config_update(mutate)


@router.put("/admin/emullm/backends/configured/{source}/{index}")
@router.put("/emullm/admin/backends/configured/{source}/{index}")
def admin_update_backend(
    source: str,
    index: int,
    body: AdminBackendConfig,
) -> dict[str, Any]:
    if body.expected_revision is None:
        raise HTTPException(
            status_code=428,
            detail="expected_revision is required; refresh before editing",
        )

    def mutate(config: dict[str, Any]) -> dict[str, Any]:
        entry = _backend_entry(
            config,
            source,
            index,
            body.expected_name,
            body.expected_revision,
        )
        if body.default:
            _clear_backend_defaults(config)
        _apply_backend_config(entry, body)
        _ensure_unique_backend_names(config)
        record = next(
            item
            for item in _configured_backend_records(config)
            if item["source"] == source and item["index"] == index
        )
        return {
            "saved": True,
            "restart_required": True,
            "backend": record,
            "backends": _configured_backend_records(config),
        }

    return _atomic_config_update(mutate)


@router.delete("/admin/emullm/backends/configured/{source}/{index}")
@router.delete("/emullm/admin/backends/configured/{source}/{index}")
def admin_delete_backend(
    source: str,
    index: int,
    expected_name: str | None = None,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    if expected_revision is None:
        raise HTTPException(
            status_code=428,
            detail="expected_revision is required; refresh before deleting",
        )

    def mutate(config: dict[str, Any]) -> dict[str, Any]:
        _backend_entry(
            config,
            source,
            index,
            expected_name,
            expected_revision,
        )
        entries = config[source]
        deleted = entries.pop(index)
        return {
            "saved": True,
            "restart_required": True,
            "deleted": str(deleted.get("name") or f"{source}:{index}"),
            "backends": _configured_backend_records(config),
        }

    return _atomic_config_update(mutate)


def _configured_codex_suppliers(
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    document = config if config is not None else _read_config()
    raw_suppliers = document.get("codex_suppliers")
    if raw_suppliers is None:
        raw_suppliers = [_DEFAULT_CODEX_SUPPLIER]
    suppliers = []
    for supplier in raw_suppliers:
        if not isinstance(supplier, dict):
            continue
        normalized = CodexSupplierConfig.model_validate(supplier).model_dump(
            mode="json"
        )
        normalized["revision"] = hashlib.sha256(
            json.dumps(
                supplier,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()[:24]
        suppliers.append(normalized)
    return suppliers


def _stored_codex_suppliers(
    suppliers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in supplier.items() if key != "revision"}
        for supplier in suppliers
    ]


def _codex_supplier_for_model(backing_model: str) -> dict[str, Any] | None:
    matches: list[tuple[int, int, dict[str, Any]]] = []
    for supplier in _configured_codex_suppliers():
        if not supplier.get("enabled"):
            continue
        patterns = supplier.get("model_patterns") or []
        matching_patterns = [
            str(pattern)
            for pattern in patterns
            if fnmatchcase(backing_model.lower(), str(pattern).lower())
        ]
        if matching_patterns:
            specificity = max(
                len(pattern.replace("*", "").replace("?", ""))
                for pattern in matching_patterns
            )
            matches.append(
                (int(supplier.get("priority") or 0), specificity, supplier)
            )
    return max(matches, key=lambda match: (match[0], match[1]))[2] if matches else None


@router.get("/admin/emullm/codex-suppliers")
@router.get("/emullm/admin/codex-suppliers")
def admin_list_codex_suppliers() -> dict[str, Any]:
    suppliers = _configured_codex_suppliers()
    return {"count": len(suppliers), "suppliers": suppliers}


@router.post("/admin/emullm/codex-suppliers", status_code=201)
@router.post("/emullm/admin/codex-suppliers", status_code=201)
def admin_create_codex_supplier(body: CodexSupplierConfig) -> dict[str, Any]:
    def mutate(config: dict[str, Any]) -> dict[str, Any]:
        suppliers = _configured_codex_suppliers(config)
        if any(supplier["id"] == body.id for supplier in suppliers):
            raise HTTPException(
                status_code=409,
                detail=f"Codex supplier '{body.id}' already exists",
            )
        suppliers.append(body.model_dump(mode="json"))
        config["codex_suppliers"] = _stored_codex_suppliers(suppliers)
        records = _configured_codex_suppliers(config)
        return {"saved": True, "supplier": records[-1], "suppliers": records}

    return _atomic_config_update(mutate)


@router.put("/admin/emullm/codex-suppliers/{supplier_id}")
@router.put("/emullm/admin/codex-suppliers/{supplier_id}")
def admin_update_codex_supplier(
    supplier_id: str,
    body: CodexSupplierConfig,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    if supplier_id != body.id:
        raise HTTPException(status_code=422, detail="path supplier ID must match body ID")
    if expected_revision is None:
        raise HTTPException(
            status_code=428,
            detail="expected_revision is required; refresh before editing",
        )
    def mutate(config: dict[str, Any]) -> dict[str, Any]:
        suppliers = _configured_codex_suppliers(config)
        for index, supplier in enumerate(suppliers):
            if supplier["id"] == supplier_id:
                if supplier.get("revision") != expected_revision:
                    raise HTTPException(
                        status_code=409,
                        detail="Codex supplier changed; refresh before editing",
                    )
                merged = {
                    key: value
                    for key, value in supplier.items()
                    if key != "revision"
                }
                merged.update(body.model_dump(mode="json"))
                suppliers[index] = merged
                break
        else:
            raise HTTPException(
                status_code=404,
                detail=f"no Codex supplier '{supplier_id}'",
            )
        config["codex_suppliers"] = _stored_codex_suppliers(suppliers)
        records = _configured_codex_suppliers(config)
        return {
            "saved": True,
            "supplier": records[index],
            "suppliers": records,
        }

    return _atomic_config_update(mutate)


@router.delete("/admin/emullm/codex-suppliers/{supplier_id}")
@router.delete("/emullm/admin/codex-suppliers/{supplier_id}")
def admin_delete_codex_supplier(
    supplier_id: str,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    if expected_revision is None:
        raise HTTPException(
            status_code=428,
            detail="expected_revision is required; refresh before deleting",
        )

    def mutate(config: dict[str, Any]) -> dict[str, Any]:
        suppliers = _configured_codex_suppliers(config)
        current = next(
            (
                supplier
                for supplier in suppliers
                if supplier["id"] == supplier_id
            ),
            None,
        )
        if current is None:
            raise HTTPException(
                status_code=404,
                detail=f"no Codex supplier '{supplier_id}'",
            )
        if current.get("revision") != expected_revision:
            raise HTTPException(
                status_code=409,
                detail="Codex supplier changed; refresh before deleting",
            )
        remaining = [
            supplier for supplier in suppliers if supplier["id"] != supplier_id
        ]
        config["codex_suppliers"] = _stored_codex_suppliers(remaining)
        records = _configured_codex_suppliers(config)
        return {"saved": True, "deleted": supplier_id, "suppliers": records}

    return _atomic_config_update(mutate)


@router.get("/admin/emullm/model-config")
@router.get("/emullm/admin/model-config")
def admin_model_config() -> dict[str, Any]:
    manager = _copilot_api.get_manager()
    slots = []
    if manager is not None:
        slots = [
            instance
            for instance in manager.list()
            if _is_on_demand_copilot(
                str(instance.get("worker_id") or "")
            )
        ]
    try:
        config = _copilot_api.read_config_document(_CONFIG_PATH)
    except _copilot_api.CopilotInstanceError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {
        "models": list_models(hidden=True)["data"],
        "overrides": dict(_model_catalog_overrides),
        "routes": dict(_model_routes),
        "backends": _all_backends(),
        "codex_suppliers": _configured_codex_suppliers(),
        "revision": _copilot_api.config_document_revision(config),
        "on_demand": {
            "worker_prefix": _ON_DEMAND_COPILOT_PREFIX,
            "worker_start": _ON_DEMAND_COPILOT_START,
            "limit": _on_demand_copilot_limit(),
            "max_concurrent_calls": _max_concurrent_calls,
            "slots": slots,
        },
    }


@router.put("/admin/emullm/model-config")
@router.put("/emullm/admin/model-config")
def admin_put_model_config(body: AdminModelConfigRequest) -> dict[str, Any]:
    def mutate(config: dict[str, Any]) -> dict[str, Any]:
        overrides = dict(config.get("model_catalog_overrides") or {})
        if body.reset:
            overrides.pop(body.model_id, None)
        else:
            patch = dict(body.patch)
            for derived in (
                "id",
                "connected",
                "active_workers",
                "backing_models",
                "route_targets",
                "routing_mode",
                "hidden",
                "exported",
            ):
                patch.pop(derived, None)
            overrides[body.model_id] = {
                "hidden": body.hidden,
                "patch": patch,
            }
        if overrides:
            config["model_catalog_overrides"] = overrides
        else:
            config.pop("model_catalog_overrides", None)

        routes = dict(config.get("model_routes") or {})
        if body.set_route:
            normalized = _normalise_model_route(body.route)
            if normalized:
                routes[body.model_id] = normalized
            else:
                routes.pop(body.model_id, None)
            if routes:
                config["model_routes"] = routes
            else:
                config.pop("model_routes", None)
        return {"saved": True}

    result = _atomic_config_update(
        mutate,
        expected_revision=body.expected_revision,
    )
    config = _copilot_api.read_config_document(_CONFIG_PATH)
    apply_agent_policies(config)
    entry = None
    try:
        entry = get_model(body.model_id)
    except HTTPException as error:
        if error.status_code != 404:
            raise
    return {
        **result,
        "model_id": body.model_id,
        "override": _model_catalog_overrides.get(body.model_id),
        "route": _model_routes.get(body.model_id),
        "model": entry,
    }


@router.post("/admin/emullm/model-config/load/{model_id:path}")
@router.post("/emullm/admin/model-config/load/{model_id:path}")
async def admin_load_copilot_model(model_id: str) -> dict[str, Any]:
    backing_model = _copilot_backing_model(model_id)
    if backing_model is None:
        raise HTTPException(
            status_code=422,
            detail="only exported copilot/<model-id> entries can be loaded on demand",
        )
    worker_id = await _ensure_on_demand_copilot(model_id, backing_model)
    _release_worker_reservation(worker_id)
    manager = _copilot_api.get_manager()
    return {
        "model_id": model_id,
        "backing_model": backing_model,
        "worker": manager.get(worker_id) if manager is not None else {"worker_id": worker_id},
        "loaded": True,
    }


@router.get("/admin/emullm/test-samples")
@router.get("/emullm/admin/test-samples")
def admin_test_samples() -> dict[str, Any]:
    return {
        "samples": [
            {
                key: value
                for key, value in sample.items()
                if key != "data"
            }
            | {
                "bytes": len(sample["data"]),
                "url": f"/emullm/admin/test-samples/{sample_id}",
            }
            for sample_id, sample in test_media_samples().items()
        ]
    }


@router.get("/admin/emullm/test-samples/{sample_id}")
@router.get("/emullm/admin/test-samples/{sample_id}")
def admin_test_sample(sample_id: str) -> Response:
    sample = test_media_samples().get(sample_id)
    if sample is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown test media sample '{sample_id}'",
        )
    return Response(
        content=sample["data"],
        media_type=sample["mime_type"],
        headers={
            "Content-Disposition": f'inline; filename="{sample["name"]}"',
            "Cache-Control": "public, max-age=3600",
        },
    )


@router.get("/admin/emullm/health")
@router.get("/emullm/admin/health")
async def admin_health() -> dict[str, Any]:
    """Return an in-memory readiness snapshot suitable for rollout polling."""
    connected_workers = len(_connected_workers)
    return {
        "status": "ready",
        "ready": connected_workers > 0,
        "pid": os.getpid(),
        "uptime_seconds": round(time.time() - _SERVER_STARTED_AT, 1),
        "connected_workers": connected_workers,
        "max_calls": _max_concurrent_calls,
        "active_calls": sum(_worker_inflight.values()),
        "waiting_for_worker": len(_waiting_for_worker_snapshot()),
        "stuck_workers": sum(
            1
            for metadata in _active_service_requests.values()
            if time.monotonic()
            - float(metadata.get("started_monotonic") or time.monotonic())
            >= _STUCK_WORKER_SECONDS
        ),
        "restart_in_progress": _process_control.restart_in_progress(),
    }


async def _shutdown_connected_worker(
    worker_id: str,
    reason: str,
) -> bool:
    peer = _connected_workers.get(worker_id)
    if peer is None:
        return False
    await _send_worker_json(
        worker_id,
        peer,
        {"type": "shutdown", "reason": reason},
    )
    deadline = time.monotonic() + 15
    while worker_id in _connected_workers and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    return worker_id not in _connected_workers


async def _wait_for_managed_worker_offline(
    manager: Any,
    worker_id: str,
) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        instance = await asyncio.to_thread(manager.get, worker_id)
        if not instance.get("running") and not instance.get("connected"):
            return
        await asyncio.sleep(0.1)


async def _wait_for_connected_worker(
    worker_id: str,
    timeout_seconds: float = _WORKER_RECONNECT_TIMEOUT_SECONDS,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if worker_id in _connected_workers:
            return True
        await asyncio.sleep(0.1)
    return worker_id in _connected_workers


@router.post("/admin/emullm/copilots/{worker_id}/online-action/{action}")
@router.post("/emullm/admin/copilots/{worker_id}/online-action/{action}")
async def admin_online_copilot_action(
    worker_id: str,
    action: str,
) -> dict[str, Any]:
    global _idle_maintenance_paused
    if action not in {"start", "stop", "restart", "reset-session"}:
        raise HTTPException(status_code=404, detail=f"unknown action '{action}'")
    manager = _copilot_api.get_manager()
    if manager is None:
        raise HTTPException(status_code=409, detail="headless Copilot manager unavailable")
    try:
        instance = manager.get(worker_id)
    except Exception as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    online = bool(instance.get("running") or instance.get("connected"))
    if action == "start":
        if online:
            raise HTTPException(status_code=409, detail=f"worker '{worker_id}' is online")
        _idle_maintenance_paused = False
        return await asyncio.to_thread(manager.start, worker_id)
    if not online:
        raise HTTPException(status_code=409, detail=f"worker '{worker_id}' is offline")
    if instance.get("connected"):
        await _shutdown_connected_worker(worker_id, f"operator {action}")
        await _wait_for_managed_worker_offline(manager, worker_id)
    else:
        await asyncio.to_thread(manager.stop, worker_id)
    if action == "stop":
        return {"worker_id": worker_id, "stopped": True}
    if action == "reset-session":
        result = await asyncio.to_thread(manager.reset_session, worker_id)
        await asyncio.to_thread(manager.start, worker_id)
        return {**result, "started": True}
    _idle_maintenance_paused = False
    return await asyncio.to_thread(manager.start, worker_id)


@router.post("/admin/emullm/copilots/bulk/{action}")
@router.post("/emullm/admin/copilots/bulk/{action}")
async def admin_bulk_copilot_action(
    action: str,
    batch_size: int = Query(
        _BULK_WORKER_ACTION_BATCH_SIZE,
        ge=1,
        le=50,
    ),
) -> dict[str, Any]:
    global _idle_maintenance_paused
    if action not in {
        "start",
        "stop",
        "stop-idle",
        "restart",
        "reset-session",
    }:
        raise HTTPException(
            status_code=404,
            detail=f"unknown bulk Copilot action '{action}'",
        )
    manager = _copilot_api.get_manager()
    if manager is None:
        raise HTTPException(
            status_code=409,
            detail="headless Copilot manager is unavailable",
        )
    instances = await asyncio.to_thread(manager.list)
    results = []
    if action in {"stop", "stop-idle"}:
        _idle_maintenance_paused = True
    elif action in {"start", "restart"}:
        _idle_maintenance_paused = False

    eligible = []
    for instance in instances:
        worker_id = str(instance["worker_id"])
        online = bool(instance.get("running") or instance.get("connected"))
        if action == "start":
            if online:
                continue
        elif action in {"stop", "stop-idle"}:
            if not online:
                continue
            if action == "stop-idle" and not _worker_is_idle(worker_id):
                continue
        elif not online:
            continue
        eligible.append(instance)

    async def apply(instance: dict[str, Any]) -> dict[str, Any]:
        worker_id = str(instance["worker_id"])
        if action == "start":
            return await asyncio.to_thread(manager.start, worker_id)
        if action in {"stop", "stop-idle"}:
            if instance.get("connected"):
                stopped = await _shutdown_connected_worker(
                    worker_id,
                    f"bulk {action}",
                )
                result = {"worker_id": worker_id, "stopped": stopped}
            else:
                result = await asyncio.to_thread(manager.stop, worker_id)
            return result
        if instance.get("connected"):
            await _shutdown_connected_worker(
                worker_id,
                f"bulk {action}",
            )
            await _wait_for_managed_worker_offline(manager, worker_id)
        else:
            await asyncio.to_thread(manager.stop, worker_id)
        if action == "reset-session":
            result = await asyncio.to_thread(
                manager.reset_session,
                worker_id,
            )
            await asyncio.to_thread(manager.start, worker_id)
        else:
            result = await asyncio.to_thread(manager.start, worker_id)
        connected = await _wait_for_connected_worker(worker_id)
        return {**result, "connected": connected}

    for offset in range(0, len(eligible), batch_size):
        batch = eligible[offset : offset + batch_size]
        results.extend(await asyncio.gather(*(apply(instance) for instance in batch)))
    return {
        "action": action,
        "affected": len(results),
        "batch_size": batch_size,
        "batches": (len(eligible) + batch_size - 1) // batch_size,
        "idle_maintenance_paused": _idle_maintenance_paused,
        "results": results,
    }


def _configured_agent_rows() -> list[dict[str, Any]]:
    config = _read_config()
    agents = config.get("agents") if isinstance(config.get("agents"), list) else []
    return [
        {
            "id": str(agent.get("id") or agent.get("worker_id") or ""),
            "launch": str(agent.get("launch") or ""),
            "description": str(agent.get("description") or ""),
            "enabled": agent.get("enabled") is not False,
            "connected": str(agent.get("id") or agent.get("worker_id") or "") in _connected_workers,
            "mock_registered": isinstance(
                _connected_workers.get(str(agent.get("id") or agent.get("worker_id") or "")),
                _MockWorker,
            ),
        }
        for agent in agents
        if isinstance(agent, dict) and (agent.get("id") or agent.get("worker_id"))
    ]


@router.get("/admin/emullm/agents")
@router.get("/emullm/admin/agents")
def admin_list_agents() -> dict[str, Any]:
    """List configured agents and whether each is enabled and connected."""
    return {"agents": _configured_agent_rows()}


@router.get("/admin/emullm/websockets")
@router.get("/emullm/admin/websockets")
def admin_list_websockets() -> dict[str, Any]:
    """List every active EMULLM WebSocket and its inbound/outbound frame counts."""
    connections = _active_websocket_rows()
    return {
        "count": len(connections),
        "connections": connections,
        "logs": {
            "directory": str(_socket_worker_log_dir),
            "segments_per_worker": 3,
            "segment_bytes": _socket_worker_log_segment_bytes,
            "max_bytes_per_worker": 3 * _socket_worker_log_segment_bytes,
        },
    }


@router.get("/admin/emullm/websockets/{worker_id}/log")
@router.get("/emullm/admin/websockets/{worker_id}/log")
def admin_worker_socket_log(worker_id: str) -> PlainTextResponse:
    try:
        normalized = _mailbox_id(worker_id)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    first, current, previous = _socket_log_paths(normalized)
    with _socket_worker_log_lock:
        segments = [
            (name, path)
            for name, path in (
                ("first", first),
                ("previous", previous),
                ("current", current),
            )
            if path.is_file()
        ]
        if not segments:
            raise HTTPException(
                status_code=404,
                detail=f"no socket log for worker '{normalized}'",
            )
        startup_prompt, prompt_source = _worker_start_prompt(normalized, first)
        prompt_header = {
            **_socket_log_clock_fields(),
            "record_type": "worker_start_prompt",
            "worker_id": normalized,
            "source": prompt_source,
            "from": "SYSTEM",
            "sender": "SYSTEM",
            "prompt": _socket_log_value(startup_prompt),
        }
        content_parts = [
            (
                json.dumps(
                    prompt_header,
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
        ]
        for segment, path in segments:
            boundary = {
                **_socket_log_clock_fields(),
                "record_type": "segment_boundary",
                "segment": segment,
                "file": path.name,
                "bytes": path.stat().st_size,
                "from": "SYSTEM",
                "sender": "SYSTEM",
            }
            content_parts.append(
                (
                    json.dumps(
                        boundary,
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
            )
            content_parts.append(path.read_bytes())
        content = b"".join(content_parts)
    return PlainTextResponse(
        content.decode("utf-8", errors="replace"),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/admin/emullm/websockets/{worker_id}/log/view", response_class=HTMLResponse)
@router.get("/emullm/admin/websockets/{worker_id}/log/view", response_class=HTMLResponse)
def admin_worker_socket_log_viewer(worker_id: str) -> HTMLResponse:
    try:
        normalized = _mailbox_id(worker_id)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    raw_url = f"/emullm/admin/websockets/{normalized}/log"
    worker_json = json.dumps(normalized)
    raw_url_json = json.dumps(raw_url)
    return HTMLResponse(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='12' fill='%23071720'/%3E%3Ctext x='32' y='43' text-anchor='middle' font-size='38' font-family='monospace' font-weight='700' fill='%2325d5c8'%3EE%3C/text%3E%3C/svg%3E">
<title>EMULLM socket log · {normalized}</title>
<style>
  :root {{ color-scheme: dark; --bg:#061017; --panel:#0a1b25; --line:#19414d;
    --text:#d9edf1; --muted:#7895a0; --in:#12313d; --out:#16483f; --accent:#25d5c8; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; height:100vh; display:flex; flex-direction:column;
    background:var(--bg); color:var(--text); font-family:Inter,"Segoe UI",sans-serif; }}
  header {{ display:flex; flex-wrap:wrap; align-items:center; gap:10px; padding:12px 16px;
    border-bottom:1px solid var(--line); background:#071720; }}
  h1 {{ margin:0; font-size:.95rem; color:var(--accent); }}
  button,a {{ padding:6px 10px; border:1px solid var(--line); border-radius:6px;
    background:#0e2b36; color:var(--text); text-decoration:none; cursor:pointer; }}
  #status {{ color:var(--muted); font-size:.75rem; }}
  #messages {{ flex:1; min-height:0; overflow-y:auto; display:flex; flex-direction:column;
    gap:9px; padding:16px; scroll-behavior:smooth; }}
  .bubble {{ width:fit-content; max-width:80%; padding:9px 11px; border:1px solid var(--line);
    border-radius:11px; box-shadow:0 5px 18px #0005; }}
  .bubble.server {{ align-self:flex-start; background:var(--in); border-bottom-left-radius:2px; }}
  .bubble.worker {{ align-self:flex-end; background:var(--out); border-bottom-right-radius:2px; }}
  .bubble.system {{ align-self:center; width:min(92%,900px); max-width:92%;
    background:#151d27; border-style:dashed; }}
  .meta {{ margin-bottom:5px; color:#8bb2bc; font:600 .68rem/1.35 Consolas,monospace; }}
  .content {{ white-space:pre-wrap; overflow-wrap:anywhere; font:.8rem/1.45 Consolas,monospace; }}
  .media {{ display:grid; gap:7px; margin-top:8px; }}
  .media figure {{ margin:0; }}
  .media img {{ display:block; max-width:100%; max-height:480px; border:1px solid var(--line);
    border-radius:7px; background:#02090d; object-fit:contain; }}
  .media audio {{ display:block; width:min(100%,520px); }}
  .media figcaption {{ margin-top:4px; color:var(--muted); font-size:.67rem; }}
  .media-error {{ color:#e0ad68; font-size:.72rem; }}
</style>
</head>
<body>
<header>
  <h1>Socket worker · <code>{normalized}</code></h1>
  <button id="autoscroll" type="button" aria-pressed="true">Autoscroll: on</button>
  <button id="reload" type="button">Reload</button>
  <a href="{raw_url}" target="_blank">Raw JSONL</a>
  <span id="status">loading…</span>
</header>
<main id="messages" tabindex="0" aria-label="Worker socket conversation"></main>
<script>
const WORKER = {worker_json};
const RAW_URL = {raw_url_json};
const messages = document.getElementById('messages');
const autoscrollButton = document.getElementById('autoscroll');
let autoscroll = true;
let lastText = '';
function esc(value) {{
  return String(value).replace(/[&<>"']/g, c => ({{
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
  }}[c]));
}}
function setAutoscroll(enabled) {{
  autoscroll = enabled;
  autoscrollButton.textContent = 'Autoscroll: ' + (enabled ? 'on' : 'off');
  autoscrollButton.setAttribute('aria-pressed', String(enabled));
  if (enabled) scrollToBottom();
}}
function scrollToBottom() {{
  messages.scrollTop = messages.scrollHeight;
}}
function recordContent(record) {{
  if (record.record_type === 'worker_start_prompt') return record.prompt || '(startup prompt unavailable)';
  if (record.record_type === 'segment_boundary') {{
    return 'Segment ' + record.segment + ' · ' + record.file + ' · ' + record.bytes + ' bytes';
  }}
  const frame = record.frame || {{}};
  for (const key of ['prompt','content','reason','reply']) {{
    if (frame[key] != null) return typeof frame[key] === 'string'
      ? frame[key] : JSON.stringify(frame[key], null, 2);
  }}
  return JSON.stringify(frame, null, 2);
}}
function mediaContent(record) {{
  const media = Array.isArray(record.media) ? record.media : [];
  if (!media.length) return '';
  return '<div class="media">' + media.map(item => {{
    const description = esc(
      (item.mime_type || item.kind || 'media') + ' · ' +
      (item.bytes == null ? 'unknown size' : item.bytes + ' bytes')
    );
    if (!item.available || !item.url) {{
      return '<div class="media-error">' + description + ' · ' +
        esc(item.error || 'preview unavailable') + '</div>';
    }}
    const url = esc(item.url);
    if (item.kind === 'image') {{
      return '<figure><img src="' + url + '" alt="Socket image preview" loading="lazy">' +
        '<figcaption>' + description + '</figcaption></figure>';
    }}
    if (item.kind === 'audio') {{
      return '<figure><audio controls preload="metadata" src="' + url + '"></audio>' +
        '<figcaption>' + description + '</figcaption></figure>';
    }}
    return '';
  }}).join('') + '</div>';
}}
function render(text) {{
  if (text === lastText) {{ if (autoscroll) scrollToBottom(); return; }}
  const previousTop = messages.scrollTop;
  const records = text.split(/\\r?\\n/).filter(Boolean).map(line => {{
    try {{ return JSON.parse(line); }}
    catch (error) {{ return {{record_type:'parse_error', line, error:String(error)}}; }}
  }});
  messages.innerHTML = records.map(record => {{
    const side = record.direction === 'outbound' ? 'server' :
      (record.direction === 'inbound' ? 'worker' : 'system');
    const from = record.sender || record.from || (side === 'server' ? 'EMULLM' :
      (side === 'worker' ? WORKER : 'SYSTEM'));
    const speaker = 'from: ' + from;
    const frame = record.frame || {{}};
    const label = record.record_type || frame.type || 'frame';
    const id = frame.id ? ' · ' + frame.id : '';
    const duration = frame.duration_ms != null ? ' · ' + frame.duration_ms + ' ms' : '';
    const timestamp = record.timestamp ? ' · ' + new Date(record.timestamp).toLocaleTimeString() : '';
    const precision = record.precision_clock_decimal
      ? ' · clock ' + record.precision_clock_decimal : '';
    return '<article class="bubble ' + side + '"><div class="meta">' +
      esc(speaker + ' · ' + label + id + duration + timestamp + precision) +
      '</div><div class="content">' + esc(recordContent(record)) + '</div>' +
      mediaContent(record) + '</article>';
  }}).join('');
  lastText = text;
  if (autoscroll) scrollToBottom(); else messages.scrollTop = previousTop;
}}
async function loadLog() {{
  try {{
    const response = await fetch(RAW_URL, {{cache:'no-store'}});
    if (!response.ok) throw new Error('HTTP ' + response.status);
    const text = await response.text();
    render(text);
    document.getElementById('status').textContent =
      WORKER + ' · updated ' + new Date().toLocaleTimeString();
  }} catch (error) {{
    document.getElementById('status').textContent = 'load failed: ' + error;
  }}
}}
autoscrollButton.addEventListener('click', () => setAutoscroll(!autoscroll));
document.getElementById('reload').addEventListener('click', loadLog);
for (const eventName of ['wheel','touchstart','pointerdown']) {{
  messages.addEventListener(eventName, () => setAutoscroll(false), {{passive:true}});
}}
messages.addEventListener('keydown', event => {{
  if (['ArrowUp','ArrowDown','PageUp','PageDown','Home','End',' '].includes(event.key)) {{
    setAutoscroll(false);
  }}
}});
loadLog();
setInterval(loadLog, 2000);
</script>
</body>
</html>""",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/admin/emullm/websockets/{worker_id}/media/{filename}")
@router.get("/emullm/admin/websockets/{worker_id}/media/{filename}")
def admin_worker_socket_media(worker_id: str, filename: str) -> FileResponse:
    try:
        normalized = _mailbox_id(worker_id)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if not re.fullmatch(r"[0-9a-f]{64}\.(?:png|jpg|gif|webp|bmp|wav|mp3|ogg|flac|m4a|webm|bin)", filename):
        raise HTTPException(status_code=404, detail="unknown socket media artifact")
    path = _socket_worker_media_dir(normalized) / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="socket media artifact is unavailable")
    media_type = {
        ".m4a": "audio/mp4",
        ".bin": "application/octet-stream",
    }.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0]
    return FileResponse(
        path,
        media_type=media_type or "application/octet-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/admin/emullm/clients")
@router.get("/emullm/admin/clients")
def admin_list_clients() -> dict[str, Any]:
    """List server-lifetime logical clients observed on OpenAI-compatible HTTP routes."""
    clients = _openai_client_rows()
    requests = _openai_request_rows()
    return {
        "count": len(clients),
        "active_count": sum(1 for client in clients if client["connected"]),
        "active_requests": sum(int(client["active_requests"]) for client in clients),
        "clients": clients,
        "request_count": len(requests),
        "requests": requests,
    }


def _require_local_process_control(request: Request) -> None:
    host = request.client.host if request.client is not None else ""
    if host == "testclient":
        return
    try:
        is_loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loopback = host.lower() == "localhost"
    if not is_loopback:
        raise HTTPException(
            status_code=403,
            detail="server process controls are restricted to loopback clients",
        )


@router.post("/admin/emullm/shutdown", status_code=202)
@router.post("/emullm/admin/shutdown", status_code=202)
def admin_shutdown_server(request: Request) -> dict[str, Any]:
    """Gracefully stop the standalone EMULLM process after returning."""
    _require_local_process_control(request)
    if not _process_control.schedule_shutdown():
        raise HTTPException(
            status_code=409,
            detail="shutdown is unavailable in embedded mode",
        )
    return {"status": "shutting_down", "pid": os.getpid()}


@router.post("/admin/emullm/restart", status_code=202)
@router.post("/emullm/admin/restart", status_code=202)
def admin_restart_server(request: Request) -> dict[str, Any]:
    """Gracefully replace the standalone EMULLM process after returning."""
    _require_local_process_control(request)
    host = os.environ.get("EMULLM_HOST", "127.0.0.1")
    port = int(os.environ.get("EMULLM_HTTP_PORT", "8801"))
    try:
        helper_pid = _process_control.schedule_restart(host, port)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {
        "status": "restarting",
        "pid": os.getpid(),
        "helper_pid": helper_pid,
        "host": host,
        "port": port,
    }


@router.post("/admin/emullm/test-chat")
@router.post("/emullm/admin/test-chat")
async def admin_test_chat(body: AdminTestChatRequest) -> Any:
    """Run one cancellable multimodal request for the admin test client."""
    if body.request_id in _admin_test_tasks:
        raise HTTPException(
            status_code=409,
            detail=f"test request '{body.request_id}' is already active",
        )
    attachment_records: list[dict[str, Any]] = []
    total_bytes = 0
    for index, attachment in enumerate(body.attachments, start=1):
        mime_type = (
            attachment.mime_type.split(";", 1)[0].strip().lower()
            or "application/octet-stream"
        )
        anonymous_name = _anonymous_attachment_name(index)
        try:
            data = base64.b64decode(attachment.data_b64, validate=True)
        except (binascii.Error, ValueError) as error:
            raise HTTPException(
                status_code=422,
                detail=f"attachment '{anonymous_name}' is not valid base64",
            ) from error
        if not data:
            raise HTTPException(
                status_code=422,
                detail=f"attachment '{anonymous_name}' is empty",
            )
        if len(data) > _MAX_ADMIN_TEST_ATTACHMENT_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"attachment '{anonymous_name}' exceeds the "
                    f"{_MAX_ADMIN_TEST_ATTACHMENT_BYTES}-byte limit"
                ),
            )
        total_bytes += len(data)
        if total_bytes > _MAX_ADMIN_TEST_ATTACHMENTS_TOTAL_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    "attachments exceed the "
                    f"{_MAX_ADMIN_TEST_ATTACHMENTS_TOTAL_BYTES}-byte total limit"
                ),
            )
        record = _store_cloud_bytes(
            data,
            anonymous_name,
            purpose="user_data",
            mime_type=mime_type,
        )
        attachment_records.append(
            {
                "file_id": record["id"],
                "name": record["filename"],
                "mime_type": mime_type,
                "bytes": len(data),
                "url": _cloud_file_url(record["id"]),
            }
        )

    images = [
        attachment["url"]
        for attachment in attachment_records
        if attachment["mime_type"].startswith("image/")
    ]
    audio_files = [
        attachment["url"]
        for attachment in attachment_records
        if attachment["mime_type"].startswith("audio/")
    ]
    if images:
        kind = "vision"
    elif audio_files:
        kind = "audio_attachment"
    elif attachment_records:
        kind = "file_attachment"
    else:
        kind = "chat"

    async def execute() -> dict[str, Any]:
        prompt_text = f"[user] {body.prompt}"
        result = await _relay_full(
            body.model,
            prompt_text,
            images=images or None,
            audio=audio_files[0] if audio_files else None,
            files={"attachments": attachment_records} if attachment_records else None,
            kind=kind,
            attachments=attachment_records or None,
            required_capabilities={
                capability.strip()
                for capability in body.required_capabilities
                if capability.strip()
            },
        )
        reply_text = _reply_content(result)
        usage = {
            "prompt_tokens": _token_count(prompt_text),
            "completion_tokens": _token_count(reply_text),
        }
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        return {
            "id": _new_resource_id("chatcmpl"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
            "attachments": attachment_records,
            "request_kind": kind,
        }

    task = asyncio.create_task(execute())
    _admin_test_tasks[body.request_id] = task
    try:
        return await task
    except asyncio.CancelledError as error:
        raise HTTPException(status_code=499, detail="test request cancelled") from error
    finally:
        if _admin_test_tasks.get(body.request_id) is task:
            _admin_test_tasks.pop(body.request_id, None)


@router.delete("/admin/emullm/test-chat/{request_id}")
@router.delete("/emullm/admin/test-chat/{request_id}")
async def admin_cancel_test_chat(request_id: str) -> dict[str, Any]:
    """Cancel an active admin test request and notify its worker."""
    task = _admin_test_tasks.get(request_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"no active test request '{request_id}'")
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    return {"request_id": request_id, "cancelled": True}


@router.put("/admin/emullm/agents/{worker_id}/enabled")
@router.put("/emullm/admin/agents/{worker_id}/enabled")
def admin_set_agent_enabled(worker_id: str, body: AgentEnabledRequest) -> dict[str, Any]:
    """Persist an agent's enabled flag; mock agents are toggled immediately."""
    config = _read_config()
    agents = config.get("agents") if isinstance(config.get("agents"), list) else []
    agent = next(
        (
            candidate
            for candidate in agents
            if isinstance(candidate, dict)
            and str(candidate.get("id") or candidate.get("worker_id") or "") == worker_id
        ),
        None,
    )
    if agent is None:
        raise HTTPException(status_code=404, detail=f"no configured agent '{worker_id}'")
    agent["enabled"] = body.enabled
    try:
        EmullmConfig.model_validate(config)
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=error.errors()) from error
    _copilot_api.write_config_document(_CONFIG_PATH, config)

    launch = str(agent.get("launch") or "").lower()
    if launch == "mock" and not body.enabled:
        unregister_mock_workers([worker_id])
    expanded = _sup.expand_agents(config)
    apply_agent_policies(expanded)
    if launch == "mock" and body.enabled:
        spec = next(
            (
                item
                for item in expanded.get("mock_workers", [])
                if isinstance(item, dict)
                and str(item.get("id") or item.get("worker_id") or "") == worker_id
            ),
            None,
        )
        if spec is not None:
            register_mock_workers([spec])
    return {
        "worker_id": worker_id,
        "enabled": body.enabled,
        "applied": launch == "mock",
        "restart_required": launch != "mock",
        "agents": _configured_agent_rows(),
    }


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
<meta name="description" content="EMULLM operations console for resident model servants, routing, WebSockets, and configuration.">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='12' fill='%23071720'/%3E%3Ctext x='32' y='43' text-anchor='middle' font-size='38' font-family='monospace' font-weight='700' fill='%2325d5c8'%3EE%3C/text%3E%3C/svg%3E">
<title>emullm -- admin</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #050d13; --sidebar: #071720; --panel: #0a1b25; --panel-2: #0d2430;
    --line: #173844; --line-soft: #102b35; --text: #d9edf1; --muted: #7895a0;
    --accent: #25d5c8; --accent-soft: #123d42; --green: #35d48a;
    --amber: #f3b95f; --red: #ff6b72; --shadow: 0 14px 38px #0008;
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body { margin: 0; min-height: 100vh; background: radial-gradient(circle at 75% -20%, #123341 0, transparent 38%), var(--bg); color: var(--text); font-family: Inter, "Segoe UI", system-ui, sans-serif; }
  .app-shell { min-height: 100vh; display: grid; grid-template-columns: 220px minmax(0, 1fr); }
  .sidebar { position: sticky; top: 0; height: 100vh; padding: 18px 14px; background: linear-gradient(180deg, #081b25, #06131b); border-right: 1px solid var(--line); display: flex; flex-direction: column; gap: 18px; }
  .brand { display: flex; align-items: center; gap: 10px; padding: 4px 6px 14px; border-bottom: 1px solid var(--line-soft); }
  .brand-mark { width: 30px; height: 30px; display: grid; place-items: center; border: 1px solid #25d5c899; border-radius: 7px; background: #25d5c81c; color: var(--accent); font: 800 14px/1 monospace; box-shadow: 0 0 24px #25d5c822; }
  .brand strong { color: var(--accent); letter-spacing: .16em; font-size: .83rem; }
  .brand small { display: block; margin-top: 2px; color: var(--muted); font-size: .65rem; letter-spacing: .04em; }
  .nav-group { display: grid; gap: 5px; }
  .nav-label { padding: 7px 9px 3px; color: #4f717d; font-size: .61rem; font-weight: 800; letter-spacing: .15em; text-transform: uppercase; }
  .nav-link { display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 8px 9px; border: 1px solid transparent; border-radius: 6px; color: #92b2bc; text-decoration: none; font-size: .78rem; transition: .15s ease; }
  .nav-link:hover { color: var(--text); border-color: var(--line); background: #0d2631; transform: translateX(2px); }
  .nav-link b { min-width: 22px; padding: 1px 6px; border-radius: 999px; background: #102d38; color: var(--accent); text-align: center; font-size: .65rem; }
  .sidebar-foot { margin-top: auto; padding: 10px 9px; border-top: 1px solid var(--line-soft); color: var(--muted); font-size: .68rem; line-height: 1.5; }
  .workspace { min-width: 0; }
  .topbar { position: sticky; top: 0; z-index: 10; display: flex; justify-content: space-between; align-items: center; gap: 18px; padding: 13px 22px; border-bottom: 1px solid var(--line); background: #07141de8; backdrop-filter: blur(14px); }
  .topbar h1 { margin: 0; color: var(--accent); font: 800 .95rem/1.1 monospace; letter-spacing: .1em; }
  .topbar .sub { margin: 3px 0 0; }
  .chips { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 7px; }
  .chip { padding: 5px 9px; border: 1px solid var(--line); border-radius: 999px; background: #0b222c; color: #9bb8c1; font-size: .68rem; }
  .chip b { color: var(--accent); }
  .content { width: min(1500px, 100%); margin: 0 auto; padding: 18px 22px 50px; display: grid; gap: 15px; }
  .warn { margin: 0; padding: 9px 12px; border: 1px solid #f3b95f55; border-radius: 7px; background: #5a3b101f; color: #d7b77f; font-size: .73rem; }
  .panel { min-width: 0; padding: 15px 16px 16px; border: 1px solid var(--line); border-radius: 9px; background: linear-gradient(145deg, #0a1d27f2, #071720f2); box-shadow: var(--shadow); }
  .panel:target { border-color: #25d5c888; box-shadow: 0 0 0 1px #25d5c822, var(--shadow); }
  .panel h2 { display: flex; flex-wrap: wrap; align-items: center; gap: 7px; margin: 0 0 9px; color: #c7e1e6; font-size: .9rem; letter-spacing: .02em; }
  .panel h3 { margin: 14px 0 6px; color: #92b7c0; font-size: .75rem; text-transform: uppercase; letter-spacing: .08em; }
  .sub { color: var(--muted); font-size: .74rem; line-height: 1.55; }
  .muted { color: var(--muted); }
  .msg { margin-left: 5px; font-size: .7rem; }
  .ok { color: var(--green); } .err { color: var(--red); }
  a { color: #5be2d7; }
  button { min-height: 29px; margin: 0 3px 3px 0; padding: 5px 10px; border: 1px solid #24505d; border-radius: 5px; background: linear-gradient(#113440, #0c2832); color: #cce5e9; font: 650 .69rem/1 "Segoe UI", sans-serif; cursor: pointer; transition: .15s ease; }
  button:hover:not(:disabled) { border-color: var(--accent); color: white; background: #15505a; box-shadow: 0 0 14px #25d5c822; }
  button:active:not(:disabled) { transform: translateY(1px); }
  button:disabled { cursor: not-allowed; opacity: .38; }
  button[type="submit"] { border-color: #25d5c899; color: #06171b; background: var(--accent); }
  button.danger { border-color: #ff6b7266; color: #ff9da2; background: #401b22; }
  input, select, textarea { width: 100%; border: 1px solid #1a414d; border-radius: 5px; outline: none; background: #06151d; color: #d8eef1; font: 400 .74rem/1.35 "Segoe UI", sans-serif; transition: border-color .15s, box-shadow .15s; }
  input, select { min-height: 32px; padding: 5px 8px; }
  textarea { min-height: 16rem; padding: 9px; resize: vertical; font-family: "Cascadia Code", Consolas, monospace; }
  input:focus, select:focus, textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 2px #25d5c81f; }
  input[type="checkbox"] { width: 14px; min-height: 14px; accent-color: var(--accent); }
  .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(185px, 1fr)); gap: 9px 12px; }
  .form-grid label { display: flex; flex-direction: column; gap: 4px; color: #71949f; font-size: .67rem; font-weight: 650; }
  .checks { display: flex; flex-wrap: wrap; gap: 8px 15px; margin: 10px 0; padding: 9px; border: 1px solid var(--line-soft); border-radius: 6px; background: #06141c; }
  .checks label { display: inline-flex; align-items: center; gap: 6px; color: #8eacb5; font-size: .68rem; }
  .poll-control { display: inline-flex; align-items: center; gap: 5px; }
  .poll-control select { width: auto; min-height: 25px; padding: 2px 5px; font-size: .65rem; }
  #copilot-prompt { min-height: 90px; }
  #model-test-prompt { min-height: 84px; }
  .attachment-drop { margin-top: 9px; padding: 12px; border: 1px dashed #2b6370; border-radius: 7px; background: #06141c; text-align: center; color: #83a6b0; transition: .15s ease; }
  .attachment-drop.dragging { border-color: var(--accent); background: #0d3137; color: var(--accent); }
  .attachment-drop input { width: auto; max-width: 100%; }
  .attachment-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(205px, 1fr)); gap: 8px; margin-top: 9px; }
  .attachment-card { min-width: 0; padding: 8px; border: 1px solid var(--line-soft); border-radius: 6px; background: #06141c; }
  .attachment-card img, .attachment-card video { display: block; width: 100%; height: 110px; margin-bottom: 7px; border-radius: 4px; background: #02090d; object-fit: contain; }
  .attachment-card audio { width: 100%; height: 34px; margin: 5px 0; }
  #image-generation-preview { display: none; width: min(100%, 640px); max-height: 480px; margin-top: 9px; border: 1px solid var(--line); border-radius: 6px; background: #02090d; object-fit: contain; }
  .attachment-card .file-icon { display: grid; place-items: center; height: 58px; margin-bottom: 6px; border-radius: 4px; background: #0d2833; color: var(--accent); font: 800 1.1rem monospace; }
  .attachment-name { overflow: hidden; color: #c5e0e5; font-size: .7rem; font-weight: 700; text-overflow: ellipsis; white-space: nowrap; }
  .attachment-meta { color: var(--muted); font-size: .62rem; line-height: 1.45; overflow-wrap: anywhere; }
  .attachment-result { margin-top: 10px; }
  .capability-card { margin: 9px 0; padding: 10px 12px; border: 1px solid #1b4651; border-radius: 7px; background: linear-gradient(135deg, #071921, #0b2630); color: #91adb6; font-size: .68rem; line-height: 1.55; }
  .capability-card:empty { display: none; }
  .capability-card strong { color: #d5edf1; }
  .capability-badges { display: flex; flex-wrap: wrap; gap: 5px; margin: 6px 0; }
  .capability-badge { padding: 2px 6px; border: 1px solid #27606d; border-radius: 999px; color: #86dcd5; background: #0b3038; font-size: .61rem; }
  .table-wrap { width: 100%; overflow-x: auto; border: 1px solid var(--line-soft); border-radius: 6px; }
  table { width: 100%; border-collapse: collapse; }
  th, td { padding: 8px 9px; border-bottom: 1px solid var(--line-soft); text-align: left; vertical-align: top; font-size: .7rem; }
  th { position: sticky; top: 0; background: #0b222c; color: #7fa5b0; font-size: .62rem; letter-spacing: .08em; text-transform: uppercase; }
  tbody tr:hover { background: #0e2a354f; }
  tbody tr.selected-row { background: #16404b99; }
  tbody tr:last-child td { border-bottom: 0; }
  pre { min-height: 42px; margin: 6px 0; padding: 10px; border: 1px solid var(--line-soft); border-radius: 6px; background: #041117; color: #b9d7dd; white-space: pre-wrap; overflow-wrap: anywhere; font: .72rem/1.5 "Cascadia Code", Consolas, monospace; }
  details { margin-top: 9px; padding: 9px 11px; border: 1px solid var(--line-soft); border-radius: 6px; background: #071821; }
  summary { cursor: pointer; color: #9cc0c7; font-size: .72rem; font-weight: 700; }
  details[open] summary { margin-bottom: 10px; color: var(--accent); }
  code { padding: 1px 4px; border-radius: 3px; background: #102b35; color: #8be9e0; font: .66rem "Cascadia Code", Consolas, monospace; }
  .action-link { display: inline-flex; min-width: 28px; min-height: 28px; align-items: center; justify-content: center; margin-left: 3px; border: 1px solid var(--line); border-radius: 5px; text-decoration: none; }
  .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
  .dot { display: inline-block; width: 7px; height: 7px; margin-right: 5px; border-radius: 50%; box-shadow: 0 0 8px currentColor; }
  .on { color: var(--green); background: var(--green); } .off { color: #607780; background: #607780; }
  .two-column { display: grid; grid-template-columns: minmax(0, 1fr) minmax(320px, .72fr); gap: 15px; align-items: start; }
  .section-editor { display: grid; grid-template-columns: 210px minmax(0, 1fr); gap: 12px; align-items: start; }
  .section-tabs { display: grid; gap: 5px; }
  .section-tab { width: 100%; display: flex; justify-content: space-between; text-align: left; }
  .section-tab.active { border-color: var(--accent); color: var(--accent); background: var(--accent-soft); }
  #config-section-editor { min-height: 300px; }
  .model-config-layout { display: grid; grid-template-columns: minmax(240px, .7fr) minmax(0, 1.8fr); gap: 12px; align-items: start; }
  #model-config-list { min-height: 390px; font-family: "Cascadia Code", Consolas, monospace; }
  #model-config-json { min-height: 320px; }
  #model-config-route { min-height: 105px; }
  .slot-grid { display: grid; gap: 5px; margin-top: 9px; }
  .slot-card { padding: 7px 8px; border: 1px solid var(--line-soft); border-radius: 5px; background: #06141c; color: var(--muted); font-size: .64rem; overflow-wrap: anywhere; }
  .slot-card b { color: #c7e2e7; }
  .route-order { display: flex; flex-wrap: wrap; gap: 6px; margin: 7px 0 9px; }
  .route-order-item { width: max-content; max-width: 100%; display: grid; grid-template-columns: auto max-content auto; gap: 5px; align-items: center; padding: 5px 6px; border: 1px solid var(--line-soft); border-radius: 5px; background: #06141c; }
  .route-order-item code { max-width: 310px; overflow: hidden; color: #b9d8de; text-overflow: ellipsis; white-space: nowrap; }
  .route-order-item button { min-width: 32px; padding: 3px 7px; }
  .sort-button { padding: 0; border: 0; background: transparent; color: inherit; font: inherit; }
  .stats-scroll { max-height: 210px; overflow: auto; }
  .stats-scroll.expanded { max-height: none; }
  .stats-scroll thead { position: sticky; top: 0; z-index: 2; background: var(--panel-2); }
  .stats-scroll tfoot { position: sticky; bottom: 0; z-index: 2; background: var(--panel-2); }
  #anti-idle-table { width: max-content; min-width: 100%; }
  #anti-idle-table th, #anti-idle-table td {
    height: 38px; vertical-align: middle; white-space: nowrap;
  }
  @media (max-width: 1050px) { .two-column { grid-template-columns: 1fr; } }
  @media (max-width: 760px) {
    .app-shell { grid-template-columns: 1fr; }
    .sidebar { position: static; width: auto; height: auto; }
    .nav-group { grid-template-columns: repeat(2, 1fr); }
    .nav-label, .sidebar-foot { display: none; }
    .topbar { position: static; align-items: flex-start; flex-direction: column; }
    .chips { justify-content: flex-start; }
    .content { padding: 12px; }
    .section-editor { grid-template-columns: 1fr; }
    .model-config-layout { grid-template-columns: 1fr; }
    .section-tabs { grid-template-columns: repeat(2, 1fr); }
  }
</style>
</head>
<body>
<div class="app-shell">
<aside class="sidebar">
  <div class="brand"><div class="brand-mark">E</div><div><strong>EMULLM</strong><small>operations console</small></div></div>
  <nav class="nav-group">
    <div class="nav-label">Live</div>
    <a class="nav-link" href="#overview"><span>Overview</span><b>LIVE</b></a>
    <a class="nav-link" href="#connections"><span>WebSockets</span><b id="nav-socket-count">0</b></a>
    <a class="nav-link" href="#clients"><span>FastAPI requests</span><b id="nav-client-count">0</b></a>
    <a class="nav-link" href="#telemetry"><span>Request telemetry</span><b id="nav-waiting-count">0</b></a>
    <a class="nav-link" href="#servants"><span>Headless servants</span><b id="nav-servant-count">0</b></a>
    <div class="nav-label">Control</div>
    <a class="nav-link" href="#agents"><span>Agent controls</span><b>CFG</b></a>
    <a class="nav-link" href="#server-settings"><span>Server settings</span><b>CFG</b></a>
    <a class="nav-link" href="#backend-config"><span>Backends</span><b id="nav-backend-count">0</b></a>
    <a class="nav-link" href="#codex-suppliers"><span>Codex suppliers</span><b id="nav-supplier-count">0</b></a>
    <a class="nav-link" href="#anti-idle"><span>Anti-idle prompts</span><b id="nav-anti-idle-count">0</b></a>
    <a class="nav-link" href="#managed"><span>Managed workers</span><b id="nav-managed-count">0</b></a>
    <a class="nav-link" href="#model-configurator"><span>Models configurator</span><b id="nav-config-model-count">0</b></a>
    <a class="nav-link" href="#test-client"><span>Model test client</span><b id="nav-model-count">0</b></a>
    <a class="nav-link" href="#config-sections"><span>Config sections</span><b>EDIT</b></a>
    <a class="nav-link" href="#configuration"><span>Raw configuration</span><b>JSON</b></a>
  </nav>
  <div class="sidebar-foot">OpenAI-compatible relay<br><code>/v1</code> &middot; <code>/emullm/ws</code></div>
</aside>
<main class="workspace">
<header class="topbar" id="overview">
  <div><h1>EMULLM // CONTROL PLANE</h1><p class="sub">resident model servants and relay operations</p></div>
  <div class="chips">
    <span class="chip">MODE <b id="mode">-</b></span>
    <span class="chip">SOCKETS <b id="top-socket-count">0</b></span>
    <span class="chip">CLIENTS <b id="top-client-count">0/0</b></span>
    <span class="chip">SERVANTS <b id="top-servant-count">0</b></span>
    <span class="chip">MODELS <b id="top-model-count">0</b></span>
    <span class="chip">ACTIVE <b id="top-active-count">0</b></span>
    <span class="chip">WAITING <b id="top-waiting-count">0</b></span>
    <span class="chip">STUCK <b id="top-stuck-count">0</b></span>
    <span class="chip">SERVED <b id="top-served-count">0</b></span>
    <span class="chip">SWITCHES <b id="top-switch-count">0</b></span>
    <span class="chip">UPTIME <b id="top-uptime">0s</b></span>
    <span class="chip"><a href="/emullm/status">STATUS</a></span>
    <label class="chip poll-control">POLL
      <select id="poll-window">
        <option value="0">continuous</option>
        <option value="60000">1 min</option>
        <option value="120000" selected>2 min</option>
        <option value="300000">5 min</option>
      </select>
    </label>
    <label class="chip poll-control"><input id="poll-hidden" type="checkbox"> HIDDEN / 2 MIN</label>
    <button id="poll-wake" type="button">Wake / refresh</button>
    <span class="chip" id="poll-status">polling</span>
    <span class="chip" id="updated">-</span>
    <button id="server-restart" type="button">Restart</button>
    <button id="server-shutdown" class="danger" type="button">Shutdown</button>
  </div>
</header>
<div class="content">
<p class="warn">LOCAL CONTROL SURFACE &mdash; edits configuration, launches processes, and controls active model requests. Keep it bound to localhost.</p>

<section class="panel" id="agents">
<h2>Configured agents</h2>
<label><input id="carol-enabled" type="checkbox" style="width:auto"> Enable Carol mock worker</label>
<span id="carol-note" class="msg muted"></span>
</section>

<section class="panel" id="server-settings">
<h2>Server configuration</h2>
<p class="sub">Common deployment settings. Save persists them to config.json; restart EMULLM to apply startup-level changes.</p>
<div class="form-grid">
  <label>Description<input id="server-description"></label>
  <label>Mode chain<input id="server-mode" placeholder="recruit,mock"></label>
  <label>Capability fallback
    <select id="server-capability-fallback"><option value="stub">stub</option><option value="wait">wait</option><option value="error">error</option></select>
  </label>
  <label>Default subagent model<input id="server-subagent-model"></label>
  <label>Maximum simultaneous calls<input id="server-max-concurrent" type="number" min="4" max="50" value="50"></label>
  <label>Idle worker target<input id="server-idle-workers" type="number" min="0" max="50" value="5"></label>
  <label>Idle grace (seconds)<input id="server-idle-grace" type="number" min="0" max="3600" step="1" value="30"></label>
  <label>Backend fallback delay (seconds)<input id="server-backend-delay" type="number" min="0" max="300" step="0.5" value="5"></label>
  <label>Validation interval default<input id="server-validation-default" placeholder="never, 1day, 12h"></label>
  <label>Validation override<input id="server-validation-override" placeholder="blank = none"></label>
</div>
<div style="margin-top:10px">
  <button id="server-settings-save" type="button">Save server settings</button>
  <button id="server-settings-reload" type="button">Reload</button>
  <span id="server-settings-msg" class="msg"></span>
</div>
</section>

<section class="panel" id="backend-config">
<h2>OpenAI-compatible backends <button id="backend-new" type="button">Add backend</button>
  <button id="backend-refresh" type="button">Refresh</button>
  <span id="backend-note" class="muted"></span></h2>
<p class="sub">Configures both direct <code>backends[]</code> entries and
  <code>launch: proxy</code> agents. Existing proxy-agent service catalogs are
  preserved. Inline API keys are write-only here; environment-variable keys are preferred.</p>
<div class="table-wrap">
<table>
  <thead><tr><th>Name</th><th>Source</th><th>Base URL</th><th>Model</th><th>Credential</th><th>Default</th><th>Actions</th></tr></thead>
  <tbody id="backend-config-rows"><tr><td colspan="7" class="muted">loading...</td></tr></tbody>
</table>
</div>
<details id="backend-editor">
  <summary id="backend-editor-title">Add backend</summary>
  <form id="backend-form">
    <div class="form-grid">
      <label>Name<input id="backend-name" required pattern="[A-Za-z0-9][A-Za-z0-9_.-]*" maxlength="100"></label>
      <label>Base URL<input id="backend-base-url" required placeholder="https://provider.example/v1"></label>
      <label>Default model<input id="backend-model" placeholder="provider/model"></label>
      <label>API-key environment variable<input id="backend-api-key-env" placeholder="PROVIDER_API_KEY"></label>
      <label>Inline API key (leave blank to preserve)<input id="backend-api-key" type="password" autocomplete="new-password"></label>
      <label>Validation interval<input id="backend-validation" placeholder="never, 1day, 12h"></label>
      <label>Description<input id="backend-description"></label>
    </div>
    <div class="checks">
      <label><input id="backend-default" type="checkbox"> default backend</label>
      <label><input id="backend-clear-api-key" type="checkbox"> clear stored inline API key</label>
    </div>
    <button type="submit">Save backend</button>
    <button id="backend-cancel" type="button">Cancel</button>
    <span id="backend-msg" class="msg"></span>
  </form>
</details>
</section>

<section class="panel" id="codex-suppliers">
<h2>Codex suppliers <button id="supplier-new" type="button">Add supplier</button>
  <button id="supplier-refresh" type="button">Refresh</button>
  <span id="supplier-note" class="muted"></span></h2>
<p class="sub">Declares providers capable of supplying Codex-family workers and models.
  The current supplier is GitHub Copilot through resident <code>worker-copilot-*</code>
  sessions.</p>
<div class="table-wrap">
<table>
  <thead><tr><th>Supplier</th><th>Kind</th><th>Worker pattern</th><th>Models</th><th>State</th><th>Actions</th></tr></thead>
  <tbody id="supplier-rows"><tr><td colspan="6" class="muted">loading...</td></tr></tbody>
</table>
</div>
<details id="supplier-editor">
  <summary id="supplier-editor-title">Add Codex supplier</summary>
  <form id="supplier-form">
    <div class="form-grid">
      <label>Supplier ID<input id="supplier-id" required pattern="[A-Za-z0-9][A-Za-z0-9_.-]*" maxlength="100"></label>
      <label>Name<input id="supplier-name" required maxlength="200"></label>
      <label>Kind<select id="supplier-kind"><option value="copilot">copilot</option><option value="codex-cli">codex-cli</option><option value="openai-compatible">openai-compatible</option><option value="custom">custom</option></select></label>
      <label>Priority<input id="supplier-priority" type="number" min="-10000" max="10000" value="0"></label>
      <label>Worker pattern<input id="supplier-worker-pattern" placeholder="worker-codex-*"></label>
      <label>Model prefix<input id="supplier-model-prefix" placeholder="copilot/"></label>
      <label>Model patterns<input id="supplier-model-patterns" placeholder="*codex*,gpt-5.3-*"></label>
      <label>Command<input id="supplier-command" placeholder="copilot or codex"></label>
      <label>Base URL<input id="supplier-base-url" placeholder="https://provider.example/v1"></label>
      <label>API-key environment variable<input id="supplier-api-key-env" placeholder="PROVIDER_API_KEY"></label>
      <label>Description<input id="supplier-description"></label>
    </div>
    <div class="checks"><label><input id="supplier-enabled" type="checkbox" checked> enabled</label></div>
    <button type="submit">Save supplier</button>
    <button id="supplier-cancel" type="button">Cancel</button>
    <span id="supplier-msg" class="msg"></span>
  </form>
</details>
</section>

<section class="panel" id="anti-idle">
<h2>Anti-idle conversation prompts
  <button id="anti-idle-refresh" type="button">Reload</button>
  <button id="anti-idle-reset-stats" type="button">Reset stats</button>
  <span id="anti-idle-note" class="muted"></span></h2>
<p class="sub">Workers receive these as ordinary short completion turns; the model is not
  told they are maintenance. The initial 50 may grow. Runtime timing is aggregated
  across resident workers, and slow prompts may also be retired independently by a worker.</p>
<div class="form-grid">
  <label>Frequency (seconds)<input id="anti-idle-interval" type="number" min="0" max="3600" step="1" value="40"></label>
  <label>Task timeout (seconds, maximum 10)<input id="anti-idle-timeout" type="number" min="0.1" max="10" step="0.1" value="10"></label>
  <label>Slow budget (seconds)<input id="anti-idle-slow-budget" type="number" min="0.1" max="10" step="0.1" value="8"></label>
</div>
<div class="checks"><label><input id="anti-idle-enabled" type="checkbox" checked> scheduler enabled (applies immediately)</label></div>
<div class="two-column">
  <div>
    <div class="table-wrap stats-scroll" style="max-height:430px">
    <table id="anti-idle-table">
      <thead><tr>
        <th><button class="sort-button" data-anti-sort="number">#</button></th>
        <th><button class="sort-button" data-anti-sort="average_duration_ms">Average</button></th>
        <th><button class="sort-button" data-anti-sort="min_duration_ms">Shortest / worker</button></th>
        <th><button class="sort-button" data-anti-sort="max_duration_ms">Longest / worker</button></th>
        <th><button class="sort-button" data-anti-sort="attempts">Attempts</button></th>
        <th><button class="sort-button" data-anti-sort="timeouts">Timeouts</button></th>
        <th><button class="sort-button" data-anti-sort="slow">Over budget</button></th>
        <th><button class="sort-button" data-anti-sort="retired_workers">Retired by</button></th>
        <th><button class="sort-button" data-anti-sort="deprecated">Deprecated</button></th>
        <th><button class="sort-button" data-anti-sort="prompt">Conversation</button></th>
      </tr></thead>
      <tbody id="anti-idle-list"><tr><td colspan="10" class="muted">loading...</td></tr></tbody>
    </table>
    </div>
    <div style="margin-top:8px">
      <button id="anti-idle-add" type="button">Add prompt</button>
      <button id="anti-idle-save" type="button">Save configuration</button>
      <span id="anti-idle-msg" class="msg"></span>
    </div>
  </div>
  <div>
    <label>Stable prompt ID<input id="anti-idle-id" pattern="[A-Za-z0-9][A-Za-z0-9_.-]*" maxlength="100"></label>
    <label for="anti-idle-text" style="display:block;margin-top:8px">Conversation prompt</label>
    <textarea id="anti-idle-text" style="min-height:150px" maxlength="1000"></textarea>
    <div class="checks">
      <label><input id="anti-idle-deprecated" type="checkbox"> deprecated (scheduler skips it)</label>
    </div>
    <div id="anti-idle-stats" class="capability-card"></div>
  </div>
</div>
</section>

<section class="panel" id="connections">
<h2>Connected WebSockets <button id="refresh-websockets" type="button">List / refresh</button>
  <span id="websocket-note" class="muted"></span></h2>
<div class="table-wrap">
<table>
  <thead><tr><th>Connection</th><th>Kind / endpoint</th><th>Identity / subscription</th><th>Client</th><th>Messages</th><th>Age</th><th>Last satisfied</th><th>Last client work</th></tr></thead>
  <tbody id="websocket-connections"><tr><td colspan="8" class="muted">Click List / refresh.</td></tr></tbody>
</table>
</div>
</section>

<section class="panel" id="clients">
<h2>FastAPI requests <button id="refresh-clients" type="button">List / refresh</button>
  <span id="client-note" class="muted"></span></h2>
<p class="sub">Logical sessions observed on <code>/v1/*</code>. Supply
  <code>X-EmuLLM-Client-ID</code> for a stable identity; otherwise EMULLM groups by
  remote host and User-Agent. “Connected” means at least one request is active.</p>
<h3>Active / recent requests</h3>
<div class="table-wrap stats-scroll">
<table>
  <thead><tr><th>Request</th><th>Client</th><th>Method / endpoint</th><th>State</th><th>Started</th><th>Duration</th></tr></thead>
  <tbody id="fastapi-requests"><tr><td colspan="6" class="muted">Click List / refresh.</td></tr></tbody>
</table>
</div>
<h3>Logical client sessions</h3>
<div class="table-wrap stats-scroll">
<table>
  <thead><tr><th>Client</th><th>Address</th><th>State</th><th>Requests</th><th>Last endpoint</th><th>First seen</th><th>Last seen</th></tr></thead>
  <tbody id="openai-clients"><tr><td colspan="7" class="muted">Click List / refresh.</td></tr></tbody>
</table>
</div>
</section>

<section class="panel" id="telemetry">
<h2>Request telemetry <span id="telemetry-summary" class="muted"></span>
  <label style="display:inline-flex;align-items:center;gap:5px">Footer
    <select id="telemetry-footer-mode" style="width:auto">
      <option value="both">Totals + weighted averages</option>
      <option value="totals">Cumulative totals</option>
      <option value="averages">Cumulative averages</option>
    </select>
  </label>
</h2>
<p class="sub">Server-lifetime service time starts when a request reaches a worker. Waiting shows
  client requests deliberately held because no worker is connected/ready or before backend fallback.</p>
<div id="telemetry-alerts" class="capability-card"></div>
<h3>By service kind</h3>
<div class="table-wrap"><table>
  <thead><tr><th>Service</th><th>Active</th><th>Attempts</th><th>Served</th><th>Deferred</th><th>Rejected</th><th>Failed</th><th>Total time</th><th>Average</th></tr></thead>
  <tbody id="telemetry-services"><tr><td colspan="9" class="muted">No requests served yet.</td></tr></tbody>
  <tfoot id="telemetry-services-total"></tfoot>
</table></div>
<h3>By worker / relayed backend <button id="telemetry-workers-toggle" type="button">Show all</button></h3>
<div class="table-wrap stats-scroll" id="telemetry-workers-wrap"><table>
  <thead><tr>
    <th><button class="sort-button" data-stats-table="workers" data-stats-key="id">Worker</button></th>
    <th><button class="sort-button" data-stats-table="workers" data-stats-key="active">Active</button></th>
    <th><button class="sort-button" data-stats-table="workers" data-stats-key="reserved">Reserved</button></th>
    <th><button class="sort-button" data-stats-table="workers" data-stats-key="attempts">Attempts</button></th>
    <th><button class="sort-button" data-stats-table="workers" data-stats-key="served">Served</button></th>
    <th><button class="sort-button" data-stats-table="workers" data-stats-key="deferred">Deferred</button></th>
    <th><button class="sort-button" data-stats-table="workers" data-stats-key="failed">Failed / rejected</button></th>
    <th><button class="sort-button" data-stats-table="workers" data-stats-key="switches">Model switches</button></th>
    <th><button class="sort-button" data-stats-table="workers" data-stats-key="total_seconds">Total time</button></th>
    <th><button class="sort-button" data-stats-table="workers" data-stats-key="average_seconds">Average</button></th>
  </tr></thead>
  <tbody id="telemetry-workers"><tr><td colspan="10" class="muted">No worker timing yet.</td></tr></tbody>
  <tfoot id="telemetry-workers-total"></tfoot>
</table></div>
<h3>By requested model <button id="telemetry-models-toggle" type="button">Show all</button></h3>
<div class="table-wrap stats-scroll" id="telemetry-models-wrap"><table>
  <thead><tr>
    <th><button class="sort-button" data-stats-table="models" data-stats-key="id">Model</button></th>
    <th><button class="sort-button" data-stats-table="models" data-stats-key="active">Active</button></th>
    <th><button class="sort-button" data-stats-table="models" data-stats-key="attempts">Requests</button></th>
    <th><button class="sort-button" data-stats-table="models" data-stats-key="served">Served</button></th>
    <th><button class="sort-button" data-stats-table="models" data-stats-key="deferred">Deferred</button></th>
    <th><button class="sort-button" data-stats-table="models" data-stats-key="failed">Failed / rejected</button></th>
    <th><button class="sort-button" data-stats-table="models" data-stats-key="total_seconds">Total time</button></th>
    <th><button class="sort-button" data-stats-table="models" data-stats-key="average_seconds">Average</button></th>
  </tr></thead>
  <tbody id="telemetry-models"><tr><td colspan="8" class="muted">No model timing yet.</td></tr></tbody>
  <tfoot id="telemetry-models-total"></tfoot>
</table></div>
<div id="telemetry-waiting" class="capability-card"></div>
</section>

<section class="panel" id="managed">
<h2>Managed workers <button id="refresh-services" type="button">Refresh services</button>
  <span id="sup-note" class="muted"></span></h2>
<div class="table-wrap">
<table>
  <thead><tr><th>Worker</th><th>Role</th><th>PID</th><th>State</th><th>Actions</th></tr></thead>
  <tbody id="workers"><tr><td colspan="5" class="muted">loading...</td></tr></tbody>
</table>
</div>
</section>

<section class="panel" id="servants">
<h2>Headless Copilot servants <button id="refresh-copilots" type="button">Refresh</button>
  <button id="refresh-copilot-models" type="button">Refresh Copilot LLMs</button>
  <span id="copilot-note" class="muted"></span></h2>
<div style="margin-bottom:9px">
  <button id="copilot-start-all" type="button">Start all offline</button>
  <button id="copilot-stop-all" type="button">Stop all online</button>
  <button id="copilot-stop-idle" type="button">Stop idle workers</button>
  <button id="copilot-restart-all" type="button">Restart all online</button>
  <button id="copilot-reset-all" type="button">New sessions for all online</button>
  <span id="copilot-bulk-msg" class="msg"></span>
</div>
<p class="sub">Each servant connects to <code>/emullm/ws</code> and reuses one Copilot
  SDK/CLI process and session across requests. Tool access is disabled unless explicitly enabled.
  <span id="copilot-model-note"></span></p>
<div class="table-wrap">
<table>
  <thead><tr><th>Servant</th><th>Copilot / session</th><th>Models</th><th>State</th><th>Actions</th></tr></thead>
  <tbody id="copilots"><tr><td colspan="5" class="muted">loading...</td></tr></tbody>
</table>
</div>

<details id="copilot-editor">
  <summary id="copilot-editor-title">Add headless Copilot servant</summary>
  <form id="copilot-form">
    <div class="form-grid">
      <label>Worker ID<input id="cp-worker-id" required pattern="[A-Za-z0-9][A-Za-z0-9_.-]*" maxlength="128"></label>
      <label>Copilot model ID<input id="cp-model" placeholder="blank = random"></label>
      <label>Choose Copilot model
        <select id="cp-model-picker"><option value="">Select a Copilot model...</option></select>
      </label>
      <label>Selection model pool<input id="cp-model-pool" placeholder="blank = all discovered models"></label>
      <label>Model selector
        <select id="cp-model-selector">
          <option value="random">random</option>
          <option value="best-1">best-1</option><option value="best-3">best-3</option><option value="best-5">best-5</option>
          <option value="worst-1">worst-1</option><option value="worst-3">worst-3</option><option value="worst-5">worst-5</option>
        </select>
      </label>
      <label>Handled model masks<input id="cp-modelmasks" placeholder="openai/*,gpt-*"></label>
      <label>Role<input id="cp-role" value="headless-copilot"></label>
      <label>Capabilities (prefix ! to decline)<input id="cp-capabilities" placeholder="audio_input,vision_input,!file_input"></label>
      <label>Working directory<input id="cp-cwd" placeholder="runtime-managed workspace"></label>
      <label>Relay WebSocket base<input id="cp-host-ws-url" placeholder="ws://127.0.0.1:8801"></label>
      <label>Copilot executable<input id="cp-command" placeholder="auto-detect copilot.cmd / copilot"></label>
      <label>Session UUID<input id="cp-session-id" placeholder="generated automatically"></label>
      <label>Context
        <select id="cp-context"><option value="default">default</option><option value="long_context">long_context</option></select>
      </label>
      <label>Reasoning effort / selector
        <select id="cp-effort">
          <option value="">model default</option><option>none</option><option>minimal</option><option>low</option>
          <option>medium</option><option>high</option><option>xhigh</option><option>max</option>
        </select>
      </label>
      <label>Max AI credits<input id="cp-credits" type="number" min="0.01" step="0.01" placeholder="unlimited"></label>
      <label>Request timeout (seconds)<input id="cp-timeout" type="number" min="1" max="86400" value="900"></label>
      <label>Reconnect delay (seconds)<input id="cp-reconnect" type="number" min="0.1" max="300" step="0.1" value="2"></label>
      <label>Anti-idle interval override<input id="cp-keepalive-interval" type="number" min="0" max="3600" step="1" placeholder="shared setting"></label>
      <label>Anti-idle timeout override<input id="cp-keepalive-timeout" type="number" min="0.1" max="10" step="0.1" placeholder="shared setting"></label>
      <label>Startup warmup prompt<input id="cp-warmup-prompt" value="Startup warmup: reply only READY."></label>
      <label>Chunk token override<input id="cp-chunk-tokens" type="number" min="1000" max="1000000" placeholder="derive from selected model"></label>
      <label>Maximum chunks<input id="cp-max-chunks" type="number" min="1" max="1000" value="64"></label>
      <label>Max prompt characters<input id="cp-max-prompt" type="number" min="1000" max="20000000" value="4000000"></label>
      <label>Max output characters<input id="cp-max-output" type="number" min="1000" max="2000000" value="200000"></label>
      <label>Max attachment bytes<input id="cp-max-attachment" type="number" min="1024" max="104857600" value="26214400"></label>
    </div>
    <div id="cp-model-capabilities" class="capability-card"></div>
    <label for="copilot-prompt" style="display:block;margin-top:0.7rem;color:#888;font-size:0.8rem">System prompt</label>
    <textarea id="copilot-prompt" spellcheck="false"></textarea>
    <div class="checks">
      <label><input id="cp-autostart" type="checkbox" checked> autostart with EMULLM</label>
      <label><input id="cp-chunk-prompts" type="checkbox" checked> chunk prompts larger than the servant model</label>
      <label><input id="cp-allow-all" type="checkbox" checked> allow all tools, paths, and URLs (unsafe)</label>
      <label><input id="cp-custom-instructions" type="checkbox" checked> load repository instructions</label>
      <label><input id="cp-builtin-mcps" type="checkbox" checked> enable built-in MCPs</label>
      <label><input id="cp-shared-anti-idle" type="checkbox" checked> inherit shared anti-idle configuration</label>
    </div>
    <button type="submit">Save configuration</button>
    <button id="copilot-new" type="button">Create new servant</button>
    <span id="copilot-msg" class="msg"></span>
  </form>
</details>
<div style="margin-top:10px">
  <button id="copilot-add-another" type="button">+ Create another servant</button>
</div>
</section>

<section class="panel" id="model-configurator">
<h2>Models configurator <button id="model-config-refresh" type="button">Pull /v1/models</button></h2>
<p class="sub">Edit any live <code>/v1/models</code> record, hide/export it, change media and
  on-demand flags, or persist an ordered route to servants and real backends.</p>
<div class="model-config-layout">
  <div>
    <label>Filter models<input id="model-config-search" placeholder="copilot/, audio, worker..."></label>
    <label class="sr-only" for="model-config-list">Exported models (multiple selection allowed)</label>
    <select id="model-config-list" size="18" multiple></select>
    <div class="sub">Use Ctrl/Command or Shift to select multiple models for bulk edits.</div>
    <h3>On-demand Copilot slots <span id="model-config-slot-count" class="muted"></span></h3>
    <div id="model-config-slots" class="slot-grid"></div>
  </div>
  <div>
    <h3 id="model-config-title">Select a model</h3>
    <div id="model-config-selection-note" class="sub"></div>
    <div class="checks">
      <label><input id="model-config-export" type="checkbox" checked> Export in /v1/models</label>
      <label><input id="model-config-ondemand" type="checkbox"> on-demand worker</label>
      <label><input id="model-config-simulated" type="checkbox"> simulated</label>
      <label><input id="model-config-image" type="checkbox"> image input</label>
      <label><input id="model-config-audio" type="checkbox"> audio input</label>
      <label><input id="model-config-file" type="checkbox"> general files</label>
      <label><input id="model-config-code" type="checkbox"> code</label>
      <label><input id="model-config-image-output" type="checkbox"> image output</label>
      <label><input id="model-config-summary" type="checkbox"> summarization</label>
    </div>
    <div class="checks">
      <label><input id="model-route-worker-name" type="checkbox"> worker-in-name</label>
      <label><input id="model-route-copilot" type="checkbox"> worker-copilot-*</label>
      <label><input id="model-route-codex" type="checkbox"> worker-codex-*</label>
      <label><input id="model-route-unknown" type="checkbox"> worker-unknown-*</label>
      <label><input id="model-route-backends" type="checkbox"> backend-*</label>
      <span id="model-route-specific-backends"></span>
    </div>
    <div class="sub">Ordered active targets (move left for higher priority, right for lower)</div>
    <div id="model-route-order" class="route-order"></div>
    <label><code>route_targets</code> editor (ordered, one target per line)
      <textarea id="model-config-route" spellcheck="false" placeholder="worker-in-name&#10;worker-copilot-*&#10;worker-codex-*&#10;worker-unknown-*&#10;backend-*"></textarea>
    </label>
    <label>Effective exported model JSON
      <textarea id="model-config-json" spellcheck="false"></textarea>
    </label>
    <div style="margin-top:8px">
      <button id="model-config-save" type="button">Save model + route</button>
      <button id="model-config-load" type="button">Load copilot worker</button>
      <button id="model-config-reset" type="button">Reset model override</button>
      <button id="model-config-clear-route" type="button">Clear route</button>
      <span id="model-config-msg" class="msg"></span>
    </div>
  </div>
</div>
</section>

<section class="panel" id="test-client">
<h2>Model test client <button id="refresh-api-models" type="button">Refresh API models</button>
  <button id="model-test-configure" type="button">Configure selected model</button></h2>
<p class="sub">Send a chat-completions request through this EMULLM server. Pick an advertised
  model or type any model ID.</p>
<form id="model-test-form">
  <div class="form-grid">
    <label>API model ID<input id="model-test-model" required placeholder="Type any model/id"></label>
    <label>Choose advertised model <span id="api-model-count"></span>
      <select id="model-test-model-picker"><option value="">Select a model...</option></select>
    </label>
    <label>Required capabilities<input id="model-test-capabilities" placeholder="code,summarization,audio_input"></label>
  </div>
  <div id="api-model-capabilities" class="capability-card"></div>
  <label for="model-test-prompt" style="display:block;margin-top:0.7rem;color:#888;font-size:0.8rem">User prompt</label>
  <textarea id="model-test-prompt" required>Say hello in five words.</textarea>
  <div class="attachment-drop" id="model-test-drop">
    <label for="model-test-files"><b>Attach images, audio, video, documents, or any other files</b></label><br>
    <input id="model-test-files" type="file" multiple>
    <div class="sub">Choose files or drag and drop here. Up to 12 files, 25 MiB each, 50 MiB total.</div>
    <div id="model-test-samples" style="margin-top:8px"></div>
  </div>
  <div id="model-test-attachment-summary" class="sub"></div>
  <div id="model-test-attachments" class="attachment-grid"></div>
  <button id="model-test-clear-attachments" type="button">Clear attachments</button>
  <div style="margin-top:0.5rem">
    <button id="model-test-send" type="submit">Send request</button>
    <button id="model-test-cancel" type="button" disabled>Cancel</button>
    <button id="model-test-clear" type="button">Clear result</button>
    <span id="model-test-status" class="msg"></span>
  </div>
</form>
<h3>Assistant response</h3>
<pre id="model-test-answer" class="muted">No request sent.</pre>
<div id="model-test-uploaded" class="attachment-grid attachment-result"></div>
<details>
  <summary>Raw response</summary>
  <pre id="model-test-raw"></pre>
</details>
<h3>Image generation test</h3>
<p class="sub">Uses the real <code>POST /v1/images/generations</code> surface. The default
  Copilot model generates a workspace PNG with tools; the result explicitly says whether
  it came from a worker or the simulated placeholder.</p>
<div class="form-grid">
  <label>Image model<input id="image-generation-model" value="copilot/gpt-5.3-codex"></label>
  <label>Image prompt<input id="image-generation-prompt" value="A bright red circle centered on a clean white background"></label>
</div>
<button id="image-generation-run" type="button">Generate test image</button>
<span id="image-generation-status" class="msg"></span>
<img id="image-generation-preview" alt="Generated image test result">
<pre id="image-generation-description" class="muted"></pre>
<details><summary>Raw image response</summary><pre id="image-generation-raw"></pre></details>
</section>

<section class="panel" id="config-sections">
<h2>Configuration sections</h2>
<p class="sub">Edit one validated section without replacing unrelated settings. Startup-owned changes report that an EMULLM restart is required.</p>
<div class="section-editor">
  <div class="section-tabs" id="config-section-tabs"></div>
  <div>
    <h3 id="config-section-title">Select a section</h3>
    <p class="sub" id="config-section-help"></p>
    <label class="sr-only" for="config-section-editor">Selected configuration section JSON</label>
    <textarea id="config-section-editor" spellcheck="false"></textarea>
    <div style="margin-top:8px">
      <button id="config-section-save" type="button">Save section</button>
      <button id="config-section-reload" type="button">Reload section</button>
      <button id="config-section-delete" type="button">Delete section</button>
      <span id="config-section-msg" class="msg"></span>
    </div>
  </div>
</div>
</section>

<section class="panel" id="configuration">
<h2>config.json <span id="cfg-path" class="muted"></span></h2>
<label class="sr-only" for="config">Raw EMULLM JSON configuration</label>
<textarea id="config" spellcheck="false"></textarea>
<div style="margin-top:0.5rem">
  <button id="reload">Reload</button>
  <button id="save">Save</button>
  <span id="cfg-msg" class="msg"></span>
</div>
</section>
</div>
</main>
</div>

<script>
// Resolve REST calls relative to wherever this page is served, so it works
// under either admin prefix (/emullm/admin or /admin/emullm) -- and would
// survive being mounted under a sub-path. Both alias trees exist server-side.
const PAGE_PATH = location.pathname.replace(/\\/+$/, '');
const ADMIN = PAGE_PATH === '/emullm' ? '/emullm/admin' : (PAGE_PATH || '/emullm/admin');
const POLL_VISIBLE_MS = 3000;
const POLL_HIDDEN_MS = 120000;
const POLL_WINDOW_KEY = 'emullm.poll.window';
const POLL_HIDDEN_KEY = 'emullm.poll.hidden';
let pollTimer = null;
let pollWindowStartedAt = Date.now();
let pollInFlight = false;
async function getJSON(u, opts) { const r = await fetch(u, opts); return { ok: r.ok, status: r.status, body: await r.json().catch(() => ({})) }; }
function esc(x) { return String(x).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function formatDuration(value) {
  let seconds = Math.max(0, Math.floor(Number(value) || 0));
  const days = Math.floor(seconds / 86400); seconds %= 86400;
  const hours = Math.floor(seconds / 3600); seconds %= 3600;
  const minutes = Math.floor(seconds / 60); seconds %= 60;
  return [days ? days + 'd' : '', hours ? hours + 'h' : '', minutes ? minutes + 'm' : '', seconds + 's'].filter(Boolean).join(' ');
}
function formatActivityTime(timestamp, elapsedSeconds, kind) {
  if (!timestamp) return '<span class="muted">never</span>';
  const date = new Date(timestamp);
  const clock = Number.isNaN(date.getTime()) ? String(timestamp) : date.toLocaleTimeString();
  return (kind ? '<b>' + esc(kind) + '</b><br>' : '') + esc(clock) +
    '<br><span class="muted">' + formatDuration(elapsedSeconds) + ' ago</span>';
}
function formatTokens(value) {
  const tokens = Number(value) || 0;
  if (!tokens) return 'unknown';
  return tokens >= 1000000 ? (tokens / 1000000).toFixed(2).replace(/0+$/, '').replace(/\\.$/, '') + 'M' :
    (tokens >= 1000 ? Math.round(tokens / 1000) + 'K' : String(tokens));
}
function capabilityCard(model, options = {}) {
  if (!model) return '<span class="muted">No capability metadata for this model.</span>';
  const capabilities = model.capabilities || {};
  const limits = capabilities.limits || {};
  const vision = limits.vision || {};
  const mediaTypes = Array.isArray(vision.supported_media_types)
    ? vision.supported_media_types : [];
  const audioTypes = mediaTypes.filter(value => value.startsWith('audio/'));
  const inputModalities = model.input_modalities || {};
  const audioModality = inputModalities.audio || {};
  const fileModality = inputModalities.general_file || {};
  const taskCapabilities = model.task_capabilities || {};
  const taskBadges = Object.entries(taskCapabilities)
    .filter(([, value]) => value && value.enabled === true)
    .map(([name]) => 'task ' + name.replaceAll('_', ' '));
  const supports = capabilities.supports || {};
  const reasoning = Array.isArray(supports.reasoning_effort) ? supports.reasoning_effort : [];
  const supported = Object.entries(supports)
    .filter(([key, value]) => value === true && !['reasoningEffort'].includes(key))
    .map(([key]) => key.replaceAll('_', ' '));
  const badges = [
    model.quality_rank ? ('quality #' + model.quality_rank) : null,
    model.quality_tier,
    ...supported,
    ...taskBadges,
    ...reasoning.map(level => 'effort ' + level),
  ].filter(Boolean);
  const prices = model.billing && model.billing.tokenPrices ? model.billing.tokenPrices : {};
  const route = options.routeTargets && options.routeTargets.length
    ? '<div><strong>Route:</strong> ' + esc(options.routeTargets.join(' → ')) + '</div>' : '';
  const active = options.activeWorkers && options.activeWorkers.length
    ? '<div><strong>Active workers:</strong> ' + esc(options.activeWorkers.join(', ')) + '</div>' : '';
  const backing = options.backingModel
    ? '<div><strong>Backing model:</strong> ' + esc(options.backingModel) + '</div>' : '';
  return '<div><strong>' + esc(model.name || model.display_name || model.id) + '</strong> <code>' + esc(model.id) + '</code></div>' +
    '<div class="capability-badges">' + badges.map(value => '<span class="capability-badge">' + esc(value) + '</span>').join('') + '</div>' +
    '<div><strong>Context:</strong> ' + formatTokens(limits.max_context_window_tokens || model.context_length) +
    ' · <strong>prompt:</strong> ' + formatTokens(limits.max_prompt_tokens || model.context_length) +
    ' · <strong>output:</strong> ' + formatTokens(limits.max_output_tokens) + '</div>' +
    (limits.vision ? '<div><strong>Vision:</strong> ' + (vision.max_prompt_images || '?') +
      ' image(s), max ' + formatBytes(vision.max_prompt_image_size || 0) + ' each</div>' : '') +
    (mediaTypes.length ? '<div><strong>Advertised media:</strong> ' + esc(mediaTypes.join(', ')) + '</div>' : '') +
    '<div><strong>Native audio:</strong> ' +
      esc(audioModality.status || (audioTypes.length ? audioTypes.join(', ') : 'not advertised')) +
      (audioModality.source ? ' · ' + esc(audioModality.source) : '') + '</div>' +
    '<div><strong>General files:</strong> ' +
      esc(fileModality.status || 'transport supported') +
      '; comprehension ' + esc(fileModality.model_comprehension || 'model-dependent') + '.</div>' +
    (prices.inputPrice != null ? '<div><strong>Credits / 1M:</strong> input ' + prices.inputPrice +
      ' · cached ' + (prices.cacheReadPrice ?? prices.cachePrice ?? '?') + ' · output ' + (prices.outputPrice ?? '?') + '</div>' : '') +
    route + active + backing +
    (options.description ? '<div>' + esc(options.description) + '</div>' : '');
}
function renderCopilotCapabilities() {
  const id = field('cp-model').value.trim() || field('cp-model-picker').value;
  const model = copilotModelCatalog.find(item => item.id === id);
  field('cp-model-capabilities').innerHTML = id ? capabilityCard(model) :
    '<span class="muted">Model is selected at servant startup by the configured strategy and pool.</span>';
}
function renderApiModelCapabilities() {
  const id = field('model-test-model').value.trim();
  const entry = apiModelCatalog.find(item => item.id === id);
  if (!entry) {
    field('api-model-capabilities').innerHTML = id
      ? '<strong>' + esc(id) + '</strong><div class="muted">Free-form model ID; EMULLM will apply generic routing.</div>'
      : '';
    return;
  }
  const backingIds = [
    entry.backing_model,
    ...Object.values(entry.backing_models || {}),
  ].filter(Boolean);
  const backing = copilotModelCatalog.find(item => backingIds.includes(item.id));
  field('api-model-capabilities').innerHTML = capabilityCard(
    backing || entry,
    {
      backingModel: backing ? backing.id : entry.backing_model,
      routeTargets: entry.route_targets || [],
      activeWorkers: entry.active_workers || (entry.worker_id ? [entry.worker_id] : []),
      description: entry.description || '',
    },
  );
}
const DEFAULT_COPILOT_PROMPT = 'Act as the model requested by the OpenAI-compatible caller. Answer the request directly and return only the assistant response that should be sent to the caller.';
let copilotInstances = [];
let copilotModelCatalog = [];
let apiModelCatalog = [];
let modelConfigCatalog = [];
let modelConfigMeta = { revision: '', overrides: {}, routes: {}, backends: [], on_demand: { slots: [], limit: 46, max_concurrent_calls: 50 } };
let selectedModelConfigId = '';
let selectedModelConfigIds = [];
let latestTelemetryState = {};
let telemetryWorkerSort = { key: 'served', direction: -1 };
let telemetryModelSort = { key: 'attempts', direction: -1 };
let telemetryShowAllWorkers = false;
let telemetryShowAllModels = false;
let editingCopilot = null;
let suggestedCopilotId = 'worker-copilot-1';
let modelTestController = null;
let modelTestTimer = null;
let modelTestStartedAt = 0;
let modelTestRequestId = null;
let modelTestAttachments = [];
let serverPid = null;
let currentConfig = {};
let currentConfigRevision = '';
let activeConfigSection = 'services';
let configuredBackends = [];
let editingBackend = null;
let codexSuppliers = [];
let editingSupplier = null;
let antiIdleConfig = null;
let antiIdlePrompts = [];
let antiIdleRevision = '';
let selectedAntiIdlePromptId = '';
let antiIdleSort = { key: 'number', direction: 1 };
const CONFIG_SECTIONS = {
  services: ['Service catalog and per-service fallback policies.', 'object'],
  agents: ['Unified recruit, subagent, mock, and proxy agent definitions.', 'array'],
  model_routes: ['Model IDs mapped to one worker or an ordered worker-glob/backend chain.', 'object'],
  workers: ['Legacy managed worker definitions used by auto mode.', 'array'],
  mock_workers: ['Legacy in-process mock worker definitions.', 'array'],
  backends: ['Legacy OpenAI-compatible proxy backend definitions.', 'array'],
  codex_suppliers: ['Codex worker/model supplier definitions.', 'array'],
  anti_idle: ['Shared anti-idle frequency, deadline, and conversational prompt catalog.', 'object'],
  mock: ['Global mock reply and template used by mock mode.', 'object'],
};

async function waitForServerRestart(previousPid) {
  const deadline = Date.now() + 120000;
  while (Date.now() < deadline) {
    await new Promise(resolve => setTimeout(resolve, 500));
    try {
      const state = await getJSON(ADMIN + '/state', { cache: 'no-store' });
      if (state.ok && state.body.process && state.body.process.pid !== previousPid) {
        location.reload();
        return;
      }
    } catch (_) {
      // Expected while the old process releases the port.
    }
  }
  field('updated').textContent = 'restart timed out';
}
field('server-restart').addEventListener('click', async () => {
  if (!confirm('Gracefully restart EMULLM and all resident servants?')) return;
  const previousPid = serverPid;
  field('server-restart').disabled = true;
  field('updated').textContent = 'restarting...';
  const response = await getJSON(ADMIN + '/restart', { method: 'POST' });
  if (!response.ok) {
    field('updated').textContent = response.body.detail || 'restart failed';
    field('server-restart').disabled = false;
    return;
  }
  void waitForServerRestart(previousPid);
});
field('server-shutdown').addEventListener('click', async () => {
  if (!confirm('Gracefully shut down EMULLM and all resident servants?')) return;
  field('server-shutdown').disabled = true;
  field('updated').textContent = 'shutting down...';
  const response = await getJSON(ADMIN + '/shutdown', { method: 'POST' });
  if (!response.ok) {
    field('updated').textContent = response.body.detail || 'shutdown failed';
    field('server-shutdown').disabled = false;
  }
});

async function refreshConfiguredAgents() {
  const r = await getJSON(ADMIN + '/agents', { cache: 'no-store' });
  const carol = ((r.body && r.body.agents) || []).find(agent => agent.id === 'carol');
  field('carol-enabled').disabled = !carol;
  field('carol-enabled').checked = !!(carol && carol.enabled);
  field('carol-note').textContent = !carol ? 'Carol is not present in config.json' :
    (carol.mock_registered ? 'enabled and registered' : (carol.enabled ? 'enabled; waiting' : 'disabled'));
}
field('carol-enabled').addEventListener('change', async () => {
  const enabled = field('carol-enabled').checked;
  field('carol-enabled').disabled = true;
  const r = await getJSON(ADMIN + '/agents/carol/enabled', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled }),
  });
  field('carol-note').textContent = r.ok ? (enabled ? 'enabled and registered' : 'disabled') :
    ('update failed (' + r.status + ')');
  await Promise.all([refreshConfiguredAgents(), refreshWorkers(), refreshCopilots(), loadConfig()]);
});

function setBackendMsg(text, cls) {
  field('backend-msg').textContent = text;
  field('backend-msg').className = 'msg ' + (cls || '');
}
function resetBackendEditor() {
  editingBackend = null;
  field('backend-editor').open = true;
  field('backend-editor-title').textContent = 'Add backend';
  field('backend-form').reset();
  field('backend-default').checked = false;
  field('backend-clear-api-key').checked = false;
  setBackendMsg('', '');
}
function editBackend(record) {
  editingBackend = {
    source: record.source,
    index: record.index,
    name: record.name,
    revision: record.revision,
  };
  field('backend-editor').open = true;
  field('backend-editor-title').textContent = 'Edit ' + (record.name || record.record_id);
  field('backend-name').value = record.name || '';
  field('backend-base-url').value = record.base_url || '';
  field('backend-model').value = record.model || '';
  field('backend-api-key-env').value = record.api_key_env || '';
  field('backend-api-key').value = '';
  field('backend-validation').value = record.validation_interval ?? '';
  field('backend-description').value = record.description || '';
  field('backend-default').checked = !!record.default;
  field('backend-clear-api-key').checked = false;
  setBackendMsg(
    record.has_api_key ? 'A stored inline API key is present and remains unchanged.' : '',
    '',
  );
}
async function refreshBackendConfigs() {
  const response = await getJSON(ADMIN + '/backends/configured', { cache: 'no-store' });
  configuredBackends = (response.body && response.body.backends) || [];
  field('backend-note').textContent = configuredBackends.length + ' configured';
  field('nav-backend-count').textContent = configuredBackends.length;
  field('backend-config-rows').innerHTML = configuredBackends.length
    ? configuredBackends.map(record =>
      '<tr><td><b>' + esc(record.name || record.record_id) + '</b><br><span class="muted">' +
      esc(record.description || '') + '</span></td><td><code>' + esc(record.source) +
      ':' + record.index + '</code></td><td><code>' + esc(record.base_url || '--') +
      '</code></td><td>' + esc(record.model || '--') + '</td><td>' +
      (record.has_api_key ? 'stored key' : (record.api_key_env ? ('env ' + esc(record.api_key_env)) : 'none')) +
      '</td><td>' + (record.default ? 'yes' : 'no') + '</td><td>' +
      '<button data-backend-action="edit" data-source="' + esc(record.source) +
      '" data-index="' + record.index + '">Edit</button>' +
      '<button data-backend-action="delete" class="danger" data-source="' +
      esc(record.source) + '" data-index="' + record.index + '">Delete</button></td></tr>'
    ).join('')
    : '<tr><td colspan="7" class="muted">none configured</td></tr>';
}
field('backend-new').addEventListener('click', resetBackendEditor);
field('backend-cancel').addEventListener('click', () => {
  field('backend-editor').open = false;
  editingBackend = null;
});
field('backend-refresh').addEventListener('click', refreshBackendConfigs);
field('backend-config-rows').addEventListener('click', async event => {
  const button = event.target.closest('button[data-backend-action]');
  if (!button) return;
  const source = button.dataset.source;
  const index = Number(button.dataset.index);
  const record = configuredBackends.find(
    item => item.source === source && item.index === index
  );
  if (!record) return;
  if (button.dataset.backendAction === 'edit') {
    editBackend(record);
    return;
  }
  if (!confirm('Delete backend ' + (record.name || record.record_id) + '?')) return;
  const deletePath = ADMIN + '/backends/configured/' + encodeURIComponent(source) +
    '/' + index + (record.name
      ? '?expected_name=' + encodeURIComponent(record.name) +
        '&expected_revision=' + encodeURIComponent(record.revision)
      : '');
  const response = await getJSON(deletePath, { method: 'DELETE' });
  setBackendMsg(
    response.ok ? 'backend deleted' : JSON.stringify(response.body.detail || response.status),
    response.ok ? 'ok' : 'err',
  );
  if (response.ok) await Promise.all([refreshBackendConfigs(), loadConfig()]);
});
field('backend-form').addEventListener('submit', async event => {
  event.preventDefault();
  const apiKey = field('backend-api-key').value;
  const body = {
    name: field('backend-name').value.trim(),
    base_url: field('backend-base-url').value.trim(),
    description: field('backend-description').value.trim() || null,
    api_key_env: field('backend-api-key-env').value.trim() || null,
    model: field('backend-model').value.trim() || null,
    default: field('backend-default').checked,
    validation_interval: field('backend-validation').value.trim() || null,
    clear_api_key: field('backend-clear-api-key').checked,
  };
  if (editingBackend) {
    body.expected_name = editingBackend.name;
    body.expected_revision = editingBackend.revision;
  }
  if (apiKey) body.api_key = apiKey;
  const path = editingBackend
    ? ADMIN + '/backends/configured/' + encodeURIComponent(editingBackend.source) +
      '/' + editingBackend.index
    : ADMIN + '/backends/configured';
  const response = await getJSON(path, {
    method: editingBackend ? 'PUT' : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  setBackendMsg(
    response.ok ? 'saved; restart to refresh advertised backend catalogs' :
      JSON.stringify(response.body.detail || response.status),
    response.ok ? 'ok' : 'err',
  );
  if (response.ok) {
    editingBackend = null;
    field('backend-editor').open = false;
    await Promise.all([refreshBackendConfigs(), loadConfig(), refreshModelConfigurator()]);
  }
});

function setSupplierMsg(text, cls) {
  field('supplier-msg').textContent = text;
  field('supplier-msg').className = 'msg ' + (cls || '');
}
function resetSupplierEditor() {
  editingSupplier = null;
  field('supplier-editor').open = true;
  field('supplier-editor-title').textContent = 'Add Codex supplier';
  field('supplier-form').reset();
  field('supplier-kind').value = 'custom';
  field('supplier-priority').value = '0';
  field('supplier-enabled').checked = true;
  field('supplier-id').readOnly = false;
  setSupplierMsg('', '');
}
function editSupplier(supplier) {
  editingSupplier = { id: supplier.id, revision: supplier.revision };
  field('supplier-editor').open = true;
  field('supplier-editor-title').textContent = 'Edit ' + supplier.name;
  field('supplier-id').value = supplier.id;
  field('supplier-id').readOnly = true;
  field('supplier-name').value = supplier.name || '';
  field('supplier-kind').value = supplier.kind || 'custom';
  field('supplier-priority').value = supplier.priority ?? 0;
  field('supplier-worker-pattern').value = supplier.worker_pattern || '';
  field('supplier-model-prefix').value = supplier.model_prefix || '';
  field('supplier-model-patterns').value = (supplier.model_patterns || []).join(',');
  field('supplier-command').value = supplier.command || '';
  field('supplier-base-url').value = supplier.base_url || '';
  field('supplier-api-key-env').value = supplier.api_key_env || '';
  field('supplier-description').value = supplier.description || '';
  field('supplier-enabled').checked = supplier.enabled !== false;
  setSupplierMsg('', '');
}
async function refreshCodexSuppliers() {
  const response = await getJSON(ADMIN + '/codex-suppliers', { cache: 'no-store' });
  codexSuppliers = (response.body && response.body.suppliers) || [];
  field('supplier-note').textContent = codexSuppliers.length + ' configured';
  field('nav-supplier-count').textContent = codexSuppliers.length;
  field('supplier-rows').innerHTML = codexSuppliers.length
    ? codexSuppliers.map(supplier =>
      '<tr><td><b>' + esc(supplier.name) + '</b><br><code>' + esc(supplier.id) +
      '</code></td><td>' + esc(supplier.kind) + '</td><td><code>' +
      esc(supplier.worker_pattern || '--') + '</code></td><td>' +
      esc((supplier.model_patterns || []).join(', ') || '--') + '</td><td>' +
      (supplier.enabled ? '<span class="dot on"></span>enabled' :
        '<span class="dot off"></span>disabled') + '</td><td>' +
      '<button data-supplier-action="edit" data-id="' + esc(supplier.id) +
      '">Edit</button><button data-supplier-action="delete" class="danger" data-id="' +
      esc(supplier.id) + '">Delete</button></td></tr>'
    ).join('')
    : '<tr><td colspan="6" class="muted">none configured</td></tr>';
}
field('supplier-new').addEventListener('click', resetSupplierEditor);
field('supplier-cancel').addEventListener('click', () => {
  field('supplier-editor').open = false;
  editingSupplier = null;
});
field('supplier-refresh').addEventListener('click', refreshCodexSuppliers);
field('supplier-rows').addEventListener('click', async event => {
  const button = event.target.closest('button[data-supplier-action]');
  if (!button) return;
  const supplier = codexSuppliers.find(item => item.id === button.dataset.id);
  if (!supplier) return;
  if (button.dataset.supplierAction === 'edit') {
    editSupplier(supplier);
    return;
  }
  if (!confirm('Delete Codex supplier ' + supplier.name + '?')) return;
  const response = await getJSON(
    ADMIN + '/codex-suppliers/' + encodeURIComponent(supplier.id) +
      '?expected_revision=' + encodeURIComponent(supplier.revision),
    { method: 'DELETE' },
  );
  setSupplierMsg(
    response.ok ? 'supplier deleted' : JSON.stringify(response.body.detail || response.status),
    response.ok ? 'ok' : 'err',
  );
  if (response.ok) await Promise.all([refreshCodexSuppliers(), loadConfig()]);
});
field('supplier-form').addEventListener('submit', async event => {
  event.preventDefault();
  const body = {
    id: field('supplier-id').value.trim(),
    name: field('supplier-name').value.trim(),
    kind: field('supplier-kind').value,
    enabled: field('supplier-enabled').checked,
    priority: Number(field('supplier-priority').value) || 0,
    description: field('supplier-description').value.trim() || null,
    worker_pattern: field('supplier-worker-pattern').value.trim() || null,
    model_prefix: field('supplier-model-prefix').value.trim() || null,
    model_patterns: csv(field('supplier-model-patterns').value),
    command: field('supplier-command').value.trim() || null,
    base_url: field('supplier-base-url').value.trim() || null,
    api_key_env: field('supplier-api-key-env').value.trim() || null,
  };
  const response = await getJSON(
    editingSupplier
      ? ADMIN + '/codex-suppliers/' + encodeURIComponent(editingSupplier.id) +
        '?expected_revision=' + encodeURIComponent(editingSupplier.revision)
      : ADMIN + '/codex-suppliers',
    {
      method: editingSupplier ? 'PUT' : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    },
  );
  setSupplierMsg(
    response.ok ? 'supplier saved' : JSON.stringify(response.body.detail || response.status),
    response.ok ? 'ok' : 'err',
  );
  if (response.ok) {
    editingSupplier = null;
    field('supplier-editor').open = false;
    await Promise.all([
      refreshCodexSuppliers(),
      loadConfig(),
      refreshApiModels(),
      refreshModelConfigurator(),
    ]);
  }
});

function antiIdlePrompt() {
  return antiIdlePrompts.find(prompt => prompt.id === selectedAntiIdlePromptId) || null;
}
function antiIdleDuration(value) {
  return value == null ? '—' : (Number(value) / 1000).toFixed(2) + 's';
}
function renderAntiIdleEditor() {
  const prompt = antiIdlePrompt();
  field('anti-idle-id').readOnly = true;
  field('anti-idle-id').value = prompt ? prompt.id : '';
  field('anti-idle-text').value = prompt ? prompt.prompt : '';
  field('anti-idle-deprecated').checked = !!(prompt && prompt.deprecated);
  field('anti-idle-text').disabled = !prompt;
  field('anti-idle-deprecated').disabled = !prompt;
  field('anti-idle-stats').innerHTML = prompt
    ? '<strong>Item #' + prompt.number + '</strong><br>average ' +
      antiIdleDuration(prompt.average_duration_ms) + ' · shortest ' +
      antiIdleDuration(prompt.min_duration_ms) + ' by ' +
      esc(prompt.shortest_worker_id || '—') + ' · longest ' +
      antiIdleDuration(prompt.max_duration_ms) + ' by ' +
      esc(prompt.longest_worker_id || '—') + '<br>attempts ' +
      (prompt.attempts || 0) + ' · completed ' + (prompt.completed || 0) +
      ' · timeouts ' + (prompt.timeouts || 0) + ' · over budget ' +
      (prompt.slow || 0) + ' · adaptively retired by ' +
      (prompt.retired_workers || 0) + ' worker(s)'
    : '<span class="muted">Select a prompt.</span>';
}
function renderAntiIdleList() {
  antiIdlePrompts.forEach((prompt, index) => { prompt.number = index + 1; });
  if (!antiIdlePrompts.some(prompt => prompt.id === selectedAntiIdlePromptId)) {
    selectedAntiIdlePromptId = antiIdlePrompts[0] ? antiIdlePrompts[0].id : '';
  }
  const visible = [...antiIdlePrompts].sort((left, right) => {
    let a = left[antiIdleSort.key];
    let b = right[antiIdleSort.key];
    if (antiIdleSort.key === 'prompt') {
      a = String(a || '').toLowerCase();
      b = String(b || '').toLowerCase();
    } else if (antiIdleSort.key === 'deprecated') {
      a = a ? 1 : 0;
      b = b ? 1 : 0;
    } else {
      a = a == null ? Number.POSITIVE_INFINITY : Number(a);
      b = b == null ? Number.POSITIVE_INFINITY : Number(b);
    }
    return (a < b ? -1 : (a > b ? 1 : 0)) * antiIdleSort.direction;
  });
  field('anti-idle-list').innerHTML = visible.length ? visible.map(prompt =>
    '<tr data-anti-id="' + prompt.id + '" class="' +
    (prompt.id === selectedAntiIdlePromptId ? 'selected-row' : '') + '">' +
    '<td>' + prompt.number + '</td><td>' +
    antiIdleDuration(prompt.average_duration_ms) + '</td><td>' +
    antiIdleDuration(prompt.min_duration_ms) + ' · <code>' +
    esc(prompt.shortest_worker_id || '—') + '</code></td><td>' +
    antiIdleDuration(prompt.max_duration_ms) + ' · <code>' +
    esc(prompt.longest_worker_id || '—') + '</code></td><td>' +
    (prompt.attempts || 0) + '</td><td>' + (prompt.timeouts || 0) +
    '</td><td>' + (prompt.slow || 0) + '</td><td>' +
    (prompt.retired_workers || 0) + '</td><td><input type="checkbox" ' +
    'data-anti-deprecated="' + prompt.id + '"' +
    (prompt.deprecated ? ' checked' : '') + '></td><td>' +
    esc(prompt.prompt) + '</td></tr>'
  ).join('') : '<tr><td colspan="10" class="muted">no prompts</td></tr>';
  field('nav-anti-idle-count').textContent = antiIdlePrompts.length;
  renderAntiIdleEditor();
}
async function refreshAntiIdle() {
  const response = await getJSON(ADMIN + '/anti-idle', { cache: 'no-store' });
  if (!response.ok) {
    field('anti-idle-note').textContent = 'load failed';
    return;
  }
  antiIdleConfig = response.body.config;
  antiIdlePrompts = response.body.prompts || [];
  antiIdleRevision = response.body.revision || '';
  field('anti-idle-enabled').checked = antiIdleConfig.enabled !== false;
  field('anti-idle-interval').value = antiIdleConfig.interval_seconds ?? 60;
  field('anti-idle-timeout').value = antiIdleConfig.timeout_seconds ?? 10;
  field('anti-idle-slow-budget').value = antiIdleConfig.slow_budget_seconds ?? 8;
  field('anti-idle-note').textContent = antiIdlePrompts.length + ' prompt(s) · ' +
    (response.body.worker_count || 0) + ' worker status file(s)';
  renderAntiIdleList();
}
field('anti-idle-refresh').addEventListener('click', refreshAntiIdle);
field('anti-idle-enabled').addEventListener('change', async event => {
  const checkbox = event.target;
  const enabled = checkbox.checked;
  checkbox.disabled = true;
  field('anti-idle-msg').textContent = enabled ? 'enabling immediately...' : 'disabling immediately...';
  const response = await getJSON(ADMIN + '/anti-idle/enabled', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      enabled,
      expected_revision: antiIdleRevision,
    }),
  });
  checkbox.disabled = false;
  if (!response.ok) {
    checkbox.checked = !enabled;
    field('anti-idle-msg').textContent = JSON.stringify(response.body.detail || response.status);
    field('anti-idle-msg').className = 'msg err';
    return;
  }
  antiIdleRevision = response.body.revision;
  antiIdleConfig.enabled = enabled;
  field('anti-idle-msg').textContent =
    (enabled ? 'enabled' : 'disabled') + ' immediately on ' +
    response.body.updated_workers + ' connected worker(s)' +
    (response.body.failed_workers ? ' · ' + response.body.failed_workers + ' failed' : '');
  field('anti-idle-msg').className =
    'msg ' + (response.body.failed_workers ? 'err' : 'ok');
});
field('anti-idle-table').addEventListener('click', event => {
  const sort = event.target.closest('button[data-anti-sort]');
  if (sort) {
    const key = sort.dataset.antiSort;
    antiIdleSort = {
      key,
      direction: antiIdleSort.key === key ? -antiIdleSort.direction : 1,
    };
    renderAntiIdleList();
    return;
  }
  if (event.target.matches('input[data-anti-deprecated]')) return;
  const row = event.target.closest('tr[data-anti-id]');
  if (!row) return;
  selectedAntiIdlePromptId = row.dataset.antiId;
  renderAntiIdleEditor();
  renderAntiIdleList();
});
field('anti-idle-list').addEventListener('change', event => {
  const checkbox = event.target.closest('input[data-anti-deprecated]');
  if (!checkbox) return;
  const prompt = antiIdlePrompts.find(item => item.id === checkbox.dataset.antiDeprecated);
  if (!prompt) return;
  prompt.deprecated = checkbox.checked;
  selectedAntiIdlePromptId = prompt.id;
  renderAntiIdleList();
});
field('anti-idle-text').addEventListener('input', event => {
  const prompt = antiIdlePrompt();
  if (!prompt) return;
  prompt.prompt = event.target.value;
});
field('anti-idle-text').addEventListener('blur', renderAntiIdleList);
field('anti-idle-deprecated').addEventListener('change', event => {
  const prompt = antiIdlePrompt();
  if (!prompt) return;
  prompt.deprecated = event.target.checked;
  renderAntiIdleList();
});
field('anti-idle-add').addEventListener('click', () => {
  let number = antiIdlePrompts.length + 1;
  let id = 'conversation-' + String(number).padStart(2, '0');
  const used = new Set(antiIdlePrompts.map(prompt => prompt.id));
  while (used.has(id)) {
    number += 1;
    id = 'conversation-' + String(number).padStart(2, '0');
  }
  antiIdlePrompts.push({
    id,
    prompt: 'Ask the worker an interesting short question.',
    deprecated: false,
    number: antiIdlePrompts.length + 1,
    attempts: 0,
    completed: 0,
    slow: 0,
    timeouts: 0,
    total_duration_ms: 0,
    average_duration_ms: 0,
    min_duration_ms: null,
    max_duration_ms: 0,
    shortest_worker_id: null,
    longest_worker_id: null,
    retired_workers: 0,
  });
  selectedAntiIdlePromptId = id;
  renderAntiIdleList();
  field('anti-idle-text').focus();
});
field('anti-idle-save').addEventListener('click', async () => {
  const prompts = antiIdlePrompts.map(prompt => ({
    id: prompt.id,
    prompt: prompt.prompt.trim(),
    deprecated: !!prompt.deprecated,
  }));
  const response = await getJSON(ADMIN + '/anti-idle', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      expected_revision: antiIdleRevision,
      config: {
        enabled: field('anti-idle-enabled').checked,
        interval_seconds: Number(field('anti-idle-interval').value),
        timeout_seconds: Number(field('anti-idle-timeout').value),
        slow_budget_seconds: Number(field('anti-idle-slow-budget').value),
        prompts,
      },
    }),
  });
  field('anti-idle-msg').textContent = response.ok
    ? 'saved; restart workers to apply'
    : JSON.stringify(response.body.detail || response.status);
  field('anti-idle-msg').className = 'msg ' + (response.ok ? 'ok' : 'err');
  if (response.ok) {
    antiIdleRevision = response.body.revision;
    await Promise.all([refreshAntiIdle(), loadConfig()]);
  }
});
field('anti-idle-reset-stats').addEventListener('click', async () => {
  if (!confirm('Reset anti-idle timing and adaptive retirement for every worker?')) return;
  const button = field('anti-idle-reset-stats');
  button.disabled = true;
  const response = await getJSON(ADMIN + '/anti-idle/reset-stats', {
    method: 'POST',
  });
  field('anti-idle-msg').textContent = response.ok
    ? ('reset ' + (response.body.reset || 0) + ' worker(s)')
    : JSON.stringify(response.body.detail || response.status);
  field('anti-idle-msg').className = 'msg ' + (response.ok ? 'ok' : 'err');
  button.disabled = false;
  if (response.ok) await refreshAntiIdle();
});

async function refreshWebsockets() {
  const r = await getJSON(ADMIN + '/websockets', { cache: 'no-store' });
  const connections = (r.body && r.body.connections) || [];
  const logs = (r.body && r.body.logs) || {};
  field('websocket-note').textContent = connections.length + ' active' +
    (logs.directory
      ? (' · logs ' + logs.directory + ' · first 2 MiB + newest 4 MiB per worker')
      : '');
  field('nav-socket-count').textContent = connections.length;
  field('top-socket-count').textContent = connections.length;
  const tbody = field('websocket-connections');
  if (!connections.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="muted">none</td></tr>';
    return;
  }
  tbody.innerHTML = connections.map(connection => {
    const identity = connection.worker_id || ((connection.subscriptions || []).join(', ')) ||
      (connection.filters ? JSON.stringify(connection.filters) : '--');
    return '<tr><td><code>' + esc(connection.connection_id) + '</code></td>' +
      '<td><b>' + esc(connection.kind) + '</b><br><code>' + esc(connection.endpoint) + '</code></td>' +
      '<td>' + esc(identity) + '</td><td>' + esc(connection.client || '--') + '</td>' +
      '<td>in ' + (connection.messages_in || 0) + ' / out ' + (connection.messages_out || 0) +
      (connection.log_url
        ? ('<br><a href="' + esc(connection.log_url) + '/view" target="_blank">Viewer</a>' +
          ' · <a href="' + esc(connection.log_url) + '" target="_blank">Raw JSONL</a> · ' +
          formatBytes(connection.log_bytes || 0) + ' / 6 MiB')
        : '') + '</td>' +
      '<td>' + formatDuration(connection.connected_seconds) + '</td><td>' +
      formatActivityTime(
        connection.last_satisfied_at,
        connection.last_satisfied_seconds,
        connection.last_satisfied_kind,
      ) + '</td><td>' +
      formatActivityTime(connection.last_client_work_at, connection.last_client_work_seconds) +
      '</td></tr>';
  }).join('');
}
field('refresh-websockets').addEventListener('click', refreshWebsockets);

async function refreshClients() {
  const r = await getJSON(ADMIN + '/clients', { cache: 'no-store' });
  const clients = (r.body && r.body.clients) || [];
  const requests = (r.body && r.body.requests) || [];
  const active = Number((r.body && r.body.active_count) || 0);
  const activeRequests = Number((r.body && r.body.active_requests) || 0);
  field('client-note').textContent = active + ' connected · ' + clients.length +
    ' known · ' + activeRequests + ' active request(s)';
  field('nav-client-count').textContent = clients.length;
  field('top-client-count').textContent = active + '/' + clients.length;
  const requestBody = field('fastapi-requests');
  requestBody.innerHTML = requests.length ? requests.map(request => {
    const client = request.declared_client_id || request.client_id;
    const state = request.active
      ? '<span class="dot on"></span>active'
      : '<span class="dot off"></span>HTTP ' + esc(request.status ?? '--');
    return '<tr><td><code>' + esc(request.external_request_id || request.request_id) +
      '</code></td><td>' + esc(client) + '</td><td><b>' + esc(request.method) +
      '</b><br><code>' + esc(request.endpoint) + '</code></td><td>' + state +
      '</td><td>' + formatActivityTime(request.started_at, request.age_seconds) +
      '</td><td>' + formatDuration(request.duration_seconds) + '</td></tr>';
  }).join('') : '<tr><td colspan="6" class="muted">none observed</td></tr>';
  const tbody = field('openai-clients');
  if (!clients.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="muted">none observed</td></tr>';
    return;
  }
  tbody.innerHTML = clients.map(client => {
    const name = client.declared_id || client.client_id;
    const address = client.host + (client.last_port != null ? ':' + client.last_port : '');
    const state = client.connected
      ? '<span class="dot on"></span>connected · ' + client.active_requests + ' active'
      : '<span class="dot off"></span>idle';
    return '<tr><td><b>' + esc(name) + '</b><br><span class="muted">' +
      esc(client.user_agent || 'unknown') + '</span></td><td><code>' + esc(address) +
      '</code></td><td>' + state + '</td><td>' + client.requests + '</td><td><code>' +
      esc((client.last_method || '') + ' ' + (client.last_endpoint || '')) +
      '</code><br><span class="muted">HTTP ' + esc(client.last_status ?? '--') +
      '</span></td><td>' +
      formatActivityTime(client.first_seen_at, client.first_seen_seconds) + '</td><td>' +
      formatActivityTime(client.last_seen_at, client.last_seen_seconds) + '</td></tr>';
  }).join('');
}
field('refresh-clients').addEventListener('click', refreshClients);

async function refreshWorkers() {
  const r = await getJSON(ADMIN + '/workers', { cache: 'no-store' });
  const note = document.getElementById('sup-note');
  const tbody = document.getElementById('workers');
  const workers = (r.body && r.body.workers) || [];
  field('nav-managed-count').textContent = workers.length;
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
document.getElementById('refresh-services').addEventListener('click', tick);

function csv(value) { return value.split(',').map(v => v.trim()).filter(Boolean); }
function field(id) { return document.getElementById(id); }
function setCopilotMsg(text, cls) {
  const node = field('copilot-msg'); node.textContent = text; node.className = 'msg ' + cls;
}
function syncCopilotAntiIdleOverride() {
  const inherited = field('cp-shared-anti-idle').checked;
  field('cp-keepalive-interval').disabled = inherited;
  field('cp-keepalive-timeout').disabled = inherited;
}
function resetCopilotForm() {
  editingCopilot = null;
  field('copilot-editor-title').textContent = 'Add headless Copilot servant';
  field('cp-worker-id').readOnly = false;
  field('cp-worker-id').value = suggestedCopilotId;
  field('cp-model').value = '';
  field('cp-model-pool').value = '';
  field('cp-model-selector').value = 'random';
  field('cp-modelmasks').value = '';
  field('cp-role').value = 'headless-copilot';
  field('cp-capabilities').value = '';
  field('cp-cwd').value = '';
  field('cp-host-ws-url').value = '';
  field('cp-command').value = '';
  field('cp-session-id').value = '';
  field('cp-context').value = 'default';
  field('cp-effort').value = '';
  field('cp-credits').value = '';
  field('cp-timeout').value = '900';
  field('cp-reconnect').value = '2';
  field('cp-keepalive-interval').value = '';
  field('cp-keepalive-timeout').value = '';
  field('cp-warmup-prompt').value = 'Startup warmup: reply only READY.';
  field('cp-chunk-tokens').value = '';
  field('cp-max-chunks').value = '64';
  field('cp-max-prompt').value = '4000000';
  field('cp-max-output').value = '200000';
  field('cp-max-attachment').value = '26214400';
  field('copilot-prompt').value = DEFAULT_COPILOT_PROMPT;
  field('cp-autostart').checked = true;
  field('cp-chunk-prompts').checked = true;
  field('cp-allow-all').checked = true;
  field('cp-custom-instructions').checked = true;
  field('cp-builtin-mcps').checked = true;
  field('cp-shared-anti-idle').checked = true;
  syncCopilotAntiIdleOverride();
  updateReasoningOptions();
  renderCopilotCapabilities();
  setCopilotMsg('', '');
}
function editCopilot(config) {
  editingCopilot = config.worker_id;
  field('copilot-editor').open = true;
  field('copilot-editor-title').textContent = 'Edit ' + config.worker_id;
  field('cp-worker-id').value = config.worker_id || '';
  field('cp-worker-id').readOnly = true;
  field('cp-model').value = config.model || '';
  field('cp-model-pool').value = (config.model_pool || []).join(',');
  field('cp-model-selector').value = config.model_selector || 'random';
  field('cp-modelmasks').value = (config.modelmasks || []).join(',');
  field('cp-role').value = config.role || 'headless-copilot';
  field('cp-capabilities').value = (config.capabilities || []).join(',');
  field('cp-cwd').value = config.cwd || '';
  field('cp-host-ws-url').value = config.host_ws_url || '';
  field('cp-command').value = config.copilot_command || '';
  field('cp-session-id').value = config.session_id || '';
  field('cp-context').value = config.context || 'default';
  field('cp-effort').value = config.reasoning_effort || '';
  field('cp-credits').value = config.max_ai_credits ?? '';
  field('cp-timeout').value = config.timeout_seconds ?? 900;
  field('cp-reconnect').value = config.reconnect_seconds ?? 2;
  field('cp-keepalive-interval').value = config.keepalive_interval_seconds ?? '';
  field('cp-keepalive-timeout').value = config.keepalive_timeout_seconds ?? '';
  field('cp-warmup-prompt').value = config.warmup_prompt || 'Startup warmup: reply only READY.';
  field('cp-chunk-tokens').value = config.chunk_tokens ?? '';
  field('cp-max-chunks').value = config.max_chunks ?? 64;
  field('cp-max-prompt').value = config.max_prompt_chars ?? 4000000;
  field('cp-max-output').value = config.max_output_chars ?? 200000;
  field('cp-max-attachment').value = config.max_attachment_bytes ?? 26214400;
  field('copilot-prompt').value = config.system_prompt || DEFAULT_COPILOT_PROMPT;
  field('cp-autostart').checked = config.autostart !== false;
  field('cp-chunk-prompts').checked = config.chunk_long_prompts !== false;
  field('cp-allow-all').checked = !!config.allow_all;
  field('cp-custom-instructions').checked = !!config.load_custom_instructions;
  field('cp-builtin-mcps').checked = !!config.enable_builtin_mcps;
  field('cp-shared-anti-idle').checked = config.use_shared_anti_idle !== false;
  syncCopilotAntiIdleOverride();
  updateReasoningOptions(config.reasoning_effort || '');
  renderCopilotCapabilities();
  setCopilotMsg('', '');
}
function copilotConfigFromForm() {
  const config = {
    worker_id: field('cp-worker-id').value.trim(),
    model_pool: csv(field('cp-model-pool').value),
    model_selector: field('cp-model-selector').value,
    modelmasks: csv(field('cp-modelmasks').value),
    role: field('cp-role').value.trim() || 'headless-copilot',
    capabilities: csv(field('cp-capabilities').value),
    system_prompt: field('copilot-prompt').value,
    autostart: field('cp-autostart').checked,
    warmup: true,
    warmup_prompt: field('cp-warmup-prompt').value,
    chunk_long_prompts: field('cp-chunk-prompts').checked,
    max_chunks: Number(field('cp-max-chunks').value),
    timeout_seconds: Number(field('cp-timeout').value),
    reconnect_seconds: Number(field('cp-reconnect').value),
    context: field('cp-context').value,
    allow_all: field('cp-allow-all').checked,
    load_custom_instructions: field('cp-custom-instructions').checked,
    enable_builtin_mcps: field('cp-builtin-mcps').checked,
    use_shared_anti_idle: field('cp-shared-anti-idle').checked,
    max_prompt_chars: Number(field('cp-max-prompt').value),
    max_output_chars: Number(field('cp-max-output').value),
    max_attachment_bytes: Number(field('cp-max-attachment').value),
  };
  for (const [key, id] of [
    ['session_id','cp-session-id'], ['model','cp-model'], ['cwd','cp-cwd'],
    ['host_ws_url','cp-host-ws-url'], ['copilot_command','cp-command'],
    ['reasoning_effort','cp-effort'],
  ]) {
    const value = field(id).value.trim(); if (value) config[key] = value;
  }
  const credits = field('cp-credits').value;
  if (credits !== '') config.max_ai_credits = Number(credits);
  const chunkTokens = field('cp-chunk-tokens').value;
  if (chunkTokens !== '') config.chunk_tokens = Number(chunkTokens);
  const keepaliveInterval = field('cp-keepalive-interval').value;
  if (!config.use_shared_anti_idle && keepaliveInterval !== '') {
    config.keepalive_interval_seconds = Number(keepaliveInterval);
  }
  const keepaliveTimeout = field('cp-keepalive-timeout').value;
  if (!config.use_shared_anti_idle && keepaliveTimeout !== '') {
    config.keepalive_timeout_seconds = Number(keepaliveTimeout);
  }
  return config;
}
async function refreshCopilots() {
  const r = await getJSON(ADMIN + '/copilots', { cache: 'no-store' });
  const tbody = field('copilots');
  const note = field('copilot-note');
  copilotInstances = (r.body && r.body.instances) || [];
  const online = copilotInstances.filter(item => item.running || item.connected);
  const offline = copilotInstances.filter(item => !item.running && !item.connected);
  field('copilot-start-all').disabled = offline.length === 0;
  field('copilot-stop-all').disabled = online.length === 0;
  field('copilot-stop-idle').disabled = online.length === 0;
  field('copilot-restart-all').disabled = online.length === 0;
  field('copilot-reset-all').disabled = online.length === 0;
  field('nav-servant-count').textContent = copilotInstances.length;
  field('top-servant-count').textContent = copilotInstances.length;
  suggestedCopilotId = (r.body && r.body.next_worker_id) || 'worker-copilot-1';
  if (!editingCopilot) field('cp-worker-id').value = suggestedCopilotId;
  note.textContent = !r.body.manager_active ? '(manager unavailable)' :
    (r.body.copilot_available ? ('CLI: ' + (r.body.copilot_command || 'available')) : '(Copilot CLI not found)');
  if (!copilotInstances.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="muted">none configured</td></tr>';
    return;
  }
  tbody.innerHTML = copilotInstances.map(item => {
    const cfg = item.config || {};
    const runtime = item.runtime || {};
    const state = item.connected ? 'connected' : (item.running ? 'starting / reconnecting' : 'stopped');
    const online = item.running || item.connected;
    const masks = (item.modelmasks || []).length ? item.modelmasks.join(', ') : 'all API models';
    return '<tr><td><span class="dot ' + (item.running ? 'on' : 'off') + '"></span><b>' + esc(item.worker_id) + '</b></td>' +
      '<td>' + esc(item.selected_model || item.model || 'not started') +
      (item.model ? '' : ' <span class="muted">(' + esc(item.model_selector || 'random') + ')</span>') +
      (item.selected_reasoning_effort ? ('<br>effort ' + esc(item.selected_reasoning_effort)) :
        (item.reasoning_effort ? ('<br>effort ' + esc(item.reasoning_effort)) : '')) +
      '<br><code title="' + esc(item.session_id) + '">' + esc(item.session_id.slice(0, 12)) + '...</code></td>' +
      '<td>' + esc(masks) + '</td><td>' + esc(state) + (item.pid ? (' · adapter ' + item.pid) : '') +
      (runtime.runtime_pid ? (' · CLI ' + runtime.runtime_pid) : '') +
      (runtime.bridge_pid ? (' · bridge ' + runtime.bridge_pid) : '') +
      (runtime.warmup_duration_ms != null ? ('<br>warmup ' + (runtime.warmup_duration_ms / 1000).toFixed(1) + 's') : '') +
      (runtime.last_chunk_count > 1 ? (' · ' + runtime.last_chunk_count + ' chunks') : '') +
      (runtime.last_duration_ms != null ? ('<br>last ' + (runtime.last_duration_ms / 1000).toFixed(1) + 's') : '') +
      (runtime.keepalives ? (' · keepalives ' + runtime.keepalives) : '') +
      ((runtime.retired_keepalive_tasks || []).length
        ? (' · retired tasks ' + runtime.retired_keepalive_tasks.length)
        : '') + '</td>' +
      '<td><button data-cp-act="edit" data-id="' + esc(item.worker_id) + '">Edit</button>' +
      '<button data-cp-act="start" data-id="' + esc(item.worker_id) + '"' + (online ? ' disabled' : '') + '>Start</button>' +
      '<button data-cp-act="stop" data-id="' + esc(item.worker_id) + '"' + (!online ? ' disabled' : '') + '>Stop</button>' +
      '<button data-cp-act="restart" data-id="' + esc(item.worker_id) + '"' + (!online ? ' disabled' : '') + '>Restart</button>' +
      '<button data-cp-act="reset-session" data-id="' + esc(item.worker_id) + '"' + (!online ? ' disabled' : '') + '>New session</button>' +
      '<button data-cp-act="delete" data-id="' + esc(item.worker_id) + '">Delete</button>' +
      '<a class="action-link" href="' + ADMIN + '/copilots/' + encodeURIComponent(item.worker_id) + '/log" target="_blank">log</a></td></tr>';
  }).join('');
}
async function refreshCopilotModels(force) {
  const r = await getJSON(ADMIN + '/copilots/models?refresh=' + (force ? 'true' : 'false'), { cache: 'no-store' });
  const models = (r.body && r.body.models) || [];
  copilotModelCatalog = models;
  field('cp-model-picker').innerHTML = '<option value="">Select a Copilot model...</option>' +
    models.filter(model => model.id !== 'auto').map(model =>
      '<option value="' + esc(model.id) + '">' +
      (model.quality_rank ? ('#' + model.quality_rank + ' ') : '') +
      esc(model.name || model.id) + ' · ' + esc(model.quality_tier || 'unranked') + '</option>'
    ).join('');
  field('cp-model-picker').disabled = models.length === 0;
  field('cp-model-picker').value = [...field('cp-model-picker').options].some(
    option => option.value === field('cp-model').value
  ) ? field('cp-model').value : '';
  const suffix = r.body.error ? ('; fallback: ' + r.body.error) : '';
  field('copilot-model-note').textContent = models.length
    ? ('Available Copilot models: ' + models.length + ' (' + (r.body.source || 'unknown') + ')' + suffix)
    : 'No Copilot models discovered';
  updateReasoningOptions();
  renderCopilotCapabilities();
  renderApiModelCapabilities();
}
field('refresh-copilots').addEventListener('click', refreshCopilots);
field('cp-shared-anti-idle').addEventListener(
  'change',
  syncCopilotAntiIdleOverride,
);
field('refresh-copilot-models').addEventListener('click', () => refreshCopilotModels(true));
async function runBulkCopilotAction(action, prompt) {
  if (prompt && !confirm(prompt)) return;
  const buttons = [
    'copilot-start-all','copilot-stop-all','copilot-stop-idle',
    'copilot-restart-all','copilot-reset-all',
  ];
  for (const id of buttons) field(id).disabled = true;
  field('copilot-bulk-msg').textContent = action + '...';
  const response = await getJSON(ADMIN + '/copilots/bulk/' + action, {
    method: 'POST',
  });
  field('copilot-bulk-msg').textContent = response.ok
    ? (response.body.affected + ' worker(s) affected')
    : ('bulk action failed: ' + JSON.stringify((response.body && response.body.detail) || response.status));
  field('copilot-bulk-msg').className = 'msg ' + (response.ok ? 'ok' : 'err');
  await refreshCopilots();
}
field('copilot-start-all').addEventListener('click', () =>
  runBulkCopilotAction('start', 'Start every offline Copilot worker?'));
field('copilot-stop-all').addEventListener('click', () =>
  runBulkCopilotAction('stop', 'Stop every online Copilot worker and pause idle maintenance?'));
field('copilot-stop-idle').addEventListener('click', () =>
  runBulkCopilotAction('stop-idle', 'Stop workers idle for at least the configured grace period?'));
field('copilot-restart-all').addEventListener('click', () =>
  runBulkCopilotAction('restart', 'Restart every online Copilot worker?'));
field('copilot-reset-all').addEventListener('click', () =>
  runBulkCopilotAction('reset-session', 'Replace the persistent sessions of every online worker?'));
field('cp-model-picker').addEventListener('change', () => {
  if (field('cp-model-picker').value) field('cp-model').value = field('cp-model-picker').value;
  updateReasoningOptions();
  renderCopilotCapabilities();
});
field('cp-model').addEventListener('input', () => {
  updateReasoningOptions();
  renderCopilotCapabilities();
});
field('cp-model-pool').addEventListener('input', () => updateReasoningOptions());
function updateReasoningOptions(preferred) {
  const order = ['none','minimal','low','medium','high','xhigh','max'];
  const selectors = ['random','most-1','most-2','most-3','least-1','least-2','least-3'];
  const explicit = field('cp-model').value.trim();
  const pool = csv(field('cp-model-pool').value);
  const candidates = explicit ? copilotModelCatalog.filter(model => model.id === explicit) :
    (pool.length ? copilotModelCatalog.filter(model => pool.includes(model.id)) :
      copilotModelCatalog.filter(model => model.id !== 'auto'));
  const supported = new Set();
  for (const model of candidates) {
    const levels = model.capabilities && model.capabilities.supports
      ? model.capabilities.supports.reasoning_effort : [];
    for (const level of (Array.isArray(levels) ? levels : [])) supported.add(level);
  }
  const current = preferred || field('cp-effort').value;
  const levels = order.filter(level => supported.size === 0 || supported.has(level));
  field('cp-effort').innerHTML = '<option value="">model default</option>' +
    selectors.map(value => '<option value="' + value + '">' + value + '</option>').join('') +
    levels.map(level => '<option value="' + level + '">' + level + '</option>').join('');
  field('cp-effort').value = [...selectors, ...levels].includes(current) ? current : '';
}
function beginNewCopilot() {
  resetCopilotForm();
  field('copilot-editor').open = true;
  field('copilot-editor').scrollIntoView({ behavior: 'smooth', block: 'start' });
  field('cp-worker-id').focus();
}
field('copilot-new').addEventListener('click', beginNewCopilot);
field('copilot-add-another').addEventListener('click', beginNewCopilot);
field('copilots').addEventListener('click', async (event) => {
  const button = event.target.closest('button[data-cp-act]'); if (!button) return;
  const action = button.dataset.cpAct;
  const id = button.dataset.id;
  const item = copilotInstances.find(candidate => candidate.worker_id === id);
  if (action === 'edit') { editCopilot((item && item.config) || {}); return; }
  if (action === 'delete') {
    if (!confirm('Delete headless Copilot servant ' + id + '? Logs are retained.')) return;
    if (item && (item.running || item.connected)) {
      await getJSON(
        ADMIN + '/copilots/' + encodeURIComponent(id) + '/online-action/stop',
        { method: 'POST' },
      );
    }
    await getJSON(ADMIN + '/copilots/' + encodeURIComponent(id), { method: 'DELETE' });
  } else {
    await getJSON(
      ADMIN + '/copilots/' + encodeURIComponent(id) + '/online-action/' + action,
      { method: 'POST' },
    );
  }
  await refreshCopilots(); await loadConfig();
});
field('copilot-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const config = copilotConfigFromForm();
  if (!config.worker_id) { setCopilotMsg('worker ID is required', 'err'); return; }
  const path = editingCopilot
    ? ADMIN + '/copilots/' + encodeURIComponent(editingCopilot) + '?restart=true'
    : ADMIN + '/copilots?start=true';
  const r = await getJSON(path, {
    method: editingCopilot ? 'PUT' : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  const detail = r.body && r.body.detail;
  setCopilotMsg(r.ok ? 'saved and active' : ('save failed: ' + (typeof detail === 'string' ? detail : JSON.stringify(detail || r.status))), r.ok ? 'ok' : 'err');
  if (r.ok) {
    editCopilot(r.body.config);
    await refreshCopilots(); await loadConfig();
  }
});

function configSectionDefault(section) {
  return CONFIG_SECTIONS[section][1] === 'array' ? [] : {};
}
function configSectionSummary(value) {
  if (Array.isArray(value)) return value.length;
  if (value && typeof value === 'object') return Object.keys(value).length;
  return value == null ? 0 : 1;
}
function renderConfigSection() {
  field('config-section-tabs').innerHTML = Object.keys(CONFIG_SECTIONS).map(section =>
    '<button type="button" class="section-tab ' + (section === activeConfigSection ? 'active' : '') +
    '" data-section="' + section + '"><span>' + esc(section) + '</span><b>' +
    configSectionSummary(currentConfig[section]) + '</b></button>'
  ).join('');
  field('config-section-title').textContent = activeConfigSection;
  field('config-section-help').textContent = CONFIG_SECTIONS[activeConfigSection][0];
  const value = currentConfig[activeConfigSection] ?? configSectionDefault(activeConfigSection);
  field('config-section-editor').value = JSON.stringify(value, null, 2);
}
field('config-section-tabs').addEventListener('click', event => {
  const button = event.target.closest('button[data-section]'); if (!button) return;
  activeConfigSection = button.dataset.section;
  renderConfigSection();
});
field('config-section-reload').addEventListener('click', renderConfigSection);
field('config-section-save').addEventListener('click', async () => {
  let value;
  try { value = JSON.parse(field('config-section-editor').value); }
  catch (error) {
    field('config-section-msg').textContent = 'invalid JSON: ' + error.message;
    field('config-section-msg').className = 'msg err';
    return;
  }
  const r = await getJSON(ADMIN + '/config/section/' + activeConfigSection, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ value, expected_revision: currentConfigRevision }),
  });
  const detail = r.body && r.body.detail;
  field('config-section-msg').textContent = r.ok ? 'saved; restart required' :
    ('save failed: ' + (typeof detail === 'string' ? detail : JSON.stringify(detail || r.status)));
  field('config-section-msg').className = 'msg ' + (r.ok ? 'ok' : 'err');
  if (r.ok) {
    currentConfig = r.body.config;
    currentConfigRevision = r.body.revision;
    field('config').value = JSON.stringify(currentConfig, null, 2);
    renderConfigSection();
  }
});
field('config-section-delete').addEventListener('click', async () => {
  if (!confirm('Delete the ' + activeConfigSection + ' section from config.json?')) return;
  const r = await getJSON(ADMIN + '/config/section/' + activeConfigSection, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ delete: true, expected_revision: currentConfigRevision }),
  });
  field('config-section-msg').textContent = r.ok ? 'deleted; restart required' : ('delete failed (' + r.status + ')');
  field('config-section-msg').className = 'msg ' + (r.ok ? 'ok' : 'err');
  if (r.ok) {
    currentConfig = r.body.config;
    currentConfigRevision = r.body.revision;
    field('config').value = JSON.stringify(currentConfig, null, 2);
    renderConfigSection();
  }
});

function formatBytes(value) {
  const bytes = Number(value) || 0;
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KiB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MiB';
}
function attachmentPreview(entry, index) {
  const type = entry.file.type || 'application/octet-stream';
  const anonymousName = 'attachment-' + (index + 1);
  if (type.startsWith('image/')) return '<img src="' + esc(entry.url) + '" alt="Preview of ' + anonymousName + '">';
  if (type.startsWith('audio/')) return '<audio controls src="' + esc(entry.url) + '"></audio>';
  if (type.startsWith('video/')) return '<video controls src="' + esc(entry.url) + '"></video>';
  return '<div class="file-icon">FILE</div>';
}
function renderModelTestAttachments() {
  const total = modelTestAttachments.reduce((sum, entry) => sum + entry.file.size, 0);
  field('model-test-attachment-summary').textContent = modelTestAttachments.length
    ? modelTestAttachments.length + ' attachment(s) · ' + formatBytes(total)
    : 'No attachments selected.';
  field('model-test-attachments').innerHTML = modelTestAttachments.map((entry, index) =>
    '<div class="attachment-card">' + attachmentPreview(entry, index) +
    '<div class="attachment-name">attachment-' + (index + 1) + '</div>' +
    '<div class="attachment-meta">' + esc(entry.file.type || 'application/octet-stream') + ' · ' +
    formatBytes(entry.file.size) + '</div>' +
    '<button type="button" data-remove-attachment="' + index + '">Remove</button></div>'
  ).join('');
}
function clearModelTestAttachments() {
  for (const entry of modelTestAttachments) URL.revokeObjectURL(entry.url);
  modelTestAttachments = [];
  field('model-test-files').value = '';
  renderModelTestAttachments();
}
function addModelTestFiles(files) {
  const incoming = [...files];
  const existingKeys = new Set(modelTestAttachments.map(entry =>
    entry.file.name + ':' + entry.file.size + ':' + entry.file.lastModified
  ));
  for (const file of incoming) {
    const key = file.name + ':' + file.size + ':' + file.lastModified;
    if (existingKeys.has(key)) continue;
    if (modelTestAttachments.length >= 12) {
      field('model-test-status').textContent = 'maximum 12 attachments';
      break;
    }
    if (file.size > 25 * 1024 * 1024) {
      field('model-test-status').textContent = file.name + ' exceeds 25 MiB';
      continue;
    }
    const total = modelTestAttachments.reduce((sum, entry) => sum + entry.file.size, 0);
    if (total + file.size > 50 * 1024 * 1024) {
      field('model-test-status').textContent = 'attachments exceed 50 MiB total';
      break;
    }
    modelTestAttachments.push({ file, url: URL.createObjectURL(file) });
    existingKeys.add(key);
  }
  field('model-test-files').value = '';
  renderModelTestAttachments();
}
async function refreshModelTestSamples() {
  const response = await getJSON(ADMIN + '/test-samples', { cache: 'no-store' });
  const samples = (response.body && response.body.samples) || [];
  field('model-test-samples').innerHTML = samples.map(sample =>
    '<button type="button" data-test-sample="' + esc(sample.id) + '" data-url="' +
    esc(sample.url) + '" data-name="' + esc(sample.name) + '" data-mime="' +
    esc(sample.mime_type) + '" title="' + esc(sample.description) + '">+ ' +
    esc(sample.description) + '</button>'
  ).join(' ');
}
field('model-test-samples').addEventListener('click', async event => {
  const button = event.target.closest('button[data-test-sample]'); if (!button) return;
  button.disabled = true;
  try {
    const response = await fetch(button.dataset.url, { cache: 'no-store' });
    if (!response.ok) throw new Error('HTTP ' + response.status);
    const blob = await response.blob();
    const file = new File([blob], button.dataset.name, {
      type: button.dataset.mime || blob.type || 'application/octet-stream',
      lastModified: 0,
    });
    addModelTestFiles([file]);
    field('model-test-status').textContent = 'sample attached as attachment-' +
      modelTestAttachments.length;
  } catch (error) {
    field('model-test-status').textContent = 'sample failed: ' + error.message;
  } finally {
    button.disabled = false;
  }
});
field('model-test-files').addEventListener('change', event => addModelTestFiles(event.target.files || []));
field('model-test-attachments').addEventListener('click', event => {
  const button = event.target.closest('button[data-remove-attachment]'); if (!button) return;
  const index = Number(button.dataset.removeAttachment);
  const [removed] = modelTestAttachments.splice(index, 1);
  if (removed) URL.revokeObjectURL(removed.url);
  renderModelTestAttachments();
});
field('model-test-clear-attachments').addEventListener('click', clearModelTestAttachments);
for (const eventName of ['dragenter', 'dragover']) {
  field('model-test-drop').addEventListener(eventName, event => {
    event.preventDefault(); field('model-test-drop').classList.add('dragging');
  });
}
for (const eventName of ['dragleave', 'drop']) {
  field('model-test-drop').addEventListener(eventName, event => {
    event.preventDefault(); field('model-test-drop').classList.remove('dragging');
  });
}
field('model-test-drop').addEventListener('drop', event => addModelTestFiles(event.dataTransfer.files || []));
function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error || new Error('file read failed'));
    reader.onload = () => resolve(String(reader.result).split(',', 2)[1] || '');
    reader.readAsDataURL(file);
  });
}
async function encodedModelTestAttachments() {
  return Promise.all(modelTestAttachments.map(async (entry, index) => ({
    name: 'attachment-' + (index + 1),
    mime_type: entry.file.type || 'application/octet-stream',
    data_b64: await fileToBase64(entry.file),
  })));
}
function renderUploadedAttachments(attachments) {
  field('model-test-uploaded').innerHTML = (attachments || []).map(attachment => {
    const url = attachment.url || '';
    const type = attachment.mime_type || 'application/octet-stream';
    let preview = '<div class="file-icon">FILE</div>';
    if (type.startsWith('image/')) preview = '<img src="' + esc(url) + '" alt="Uploaded ' + esc(attachment.name) + '">';
    else if (type.startsWith('audio/')) preview = '<audio controls src="' + esc(url) + '"></audio>';
    else if (type.startsWith('video/')) preview = '<video controls src="' + esc(url) + '"></video>';
    return '<div class="attachment-card">' + preview +
      '<div class="attachment-name">' + esc(attachment.name) + '</div>' +
      '<div class="attachment-meta">' + esc(type) + ' · ' + formatBytes(attachment.bytes) + '</div>' +
      '<a class="action-link" href="' + esc(url) + '" target="_blank" download>open</a></div>';
  }).join('');
}

function modelConfigRouteTargets() {
  return field('model-config-route').value.split(/\\r?\\n/)
    .map(value => value.trim()).filter(Boolean);
}
function syncModelRouteShortcuts() {
  const targets = modelConfigRouteTargets();
  field('model-route-worker-name').checked = targets.includes('worker-in-name');
  field('model-route-copilot').checked = targets.includes('worker-copilot-*');
  field('model-route-codex').checked = targets.includes('worker-codex-*');
  field('model-route-unknown').checked = targets.includes('worker-unknown-*');
  field('model-route-backends').checked = targets.includes('backend-*');
  for (const input of field('model-route-specific-backends').querySelectorAll('input[data-backend-target]')) {
    input.checked = targets.includes(input.dataset.backendTarget);
  }
  renderModelRouteOrder();
}
function renderSpecificBackendShortcuts() {
  field('model-route-specific-backends').innerHTML = (modelConfigMeta.backends || []).map(backend => {
    const target = 'backend-' + backend.name;
    return '<label><input type="checkbox" data-backend-target="' + esc(target) + '"> ' +
      esc(target) + '</label>';
  }).join('');
}
function renderModelRouteOrder() {
  const targets = modelConfigRouteTargets();
  field('model-route-order').innerHTML = targets.length ? targets.map((target, index) =>
    '<div class="route-order-item"><button type="button" data-route-move="-1" data-index="' + index + '"' +
    (index === 0 ? ' disabled' : '') + ' aria-label="Move ' + esc(target) + ' earlier">←</button>' +
    '<code title="' + esc(target) + '">' + esc(target) + '</code>' +
    '<button type="button" data-route-move="1" data-index="' + index + '"' +
    (index === targets.length - 1 ? ' disabled' : '') + ' aria-label="Move ' + esc(target) + ' later">→</button></div>'
  ).join('') : '<div class="muted">No explicit route; model default behavior applies.</div>';
}
function setModelRouteShortcut(kind, checked) {
  let targets = modelConfigRouteTargets();
  const values = [kind];
  if (checked) {
    for (const value of values) if (!targets.includes(value)) targets.push(value);
  } else {
    targets = targets.filter(value => !values.includes(value));
  }
  field('model-config-route').value = targets.join('\\n');
  syncModelRouteShortcuts();
}
function modelConfigEntry() {
  return modelConfigCatalog.find(model => model.id === selectedModelConfigId);
}
function modelConfigEntries() {
  return selectedModelConfigIds
    .map(modelId => modelConfigCatalog.find(model => model.id === modelId))
    .filter(Boolean);
}
function modelConfigJson() {
  try { return JSON.parse(field('model-config-json').value || '{}'); }
  catch (error) {
    field('model-config-msg').textContent = 'invalid model JSON: ' + error.message;
    field('model-config-msg').className = 'msg err';
    return null;
  }
}
function modelConfigModality(model, name) {
  return !!(model && model.input_modalities && model.input_modalities[name] &&
    model.input_modalities[name].enabled);
}
function modelConfigTask(model, name) {
  return !!(model && model.task_capabilities && model.task_capabilities[name] &&
    model.task_capabilities[name].enabled);
}
function syncModelConfigCheckboxes(model) {
  const override = (modelConfigMeta.overrides || {})[selectedModelConfigId] || {};
  field('model-config-export').checked = override.hidden !== true;
  field('model-config-ondemand').checked = !!model.on_demand;
  field('model-config-simulated').checked = !!model.simulated;
  field('model-config-image').checked = modelConfigModality(model, 'image');
  field('model-config-audio').checked = modelConfigModality(model, 'audio');
  field('model-config-file').checked = modelConfigModality(model, 'general_file');
  field('model-config-code').checked = modelConfigTask(model, 'code');
  field('model-config-image-output').checked = modelConfigTask(model, 'image_output');
  field('model-config-summary').checked = modelConfigTask(model, 'summarization');
  field('model-config-load').disabled = !model.id.startsWith('copilot/') || !model.on_demand;
}
function applyModelConfigCheckboxes() {
  const model = modelConfigJson(); if (!model) return;
  if (selectedModelConfigIds.length === 1) model.id = selectedModelConfigId;
  else delete model.id;
  model.on_demand = field('model-config-ondemand').checked;
  model.simulated = field('model-config-simulated').checked;
  model.input_modalities = model.input_modalities || {};
  for (const [name, id] of [
    ['image', 'model-config-image'],
    ['audio', 'model-config-audio'],
    ['general_file', 'model-config-file'],
  ]) {
    model.input_modalities[name] = model.input_modalities[name] || {};
    model.input_modalities[name].enabled = field(id).checked;
  }
  model.task_capabilities = model.task_capabilities || {};
  for (const [name, id] of [
    ['code', 'model-config-code'],
    ['image_output', 'model-config-image-output'],
    ['summarization', 'model-config-summary'],
  ]) {
    model.task_capabilities[name] = model.task_capabilities[name] || {};
    model.task_capabilities[name].enabled = field(id).checked;
  }
  field('model-config-json').value = JSON.stringify(model, null, 2);
  field('model-config-load').disabled = !selectedModelConfigIds.length ||
    !selectedModelConfigIds.every(modelId => modelId.startsWith('copilot/')) ||
    !model.on_demand;
}
function renderModelConfigSlots() {
  const onDemand = modelConfigMeta.on_demand || {};
  const slots = onDemand.slots || [];
  const limit = Number(onDemand.limit) || 4;
  field('model-config-slot-count').textContent = slots.length + '/' + limit;
  const cards = [...slots].sort((left, right) =>
    String(left.worker_id).localeCompare(String(right.worker_id), undefined, { numeric: true })
  ).map(slot => {
    const workerId = slot.worker_id;
    const state = slot.connected ? 'connected' : (slot.running ? 'starting' : 'stopped');
    return '<div class="slot-card"><b>' + esc(workerId) + '</b> · ' + esc(state) +
      '<br>' + esc(slot.selected_model || slot.model || ((slot.config || {}).model) || 'unassigned') + '</div>';
  });
  const available = Math.max(0, limit - slots.length);
  if (available) cards.push('<div class="slot-card"><b>+' + available + '</b> elastic slots available</div>');
  field('model-config-slots').innerHTML = cards.join('');
}
function renderModelConfigList() {
  const search = field('model-config-search').value.trim().toLowerCase();
  const models = modelConfigCatalog.filter(model =>
    !search || JSON.stringify(model).toLowerCase().includes(search));
  field('model-config-list').innerHTML = models.map(model => {
    const hidden = model.hidden === true ||
      ((modelConfigMeta.overrides || {})[model.id] || {}).hidden === true;
    return '<option value="' + esc(model.id) + '">' + (hidden ? '[unexported] ' : '') +
      esc(model.id) + '</option>';
  }).join('');
  const visibleIds = new Set(models.map(model => model.id));
  selectedModelConfigIds = selectedModelConfigIds.filter(modelId => visibleIds.has(modelId));
  if (!selectedModelConfigIds.length && models.length) {
    selectedModelConfigIds = [models[0].id];
  }
  selectedModelConfigId = selectedModelConfigIds[0] || '';
  for (const option of field('model-config-list').options) {
    option.selected = selectedModelConfigIds.includes(option.value);
  }
  renderSelectedModelConfig();
}
function renderSelectedModelConfig() {
  const model = modelConfigEntry();
  const selected = modelConfigEntries();
  const controls = [
    'model-config-export','model-config-ondemand','model-config-simulated',
    'model-config-image','model-config-audio','model-config-file',
    'model-config-code','model-config-summary',
    'model-config-image-output',
    'model-config-save','model-config-reset','model-config-clear-route',
  ];
  for (const id of controls) field(id).disabled = !model;
  if (!model) {
    field('model-config-title').textContent = 'No model selected';
    field('model-config-selection-note').textContent = '';
    field('model-config-json').value = '';
    field('model-config-route').value = '';
    field('model-config-load').disabled = true;
    return;
  }
  const bulk = selected.length > 1;
  field('model-config-title').textContent = bulk
    ? selected.length + ' models selected'
    : model.id;
  field('model-config-selection-note').textContent = bulk
    ? ('Bulk save applies this shared merge patch, checkbox state, and route order to: ' +
      selectedModelConfigIds.join(', '))
    : 'Editing the effective exported record for this model.';
  field('model-config-json').value = JSON.stringify(bulk ? {} : model, null, 2);
  const route = (modelConfigMeta.routes || {})[model.id];
  field('model-config-route').value = Array.isArray(route) ? route.join('\\n') : (route || '');
  syncModelConfigCheckboxes(model);
  field('model-config-load').disabled = !selectedModelConfigIds.every(
    modelId => modelId.startsWith('copilot/')
  ) || !field('model-config-ondemand').checked;
  syncModelRouteShortcuts();
}
async function refreshModelConfigurator() {
  const [catalog, metadata] = await Promise.all([
    getJSON('/v1/models?hidden=true', { cache: 'no-store' }),
    getJSON(ADMIN + '/model-config', { cache: 'no-store' }),
  ]);
  modelConfigMeta = metadata.body || modelConfigMeta;
  const byId = new Map(((catalog.body && catalog.body.data) || []).map(model => [model.id, model]));
  for (const [modelId, override] of Object.entries(modelConfigMeta.overrides || {})) {
    if (!byId.has(modelId)) {
      byId.set(modelId, { id: modelId, object: 'model', ...(override.patch || {}) });
    }
  }
  modelConfigCatalog = [...byId.values()];
  field('nav-config-model-count').textContent = modelConfigCatalog.length;
  renderSpecificBackendShortcuts();
  renderModelConfigSlots();
  renderModelConfigList();
}
async function saveSelectedModelConfig(reset = false) {
  const model = modelConfigJson(); if (!model || !selectedModelConfigIds.length) return;
  applyModelConfigCheckboxes();
  const edited = modelConfigJson(); if (!edited) return;
  const patch = { ...edited }; delete patch.id;
  const route = modelConfigRouteTargets();
  field('model-config-msg').textContent = reset ? 'resetting...' : 'saving...';
  field('model-config-msg').className = 'msg';
  const responses = [];
  let revision = modelConfigMeta.revision;
  for (const modelId of selectedModelConfigIds) {
    const response = await getJSON(ADMIN + '/model-config', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_id: modelId,
        hidden: !field('model-config-export').checked,
        patch,
        set_route: !reset,
        route: route.length ? route : null,
        reset,
        expected_revision: revision,
      }),
    });
    responses.push(response);
    if (!response.ok) break;
    revision = response.body.revision;
  }
  const failed = responses.find(response => !response.ok);
  field('model-config-msg').textContent = failed
    ? ('save failed: ' + JSON.stringify((failed.body && failed.body.detail) || failed.status))
    : ((reset ? 'reset ' : 'saved ') + responses.length + ' model(s)');
  field('model-config-msg').className = 'msg ' + (failed ? 'err' : 'ok');
  if (!failed) {
    await Promise.all([refreshApiModels(), refreshModelConfigurator(), loadConfig()]);
  }
}
field('model-config-refresh').addEventListener('click', refreshModelConfigurator);
field('model-config-search').addEventListener('input', renderModelConfigList);
field('model-config-list').addEventListener('change', () => {
  selectedModelConfigIds = [...field('model-config-list').selectedOptions]
    .map(option => option.value);
  selectedModelConfigId = selectedModelConfigIds[0] || '';
  renderSelectedModelConfig();
});
for (const id of [
  'model-config-ondemand','model-config-simulated','model-config-image',
  'model-config-audio','model-config-file','model-config-code',
  'model-config-image-output','model-config-summary',
]) field(id).addEventListener('change', applyModelConfigCheckboxes);
field('model-config-json').addEventListener('change', () => {
  const model = modelConfigJson(); if (model) syncModelConfigCheckboxes(model);
});
field('model-config-route').addEventListener('input', syncModelRouteShortcuts);
field('model-route-order').addEventListener('click', event => {
  const button = event.target.closest('button[data-route-move]'); if (!button) return;
  const targets = modelConfigRouteTargets();
  const index = Number(button.dataset.index);
  const next = index + Number(button.dataset.routeMove);
  if (index < 0 || next < 0 || index >= targets.length || next >= targets.length) return;
  [targets[index], targets[next]] = [targets[next], targets[index]];
  field('model-config-route').value = targets.join('\\n');
  syncModelRouteShortcuts();
});
field('model-route-worker-name').addEventListener('change', event =>
  setModelRouteShortcut('worker-in-name', event.target.checked));
field('model-route-copilot').addEventListener('change', event =>
  setModelRouteShortcut('worker-copilot-*', event.target.checked));
field('model-route-codex').addEventListener('change', event =>
  setModelRouteShortcut('worker-codex-*', event.target.checked));
field('model-route-unknown').addEventListener('change', event =>
  setModelRouteShortcut('worker-unknown-*', event.target.checked));
field('model-route-backends').addEventListener('change', event =>
  setModelRouteShortcut('backend-*', event.target.checked));
field('model-route-specific-backends').addEventListener('change', event => {
  const input = event.target.closest('input[data-backend-target]'); if (!input) return;
  setModelRouteShortcut(input.dataset.backendTarget, input.checked);
});
field('model-config-save').addEventListener('click', () => saveSelectedModelConfig(false));
field('model-config-reset').addEventListener('click', () => saveSelectedModelConfig(true));
field('model-config-clear-route').addEventListener('click', () => {
  field('model-config-route').value = '';
  syncModelRouteShortcuts();
  field('model-config-msg').textContent = 'route cleared locally; save to apply';
  field('model-config-msg').className = 'msg';
});
field('model-config-load').addEventListener('click', async () => {
  const modelIds = selectedModelConfigIds.filter(modelId => modelId.startsWith('copilot/'));
  if (!modelIds.length || modelIds.length !== selectedModelConfigIds.length) return;
  field('model-config-load').disabled = true;
  field('model-config-msg').textContent = 'loading ' + modelIds.length + ' on-demand worker(s)...';
  field('model-config-msg').className = 'msg';
  const responses = [];
  for (const modelId of modelIds) {
    const encoded = modelId.split('/').map(encodeURIComponent).join('/');
    responses.push(await getJSON(ADMIN + '/model-config/load/' + encoded, { method: 'POST' }));
  }
  const failed = responses.find(response => !response.ok);
  field('model-config-msg').textContent = failed
    ? ('load failed: ' + JSON.stringify((failed.body && failed.body.detail) || failed.status))
    : ('loaded ' + responses.map(response => response.body.worker.worker_id).join(', '));
  field('model-config-msg').className = 'msg ' + (failed ? 'err' : 'ok');
  await Promise.all([refreshCopilots(), refreshModelConfigurator(), refreshApiModels()]);
});

async function refreshApiModels() {
  const [r, state] = await Promise.all([
    getJSON('/v1/models', { cache: 'no-store' }),
    getJSON(ADMIN + '/state', { cache: 'no-store' }),
  ]);
  const byId = new Map();
  for (const model of ((r.body && r.body.data) || [])) byId.set(model.id, model);
  for (const id of ((state.body && state.body.advertised_models) || [])) {
    if (!byId.has(id)) byId.set(id, { id, name: id });
  }
  for (const id of Object.keys((state.body && state.body.model_routes) || {})) {
    if (!byId.has(id)) byId.set(id, { id, name: id });
  }
  const models = [...byId.values()];
  apiModelCatalog = models;
  field('nav-model-count').textContent = models.length;
  field('top-model-count').textContent = models.length;
  field('model-test-model-picker').innerHTML = '<option value="">Select a model...</option>' + models.map(model => {
    const label = model.display_name || model.name || model.id;
    const suffix = label === model.id ? '' : ' · ' + model.id;
    return '<option value="' + esc(model.id) + '">' + esc(label + suffix) + '</option>';
  }).join('');
  field('model-test-model-picker').disabled = models.length === 0;
  field('api-model-count').textContent = '(' + models.length + ')';
  if (!field('model-test-model').value && models.length) field('model-test-model').value = models[0].id;
  field('model-test-model-picker').value = [...field('model-test-model-picker').options].some(
    option => option.value === field('model-test-model').value
  ) ? field('model-test-model').value : '';
  field('model-test-status').textContent = models.length + ' advertised model(s)';
  renderApiModelCapabilities();
}
field('refresh-api-models').addEventListener('click', refreshApiModels);
field('model-test-model-picker').addEventListener('change', () => {
  if (field('model-test-model-picker').value) {
    field('model-test-model').value = field('model-test-model-picker').value;
  }
  renderApiModelCapabilities();
});
field('model-test-model').addEventListener('input', renderApiModelCapabilities);
field('model-test-configure').addEventListener('click', () => {
  const modelId = field('model-test-model').value.trim();
  if (modelId && modelConfigCatalog.some(model => model.id === modelId)) {
    selectedModelConfigId = modelId;
    selectedModelConfigIds = [modelId];
    field('model-config-search').value = '';
    renderModelConfigList();
  }
  document.getElementById('model-configurator').scrollIntoView({ behavior: 'smooth' });
});
field('model-test-clear').addEventListener('click', () => {
  field('model-test-answer').textContent = 'No request sent.';
  field('model-test-answer').className = 'muted';
  field('model-test-raw').textContent = '';
  field('model-test-uploaded').innerHTML = '';
  field('model-test-status').textContent = '';
});
field('model-test-cancel').addEventListener('click', async () => {
  if (modelTestController && modelTestRequestId) {
    const controller = modelTestController;
    const requestId = modelTestRequestId;
    field('model-test-status').textContent = 'cancelling... ' +
      ((performance.now() - modelTestStartedAt) / 1000).toFixed(1) + 's';
    await getJSON(ADMIN + '/test-chat/' + encodeURIComponent(requestId), {
      method: 'DELETE',
    }).catch(() => ({}));
    controller.abort();
  }
});
field('model-test-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const model = field('model-test-model').value.trim();
  const prompt = field('model-test-prompt').value;
  if (!model || !prompt.trim()) return;
  const button = field('model-test-send');
  let attachments;
  try {
    attachments = await encodedModelTestAttachments();
  } catch (error) {
    field('model-test-answer').textContent = 'Could not read attachment: ' + error;
    field('model-test-answer').className = 'err';
    return;
  }
  modelTestController = new AbortController();
  modelTestRequestId = (crypto.randomUUID && crypto.randomUUID()) ||
    ('test-' + Date.now() + '-' + Math.random().toString(16).slice(2));
  modelTestStartedAt = performance.now();
  button.disabled = true;
  field('model-test-cancel').disabled = false;
  modelTestTimer = window.setInterval(() => {
    field('model-test-status').textContent = 'waiting... ' +
      ((performance.now() - modelTestStartedAt) / 1000).toFixed(1) + 's';
  }, 100);
  try {
    const r = await getJSON(ADMIN + '/test-chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        request_id: modelTestRequestId,
        model,
        prompt,
        attachments,
        required_capabilities: csv(field('model-test-capabilities').value),
      }),
      signal: modelTestController.signal,
    });
    const elapsed = ((performance.now() - modelTestStartedAt) / 1000).toFixed(1);
    const answer = r.body && r.body.choices && r.body.choices[0] && r.body.choices[0].message
      ? r.body.choices[0].message.content : '';
    const cancelled = r.status === 499;
    field('model-test-answer').textContent = cancelled ? 'Request cancelled.' :
      (answer || (r.ok ? '(empty response)' : (r.body.detail || 'request failed')));
    field('model-test-answer').className = cancelled ? 'muted' : (r.ok ? '' : 'err');
    field('model-test-raw').textContent = JSON.stringify(r.body, null, 2);
    renderUploadedAttachments(r.body.attachments || []);
    field('model-test-status').textContent = cancelled ? ('cancelled · ' + elapsed + 's') :
      ('HTTP ' + r.status + ' · ' + elapsed + 's');
  } catch (error) {
    const elapsed = ((performance.now() - modelTestStartedAt) / 1000).toFixed(1);
    if (error && error.name === 'AbortError') {
      field('model-test-answer').textContent = 'Request cancelled.';
      field('model-test-answer').className = 'muted';
      field('model-test-status').textContent = 'cancelled · ' + elapsed + 's';
    } else {
      field('model-test-answer').textContent = String(error);
      field('model-test-answer').className = 'err';
      field('model-test-status').textContent = 'request failed · ' + elapsed + 's';
    }
  } finally {
    window.clearInterval(modelTestTimer);
    modelTestTimer = null;
    modelTestController = null;
    modelTestRequestId = null;
    button.disabled = false;
    field('model-test-cancel').disabled = true;
  }
});

field('image-generation-run').addEventListener('click', async () => {
  const model = field('image-generation-model').value.trim();
  const prompt = field('image-generation-prompt').value.trim();
  if (!model || !prompt) return;
  field('image-generation-run').disabled = true;
  field('image-generation-status').textContent = 'generating...';
  field('image-generation-status').className = 'msg';
  field('image-generation-preview').style.display = 'none';
  try {
    const response = await getJSON('/v1/images/generations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, prompt, response_format: 'url' }),
    });
    const entry = response.body && response.body.data ? response.body.data[0] : null;
    field('image-generation-raw').textContent = JSON.stringify(response.body, null, 2);
    if (!response.ok || !entry) {
      field('image-generation-status').textContent = 'unavailable · HTTP ' + response.status;
      field('image-generation-status').className = 'msg err';
      field('image-generation-description').textContent =
        JSON.stringify((response.body && response.body.detail) || response.body || '');
      return;
    }
    const source = entry.source || 'simulated';
    field('image-generation-status').textContent = source === 'worker'
      ? 'worker-generated image'
      : 'simulated placeholder';
    field('image-generation-status').className = 'msg ' + (source === 'worker' ? 'ok' : '');
    field('image-generation-description').textContent =
      entry.pretend_description || entry.revised_prompt || '';
    const imageUrl = entry.url || (
      entry.b64_json ? 'data:image/png;base64,' + entry.b64_json : ''
    );
    if (imageUrl) {
      field('image-generation-preview').src = imageUrl;
      field('image-generation-preview').style.display = 'block';
    }
  } catch (error) {
    field('image-generation-status').textContent = 'generation failed';
    field('image-generation-status').className = 'msg err';
    field('image-generation-description').textContent = String(error);
  } finally {
    field('image-generation-run').disabled = false;
  }
});

async function loadConfig() {
  const r = await getJSON(ADMIN + '/config', { cache: 'no-store' });
  document.getElementById('cfg-path').textContent = r.body.path || '';
  const config = r.body.config || {};
  currentConfig = config;
  currentConfigRevision = r.body.revision || '';
  document.getElementById('config').value = JSON.stringify(config, null, 2);
  field('server-description').value = config.description || '';
  field('server-mode').value = Array.isArray(config.mode) ? config.mode.join(',') : (config.mode || '');
  field('server-capability-fallback').value = config.capability_fallback || 'stub';
  field('server-subagent-model').value = config.subagent_model || '';
  field('server-max-concurrent').value = config.max_concurrent_calls ?? 50;
  field('server-idle-workers').value = config.idle_worker_target ?? 5;
  field('server-idle-grace').value = config.idle_grace_seconds ?? 30;
  field('server-backend-delay').value = config.backend_fallback_delay_seconds ?? 5;
  field('server-validation-default').value = config.validation_interval_default ?? config.validation_interval ?? '';
  field('server-validation-override').value = config.validation_interval_override ?? '';
  renderConfigSection();
  setMsg('', '');
}
function setMsg(text, cls) { const m = document.getElementById('cfg-msg'); m.textContent = text; m.className = 'msg ' + cls; }
document.getElementById('reload').addEventListener('click', loadConfig);
field('server-settings-reload').addEventListener('click', loadConfig);
field('server-settings-save').addEventListener('click', async () => {
  const settings = {
    description: field('server-description').value.trim() || null,
    mode: field('server-mode').value.trim() || null,
    capability_fallback: field('server-capability-fallback').value,
    subagent_model: field('server-subagent-model').value.trim() || null,
    max_concurrent_calls: Number(field('server-max-concurrent').value) || 50,
    idle_worker_target: Number(field('server-idle-workers').value) || 0,
    idle_grace_seconds: Number(field('server-idle-grace').value) || 0,
    backend_fallback_delay_seconds: Number(field('server-backend-delay').value) || 0,
    validation_interval_default: field('server-validation-default').value.trim() || null,
    validation_interval_override: field('server-validation-override').value.trim() || null,
    expected_revision: currentConfigRevision,
  };
  const r = await getJSON(ADMIN + '/config/server-settings', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(settings),
  });
  field('server-settings-msg').textContent = r.ok ? 'saved; restart to apply startup settings' : ('save failed (' + r.status + ')');
  field('server-settings-msg').className = 'msg ' + (r.ok ? 'ok' : 'err');
  if (r.ok) {
    currentConfigRevision = r.body.revision;
    await loadConfig();
  }
});
document.getElementById('save').addEventListener('click', async () => {
  let parsed;
  try { parsed = JSON.parse(document.getElementById('config').value); }
  catch (err) { setMsg('invalid JSON: ' + err.message, 'err'); return; }
  const r = await getJSON(ADMIN + '/config', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config: parsed, expected_revision: currentConfigRevision }),
  });
  setMsg(r.ok ? 'saved' : ('save failed (' + r.status + ')'), r.ok ? 'ok' : 'err');
  if (r.ok) await loadConfig();
});

function telemetryTotals(services) {
  const totals = { attempts: 0, served: 0, failed: 0, rejected: 0, deferred: 0, cancelled: 0, total_seconds: 0 };
  for (const stats of Object.values(services || {})) {
    for (const key of ['attempts','served','failed','rejected','deferred','cancelled']) {
      totals[key] += Number(stats[key]) || 0;
    }
    totals.total_seconds += Number(stats.total_seconds) || 0;
  }
  totals.average_seconds = totals.attempts ? totals.total_seconds / totals.attempts : 0;
  return totals;
}
function telemetrySeconds(value) {
  return (Number(value) || 0).toFixed(2) + 's';
}
function sortTelemetryRows(rows, sort) {
  return [...rows].sort((left, right) => {
    const a = left[sort.key];
    const b = right[sort.key];
    const compared = typeof a === 'string'
      ? String(a).localeCompare(String(b), undefined, { numeric: true })
      : (Number(a) || 0) - (Number(b) || 0);
    return compared * sort.direction;
  });
}
function telemetryFooterValues(totals) {
  const mode = field('telemetry-footer-mode').value;
  return {
    count: value => mode === 'averages' ? '—' : value,
    total: mode === 'averages' ? '—' : telemetrySeconds(totals.total_seconds),
    average: mode === 'totals' ? '—' : telemetrySeconds(totals.average_seconds),
    label: mode === 'averages' ? 'Weighted average' : 'Cumulative total',
  };
}
function renderTelemetry(state) {
  latestTelemetryState = state;
  const concurrency = state.concurrency || {};
  const team = state.team_service_stats || { totals: {}, services: {}, models: {} };
  const totals = team.totals || {};
  const waiting = state.waiting_for_worker || [];
  const capacityWaiting = concurrency.client_capacity_waiting || 0;
  const stuckWorkers = state.stuck_workers || [];
  const connectionErrors = state.connection_errors || [];
  field('top-active-count').textContent = concurrency.active_calls || 0;
  field('top-waiting-count').textContent = (concurrency.waiting_for_worker || 0) + capacityWaiting;
  field('nav-waiting-count').textContent = (concurrency.waiting_for_worker || 0) + capacityWaiting;
  field('top-stuck-count').textContent = stuckWorkers.length;
  field('top-served-count').textContent = totals.served || 0;
  field('top-switch-count').textContent = state.team_model_switches || 0;
  field('telemetry-summary').textContent =
    (concurrency.active_calls || 0) + '/' + (concurrency.max_calls || 50) + ' active · ' +
    (concurrency.idle_workers || 0) + ' idle / target ' + (concurrency.idle_worker_target ?? 5) +
    ' · ' + (concurrency.recently_busy_workers || 0) + ' cooling (' +
    (concurrency.idle_grace_seconds ?? 30) + 's)' +
    (concurrency.idle_maintenance_paused ? ' · idle maintenance paused' : '') +
    ' · ' + (concurrency.waiting_for_worker || 0) + ' waiting · backend delay ' +
    (concurrency.backend_fallback_delay_seconds ?? 5) + 's · ' +
    'client max ' + (concurrency.client_worker_limit || 1) + ' · reserve ' +
    (concurrency.client_worker_reserve || 0) + ' · ' + capacityWaiting + ' capacity waiting · ' +
    (state.team_model_switches || 0) + ' model switches';
  field('telemetry-alerts').innerHTML =
    '<b>Stuck workers</b> ' + (stuckWorkers.length
      ? stuckWorkers.map(item => '<code>' + esc(item.worker_id) + '</code> ' +
          esc(item.service_kind) + ' · ' + telemetrySeconds(item.age_seconds) +
          ' · ' + esc(item.model)).join('<br>')
      : '<span class="ok">none</span>') +
    '<br><b>Connection errors</b> ' + (connectionErrors.length
      ? connectionErrors.map(item => '<code>' + esc(item.worker_id) + '</code> ' +
          item.connection_errors + ' · ' + esc(item.last_connection_error || 'disconnected')).join('<br>')
      : '<span class="ok">none</span>');
  const services = Object.entries(team.services || {});
  field('telemetry-services').innerHTML = services.length ? services.map(([kind, stats]) =>
    '<tr><td><code>' + esc(kind) + '</code></td><td>' + (stats.active || 0) +
    '</td><td>' + (stats.attempts || 0) + '</td><td>' + (stats.served || 0) +
    '</td><td>' + (stats.deferred || 0) + '</td><td>' + (stats.rejected || 0) +
    '</td><td>' + ((stats.failed || 0) + (stats.cancelled || 0)) +
    '</td><td>' + telemetrySeconds(stats.total_seconds) +
    '</td><td>' + telemetrySeconds(stats.average_seconds) + '</td></tr>'
  ).join('') : '<tr><td colspan="9" class="muted">No requests served yet.</td></tr>';
  const serviceFooter = telemetryFooterValues(totals);
  field('telemetry-services-total').innerHTML =
    '<tr><th>' + serviceFooter.label + '</th><th>' +
    services.reduce((sum, [, stats]) => sum + (Number(stats.active) || 0), 0) +
    '</th><th>' + serviceFooter.count(totals.attempts || 0) +
    '</th><th>' + serviceFooter.count(totals.served || 0) +
    '</th><th>' + serviceFooter.count(totals.deferred || 0) +
    '</th><th>' + serviceFooter.count(totals.rejected || 0) +
    '</th><th>' + serviceFooter.count((totals.failed || 0) + (totals.cancelled || 0)) +
    '</th><th>' + serviceFooter.total + '</th><th>' + serviceFooter.average + '</th></tr>';
  const workers = Object.entries(state.worker_service_stats || {}).map(([workerId, worker]) => {
    const stats = telemetryTotals(worker.services);
    const switches = (state.worker_model_switches || {})[workerId] || {};
    return {
      id: workerId,
      kind: worker.kind || 'worker',
      active: Number(worker.active) || 0,
      reserved: Number(worker.reserved) || 0,
      attempts: stats.attempts,
      served: stats.served,
      deferred: stats.deferred,
      failed: stats.failed + stats.rejected + stats.cancelled,
      switches: Number(switches.count) || 0,
      switchTitle: (switches.previous_model || '') + ' → ' + (switches.new_model || ''),
      total_seconds: stats.total_seconds,
      average_seconds: stats.average_seconds,
    };
  });
  const visibleWorkers = sortTelemetryRows(workers, telemetryWorkerSort);
  field('telemetry-workers-wrap').classList.toggle(
    'expanded',
    telemetryShowAllWorkers,
  );
  field('telemetry-workers').innerHTML = visibleWorkers.length ? visibleWorkers.map(worker =>
    '<tr><td><code>' + esc(worker.id) + '</code><br><span class="muted">' +
    esc(worker.kind) + '</span></td><td>' + worker.active +
    '</td><td>' + worker.reserved + '</td><td>' +
    (worker.kind === 'backend' && worker.attempts === 0 ? '—' : worker.attempts) +
    '</td><td>' + (worker.kind === 'backend' && worker.attempts === 0 ? '—' : worker.served) +
    '</td><td>' + (worker.kind === 'backend' && worker.attempts === 0 ? '—' : worker.deferred) +
    '</td><td>' + (worker.kind === 'backend' && worker.attempts === 0 ? '—' : worker.failed) +
    '</td><td title="' + esc(worker.switchTitle) + '">' +
    (worker.kind === 'backend' ? '—' : worker.switches) + '</td><td>' +
    (worker.kind === 'backend' && worker.attempts === 0 ? '—' : telemetrySeconds(worker.total_seconds)) +
    '</td><td>' + (worker.kind === 'backend' && worker.attempts === 0 ? '—' : telemetrySeconds(worker.average_seconds)) +
    '</td></tr>'
  ).join('') : '<tr><td colspan="10" class="muted">No worker timing yet.</td></tr>';
  field('telemetry-workers-toggle').textContent = telemetryShowAllWorkers
    ? 'Show four-row scroll'
    : ('Expand all (' + workers.length + ')');
  field('telemetry-workers-toggle').disabled = workers.length <= 4;
  const workerTotals = telemetryTotals(
    Object.fromEntries(workers.map(worker => [worker.id, {
      attempts: worker.attempts, served: worker.served, deferred: worker.deferred,
      failed: worker.failed, rejected: 0, cancelled: 0, total_seconds: worker.total_seconds,
    }]))
  );
  const workerFooter = telemetryFooterValues(workerTotals);
  field('telemetry-workers-total').innerHTML =
    '<tr><th>' + workerFooter.label + '</th><th>' +
    workers.reduce((sum, worker) => sum + worker.active, 0) +
    '</th><th>' + workers.reduce((sum, worker) => sum + worker.reserved, 0) +
    '</th><th>' + workerFooter.count(workerTotals.attempts) +
    '</th><th>' + workerFooter.count(workerTotals.served) +
    '</th><th>' + workerFooter.count(workerTotals.deferred) +
    '</th><th>' + workerFooter.count(workerTotals.failed) +
    '</th><th>' + workerFooter.count(state.team_model_switches || 0) +
    '</th><th>' + workerFooter.total + '</th><th>' + workerFooter.average + '</th></tr>';
  const models = Object.entries(team.models || {}).map(([modelId, model]) => {
    const stats = model.totals || {};
    return {
      id: modelId,
      active: Number(model.active) || 0,
      attempts: Number(stats.attempts) || 0,
      served: Number(stats.served) || 0,
      deferred: Number(stats.deferred) || 0,
      failed: (Number(stats.failed) || 0) + (Number(stats.rejected) || 0) +
        (Number(stats.cancelled) || 0),
      total_seconds: Number(stats.total_seconds) || 0,
      average_seconds: Number(stats.average_seconds) || 0,
    };
  });
  const visibleModels = sortTelemetryRows(models, telemetryModelSort);
  field('telemetry-models-wrap').classList.toggle(
    'expanded',
    telemetryShowAllModels,
  );
  field('telemetry-models').innerHTML = visibleModels.length ? visibleModels.map(model =>
    '<tr><td><code>' + esc(model.id) + '</code></td><td>' + model.active +
    '</td><td>' + model.attempts + '</td><td>' + model.served +
    '</td><td>' + model.deferred + '</td><td>' + model.failed +
    '</td><td>' + telemetrySeconds(model.total_seconds) +
    '</td><td>' + telemetrySeconds(model.average_seconds) + '</td></tr>'
  ).join('') : '<tr><td colspan="8" class="muted">No model timing yet.</td></tr>';
  field('telemetry-models-toggle').textContent = telemetryShowAllModels
    ? 'Show four-row scroll'
    : ('Expand all (' + models.length + ')');
  field('telemetry-models-toggle').disabled = models.length <= 4;
  const modelFooter = telemetryFooterValues(totals);
  field('telemetry-models-total').innerHTML =
    '<tr><th>' + modelFooter.label + '</th><th>' +
    models.reduce((sum, model) => sum + model.active, 0) +
    '</th><th>' + modelFooter.count(totals.attempts || 0) +
    '</th><th>' + modelFooter.count(totals.served || 0) +
    '</th><th>' + modelFooter.count(totals.deferred || 0) +
    '</th><th>' + modelFooter.count(
      (totals.failed || 0) + (totals.rejected || 0) + (totals.cancelled || 0)
    ) + '</th><th>' + modelFooter.total + '</th><th>' + modelFooter.average + '</th></tr>';
  field('telemetry-waiting').innerHTML = waiting.length
    ? '<strong>Waiting client requests</strong>' + waiting.map(item =>
      '<div><code>' + esc(item.model) + '</code> → ' + esc(item.worker_id) +
      ' · ' + telemetrySeconds(item.waiting_seconds) + ' · ' + esc(item.reason) + '</div>'
    ).join('')
    : '';
}
field('telemetry-workers-toggle').addEventListener('click', () => {
  telemetryShowAllWorkers = !telemetryShowAllWorkers;
  renderTelemetry(latestTelemetryState);
});
field('telemetry-models-toggle').addEventListener('click', () => {
  telemetryShowAllModels = !telemetryShowAllModels;
  renderTelemetry(latestTelemetryState);
});
field('telemetry-footer-mode').addEventListener('change', () =>
  renderTelemetry(latestTelemetryState));
field('telemetry').addEventListener('click', event => {
  const button = event.target.closest('button[data-stats-table]'); if (!button) return;
  const sort = button.dataset.statsTable === 'workers'
    ? telemetryWorkerSort
    : telemetryModelSort;
  if (sort.key === button.dataset.statsKey) sort.direction *= -1;
  else {
    sort.key = button.dataset.statsKey;
    sort.direction = button.dataset.statsKey === 'id' ? 1 : -1;
  }
  renderTelemetry(latestTelemetryState);
});

async function tick() {
  try {
    const s = await getJSON(ADMIN + '/state', { cache: 'no-store' });
    document.getElementById('mode').textContent = (s.body && s.body.mode) || 'relay';
    const nextServerPid = s.body && s.body.process ? s.body.process.pid : null;
    if (serverPid && nextServerPid && nextServerPid !== serverPid) {
      location.reload();
      return;
    }
    serverPid = nextServerPid;
    field('top-uptime').textContent = formatDuration(s.body && s.body.uptime_seconds);
    renderTelemetry(s.body || {});
    await Promise.all([
      refreshConfiguredAgents(),
      refreshWebsockets(),
      refreshClients(),
      refreshWorkers(),
      refreshCopilots(),
    ]);
    document.getElementById('updated').textContent = 'updated ' + new Date().toLocaleTimeString();
  } catch (e) { document.getElementById('updated').textContent = 'error'; }
}
function storedPollValue(key, fallback) {
  try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; }
}
function savePollValue(key, value) {
  try { localStorage.setItem(key, value); } catch {}
}
function schedulePoll() {
  clearTimeout(pollTimer);
  const hiddenPolling = field('poll-hidden').checked;
  if (document.hidden && !hiddenPolling) {
    field('poll-status').textContent = 'PAUSED / HIDDEN';
    return;
  }
  const windowMs = Number(field('poll-window').value);
  if (!document.hidden && windowMs > 0 && Date.now() - pollWindowStartedAt >= windowMs) {
    field('poll-status').textContent = 'PAUSED / WAKE';
    return;
  }
  const delay = document.hidden ? POLL_HIDDEN_MS : POLL_VISIBLE_MS;
  field('poll-status').textContent = document.hidden ? 'HIDDEN / 2 MIN' : 'LIVE / 3 SEC';
  pollTimer = setTimeout(runPoll, delay);
}
async function runPoll(wake = false) {
  if (wake) pollWindowStartedAt = Date.now();
  if (pollInFlight) return;
  pollInFlight = true;
  try { await tick(); } finally {
    pollInFlight = false;
    schedulePoll();
  }
}
const pollWindow = field('poll-window');
const savedPollWindow = storedPollValue(POLL_WINDOW_KEY, '120000');
pollWindow.value = Array.from(pollWindow.options).some(option => option.value === savedPollWindow)
  ? savedPollWindow : '120000';
const pollHidden = field('poll-hidden');
pollHidden.checked = storedPollValue(POLL_HIDDEN_KEY, 'false') === 'true';
pollWindow.addEventListener('change', () => {
  savePollValue(POLL_WINDOW_KEY, pollWindow.value);
  runPoll(true);
});
pollHidden.addEventListener('change', () => {
  savePollValue(POLL_HIDDEN_KEY, String(pollHidden.checked));
  runPoll(true);
});
field('poll-wake').addEventListener('click', () => runPoll(true));
document.addEventListener('visibilitychange', () => {
  if (document.hidden) schedulePoll();
  else runPoll(true);
});
loadConfig();
resetCopilotForm();
renderModelTestAttachments();
refreshModelTestSamples();
refreshCopilotModels(false);
refreshApiModels();
refreshModelConfigurator();
refreshBackendConfigs();
refreshCodexSuppliers();
refreshAntiIdle();
runPoll(true);
</script>
</body>
</html>"""


@router.get("/admin/emullm", response_class=HTMLResponse)
@router.get("/emullm/admin", response_class=HTMLResponse)
@router.get("/emullm/", response_class=HTMLResponse)
@router.get("/emullm", response_class=HTMLResponse)
def admin_page() -> HTMLResponse:
    """Operator control page: configure EMULLM and manage model workers."""
    return HTMLResponse(
        _ADMIN_PAGE_HTML,
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )


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
    routes: dict[str, str | list[str]] = Field(default_factory=dict)
    replace: bool = False


@router.get("/admin/emullm/model_routes")
@router.get("/emullm/admin/model_routes")
def admin_get_model_routes() -> dict[str, Any]:
    return {"model_routes": dict(_model_routes)}


@router.post("/admin/emullm/model_routes")
@router.post("/emullm/admin/model_routes")
def admin_set_model_routes(body: ModelRoutesRequest) -> dict[str, Any]:
    """Set model routes at runtime.

    A value may be one exact worker ID or an ordered list of worker-ID globs and
    OpenAI-compatible backend URLs. Merges by default; ``replace: true`` clears
    first, and an empty value removes a route.
    """
    if body.replace:
        _model_routes.clear()
    for mid, route in body.routes.items():
        if not isinstance(mid, str) or not mid:
            continue
        normalized = _normalise_model_route(route)
        if normalized:
            _model_routes[mid] = normalized
        else:
            _model_routes.pop(mid, None)
    return {"model_routes": dict(_model_routes)}


@router.post("/admin/emullm/reset")
@router.post("/emullm/admin/reset")
def admin_reset() -> dict[str, Any]:
    """Deletes every persisted record (files/assistants/threads/fine-tuning
    jobs/events and mailbox configuration/event logs) under the current runtime
    dir. Does not touch a
    connected worker or in-flight relayed requests."""
    removed = {kind: store.clear() for kind, store in _KIND_STORES.items()}
    removed.update(_clear_mailbox_storage())
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


def _record_worker_model_switch(
    worker_id: str,
    previous_model: str,
    new_model: str,
) -> None:
    stats = _worker_model_switch_stats.setdefault(
        worker_id,
        {"count": 0},
    )
    stats.update(
        {
            "count": int(stats["count"]) + 1,
            "previous_model": previous_model,
            "new_model": new_model,
            "changed_at": time.time(),
        }
    )


def _apply_worker_registration(worker_id: str, registration: dict[str, Any]) -> None:
    models = registration.get("models")
    if isinstance(models, dict) and models:
        _worker_models[worker_id] = models
    capabilities = registration.get("capabilities")
    if isinstance(capabilities, dict):
        _worker_capabilities[worker_id] = {str(key): bool(value) for key, value in capabilities.items()}
    role = registration.get("role")
    if isinstance(role, str) and role.strip():
        _worker_roles[worker_id] = role.strip()
    worker_kind = registration.get("worker_kind")
    if isinstance(worker_kind, str) and worker_kind.strip():
        _worker_kinds[worker_id] = worker_kind.strip()
    runtime_model = registration.get("runtime_model")
    if isinstance(runtime_model, str) and runtime_model.strip():
        next_model = runtime_model.strip()
        previous_model = _worker_runtime_models.get(worker_id)
        if previous_model and previous_model != next_model:
            _record_worker_model_switch(
                worker_id,
                previous_model,
                next_model,
            )
        _worker_runtime_models[worker_id] = next_model
    description = registration.get("description")
    if isinstance(description, str) and description.strip():
        _worker_descriptions[worker_id] = description.strip()
    if "modelmasks" in registration:
        _set_worker_model_masks(worker_id, _normalise_model_masks(registration["modelmasks"]))


async def _serve_worker_socket(
    websocket: WebSocket, worker_id: str, model_masks: tuple[str, ...] | None
) -> None:
    """Serve one native worker connection after its query parameters are parsed."""
    try:
        _mailbox_id(worker_id)
    except ValueError:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    connection_id = _register_active_websocket(
        websocket,
        "worker",
        worker_id=worker_id,
        modelmasks=list(model_masks) if model_masks is not None else None,
    )
    _ensure_worker_mailbox(worker_id)
    duplicate = False
    async with _worker_lock:
        duplicate = worker_id in _connected_workers
        if not duplicate:
            _connected_workers[worker_id] = websocket
            _native_worker_ids.add(worker_id)
            _worker_connection_ids[worker_id] = connection_id
            _set_worker_model_masks(worker_id, model_masks)
    if not duplicate:
        _record_worker_connection(worker_id, model_masks)
        _track_worker_socket_frame(
            connection_id,
            "lifecycle",
            {"type": "connected", "modelmasks": list(model_masks or [])},
        )
    try:
        hello: dict[str, Any] = {"type": "hello", "worker_id": worker_id}
        if model_masks is not None:
            hello["modelmasks"] = list(model_masks)
        await _tracked_ws_send_json(websocket, connection_id, hello)
        if duplicate:
            reason = f"duplicate worker_id '{worker_id}' is already connected"
            await _tracked_ws_send_json(
                websocket,
                connection_id,
                {"type": "shutdown", "reason": reason},
            )
            _track_worker_socket_frame(
                connection_id,
                "lifecycle",
                {"type": "duplicate-rejected", "reason": reason},
            )
            return
        while True:
            data = await _tracked_ws_receive_json(websocket, connection_id)
            if isinstance(data, dict) and data.get("type") == "register":
                _apply_worker_registration(worker_id, data)
            elif isinstance(data, dict):
                await _handle_worker_message(
                    worker_id,
                    data,
                    connection_id=connection_id,
                )
    except WebSocketDisconnect:
        pass
    except RuntimeError as exc:
        # Starlette's in-process WebSocket test transport can surface a
        # completed disconnect as this RuntimeError instead of
        # WebSocketDisconnect. Preserve real protocol errors.
        if "disconnect message" not in str(exc):
            raise
    finally:
        disconnected = False
        async with _worker_lock:
            if _connected_workers.get(worker_id) is websocket:
                del _connected_workers[worker_id]
                _native_worker_ids.discard(worker_id)
                _worker_connection_ids.pop(worker_id, None)
                _worker_model_masks.pop(worker_id, None)
                disconnected = True
        _track_worker_socket_frame(
            connection_id,
            "lifecycle",
            {"type": "disconnected"},
        )
        _remove_active_websocket(connection_id)
        if disconnected:
            _record_worker_disconnection(worker_id)


@router.websocket("/emullm/ws")
async def emullm_socket(websocket: WebSocket) -> None:
    """The single native servant socket.

    ``worker_id`` and optional comma-delimited ``modelmasks`` are query
    parameters. Omit ``worker_id`` to receive a generated identity in the
    hello frame, and omit ``modelmasks`` to accept offers for every model.
    """
    worker_id = (websocket.query_params.get("worker_id") or "").strip() or _new_automatic_worker_id()
    raw_masks = websocket.query_params.getlist("modelmasks")
    model_masks = _normalise_model_masks(raw_masks) if raw_masks else None
    await _serve_worker_socket(websocket, worker_id, model_masks)


async def _handle_worker_message(
    worker_id: str,
    data: dict[str, Any],
    *,
    connection_id: str | None = None,
) -> None:
    active_connection_id = connection_id or _worker_connection_ids.get(worker_id)
    if data.get("type") == "keepalive_reply":
        _update_active_websocket(
            active_connection_id or "",
            last_keepalive_duration_ms=data.get("duration_ms"),
            last_keepalive_succeeded=True,
        )
        _mark_active_websocket_satisfied(
            active_connection_id,
            kind="keepalive",
        )
        return
    if data.get("type") == "keepalive_error":
        _update_active_websocket(
            active_connection_id or "",
            last_keepalive_duration_ms=data.get("duration_ms"),
            last_keepalive_succeeded=False,
            last_keepalive_error=data.get("reason"),
        )
        return
    if data.get("type") in (
        "model_changed",
        "model_change_error",
        "keepalive_stats_reset",
        "anti_idle_changed",
    ):
        control_id = str(data.get("id") or "")
        future = _pending_worker_controls.get(control_id)
        if future is not None and not future.done():
            future.set_result(dict(data))
        return
    if data.get("type") == "accept":
        request_id = str(data.get("id") or "")
        if request_id and request_id in _pending:
            _record_relay_accept(worker_id, request_id, _pending_models.get(request_id))
        return
    if data.get("type") in ("not_ready", "retry_later", "defer"):
        request_id = str(data.get("id") or "")
        reason = str(
            data.get("reason")
            or data.get("content")
            or "worker temporarily unavailable"
        )
        try:
            retry_after = float(data.get("retry_after") or 15)
        except (TypeError, ValueError):
            retry_after = 15.0
        retry_after = min(3600.0, max(0.1, retry_after))
        future = _pending.get(request_id)
        if future and not future.done():
            _record_relay_not_ready(
                worker_id,
                request_id,
                reason,
                retry_after,
                _pending_models.get(request_id),
            )
            future.set_exception(
                _WorkerNotReady(worker_id, reason, retry_after)
            )
        return
    if data.get("type") == "reject":
        request_id = str(data.get("id") or "")
        reason = str(data.get("reason") or data.get("content") or "worker declined the request")
        future = _pending.get(request_id)
        if future and not future.done():
            _record_relay_rejection(worker_id, request_id, reason, _pending_models.get(request_id))
            future.set_exception(_WorkerRejected(worker_id, reason))
        return
    if data.get("type") == "reply":
        request_id = str(data.get("id") or "")
        # A two-way worker may return real media alongside (or instead of)
        # text: image_b64 / image_url / mime. Keep the reply structured so
        # image-gen and vision can hand back actual bytes.
        reply: dict[str, Any] = {"content": str(data.get("content") or "")}
        for key in ("image_b64", "image_url", "audio_b64", "audio_url", "mime", "images", "file_id", "file_url"):
            if data.get(key) is not None:
                reply[key] = data[key]
        _record_relay_reply(worker_id, request_id, reply, _pending_models.get(request_id))
        future = _pending.pop(request_id, None)
        if future and not future.done():
            future.set_result(reply)
            _mark_active_websocket_satisfied(
                active_connection_id,
                kind="client",
                client_work=True,
            )


router.include_router(
    _copilot_api.router,
    dependencies=[Depends(_require_local_process_control)],
)

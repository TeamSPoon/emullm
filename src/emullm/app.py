"""Standalone FastAPI app for the emullm relay.

Exposes the emullm router as its own app so it can run on its own port
(see run.py). Point OpenAI-compatible clients at http://<host>/v1 -- no
API key/token required -- and connect workers at
ws://<host>/emullm/ws?worker_id=<worker_id>.

In `auto` mode (EMULLM_MODE=auto) the app also starts its own worker
subprocesses on startup and stops them on shutdown -- see
emullm.supervisor.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path

from fastapi import FastAPI

from . import supervisor as _sup
from .api import router as emullm_router

# Repo root (holds the subagents/ folder), one level above this package.
_BASE_DIR = Path(__file__).resolve().parent.parent


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    from . import api as _api  # local import to avoid any import-order surprises

    config = _sup.load_config(_api._CONFIG_PATH)
    config = _sup.expand_agents(config)
    _api.apply_agent_policies(config)
    prev_mode = _api._SERVER_MODE
    prev_capability_fallback = _api._CAPABILITY_FALLBACK
    if config.get("mode") is not None:
        _api._SERVER_MODE = config["mode"]
    if config.get("capability_fallback") is not None:
        _api._CAPABILITY_FALLBACK = config["capability_fallback"]
    modes = _api._current_modes()
    supervisor = None
    mock_worker_ids: list[str] = []

    mock_workers = config.get("mock_workers")
    if isinstance(mock_workers, list) and mock_workers:
        mock_worker_ids = _api.register_mock_workers(mock_workers)
        print(
            f"[emullm] registered {len(mock_worker_ids)} mock copilot(s): "
            f"{', '.join(mock_worker_ids) or '(none)'}",
            flush=True,
        )

    if "auto" in modes:
        host_ws_url = os.environ.get("EMULLM_HOST_WS_URL", "ws://127.0.0.1:8801")
        env_launch = os.environ.get("EMULLM_SUBAGENT_LAUNCH")
        if env_launch and config.get("subagent_launch") is None:
            config = {**config, "subagent_launch": env_launch}
        specs = _sup.build_specs(_BASE_DIR, host_ws_url, config)
        supervisor = _sup.Supervisor(specs)
        _sup.set_supervisor(supervisor)
        ids = supervisor.start_autostart()
        source = "config.json" if config.get("workers") else "subagents/ discovery"
        print(
            f"[emullm] auto mode ({source}): started {len(ids)} worker(s): "
            f"{', '.join(ids) or '(none found)'}",
            flush=True,
        )
    try:
        yield
    finally:
        if supervisor is not None:
            supervisor.stop_all()
            _sup.set_supervisor(None)
        if mock_worker_ids:
            _api.unregister_mock_workers(mock_worker_ids)
        _api.clear_agent_policies()
        _api._SERVER_MODE = prev_mode
        _api._CAPABILITY_FALLBACK = prev_capability_fallback


app = FastAPI(title="EMULLM relay", lifespan=_lifespan)
app.include_router(emullm_router)

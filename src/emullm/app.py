"""Standalone FastAPI app for the emullm relay.

Exposes the emullm router as its own app so it can run on its own port
(see :mod:`emullm.standalone`). Point OpenAI-compatible clients at
http://<host>/v1 -- no API key/token required -- and connect workers at
ws://<host>/emullm/ws?worker_id=<worker_id>.

In `auto` mode (EMULLM_MODE=auto) the app also starts its own worker
subprocesses on startup and stops them on shutdown -- see
emullm.supervisor.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path

from fastapi import FastAPI

from . import copilot_api as _copilot
from . import supervisor as _sup
from .api import router as emullm_router

# Repo root (holds the subagents/ folder), one level above this package.
_BASE_DIR = Path(__file__).resolve().parent.parent


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    from . import api as _api  # local import to avoid any import-order surprises

    # First-run bootstrap: create the emullm_runtime container (config, logs,
    # metrics, state) and seed the live config before anything reads it.
    _api._paths.ensure_layout()

    restart_handoff = os.environ.pop("EMULLM_RESTART_HANDOFF", "") == "1"
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
    copilot_manager = None
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
    definitions = config.get("headless_copilots")
    if definitions is not None and not isinstance(definitions, list):
        raise ValueError("config headless_copilots must be a list")
    copilot_manager = _copilot.CopilotInstanceManager(
        config_path=_api._CONFIG_PATH,
        runtime_dir=_api._RUNTIME_DIR / "headless_copilots",
        base_dir=_api._PLUGIN_ROOT,
        default_host_ws_url=os.environ.get("EMULLM_HOST_WS_URL", "ws://127.0.0.1:8801"),
        definitions=definitions or [],
        connected=lambda worker_id: worker_id in _api._connected_workers,
    )
    _copilot.set_manager(copilot_manager)
    idle_worker_task: asyncio.Task[None] | None = None
    restart_reconcile_task: asyncio.Task[None] | None = None
    try:
        started_copilots = (
            []
            if restart_handoff
            else copilot_manager.start_autostart()
        )
        if restart_handoff:
            async def reconcile_preserved_workers() -> None:
                await asyncio.sleep(5)
                started = await asyncio.to_thread(
                    copilot_manager.start_autostart
                )
                print(
                    "[emullm] restart handoff: preserved connected servants; "
                    f"started missing: {', '.join(started) or '(none)'}",
                    flush=True,
                )

            restart_reconcile_task = asyncio.create_task(
                reconcile_preserved_workers()
            )
        idle_worker_task = asyncio.create_task(
            _api.maintain_idle_copilot_workers(
                initial_delay_seconds=8 if restart_handoff else 1
            )
        )
        if definitions:
            print(
                f"[emullm] started {len(started_copilots)} headless Copilot servant(s): "
                f"{', '.join(started_copilots) or '(none)'}",
                flush=True,
            )
        yield
    finally:
        if restart_reconcile_task is not None:
            restart_reconcile_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await restart_reconcile_task
        if idle_worker_task is not None:
            idle_worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await idle_worker_task
        if copilot_manager is not None:
            if not _api._process_control.restart_in_progress():
                for instance in copilot_manager.list():
                    worker_id = str(instance["worker_id"])
                    if instance.get("connected"):
                        with contextlib.suppress(Exception):
                            await _api._shutdown_connected_worker(
                                worker_id,
                                "server shutdown",
                            )
                copilot_manager.stop_all()
            _copilot.set_manager(None)
        if supervisor is not None:
            supervisor.stop_all()
            _sup.set_supervisor(None)
        registered_mock_ids = [
            worker_id
            for worker_id, worker in list(_api._connected_workers.items())
            if isinstance(worker, _api._MockWorker)
        ]
        if registered_mock_ids:
            _api.unregister_mock_workers(registered_mock_ids)
        _api.clear_agent_policies()
        _api._SERVER_MODE = prev_mode
        _api._CAPABILITY_FALLBACK = prev_capability_fallback


app = FastAPI(title="EMULLM relay", lifespan=_lifespan)
app.include_router(emullm_router)

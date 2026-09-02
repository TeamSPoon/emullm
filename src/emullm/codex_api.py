"""Headless Codex servant configuration, lifecycle, and admin API.

The counterpart to :mod:`emullm.copilot_api`, but for the OpenAI Codex CLI.
Each configured servant is a small adapter process (:mod:`emullm.codex_servant`)
that connects to EMULLM's shared ``/emullm/ws`` endpoint as a
``worker-codex-*`` worker and answers relayed requests by running ``codex exec``
non-interactively. Every servant gets an isolated ``CODEX_HOME`` (its own
``config.toml`` + auth + sessions) so multiple codex workers can back different
LAN models at once without colliding.

Codex, unlike the resident Copilot SDK, is a full agentic harness: each request
runs the harness once in the servant's workspace with its own tools. EMULLM
never executes anything itself -- it only relays what Codex returns.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .supervisor import Supervisor, WorkerSpec

router = APIRouter()

# Kept in sync with emullm.codex_servant.DEFAULT_SYSTEM_PROMPT. Defined locally
# (rather than imported) so the codex_servant subprocess -- launched via
# ``python -m emullm.codex_servant`` -- is not pulled into sys.modules at
# package import time (which would trigger a runpy re-import warning), matching
# how copilot_api references copilot_servant only by module name.
DEFAULT_SYSTEM_PROMPT = (
    "Act as the model requested by the OpenAI-compatible caller. Answer the "
    "request directly and return only the assistant response that should be "
    "sent to the caller. You MAY use your own tools and internal abilities "
    "(shell, file inspection, retrieval, reasoning) to produce the answer, and "
    "the caller need not know how you did it. But if the request lists the "
    "CALLER'S tools and asks you to emit tool calls, do NOT execute those "
    "caller tools yourself -- they run in the caller's environment; instead "
    "respond with exactly the requested tool-call JSON envelope and nothing "
    "else, so the caller can run them."
)

_IMPORT_ROOT = Path(__file__).resolve().parent.parent
# ``.../workbench/plugins/emullm`` -> sibling ``codex_cli`` plugin holds the
# vendored Codex CLI install (node_modules/@openai/codex/bin/codex.js).
_PLUGIN_ROOT = _IMPORT_ROOT.parent
_SIBLING_CODEX_ENTRY = (
    _PLUGIN_ROOT.parent
    / "codex_cli"
    / "node_modules"
    / "@openai"
    / "codex"
    / "bin"
    / "codex.js"
)

_SANDBOX_MODES = ("read-only", "workspace-write", "danger-full-access")
_WIRE_APIS = ("responses", "chat")


# --------------------------------------------------------------------------- #
# Codex launch resolution
# --------------------------------------------------------------------------- #
def resolve_node_command(configured: str | None = None) -> str | None:
    if configured and configured.strip():
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        found = shutil.which(configured)
        if found:
            return found
    return shutil.which("node")


def resolve_codex_entry(configured: str | None = None) -> Path | None:
    """Resolve the Codex Node entrypoint (codex.js) for node execution.

    Order: explicit config -> ``EMULLM_CODEX_ENTRY`` -> the sibling codex_cli
    plugin's vendored install. Returns ``None`` when no Node entrypoint is
    found (a native ``codex`` binary may still be usable, see
    :func:`resolve_codex_command`).
    """

    candidates: list[Path] = []
    if configured and configured.strip():
        candidates.append(Path(configured).expanduser())
    env_entry = os.environ.get("EMULLM_CODEX_ENTRY")
    if env_entry:
        candidates.append(Path(env_entry).expanduser())
    candidates.append(_SIBLING_CODEX_ENTRY)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_codex_command(configured: str | None = None) -> str | None:
    """Resolve a native ``codex`` binary on PATH (fallback when no codex.js)."""

    if configured and configured.strip():
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        found = shutil.which(configured)
        if found:
            return found
    return shutil.which("codex") or shutil.which("codex.cmd")


def codex_available(config: "HeadlessCodexConfig | None" = None) -> dict[str, Any]:
    """Describe how a Codex servant would be launched, for the admin UI."""

    node = resolve_node_command(config.node_command if config else None)
    entry = resolve_codex_entry(config.codex_entry if config else None)
    native = resolve_codex_command(config.codex_command if config else None)
    node_launch = bool(node and entry)
    return {
        "available": bool(node_launch or native),
        "node_command": node,
        "codex_entry": str(entry) if entry else None,
        "codex_command": native,
        "launch": "node" if node_launch else ("native" if native else None),
    }


# --------------------------------------------------------------------------- #
# Config document helpers (shared config.json under the emullm runtime)
# --------------------------------------------------------------------------- #
_CONFIG_IO_LOCK = threading.RLock()


def _read_config_document(path: Path) -> dict[str, Any]:
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(data, dict):
                return data
    except (OSError, ValueError):
        pass
    return {}


def _write_config_document(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _update_config_document(
    path: Path, mutate: Callable[[dict[str, Any]], None]
) -> dict[str, Any]:
    with _CONFIG_IO_LOCK:
        document = _read_config_document(path)
        mutate(document)
        _write_config_document(path, document)
        return document


# --------------------------------------------------------------------------- #
# Configuration model
# --------------------------------------------------------------------------- #
class HeadlessCodexConfig(BaseModel):
    """Configuration for one non-interactive Codex CLI servant."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    worker_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
        description="Stable EMULLM worker identity (e.g. worker-codex-1).",
    )
    session_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    modelmasks: list[str] = Field(
        default_factory=list,
        description="OpenAI-compatible model glob patterns handled by this servant; empty accepts all.",
    )
    role: str = Field(default="headless-codex", min_length=1, max_length=100)
    capabilities: list[str] = Field(
        default_factory=list,
        description="Declared capabilities; prefix a name with ! or - to decline it.",
    )
    system_prompt: str = Field(default=DEFAULT_SYSTEM_PROMPT, min_length=1, max_length=20_000)
    cwd: str | None = Field(default=None, description="Workspace root Codex runs in; defaults to an isolated per-worker dir.")
    host_ws_url: str | None = None
    autostart: bool = True

    # --- backing model (written into CODEX_HOME/config.toml when base_url set) ---
    model: str | None = Field(default=None, max_length=200, description="Model Codex should use (codex -m / config.toml model).")
    base_url: str | None = Field(
        default=None,
        max_length=2_000,
        description="OpenAI-compatible base URL Codex points at (any LAN model). Leave blank to use the CODEX_HOME login as-is.",
    )
    wire_api: str = Field(default="responses", description="Codex model_providers wire_api: responses or chat.")
    provider_id: str = Field(default="lan", max_length=100)
    provider_label: str = Field(default="LAN", max_length=200)
    env_key: str = Field(default="CODEX_API_KEY", max_length=200)
    api_key: str | None = Field(default=None, max_length=4_000, description="Value exported for env_key; defaults to a local placeholder.")

    # --- sandbox / execution controls ---
    sandbox_mode: str = Field(default="workspace-write")
    full_auto: bool = Field(
        default=True,
        description="Run codex exec with --dangerously-bypass-approvals-and-sandbox (unattended worker). Disable to enforce sandbox_mode.",
    )
    ephemeral: bool = Field(default=False, description="Run codex exec with --ephemeral (do not persist session files).")
    extra_config: list[str] = Field(default_factory=list, description="Extra `codex -c key=value` overrides.")

    codex_entry: str | None = Field(default=None, description="Override path to codex.js (Node entrypoint).")
    codex_command: str | None = Field(default=None, description="Override native codex binary (used if no Node entrypoint).")
    node_command: str | None = None

    # --- timing / limits ---
    idle_timeout_seconds: float = Field(default=15.0, ge=1.0, le=600.0)
    reply_timeout_seconds: float = Field(default=1800.0, ge=5.0, le=86_400.0)
    reconnect_seconds: float = Field(default=2.0, ge=0.1, le=300.0)
    max_prompt_chars: int = Field(default=4_000_000, ge=1_000, le=20_000_000)

    @field_validator("modelmasks", "capabilities", "extra_config")
    @classmethod
    def _clean_string_list(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    @field_validator("host_ws_url")
    @classmethod
    def _validate_ws(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith(("ws://", "wss://")):
            raise ValueError("host_ws_url must start with ws:// or wss://")
        return value

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str | None) -> str | None:
        if value is not None and value and not value.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return value

    @field_validator("sandbox_mode")
    @classmethod
    def _validate_sandbox(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in _SANDBOX_MODES:
            raise ValueError(f"sandbox_mode must be one of {_SANDBOX_MODES}")
        return normalized

    @field_validator("wire_api")
    @classmethod
    def _validate_wire_api(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in _WIRE_APIS:
            raise ValueError(f"wire_api must be one of {_WIRE_APIS}")
        return normalized


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class CodexInstanceError(RuntimeError):
    pass


class CodexInstanceExists(CodexInstanceError):
    pass


class CodexInstanceMissing(CodexInstanceError):
    pass


class CodexInstanceRunning(CodexInstanceError):
    pass


# --------------------------------------------------------------------------- #
# config.toml rendering (matches the codex_cli plugin's format)
# --------------------------------------------------------------------------- #
def _toml_str(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_config_toml(config: HeadlessCodexConfig) -> str:
    lines = [
        f"# Generated by EMULLM for headless Codex worker '{config.worker_id}'.",
        "# base_url points Codex at the chosen OpenAI-compatible LAN model.",
        "",
    ]
    if config.model:
        lines.append(f"model = {_toml_str(config.model)}")
    provider = config.provider_id or "lan"
    lines.append(f"model_provider = {_toml_str(provider)}")
    lines.append("")
    lines.append(f"[model_providers.{provider}]")
    lines.append(f"name = {_toml_str(config.provider_label or provider)}")
    lines.append(f"base_url = {_toml_str(config.base_url or '')}")
    lines.append(f"wire_api = {_toml_str(config.wire_api)}")
    if config.env_key:
        lines.append(f"env_key = {_toml_str(config.env_key)}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Instance manager
# --------------------------------------------------------------------------- #
class CodexInstanceManager:
    """Persist and supervise non-interactive Codex CLI servant processes."""

    def __init__(
        self,
        *,
        config_path: Path,
        runtime_dir: Path,
        base_dir: Path,
        default_host_ws_url: str,
        definitions: list[dict[str, Any] | HeadlessCodexConfig] | None = None,
        connected: Callable[[str], bool] | None = None,
        spawn: Callable[[WorkerSpec], Any] | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.runtime_dir = Path(runtime_dir)
        self.base_dir = Path(base_dir)
        self.default_host_ws_url = default_host_ws_url.rstrip("/")
        self._connected = connected or (lambda _worker_id: False)
        self._spawn_override = spawn
        self._lock = threading.RLock()
        self._configs: dict[str, HeadlessCodexConfig] = {}
        self._supervisor = Supervisor([], spawn=self._spawn)

        for raw_definition in definitions or []:
            if isinstance(raw_definition, HeadlessCodexConfig):
                config = raw_definition
            else:
                config = HeadlessCodexConfig.model_validate(raw_definition)
            if config.worker_id in self._configs:
                raise ValueError(f"duplicate headless Codex worker_id '{config.worker_id}'")
            self._configs[config.worker_id] = config
            self._supervisor.add_spec(self._spec_for(config))

    # -- paths --------------------------------------------------------------
    def _instance_dir(self, worker_id: str) -> Path:
        return self.runtime_dir / worker_id

    def _working_dir(self, config: HeadlessCodexConfig) -> Path:
        if config.cwd:
            configured = Path(config.cwd).expanduser()
            path = configured if configured.is_absolute() else self.base_dir / configured
        else:
            path = self._instance_dir(config.worker_id) / "workspace"
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    def _codex_home(self, worker_id: str) -> Path:
        path = self._instance_dir(worker_id) / "codex_home"
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    def _runtime_config_path(self, worker_id: str) -> Path:
        return self._instance_dir(worker_id) / "servant-config.json"

    def _runtime_status_path(self, worker_id: str) -> Path:
        return self._instance_dir(worker_id) / "runtime-status.json"

    def _log_path(self, worker_id: str) -> Path:
        return self._instance_dir(worker_id) / "servant.log"

    # -- spec / runtime config ---------------------------------------------
    def _spec_for(self, config: HeadlessCodexConfig) -> WorkerSpec:
        return WorkerSpec(
            worker_id=config.worker_id,
            argv=[
                sys.executable,
                "-m",
                "emullm.codex_servant",
                "--config",
                str(self._runtime_config_path(config.worker_id)),
            ],
            cwd=self._working_dir(config),
            role=config.role,
            autostart=config.autostart,
            modelmasks=",".join(config.modelmasks),
        )

    def _resolve_launch(self, config: HeadlessCodexConfig) -> dict[str, Any]:
        node = resolve_node_command(config.node_command)
        entry = resolve_codex_entry(config.codex_entry)
        native = resolve_codex_command(config.codex_command)
        if not ((node and entry) or native):
            raise FileNotFoundError(
                "Codex CLI was not found. Install it via the codex_cli plugin "
                "(npm install) or set codex_entry/codex_command."
            )
        return {
            "node_command": node or "",
            "codex_entry": str(entry) if entry else "",
            "codex_command": native or "",
        }

    def _write_runtime_config(self, config: HeadlessCodexConfig) -> None:
        instance_dir = self._instance_dir(config.worker_id)
        instance_dir.mkdir(parents=True, exist_ok=True)
        codex_home = self._codex_home(config.worker_id)
        # Only (re)write config.toml when a backing endpoint is configured, so
        # an operator-provided CODEX_HOME login is left untouched otherwise.
        if config.base_url:
            (codex_home / "config.toml").write_text(
                render_config_toml(config), encoding="utf-8"
            )
        launch = self._resolve_launch(config)
        payload = {
            "worker_id": config.worker_id,
            "session_id": str(config.session_id),
            "host_ws_url": config.host_ws_url or self.default_host_ws_url,
            "role": config.role,
            "modelmasks": list(config.modelmasks),
            "capabilities": list(config.capabilities),
            "system_prompt": config.system_prompt,
            "node_command": launch["node_command"],
            "codex_entry": launch["codex_entry"],
            "codex_command": launch["codex_command"],
            "resolved_cwd": str(self._working_dir(config)),
            "codex_home": str(codex_home),
            "env_key": config.env_key,
            "api_key": config.api_key or "emullm-local",
            "selected_model": config.model or "",
            "sandbox_mode": config.sandbox_mode,
            "full_auto": config.full_auto,
            "ephemeral": config.ephemeral,
            "extra_config": list(config.extra_config),
            "idle_timeout_seconds": config.idle_timeout_seconds,
            "reply_timeout_seconds": config.reply_timeout_seconds,
            "reconnect_seconds": config.reconnect_seconds,
            "max_prompt_chars": config.max_prompt_chars,
            "runtime_status_path": str(self._runtime_status_path(config.worker_id).resolve()),
        }
        _write_config_document(self._runtime_config_path(config.worker_id), payload)

    def _spawn(self, spec: WorkerSpec) -> Any:
        config = self._configs[spec.worker_id]
        self._write_runtime_config(config)
        if self._spawn_override is not None:
            return self._spawn_override(spec)
        log_path = self._log_path(spec.worker_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(_IMPORT_ROOT), env.get("PYTHONPATH", "")) if value
        )
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True
        with log_path.open("ab", buffering=0) as log:
            return subprocess.Popen(  # noqa: S603 - argv is generated without a shell
                spec.argv,
                cwd=str(spec.cwd) if spec.cwd else None,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                close_fds=True,
                **kwargs,
            )

    def _persist_locked(self) -> None:
        def update(document: dict[str, Any]) -> None:
            document["headless_codexes"] = [
                config.model_dump(mode="json") for config in self._configs.values()
            ]

        _update_config_document(self.config_path, update)

    def _require(self, worker_id: str) -> HeadlessCodexConfig:
        try:
            return self._configs[worker_id]
        except KeyError as error:
            raise CodexInstanceMissing(f"no headless Codex servant '{worker_id}'") from error

    # -- CRUD ---------------------------------------------------------------
    def create(self, config: HeadlessCodexConfig, *, start: bool = True) -> dict[str, Any]:
        with self._lock:
            if config.worker_id in self._configs:
                raise CodexInstanceExists(
                    f"headless Codex servant '{config.worker_id}' already exists"
                )
            if start:
                self._resolve_launch(config)
            self._configs[config.worker_id] = config
            self._supervisor.add_spec(self._spec_for(config))
            try:
                if start:
                    self._supervisor.start(config.worker_id)
                self._persist_locked()
            except Exception:
                self._supervisor.remove_spec(config.worker_id)
                self._configs.pop(config.worker_id, None)
                raise
            return self.get(config.worker_id)

    def update(
        self, worker_id: str, config: HeadlessCodexConfig, *, restart: bool = True
    ) -> dict[str, Any]:
        if worker_id != config.worker_id:
            raise ValueError("path worker_id must match config.worker_id")
        with self._lock:
            previous = self._require(worker_id)
            was_running = self._is_running(worker_id)
            if was_running and not restart:
                raise CodexInstanceRunning(
                    "a running servant must be restarted to apply configuration"
                )
            if was_running:
                self._supervisor.stop(worker_id)
            self._supervisor.remove_spec(worker_id)
            self._configs[worker_id] = config
            self._supervisor.add_spec(self._spec_for(config))
            try:
                if was_running:
                    self._resolve_launch(config)
                    self._supervisor.start(worker_id)
                self._persist_locked()
            except Exception:
                self._supervisor.remove_spec(worker_id)
                self._configs[worker_id] = previous
                self._supervisor.add_spec(self._spec_for(previous))
                if was_running:
                    self._supervisor.start(worker_id)
                raise
            return self.get(worker_id)

    def delete(self, worker_id: str) -> dict[str, Any]:
        with self._lock:
            self._require(worker_id)
            self._graceful_stop(worker_id)
            self._supervisor.remove_spec(worker_id)
            self._configs.pop(worker_id)
            self._persist_locked()
            return {"worker_id": worker_id, "deleted": True}

    def start(self, worker_id: str) -> dict[str, Any]:
        with self._lock:
            config = self._require(worker_id)
            if self._is_running(worker_id):
                return {**self.get(worker_id), "started": False}
            self._resolve_launch(config)
            started = self._supervisor.start(worker_id)
            return {**self.get(worker_id), "started": started}

    def stop(self, worker_id: str) -> dict[str, Any]:
        with self._lock:
            self._require(worker_id)
            stopped = self._graceful_stop(worker_id)
            return {**self.get(worker_id), "stopped": stopped}

    def restart(self, worker_id: str) -> dict[str, Any]:
        with self._lock:
            config = self._require(worker_id)
            self._resolve_launch(config)
            self._graceful_stop(worker_id)
            started = self._supervisor.start(worker_id)
            return {**self.get(worker_id), "restarted": started}

    def reset_session(self, worker_id: str) -> dict[str, Any]:
        with self._lock:
            config = self._require(worker_id)
            replacement = config.model_copy(update={"session_id": uuid.uuid4()})
            return self.update(worker_id, replacement, restart=True)

    # -- lifecycle helpers --------------------------------------------------
    def _is_running(self, worker_id: str) -> bool:
        row = next(
            (row for row in self._supervisor.status() if row["worker_id"] == worker_id),
            None,
        )
        running = bool(row and row["running"])
        return running or self._connected(worker_id)

    def _graceful_stop(self, worker_id: str) -> bool:
        spec = next(
            (spec for spec in self._supervisor.specs() if spec.worker_id == worker_id),
            None,
        )
        if spec is None:
            return False
        process = spec.process
        if process is None or process.poll() is not None:
            return False
        if (
            os.name == "nt"
            and hasattr(signal, "CTRL_BREAK_EVENT")
            and hasattr(process, "send_signal")
        ):
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT)
                process.wait(timeout=15)
                spec.process = None
                return True
            except (OSError, subprocess.TimeoutExpired):
                pass
        return self._supervisor.stop(worker_id)

    def start_autostart(self) -> list[str]:
        started: list[str] = []
        with self._lock:
            for config in self._configs.values():
                if not config.autostart:
                    continue
                if self._connected(config.worker_id) or self._is_running(config.worker_id):
                    continue
                try:
                    self._resolve_launch(config)
                except FileNotFoundError:
                    continue
                if self._supervisor.start(config.worker_id):
                    started.append(config.worker_id)
        return started

    def stop_all(self) -> None:
        with self._lock:
            for worker_id in list(self._configs):
                self._graceful_stop(worker_id)

    # -- introspection ------------------------------------------------------
    def _runtime_status(self, worker_id: str) -> dict[str, Any]:
        path = self._runtime_status_path(worker_id)
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    def get(self, worker_id: str) -> dict[str, Any]:
        with self._lock:
            config = self._require(worker_id)
            process = next(
                (row for row in self._supervisor.status() if row["worker_id"] == worker_id),
                {"running": False, "pid": None, "returncode": None},
            )
            connected = bool(self._connected(worker_id))
            running = bool(process["running"] or connected)
            runtime = self._runtime_status(worker_id)
            if not running and runtime:
                runtime = {**runtime, "running": False}
            return {
                "worker_id": worker_id,
                "kind": "headless-codex",
                "running": running,
                "connected": connected,
                "external": bool(connected and not process["running"]),
                "pid": (
                    runtime.get("adapter_pid")
                    if runtime and runtime.get("adapter_pid")
                    else process["pid"]
                ),
                "launcher_pid": process["pid"],
                "returncode": process["returncode"],
                "session_id": str(config.session_id),
                "model": config.model,
                "base_url": config.base_url,
                "wire_api": config.wire_api,
                "sandbox_mode": config.sandbox_mode,
                "full_auto": config.full_auto,
                "modelmasks": list(config.modelmasks),
                "role": config.role,
                "autostart": config.autostart,
                "cwd": str(self._working_dir(config)),
                "codex_home": str(self._codex_home(worker_id)),
                "log_path": str(self._log_path(worker_id).resolve()),
                "runtime": runtime,
                "config": config.model_dump(mode="json"),
            }

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self.get(worker_id) for worker_id in self._configs]

    def next_worker_id(self) -> str:
        with self._lock:
            used = set(self._configs)
        index = 1
        while f"worker-codex-{index}" in used:
            index += 1
        return f"worker-codex-{index}"

    def tail_log(self, worker_id: str, lines: int) -> str:
        with self._lock:
            self._require(worker_id)
            path = self._log_path(worker_id)
        if not path.exists():
            return ""
        try:
            return "\n".join(
                path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
            )
        except OSError as error:
            raise CodexInstanceError(f"cannot read servant log '{path}': {error}") from error


# --------------------------------------------------------------------------- #
# Module-level manager singleton + status
# --------------------------------------------------------------------------- #
_manager: CodexInstanceManager | None = None
_manager_status_lock = threading.RLock()
_manager_status_source: object | None = None
_manager_status_at = 0.0
_manager_status_cache: list[dict[str, Any]] = []
_MANAGER_STATUS_CACHE_SECONDS = 0.5


def set_manager(manager: CodexInstanceManager | None) -> None:
    global _manager, _manager_status_source, _manager_status_at, _manager_status_cache
    _manager = manager
    with _manager_status_lock:
        _manager_status_source = None
        _manager_status_at = 0.0
        _manager_status_cache = []


def get_manager() -> CodexInstanceManager | None:
    return _manager


def manager_status() -> list[dict[str, Any]]:
    global _manager_status_source, _manager_status_at, _manager_status_cache
    manager = get_manager()
    if manager is None:
        return []
    now = time.monotonic()
    with _manager_status_lock:
        if (
            manager is _manager_status_source
            and now - _manager_status_at < _MANAGER_STATUS_CACHE_SECONDS
        ):
            return _manager_status_cache
        _manager_status_cache = manager.list()
        _manager_status_source = manager
        _manager_status_at = time.monotonic()
        return _manager_status_cache


def _require_manager() -> CodexInstanceManager:
    if _manager is None:
        raise HTTPException(status_code=409, detail="headless Codex manager is not active")
    return _manager


def _map_error(error: Exception) -> HTTPException:
    if isinstance(error, CodexInstanceMissing):
        return HTTPException(status_code=404, detail=str(error))
    if isinstance(error, (CodexInstanceExists, CodexInstanceRunning)):
        return HTTPException(status_code=409, detail=str(error))
    if isinstance(error, (ValueError, FileNotFoundError)):
        return HTTPException(status_code=422, detail=str(error))
    if isinstance(error, (OSError, subprocess.SubprocessError)):
        return HTTPException(status_code=503, detail=str(error))
    if isinstance(error, CodexInstanceError):
        return HTTPException(status_code=500, detail=str(error))
    return HTTPException(status_code=500, detail=f"{type(error).__name__}: {error}")


# --------------------------------------------------------------------------- #
# Admin API (mounted under the main emullm router)
# --------------------------------------------------------------------------- #
@router.get("/admin/emullm/codexes")
@router.get("/emullm/admin/codexes")
def list_codexes() -> dict[str, Any]:
    manager = get_manager()
    availability = codex_available()
    return {
        "manager_active": manager is not None,
        "codex_available": availability["available"],
        "codex_launch": availability,
        "next_worker_id": manager.next_worker_id() if manager else "worker-codex-1",
        "instances": manager_status(),
    }


@router.get("/admin/emullm/codexes/schema")
@router.get("/emullm/admin/codexes/schema")
def codex_config_schema() -> dict[str, Any]:
    return HeadlessCodexConfig.model_json_schema()


@router.get("/admin/emullm/codexes/{worker_id}")
@router.get("/emullm/admin/codexes/{worker_id}")
def get_codex(worker_id: str) -> dict[str, Any]:
    try:
        return _require_manager().get(worker_id)
    except Exception as error:
        raise _map_error(error) from error


@router.post("/admin/emullm/codexes")
@router.post("/emullm/admin/codexes")
def create_codex(config: HeadlessCodexConfig, start: bool = True) -> dict[str, Any]:
    try:
        return _require_manager().create(config, start=start)
    except Exception as error:
        raise _map_error(error) from error


@router.put("/admin/emullm/codexes/{worker_id}")
@router.put("/emullm/admin/codexes/{worker_id}")
def update_codex(
    worker_id: str, config: HeadlessCodexConfig, restart: bool = True
) -> dict[str, Any]:
    try:
        return _require_manager().update(worker_id, config, restart=restart)
    except Exception as error:
        raise _map_error(error) from error


@router.delete("/admin/emullm/codexes/{worker_id}")
@router.delete("/emullm/admin/codexes/{worker_id}")
def delete_codex(worker_id: str) -> dict[str, Any]:
    try:
        return _require_manager().delete(worker_id)
    except Exception as error:
        raise _map_error(error) from error


def _codex_action(worker_id: str, action: str) -> dict[str, Any]:
    manager = _require_manager()
    try:
        return getattr(manager, action)(worker_id)
    except Exception as error:
        raise _map_error(error) from error


@router.post("/admin/emullm/codexes/{worker_id}/start")
@router.post("/emullm/admin/codexes/{worker_id}/start")
def start_codex(worker_id: str) -> dict[str, Any]:
    return _codex_action(worker_id, "start")


@router.post("/admin/emullm/codexes/{worker_id}/stop")
@router.post("/emullm/admin/codexes/{worker_id}/stop")
def stop_codex(worker_id: str) -> dict[str, Any]:
    return _codex_action(worker_id, "stop")


@router.post("/admin/emullm/codexes/{worker_id}/restart")
@router.post("/emullm/admin/codexes/{worker_id}/restart")
def restart_codex(worker_id: str) -> dict[str, Any]:
    return _codex_action(worker_id, "restart")


@router.post("/admin/emullm/codexes/{worker_id}/reset-session")
@router.post("/emullm/admin/codexes/{worker_id}/reset-session")
def reset_codex_session(worker_id: str) -> dict[str, Any]:
    return _codex_action(worker_id, "reset_session")


@router.get("/admin/emullm/codexes/{worker_id}/log", response_class=PlainTextResponse)
@router.get("/emullm/admin/codexes/{worker_id}/log", response_class=PlainTextResponse)
def codex_log(worker_id: str, lines: int = 200) -> str:
    if not 1 <= lines <= 5_000:
        raise HTTPException(status_code=422, detail="lines must be between 1 and 5000")
    try:
        return _require_manager().tail_log(worker_id, lines)
    except Exception as error:
        raise _map_error(error) from error

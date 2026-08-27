"""Worker process supervisor.

In `auto` mode the relay starts its own worker subprocesses (the
"subagents") instead of waiting for someone to run them by hand. This
module owns that: it discovers worker specs (by default one per
``subagents/emullm_worker_*`` folder), launches each as a subprocess in
its own working directory, and can stop/restart them and report status.

The launcher is injectable (``Supervisor(spawn=...)``) so tests can drive
the lifecycle without actually starting processes, and so a deployment can
swap the default ``python -m emullm.worker`` command for the Copilot CLI
(``copilot``) or anything else via a spec's ``argv``.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class WorkerSpec:
    """How to launch one managed worker."""

    worker_id: str
    argv: list[str]
    cwd: Optional[Path] = None
    role: str = "trusted"
    autostart: bool = True
    modelmasks: str = ""
    # non-init runtime handle for the running process (set by the supervisor)
    process: Any = field(default=None, init=False, repr=False)


def default_worker_argv(
    worker_id: str, host_ws_url: str, role: str = "trusted", modelmasks: str = ""
) -> list[str]:
    """The plain worker-loop launch command: this package's own worker loop,
    writing its handoff files into the (per-worker) current directory. Used
    for explicit config workers that don't specify a ``launch``."""
    argv = [
        sys.executable,
        "-m",
        "emullm.worker",
        "--worker-id",
        worker_id,
        "--host-ws-url",
        host_ws_url,
        "--request-file",
        "request.json",
        "--reply-file",
        "reply.json",
        "--role",
        role,
    ]
    if modelmasks:
        argv += ["--modelmasks", modelmasks]
    return argv


# Kickoff prompt for a Copilot-agent worker launched in a subagent folder.
# Copilot auto-loads that folder's AGENTS.md as context, so the prompt just
# tells it to follow it and keep the worker loop alive.
DEFAULT_SUBAGENT_PROMPT = (
    "Read AGENTS.md in this folder and act as an emullm worker: start your "
    "worker loop (python -m emullm.worker with your worker_id and this "
    "folder's request.json/reply.json) and keep answering relayed requests as "
    "the model until stopped."
)


def copilot_launch_argv(prompt: str = DEFAULT_SUBAGENT_PROMPT, model: str | None = None) -> list[str]:
    """Launch the GitHub Copilot CLI as an agent worker in a subagent folder.
    It reads the folder's AGENTS.md and runs the worker loop per the prompt.

    A managed worker runs fully unattended, so it needs broad permissions to do
    the "insecure" things a worker legitimately does, with no prompt to a human:
      * run shell (its worker loop + python)     -> --allow-all-tools
      * read/write files outside its own folder  -> --allow-all-paths
        (the shared venv, the request/reply handoff, cloud-file blobs on disk)
      * fetch URLs                               -> --allow-all-urls
        (pull cloud-file media handed to it for vision/transcription jobs)
    ``--allow-all`` is the single umbrella for all three (same as ``--yolo``);
    ``--no-ask-user`` stops it ever blocking on a question. ``model`` (from a
    worker's config, e.g. ``"model": "claude-sonnet-4.5"``) picks the AI model
    the student runs as; omit it to let Copilot choose ('auto'). See ``copilot
    --help`` for the flag reference."""
    argv = ["copilot", "--allow-all", "--no-ask-user"]
    if model:
        argv += ["--model", str(model)]
    argv += ["-p", prompt]
    return argv


# --- Worker launch types -------------------------------------------------
#
# A subagent folder can be turned into a worker in a few different ways. We
# name the ways so config can pick one by keyword:
#
#   * "copilot" (default) -- an **auto-configured** agent: the supervisor
#     spawns the Copilot CLI in the folder (it auto-reads AGENTS.md and runs
#     the worker loop unattended). This is what the subagents are for.
#   * "worker" -- spawn the plain `python -m emullm.worker` loop instead.
#   * "recruit"/"interactive" -- an **interactive recruit**: a human-driven
#     copilot sitting in an IDE that connects itself. The supervisor does NOT
#     spawn these; the `recruit`/`self` run modes just use them once they
#     join. Discovery returns no spawn spec for a folder of this type.
#
# Anything else that looks like an argv (a list, or a string with spaces) is
# taken literally as the launch command.
_RECRUIT_KINDS = {"recruit", "interactive", "ide", "attended", "self", "manual", "none"}
_COPILOT_KINDS = {"copilot", "auto", "agent", "subagent"}
_WORKER_KINDS = {"worker", "loop", "python"}


def _launch_for(
    launch: Any, worker_id: str, host_ws_url: str, role: str, model: str | None = None, modelmasks: str = ""
) -> tuple[Optional[list[str]], bool]:
    """Resolve a launch override into ``(argv, spawn)``.

    ``launch`` may be ``None`` (default: spawn Copilot), a named type
    (see above), or an explicit argv (list, or a space-split string).
    ``spawn`` is ``False`` for interactive-recruit types, which the
    supervisor must not launch (they connect themselves). ``model`` picks the
    AI model for Copilot-agent launches (ignored for the plain worker loop and
    for explicit argv overrides, which specify their own command).
    """
    if isinstance(launch, (list, tuple)) and launch:
        return [str(x) for x in launch], True
    if isinstance(launch, str) and launch.strip():
        token = launch.strip()
        low = token.lower()
        if low in _RECRUIT_KINDS:
            return None, False
        if low in _COPILOT_KINDS:
            return copilot_launch_argv(model=model), True
        if low in _WORKER_KINDS:
            return default_worker_argv(worker_id, host_ws_url, role, modelmasks), True
        return token.split(), True
    return copilot_launch_argv(model=model), True


def discover_worker_specs(
    base_dir: Path,
    host_ws_url: str = "ws://127.0.0.1:8801",
    *,
    role: str = "trusted",
    glob: str = "emullm_worker_*",
    launch: Any = None,
    model: str | None = None,
) -> list[WorkerSpec]:
    """Build one spec per ``subagents/<glob>`` folder found under base_dir.

    Each worker runs in its own folder so its request.json/reply.json
    handoff files stay isolated, and takes the folder name as its worker_id
    (matching the folders' AGENTS.md convention).

    ``launch`` picks the worker *type* (see :data:`_COPILOT_KINDS` etc.):
    by default each folder spawns the **Copilot CLI** (auto-configured
    agent). Use ``"worker"`` for the plain worker loop, ``"recruit"`` for
    interactive recruits (not spawned -- they connect themselves), or pass an
    explicit argv. Config sets this via ``subagent_launch``. ``model`` (config
    ``subagent_model``) picks the AI model for Copilot-agent launches.
    """
    subagents = base_dir / "subagents"
    root = subagents if subagents.is_dir() else base_dir
    specs: list[WorkerSpec] = []
    for path in sorted(root.glob(glob)):
        if not path.is_dir():
            continue
        worker_id = path.name
        argv, spawn = _launch_for(launch, worker_id, host_ws_url, role, model=model)
        if not spawn or not argv:
            continue  # interactive recruit: it connects itself, nothing to spawn
        specs.append(WorkerSpec(worker_id=worker_id, argv=argv, cwd=path, role=role))
    return specs


def _default_spawn(spec: WorkerSpec) -> subprocess.Popen:
    return subprocess.Popen(  # noqa: S603 -- argv is code-controlled, not shell
        spec.argv,
        cwd=str(spec.cwd) if spec.cwd else None,
    )


def load_config(path: Path) -> dict[str, Any]:
    """Read a config.json (an object) or return {} if missing/invalid."""
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except (OSError, ValueError):
        pass
    return {}


def specs_from_config(
    config: dict[str, Any],
    base_dir: Path,
    host_ws_url: str = "ws://127.0.0.1:8801",
    *,
    default_role: str = "trusted",
) -> list[WorkerSpec]:
    """Build worker specs from a config document's ``workers`` list.

    Each worker entry:
        { "id": "w1", "role": "training", "autostart": true,
          "cwd": "subagents/w1", "launch": "copilot" }
    ``cwd`` is resolved relative to base_dir. ``launch`` picks the worker
    type the same way discovery does (``copilot``/``worker``/``recruit`` or an
    explicit argv list/string); if omitted it defaults to the plain
    ``python -m emullm.worker`` loop. A ``recruit`` launch is skipped (it
    connects itself). Entries without an ``id`` are skipped.
    """
    workers = config.get("workers")
    if not isinstance(workers, list):
        return []
    specs: list[WorkerSpec] = []
    for entry in workers:
        if not isinstance(entry, dict):
            continue
        worker_id = str(entry.get("id") or entry.get("worker_id") or "").strip()
        if not worker_id:
            continue
        role = str(entry.get("role") or default_role)
        autostart = bool(entry.get("autostart", True))
        cwd_val = entry.get("cwd")
        cwd = (base_dir / cwd_val) if cwd_val else None
        launch = entry.get("launch")
        model = entry.get("model")
        raw_modelmasks = entry.get("modelmasks")
        if isinstance(raw_modelmasks, (list, tuple)):
            modelmasks = ",".join(str(mask).strip() for mask in raw_modelmasks if str(mask).strip())
        elif isinstance(raw_modelmasks, str):
            modelmasks = raw_modelmasks.strip()
        else:
            modelmasks = ""
        if launch is None:
            argv, spawn = default_worker_argv(worker_id, host_ws_url, role, modelmasks), True
        else:
            argv, spawn = _launch_for(launch, worker_id, host_ws_url, role, model=model, modelmasks=modelmasks)
        if not spawn or not argv:
            continue  # recruit: connects itself, nothing to spawn
        specs.append(
            WorkerSpec(
                worker_id=worker_id,
                argv=argv,
                cwd=cwd,
                role=role,
                modelmasks=modelmasks,
                autostart=autostart,
            )
        )
    return specs


def expand_agents(config: dict[str, Any]) -> dict[str, Any]:
    """Translate the unified ``agents`` list into the flat
    ``workers``/``mock_workers``/``backends`` structures the runtime consumes,
    appended to any explicit flat entries. Returns a new merged dict (the
    input is left unchanged); a config without ``agents`` passes through.

    Mapping by ``launch``:
      * ``subagent`` -> a managed worker (``command`` -> ``launch``, default
        ``copilot``; ``cwd`` defaults to ``subagents/<id>``)
      * ``mock``     -> a mock_workers entry
      * ``proxy``    -> a backends entry
      * ``recruit``  -> nothing (it connects itself)
    """
    agents = config.get("agents")
    if not isinstance(agents, list) or not agents:
        return dict(config)
    workers = list(config.get("workers") or [])
    mock_workers = list(config.get("mock_workers") or [])
    backends = list(config.get("backends") or [])
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        launch = str(agent.get("launch") or "").strip().lower()
        worker_id = str(agent.get("id") or agent.get("worker_id") or "").strip()
        if launch == "subagent":
            workers.append(
                {
                    "id": worker_id,
                    "role": agent.get("role"),
                    "cwd": agent.get("cwd") or (f"subagents/{worker_id}" if worker_id else None),
                    "launch": agent.get("command") or "copilot",
                    "model": agent.get("model"),
                    "modelmasks": agent.get("modelmasks"),
                }
            )
        elif launch == "mock":
            mock_workers.append(
                {
                    "id": worker_id,
                    "reply": agent.get("reply"),
                    "template": agent.get("template"),
                    "capabilities": agent.get("capabilities"),
                    "role": agent.get("role"),
                    "models": agent.get("models"),
                    "modelmasks": agent.get("modelmasks"),
                }
            )
        elif launch == "proxy":
            backends.append(
                {
                    "name": agent.get("name") or worker_id or "backend",
                    "base_url": agent.get("base_url"),
                    "api_key": agent.get("api_key"),
                    "api_key_env": agent.get("api_key_env"),
                    "model": agent.get("model"),
                    "default": agent.get("default"),
                }
            )
        # recruit: connects itself; nothing to spawn/register
    merged = dict(config)
    if workers:
        merged["workers"] = workers
    if mock_workers:
        merged["mock_workers"] = mock_workers
    if backends:
        merged["backends"] = backends
    return merged


def build_specs(
    base_dir: Path,
    host_ws_url: str = "ws://127.0.0.1:8801",
    config: Optional[dict[str, Any]] = None,
) -> list[WorkerSpec]:
    """Worker specs from config if it declares any, else discovered from
    ``subagents/emullm_worker_*`` folders. A config ``subagent_launch``
    (list or string) overrides the default Copilot launch for discovery, and
    ``subagent_model`` sets the default AI model for discovered Copilot workers."""
    if config:
        specs = specs_from_config(config, base_dir, host_ws_url)
        if specs:
            return specs
    launch = config.get("subagent_launch") if config else None
    model = config.get("subagent_model") if config else None
    return discover_worker_specs(base_dir, host_ws_url, launch=launch, model=model)


class Supervisor:
    """Starts, stops, and reports on managed worker subprocesses."""

    def __init__(
        self,
        specs: Optional[list[WorkerSpec]] = None,
        *,
        spawn: Callable[[WorkerSpec], Any] = _default_spawn,
    ) -> None:
        self._specs: dict[str, WorkerSpec] = {}
        self._spawn = spawn
        for spec in specs or []:
            self._specs[spec.worker_id] = spec

    # -- registration -------------------------------------------------------
    def add_spec(self, spec: WorkerSpec) -> None:
        self._specs[spec.worker_id] = spec

    def remove_spec(self, worker_id: str) -> None:
        self.stop(worker_id)
        self._specs.pop(worker_id, None)

    def specs(self) -> list[WorkerSpec]:
        return list(self._specs.values())

    # -- lifecycle ----------------------------------------------------------
    def start(self, worker_id: str) -> bool:
        """Start (or restart if exited) one worker. Returns True if a new
        process was launched, False if it was already running or unknown."""
        spec = self._specs.get(worker_id)
        if spec is None:
            return False
        if self._is_running(spec):
            return False
        spec.process = self._spawn(spec)
        return True

    def stop(self, worker_id: str) -> bool:
        """Terminate one worker if running. Returns True if it was running."""
        spec = self._specs.get(worker_id)
        if spec is None or not self._is_running(spec):
            return False
        proc = spec.process
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        spec.process = None
        return True

    def start_autostart(self) -> list[str]:
        """Start every spec flagged autostart. Returns the ids started."""
        started: list[str] = []
        for spec in self._specs.values():
            if spec.autostart and self.start(spec.worker_id):
                started.append(spec.worker_id)
        return started

    def stop_all(self) -> None:
        for worker_id in list(self._specs):
            self.stop(worker_id)

    # -- introspection ------------------------------------------------------
    @staticmethod
    def _is_running(spec: WorkerSpec) -> bool:
        proc = spec.process
        return proc is not None and proc.poll() is None

    def status(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for spec in self._specs.values():
            proc = spec.process
            running = self._is_running(spec)
            rows.append(
                {
                    "worker_id": spec.worker_id,
                    "managed": True,
                    "running": running,
                    "pid": getattr(proc, "pid", None) if proc is not None else None,
                    "returncode": (proc.poll() if proc is not None else None),
                    "role": spec.role,
                    "autostart": spec.autostart,
                    "cwd": str(spec.cwd) if spec.cwd else None,
                    "argv": list(spec.argv),
                }
            )
        return rows


# --- module-level singleton, shared by app.py (owner) and api.py (reader) ---
_supervisor: Optional[Supervisor] = None


def set_supervisor(supervisor: Optional[Supervisor]) -> None:
    global _supervisor
    _supervisor = supervisor


def get_supervisor() -> Optional[Supervisor]:
    return _supervisor

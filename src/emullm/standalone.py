"""Standalone runner: run EMULLM as its own OS process.

Entry points:

* ``python -m emullm.standalone [host] [http_port]`` (or the explicit
  ``serve`` subcommand) runs the server in the foreground.
* ``python -m emullm.standalone install <dir>`` creates the ``emullm_runtime``
  container inside ``<dir>`` (``<dir>/emullm_runtime``) and seeds a live config
  from the shipped default, so the server can be run against a self-contained
  directory.
* :func:`launch` spawns the server as a *detached* background process. It is
  idempotent: if something is already serving the target port it does nothing.
  The workbench plugin uses this in "standalone" mode and reaches the server
  through its ``web_proxy`` (see ``plugin.json``).

Run ``python -m emullm.standalone --help`` for the full command reference.

Contrast with :mod:`emullm.embedded`, which mounts the service in-process.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# .../emullm (package dir) and its parent (import root for ``emullm``).
_PACKAGE_DIR = Path(__file__).resolve().parent
_IMPORT_ROOT = _PACKAGE_DIR.parent

DEFAULT_HOST = os.environ.get("EMULLM_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("EMULLM_HTTP_PORT", "8801"))

# Windows process-creation flags (no dependency on the ``subprocess`` constants,
# which are only defined on Windows): detach from the parent console and start a
# new process group so the child survives the host exiting.
_DETACHED_PROCESS = 0x00000008
_CREATE_NEW_PROCESS_GROUP = 0x00000200


def _probe_host(host: str) -> str:
    """A connectable address for a bind host (wildcards map to loopback)."""

    if host in ("0.0.0.0", "::", ""):
        return "127.0.0.1"
    return host


def is_listening(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 0.5) -> bool:
    """Return True when a TCP connection to ``host:port`` succeeds."""

    try:
        with socket.create_connection((_probe_host(host), port), timeout=timeout):
            return True
    except OSError:
        return False


# Subcommands recognised at the top level. Anything else (a bare ``[host]
# [port]`` / flag form, or nothing at all) is treated as the implicit ``serve``
# command so the historical invocation -- and :func:`launch` -- keep working.
_SUBCOMMANDS = frozenset({"serve", "run", "install"})


def _add_serve_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("host_pos", nargs="?", default=None, help="host to bind (positional)")
    parser.add_argument("port_pos", nargs="?", type=int, default=None, help="port to bind (positional)")
    parser.add_argument("--host", default=None, help="host to bind to")
    parser.add_argument("--port", type=int, default=None, help="port to bind to")
    parser.add_argument("--reload", action="store_true", help="enable autoreload (development)")
    parser.add_argument("--no-reload", action="store_true", help="disable autoreload (default)")


def _config_host_port() -> tuple[str | None, int | None]:
    """Read ``host`` / ``http_port`` from the live config, if present."""

    from . import paths

    try:
        data = json.loads(paths.config_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return (None, None)
    if not isinstance(data, dict):
        return (None, None)
    host_raw = data.get("host")
    port_raw = data.get("http_port", data.get("port"))
    host = host_raw.strip() if isinstance(host_raw, str) and host_raw.strip() else None
    port: int | None = None
    if isinstance(port_raw, bool):
        port = None
    elif isinstance(port_raw, int):
        port = port_raw
    elif isinstance(port_raw, str) and port_raw.strip().isdigit():
        port = int(port_raw.strip())
    return (host, port)


def _resolve_host_port(cli_host: str | None, cli_port: int | None) -> tuple[str, int]:
    """Resolve the bind host/port: CLI > env > config file > built-in default."""

    cfg_host, cfg_port = _config_host_port()
    env_host = os.environ.get("EMULLM_HOST") or None
    env_port_raw = os.environ.get("EMULLM_HTTP_PORT")
    env_port = int(env_port_raw) if env_port_raw and env_port_raw.isdigit() else None

    host = cli_host or env_host or cfg_host or "127.0.0.1"
    port = cli_port or env_port or cfg_port or 8801
    return (host, port)


def _patch_config_host_port(
    config_file: Path, host: str | None, port: int | None
) -> None:
    """Write ``host`` / ``http_port`` into an existing (or new) config file."""

    try:
        data = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    if host is not None:
        data["host"] = host
    if port is not None:
        data["http_port"] = port
    config_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _serve(args: argparse.Namespace) -> None:
    """Run the relay server in the foreground (the ``serve`` subcommand)."""

    import uvicorn

    from . import paths

    # First-run bootstrap: create the config/runtime layout and seed the config
    # so a fresh (including non-editable) install works out of the box. Done
    # first so the config is present when we resolve the bind host/port from it.
    paths.ensure_layout()

    host, port = _resolve_host_port(
        args.host_pos or args.host, args.port_pos or args.port
    )

    os.environ["EMULLM_HOST"] = host
    os.environ["EMULLM_HTTP_PORT"] = str(port)

    if args.reload and not args.no_reload:
        uvicorn.run(
            "emullm.app:app",
            host=host,
            port=port,
            reload=True,
            reload_dirs=[str(_PACKAGE_DIR)],
            reload_includes=["*.py"],
            reload_excludes=["runtime/*", "__pycache__/*"],
        )
        return

    from . import process_control

    server = uvicorn.Server(
        uvicorn.Config(
            "emullm.app:app",
            host=host,
            port=port,
            reload=False,
        )
    )
    process_control.register_shutdown_callback(
        lambda: setattr(server, "should_exit", True)
    )
    try:
        server.run()
    finally:
        process_control.register_shutdown_callback(None)


def _install(
    directory: str,
    *,
    force: bool = False,
    host: str | None = None,
    port: int | None = None,
) -> Path:
    """Create the runtime container under ``directory`` and seed the config.

    ``directory`` is the *parent*: the ``emullm_runtime`` container is created
    inside it (``<directory>/emullm_runtime``) and gains ``config/``, ``logs/``,
    ``metrics/`` and ``state/`` subdirectories plus a live
    ``config/server_config.json`` seeded from the shipped default. Pointing
    ``EMULLM_RUNTIME_DIR`` at the container then serves from this install.

    ``host`` / ``port``, when given, are written into the seeded config's
    top-level ``host`` / ``http_port`` keys so ``serve`` picks them up -- this
    also works as an in-place edit of an existing install (no ``--force``
    needed just to change the port).
    """

    from . import paths

    parent = Path(directory).expanduser()
    container_root = parent / paths.EmullmRuntime.CONTAINER_NAME
    # ``env={}`` isolates the install from ambient EMULLM_* overrides so the
    # layout is always rooted at the container in a predictable way.
    runtime = paths.EmullmRuntime(root=container_root, env={})
    root = runtime.root

    for created in (runtime.config_dir, runtime.logs_dir, runtime.metrics_dir, runtime.state_dir):
        created.mkdir(parents=True, exist_ok=True)

    dist = paths.DIST_CONFIG_PATH
    config_file = runtime.config_file
    if dist.exists():
        # Keep a human-visible reference copy of the shipped default beside the
        # live file so it can be inspected / diffed.
        shutil.copy2(dist, runtime.dist_config_reference)

    if config_file.exists() and not force:
        print(f"emullm: live config already exists, left unchanged -> {config_file}")
        print("        (re-run with --force to reseed it from the shipped default)")
    elif dist.exists():
        shutil.copy2(dist, config_file)
        print(f"emullm: seeded live config from shipped default -> {config_file}")
    else:
        # No packaged default available (unusual); fall back to the standard
        # best-effort seeding so at least the directories exist.
        config_file = runtime.ensure_layout()
        print(f"emullm: created runtime layout -> {config_file}")

    if host is not None or port is not None:
        _patch_config_host_port(config_file, host, port)
        changed = ", ".join(
            part
            for part in (
                f"host={host}" if host is not None else "",
                f"http_port={port}" if port is not None else "",
            )
            if part
        )
        print(f"emullm: set {changed} in {config_file}")

    print(f"emullm: runtime installed under {root}")
    print("        (config/  logs/  metrics/  state/)")
    print()
    print("Run the server against this install with:")
    if os.name == "nt":
        print(f"    set EMULLM_RUNTIME_DIR={root}")
        print("    python -m emullm.standalone")
    else:
        print(f'    EMULLM_RUNTIME_DIR="{root}" python -m emullm.standalone')
    return root


def main(argv: list[str] | None = None) -> None:
    """Dispatch the standalone CLI.

    ``python -m emullm.standalone [host] [port]`` runs the relay directly
    (equivalently ``... serve [host] [port]``); ``--reload`` enables autoreload
    for development (this replaces the former ``run.py`` / ``emullm-serve
    --reload`` runners). ``python -m emullm.standalone install <dir>`` scaffolds
    a self-contained config/runtime directory.
    """

    raw = list(sys.argv[1:] if argv is None else argv)
    # Backward-compatible dispatch: the historical "[host] [port]" / flag form
    # (and the empty form used by launch()) is the implicit "serve" command.
    # Only an explicit subcommand, or a top-level -h/--help, bypasses it so the
    # subcommand help remains reachable.
    if not raw:
        raw = ["serve"]
    elif raw[0] not in _SUBCOMMANDS and raw[0] not in ("-h", "--help"):
        raw = ["serve", *raw]

    parser = argparse.ArgumentParser(
        prog="python -m emullm.standalone",
        description="Run the EMULLM relay server, or install its runtime layout.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m emullm.standalone                 serve on 127.0.0.1:8801\n"
            "  python -m emullm.standalone 0.0.0.0 9000    serve on a chosen host/port\n"
            "  python -m emullm.standalone serve --reload  serve with autoreload (dev)\n"
            "  python -m emullm.standalone install ./run   create a runtime directory\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve_parser = sub.add_parser(
        "serve",
        aliases=["run"],
        help="run the relay server in the foreground (default)",
        description="Run the EMULLM relay server in the foreground.",
    )
    _add_serve_arguments(serve_parser)

    install_parser = sub.add_parser(
        "install",
        help="create the emullm_runtime container in a target directory",
        description=(
            "Create the emullm_runtime container under DIRECTORY "
            "(DIRECTORY/emullm_runtime with config/, logs/, metrics/, state/) "
            "and seed a live config/server_config.json from the shipped default. "
            "Set EMULLM_RUNTIME_DIR to the created container to serve from it."
        ),
    )
    install_parser.add_argument(
        "directory",
        help="parent directory the emullm_runtime container is created in",
    )
    install_parser.add_argument(
        "--host",
        default=None,
        help="host to write into the seeded config (top-level 'host')",
    )
    install_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="port to write into the seeded config (top-level 'http_port')",
    )
    install_parser.add_argument(
        "--force",
        action="store_true",
        help="reseed the live config even if it already exists",
    )

    args = parser.parse_args(raw)

    if args.command == "install":
        _install(args.directory, force=args.force, host=args.host, port=args.port)
        return
    _serve(args)


def launch(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    wait: bool = True,
    timeout: float = 20.0,
) -> subprocess.Popen | None:
    """Spawn the standalone server as a detached background process.

    Idempotent: returns ``None`` immediately if ``host:port`` is already serving.
    Otherwise starts ``python -m emullm.standalone host port`` detached, with
    stdout/stderr redirected to ``<state_dir>/standalone.log``. When ``wait`` is
    true, blocks until the port accepts connections (or raises on failure).
    """

    if is_listening(host, port):
        return None

    env = os.environ.copy()
    # The child must import ``emullm`` regardless of its cwd.
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(_IMPORT_ROOT), existing) if p)

    from . import paths

    paths.ensure_layout()
    log_dir = paths.log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "standalone.log"
    log = open(log_path, "ab", buffering=0)  # noqa: SIM115 - handed to the child

    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = _DETACHED_PROCESS | _CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(
        [sys.executable, "-m", "emullm.standalone", host, str(port)],
        cwd=str(_IMPORT_ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=log,
        close_fds=True,
        **kwargs,
    )

    if not wait:
        return proc

    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_listening(host, port):
            return proc
        if proc.poll() is not None:
            raise RuntimeError(
                f"emullm standalone exited (code {proc.returncode}) during "
                f"startup; see {log_path}"
            )
        time.sleep(0.25)
    raise TimeoutError(
        f"emullm standalone did not start listening on {host}:{port} within "
        f"{timeout:.0f}s; see {log_path}"
    )


def create_router(manifest: dict[str, Any] | None = None):
    """Plugin hook for standalone mode.

    Ensures the standalone server is running, then returns an empty router. The
    workbench serves ``/emullm`` by proxying to the standalone process (see
    the ``web_proxy`` entry in ``plugin.json``), so this router intentionally
    contributes no in-process routes.
    """

    launch(DEFAULT_HOST, DEFAULT_PORT)
    try:
        from fastapi import APIRouter
    except Exception:  # pragma: no cover - FastAPI always present in practice
        return None
    return APIRouter()


if __name__ == "__main__":
    main()

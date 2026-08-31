"""Graceful standalone-process shutdown and restart coordination."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections.abc import Callable
from typing import Any

_lock = threading.RLock()
_shutdown_callback: Callable[[], None] | None = None


def register_shutdown_callback(callback: Callable[[], None] | None) -> None:
    global _shutdown_callback
    with _lock:
        _shutdown_callback = callback


def shutdown_available() -> bool:
    with _lock:
        return _shutdown_callback is not None


def schedule_shutdown(delay_seconds: float = 0.5) -> bool:
    with _lock:
        callback = _shutdown_callback
    if callback is None:
        return False
    timer = threading.Timer(delay_seconds, callback)
    timer.daemon = True
    timer.start()
    return True


def schedule_restart(
    host: str,
    port: int,
    *,
    delay_seconds: float = 0.5,
) -> int:
    """Start a detached helper, then gracefully stop the current server."""
    if not shutdown_available():
        raise RuntimeError("standalone process control is not available")
    env = os.environ.copy()
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = 0x00000008 | 0x00000200
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(  # noqa: S603 - fixed package module and validated scalar args
        [
            sys.executable,
            "-m",
            "emullm.restart_helper",
            "--wait-pid",
            str(os.getpid()),
            "--host",
            host,
            "--port",
            str(port),
        ],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        **kwargs,
    )
    if not schedule_shutdown(delay_seconds):
        process.terminate()
        raise RuntimeError("standalone process control became unavailable")
    return process.pid

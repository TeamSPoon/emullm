"""Wait for one EMULLM process to exit, then start its replacement."""
from __future__ import annotations

import argparse
import os
import socket
import time

from . import standalone


def _process_exists(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except OSError:
        return False
    return True


def _wait_until(predicate, timeout: float, description: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for {description}")


def _port_is_free(host: str, port: int) -> bool:
    probe_host = "127.0.0.1" if host in {"", "0.0.0.0", "::"} else host
    try:
        with socket.create_connection((probe_host, port), timeout=0.2):
            return False
    except OSError:
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    _wait_until(
        lambda: not _process_exists(args.wait_pid),
        90,
        f"process {args.wait_pid} to exit",
    )
    _wait_until(
        lambda: _port_is_free(args.host, args.port),
        30,
        f"{args.host}:{args.port} to become free",
    )
    standalone.launch(args.host, args.port, wait=False)


if __name__ == "__main__":
    main()

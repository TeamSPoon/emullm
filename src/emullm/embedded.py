"""Embedded runner: run EMULLM *inside* the host (workbench) process.

In embedded mode the service shares the host's event loop and is mounted directly
into the host application. ``create_router(manifest)`` returns the relay's
FastAPI routes with their startup and shutdown lifespan. No separate OS process
and no proxy are involved.

Contrast with :mod:`emullm.standalone`, which runs the service as its own
process reached through the workbench ``web_proxy``.
"""

from __future__ import annotations

from typing import Any


def create_router(manifest: dict[str, Any] | None = None):
    """Return the in-process router the host mounts directly.

    The relay router already declares its public ``/v1``, ``/emullm``, and
    ``/admin/emullm`` paths. Its lifespan applies config and manages workers.
    """

    from fastapi import APIRouter

    from .api import router as relay_router
    from .app import _lifespan

    router = APIRouter(lifespan=_lifespan)
    router.include_router(relay_router)
    return router


def main(argv: list[str] | None = None) -> None:
    """Convenience: serve the same wiring in *this* process (foreground).

    This is handy for local runs/tests; the workbench itself consumes
    :func:`create_router` rather than calling ``main``.
    """

    from .standalone import main as standalone_main

    standalone_main(argv)


if __name__ == "__main__":
    main()

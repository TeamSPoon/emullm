"""EMULLM workbench plugin entrypoint.

The workbench plugin loader imports this file directly (not as part of a package)
and calls ``create_router(manifest)``. We add the project's ``src`` directory to
``sys.path`` so the ``emullm`` package is importable, then delegate to one of two
runners:

* :mod:`emullm.standalone` — run EMULLM as its own process; the workbench
  reaches it through its ``web_proxy`` (see ``plugin.json``). This is the default.
* :mod:`emullm.embedded` — mount EMULLM in-process and return its router.

Select the runner with ``EMULLM_PLUGIN_MODE=standalone|embedded``
(default: ``standalone``).

This file is also the single canonical way to start the relay by hand::

    python plugin.py                 # 127.0.0.1:8801
    python plugin.py --port 9001
    python plugin.py --reload        # development autoreload
    python -m emullm.standalone      # equivalent module form

``main`` is a thin delegate to :func:`emullm.standalone.main` (the former
``run.py`` / ``emullm-serve`` runners have been folded into it). On both the
loader path and the manual path we first create the ``emullm_runtime`` container
(config, logs, metrics, state) so a fresh or non-editable install works out of
the box.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_SOURCE_ROOT = _HERE / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))


def _bootstrap() -> None:
    """Create the emullm_runtime container + seed config on first run."""
    try:
        from emullm import paths

        paths.ensure_layout()
    except Exception:
        # Best-effort: the individual entry points also call ensure_layout.
        pass


def _mode() -> str:
    return (os.environ.get("EMULLM_PLUGIN_MODE") or "standalone").strip().lower()


def resolve_ui_pages(manifest: dict[str, Any], pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve where this plugin's declared pages live.

    EMULLM serves its own administration console, so every page is an
    absolute URL under the manifest's ``configPage`` base rather than a
    workbench-rendered descriptor. A relative descriptor is joined onto that
    base, and any fragment the manifest declares is preserved.
    """

    base = str(manifest.get("configPage") or "").rstrip("/")
    resolved: list[dict[str, Any]] = []
    for page in pages:
        descriptor = str(page.get("descriptor") or "")
        if descriptor.startswith(("http://", "https://")):
            address = descriptor
        elif base:
            address = f"{base}/{descriptor.lstrip('/')}" if descriptor else base
        else:
            address = descriptor
        resolved.append({**page, "address": address, "external": address.startswith(("http://", "https://"))})
    return resolved


def create_router(manifest: dict[str, Any] | None = None):
    """Delegate to the selected runner's ``create_router``."""

    _bootstrap()
    if _mode() == "embedded":
        from emullm.embedded import create_router as _create_router
    else:
        from emullm.standalone import create_router as _create_router
    return _create_router(manifest)


def main(argv: list[str] | None = None) -> None:
    """Start the relay directly (``python plugin.py ...``).

    Thin delegate to the canonical runner :func:`emullm.standalone.main`, also
    reachable as ``python -m emullm.standalone``.
    """

    _bootstrap()
    from emullm.standalone import main as _main

    _main(argv)


__all__ = ["create_router", "resolve_ui_pages", "main"]


if __name__ == "__main__":
    main()

"""EMULLM workbench plugin entrypoint.

The workbench plugin loader imports this file directly (not as part of a package)
and calls ``create_router(manifest)``. We add this directory to ``sys.path`` so
the ``emullm`` package is importable, then delegate to one of two runners:

* :mod:`emullm.standalone` — run EMULLM as its own process; the workbench
  reaches it through its ``web_proxy`` (see ``plugin.json``). This is the default.
* :mod:`emullm.embedded` — mount EMULLM in-process and return its router.

Select the runner with ``EMULLM_PLUGIN_MODE=standalone|embedded``
(default: ``standalone``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


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

    if _mode() == "embedded":
        from emullm.embedded import create_router as _create_router
    else:
        from emullm.standalone import create_router as _create_router
    return _create_router(manifest)


__all__ = ["create_router", "resolve_ui_pages"]

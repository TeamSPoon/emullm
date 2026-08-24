"""Console-script entry point for the emullm relay.

This lives inside the ``emullm`` package (unlike ``run.py`` at the repo
root) so it is importable from an installed console script -- an installed
``.exe`` does not put the repo root on ``sys.path``, which is why an entry
point of ``run:main`` fails with ``ModuleNotFoundError: No module named
'run'``. Use ``emullm.cli:main`` instead.

Usage:
    emullm-serve                  # 127.0.0.1:8801
    emullm-serve --port 9001
    emullm-serve --no-reload      # for a longer-lived background run
"""
from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8801

# Directory of the installed package, used to scope autoreload watching.
_PACKAGE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", default=DEFAULT_PORT, type=int)
    parser.add_argument("--no-reload", action="store_true", help="disable autoreload")
    args = parser.parse_args()

    reload = not args.no_reload
    uvicorn.run(
        "emullm.app:app",
        host=args.host,
        port=args.port,
        reload=reload,
        reload_dirs=[str(_PACKAGE_DIR)] if reload else None,
        reload_includes=["*.py"] if reload else None,
        reload_excludes=["runtime/*", "__pycache__/*"] if reload else None,
    )


if __name__ == "__main__":
    main()

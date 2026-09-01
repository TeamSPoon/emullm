"""Run the emullm relay standalone.

Usage:
    python run.py                 # 127.0.0.1:8801
    python run.py --port 9001
    python run.py --no-reload     # for a longer-lived background run
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
DEFAULT_PORT = 8801


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="host to bind to")
    parser.add_argument("--port", default=DEFAULT_PORT, type=int, help="port to bind to")
    parser.add_argument("--no-reload", action="store_true", help="disable autoreload")
    args = parser.parse_args()

    sys.path.insert(0, str(SOURCE_ROOT))
    if args.no_reload:
        os.environ["EMULLM_HOST"] = args.host
        os.environ["EMULLM_HTTP_PORT"] = str(args.port)
        from emullm import standalone

        standalone.main([args.host, str(args.port)])
        return
    uvicorn.run(
        "emullm.app:app",
        host=args.host,
        port=args.port,
        reload=True,
        reload_dirs=[str(SOURCE_ROOT / "emullm")],
        reload_includes=["*.py"],
        reload_excludes=["runtime/*", "__pycache__/*"],
    )


if __name__ == "__main__":
    main()

"""Run the emullm relay standalone.

Usage:
    python run.py                 # 127.0.0.1:8801
    python run.py --port 9001
    python run.py --no-reload     # for a longer-lived background run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
DEFAULT_PORT = 8801


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0", help="host to bind to")
    parser.add_argument("--port", default=DEFAULT_PORT, type=int, help="port to bind to")
    parser.add_argument("--no-reload", action="store_true", help="disable autoreload")
    args = parser.parse_args()

    sys.path.insert(0, str(SOURCE_ROOT))
    uvicorn.run(
        "emullm.app:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        reload_dirs=[str(SOURCE_ROOT / "emullm")] if not args.no_reload else None,
        reload_includes=["*.py"] if not args.no_reload else None,
        reload_excludes=["runtime/*", "__pycache__/*"] if not args.no_reload else None,
    )


if __name__ == "__main__":
    main()

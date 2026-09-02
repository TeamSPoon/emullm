"""Filesystem layout for emullm: one class that owns everything it creates.

:class:`EmullmRuntime` is the single container for every mutable thing the
server writes -- configuration, logs, metrics, state, and the runtime records
(mailboxes, files, event logs, headless-copilot data). Keeping it all under one
directory means there is exactly one thing to gitignore, relocate, or create on
first run::

    <emullm_runtime>/
      config/
        server_config.json        # live, runtime-mutated config (git-ignored)
        server_config_dist.json   # reference copy of the shipped default
        mailboxes.json            # existing mailbox config
      logs/                       # standalone.log, ...
      metrics/
      state/
      files/                      # existing runtime records
      events_logs/
      headless_copilots/

The shipped default template ships as package data (``server_config_dist.json``
next to this module) so non-editable / wheel installs have it available.

Container location (``EmullmRuntime.root``):

* ``EMULLM_RUNTIME_DIR`` -- explicit override (used everywhere), else
* source checkout: the existing ``src/runtime`` when present (so existing live
  data is never orphaned), otherwise ``<plugin root>/emullm_runtime``, else
* non-editable install: ``<user data dir>/emullm_runtime``
  (``%LOCALAPPDATA%`` on Windows, ``$XDG_DATA_HOME`` / ``~/.local/share`` on
  POSIX).

Optional finer-grained overrides:

* ``EMULLM_CONFIG_DIR``   -- directory holding ``server_config.json``
* ``EMULLM_CONFIG_FILE``  -- explicit path to the live config file
* ``EMULLM_LOG_DIR``      -- directory holding ``standalone.log``
* ``EMULLM_DATA_DIR``     -- legacy alias for ``EMULLM_CONFIG_DIR``
* ``EMULLM_STATE_DIR``    -- legacy alias for ``EMULLM_LOG_DIR``

A module-level default instance (:data:`RUNTIME`) plus thin module-level
delegators are provided for convenience and backwards compatibility.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Mapping

_CONTAINER_NAME = "emullm_runtime"

# ``.../emullm`` (package dir), ``.../src`` (or site-packages) and the plugin
# root above it. ``PLUGIN_ROOT`` is only meaningful for a source checkout.
PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PLUGIN_ROOT = SRC_DIR.parent

# The shipped, immutable default config. Stored as package data so it is present
# in both source checkouts and wheels.
DIST_CONFIG_PATH = PACKAGE_DIR / "server_config_dist.json"


class EmullmRuntime:
    """The single container that owns every path emullm writes to.

    Paths are resolved lazily from the environment on each access, so tests can
    point ``EMULLM_RUNTIME_DIR`` (or pass ``root=``) at a temporary directory.
    """

    #: Container directory name used for fresh / non-editable installs.
    CONTAINER_NAME = _CONTAINER_NAME
    PACKAGE_DIR = PACKAGE_DIR
    SRC_DIR = SRC_DIR
    PLUGIN_ROOT = PLUGIN_ROOT
    DIST_CONFIG_PATH = DIST_CONFIG_PATH

    def __init__(
        self,
        root: str | os.PathLike[str] | None = None,
        *,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._env = env if env is not None else os.environ
        self._explicit_root = Path(root) if root is not None else None

    # -- resolution --------------------------------------------------------
    def _get(self, *names: str) -> str | None:
        for name in names:
            value = self._env.get(name)
            if value:
                return value
        return None

    def is_source_checkout(self) -> bool:
        """True when running from a source/editable checkout (writable root)."""
        return (self.PLUGIN_ROOT / "pyproject.toml").exists() or (
            self.PLUGIN_ROOT / ".git"
        ).exists()

    def user_data_root(self) -> Path:
        """Per-user writable base for non-editable installs."""
        if os.name == "nt":
            base = self._env.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        else:
            base = self._env.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
        return Path(base)

    @property
    def root(self) -> Path:
        """The single container directory for everything the server creates."""
        if self._explicit_root is not None:
            return self._explicit_root
        override = self._get("EMULLM_RUNTIME_DIR")
        if override:
            return Path(override)
        if self.is_source_checkout():
            # Preserve the historical source-checkout location when it already
            # holds live data, so mailboxes/files/events are never orphaned.
            legacy = self.SRC_DIR / "runtime"
            if legacy.exists():
                return legacy
            return self.PLUGIN_ROOT / self.CONTAINER_NAME
        return self.user_data_root() / self.CONTAINER_NAME

    # ``RUNTIME_DIR`` is the historical name for the container.
    runtime_dir = root

    @property
    def config_dir(self) -> Path:
        override = self._get("EMULLM_CONFIG_DIR", "EMULLM_DATA_DIR")
        if override:
            return Path(override)
        return self.root / "config"

    @property
    def logs_dir(self) -> Path:
        override = self._get("EMULLM_LOG_DIR", "EMULLM_STATE_DIR")
        if override:
            return Path(override)
        return self.root / "logs"

    @property
    def metrics_dir(self) -> Path:
        return self.root / "metrics"

    @property
    def state_dir(self) -> Path:
        return self.root / "state"

    @property
    def files_dir(self) -> Path:
        return self.root / "files"

    @property
    def events_logs_dir(self) -> Path:
        return self.root / "events_logs"

    @property
    def headless_copilots_dir(self) -> Path:
        return self.root / "headless_copilots"

    @property
    def config_file(self) -> Path:
        """The live config path (honours ``EMULLM_CONFIG_FILE``)."""
        override = self._get("EMULLM_CONFIG_FILE")
        if override:
            return Path(override)
        return self.config_dir / "server_config.json"

    @property
    def dist_config_reference(self) -> Path:
        """The human-visible copy of the shipped default kept beside the live file."""
        return self.config_dir / "server_config_dist.json"

    @property
    def mailboxes_file(self) -> Path:
        return self.config_dir / "mailboxes.json"

    # -- config defaults / seeding ----------------------------------------
    def _config_default_sources(self) -> tuple[Path, ...]:
        """Ordered candidates the live config is seeded from when it is missing.

        Existing installs are preferred over the pristine shipped default so an
        upgrade keeps the user's settings: the earlier ``data/config.json`` and
        the oldest tracked ``config.json`` come first, and a fresh install falls
        through to the packaged default.
        """
        return (
            self.PLUGIN_ROOT / "data" / "config.json",   # earlier interim live config
            self.PLUGIN_ROOT / "config.json",            # oldest tracked/live config
            self.dist_config_reference,                  # refreshed reference copy
            self.DIST_CONFIG_PATH,                       # shipped default (package data)
        )

    def read_config_defaults(self) -> dict:
        """Load the shipped default document, used for top-level key backfill."""
        for source in (self.dist_config_reference, self.DIST_CONFIG_PATH):
            try:
                if source.is_file():
                    data = json.loads(source.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        return data
            except (OSError, ValueError):
                continue
        return {}

    def ensure_layout(self) -> Path:
        """Create the container directories and seed the live config file.

        Idempotent and best-effort -- safe to call on every startup and from the
        plugin loader so the directories are "made the first time" regardless of
        how the server is launched. Returns the resolved live config path.
        """
        config_dir = self.config_dir
        try:
            config_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        for directory in (self.logs_dir, self.metrics_dir, self.state_dir):
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
        # Keep a human-visible copy of the shipped default beside the live file
        # so it can be inspected/diffed and always reflects the package version.
        if self.DIST_CONFIG_PATH.exists():
            try:
                shutil.copy2(self.DIST_CONFIG_PATH, self.dist_config_reference)
            except OSError:
                pass
        # Seed the live config exactly once from the first available
        # default/legacy file: a fresh (or non-editable) install starts from the
        # shipped default, while an upgrade preserves existing settings.
        config_file = self.config_file
        if not config_file.exists():
            for source in self._config_default_sources():
                if source.exists() and source != config_file:
                    try:
                        shutil.copy2(source, config_file)
                    except OSError:
                        pass
                    break
        return config_file


# Default instance + module-level delegators (backwards-compatible surface).
RUNTIME = EmullmRuntime()


def is_source_checkout() -> bool:
    return RUNTIME.is_source_checkout()


def user_data_root() -> Path:
    return RUNTIME.user_data_root()


def runtime_dir() -> Path:
    return RUNTIME.root


def config_dir() -> Path:
    return RUNTIME.config_dir


def log_dir() -> Path:
    return RUNTIME.logs_dir


def config_file() -> Path:
    return RUNTIME.config_file


def read_config_defaults() -> dict:
    return RUNTIME.read_config_defaults()


def ensure_layout() -> Path:
    return RUNTIME.ensure_layout()


# Module-level snapshots for cheap access; call :func:`ensure_layout` (idempotent)
# before first use to create + seed them.
RUNTIME_DIR = RUNTIME.root
CONFIG_DIR = RUNTIME.config_dir
RUNTIME_LOGS_DIR = RUNTIME.logs_dir
RUNTIME_METRICS_DIR = RUNTIME.metrics_dir
RUNTIME_STATE_DIR = RUNTIME.state_dir
LOG_DIR = RUNTIME.logs_dir
CONFIG_PATH = RUNTIME.config_file

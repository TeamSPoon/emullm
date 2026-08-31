"""Headless Copilot servant configuration, lifecycle, and admin API.

Each configured servant is a small adapter process that connects to EMULLM's
shared ``/emullm/ws`` endpoint and owns one resident Copilot SDK/CLI runtime
with a stable session ID. Requests reuse both the process and conversation.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .supervisor import Supervisor, WorkerSpec

router = APIRouter()

DEFAULT_SYSTEM_PROMPT = (
    "Act as the model requested by the OpenAI-compatible caller. Answer the "
    "request directly and return only the assistant response that should be "
    "sent to the caller."
)
_CONFIG_IO_LOCK = threading.RLock()
_IMPORT_ROOT = Path(__file__).resolve().parent.parent
_MODEL_CACHE_LOCK = threading.RLock()
_MODEL_CACHE: dict[str, Any] | None = None
_MODEL_CACHE_TTL_SECONDS = 300.0
_FALLBACK_MODELS = [
    {"id": "gpt-5.6-sol", "name": "GPT-5.6 Sol"},
    {"id": "gpt-5.3-codex", "name": "GPT-5.3-Codex"},
    {"id": "claude-sonnet-5", "name": "Claude Sonnet 5"},
    {"id": "claude-fable-5", "name": "Claude Fable 5"},
    {"id": "claude-opus-5", "name": "Claude Opus 5"},
    {"id": "gemini-3.1-pro-preview", "name": "Gemini 3.1 Pro"},
    {"id": "gpt-5.5", "name": "GPT-5.5"},
    {"id": "kimi-k3", "name": "Kimi K3"},
    {"id": "gemini-3.7-flash", "name": "Gemini 3.7 Flash"},
    {"id": "gpt-5.6-terra", "name": "GPT-5.6 Terra"},
    {"id": "grok-4.6", "name": "Grok 4.6"},
    {"id": "claude-haiku-4.5", "name": "Claude Haiku 4.5"},
    {"id": "gpt-5.4-mini", "name": "GPT-5.4 mini"},
    {"id": "gpt-5.6-luna", "name": "GPT-5.6 Luna"},
    {"id": "kimi-k2.7-code", "name": "Kimi K2.7 Code"},
    {"id": "mai-code-1-flash-picker", "name": "MAI-Code-1-Flash"},
    {"id": "mai-code-1.1-flash", "name": "MAI-Code-1.1-Flash"},
]
_MODEL_QUALITY_ORDER = [
    "claude-opus-5",
    "gpt-5.6-sol",
    "claude-opus-4.8-fast",
    "claude-sonnet-5",
    "gpt-5.5",
    "gpt-5.6-terra",
    "gemini-3.1-pro-preview",
    "kimi-k3",
    "grok-4.6",
    "gpt-5.4",
    "claude-sonnet-4.6",
    "gpt-5.3-codex",
    "claude-fable-5",
    "gpt-5.6-luna",
    "gemini-3.7-flash",
    "grok-4.5",
    "claude-opus-4.5",
    "claude-sonnet-4.5",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "kimi-k2.7-code",
    "gpt-5.4-mini",
    "gpt-5-mini",
    "claude-haiku-4.5",
    "mai-code-1.1-flash",
    "mai-code-1-flash-picker",
]
_MODEL_QUALITY_INDEX = {
    model_id: index for index, model_id in enumerate(_MODEL_QUALITY_ORDER)
}
_REASONING_LEVELS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]


class HeadlessCopilotConfig(BaseModel):
    """Configuration for one persistent headless Copilot servant."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    worker_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
        description="Stable EMULLM worker and mailbox identity.",
    )
    session_id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Persistent Copilot conversation UUID; reset it to discard conversation history.",
    )
    model: str | None = Field(
        default=None,
        max_length=200,
        description="Explicit Copilot model ID. When omitted, one model is chosen randomly at servant start.",
    )
    model_pool: list[str] = Field(
        default_factory=list,
        description="Random-selection model IDs. An empty list uses the live discovered Copilot catalog.",
    )
    model_selector: str = Field(
        default="random",
        description="Model strategy when model is blank: random, best-N, or worst-N (worse-N alias accepted).",
    )
    modelmasks: list[str] = Field(
        default_factory=list,
        description="OpenAI-compatible model glob patterns handled by this servant; empty accepts all.",
    )
    role: str = Field(default="headless-copilot", min_length=1, max_length=100)
    capabilities: list[str] = Field(
        default_factory=list,
        description=(
            "Declared service/input capabilities such as audio_input; prefix a "
            "name with ! or - to explicitly decline it."
        ),
    )
    system_prompt: str = Field(default=DEFAULT_SYSTEM_PROMPT, min_length=1, max_length=20_000)
    cwd: str | None = None
    host_ws_url: str | None = None
    copilot_command: str | None = None
    autostart: bool = True
    warmup: bool = True
    warmup_prompt: str = Field(
        default="Startup warmup: reply only READY.",
        min_length=1,
        max_length=2_000,
        description="Prompt sent once after the resident runtime starts and before its WebSocket connects.",
    )
    timeout_seconds: float = Field(default=900.0, ge=1.0, le=86_400.0)
    reconnect_seconds: float = Field(default=2.0, ge=0.1, le=300.0)
    context: Literal["default", "long_context"] = "default"
    reasoning_effort: str | None = Field(
        default=None,
        description="Exact effort or selector: random, most-N, or least-N.",
    )
    max_ai_credits: float | None = Field(default=None, gt=0)
    allow_all: bool = False
    load_custom_instructions: bool = False
    enable_builtin_mcps: bool = False
    chunk_long_prompts: bool = True
    chunk_tokens: int | None = Field(
        default=None,
        ge=1_000,
        le=1_000_000,
        description="Optional per-ingestion-chunk token budget; omitted derives it from SDK model limits.",
    )
    max_chunks: int = Field(default=64, ge=1, le=1_000)
    max_prompt_chars: int = Field(default=4_000_000, ge=1_000, le=20_000_000)
    max_output_chars: int = Field(default=200_000, ge=1_000, le=2_000_000)
    max_attachment_bytes: int = Field(
        default=25 * 1024 * 1024,
        ge=1_024,
        le=100 * 1024 * 1024,
        description="Maximum bytes fetched for one native SDK attachment.",
    )

    @field_validator("model_pool", "modelmasks", "capabilities")
    @classmethod
    def _clean_string_list(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    @field_validator("host_ws_url")
    @classmethod
    def _validate_websocket_url(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith(("ws://", "wss://")):
            raise ValueError("host_ws_url must start with ws:// or wss://")
        return value

    @field_validator("model_selector")
    @classmethod
    def _validate_model_selector(cls, value: str) -> str:
        normalized = value.strip().lower().replace("worse-", "worst-")
        if normalized == "random" or re.fullmatch(r"(?:best|worst)-[1-9][0-9]*", normalized):
            return normalized
        raise ValueError("model_selector must be random, best-N, worst-N, or worse-N")

    @field_validator("reasoning_effort")
    @classmethod
    def _validate_reasoning_effort(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if (
            normalized in _REASONING_LEVELS
            or normalized == "random"
            or re.fullmatch(r"(?:most|least)-[1-9][0-9]*", normalized)
        ):
            return normalized
        raise ValueError(
            "reasoning_effort must be an exact level, random, most-N, or least-N"
        )


class CopilotInstanceError(RuntimeError):
    """Base class for manager operations that map to an HTTP error."""


class CopilotInstanceExists(CopilotInstanceError):
    pass


class CopilotInstanceMissing(CopilotInstanceError):
    pass


class CopilotInstanceRunning(CopilotInstanceError):
    pass


def _read_config_document(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise CopilotInstanceError(f"cannot read config document '{path}': {error}") from error
    if not isinstance(document, dict):
        raise CopilotInstanceError(f"config document '{path}' is not a JSON object")
    return document


def write_config_document(path: Path, document: dict[str, Any]) -> None:
    """Atomically write the shared EMULLM configuration document."""
    with _CONFIG_IO_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        trailing_newline = ""
        if path.exists():
            existing = path.read_text(encoding="utf-8-sig")
            if existing.endswith("\r\n"):
                trailing_newline = "\r\n"
            elif existing.endswith("\n"):
                trailing_newline = "\n"
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            serialized = json.dumps(document, indent=1, ensure_ascii=True) + trailing_newline
            temporary.write_text(serialized, encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def resolve_copilot_command(configured: str | None = None) -> str:
    """Resolve the executable without selecting PowerShell's blocked .ps1 shim."""
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_absolute() or candidate.parent != Path("."):
            if candidate.is_file():
                return str(candidate.resolve())
            raise FileNotFoundError(f"Copilot command does not exist: {candidate}")
        resolved = shutil.which(configured)
        if resolved:
            return resolved
        raise FileNotFoundError(f"Copilot command is not on PATH: {configured}")

    names = ("copilot.cmd", "copilot.exe", "copilot") if os.name == "nt" else ("copilot",)
    for name in names:
        if resolved := shutil.which(name):
            return resolved
    raise FileNotFoundError("GitHub Copilot CLI was not found on PATH")


def resolve_copilot_runtime(configured: str | None = None) -> str:
    """Resolve the executable/JS entrypoint the SDK should keep resident."""
    command = Path(resolve_copilot_command(configured))
    if os.name == "nt" and command.suffix.lower() in {".cmd", ".bat", ".ps1"}:
        loader = command.parent / "node_modules" / "@github" / "copilot" / "npm-loader.js"
        if not loader.is_file():
            raise FileNotFoundError(
                f"cannot resolve the Node entrypoint behind Copilot shim '{command}'"
            )
        return str(loader.resolve())
    return str(command)


def _find_copilot_sdk(copilot_command: str) -> Path:
    command = Path(copilot_command).resolve()
    roots = [command.parent, *list(command.parents)[:3]]
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            (root / "node_modules" / "@github" / "copilot" / "node_modules" / "@github").glob(
                "copilot-*/copilot-sdk/index.js"
            )
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("the bundled Copilot SDK model catalog was not found")


def _query_copilot_models() -> list[dict[str, Any]]:
    copilot_command = resolve_copilot_command()
    node = shutil.which("node")
    if not node:
        raise FileNotFoundError("Node.js is required to query Copilot's model catalog")
    sdk = _find_copilot_sdk(copilot_command)
    script = f"""
import {{ CopilotClient }} from {json.dumps(sdk.as_uri())};
const client = new CopilotClient({{ useLoggedInUser: true, logLevel: "none" }});
try {{
  await client.start();
  console.log(JSON.stringify(await client.listModels()));
}} finally {{
  await client.stop();
}}
"""
    env = os.environ.copy()
    env.pop("COPILOT_AGENT_SESSION_ID", None)
    result = subprocess.run(  # noqa: S603 - fixed node executable and in-memory script
        [node, "--input-type=module", "-"],
        input=script,
        text=True,
        capture_output=True,
        timeout=60,
        env=env,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.strip()[-2_000:] or "no process output"
        raise CopilotInstanceError(f"Copilot model discovery failed: {detail}")
    try:
        raw_models = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise CopilotInstanceError("Copilot model discovery returned invalid JSON") from error
    if not isinstance(raw_models, list):
        raise CopilotInstanceError("Copilot model discovery did not return a list")
    models: list[dict[str, Any]] = []
    for raw in raw_models:
        if not isinstance(raw, dict) or not raw.get("id"):
            continue
        models.append(
            {
                key: raw[key]
                for key in ("id", "name", "capabilities", "billing", "policy")
                if key in raw
            }
        )
    if not models:
        raise CopilotInstanceError("Copilot model discovery returned no models")
    return models


def _annotate_model_ranks(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated = []
    for model in models:
        model_id = str(model.get("id") or "")
        rank_index = _MODEL_QUALITY_INDEX.get(model_id)
        rank = rank_index + 1 if rank_index is not None else None
        if rank is None:
            tier = "unranked"
        elif rank <= 5:
            tier = "highest"
        elif rank <= 13:
            tier = "powerful"
        elif rank <= 20:
            tier = "versatile"
        else:
            tier = "lightweight"
        annotated.append({**model, "quality_rank": rank, "quality_tier": tier})
    return sorted(
        annotated,
        key=lambda model: (
            -1
            if model.get("id") == "auto"
            else (
                int(model["quality_rank"])
                if model.get("quality_rank") is not None
                else len(_MODEL_QUALITY_ORDER) + 1
            ),
            str(model.get("name") or model.get("id") or ""),
        ),
    )


def copilot_models(*, refresh: bool = False) -> dict[str, Any]:
    """Return the live account catalog, with the user-supplied list as fallback."""
    global _MODEL_CACHE
    with _MODEL_CACHE_LOCK:
        now = time.time()
        if (
            not refresh
            and _MODEL_CACHE is not None
            and now - float(_MODEL_CACHE["fetched_at"]) < _MODEL_CACHE_TTL_SECONDS
        ):
            return dict(_MODEL_CACHE)
        try:
            models = _annotate_model_ranks(_query_copilot_models())
            result: dict[str, Any] = {
                "source": "copilot-sdk",
                "models": models,
                "fetched_at": now,
                "error": None,
            }
        except (CopilotInstanceError, FileNotFoundError, OSError, subprocess.SubprocessError) as error:
            result = {
                "source": "fallback",
                "models": _annotate_model_ranks(list(_FALLBACK_MODELS)),
                "fetched_at": now,
                "error": str(error),
            }
        _MODEL_CACHE = result
        return dict(result)


def _model_reasoning_levels(model: dict[str, Any]) -> list[str]:
    capabilities = model.get("capabilities")
    supports = capabilities.get("supports") if isinstance(capabilities, dict) else None
    levels = supports.get("reasoning_effort") if isinstance(supports, dict) else None
    return [str(level) for level in levels] if isinstance(levels, list) else []


def _is_exact_reasoning_effort(value: str | None) -> bool:
    return value in _REASONING_LEVELS


def select_copilot_model(
    config: HeadlessCopilotConfig,
    models: list[dict[str, Any]],
) -> str:
    """Choose an available model using the configured quality selector."""
    by_id = {
        str(model["id"]): model
        for model in models
        if model.get("id") and model.get("id") != "auto"
    }
    if config.model:
        metadata = by_id.get(config.model)
        if (
            metadata is not None
            and _is_exact_reasoning_effort(config.reasoning_effort)
            and config.reasoning_effort not in _model_reasoning_levels(metadata)
        ):
            raise CopilotInstanceError(
                f"model '{config.model}' does not support reasoning effort "
                f"'{config.reasoning_effort}'"
            )
        return config.model

    candidates = [
        model_id
        for model_id in (config.model_pool or list(by_id))
        if model_id != "auto"
    ]
    if _is_exact_reasoning_effort(config.reasoning_effort):
        candidates = [
            model_id
            for model_id in candidates
            if model_id not in by_id
            or config.reasoning_effort in _model_reasoning_levels(by_id[model_id])
        ]
    if not candidates:
        raise CopilotInstanceError(
            "no Copilot models satisfy the configured pool and reasoning effort"
        )

    ranked = sorted(
        candidates,
        key=lambda model_id: (
            _MODEL_QUALITY_INDEX.get(model_id, len(_MODEL_QUALITY_ORDER)),
            model_id,
        ),
    )
    if config.model_selector == "random":
        selection_pool = candidates
    else:
        direction, raw_count = config.model_selector.split("-", 1)
        count = min(int(raw_count), len(ranked))
        selection_pool = ranked[:count] if direction == "best" else list(reversed(ranked))[:count]
    return secrets.choice(selection_pool)


def select_reasoning_effort(
    configured: str | None,
    model: dict[str, Any] | None,
) -> str | None:
    """Resolve an exact/random/most-N/least-N effort for the selected model."""
    if configured is None:
        return None
    if _is_exact_reasoning_effort(configured):
        return configured
    supported = [
        level
        for level in _REASONING_LEVELS
        if model is not None and level in _model_reasoning_levels(model)
    ]
    if not supported:
        return None
    if configured == "random":
        choices = supported
    else:
        direction, raw_count = configured.split("-", 1)
        count = min(int(raw_count), len(supported))
        choices = (
            list(reversed(supported))[:count]
            if direction == "most"
            else supported[:count]
        )
    return secrets.choice(choices)


def _selected_model_prompt_limit(
    model: dict[str, Any] | None,
    context: str,
) -> int:
    if model is None:
        return 64_000
    capabilities = model.get("capabilities")
    limits = capabilities.get("limits") if isinstance(capabilities, dict) else None
    maximum = int(limits.get("max_prompt_tokens") or 0) if isinstance(limits, dict) else 0
    if context == "default":
        billing = model.get("billing")
        prices = billing.get("tokenPrices") if isinstance(billing, dict) else None
        default_limit = int(prices.get("contextMax") or 0) if isinstance(prices, dict) else 0
        if default_limit:
            maximum = min(value for value in (maximum, default_limit) if value > 0)
    return maximum or 64_000


class CopilotInstanceManager:
    """Persist and supervise headless Copilot WebSocket servant processes."""

    def __init__(
        self,
        *,
        config_path: Path,
        runtime_dir: Path,
        base_dir: Path,
        default_host_ws_url: str,
        definitions: list[dict[str, Any] | HeadlessCopilotConfig] | None = None,
        connected: Callable[[str], bool] | None = None,
        spawn: Callable[[WorkerSpec], Any] | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.runtime_dir = Path(runtime_dir)
        self.base_dir = Path(base_dir)
        self.default_host_ws_url = default_host_ws_url.rstrip("/")
        self._connected = connected or (lambda _worker_id: False)
        self._spawn_override = spawn
        self._lock = threading.RLock()
        self._configs: dict[str, HeadlessCopilotConfig] = {}
        self._supervisor = Supervisor([], spawn=self._spawn)
        generated_session = False

        for raw_definition in definitions or []:
            if isinstance(raw_definition, HeadlessCopilotConfig):
                config = raw_definition
            else:
                generated_session = generated_session or "session_id" not in raw_definition
                config = HeadlessCopilotConfig.model_validate(raw_definition)
            if config.worker_id in self._configs:
                raise ValueError(f"duplicate headless Copilot worker_id '{config.worker_id}'")
            self._configs[config.worker_id] = config
            self._supervisor.add_spec(self._spec_for(config))

        if generated_session:
            self._persist_locked()

    def _instance_dir(self, worker_id: str) -> Path:
        return self.runtime_dir / worker_id

    def _working_dir(self, config: HeadlessCopilotConfig) -> Path:
        if config.cwd:
            configured = Path(config.cwd).expanduser()
            path = configured if configured.is_absolute() else self.base_dir / configured
        else:
            path = self._instance_dir(config.worker_id) / "workspace"
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    def _runtime_config_path(self, worker_id: str) -> Path:
        return self._instance_dir(worker_id) / "servant-config.json"

    def _log_path(self, worker_id: str) -> Path:
        return self._instance_dir(worker_id) / "servant.log"

    def _runtime_status_path(self, worker_id: str) -> Path:
        return self._instance_dir(worker_id) / "runtime-status.json"

    def _spec_for(self, config: HeadlessCopilotConfig) -> WorkerSpec:
        return WorkerSpec(
            worker_id=config.worker_id,
            argv=[
                sys.executable,
                "-m",
                "emullm.copilot_servant",
                "--config",
                str(self._runtime_config_path(config.worker_id)),
            ],
            cwd=self._working_dir(config),
            role=config.role,
            autostart=config.autostart,
            modelmasks=",".join(config.modelmasks),
        )

    def _write_runtime_config(self, config: HeadlessCopilotConfig) -> None:
        instance_dir = self._instance_dir(config.worker_id)
        instance_dir.mkdir(parents=True, exist_ok=True)
        catalog = list(copilot_models()["models"])
        selected_model = select_copilot_model(config, catalog)
        selected_metadata = next(
            (model for model in catalog if model.get("id") == selected_model),
            None,
        )
        selected_reasoning_effort = select_reasoning_effort(
            config.reasoning_effort, selected_metadata
        )
        prompt_limit = _selected_model_prompt_limit(selected_metadata, config.context)
        capabilities = (
            selected_metadata.get("capabilities")
            if isinstance(selected_metadata, dict)
            else {}
        )
        limits = capabilities.get("limits") if isinstance(capabilities, dict) else {}
        vision = limits.get("vision") if isinstance(limits, dict) else {}
        supported_media_types = (
            vision.get("supported_media_types")
            if isinstance(vision, dict)
            else []
        )
        copilot_command = resolve_copilot_command(config.copilot_command)
        node_command = shutil.which("node")
        if not node_command:
            raise FileNotFoundError("Node.js is required for resident Copilot SDK servants")
        bridge_path = Path(__file__).with_name("copilot_sdk_bridge.mjs")
        if not bridge_path.is_file():
            raise FileNotFoundError(f"resident Copilot SDK bridge is missing: {bridge_path}")
        payload = config.model_dump(mode="json")
        payload.update(
            {
                "model": selected_model,
                "selected_model": selected_model,
                "configured_reasoning_effort": config.reasoning_effort,
                "reasoning_effort": selected_reasoning_effort,
                "selected_reasoning_effort": selected_reasoning_effort,
                "selected_model_max_prompt_tokens": prompt_limit,
                "selected_model_max_output_tokens": (
                    int(limits.get("max_output_tokens") or 0)
                    if isinstance(limits, dict)
                    else 0
                ),
                "selected_model_supported_media_types": (
                    [str(media_type) for media_type in supported_media_types]
                    if isinstance(supported_media_types, list)
                    else []
                ),
                "host_ws_url": config.host_ws_url or self.default_host_ws_url,
                "copilot_command": copilot_command,
                "copilot_runtime_path": resolve_copilot_runtime(config.copilot_command),
                "copilot_sdk_path": str(_find_copilot_sdk(copilot_command)),
                "node_command": str(Path(node_command).resolve()),
                "bridge_path": str(bridge_path.resolve()),
                "runtime_config_path": str(self._runtime_config_path(config.worker_id).resolve()),
                "runtime_status_path": str(self._runtime_status_path(config.worker_id).resolve()),
                "resolved_cwd": str(self._working_dir(config)),
            }
        )
        write_config_document(self._runtime_config_path(config.worker_id), payload)

    def _spawn(self, spec: WorkerSpec) -> Any:
        config = self._configs[spec.worker_id]
        self._write_runtime_config(config)
        if self._spawn_override is not None:
            return self._spawn_override(spec)

        log_path = self._log_path(spec.worker_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(_IMPORT_ROOT), env.get("PYTHONPATH", "")) if value
        )
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env.pop("COPILOT_AGENT_SESSION_ID", None)
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True
        with log_path.open("ab", buffering=0) as log:
            return subprocess.Popen(  # noqa: S603 - argv is generated without a shell
                spec.argv,
                cwd=str(spec.cwd) if spec.cwd else None,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                close_fds=True,
                **kwargs,
            )

    def _persist_locked(self) -> None:
        document = _read_config_document(self.config_path)
        document["headless_copilots"] = [
            config.model_dump(mode="json") for config in self._configs.values()
        ]
        write_config_document(self.config_path, document)

    def _require(self, worker_id: str) -> HeadlessCopilotConfig:
        try:
            return self._configs[worker_id]
        except KeyError as error:
            raise CopilotInstanceMissing(f"no headless Copilot servant '{worker_id}'") from error

    def create(self, config: HeadlessCopilotConfig, *, start: bool = True) -> dict[str, Any]:
        with self._lock:
            if config.worker_id in self._configs:
                raise CopilotInstanceExists(f"headless Copilot servant '{config.worker_id}' already exists")
            if start:
                resolve_copilot_command(config.copilot_command)
            self._configs[config.worker_id] = config
            self._supervisor.add_spec(self._spec_for(config))
            try:
                if start:
                    self._supervisor.start(config.worker_id)
                self._persist_locked()
            except Exception:
                self._supervisor.remove_spec(config.worker_id)
                self._configs.pop(config.worker_id, None)
                raise
            return self.get(config.worker_id)

    def update(
        self, worker_id: str, config: HeadlessCopilotConfig, *, restart: bool = True
    ) -> dict[str, Any]:
        if worker_id != config.worker_id:
            raise ValueError("path worker_id must match config.worker_id")
        with self._lock:
            previous = self._require(worker_id)
            was_running = self._is_running(worker_id)
            if was_running and not restart:
                raise CopilotInstanceRunning("a running servant must be restarted to apply configuration")
            if was_running:
                self._supervisor.stop(worker_id)
            self._supervisor.remove_spec(worker_id)
            self._configs[worker_id] = config
            self._supervisor.add_spec(self._spec_for(config))
            try:
                if was_running:
                    resolve_copilot_command(config.copilot_command)
                    self._supervisor.start(worker_id)
                self._persist_locked()
            except Exception:
                self._supervisor.remove_spec(worker_id)
                self._configs[worker_id] = previous
                self._supervisor.add_spec(self._spec_for(previous))
                if was_running:
                    self._supervisor.start(worker_id)
                raise
            return self.get(worker_id)

    def delete(self, worker_id: str) -> dict[str, Any]:
        with self._lock:
            self._require(worker_id)
            self._supervisor.remove_spec(worker_id)
            self._configs.pop(worker_id)
            self._persist_locked()
            return {"worker_id": worker_id, "deleted": True}

    def start(self, worker_id: str) -> dict[str, Any]:
        with self._lock:
            config = self._require(worker_id)
            if self._is_running(worker_id):
                return {**self.get(worker_id), "started": False}
            resolve_copilot_command(config.copilot_command)
            started = self._supervisor.start(worker_id)
            return {**self.get(worker_id), "started": started}

    def stop(self, worker_id: str) -> dict[str, Any]:
        with self._lock:
            self._require(worker_id)
            stopped = self._graceful_stop(worker_id)
            return {**self.get(worker_id), "stopped": stopped}

    def restart(self, worker_id: str) -> dict[str, Any]:
        with self._lock:
            config = self._require(worker_id)
            resolve_copilot_command(config.copilot_command)
            self._graceful_stop(worker_id)
            started = self._supervisor.start(worker_id)
            return {**self.get(worker_id), "restarted": started}

    def reset_session(self, worker_id: str) -> dict[str, Any]:
        with self._lock:
            config = self._require(worker_id)
            replacement = config.model_copy(update={"session_id": uuid.uuid4()})
            return self.update(worker_id, replacement, restart=True)

    def _is_running(self, worker_id: str) -> bool:
        row = next(row for row in self._supervisor.status() if row["worker_id"] == worker_id)
        return bool(row["running"])

    def _graceful_stop(self, worker_id: str) -> bool:
        spec = next(spec for spec in self._supervisor.specs() if spec.worker_id == worker_id)
        process = spec.process
        if process is None or process.poll() is not None:
            return False
        if (
            os.name == "nt"
            and hasattr(signal, "CTRL_BREAK_EVENT")
            and hasattr(process, "send_signal")
        ):
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT)
                process.wait(timeout=20)
                spec.process = None
                return True
            except (OSError, subprocess.TimeoutExpired):
                pass
        return self._supervisor.stop(worker_id)

    def start_autostart(self) -> list[str]:
        with self._lock:
            for config in self._configs.values():
                if config.autostart:
                    resolve_copilot_command(config.copilot_command)
            return self._supervisor.start_autostart()

    def stop_all(self) -> None:
        with self._lock:
            for worker_id in list(self._configs):
                self._graceful_stop(worker_id)

    def get(self, worker_id: str) -> dict[str, Any]:
        with self._lock:
            config = self._require(worker_id)
            process = next(
                row for row in self._supervisor.status() if row["worker_id"] == worker_id
            )
            runtime = self._runtime_status(worker_id)
            if not process["running"] and runtime:
                runtime = {**runtime, "running": False}
            return {
                "worker_id": worker_id,
                "kind": "headless-copilot",
                "running": process["running"],
                "connected": bool(self._connected(worker_id)),
                "pid": process["pid"],
                "returncode": process["returncode"],
                "session_id": str(config.session_id),
                "model": config.model,
                "selected_model": self._selected_model(worker_id),
                "model_pool": list(config.model_pool),
                "model_selector": config.model_selector,
                "reasoning_effort": config.reasoning_effort,
                "selected_reasoning_effort": self._selected_reasoning_effort(worker_id),
                "modelmasks": list(config.modelmasks),
                "role": config.role,
                "autostart": config.autostart,
                "allow_all": config.allow_all,
                "cwd": str(self._working_dir(config)),
                "log_path": str(self._log_path(worker_id).resolve()),
                "runtime": runtime,
                "config": config.model_dump(mode="json"),
            }

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self.get(worker_id) for worker_id in self._configs]

    def next_worker_id(self) -> str:
        with self._lock:
            used = set(self._configs)
        index = 1
        while (
            f"worker-copilot-{index}" in used
            or 5 <= index <= 8
        ):
            index += 1
        return f"worker-copilot-{index}"

    def _selected_model(self, worker_id: str) -> str | None:
        path = self._runtime_config_path(worker_id)
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig")).get("selected_model")
        except (OSError, json.JSONDecodeError, AttributeError):
            return None
        return str(value) if value else None

    def _selected_reasoning_effort(self, worker_id: str) -> str | None:
        path = self._runtime_config_path(worker_id)
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig")).get(
                "selected_reasoning_effort"
            )
        except (OSError, json.JSONDecodeError, AttributeError):
            return None
        return str(value) if value else None

    def _runtime_status(self, worker_id: str) -> dict[str, Any]:
        path = self._runtime_status_path(worker_id)
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def tail_log(self, worker_id: str, lines: int) -> str:
        with self._lock:
            self._require(worker_id)
            path = self._log_path(worker_id)
        if not path.exists():
            return ""
        try:
            return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
        except OSError as error:
            raise CopilotInstanceError(f"cannot read servant log '{path}': {error}") from error


_manager: CopilotInstanceManager | None = None


def set_manager(manager: CopilotInstanceManager | None) -> None:
    global _manager
    _manager = manager


def get_manager() -> CopilotInstanceManager | None:
    return _manager


def manager_status() -> list[dict[str, Any]]:
    return _manager.list() if _manager is not None else []


def _require_manager() -> CopilotInstanceManager:
    if _manager is None:
        raise HTTPException(status_code=409, detail="headless Copilot manager is not active")
    return _manager


def _map_error(error: Exception) -> HTTPException:
    if isinstance(error, CopilotInstanceMissing):
        return HTTPException(status_code=404, detail=str(error))
    if isinstance(error, (CopilotInstanceExists, CopilotInstanceRunning)):
        return HTTPException(status_code=409, detail=str(error))
    if isinstance(error, (ValueError, FileNotFoundError)):
        return HTTPException(status_code=422, detail=str(error))
    if isinstance(error, (OSError, subprocess.SubprocessError)):
        return HTTPException(status_code=503, detail=str(error))
    if isinstance(error, CopilotInstanceError):
        return HTTPException(status_code=500, detail=str(error))
    return HTTPException(status_code=500, detail=f"{type(error).__name__}: {error}")


@router.get("/admin/emullm/copilots")
@router.get("/emullm/admin/copilots")
def list_copilots() -> dict[str, Any]:
    manager = get_manager()
    try:
        command = resolve_copilot_command()
    except FileNotFoundError:
        command = None
    return {
        "manager_active": manager is not None,
        "copilot_available": command is not None,
        "copilot_command": command,
        "next_worker_id": manager.next_worker_id() if manager else "worker-copilot-1",
        "instances": manager.list() if manager else [],
    }


@router.get("/admin/emullm/copilots/schema")
@router.get("/emullm/admin/copilots/schema")
def copilot_config_schema() -> dict[str, Any]:
    return HeadlessCopilotConfig.model_json_schema()


@router.get("/admin/emullm/copilots/models")
@router.get("/emullm/admin/copilots/models")
def list_copilot_models(refresh: bool = False) -> dict[str, Any]:
    """List models available to the authenticated Copilot CLI account."""
    return copilot_models(refresh=refresh)


@router.get("/admin/emullm/copilots/{worker_id}")
@router.get("/emullm/admin/copilots/{worker_id}")
def get_copilot(worker_id: str) -> dict[str, Any]:
    try:
        return _require_manager().get(worker_id)
    except Exception as error:
        raise _map_error(error) from error


@router.post("/admin/emullm/copilots")
@router.post("/emullm/admin/copilots")
def create_copilot(config: HeadlessCopilotConfig, start: bool = True) -> dict[str, Any]:
    try:
        return _require_manager().create(config, start=start)
    except Exception as error:
        raise _map_error(error) from error


@router.put("/admin/emullm/copilots/{worker_id}")
@router.put("/emullm/admin/copilots/{worker_id}")
def update_copilot(
    worker_id: str, config: HeadlessCopilotConfig, restart: bool = True
) -> dict[str, Any]:
    try:
        return _require_manager().update(worker_id, config, restart=restart)
    except Exception as error:
        raise _map_error(error) from error


@router.delete("/admin/emullm/copilots/{worker_id}")
@router.delete("/emullm/admin/copilots/{worker_id}")
def delete_copilot(worker_id: str) -> dict[str, Any]:
    try:
        return _require_manager().delete(worker_id)
    except Exception as error:
        raise _map_error(error) from error


def _copilot_action(worker_id: str, action: str) -> dict[str, Any]:
    manager = _require_manager()
    try:
        return getattr(manager, action)(worker_id)
    except Exception as error:
        raise _map_error(error) from error


@router.post("/admin/emullm/copilots/{worker_id}/start")
@router.post("/emullm/admin/copilots/{worker_id}/start")
def start_copilot(worker_id: str) -> dict[str, Any]:
    return _copilot_action(worker_id, "start")


@router.post("/admin/emullm/copilots/{worker_id}/stop")
@router.post("/emullm/admin/copilots/{worker_id}/stop")
def stop_copilot(worker_id: str) -> dict[str, Any]:
    return _copilot_action(worker_id, "stop")


@router.post("/admin/emullm/copilots/{worker_id}/restart")
@router.post("/emullm/admin/copilots/{worker_id}/restart")
def restart_copilot(worker_id: str) -> dict[str, Any]:
    return _copilot_action(worker_id, "restart")


@router.post("/admin/emullm/copilots/{worker_id}/reset-session")
@router.post("/emullm/admin/copilots/{worker_id}/reset-session")
def reset_copilot_session(worker_id: str) -> dict[str, Any]:
    return _copilot_action(worker_id, "reset_session")


@router.get("/admin/emullm/copilots/{worker_id}/log", response_class=PlainTextResponse)
@router.get("/emullm/admin/copilots/{worker_id}/log", response_class=PlainTextResponse)
def copilot_log(worker_id: str, lines: int = 200) -> str:
    if not 1 <= lines <= 5_000:
        raise HTTPException(status_code=422, detail="lines must be between 1 and 5000")
    try:
        return _require_manager().tail_log(worker_id, lines)
    except Exception as error:
        raise _map_error(error) from error

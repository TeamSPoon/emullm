"""Non-interactive Codex CLI adapter for EMULLM's worker WebSocket.

Each servant connects to EMULLM's shared ``/emullm/ws`` endpoint as a
``worker-codex-*`` worker and answers relayed requests by running the OpenAI
Codex CLI once per request in its non-interactive ``codex exec`` mode. The
Codex agent -- with its own tools, running in its own workspace against
whatever model its ``CODEX_HOME/config.toml`` points at -- produces the
answer; EMULLM merely relays it, exactly like the headless Copilot servant.

Unlike the resident Copilot servant (one long-lived SDK process reused across
requests), a Codex servant spawns ``codex exec`` fresh per request. This keeps
the adapter tiny and lets each request run the full agentic Codex harness
(shell, file edits, its own web/vision abilities) confined to the servant's
workspace. Tools declared by the CALLER are surfaced in the prompt so the
Codex agent can emit the tool-call envelope for the caller to execute; it must
not run the caller's tools itself.

Wire protocol (identical to :mod:`emullm.copilot_servant`):

* connect, receive ``{"type": "hello", ...}``
* send ``{"type": "register", "worker_kind": "headless-codex", ...}``
* per relayed ``{"type": "request", "id", "prompt", ...}``: reply with
  ``{"type": "accept", "id"}`` then ``{"type": "reply", "id", "content"}``
  (or ``{"type": "reject"/"not_ready", ...}`` on failure).

Run it directly for debugging::

    python -m emullm.codex_servant --config <servant-config.json>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import websockets
from pydantic import BaseModel, ConfigDict, Field
from websockets.exceptions import ConnectionClosed

from .worker import worker_socket_url


DEFAULT_SYSTEM_PROMPT = (
    "Act as the model requested by the OpenAI-compatible caller. Answer the "
    "request directly and return only the assistant response that should be "
    "sent to the caller. You MAY use your own tools and internal abilities "
    "(shell, file inspection, retrieval, reasoning) to produce the answer, and "
    "the caller need not know how you did it. But if the request lists the "
    "CALLER'S tools and asks you to emit tool calls, do NOT execute those "
    "caller tools yourself -- they run in the caller's environment; instead "
    "respond with exactly the requested tool-call JSON envelope and nothing "
    "else, so the caller can run them."
)


class CodexInvocationError(RuntimeError):
    """Raised when a single ``codex exec`` invocation fails."""


class CodexServantRuntimeConfig(BaseModel):
    """Runtime document written by :class:`emullm.codex_api.CodexInstanceManager`.

    This mirrors the shape of the headless-copilot ``servant-config.json`` but
    only carries what the Codex servant actually needs: how to reach the relay,
    how to launch ``codex exec``, and which environment to launch it under.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    worker_id: str
    session_id: str = ""
    host_ws_url: str
    role: str = "headless-codex"
    modelmasks: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # Launch: prefer a Node entrypoint (codex.js) via node_command; fall back to
    # a native ``codex`` binary in codex_command when node_command is empty.
    node_command: str = ""
    codex_entry: str = ""
    codex_command: str = ""
    resolved_cwd: str
    codex_home: str = ""
    env_key: str = "CODEX_API_KEY"
    api_key: str = "emullm-local"

    # Backing model + sandbox controls passed to ``codex exec``.
    selected_model: str = ""
    sandbox_mode: str = "workspace-write"
    full_auto: bool = True
    ephemeral: bool = False
    extra_config: list[str] = Field(default_factory=list)

    # Timing.
    idle_timeout_seconds: float = 15.0
    reply_timeout_seconds: float = 1800.0
    reconnect_seconds: float = 2.0
    rest_min_seconds: float = 1.0
    rest_max_seconds: float = 20.0
    max_prompt_chars: int = 4_000_000

    runtime_status_path: str = ""


def build_prompt(config: CodexServantRuntimeConfig, request: dict[str, Any]) -> str:
    """Turn one relay offer into a bounded prompt for a single ``codex exec``."""

    sections = [
        config.system_prompt,
        f"Requested API model: {request.get('model') or '(unspecified)'}",
        f"Request kind: {request.get('kind') or 'chat'}",
    ]
    if instruction := request.get("persona_instruction"):
        sections.append(f"Persona instruction: {instruction}")
    for key in ("images", "audio", "files", "attachments"):
        if request.get(key):
            sections.append(
                f"Attached {key}: {json.dumps(request[key], ensure_ascii=False)}"
            )
    sections.extend(
        [
            "OpenAI-compatible request content:",
            str(request.get("prompt") or ""),
            "Return only the assistant response content.",
        ]
    )
    prompt = "\n\n".join(sections)
    if len(prompt) > config.max_prompt_chars:
        raise CodexInvocationError(
            f"request prompt is {len(prompt)} characters; configured maximum is "
            f"{config.max_prompt_chars}"
        )
    return prompt


def _codex_argv(config: CodexServantRuntimeConfig, last_message_file: Path) -> list[str]:
    """Build the ``codex exec`` argv (prompt is fed on stdin via ``-``)."""

    if config.node_command and config.codex_entry:
        argv = [config.node_command, config.codex_entry]
    elif config.codex_command:
        argv = [config.codex_command]
    else:
        raise CodexInvocationError(
            "no Codex launch command configured (node_command+codex_entry or "
            "codex_command required)"
        )
    argv += ["exec", "--skip-git-repo-check", "-C", config.resolved_cwd]
    argv += ["--color", "never"]
    argv += ["-o", str(last_message_file)]
    if config.ephemeral:
        argv.append("--ephemeral")
    if config.full_auto:
        # Managed, unattended worker: run without approval prompts or sandbox
        # (the equivalent of the headless Copilot's --allow-all / --no-ask-user).
        argv.append("--dangerously-bypass-approvals-and-sandbox")
    elif config.sandbox_mode:
        argv += ["-s", config.sandbox_mode]
    if config.selected_model:
        argv += ["-m", config.selected_model]
    for override in config.extra_config:
        if override:
            argv += ["-c", override]
    argv.append("-")
    return argv


def _codex_env(config: CodexServantRuntimeConfig) -> dict[str, str]:
    env = os.environ.copy()
    if config.codex_home:
        env["CODEX_HOME"] = config.codex_home
    if config.env_key:
        env[config.env_key] = config.api_key or "emullm-local"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # Keep a child Codex from inheriting an ambient session identity.
    env.pop("COPILOT_AGENT_SESSION_ID", None)
    return env


async def run_codex_once(
    config: CodexServantRuntimeConfig,
    prompt: str,
    request_id: str,
) -> str:
    """Run one ``codex exec`` and return the agent's final message text."""

    workspace = Path(config.resolved_cwd)
    workspace.mkdir(parents=True, exist_ok=True)
    last_message_file = workspace / f".emullm-codex-{request_id}.out"
    last_message_file.unlink(missing_ok=True)
    argv = _codex_argv(config, last_message_file)
    env = _codex_env(config)
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=config.resolved_cwd,
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(prompt.encode("utf-8")),
            timeout=config.reply_timeout_seconds,
        )
    except asyncio.TimeoutError as error:
        with _suppress():
            process.kill()
        raise CodexInvocationError(
            f"codex exec timed out after {config.reply_timeout_seconds}s"
        ) from error

    content = ""
    if last_message_file.is_file():
        try:
            content = last_message_file.read_text(encoding="utf-8-sig").strip()
        except OSError:
            content = ""
        last_message_file.unlink(missing_ok=True)
    stdout_text = (stdout or b"").decode("utf-8", errors="replace").strip()
    stderr_text = (stderr or b"").decode("utf-8", errors="replace").strip()
    if not content:
        content = stdout_text
    if process.returncode != 0 and not content:
        detail = stderr_text or stdout_text or f"exit code {process.returncode}"
        raise CodexInvocationError(f"codex exec failed: {detail[-2000:]}")
    if not content:
        raise CodexInvocationError(
            "codex exec produced no output" + (f": {stderr_text[-500:]}" if stderr_text else "")
        )
    return content


class _suppress:
    def __enter__(self) -> "_suppress":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return True


def declared_capabilities(config: CodexServantRuntimeConfig) -> dict[str, bool]:
    capabilities: dict[str, bool] = {}
    for configured in config.capabilities:
        normalized = configured.strip()
        if not normalized:
            continue
        if normalized[0] in "!-":
            capabilities[normalized[1:].strip()] = False
        else:
            capabilities[normalized] = True
    return capabilities


def registration_payload(config: CodexServantRuntimeConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "register",
        "role": config.role,
        "capabilities": declared_capabilities(config),
        "worker_kind": "headless-codex",
        "description": (
            "Non-interactive OpenAI Codex CLI servant"
            + (f" backed by {config.selected_model}" if config.selected_model else "")
            + "; runs `codex exec` per request."
        ),
    }
    if config.selected_model:
        payload["runtime_model"] = config.selected_model
    if config.modelmasks:
        payload["modelmasks"] = ",".join(config.modelmasks)
    return payload


def _write_status(config: CodexServantRuntimeConfig, **updates: Any) -> None:
    path = config.runtime_status_path
    if not path:
        return
    status_path = Path(path)
    try:
        existing: dict[str, Any] = {}
        if status_path.is_file():
            existing = json.loads(status_path.read_text(encoding="utf-8-sig")) or {}
        if not isinstance(existing, dict):
            existing = {}
        existing.update(updates)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except (OSError, ValueError):
        pass


async def _handle_request(
    websocket: Any,
    config: CodexServantRuntimeConfig,
    request: dict[str, Any],
    counters: dict[str, int],
) -> None:
    request_id = str(request.get("id") or "")
    if not request_id:
        return
    required = request.get("required_capabilities")
    declared = declared_capabilities(config)
    declined = (
        [
            capability
            for capability in required
            if isinstance(capability, str) and declared.get(capability) is False
        ]
        if isinstance(required, list)
        else []
    )
    if declined:
        reason = f"servant explicitly declines: {', '.join(sorted(declined))}"
        await websocket.send(
            json.dumps({"type": "reject", "id": request_id, "reason": reason})
        )
        print(f"REJECTED {request_id}: {reason}", flush=True)
        return
    await websocket.send(json.dumps({"type": "accept", "id": request_id}))
    try:
        prompt = build_prompt(config, request)
        content = await run_codex_once(config, prompt, request_id)
    except CodexInvocationError as error:
        print(f"REJECTED {request_id}: {error}", flush=True)
        await websocket.send(
            json.dumps({"type": "reject", "id": request_id, "reason": str(error)})
        )
        return
    counters["requests"] = counters.get("requests", 0) + 1
    await websocket.send(
        json.dumps({"type": "reply", "id": request_id, "content": content})
    )
    _write_status(
        config,
        requests=counters["requests"],
        last_request_at=time.time(),
        last_model=request.get("model"),
    )
    print(f"REPLIED {request_id} via codex exec", flush=True)


async def _run_connection(
    config: CodexServantRuntimeConfig,
    ws_url: str,
    counters: dict[str, int],
) -> None:
    async with websockets.connect(
        ws_url,
        open_timeout=60,
        ping_interval=30,
        ping_timeout=60,
        max_size=16 * 1024 * 1024,
    ) as websocket:
        hello = json.loads(await websocket.recv())
        if not isinstance(hello, dict) or hello.get("type") != "hello":
            raise CodexInvocationError(f"unexpected relay greeting: {hello!r}")
        await websocket.send(json.dumps(registration_payload(config)))
        print(f"CONNECTED worker={config.worker_id}", flush=True)
        _write_status(config, connected=True, connected_at=time.time())
        try:
            while True:
                try:
                    raw = await asyncio.wait_for(
                        websocket.recv(), timeout=config.idle_timeout_seconds
                    )
                except asyncio.TimeoutError:
                    return
                try:
                    message = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                if not isinstance(message, dict):
                    continue
                kind = message.get("type")
                if kind == "shutdown":
                    print(
                        f"SHUTDOWN worker={config.worker_id}: "
                        f"{message.get('reason') or 'relay requested shutdown'}",
                        flush=True,
                    )
                    raise _ServantShutdown()
                if kind in {"ping", "hello", "accept", "reply"}:
                    continue
                if kind != "request":
                    continue
                await _handle_request(websocket, config, message, counters)
        finally:
            _write_status(config, connected=False)


class _ServantShutdown(Exception):
    """Internal signal that the relay asked this servant to stop."""


async def run_servant(config: CodexServantRuntimeConfig) -> None:
    counters: dict[str, int] = {"requests": 0}
    _write_status(
        config,
        adapter_pid=os.getpid(),
        connected=False,
        running=True,
        model=config.selected_model or None,
        started_at=time.time(),
        requests=0,
    )
    ws_url = worker_socket_url(
        config.host_ws_url, config.worker_id, ",".join(config.modelmasks)
    )
    print(
        f"START worker={config.worker_id} websocket={ws_url} "
        f"model={config.selected_model or '(codex default)'}",
        flush=True,
    )
    while True:
        try:
            await _run_connection(config, ws_url, counters)
        except _ServantShutdown:
            _write_status(config, connected=False, running=False)
            return
        except (ConnectionClosed, OSError, ConnectionRefusedError) as error:
            print(f"DISCONNECTED worker={config.worker_id}: {error}", flush=True)
        except CodexInvocationError as error:
            print(f"CONNECTION ERROR worker={config.worker_id}: {error}", flush=True)
        except Exception as error:  # noqa: BLE001 -- keep the servant loop alive
            print(f"ERROR worker={config.worker_id}: {error}", flush=True)
        rest = random.uniform(
            min(config.rest_min_seconds, config.rest_max_seconds),
            config.rest_max_seconds,
        )
        await asyncio.sleep(max(config.reconnect_seconds, rest))


def load_runtime_config(path: Path) -> CodexServantRuntimeConfig:
    document = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return CodexServantRuntimeConfig.model_validate(document)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_runtime_config(args.config)
    try:
        asyncio.run(run_servant(config))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()

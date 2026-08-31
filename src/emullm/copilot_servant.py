"""Persistent-session Copilot CLI adapter for EMULLM's worker WebSocket."""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import websockets
from pydantic import BaseModel, ConfigDict, Field
from websockets.exceptions import ConnectionClosed

from .copilot_api import HeadlessCopilotConfig
from .worker import worker_socket_url


class ServantRuntimeConfig(HeadlessCopilotConfig):
    model_config = ConfigDict(extra="forbid")

    host_ws_url: str
    copilot_command: str
    resolved_cwd: str
    selected_model: str
    configured_reasoning_effort: str | None = None
    selected_reasoning_effort: str | None = None
    selected_model_max_prompt_tokens: int = 64_000
    selected_model_max_output_tokens: int = 0
    selected_model_supported_media_types: list[str] = Field(default_factory=list)
    copilot_runtime_path: str
    copilot_sdk_path: str
    node_command: str
    bridge_path: str
    runtime_config_path: str
    runtime_status_path: str


class CopilotInvocationError(RuntimeError):
    pass


def build_prompt(config: ServantRuntimeConfig, request: dict[str, Any]) -> str:
    """Turn one relay offer into a bounded prompt for the persistent session."""
    sections = [
        config.system_prompt,
        f"Requested API model: {request.get('model') or '(unspecified)'}",
        f"Request kind: {request.get('kind') or 'chat'}",
    ]
    if instruction := request.get("persona_instruction"):
        sections.append(f"Persona instruction: {instruction}")
    for key in ("images", "audio", "files", "attachments"):
        if request.get(key):
            sections.append(f"Attached {key}: {json.dumps(request[key], ensure_ascii=False)}")
    sections.extend(
        [
            "OpenAI-compatible request content:",
            str(request.get("prompt") or ""),
            "Return only the assistant response content.",
        ]
    )
    prompt = "\n\n".join(sections)
    if len(prompt) > config.max_prompt_chars:
        raise CopilotInvocationError(
            f"request prompt is {len(prompt)} characters; configured maximum is {config.max_prompt_chars}"
        )
    return prompt


class CopilotRunner:
    def __init__(self, config: ServantRuntimeConfig) -> None:
        self.config = config
        self._active: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail: list[str] = []
        self._lock = asyncio.Lock()
        self._request_count = 0
        self._cancellation_count = 0
        self._bridge_request_id: str | None = None

    async def start(self) -> None:
        if self._active is not None and self._active.returncode is None:
            return
        started = time.monotonic()
        self._active = await asyncio.create_subprocess_exec(
            self.config.node_command,
            self.config.bridge_path,
            self.config.runtime_config_path,
            cwd=self.config.resolved_cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._stderr_task = asyncio.create_task(self._pump_stderr(self._active))
        try:
            ready = await asyncio.wait_for(self._read_message(), timeout=90)
        except Exception:
            await self._terminate(self._active)
            raise
        if ready.get("type") != "ready":
            await self._terminate(self._active)
            raise CopilotInvocationError(
                f"resident Copilot bridge did not become ready: {ready}"
            )
        self._write_status(
            running=True,
            bridge_pid=self._active.pid,
            runtime_pid=ready.get("runtime_pid"),
            session_id=ready.get("session_id"),
            model=ready.get("model"),
            resumed=bool(ready.get("resumed")),
            startup_ms=round((time.monotonic() - started) * 1000),
            requests=0,
            cancellations=0,
            warmup_completed=False,
            warmup_duration_ms=None,
        )

    async def run(self, request: dict[str, Any]) -> str:
        prompt = build_prompt(self.config, request)
        request_id = str(request.get("id") or "")
        chunks = self._prompt_chunks(prompt)
        attachments = await self._sdk_attachments(request)
        async with self._lock:
            await self.start()
            started = time.monotonic()
            try:
                if len(chunks) > 1:
                    for index, chunk in enumerate(chunks, start=1):
                        await self._invoke(
                            f"{request_id}-chunk-{index}",
                            (
                                f"Ingest chunk {index}/{len(chunks)} of one oversized "
                                "request. Preserve its facts and instructions for the final "
                                "answer, but do not answer yet. Reply only CHUNK-RECEIVED.\n\n"
                                f"{chunk}"
                            ),
                        )
                    response = await self._invoke(
                        request_id,
                        (
                            f"All {len(chunks)} chunks of the oversized request have now "
                            "been ingested. Produce the final assistant answer to that "
                            "request now, following the instructions contained in the chunks."
                        ),
                        attachments=attachments,
                    )
                else:
                    response = await self._invoke(
                        request_id,
                        prompt,
                        attachments=attachments,
                    )
            except asyncio.CancelledError:
                self._cancellation_count += 1
                active_id = self._bridge_request_id
                if active_id:
                    try:
                        await asyncio.shield(self._cancel_request(active_id))
                    except (CopilotInvocationError, TimeoutError):
                        pass
                self._bridge_request_id = None
                self._write_status(cancellations=self._cancellation_count)
                raise
            if response.get("type") == "error":
                raise CopilotInvocationError(str(response.get("error") or "resident Copilot error"))
            output = str(response.get("content") or "")
            if not output:
                raise CopilotInvocationError("Copilot returned an empty response")
            if len(output) > self.config.max_output_chars:
                raise CopilotInvocationError(
                    f"Copilot returned {len(output)} characters; configured maximum is "
                    f"{self.config.max_output_chars}"
                )
            self._request_count += 1
            self._write_status(
                requests=self._request_count,
                last_request_id=request_id,
                last_duration_ms=round((time.monotonic() - started) * 1000),
                last_chunk_count=len(chunks),
                last_attachment_count=len(attachments),
                last_completed_at=time.time(),
            )
            return output

    async def _sdk_attachments(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        records = request.get("attachments")
        if not isinstance(records, list):
            return []
        return [
            await asyncio.to_thread(self._load_sdk_attachment, record)
            for record in records
            if isinstance(record, dict)
        ]

    def _load_sdk_attachment(self, record: dict[str, Any]) -> dict[str, Any]:
        raw_url = str(record.get("url") or "")
        if not raw_url:
            raise CopilotInvocationError("attachment record has no URL")
        base = self.config.host_ws_url.replace("ws://", "http://", 1).replace(
            "wss://", "https://", 1
        )
        url = urljoin(base.rstrip("/") + "/", raw_url.lstrip("/"))
        try:
            with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - relay URL is operator config
                data = response.read(self.config.max_attachment_bytes + 1)
        except OSError as error:
            raise CopilotInvocationError(
                f"could not fetch attachment '{record.get('name') or raw_url}': {error}"
            ) from error
        if len(data) > self.config.max_attachment_bytes:
            raise CopilotInvocationError(
                f"attachment '{record.get('name') or raw_url}' exceeds the "
                f"{self.config.max_attachment_bytes}-byte servant limit"
            )
        if not data:
            raise CopilotInvocationError(
                f"attachment '{record.get('name') or raw_url}' is empty"
            )
        return {
            "type": "blob",
            "data": base64.b64encode(data).decode("ascii"),
            "mimeType": str(record.get("mime_type") or "application/octet-stream"),
            "displayName": str(record.get("name") or "attachment"),
        }

    def _prompt_chunks(self, prompt: str) -> list[str]:
        token_budget = self.config.chunk_tokens or max(
            1_000, self.config.selected_model_max_prompt_tokens - 8_192
        )
        character_budget = max(3_000, token_budget * 3)
        if len(prompt) <= character_budget:
            return [prompt]
        if not self.config.chunk_long_prompts:
            raise CopilotInvocationError(
                f"request needs approximately {(len(prompt) + 2) // 3} tokens, "
                f"above the selected model chunk budget of {token_budget}"
            )
        chunks = [
            prompt[offset : offset + character_budget]
            for offset in range(0, len(prompt), character_budget)
        ]
        if len(chunks) > self.config.max_chunks:
            raise CopilotInvocationError(
                f"request requires {len(chunks)} chunks; configured maximum is "
                f"{self.config.max_chunks}"
            )
        return chunks

    async def _invoke(
        self,
        request_id: str,
        prompt: str,
        *,
        attachments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self._bridge_request_id = request_id
        self._send(
            {
                "type": "request",
                "id": request_id,
                "prompt": prompt,
                "attachments": attachments or [],
                "timeout_ms": round(self.config.timeout_seconds * 1000),
            }
        )
        try:
            response = await asyncio.wait_for(
                self._read_for_request(request_id),
                timeout=self.config.timeout_seconds + 10,
            )
        except TimeoutError as error:
            try:
                await self._cancel_request(request_id)
            except (CopilotInvocationError, TimeoutError):
                pass
            self._bridge_request_id = None
            raise CopilotInvocationError(
                f"Copilot timed out after {self.config.timeout_seconds:g} seconds"
            ) from error
        except asyncio.CancelledError:
            raise
        self._bridge_request_id = None
        if response.get("type") == "error":
            raise CopilotInvocationError(
                str(response.get("error") or "resident Copilot error")
            )
        return response

    async def warmup(self) -> str | None:
        if not self.config.warmup:
            return None
        started = time.monotonic()
        response = await self.run(
            {
                "id": "startup-warmup",
                "model": self.config.selected_model,
                "kind": "warmup",
                "prompt": self.config.warmup_prompt,
            }
        )
        self._write_status(
            warmup_completed=True,
            warmup_duration_ms=round((time.monotonic() - started) * 1000),
            warmup_reply=response[:200],
            warmup_completed_at=time.time(),
        )
        return response

    async def close(self) -> None:
        process = self._active
        if process is None:
            return
        if process.returncode is None:
            try:
                self._send({"type": "shutdown"})
                await asyncio.wait_for(process.wait(), timeout=10)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                await self._terminate(process)
        if self._stderr_task is not None:
            try:
                await asyncio.wait_for(self._stderr_task, timeout=2)
            except TimeoutError:
                self._stderr_task.cancel()
        self._active = None
        self._write_status(running=False, stopped_at=time.time())

    def _send(self, message: dict[str, Any]) -> None:
        process = self._active
        if process is None or process.stdin is None or process.returncode is not None:
            raise CopilotInvocationError("resident Copilot bridge is not running")
        process.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"))

    async def _read_message(self) -> dict[str, Any]:
        process = self._active
        if process is None or process.stdout is None:
            raise CopilotInvocationError("resident Copilot bridge has no stdout")
        line = await process.stdout.readline()
        if not line:
            detail = "\n".join(self._stderr_tail[-20:]) or "no process output"
            raise CopilotInvocationError(
                f"resident Copilot bridge exited with code {process.returncode}: {detail}"
            )
        try:
            message = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CopilotInvocationError(
                f"resident Copilot bridge returned invalid JSON: {line[:500]!r}"
            ) from error
        if not isinstance(message, dict):
            raise CopilotInvocationError("resident Copilot bridge returned a non-object message")
        return message

    async def _read_for_request(self, request_id: str) -> dict[str, Any]:
        while True:
            message = await self._read_message()
            if message.get("id") == request_id and message.get("type") in {
                "response",
                "error",
                "cancelled",
            }:
                return message

    async def _cancel_request(self, request_id: str) -> None:
        self._send({"type": "cancel", "id": request_id})
        while True:
            message = await asyncio.wait_for(self._read_message(), timeout=10)
            if message.get("id") == request_id and message.get("type") == "cancelled":
                return

    async def _pump_stderr(self, process: asyncio.subprocess.Process) -> None:
        if process.stderr is None:
            return
        while line := await process.stderr.readline():
            text = line.decode("utf-8", errors="replace").rstrip()
            if not text:
                continue
            self._stderr_tail.append(text)
            del self._stderr_tail[:-100]
            print(f"[copilot-sdk] {text}", flush=True)

    def _write_status(self, **updates: Any) -> None:
        path = Path(self.config.runtime_status_path)
        current: dict[str, Any] = {}
        try:
            if path.is_file():
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    current = loaded
        except (OSError, json.JSONDecodeError):
            current = {}
        current.update(updates)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    @staticmethod
    async def _terminate(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            process.kill()
            await process.wait()


def declared_capabilities(config: ServantRuntimeConfig) -> dict[str, bool]:
    capabilities: dict[str, bool] = {}
    for configured in config.capabilities:
        name = configured.strip()
        if not name:
            continue
        enabled = name[0] not in ("!", "-")
        normalized = name[1:] if not enabled else name
        if normalized:
            capabilities[normalized] = enabled
    media_types = config.selected_model_supported_media_types
    if any(media_type.startswith("image/") for media_type in media_types):
        capabilities.setdefault("vision_input", True)
    if any(media_type.startswith("audio/") for media_type in media_types):
        capabilities.setdefault("audio_input", True)
    return capabilities


def invocation_is_temporarily_not_ready(error: CopilotInvocationError) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "no assistant message",
            "empty response",
            "timed out",
            "bridge is not running",
            "bridge exited",
            "could not fetch attachment",
        )
    )


async def handle_request(websocket: Any, runner: CopilotRunner, request: dict[str, Any]) -> None:
    request_id = str(request.get("id") or "")
    if not request_id:
        return
    required = request.get("required_capabilities")
    declared = declared_capabilities(runner.config)
    declined = [
        capability
        for capability in required
        if isinstance(capability, str) and declared.get(capability) is False
    ] if isinstance(required, list) else []
    if declined:
        reason = f"servant explicitly declines: {', '.join(sorted(declined))}"
        await websocket.send(
            json.dumps({"type": "reject", "id": request_id, "reason": reason})
        )
        print(f"REJECTED {request_id}: {reason}", flush=True)
        return
    await websocket.send(json.dumps({"type": "accept", "id": request_id}))
    try:
        content = await runner.run(request)
    except CopilotInvocationError as error:
        if invocation_is_temporarily_not_ready(error):
            retry_after = 15
            print(f"NOT_READY {request_id}: {error}", flush=True)
            await websocket.send(
                json.dumps(
                    {
                        "type": "not_ready",
                        "id": request_id,
                        "reason": str(error),
                        "retry_after": retry_after,
                    }
                )
            )
            return
        print(f"REJECTED {request_id}: {error}", flush=True)
        await websocket.send(
            json.dumps({"type": "reject", "id": request_id, "reason": str(error)})
        )
        return
    await websocket.send(json.dumps({"type": "reply", "id": request_id, "content": content}))
    print(f"REPLIED {request_id} via Copilot session {runner.config.session_id}", flush=True)


def registration_payload(config: ServantRuntimeConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "register",
        "role": config.role,
        "capabilities": declared_capabilities(config),
        "worker_kind": "headless-copilot",
        "runtime_model": config.selected_model,
        "description": (
            f"Resident GitHub Copilot servant backed by {config.selected_model}; "
            f"session {config.session_id}."
        ),
    }
    if config.modelmasks:
        payload["modelmasks"] = config.modelmasks
    return payload


async def run_servant(config: ServantRuntimeConfig) -> None:
    runner = CopilotRunner(config)
    ws_url = worker_socket_url(
        config.host_ws_url, config.worker_id, ",".join(config.modelmasks)
    )
    print(
        f"START worker={config.worker_id} session={config.session_id} websocket={ws_url}",
        flush=True,
    )
    try:
        await runner.start()
        if config.warmup:
            warmup_reply = await runner.warmup()
            print(
                f"WARMED worker={config.worker_id} model={config.selected_model}: "
                f"{warmup_reply}",
                flush=True,
            )
        while True:
            try:
                async with websockets.connect(ws_url, open_timeout=15, max_size=8 * 1024 * 1024) as websocket:
                    hello = json.loads(await websocket.recv())
                    if not isinstance(hello, dict) or hello.get("type") != "hello":
                        raise CopilotInvocationError(f"unexpected relay greeting: {hello!r}")
                    await websocket.send(json.dumps(registration_payload(config)))
                    print(f"CONNECTED worker={config.worker_id}", flush=True)
                    active_task: asyncio.Task[None] | None = None
                    active_request_id: str | None = None
                    async for raw in websocket:
                        request = json.loads(raw)
                        if not isinstance(request, dict):
                            continue
                        if active_task is not None and active_task.done():
                            await active_task
                            active_task = None
                            active_request_id = None
                        if request.get("type") == "request":
                            request_id = str(request.get("id") or "")
                            if active_task is not None:
                                await websocket.send(
                                    json.dumps(
                                        {
                                            "type": "reject",
                                            "id": request_id,
                                            "reason": "headless Copilot servant is already processing a request",
                                        }
                                    )
                                )
                                continue
                            active_request_id = request_id
                            active_task = asyncio.create_task(
                                handle_request(websocket, runner, request)
                            )
                        elif request.get("type") == "cancel":
                            request_id = str(request.get("id") or "")
                            cancelled = bool(
                                active_task is not None
                                and request_id == active_request_id
                            )
                            if cancelled:
                                active_task.cancel()
                                try:
                                    await active_task
                                except asyncio.CancelledError:
                                    pass
                                active_task = None
                                active_request_id = None
                            await websocket.send(
                                json.dumps(
                                    {
                                        "type": "cancelled",
                                        "id": request_id,
                                        "cancelled": cancelled,
                                    }
                                )
                            )
                    if active_task is not None:
                        active_task.cancel()
                        try:
                            await active_task
                        except asyncio.CancelledError:
                            pass
            except (ConnectionClosed, OSError, asyncio.TimeoutError) as error:
                print(f"DISCONNECTED worker={config.worker_id}: {error}", flush=True)
            await asyncio.sleep(config.reconnect_seconds)
    finally:
        await runner.close()


def load_runtime_config(path: Path) -> ServantRuntimeConfig:
    try:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read servant configuration '{path}': {error}") from error
    return ServantRuntimeConfig.model_validate(document)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_runtime_config(args.config)
    try:
        asyncio.run(run_servant(config))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

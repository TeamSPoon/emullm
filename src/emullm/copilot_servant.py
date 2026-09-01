"""Persistent-session Copilot CLI adapter for EMULLM's worker WebSocket."""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import mimetypes
import os
import re
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import websockets
from pydantic import BaseModel, ConfigDict, Field
from websockets.exceptions import ConnectionClosed

from .copilot_api import (
    AntiIdlePromptConfig,
    DEFAULT_ANTI_IDLE_CONVERSATION_PROMPTS,
    HeadlessCopilotConfig,
    default_anti_idle_prompts,
)
from .worker import worker_socket_url


KEEPALIVE_TASKS = DEFAULT_ANTI_IDLE_CONVERSATION_PROMPTS
_KEEPALIVE_RETIRE_SLOW_COUNT = 2


def keepalive_prompt(
    index: int,
    prompts: list[AntiIdlePromptConfig] | None = None,
) -> str:
    configured = prompts or default_anti_idle_prompts()
    task_index = index % len(configured)
    return (
        f"{configured[task_index].prompt} Answer naturally in one short sentence "
        "of at most twelve words. Do not use tools or reveal client identities."
    )


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
    keepalive_enabled: bool = True
    keepalive_interval_seconds: float = 60.0
    keepalive_timeout_seconds: float = Field(default=10.0, ge=0.1, le=10.0)
    keepalive_slow_budget_seconds: float = Field(default=8.0, ge=0.1, le=10.0)
    keepalive_prompts: list[AntiIdlePromptConfig] = Field(
        default_factory=default_anti_idle_prompts,
        min_length=1,
        max_length=1_000,
    )
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
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail: list[str] = []
        self._response_queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()
        self._request_count = 0
        self._keepalive_count = 0
        self._keepalive_task_stats: dict[str, dict[str, Any]] = {}
        self._retired_keepalive_tasks: set[str] = set()
        self._cancellation_count = 0
        self._model_switch_count = 0
        self._bridge_request_id: str | None = None
        self._load_keepalive_task_stats()

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
        self._stdout_task = asyncio.create_task(self._pump_stdout())
        self._write_status(
            running=True,
            bridge_pid=self._active.pid,
            runtime_pid=ready.get("runtime_pid"),
            session_id=ready.get("session_id"),
            model=ready.get("model"),
            resumed=bool(ready.get("resumed")),
            startup_ms=round((time.monotonic() - started) * 1000),
            requests=0,
            keepalives=self._keepalive_count,
            cancellations=0,
            model_switches=0,
            warmup_completed=False,
            warmup_duration_ms=None,
        )

    def _load_keepalive_task_stats(self) -> None:
        path = Path(self.config.runtime_status_path)
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if (
            not isinstance(current, dict)
            or current.get("keepalive_stats_model") != self.config.selected_model
        ):
            return
        self._keepalive_count = max(0, int(current.get("keepalives") or 0))
        raw_stats = current.get("keepalive_task_stats")
        if not isinstance(raw_stats, dict):
            return
        prompt_by_id = {
            prompt.id: prompt for prompt in self.config.keepalive_prompts
        }
        for raw_prompt_id, raw_values in raw_stats.items():
            prompt_id = str(raw_prompt_id)
            if (
                prompt_id not in prompt_by_id
                and prompt_id.isdigit()
                and 0 <= int(prompt_id) < len(self.config.keepalive_prompts)
            ):
                prompt_id = self.config.keepalive_prompts[int(prompt_id)].id
            prompt = prompt_by_id.get(prompt_id)
            if prompt is None or not isinstance(raw_values, dict):
                continue
            prompt_hash = hashlib.sha256(
                prompt.prompt.encode("utf-8")
            ).hexdigest()[:16]
            if raw_values.get("prompt_hash") != prompt_hash:
                continue
            stats = {
                "prompt_hash": prompt_hash,
                "attempts": max(0, int(raw_values.get("attempts") or 0)),
                "completed": max(0, int(raw_values.get("completed") or 0)),
                "slow": max(0, int(raw_values.get("slow") or 0)),
                "consecutive_slow": max(
                    0, int(raw_values.get("consecutive_slow") or 0)
                ),
                "timeouts": max(0, int(raw_values.get("timeouts") or 0)),
                "total_duration_ms": max(
                    0, int(raw_values.get("total_duration_ms") or 0)
                ),
                "min_duration_ms": (
                    max(0, int(raw_values["min_duration_ms"]))
                    if raw_values.get("min_duration_ms") is not None
                    else None
                ),
                "average_duration_ms": max(
                    0.0, float(raw_values.get("average_duration_ms") or 0)
                ),
                "max_duration_ms": max(
                    0, int(raw_values.get("max_duration_ms") or 0)
                ),
                "retired": bool(raw_values.get("retired")),
            }
            self._keepalive_task_stats[prompt_id] = stats
            if stats["retired"]:
                self._retired_keepalive_tasks.add(prompt_id)
        active_ids = {
            prompt.id
            for prompt in self.config.keepalive_prompts
            if not prompt.deprecated
        }
        self._retired_keepalive_tasks.intersection_update(active_ids)
        if (
            active_ids
            and len(self._retired_keepalive_tasks) >= len(active_ids)
        ):
            fallback = min(
                active_ids,
                key=lambda prompt_id: float(
                    self._keepalive_task_stats.get(prompt_id, {}).get(
                        "average_duration_ms",
                        0,
                    )
                ),
            )
            self._retired_keepalive_tasks.discard(fallback)
            self._keepalive_task_stats[fallback]["retired"] = False

    def next_keepalive_task(self, start_index: int) -> int | None:
        prompts = self.config.keepalive_prompts
        for offset in range(len(prompts)):
            candidate = (start_index + offset) % len(prompts)
            prompt = prompts[candidate]
            if (
                not prompt.deprecated
                and prompt.id not in self._retired_keepalive_tasks
            ):
                return candidate
        return None

    def record_keepalive_result(
        self,
        prompt_index: int,
        duration_seconds: float,
        *,
        completed: bool,
        timed_out: bool = False,
    ) -> bool:
        prompt = self.config.keepalive_prompts[
            prompt_index % len(self.config.keepalive_prompts)
        ]
        prompt_id = prompt.id
        prompt_hash = hashlib.sha256(
            prompt.prompt.encode("utf-8")
        ).hexdigest()[:16]
        duration_ms = max(0, round(duration_seconds * 1000))
        stats = self._keepalive_task_stats.setdefault(
            prompt_id,
            {
                "prompt_hash": prompt_hash,
                "attempts": 0,
                "completed": 0,
                "slow": 0,
                "consecutive_slow": 0,
                "timeouts": 0,
                "total_duration_ms": 0,
                "min_duration_ms": None,
                "average_duration_ms": 0.0,
                "max_duration_ms": 0,
                "retired": False,
            },
        )
        stats["attempts"] = int(stats["attempts"]) + 1
        if completed:
            stats["completed"] = int(stats["completed"]) + 1
        if timed_out:
            stats["timeouts"] = int(stats["timeouts"]) + 1
        slow = (
            timed_out
            or duration_seconds >= self.config.keepalive_slow_budget_seconds
        )
        if slow:
            stats["slow"] = int(stats["slow"]) + 1
            stats["consecutive_slow"] = int(stats["consecutive_slow"]) + 1
        else:
            stats["consecutive_slow"] = 0
        stats["total_duration_ms"] = int(stats["total_duration_ms"]) + duration_ms
        stats["min_duration_ms"] = (
            duration_ms
            if stats.get("min_duration_ms") is None
            else min(int(stats["min_duration_ms"]), duration_ms)
        )
        stats["average_duration_ms"] = round(
            int(stats["total_duration_ms"]) / int(stats["attempts"]),
            1,
        )
        stats["max_duration_ms"] = max(int(stats["max_duration_ms"]), duration_ms)
        if (
            int(stats["consecutive_slow"]) >= _KEEPALIVE_RETIRE_SLOW_COUNT
            and len(self._retired_keepalive_tasks)
            < sum(
                not configured.deprecated
                for configured in self.config.keepalive_prompts
            )
            - 1
        ):
            stats["retired"] = True
            self._retired_keepalive_tasks.add(prompt_id)
        self._write_status(
            keepalive_stats_model=self.config.selected_model,
            keepalive_task_stats={
                prompt_id: values
                for prompt_id, values in sorted(
                    self._keepalive_task_stats.items()
                )
            },
            retired_keepalive_tasks=sorted(self._retired_keepalive_tasks),
        )
        return bool(stats["retired"])

    def reset_keepalive_stats(self) -> None:
        self._keepalive_count = 0
        self._keepalive_task_stats.clear()
        self._retired_keepalive_tasks.clear()
        self._write_status(
            keepalives=0,
            keepalive_stats_model=self.config.selected_model,
            keepalive_task_stats={},
            retired_keepalive_tasks=[],
            keepalive_stats_reset_at=time.time(),
        )

    async def run(self, request: dict[str, Any]) -> str:
        prompt = build_prompt(self.config, request)
        request_id = str(request.get("id") or "")
        is_keepalive = request.get("_maintenance") is True
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
                        await self._cancel_request(
                            active_id,
                            timeout_seconds=0.25 if is_keepalive else 10,
                        )
                    except (CopilotInvocationError, TimeoutError):
                        await self._discard_bridge()
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
            status = {
                "requests": self._request_count,
                "last_request_id": request_id,
                "last_duration_ms": round((time.monotonic() - started) * 1000),
                "last_chunk_count": len(chunks),
                "last_attachment_count": len(attachments),
                "last_completed_at": time.time(),
            }
            if is_keepalive:
                self._keepalive_count += 1
                status.update(
                    keepalives=self._keepalive_count,
                    last_keepalive_at=time.time(),
                    last_keepalive_duration_ms=status["last_duration_ms"],
                )
            self._write_status(
                **status,
            )
            return output

    async def set_model(self, model: str) -> bool:
        model = model.strip()
        if not model:
            raise CopilotInvocationError("model is required")
        async with self._lock:
            await self.start()
            if model == self.config.selected_model:
                return False
            request_id = f"model-switch-{uuid.uuid4().hex}"
            queue = self._message_queue(request_id)
            try:
                self._send(
                    {
                        "type": "set_model",
                        "id": request_id,
                        "model": model,
                        "reasoning_effort": self.config.selected_reasoning_effort,
                        "context": self.config.context,
                    }
                )
                response = await asyncio.wait_for(
                    self._read_for_control(request_id, queue),
                    timeout=60,
                )
            finally:
                self._response_queues.pop(request_id, None)
            if response.get("type") == "model_change_error":
                raise CopilotInvocationError(
                    str(response.get("error") or "Copilot model switch failed")
                )
            self.config.selected_model = model
            self.config.model = model
            self._keepalive_task_stats.clear()
            self._retired_keepalive_tasks.clear()
            self._model_switch_count += 1
            self._write_status(
                model=model,
                model_switches=self._model_switch_count,
                model_changed_at=time.time(),
                keepalive_stats_model=model,
                keepalive_task_stats={},
                retired_keepalive_tasks=[],
            )
            return True

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
        queue = self._message_queue(request_id)
        try:
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
                    self._read_for_request(request_id, queue),
                    timeout=self.config.timeout_seconds + 10,
                )
            except TimeoutError as error:
                try:
                    await self._cancel_request(request_id, queue)
                except (CopilotInvocationError, TimeoutError):
                    pass
                self._bridge_request_id = None
                raise CopilotInvocationError(
                    f"Copilot timed out after {self.config.timeout_seconds:g} seconds"
                ) from error
            except asyncio.CancelledError:
                raise
        finally:
            self._response_queues.pop(request_id, None)
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
        if self._stdout_task is not None:
            self._stdout_task.cancel()
            try:
                await self._stdout_task
            except asyncio.CancelledError:
                pass
        self._stdout_task = None
        self._response_queues.clear()
        self._active = None
        self._write_status(running=False, stopped_at=time.time())

    async def _discard_bridge(self) -> None:
        process = self._active
        if self._stdout_task is not None:
            self._stdout_task.cancel()
            try:
                await self._stdout_task
            except asyncio.CancelledError:
                pass
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        self._stdout_task = None
        self._stderr_task = None
        self._active = None
        self._response_queues.clear()
        self._bridge_request_id = None
        self._write_status(
            running=False,
            bridge_discarded_at=time.time(),
        )

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

    def _message_queue(self, request_id: str) -> asyncio.Queue[dict[str, Any]]:
        return self._response_queues.setdefault(request_id, asyncio.Queue())

    async def _pump_stdout(self) -> None:
        try:
            while True:
                message = await self._read_message()
                request_id = message.get("id")
                if request_id is None:
                    continue
                queue = self._response_queues.get(str(request_id))
                if queue is not None:
                    await queue.put(message)
        except asyncio.CancelledError:
            raise
        except CopilotInvocationError as error:
            for request_id, queue in list(self._response_queues.items()):
                await queue.put(
                    {
                        "type": "error",
                        "id": request_id,
                        "error": str(error),
                    }
                )

    async def _read_for_request(
        self,
        request_id: str,
        queue: asyncio.Queue[dict[str, Any]],
    ) -> dict[str, Any]:
        while True:
            message = await queue.get()
            if message.get("id") == request_id and message.get("type") in {
                "response",
                "error",
                "cancelled",
            }:
                return message

    async def _read_for_control(
        self,
        request_id: str,
        queue: asyncio.Queue[dict[str, Any]],
    ) -> dict[str, Any]:
        while True:
            message = await queue.get()
            if message.get("id") == request_id and message.get("type") in {
                "model_changed",
                "model_change_error",
            }:
                return message

    async def _cancel_request(
        self,
        request_id: str,
        queue: asyncio.Queue[dict[str, Any]] | None = None,
        timeout_seconds: float = 10,
    ) -> None:
        response_queue = queue or self._message_queue(request_id)
        owns_queue = queue is None
        try:
            self._send({"type": "cancel", "id": request_id})
            while True:
                message = await asyncio.wait_for(
                    response_queue.get(),
                    timeout=timeout_seconds,
                )
                if (
                    message.get("id") == request_id
                    and message.get("type") == "cancelled"
                ):
                    return
        finally:
            if owns_queue:
                self._response_queues.pop(request_id, None)

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
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        temporary.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        try:
            for attempt in range(10):
                try:
                    os.replace(temporary, path)
                    break
                except PermissionError:
                    if attempt == 9:
                        raise
                    time.sleep(0.02 * (attempt + 1))
        finally:
            temporary.unlink(missing_ok=True)

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


def image_file_reply(
    config: ServantRuntimeConfig,
    content: str,
) -> dict[str, str] | None:
    match = re.search(
        r"EMULLM_IMAGE_FILE:\s*[`'\"]?([^\r\n`'\"]+)",
        content,
    )
    if match is None:
        return None
    root = Path(config.resolved_cwd).resolve()
    candidate = Path(match.group(1).strip())
    path = (
        candidate.resolve()
        if candidate.is_absolute()
        else (root / candidate).resolve()
    )
    try:
        path.relative_to(root)
    except ValueError as error:
        raise CopilotInvocationError(
            "generated image path is outside the servant workspace"
        ) from error
    if not path.is_file():
        raise CopilotInvocationError(
            f"generated image file does not exist: {path.name}"
        )
    data = path.read_bytes()
    if not data or len(data) > config.max_attachment_bytes:
        raise CopilotInvocationError(
            "generated image file is empty or exceeds the attachment limit"
        )
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if not mime_type.startswith("image/"):
        raise CopilotInvocationError(
            f"generated file is not an image: {path.name}"
        )
    return {
        "image_b64": base64.b64encode(data).decode("ascii"),
        "mime": mime_type,
    }


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
    reply = {"type": "reply", "id": request_id, "content": content}
    if request.get("kind") in {"image", "image_edit"}:
        if image := image_file_reply(runner.config, content):
            reply.update(image)
    await websocket.send(json.dumps(reply))
    print(f"REPLIED {request_id} via Copilot session {runner.config.session_id}", flush=True)


async def handle_keepalive(
    websocket: Any,
    runner: CopilotRunner,
    request_id: str,
    prompt_index: int,
) -> None:
    started = time.monotonic()
    run_task = asyncio.create_task(
        runner.run(
            {
                "id": request_id,
                "model": runner.config.selected_model,
                "kind": "completion",
                "_maintenance": True,
                "prompt": keepalive_prompt(
                    prompt_index,
                    runner.config.keepalive_prompts,
                ),
            }
        )
    )
    try:
        done, _ = await asyncio.wait(
            {run_task},
            timeout=runner.config.keepalive_timeout_seconds,
        )
        if not done:
            run_task.cancel()
            asyncio.create_task(_drain_cancelled_task(run_task))
            raise TimeoutError
        content = run_task.result()
    except asyncio.CancelledError:
        run_task.cancel()
        asyncio.create_task(_drain_cancelled_task(run_task))
        raise
    except TimeoutError:
        duration_ms = round((time.monotonic() - started) * 1000)
        retired = runner.record_keepalive_result(
            prompt_index,
            time.monotonic() - started,
            completed=False,
            timed_out=True,
        )
        reason = (
            "keepalive exceeded the "
            f"{runner.config.keepalive_timeout_seconds:g}s maximum"
        )
        await websocket.send(
            json.dumps(
                {
                    "type": "keepalive_error",
                    "id": request_id,
                    "prompt_index": prompt_index,
                    "duration_ms": duration_ms,
                    "reason": reason,
                    "retired": retired,
                }
            )
        )
        print(f"KEEPALIVE_FAILED {request_id}: {reason}", flush=True)
        return
    except CopilotInvocationError as error:
        duration_ms = round((time.monotonic() - started) * 1000)
        runner.record_keepalive_result(
            prompt_index,
            time.monotonic() - started,
            completed=False,
        )
        await websocket.send(
            json.dumps(
                {
                    "type": "keepalive_error",
                    "id": request_id,
                    "prompt_index": prompt_index,
                    "duration_ms": duration_ms,
                    "reason": str(error),
                }
            )
        )
        print(f"KEEPALIVE_FAILED {request_id}: {error}", flush=True)
        return
    duration_seconds = time.monotonic() - started
    retired = runner.record_keepalive_result(
        prompt_index,
        duration_seconds,
        completed=True,
    )
    await websocket.send(
        json.dumps(
            {
                "type": "keepalive_reply",
                "id": request_id,
                "prompt_index": prompt_index,
                "content": content[:200],
                "duration_ms": round(duration_seconds * 1000),
                "retired": retired,
            }
        )
    )
    print(
        f"KEEPALIVE {request_id} prompt={prompt_index + 1}/"
        f"{len(runner.config.keepalive_prompts)}",
        flush=True,
    )


async def _drain_cancelled_task(task: asyncio.Task[Any]) -> None:
    try:
        await task
    except (asyncio.CancelledError, CopilotInvocationError):
        pass


async def _cancel_active_task(task: asyncio.Task[None] | None) -> None:
    if task is None:
        return
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, ConnectionClosed, OSError):
        pass


def registration_payload(config: ServantRuntimeConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "register",
        "role": config.role,
        "capabilities": declared_capabilities(config),
        "worker_kind": "headless-copilot",
        "runtime_model": config.selected_model,
        "startup_prompt": config.system_prompt,
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
    connection_errors = 0
    runner._write_status(
        adapter_pid=os.getpid(),
        connected=False,
        connection_errors=connection_errors,
        last_connection_error=None,
        anti_idle_enabled=config.keepalive_enabled,
    )
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
        anti_idle_enabled = config.keepalive_enabled
        while True:
            active_task: asyncio.Task[None] | None = None
            keepalive_scheduler: asyncio.Task[None] | None = None
            try:
                async with websockets.connect(
                    ws_url,
                    open_timeout=60,
                    ping_interval=30,
                    ping_timeout=60,
                    max_size=8 * 1024 * 1024,
                ) as websocket:
                    hello = json.loads(await websocket.recv())
                    if not isinstance(hello, dict) or hello.get("type") != "hello":
                        raise CopilotInvocationError(f"unexpected relay greeting: {hello!r}")
                    await websocket.send(json.dumps(registration_payload(config)))
                    print(f"CONNECTED worker={config.worker_id}", flush=True)
                    runner._write_status(
                        connected=True,
                        connected_at=time.time(),
                    )
                    active_request_id: str | None = None
                    active_task_kind: str | None = None
                    control_active = False
                    keepalive_index = sum(config.worker_id.encode("utf-8")) % len(
                        config.keepalive_prompts
                    )

                    async def schedule_keepalives() -> None:
                        nonlocal active_task, active_request_id
                        nonlocal active_task_kind, keepalive_index, control_active
                        interval = config.keepalive_interval_seconds
                        if interval <= 0:
                            return
                        prompt_count = len(config.keepalive_prompts)
                        phase = (
                            (keepalive_index % prompt_count)
                            / max(1, prompt_count - 1)
                        )
                        await asyncio.sleep(interval * (1 + phase))
                        while True:
                            if active_task is not None and active_task.done():
                                try:
                                    await active_task
                                except (ConnectionClosed, CopilotInvocationError):
                                    pass
                                active_task = None
                                active_request_id = None
                                active_task_kind = None
                            if anti_idle_enabled and active_task is None and not control_active:
                                selected_index = runner.next_keepalive_task(
                                    keepalive_index
                                )
                                if selected_index is None:
                                    await asyncio.sleep(interval)
                                    continue
                                keepalive_index = selected_index
                                active_request_id = (
                                    f"keepalive-{config.worker_id}-{uuid.uuid4().hex}"
                                )
                                active_task_kind = "keepalive"
                                active_task = asyncio.create_task(
                                    handle_keepalive(
                                        websocket,
                                        runner,
                                        active_request_id,
                                        keepalive_index,
                                    )
                                )
                                keepalive_index = (
                                    keepalive_index + 1
                                ) % prompt_count
                            await asyncio.sleep(interval)

                    keepalive_scheduler = asyncio.create_task(schedule_keepalives())
                    async for raw in websocket:
                        request = json.loads(raw)
                        if not isinstance(request, dict):
                            continue
                        if active_task is not None and active_task.done():
                            await active_task
                            active_task = None
                            active_request_id = None
                            active_task_kind = None
                        if request.get("type") == "shutdown":
                            await _cancel_active_task(keepalive_scheduler)
                            await _cancel_active_task(active_task)
                            await websocket.send(
                                json.dumps(
                                    {
                                        "type": "shutting_down",
                                        "reason": request.get("reason"),
                                    }
                                )
                            )
                            return
                        if request.get("type") == "reset_keepalive_stats":
                            if active_task_kind == "keepalive":
                                await _cancel_active_task(active_task)
                                active_task = None
                                active_request_id = None
                                active_task_kind = None
                            runner.reset_keepalive_stats()
                            await websocket.send(
                                json.dumps(
                                    {
                                        "type": "keepalive_stats_reset",
                                        "id": str(request.get("id") or ""),
                                    }
                                )
                            )
                            continue
                        if request.get("type") == "set_anti_idle":
                            anti_idle_enabled = request.get("enabled") is True
                            if not anti_idle_enabled and active_task_kind == "keepalive":
                                await _cancel_active_task(active_task)
                                active_task = None
                                active_request_id = None
                                active_task_kind = None
                            runner._write_status(
                                anti_idle_enabled=anti_idle_enabled,
                                anti_idle_changed_at=time.time(),
                            )
                            await websocket.send(
                                json.dumps(
                                    {
                                        "type": "anti_idle_changed",
                                        "id": request.get("id"),
                                        "enabled": anti_idle_enabled,
                                    }
                                )
                            )
                            continue
                        if request.get("type") == "set_model":
                            control_id = str(request.get("id") or "")
                            if active_task_kind == "keepalive":
                                await _cancel_active_task(active_task)
                                active_task = None
                                active_request_id = None
                                active_task_kind = None
                            if active_task is not None:
                                await websocket.send(
                                    json.dumps(
                                        {
                                            "type": "model_change_error",
                                            "id": control_id,
                                            "error": "servant is processing a request",
                                        }
                                    )
                                )
                                continue
                            control_active = True
                            try:
                                changed = await runner.set_model(
                                    str(request.get("model") or "")
                                )
                                modelmasks = request.get("modelmasks")
                                if isinstance(modelmasks, list):
                                    config.modelmasks = [
                                        str(mask) for mask in modelmasks if str(mask)
                                    ]
                                capabilities = request.get("capabilities")
                                if isinstance(capabilities, list):
                                    config.capabilities = [
                                        str(capability)
                                        for capability in capabilities
                                        if str(capability)
                                    ]
                                media_types = request.get("supported_media_types")
                                if isinstance(media_types, list):
                                    config.selected_model_supported_media_types = [
                                        str(media_type)
                                        for media_type in media_types
                                        if str(media_type)
                                    ]
                                await websocket.send(
                                    json.dumps(registration_payload(config))
                                )
                                await websocket.send(
                                    json.dumps(
                                        {
                                            "type": "model_changed",
                                            "id": control_id,
                                            "model": config.selected_model,
                                            "changed": changed,
                                        }
                                    )
                                )
                                print(
                                    f"MODEL_CHANGED worker={config.worker_id} "
                                    f"model={config.selected_model}",
                                    flush=True,
                                )
                            except CopilotInvocationError as error:
                                await websocket.send(
                                    json.dumps(
                                        {
                                            "type": "model_change_error",
                                            "id": control_id,
                                            "error": str(error),
                                        }
                                    )
                                )
                            finally:
                                control_active = False
                            continue
                        if request.get("type") == "request":
                            request_id = str(request.get("id") or "")
                            if active_task_kind == "keepalive":
                                await _cancel_active_task(active_task)
                                active_task = None
                                active_request_id = None
                                active_task_kind = None
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
                            active_task_kind = "client"
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
                                active_task_kind = None
                            await websocket.send(
                                json.dumps(
                                    {
                                        "type": "cancelled",
                                        "id": request_id,
                                        "cancelled": cancelled,
                                    }
                                )
                            )
                    await _cancel_active_task(keepalive_scheduler)
                    await _cancel_active_task(active_task)
            except (ConnectionClosed, OSError, asyncio.TimeoutError) as error:
                connection_errors += 1
                runner._write_status(
                    connected=False,
                    connection_errors=connection_errors,
                    last_connection_error=str(error),
                    last_disconnected_at=time.time(),
                )
                print(f"DISCONNECTED worker={config.worker_id}: {error}", flush=True)
            finally:
                await _cancel_active_task(keepalive_scheduler)
                await _cancel_active_task(active_task)
            await asyncio.sleep(config.reconnect_seconds)
    finally:
        runner._write_status(connected=False)
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

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import time
import uuid
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from emullm import copilot_api, copilot_servant
from emullm.copilot_servant import (
    CopilotInvocationError,
    KEEPALIVE_TASKS,
    ServantRuntimeConfig,
    build_prompt,
    handle_keepalive,
    handle_request,
    keepalive_prompt,
    registration_payload,
)


class FakeProc:
    def __init__(self, pid: int = 4321) -> None:
        self.pid = pid
        self._alive = True
        self.signals: list[int] = []

    def poll(self):
        return None if self._alive else 0

    def send_signal(self, value):
        self.signals.append(value)
        self._alive = False

    def terminate(self):
        self._alive = False

    def wait(self, timeout=None):
        self._alive = False
        return 0

    def kill(self):
        self._alive = False


@pytest.fixture(autouse=True)
def isolate_resident_runtime_paths(tmp_path, monkeypatch):
    sdk = tmp_path / "copilot-sdk.js"
    sdk.write_text("", encoding="utf-8")
    original_which = copilot_api.shutil.which
    monkeypatch.setattr(copilot_api, "_find_copilot_sdk", lambda _command: sdk)
    monkeypatch.setattr(
        copilot_api.shutil,
        "which",
        lambda name: sys.executable if name == "node" else original_which(name),
    )


def runtime_config(tmp_path, **updates) -> ServantRuntimeConfig:
    values = {
        "worker_id": "copilot-one",
        "session_id": str(uuid.uuid4()),
        "model": "gpt-5-mini",
        "selected_model": "gpt-5-mini",
        "modelmasks": ["openai/*", "demo-*"],
        "system_prompt": "Answer as the requested model.",
        "host_ws_url": "ws://127.0.0.1:8801",
        "copilot_command": sys.executable,
        "copilot_runtime_path": sys.executable,
        "copilot_sdk_path": str(tmp_path / "sdk.js"),
        "node_command": sys.executable,
        "bridge_path": str(tmp_path / "bridge.mjs"),
        "runtime_config_path": str(tmp_path / "servant-config.json"),
        "runtime_status_path": str(tmp_path / "runtime-status.json"),
        "resolved_cwd": str(tmp_path),
    }
    values.update(updates)
    return ServantRuntimeConfig.model_validate(values)


def test_resident_bridge_maps_session_and_security_configuration(tmp_path) -> None:
    config = runtime_config(tmp_path)
    bridge = Path(copilot_servant.__file__).with_name("copilot_sdk_bridge.mjs").read_text(
        encoding="utf-8"
    )
    assert str(config.session_id)
    assert "client.resumeSession(config.session_id" in bridge
    assert "session.sendAndWait" in bridge
    assert "session.setModel" in bridge
    assert "attachments: Array.isArray(message.attachments)" in bridge
    assert "availableTools: config.allow_all ? undefined : []" in bridge
    assert "onPermissionRequest: config.allow_all ? approveAll : undefined" in bridge
    assert "skipCustomInstructions: !config.load_custom_instructions" in bridge
    assert 'runtimeArgs.push("--disable-builtin-mcps")' in bridge


def test_build_prompt_includes_model_persona_and_content(tmp_path) -> None:
    prompt = build_prompt(
        runtime_config(tmp_path),
        {
            "model": "vendor/model",
            "kind": "chat",
            "persona_instruction": "Be concise.",
            "prompt": "[user] hello",
        },
    )
    assert "vendor/model" in prompt
    assert "Be concise." in prompt
    assert "[user] hello" in prompt


def test_build_prompt_rejects_oversized_input(tmp_path) -> None:
    config = runtime_config(tmp_path, max_prompt_chars=1000)
    with pytest.raises(CopilotInvocationError, match="configured maximum"):
        build_prompt(config, {"prompt": "x" * 2000})


def test_image_file_reply_reads_only_workspace_images(tmp_path) -> None:
    config = runtime_config(tmp_path)
    image_path = tmp_path / "generated.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nworker-image")

    reply = copilot_servant.image_file_reply(
        config,
        "EMULLM_IMAGE_FILE: generated.png",
    )

    assert reply is not None
    assert reply["mime"] == "image/png"
    assert base64.b64decode(reply["image_b64"]) == image_path.read_bytes()
    with pytest.raises(CopilotInvocationError, match="outside"):
        copilot_servant.image_file_reply(
            config,
            "EMULLM_IMAGE_FILE: ../outside.png",
        )


def test_empty_model_masks_are_omitted_from_registration_to_mean_all_models(tmp_path) -> None:
    all_models = registration_payload(runtime_config(tmp_path, modelmasks=[]))
    assert "modelmasks" not in all_models
    masked = registration_payload(runtime_config(tmp_path, modelmasks=["openai/*"]))
    assert masked["modelmasks"] == ["openai/*"]


def test_registration_declares_configured_and_model_media_capabilities(tmp_path) -> None:
    payload = registration_payload(
        runtime_config(
            tmp_path,
            capabilities=["audio_input", "!file_input"],
            selected_model_supported_media_types=["image/png", "application/pdf"],
        )
    )
    assert payload["capabilities"] == {
        "audio_input": True,
        "file_input": False,
        "vision_input": True,
    }
    assert payload["startup_prompt"] == "Answer as the requested model."


def test_manager_persists_and_controls_instances(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text('{"mode":"mock"}', encoding="utf-8")
    processes: list[FakeProc] = []

    def spawn(_spec):
        process = FakeProc(1000 + len(processes))
        processes.append(process)
        return process

    manager = copilot_api.CopilotInstanceManager(
        config_path=config_path,
        runtime_dir=tmp_path / "runtime",
        base_dir=tmp_path,
        default_host_ws_url="ws://127.0.0.1:8801",
        connected=lambda worker_id: (
            worker_id == "copilot-one"
            and bool(processes)
            and processes[-1]._alive
        ),
        spawn=spawn,
    )
    config = copilot_api.HeadlessCopilotConfig(
        worker_id="copilot-one",
        copilot_command=sys.executable,
        model="gpt-5-mini",
        modelmasks=["demo-*"],
    )

    created = manager.create(config, start=True)
    assert created["running"] is True
    assert created["connected"] is True
    assert created["pid"] == 1000
    assert manager.next_worker_id() == "worker-copilot-1"
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["mode"] == "mock"
    assert saved["headless_copilots"][0]["worker_id"] == "copilot-one"
    assert saved["headless_copilots"][0]["session_id"] == str(config.session_id)

    stopped = manager.stop("copilot-one")
    assert stopped["stopped"] is True
    original_session = stopped["session_id"]
    reset = manager.reset_session("copilot-one")
    assert reset["session_id"] != original_session
    assert manager.start("copilot-one")["started"] is True
    assert manager.restart("copilot-one")["restarted"] is True
    assert len(processes) == 3

    assert manager.delete("copilot-one") == {"worker_id": "copilot-one", "deleted": True}
    assert json.loads(config_path.read_text(encoding="utf-8"))["headless_copilots"] == []


def test_manager_migrates_legacy_shared_anti_idle_defaults(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "headless_copilots": [
                    {
                        "worker_id": "legacy",
                        "session_id": str(uuid.uuid4()),
                        "use_shared_anti_idle": True,
                        "keepalive_interval_seconds": 40,
                        "keepalive_timeout_seconds": 3,
                    },
                    {
                        "worker_id": "legacy-custom",
                        "session_id": str(uuid.uuid4()),
                        "keepalive_interval_seconds": 0,
                        "keepalive_timeout_seconds": 7,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    copilot_api.CopilotInstanceManager(
        config_path=config_path,
        runtime_dir=tmp_path / "runtime",
        base_dir=tmp_path,
        default_host_ws_url="ws://127.0.0.1:8801",
        definitions=json.loads(config_path.read_text(encoding="utf-8"))[
            "headless_copilots"
        ],
    )

    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["headless_copilots"][0]["keepalive_interval_seconds"] is None
    assert saved["headless_copilots"][0]["keepalive_timeout_seconds"] is None
    custom = saved["headless_copilots"][1]
    assert custom["use_shared_anti_idle"] is False
    assert custom["keepalive_interval_seconds"] == 0
    assert custom["keepalive_timeout_seconds"] == 7


def test_headless_copilot_admin_api_crud(tmp_path) -> None:
    manager = copilot_api.CopilotInstanceManager(
        config_path=tmp_path / "config.json",
        runtime_dir=tmp_path / "runtime",
        base_dir=tmp_path,
        default_host_ws_url="ws://127.0.0.1:8801",
        spawn=lambda _spec: FakeProc(),
    )
    copilot_api.set_manager(manager)
    app = FastAPI()
    app.include_router(copilot_api.router)
    try:
        with TestClient(app) as client:
            payload = {
                "worker_id": "api-servant",
                "copilot_command": sys.executable,
                "model": "gpt-5-mini",
                "modelmasks": ["api/*"],
            }
            response = client.post("/emullm/admin/copilots?start=true", json=payload)
            assert response.status_code == 200, response.text
            assert response.json()["running"] is True

            listing = client.get("/admin/emullm/copilots").json()
            assert listing["manager_active"] is True
            assert listing["instances"][0]["worker_id"] == "api-servant"
            assert listing["next_worker_id"] == "worker-copilot-1"

            assert client.post("/emullm/admin/copilots/api-servant/stop").status_code == 200
            assert client.post("/admin/emullm/copilots/api-servant/reset-session").status_code == 200
            assert client.delete("/emullm/admin/copilots/api-servant").json()["deleted"] is True
            assert client.get("/emullm/admin/copilots/api-servant").status_code == 404
    finally:
        manager.stop_all()
        copilot_api.set_manager(None)


def test_app_lifespan_autostarts_configured_headless_copilot(tmp_path, monkeypatch) -> None:
    from emullm import api as api_module
    from emullm import app as app_module

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "mode": "mock",
                "headless_copilots": [
                    {
                        "worker_id": "boot-servant",
                        "copilot_command": sys.executable,
                        "model": "gpt-5-mini",
                        "autostart": True,
                        "modelmasks": ["boot/*"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api_module, "_CONFIG_PATH", config_path)
    monkeypatch.setattr(api_module, "_RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(copilot_api.subprocess, "Popen", lambda *args, **kwargs: FakeProc(8181))
    monkeypatch.setattr(
        copilot_api,
        "copilot_models",
        lambda **_kwargs: {
            "models": [
                {
                    "id": "gpt-5-mini",
                    "name": "GPT-5 mini",
                    "capabilities": {},
                }
            ]
        },
    )

    with TestClient(app_module.app) as client:
        listing = client.get("/emullm/admin/copilots").json()
        assert listing["manager_active"] is True
        assert listing["instances"][0]["worker_id"] == "boot-servant"
        assert listing["instances"][0]["running"] is True
        assert listing["instances"][0]["pid"] == 8181

    assert copilot_api.get_manager() is None


class FakeWebSocket:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send(self, raw: str) -> None:
        self.messages.append(json.loads(raw))


class FakeRunner:
    def __init__(self, config: ServantRuntimeConfig, result: str | Exception) -> None:
        self.config = config
        self.result = result
        self.requests: list[dict] = []
        self.keepalive_results: list[dict] = []

    async def run(self, request):
        self.requests.append(request)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def record_keepalive_result(
        self,
        prompt_index,
        duration_seconds,
        *,
        completed,
        timed_out=False,
    ):
        self.keepalive_results.append(
            {
                "prompt_index": prompt_index,
                "duration_seconds": duration_seconds,
                "completed": completed,
                "timed_out": timed_out,
            }
        )
        return False


def test_request_handler_accepts_then_replies_or_rejects(tmp_path) -> None:
    config = runtime_config(tmp_path)
    websocket = FakeWebSocket()
    asyncio.run(
        handle_request(
            websocket,
            FakeRunner(config, "headless answer"),
            {"type": "request", "id": "req-1", "prompt": "hello"},
        )
    )
    assert websocket.messages == [
        {"type": "accept", "id": "req-1"},
        {"type": "reply", "id": "req-1", "content": "headless answer"},
    ]

    websocket = FakeWebSocket()
    asyncio.run(
        handle_request(
            websocket,
            FakeRunner(config, CopilotInvocationError("model unavailable")),
            {"type": "request", "id": "req-2", "prompt": "hello"},
        )
    )
    assert websocket.messages == [
        {"type": "accept", "id": "req-2"},
        {"type": "reject", "id": "req-2", "reason": "model unavailable"},
    ]

    websocket = FakeWebSocket()
    asyncio.run(
        handle_request(
            websocket,
            FakeRunner(
                config,
                CopilotInvocationError("Copilot SDK returned no assistant message"),
            ),
            {"type": "request", "id": "req-not-ready", "prompt": "hello"},
        )
    )
    assert websocket.messages == [
        {"type": "accept", "id": "req-not-ready"},
        {
            "type": "not_ready",
            "id": "req-not-ready",
            "reason": "Copilot SDK returned no assistant message",
            "retry_after": 15,
        },
    ]

    websocket = FakeWebSocket()
    asyncio.run(
        handle_request(
            websocket,
            FakeRunner(
                runtime_config(tmp_path, capabilities=["!audio_input"]),
                "should not run",
            ),
            {
                "type": "request",
                "id": "req-3",
                "prompt": "listen",
                "required_capabilities": ["audio_input"],
            },
        )
    )
    assert websocket.messages == [
        {
            "type": "reject",
            "id": "req-3",
            "reason": "servant explicitly declines: audio_input",
        }
    ]


def test_keepalive_pool_has_fifty_unique_bounded_text_tasks() -> None:
    assert len(KEEPALIVE_TASKS) == 50
    assert len(set(KEEPALIVE_TASKS)) == 50
    assert keepalive_prompt(50) == keepalive_prompt(0)
    assert sum("joke" in task.lower() for task in KEEPALIVE_TASKS) >= 5
    assert all(len(task) < 120 for task in KEEPALIVE_TASKS)


def test_keepalive_handler_reports_success_without_client_reply(tmp_path) -> None:
    runner = FakeRunner(runtime_config(tmp_path), "steady")
    websocket = FakeWebSocket()

    asyncio.run(handle_keepalive(websocket, runner, "keepalive-1", 0))

    assert runner.requests[0]["kind"] == "completion"
    assert runner.requests[0]["_maintenance"] is True
    assert "keepalive" not in runner.requests[0]["prompt"].lower()
    assert KEEPALIVE_TASKS[0] in runner.requests[0]["prompt"]
    assert runner.keepalive_results[0]["completed"] is True
    assert websocket.messages[0]["type"] == "keepalive_reply"
    assert websocket.messages[0]["content"] == "steady"
    assert websocket.messages[0]["prompt_index"] == 0
    assert websocket.messages[0]["retired"] is False
    assert websocket.messages[0]["duration_ms"] < 10_000


def test_keepalive_handler_aborts_at_configured_timeout(tmp_path) -> None:
    class SlowRunner(FakeRunner):
        async def run(self, request):
            self.requests.append(request)
            await asyncio.sleep(10)
            return "late"

    runner = SlowRunner(
        runtime_config(tmp_path, keepalive_timeout_seconds=0.1),
        "late",
    )
    websocket = FakeWebSocket()

    started = time.monotonic()
    asyncio.run(handle_keepalive(websocket, runner, "keepalive-slow", 1))

    assert time.monotonic() - started < 0.25
    assert runner.keepalive_results[0]["timed_out"] is True
    assert websocket.messages[0]["type"] == "keepalive_error"
    assert websocket.messages[0]["retired"] is False
    assert 50 <= websocket.messages[0]["duration_ms"] < 250


def test_slow_keepalive_task_is_retired_and_persisted(tmp_path) -> None:
    config = runtime_config(tmp_path)
    runner = copilot_servant.CopilotRunner(config)

    assert runner.record_keepalive_result(0, 8.1, completed=True) is False
    assert runner.record_keepalive_result(
        0,
        10.0,
        completed=False,
        timed_out=True,
    ) is True
    assert runner.next_keepalive_task(0) == 1

    restored = copilot_servant.CopilotRunner(config)
    assert restored.next_keepalive_task(0) == 1
    status = json.loads(Path(config.runtime_status_path).read_text(encoding="utf-8"))
    assert status["retired_keepalive_tasks"] == ["conversation-01"]
    assert status["keepalive_task_stats"]["conversation-01"]["slow"] == 2
    assert status["keepalive_task_stats"]["conversation-01"]["min_duration_ms"] == 8_100
    assert status["keepalive_task_stats"]["conversation-01"]["max_duration_ms"] == 10_000

    changed_prompts = copilot_api.default_anti_idle_prompts()
    changed_prompts[0].prompt = "What changed since your previous quick chat?"
    changed = copilot_servant.CopilotRunner(
        runtime_config(tmp_path, keepalive_prompts=changed_prompts)
    )
    assert changed.next_keepalive_task(0) == 0


def test_fast_keepalive_resets_slow_streak_and_pool_never_exhausts(tmp_path) -> None:
    runner = copilot_servant.CopilotRunner(runtime_config(tmp_path))

    runner.record_keepalive_result(0, 8.1, completed=True)
    runner.record_keepalive_result(0, 0.1, completed=True)
    assert runner.record_keepalive_result(0, 8.2, completed=True) is False
    assert runner.next_keepalive_task(0) == 0

    for prompt_index in range(len(KEEPALIVE_TASKS)):
        runner.record_keepalive_result(prompt_index, 8.1, completed=True)
        runner.record_keepalive_result(prompt_index, 8.2, completed=True)

    assert len(runner._retired_keepalive_tasks) == len(KEEPALIVE_TASKS) - 1  # noqa: SLF001
    assert runner.next_keepalive_task(0) is not None

    deprecated_config = runtime_config(tmp_path / "deprecated")
    deprecated_config.keepalive_prompts[0].deprecated = True
    deprecated = copilot_servant.CopilotRunner(deprecated_config)
    assert deprecated.next_keepalive_task(0) == 1


def test_runner_reset_keepalive_stats_clears_timings(tmp_path) -> None:
    config = runtime_config(tmp_path)
    runner = copilot_servant.CopilotRunner(config)
    runner.record_keepalive_result(0, 0.5, completed=True)

    runner.reset_keepalive_stats()

    status = json.loads(Path(config.runtime_status_path).read_text(encoding="utf-8"))
    assert status["keepalives"] == 0
    assert status["keepalive_task_stats"] == {}
    assert status["retired_keepalive_tasks"] == []


def test_request_handler_returns_generated_image_for_edits(tmp_path) -> None:
    image = b"\x89PNG\r\n\x1a\nEDITED"
    (tmp_path / "emullm-generated-image.png").write_bytes(image)
    websocket = FakeWebSocket()
    asyncio.run(
        handle_request(
            websocket,
            FakeRunner(
                runtime_config(tmp_path),
                "EMULLM_IMAGE_FILE: emullm-generated-image.png",
            ),
            {
                "type": "request",
                "id": "image-edit",
                "kind": "image_edit",
                "prompt": "edit",
            },
        )
    )
    assert websocket.messages[0] == {"type": "accept", "id": "image-edit"}
    reply = websocket.messages[1]
    assert reply["type"] == "reply"
    assert base64.b64decode(reply["image_b64"]) == image
    assert reply["mime"] == "image/png"


class FakeBridgeProcess:
    def __init__(self, *, hold_requests: bool = False) -> None:
        self.pid = 9191
        self.returncode = None
        self.hold_requests = hold_requests
        self.messages: list[dict] = []
        self.output: asyncio.Queue[bytes] = asyncio.Queue()
        self.output.put_nowait(
            json.dumps(
                {
                    "type": "ready",
                    "session_id": "session",
                    "model": "gpt-5-mini",
                    "resumed": True,
                }
            ).encode()
            + b"\n"
        )
        self.done = asyncio.Event()
        self.stdout = self
        self.stderr = _EmptyAsyncReader()
        self.stdin = _FakeBridgeStdin(self)
        self.terminated = False

    async def readline(self):
        return await self.output.get()

    async def wait(self):
        await self.done.wait()
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15
        self.done.set()

    def kill(self):
        self.terminated = True
        self.returncode = -9
        self.done.set()


class _EmptyAsyncReader:
    async def readline(self):
        return b""


class _FakeBridgeStdin:
    def __init__(self, process: FakeBridgeProcess) -> None:
        self.process = process

    def write(self, raw: bytes) -> None:
        for line in raw.decode().splitlines():
            message = json.loads(line)
            self.process.messages.append(message)
            if message["type"] == "request" and not self.process.hold_requests:
                self.process.output.put_nowait(
                    json.dumps(
                        {
                            "type": "response",
                            "id": message["id"],
                            "content": f"answer-{message['id']}",
                            "duration_ms": 25,
                        }
                    ).encode()
                    + b"\n"
                )
            elif message["type"] == "cancel":
                self.process.output.put_nowait(
                    json.dumps(
                        {"type": "cancelled", "id": message["id"], "cancelled": True}
                    ).encode()
                    + b"\n"
                )
            elif message["type"] == "set_model":
                self.process.output.put_nowait(
                    json.dumps(
                        {
                            "type": "model_changed",
                            "id": message["id"],
                            "model": message["model"],
                        }
                    ).encode()
                    + b"\n"
                )
            elif message["type"] == "shutdown":
                self.process.returncode = 0
                self.process.done.set()


class _AttachmentResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_runner_switches_models_without_restarting_bridge(
    tmp_path,
    monkeypatch,
) -> None:
    process = None

    async def create_process(*args, **kwargs):
        nonlocal process
        process = FakeBridgeProcess()
        return process

    monkeypatch.setattr(copilot_servant.asyncio, "create_subprocess_exec", create_process)

    async def scenario() -> None:
        runner = copilot_servant.CopilotRunner(runtime_config(tmp_path))
        assert await runner.set_model("claude-sonnet-5") is True
        assert await runner.set_model("claude-sonnet-5") is False
        assert runner.config.selected_model == "claude-sonnet-5"
        assert process is not None
        switches = [
            message for message in process.messages if message["type"] == "set_model"
        ]
        assert len(switches) == 1
        assert switches[0]["model"] == "claude-sonnet-5"
        status = json.loads(
            (tmp_path / "runtime-status.json").read_text(encoding="utf-8")
        )
        assert status["model"] == "claude-sonnet-5"
        assert status["model_switches"] == 1
        await runner.close()

    asyncio.run(scenario())


def test_runner_fetches_cloud_attachment_as_native_sdk_blob(
    tmp_path,
    monkeypatch,
) -> None:
    process = None

    async def create_process(*args, **kwargs):
        nonlocal process
        process = FakeBridgeProcess()
        return process

    monkeypatch.setattr(copilot_servant.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(
        copilot_servant.urllib.request,
        "urlopen",
        lambda url, timeout: _AttachmentResponse(b"image-bytes"),
    )

    async def scenario() -> None:
        runner = copilot_servant.CopilotRunner(runtime_config(tmp_path))
        response = await runner.run(
            {
                "id": "with-file",
                "model": "demo/model",
                "kind": "vision",
                "prompt": "Describe it.",
                "attachments": [
                    {
                        "name": "photo.png",
                        "mime_type": "image/png",
                        "bytes": 11,
                        "url": "/emullm/cloud/files/file-1",
                    }
                ],
            }
        )
        assert response == "answer-with-file"
        assert process is not None
        sent = next(message for message in process.messages if message["type"] == "request")
        assert sent["attachments"] == [
            {
                "type": "blob",
                "data": base64.b64encode(b"image-bytes").decode("ascii"),
                "mimeType": "image/png",
                "displayName": "photo.png",
            }
        ]
        await runner.close()

    asyncio.run(scenario())


def test_runner_reuses_one_resident_bridge_for_multiple_requests(tmp_path, monkeypatch) -> None:
    processes = []

    async def create_process(*args, **kwargs):
        process = FakeBridgeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(copilot_servant.asyncio, "create_subprocess_exec", create_process)

    async def scenario() -> None:
        runner = copilot_servant.CopilotRunner(runtime_config(tmp_path))
        first = await runner.run(
            {"id": "one", "model": "demo/model", "kind": "chat", "prompt": "first"}
        )
        second = await runner.run(
            {"id": "two", "model": "demo/model", "kind": "chat", "prompt": "second"}
        )
        assert first == "answer-one"
        assert second == "answer-two"
        assert len(processes) == 1
        assert [message["type"] for message in processes[0].messages] == [
            "request",
            "request",
        ]
        await runner.close()

    asyncio.run(scenario())
    assert processes[0].terminated is False
    assert processes[0].messages[-1]["type"] == "shutdown"


def test_runner_warms_resident_session_once_before_service(tmp_path, monkeypatch) -> None:
    process = None

    async def create_process(*args, **kwargs):
        nonlocal process
        process = FakeBridgeProcess()
        return process

    monkeypatch.setattr(copilot_servant.asyncio, "create_subprocess_exec", create_process)

    async def scenario() -> None:
        runner = copilot_servant.CopilotRunner(runtime_config(tmp_path))
        await runner.start()
        response = await runner.warmup()
        assert response == "answer-startup-warmup"
        status = json.loads((tmp_path / "runtime-status.json").read_text(encoding="utf-8"))
        assert status["warmup_completed"] is True
        assert status["warmup_duration_ms"] >= 0
        assert status["requests"] == 1
        assert process is not None
        assert [message["type"] for message in process.messages] == ["request"]
        await runner.close()

    asyncio.run(scenario())


def test_runner_ingests_oversized_prompt_in_chunks_then_answers(tmp_path, monkeypatch) -> None:
    process = None

    async def create_process(*args, **kwargs):
        nonlocal process
        process = FakeBridgeProcess()
        return process

    monkeypatch.setattr(copilot_servant.asyncio, "create_subprocess_exec", create_process)

    async def scenario() -> None:
        config = runtime_config(
            tmp_path,
            chunk_tokens=1000,
            max_chunks=10,
            max_prompt_chars=10_000,
        )
        runner = copilot_servant.CopilotRunner(config)
        response = await runner.run(
            {
                "id": "large",
                "model": "large-context/model",
                "kind": "chat",
                "prompt": "x" * 6500,
            }
        )
        assert response == "answer-large"
        assert process is not None
        request_ids = [
            message["id"]
            for message in process.messages
            if message["type"] == "request"
        ]
        assert request_ids == [
            "large-chunk-1",
            "large-chunk-2",
            "large-chunk-3",
            "large",
        ]
        status = json.loads((tmp_path / "runtime-status.json").read_text(encoding="utf-8"))
        assert status["last_chunk_count"] == 3
        await runner.close()

    asyncio.run(scenario())


def test_runner_can_reject_oversized_prompt_when_chunking_disabled(tmp_path) -> None:
    runner = copilot_servant.CopilotRunner(
        runtime_config(
            tmp_path,
            chunk_long_prompts=False,
            chunk_tokens=1000,
            max_prompt_chars=10_000,
        )
    )
    with pytest.raises(CopilotInvocationError, match="above the selected model chunk budget"):
        asyncio.run(
            runner.run(
                {
                    "id": "large",
                    "model": "model",
                    "kind": "chat",
                    "prompt": "x" * 6500,
                }
            )
        )


def test_cancelling_runner_aborts_request_but_keeps_bridge_alive(tmp_path, monkeypatch) -> None:
    process = None

    async def create_process(*args, **kwargs):
        nonlocal process
        process = FakeBridgeProcess(hold_requests=True)
        return process

    monkeypatch.setattr(copilot_servant.asyncio, "create_subprocess_exec", create_process)

    async def scenario() -> None:
        runner = copilot_servant.CopilotRunner(runtime_config(tmp_path))
        task = asyncio.create_task(
            runner.run({"id": "slow", "model": "demo/model", "kind": "chat", "prompt": "slow"})
        )
        for _ in range(100):
            if process is not None and process.messages:
                break
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert process is not None
        assert process.terminated is False
        assert [message["type"] for message in process.messages] == ["request", "cancel"]
        process.hold_requests = False
        assert await runner.run(
            {
                "id": "after-cancel",
                "model": "demo/model",
                "kind": "chat",
                "prompt": "next",
            }
        ) == "answer-after-cancel"
        await runner.close()

    asyncio.run(scenario())


def test_atomic_config_updates_do_not_lose_concurrent_writes(tmp_path) -> None:
    import concurrent.futures

    path = tmp_path / "config.json"
    path.write_text('{"values":[]}', encoding="utf-8")

    def append_value(value: int) -> None:
        copilot_api.update_config_document(
            path,
            lambda document: document["values"].append(value),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(append_value, range(50)))

    assert sorted(json.loads(path.read_text(encoding="utf-8"))["values"]) == list(
        range(50)
    )


def test_model_catalog_is_cached_and_refreshable(monkeypatch) -> None:
    calls = []

    def query():
        calls.append(True)
        return [{"id": "auto", "name": "Auto"}, {"id": "model-a", "name": "Model A"}]

    monkeypatch.setattr(copilot_api, "_MODEL_CACHE", None)
    monkeypatch.setattr(copilot_api, "_query_copilot_models", query)
    first = copilot_api.copilot_models()
    second = copilot_api.copilot_models()
    refreshed = copilot_api.copilot_models(refresh=True)

    assert first["source"] == "copilot-sdk"
    assert second["models"] == first["models"]
    assert refreshed["models"][1]["id"] == "model-a"
    assert "quality_tier" in refreshed["models"][1]
    assert len(calls) == 2


def test_model_catalog_uses_explicit_fallback_on_discovery_failure(monkeypatch) -> None:
    monkeypatch.setattr(copilot_api, "_MODEL_CACHE", None)
    monkeypatch.setattr(
        copilot_api,
        "_query_copilot_models",
        lambda: (_ for _ in ()).throw(FileNotFoundError("SDK unavailable")),
    )
    result = copilot_api.copilot_models(refresh=True)
    assert result["source"] == "fallback"
    assert "SDK unavailable" in result["error"]
    assert "gpt-5.6-sol" in {model["id"] for model in result["models"]}
    assert "auto" not in {model["id"] for model in result["models"]}


def test_unspecified_model_is_randomly_selected_from_configured_pool(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(copilot_api.secrets, "choice", lambda values: values[-1])
    manager = copilot_api.CopilotInstanceManager(
        config_path=tmp_path / "config.json",
        runtime_dir=tmp_path / "runtime",
        base_dir=tmp_path,
        default_host_ws_url="ws://127.0.0.1:8801",
        spawn=lambda _spec: FakeProc(),
    )
    created = manager.create(
        copilot_api.HeadlessCopilotConfig(
            worker_id="random-model",
            copilot_command=sys.executable,
            model_pool=["model-a", "model-b"],
        )
    )
    runtime = json.loads(
        (tmp_path / "runtime" / "random-model" / "servant-config.json").read_text(encoding="utf-8")
    )
    assert runtime["selected_model"] == "model-b"
    assert runtime["model"] == "model-b"
    assert runtime["selected_model_max_prompt_tokens"] > 0
    assert created["model"] is None
    assert created["selected_model"] == "model-b"
    manager.stop_all()


def test_runtime_status_write_retries_windows_replace_lock(tmp_path, monkeypatch) -> None:
    runner = copilot_servant.CopilotRunner(runtime_config(tmp_path))
    original_replace = os.replace
    attempts = 0

    def replace(source, target):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("temporarily locked")
        original_replace(source, target)

    monkeypatch.setattr(os, "replace", replace)

    runner._write_status(connected=True)  # noqa: SLF001

    assert attempts == 3
    assert json.loads(Path(runner.config.runtime_status_path).read_text())["connected"] is True


def test_servant_websocket_uses_restart_tolerant_timeouts() -> None:
    source = Path(copilot_servant.__file__).read_text(encoding="utf-8")
    assert "open_timeout=60" in source
    assert "ping_interval=30" in source
    assert "ping_timeout=60" in source


def test_runtime_config_resolves_shared_anti_idle_with_worker_override(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        copilot_api,
        "copilot_models",
        lambda **_kwargs: {
            "models": [{"id": "gpt-5-mini", "name": "GPT-5 mini"}],
            "source": "test",
        },
    )
    config_path = tmp_path / "config.json"
    shared = copilot_api.AntiIdleConfig(
        interval_seconds=60,
        timeout_seconds=2.5,
        slow_budget_seconds=2,
    )
    config_path.write_text(
        json.dumps({"anti_idle": shared.model_dump(mode="json")}),
        encoding="utf-8",
    )
    manager = copilot_api.CopilotInstanceManager(
        config_path=config_path,
        runtime_dir=tmp_path / "runtime",
        base_dir=tmp_path,
        default_host_ws_url="ws://127.0.0.1:8801",
        spawn=lambda _spec: FakeProc(),
    )
    manager.create(
        copilot_api.HeadlessCopilotConfig(
            worker_id="shared",
            copilot_command=sys.executable,
            model="gpt-5-mini",
        )
    )
    manager.create(
        copilot_api.HeadlessCopilotConfig(
            worker_id="overridden",
            copilot_command=sys.executable,
            model="gpt-5-mini",
            use_shared_anti_idle=False,
            keepalive_interval_seconds=75,
            keepalive_timeout_seconds=2,
        )
    )

    shared_runtime = json.loads(
        (
            tmp_path
            / "runtime"
            / "shared"
            / "servant-config.json"
        ).read_text(encoding="utf-8")
    )
    override_runtime = json.loads(
        (
            tmp_path
            / "runtime"
            / "overridden"
            / "servant-config.json"
        ).read_text(encoding="utf-8")
    )
    assert shared_runtime["keepalive_interval_seconds"] == 60
    assert shared_runtime["keepalive_timeout_seconds"] == 2.5
    assert shared_runtime["keepalive_slow_budget_seconds"] == 2
    assert len(shared_runtime["keepalive_prompts"]) == 50
    assert override_runtime["keepalive_interval_seconds"] == 75
    assert override_runtime["keepalive_timeout_seconds"] == 2
    manager.stop_all()


def test_ranked_model_selectors_and_reasoning_filter(monkeypatch) -> None:
    models = copilot_api._annotate_model_ranks(  # noqa: SLF001
        [
            {
                "id": "claude-opus-5",
                "capabilities": {"supports": {"reasoning_effort": ["low", "high", "max"]}},
            },
            {
                "id": "gpt-5.4-mini",
                "capabilities": {"supports": {"reasoning_effort": ["none", "low", "high"]}},
            },
            {
                "id": "mai-code-1-flash-picker",
                "capabilities": {"supports": {"reasoning_effort": ["low"]}},
            },
        ]
    )
    monkeypatch.setattr(copilot_api.secrets, "choice", lambda values: values[0])
    base = {
        "worker_id": "ranked",
        "model_pool": [model["id"] for model in models],
    }
    assert (
        copilot_api.select_copilot_model(
            copilot_api.HeadlessCopilotConfig(**base, model_selector="best-1"),
            models,
        )
        == "claude-opus-5"
    )
    assert (
        copilot_api.select_copilot_model(
            copilot_api.HeadlessCopilotConfig(**base, model_selector="worse-1"),
            models,
        )
        == "mai-code-1-flash-picker"
    )
    assert (
        copilot_api.select_copilot_model(
            copilot_api.HeadlessCopilotConfig(
                **base, model_selector="worst-3", reasoning_effort="max"
            ),
            models,
        )
        == "claude-opus-5"
    )
    with pytest.raises(copilot_api.CopilotInstanceError, match="does not support"):
        copilot_api.select_copilot_model(
            copilot_api.HeadlessCopilotConfig(
                worker_id="explicit",
                model="gpt-5.4-mini",
                reasoning_effort="max",
            ),
            models,
        )
    gpt_mini = next(model for model in models if model["id"] == "gpt-5.4-mini")
    assert copilot_api.select_reasoning_effort("most-1", gpt_mini) == "high"
    assert copilot_api.select_reasoning_effort("least-1", gpt_mini) == "none"
    assert copilot_api.select_reasoning_effort("random", gpt_mini) == "none"
    assert (
        copilot_api.HeadlessCopilotConfig(
            worker_id="effort", reasoning_effort="most-1"
        ).reasoning_effort
        == "most-1"
    )
    with pytest.raises(ValueError):
        copilot_api.HeadlessCopilotConfig(
            worker_id="bad-effort", reasoning_effort="maximum"
        )


def test_next_worker_id_fills_first_available_number(tmp_path) -> None:
    manager = copilot_api.CopilotInstanceManager(
        config_path=tmp_path / "config.json",
        runtime_dir=tmp_path / "runtime",
        base_dir=tmp_path,
        default_host_ws_url="ws://127.0.0.1:8801",
        definitions=[
            {"worker_id": "worker-copilot-1", "model": "gpt-5-mini"},
            {"worker_id": "worker-copilot-3", "model": "gpt-5-mini"},
            {"worker_id": "custom-name", "model": "gpt-5-mini"},
        ],
        spawn=lambda _spec: FakeProc(),
    )
    assert manager.next_worker_id() == "worker-copilot-2"


def test_next_worker_id_skips_reserved_elastic_range(tmp_path) -> None:
    manager = copilot_api.CopilotInstanceManager(
        config_path=tmp_path / "config.json",
        runtime_dir=tmp_path / "runtime",
        base_dir=tmp_path,
        default_host_ws_url="ws://127.0.0.1:8801",
        definitions=[
            {"worker_id": f"worker-copilot-{index}", "model": "gpt-5-mini"}
            for index in range(1, 5)
        ],
        spawn=lambda _spec: FakeProc(),
    )
    assert manager.next_worker_id() == "worker-copilot-51"


def test_manager_adopts_reconnected_worker_without_duplicate_spawn(tmp_path) -> None:
    manager = copilot_api.CopilotInstanceManager(
        config_path=tmp_path / "config.json",
        runtime_dir=tmp_path / "runtime",
        base_dir=tmp_path,
        default_host_ws_url="ws://127.0.0.1:8801",
        definitions=[
            {
                "worker_id": "worker-copilot-1",
                "model": "gpt-5-mini",
                "autostart": True,
            }
        ],
        connected=lambda worker_id: worker_id == "worker-copilot-1",
        spawn=lambda _spec: (_ for _ in ()).throw(
            AssertionError("reconnected worker must not be spawned again")
        ),
    )
    assert manager.start_autostart() == []
    status = manager.get("worker-copilot-1")
    assert status["running"] is True
    assert status["connected"] is True
    assert status["external"] is True


def test_manager_status_coalesces_concurrent_admin_snapshots() -> None:
    class Manager:
        calls = 0

        @classmethod
        def list(cls):
            cls.calls += 1
            return [{"worker_id": "worker-copilot-1"}]

    manager = Manager()
    copilot_api.set_manager(manager)
    try:
        assert copilot_api.manager_status() == [{"worker_id": "worker-copilot-1"}]
        assert copilot_api.manager_status() == [{"worker_id": "worker-copilot-1"}]
        assert manager.calls == 1
    finally:
        copilot_api.set_manager(None)


def test_spawn_strips_host_agent_session_from_servant_environment(tmp_path, monkeypatch) -> None:
    captured = {}

    def popen(*args, **kwargs):
        captured.update(kwargs)
        return FakeProc()

    monkeypatch.setenv("COPILOT_AGENT_SESSION_ID", "host-session")
    monkeypatch.setattr(copilot_api.subprocess, "Popen", popen)
    manager = copilot_api.CopilotInstanceManager(
        config_path=tmp_path / "config.json",
        runtime_dir=tmp_path / "runtime",
        base_dir=tmp_path,
        default_host_ws_url="ws://127.0.0.1:8801",
    )
    manager.create(
        copilot_api.HeadlessCopilotConfig(
            worker_id="isolated",
            copilot_command=sys.executable,
            model="gpt-5-mini",
        )
    )
    assert "COPILOT_AGENT_SESSION_ID" not in captured["env"]
    assert captured["env"]["PYTHONIOENCODING"] == "utf-8"
    manager.stop_all()


@pytest.mark.skipif(os.name != "nt", reason="Windows script shims are Windows-specific")
def test_windows_copilot_shim_resolves_to_sdk_runtime_entrypoint(tmp_path) -> None:
    shim = tmp_path / "copilot.cmd"
    shim.write_text("@echo off", encoding="ascii")
    loader = tmp_path / "node_modules" / "@github" / "copilot" / "npm-loader.js"
    loader.parent.mkdir(parents=True)
    loader.write_text("", encoding="ascii")
    runtime = copilot_api.resolve_copilot_runtime(str(shim))
    assert runtime == str(loader.resolve())


def test_default_config_includes_random_model_headless_copilot_one() -> None:
    config = json.loads((Path(__file__).parents[1] / "config.json").read_text(encoding="utf-8"))
    assert config["max_concurrent_calls"] == 21
    assert 0 <= config["idle_worker_target"] <= config["max_concurrent_calls"]
    assert 0 <= config["idle_grace_seconds"] <= 3_600
    assert 0 <= config["backend_fallback_delay_seconds"] <= 300
    instances = config["headless_copilots"]
    assert instances
    instance = next(item for item in instances if item["worker_id"] == "worker-copilot-1")
    assert instance["worker_id"] == "worker-copilot-1"
    assert uuid.UUID(instance["session_id"])
    assert instance["autostart"] is True
    assert instance["warmup"] is True
    assert all(item["warmup_prompt"] for item in instances)
    assert all(item["model_selector"] == "random" for item in instances)
    assert all(item["chunk_long_prompts"] is True for item in instances)
    assert all(item["allow_all"] is True for item in instances)
    assert all(item["load_custom_instructions"] is True for item in instances)
    assert all(item["enable_builtin_mcps"] is True for item in instances)
    assert all(item["max_prompt_chars"] == 4_000_000 for item in instances)
    assert all(item.get("use_shared_anti_idle", True) is True for item in instances)
    assert config["anti_idle"]["interval_seconds"] == 60
    assert config["anti_idle"]["timeout_seconds"] == 10
    assert len(config["anti_idle"]["prompts"]) == 50
    assert isinstance(instance["allow_all"], bool)
    assert instance["modelmasks"] == []
    assert not instance.get("model")
    expected_route = [
        "worker-copilot-*",
        "worker-codex-*",
        "https://llm.a.singularitycompute.com/v1",
    ]
    assert config["model_routes"]
    assert config["model_routes"]["google/gemma-4-31b-it"] == expected_route
    carol = next(agent for agent in config["agents"] if agent["id"] == "carol")
    assert carol["enabled"] is False

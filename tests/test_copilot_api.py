from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from emullm import copilot_api, copilot_servant
from emullm.copilot_servant import (
    CopilotInvocationError,
    ServantRuntimeConfig,
    build_prompt,
    handle_request,
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


def test_empty_model_masks_are_omitted_from_registration_to_mean_all_models(tmp_path) -> None:
    all_models = registration_payload(runtime_config(tmp_path, modelmasks=[]))
    assert "modelmasks" not in all_models
    masked = registration_payload(runtime_config(tmp_path, modelmasks=["openai/*"]))
    assert masked["modelmasks"] == ["openai/*"]


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
        connected=lambda worker_id: worker_id == "copilot-one",
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
    assert manager.next_worker_id() == "copilot-headless-1"
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
            assert listing["next_worker_id"] == "copilot-headless-1"

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

    async def run(self, _request):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


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
            elif message["type"] == "shutdown":
                self.process.returncode = 0
                self.process.done.set()


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
        await runner.close()

    asyncio.run(scenario())


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
            {"worker_id": "copilot-headless-1", "model": "gpt-5-mini"},
            {"worker_id": "copilot-headless-3", "model": "gpt-5-mini"},
            {"worker_id": "custom-name", "model": "gpt-5-mini"},
        ],
        spawn=lambda _spec: FakeProc(),
    )
    assert manager.next_worker_id() == "copilot-headless-2"


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
    instances = config["headless_copilots"]
    assert instances
    instance = next(item for item in instances if item["worker_id"] == "copilot-headless-1")
    assert instance["worker_id"] == "copilot-headless-1"
    assert uuid.UUID(instance["session_id"])
    assert instance["autostart"] is True
    assert all(item["warmup"] is True for item in instances)
    assert all(item["warmup_prompt"] for item in instances)
    assert all(item["model_selector"] == "random" for item in instances)
    assert all(item["chunk_long_prompts"] is True for item in instances)
    assert all(item["max_prompt_chars"] == 4_000_000 for item in instances)
    assert isinstance(instance["allow_all"], bool)
    assert instance["modelmasks"] == []
    assert not instance.get("model")
    expected_route = [
        "copilot-headless-*",
        "codex-headless-*",
        "https://llm.a.singularitycompute.com/v1",
    ]
    assert config["model_routes"]
    assert all(route == expected_route for route in config["model_routes"].values())
    carol = next(agent for agent in config["agents"] if agent["id"] == "carol")
    assert carol["enabled"] is False

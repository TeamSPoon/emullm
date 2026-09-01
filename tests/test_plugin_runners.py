from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from emullm import api, embedded, process_control, standalone
from emullm import app as app_module


def test_manifest_uses_native_service_catalog() -> None:
    manifest = json.loads((Path(__file__).parents[1] / "plugin.json").read_text(encoding="utf-8"))
    assert manifest["adminPage"] == "/emullm"
    assert manifest["configPage"].endswith("/emullm")
    services = manifest["servicesEndpoint"]
    assert services["path"] == "/emullm/endpoints"
    assert services["method"] == "GET"
    assert services["transport"] == "http"
    assert services["protocol"] == "emullm-service-catalog-v1"
    assert services["format"] == "json"
    assert services["websocket"] == "/emullm/ws"
    endpoints = manifest["plugin-endpoints"]
    assert endpoints["admin"]["path"] == "/emullm"
    assert endpoints["adminAlias"]["path"] == "/emullm/admin"
    # New declarations live in plugin-endpoints. A legacy plugin-api alias may
    # coexist while older Workbench builds still read it.
    if "plugin-api" in manifest:
        assert manifest["plugin-api"]["status"]["path"] == "/emullm/status"
        assert manifest["plugin-api"]["admin"]["path"] == "/emullm/admin"
        assert manifest["plugin-api"]["restart"]["path"] == "/emullm/admin/restart"
        assert manifest["plugin-api"]["shutdown"]["path"] == "/emullm/admin/shutdown"
    assert endpoints["services"]["path"] == "/emullm/endpoints"
    assert endpoints["workerWebSocket"]["path"] == "/emullm/ws"
    assert endpoints["mailboxWebSocket"]["path"] == "/emullm/mailbox/ws"
    assert endpoints["headlessCopilots"]["path"] == "/emullm/admin/copilots"
    assert endpoints["bulkCopilotActions"]["path"].endswith("/{action}")
    assert "/online-action/" in endpoints["onlineCopilotAction"]["path"]
    assert endpoints["chatCompletions"]["path"] == "/v1/chat/completions"
    assert endpoints["modelConfigurator"]["path"] == "/emullm/admin/model-config"
    assert endpoints["loadCopilotModel"]["path"].endswith("/{model_id}")
    assert endpoints["testMediaSamples"]["path"] == "/emullm/admin/test-samples"
    assert endpoints["imageGenerations"]["path"] == "/v1/images/generations"
    assert endpoints["imageEdits"]["path"] == "/v1/images/edits"
    assert endpoints["agents"]["path"] == "/emullm/admin/agents"
    assert endpoints["websocketInventory"]["path"] == "/emullm/admin/websockets"
    assert endpoints["workerSocketLog"]["path"].endswith("/{worker_id}/log")
    assert endpoints["workerSocketLogViewer"]["path"].endswith("/{worker_id}/log/view")
    assert endpoints["workerSocketMedia"]["path"].endswith("/{worker_id}/media/{filename}")
    assert endpoints["fastapiRequestInventory"]["path"] == "/emullm/admin/clients"
    assert endpoints["backendConfigurator"]["path"] == "/emullm/admin/backends/configured"
    assert endpoints["codexSuppliers"]["path"] == "/emullm/admin/codex-suppliers"
    assert endpoints["antiIdlePrompts"]["path"] == "/emullm/admin/anti-idle"
    assert endpoints["modelTestClient"]["path"] == "/emullm/admin/test-chat"
    assert endpoints["configSections"]["path"] == "/emullm/admin/config/section/{section}"
    assert endpoints["restart"]["path"] == "/emullm/admin/restart"
    assert endpoints["shutdown"]["path"] == "/emullm/admin/shutdown"
    assert manifest["serverEventLog"] == {
        "endpoint": "/emullm/websock_to_llm_user/events",
        "method": "GET",
        "protocol": "http",
        "format": "json",
    }
    assert 'default="127.0.0.1"' in (
        Path(__file__).parents[1] / "run.py"
    ).read_text(encoding="utf-8")
    assert "standalone.main([args.host, str(args.port)])" in (
        Path(__file__).parents[1] / "run.py"
    ).read_text(encoding="utf-8")
    modes = manifest["runtimeModes"]
    assert modes["default"] == modes["current"] == "standalone"
    assert {mode["id"] for mode in modes["available"]} == {"standalone", "embedded"}
    standalone = next(mode for mode in modes["available"] if mode["id"] == "standalone")
    embedded_mode = next(mode for mode in modes["available"] if mode["id"] == "embedded")
    assert standalone["server"]["baseUrl"] == "http://127.0.0.1:8801"
    assert standalone["environment"] == {"EMULLM_PLUGIN_MODE": "standalone"}
    assert standalone["config"]["plugin-lifecycle.standalone"] is True
    assert embedded_mode["server"]["servicePath"] == "/emullm"
    assert embedded_mode["environment"] == {"EMULLM_PLUGIN_MODE": "embedded"}
    assert embedded_mode["config"]["plugin-lifecycle.standalone"] is False


def test_embedded_router_includes_relay_routes_and_lifespan(
    monkeypatch,
    tmp_path,
) -> None:
    host = FastAPI()
    host.include_router(embedded.create_router({}))
    config_path = tmp_path / "config.json"
    config_path.write_text('{"mode":"mock","idle_worker_target":0}', encoding="utf-8")
    monkeypatch.setattr(api, "_CONFIG_PATH", config_path)
    monkeypatch.setattr(api, "_RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(
        app_module._sup,
        "load_config",
        lambda _path: {"mode": "mock", "idle_worker_target": 0},
    )
    previous_mode = api._SERVER_MODE
    api._SERVER_MODE = "recruit"
    try:
        with TestClient(host) as client:
            assert api._SERVER_MODE == "mock"
            assert client.get("/v1/models").status_code == 200
            assert client.get("/emullm/admin").status_code == 200
            assert client.get("/admin/emullm").status_code == 200
        assert api._SERVER_MODE == "recruit"
    finally:
        api._SERVER_MODE = previous_mode


def test_standalone_main_runs_existing_app(monkeypatch) -> None:
    configs: list[tuple[object, dict[str, object]]] = []
    servers = []

    class FakeServer:
        def __init__(self, config) -> None:
            self.config = config
            self.should_exit = False
            self.ran = False

        def run(self) -> None:
            self.ran = True

    def fake_config(app: object, **kwargs: object):
        configs.append((app, kwargs))
        return {"app": app, **kwargs}

    def fake_server(config):
        server = FakeServer(config)
        servers.append(server)
        return server

    monkeypatch.setattr("uvicorn.Config", fake_config)
    monkeypatch.setattr("uvicorn.Server", fake_server)
    monkeypatch.setenv("EMULLM_HOST", "before-test")
    monkeypatch.setenv("EMULLM_HTTP_PORT", "1")
    standalone.main(["127.0.0.2", "9911"])

    assert configs == [
        (
            "emullm.app:app",
            {"host": "127.0.0.2", "port": 9911, "reload": False},
        )
    ]
    assert servers[0].ran is True
    assert process_control.shutdown_available() is False


def test_restart_marks_worker_handoff_for_replacement(monkeypatch) -> None:
    captured = {}

    class FakeProcess:
        pid = 4242

        @staticmethod
        def terminate():
            return None

    def popen(*args, **kwargs):
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(process_control, "_restart_in_progress", False)
    monkeypatch.setattr(process_control.subprocess, "Popen", popen)
    process_control.register_shutdown_callback(lambda: None)
    monkeypatch.setattr(process_control, "schedule_shutdown", lambda _delay: True)
    try:
        assert process_control.schedule_restart(
            "127.0.0.1",
            8801,
            delay_seconds=60,
        ) == 4242
        assert captured["env"]["EMULLM_RESTART_HANDOFF"] == "1"
        assert process_control.restart_in_progress() is True
    finally:
        process_control.register_shutdown_callback(None)

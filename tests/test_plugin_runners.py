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
    assert endpoints["agents"]["path"] == "/emullm/admin/agents"
    assert endpoints["websocketInventory"]["path"] == "/emullm/admin/websockets"
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


def test_embedded_router_includes_relay_routes_and_lifespan(monkeypatch) -> None:
    host = FastAPI()
    host.include_router(embedded.create_router({}))
    monkeypatch.setattr(app_module._sup, "load_config", lambda _path: {"mode": "mock"})
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

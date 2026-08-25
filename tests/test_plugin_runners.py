from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from emullm import api, embedded, standalone
from emullm import app as app_module


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
    calls: list[tuple[object, dict[str, object]]] = []

    def fake_run(app: object, **kwargs: object) -> None:
        calls.append((app, kwargs))

    monkeypatch.setattr("uvicorn.run", fake_run)
    standalone.main(["127.0.0.2", "9911"])

    assert calls == [
        (
            "emullm.app:app",
            {"host": "127.0.0.2", "port": 9911, "reload": False},
        )
    ]

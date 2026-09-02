"""Lightweight smoke tests for every emulated /v1 surface in emullm_api.

These deliberately stay small: for endpoints that don't need a connected
worker (models listing, embeddings, moderations, images, audio stubs, and
the files/assistants/threads/fine_tuning CRUD stubs) we just check a 200
and the expected shape. For the worker-relayed endpoints (chat
completions, completions, responses, and messages), no worker is connected in this test
process; the relay is designed to wait (not fail fast) for one, so these
tests monkeypatch a short timeout and just assert the eventual 504 --
the actual relay round-trip (including the /emullm/ws?worker_id=...
handshake) is exercised manually via scripts/emullm_worker.py against a
live server.

Multi-worker routing, capability-gated "pretend" modes, and rate
limiting are exercised directly against the module's internal state
(registering a FakeWorker under _connected_workers), since driving a real
websocket handshake end-to-end isn't worth the weight for a smoke suite.
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from emullm import api as emullm_api


class FakeWorker:
    """A worker double: records every payload sent to it and can be told
    to answer with a canned reply."""

    def __init__(self, reply: str | None = None) -> None:
        self.sent: list[dict] = []
        self.reply = reply

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)
        future = emullm_api._pending.get(payload["id"])  # noqa: SLF001
        if future and not future.done() and self.reply is not None:
            future.set_result(self.reply)


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(emullm_api.router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def isolate_emullm_state(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The files/assistants/threads/fine_tuning-job stubs (and the tokens
    # store) persist to disk (see _JsonRecordStore) so they survive a
    # real server restart; for tests, redirect each store to a throwaway
    # tmp_path dir instead of touching the real
    # workbench/server/runtime/emullm/ directory.
    for store in (
        emullm_api._files_store,
        emullm_api._assistants_store,
        emullm_api._threads_store,
        emullm_api._fine_tuning_jobs_store,
        emullm_api._fine_tuning_events_store,
        emullm_api._tokens_store,
    ):
        monkeypatch.setattr(store, "_dir", tmp_path / store._dir.name)

    # Reset all module-level relay/routing/usage state so tests don't leak
    # into each other (e.g. rate-limit counters accumulating across tests,
    # or a FakeWorker registered by one test still being "connected" for
    # the next one).
    monkeypatch.setattr(emullm_api, "_connected_workers", {})
    monkeypatch.setattr(emullm_api, "_native_worker_ids", set())
    monkeypatch.setattr(emullm_api, "_worker_teams", {})
    monkeypatch.setattr(emullm_api, "_worker_team_of", {})
    monkeypatch.setattr(emullm_api, "_worker_models", {})
    monkeypatch.setattr(emullm_api, "_worker_kinds", {})
    monkeypatch.setattr(emullm_api, "_worker_runtime_models", {})
    monkeypatch.setattr(emullm_api, "_worker_model_switch_stats", {})
    monkeypatch.setattr(emullm_api, "_worker_descriptions", {})
    monkeypatch.setattr(emullm_api, "_worker_capabilities", {})
    monkeypatch.setattr(emullm_api, "_worker_roles", {})
    monkeypatch.setattr(emullm_api, "_worker_model_masks", {})
    monkeypatch.setattr(emullm_api, "_worker_usage", {})
    monkeypatch.setattr(emullm_api, "_pending", {})
    monkeypatch.setattr(emullm_api, "_pending_models", {})
    monkeypatch.setattr(emullm_api, "_worker_not_ready_until", {})
    monkeypatch.setattr(emullm_api, "_worker_inflight", {})
    monkeypatch.setattr(emullm_api, "_worker_reservations", {})
    monkeypatch.setattr(emullm_api, "_model_inflight", {})
    monkeypatch.setattr(emullm_api, "_worker_last_busy_at", {})
    monkeypatch.setattr(emullm_api, "_worker_service_stats", {})
    monkeypatch.setattr(emullm_api, "_model_service_stats", {})
    monkeypatch.setattr(emullm_api, "_active_service_requests", {})
    monkeypatch.setattr(emullm_api, "_waiting_for_worker", {})
    monkeypatch.setattr(emullm_api, "_admin_test_tasks", {})
    monkeypatch.setattr(emullm_api, "_active_websockets", {})
    monkeypatch.setattr(emullm_api, "_worker_connection_ids", {})
    monkeypatch.setattr(emullm_api, "_copilot_client_affinity", {})
    monkeypatch.setattr(
        emullm_api,
        "_socket_worker_log_dir",
        tmp_path / "socket-worker-logs",
    )
    monkeypatch.setattr(
        emullm_api,
        "_socket_worker_log_segment_bytes",
        2 * 1024 * 1024,
    )
    monkeypatch.setattr(emullm_api, "_openai_clients", {})
    monkeypatch.setattr(emullm_api, "_openai_requests", {})
    monkeypatch.setattr(emullm_api, "_client_capacity_waiters", {})
    monkeypatch.setattr(emullm_api, "_DOC_ALIASES", {})
    # No managed-worker supervisor by default (auto mode sets one).
    emullm_api._sup.set_supervisor(None)  # noqa: SLF001
    emullm_api._copilot_api.set_manager(None)  # noqa: SLF001
    # Reset config-derived per-agent/service policy so tests don't leak.
    emullm_api.clear_agent_policies()
    monkeypatch.setattr(emullm_api, "_max_concurrent_calls", 8)
    monkeypatch.setattr(emullm_api, "_idle_worker_target", 0)
    monkeypatch.setattr(emullm_api, "_idle_grace_seconds", 30)
    monkeypatch.setattr(emullm_api, "_idle_maintenance_paused", False)
    monkeypatch.setattr(emullm_api, "_backend_fallback_delay_seconds", 0)
    emullm_api._model_fetch_cache.clear()
    emullm_api._round_robin_state.clear()
    # /emullm/storage/* derives its root from _RUNTIME_DIR directly, so
    # isolate that too instead of touching the real runtime/emullm/ dir.
    monkeypatch.setattr(emullm_api, "_RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(emullm_api, "_CONFIG_PATH", tmp_path / "config.json")


def test_list_models_includes_personas(client: TestClient) -> None:
    data = client.get("/v1/models").json()["data"]
    ids = {entry["id"] for entry in data}
    assert {
        "worker-copilot-n/percent125",
        "worker-copilot-n/percent100",
        "worker-copilot-n/percent25",
    }.issubset(ids)
    assert not any(model_id.startswith("yourself/") for model_id in ids)
    assert not any(
        model_id.endswith(("/same", "/percent75", "/percent10"))
        for model_id in ids
    )
    assert "emullm/default" in ids
    assert any(model_id.startswith("router/") for model_id in ids)


def test_emullm_default_alias_relays_through_configured_default(
    client: TestClient,
) -> None:
    worker = FakeWorker(reply="default answer")
    emullm_api._connected_workers["worker-copilot-1"] = worker  # noqa: SLF001
    emullm_api._advertised_default = "yourself/same"  # noqa: SLF001

    model = client.get("/v1/models/emullm/default")
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "emullm/default",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert model.status_code == 200
    assert model.json()["resolved_model"] == "worker-copilot-n/percent100"
    assert response.status_code == 200
    assert response.json()["model"] == "emullm/default"
    assert response.json()["choices"][0]["message"]["content"] == "default answer"
    assert worker.sent[0]["model"] == "worker-copilot-n/percent100"


def test_literal_copilot_n_is_sticky_by_client_ip_and_port_range() -> None:
    app = FastAPI()
    app.include_router(emullm_api.router)
    first = FakeWorker(reply="worker one")
    second = FakeWorker(reply="worker two")
    emullm_api._connected_workers.update(  # noqa: SLF001
        {
            "worker-copilot-1": first,
            "worker-copilot-2": second,
        }
    )

    with TestClient(app, client=("198.51.100.20", 50001)) as client_one:
        response_one = client_one.post(
            "/v1/chat/completions",
            json={
                "model": "worker-copilot-n/percent100",
                "messages": [{"role": "user", "content": "first"}],
            },
        )
    with TestClient(app, client=("198.51.100.20", 50099)) as client_two:
        response_two = client_two.post(
            "/v1/chat/completions",
            json={
                "model": "worker-copilot-n/percent100",
                "messages": [{"role": "user", "content": "second"}],
            },
        )

    assigned = response_one.headers["x-emullm-worker-id"]
    assert assigned in {"worker-copilot-1", "worker-copilot-2"}
    assert response_two.headers["x-emullm-worker-id"] == assigned
    expected = "worker one" if assigned.endswith("-1") else "worker two"
    assert response_one.json()["choices"][0]["message"]["content"] == expected
    assert response_two.json()["choices"][0]["message"]["content"] == expected
    affinity = next(iter(emullm_api._copilot_client_affinity.values()))  # noqa: SLF001
    assert affinity["port_start"] == 49_152
    assert affinity["port_end"] == 50_175


def test_best_and_worse_capability_aliases_rank_connected_workers(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        emullm_api._copilot_api,  # noqa: SLF001
        "copilot_models",
        lambda **_kwargs: {
            "models": [
                {"id": "fast-model", "name": "Fast", "quality_rank": 1},
                {"id": "small-model", "name": "Small", "quality_rank": 20},
            ],
            "source": "test",
        },
    )
    fast = FakeWorker(reply="best answer")
    small = FakeWorker(reply="worse answer")
    emullm_api._connected_workers.update(  # noqa: SLF001
        {"worker-copilot-1": fast, "worker-copilot-2": small}
    )
    emullm_api._worker_runtime_models.update(  # noqa: SLF001
        {"worker-copilot-1": "fast-model", "worker-copilot-2": "small-model"}
    )
    capabilities = {
        "audio_input": True,
        "vision_input": True,
        "file_input": True,
        "code": True,
        "summarization": True,
        "image_generation": True,
        "image_output": True,
    }
    emullm_api._worker_capabilities.update(  # noqa: SLF001
        {
            "worker-copilot-1": dict(capabilities),
            "worker-copilot-2": dict(capabilities),
        }
    )

    model_ids = {
        entry["id"] for entry in client.get("/v1/models").json()["data"]
    }
    expected_aliases = {
        f"router/{capability}-{selector}"
        for selector in ("best", "worse")
        for capability in (
            "audio",
            "video",
            "vision",
            "file",
            "code",
            "summarization",
            "image-generation",
            "image-output",
        )
    }
    assert expected_aliases.issubset(model_ids)

    best = client.post(
        "/v1/chat/completions",
        json={
            "model": "router/audio-best",
            "messages": [{"role": "user", "content": "best"}],
        },
    )
    worse = client.post(
        "/v1/chat/completions",
        json={
            "model": "router/audio-worse",
            "messages": [{"role": "user", "content": "worse"}],
        },
    )
    assert best.headers["x-emullm-worker-id"] == "worker-copilot-1"
    assert worse.headers["x-emullm-worker-id"] == "worker-copilot-2"
    assert best.json()["choices"][0]["message"]["content"] == "best answer"
    assert worse.json()["choices"][0]["message"]["content"] == "worse answer"
    assert fast.sent[0]["required_capabilities"] == ["audio_input"]
    assert small.sent[0]["required_capabilities"] == ["audio_input"]


def test_explicit_copilot_worker_number_can_be_any_positive_integer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Manager:
        def __init__(self):
            self.created = None

        def get(self, worker_id):
            if self.created is None:
                raise emullm_api._copilot_api.CopilotInstanceMissing(worker_id)  # noqa: SLF001
            return {"worker_id": worker_id, "running": True, "connected": False}

        def create(self, config, start=True):
            self.created = config
            return {"worker_id": config.worker_id, "running": start}

        def list(self):
            return []

        def start(self, _worker_id):
            return {"started": True}

    manager = Manager()
    monkeypatch.setattr(
        emullm_api._copilot_api,  # noqa: SLF001
        "get_manager",
        lambda: manager,
    )

    assert (
        emullm_api._provision_explicit_copilot_worker(  # noqa: SLF001
            "worker-copilot-624"
        )
        is manager
    )
    assert manager.created.worker_id == "worker-copilot-624"
    assert manager.created.role == "client-requested-copilot"
    assert manager.created.autostart is True
    assert manager.created.use_shared_anti_idle is True
    resolved_worker, suffix, persona = emullm_api._require_model(  # noqa: SLF001
        "worker-copilot-624/custom-backing-model"
    )
    assert resolved_worker == "worker-copilot-624"
    assert suffix == ""
    assert persona["passthrough"] is True


def test_explicit_copilot_worker_creation_respects_resource_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Manager:
        @staticmethod
        def get(worker_id):
            raise emullm_api._copilot_api.CopilotInstanceMissing(worker_id)  # noqa: SLF001

        @staticmethod
        def list():
            return [
                {
                    "worker_id": f"worker-copilot-{index + 1000}",
                    "role": "client-requested-copilot",
                }
                for index in range(3)
            ]

    monkeypatch.setattr(
        emullm_api._copilot_api,  # noqa: SLF001
        "get_manager",
        lambda: Manager(),
    )
    monkeypatch.setattr(emullm_api, "_max_concurrent_calls", 3)

    with pytest.raises(HTTPException) as error:
        emullm_api._provision_explicit_copilot_worker(  # noqa: SLF001
            "worker-copilot-999999"
        )

    assert error.value.status_code == 503
    assert "quota reached" in str(error.value.detail)


def test_list_models_exports_complete_copilot_catalog_with_provenance(
    client: TestClient,
    monkeypatch,
) -> None:
    catalog = [
        {"id": "auto", "name": "Auto"},
        {
            "id": "gemini-test",
            "name": "Gemini Test",
            "quality_rank": 3,
            "quality_tier": "highest",
            "capabilities": {
                "supports": {"vision": True},
                "limits": {
                    "max_context_window_tokens": 123_000,
                    "vision": {
                        "max_prompt_images": 3,
                        "max_prompt_image_size": 1024,
                        "supported_media_types": ["image/png"],
                    },
                },
            },
        },
    ]
    monkeypatch.setattr(
        emullm_api._copilot_api,  # noqa: SLF001
        "copilot_models",
        lambda **_kwargs: {"models": catalog, "source": "test"},
    )

    models = {entry["id"]: entry for entry in client.get("/v1/models").json()["data"]}

    assert {"router/auto", "router/gemini-test"}.issubset(models)
    gemini = models["router/gemini-test"]
    assert gemini["owned_by"] == "github-copilot"
    assert gemini["backing_model"] == "gemini-test"
    assert gemini["on_demand"] is True
    assert gemini["on_demand_worker_limit"] == 4
    assert gemini["input_modalities"]["attachment_transport"]["supported"] is True
    assert gemini["input_modalities"]["image"]["enabled"] is True
    assert gemini["input_modalities"]["audio"]["status"] == "family_implied"
    assert gemini["input_modalities"]["audio"]["enabled"] is True
    assert gemini["task_capabilities"]["code"]["enabled"] is False
    assert gemini["task_capabilities"]["summarization"]["enabled"] is True
    assert client.get("/v1/models/router/gemini-test").json()["id"] == "router/gemini-test"


def test_model_configurator_persists_model_patch_visibility_and_route(
    client: TestClient,
    monkeypatch,
) -> None:
    catalog = [{"id": "demo-model", "name": "Demo Model", "capabilities": {}}]
    monkeypatch.setattr(
        emullm_api._copilot_api,  # noqa: SLF001
        "copilot_models",
        lambda **_kwargs: {"models": catalog, "source": "test"},
    )
    revision = client.get("/emullm/admin/model-config").json()["revision"]

    saved = client.put(
        "/emullm/admin/model-config",
        json={
            "model_id": "copilot/demo-model",
            "patch": {
                "display_name": "My edited model",
                "input_modalities": {
                    "audio": {"enabled": True, "status": "operator_enabled"}
                },
            },
            "set_route": True,
            "route": ["worker-copilot-*", "https://models.example/v1"],
            "expected_revision": revision,
        },
    )

    assert saved.status_code == 200, saved.text
    assert saved.json()["route"] == [
        "worker-copilot-*",
        "https://models.example/v1",
    ]
    model = client.get("/v1/models/copilot/demo-model").json()
    assert model["display_name"] == "My edited model"
    assert model["input_modalities"]["audio"]["enabled"] is True
    assert emullm_api._model_routes["copilot/demo-model"] == [  # noqa: SLF001
        "worker-copilot-*",
        "https://models.example/v1",
    ]

    hidden = client.put(
        "/emullm/admin/model-config",
        json={
            "model_id": "copilot/demo-model",
            "hidden": True,
            "patch": {},
            "expected_revision": saved.json()["revision"],
        },
    )
    assert hidden.status_code == 200
    assert "copilot/demo-model" not in {
        entry["id"] for entry in client.get("/v1/models").json()["data"]
    }
    assert client.get("/v1/models/copilot/demo-model").status_code == 404
    hidden_catalog = {
        entry["id"]: entry
        for entry in client.get("/v1/models?hidden=true").json()["data"]
    }
    assert hidden_catalog["copilot/demo-model"]["hidden"] is True
    assert hidden_catalog["copilot/demo-model"]["exported"] is False
    admin_catalog = {
        entry["id"]: entry
        for entry in client.get("/emullm/admin/model-config").json()["models"]
    }
    assert admin_catalog["copilot/demo-model"]["hidden"] is True

    reexported = client.put(
        "/emullm/admin/model-config",
        json={
            "model_id": "copilot/demo-model",
            "hidden": False,
            "patch": {"display_name": "Re-exported model"},
            "expected_revision": hidden.json()["revision"],
        },
    )
    assert reexported.status_code == 200
    assert client.get("/v1/models/copilot/demo-model").json()["display_name"] == (
        "Re-exported model"
    )

    reset = client.put(
        "/emullm/admin/model-config",
        json={
            "model_id": "copilot/demo-model",
            "reset": True,
            "expected_revision": reexported.json()["revision"],
        },
    )
    assert reset.status_code == 200
    assert client.get("/v1/models/copilot/demo-model").status_code == 200


def test_generated_pool_and_router_aliases_honor_unexport_overrides(
    client: TestClient,
) -> None:
    emullm_api._model_catalog_overrides.update(  # noqa: SLF001
        {
            "worker-copilot-n/percent100": {"hidden": True, "patch": {}},
            "router/audio-best": {"hidden": True, "patch": {}},
        }
    )

    public_ids = {
        entry["id"] for entry in client.get("/v1/models").json()["data"]
    }
    hidden = {
        entry["id"]: entry
        for entry in client.get("/v1/models?hidden=true").json()["data"]
    }

    assert "worker-copilot-n/percent100" not in public_ids
    assert "router/audio-best" not in public_ids
    assert hidden["worker-copilot-n/percent100"]["hidden"] is True
    assert hidden["router/audio-best"]["hidden"] is True
    assert client.get("/v1/models/worker-copilot-n/percent100").status_code == 404
    assert client.get("/v1/models/router/audio-best").status_code == 404


def test_on_demand_copilot_provisioner_uses_four_named_slots(monkeypatch) -> None:
    class FakeManager:
        def __init__(self) -> None:
            self.instances: dict[str, dict] = {}

        def list(self) -> list[dict]:
            return list(self.instances.values())

        def create(self, config, *, start=True):
            self.instances[config.worker_id] = {
                "worker_id": config.worker_id,
                "running": start,
                "config": config.model_dump(mode="json"),
            }

        def start(self, worker_id):
            self.instances[worker_id]["running"] = True

        def update(self, worker_id, config, *, restart=True):
            self.instances[worker_id] = {
                "worker_id": worker_id,
                "running": False,
                "config": config.model_dump(mode="json"),
            }

    manager = FakeManager()
    monkeypatch.setattr(emullm_api._copilot_api, "get_manager", lambda: manager)  # noqa: SLF001
    entry = {
        "id": "copilot/model",
        "on_demand": True,
        "input_modalities": {
            "image": {"enabled": True},
            "audio": {"enabled": False},
            "general_file": {"enabled": True},
        },
    }

    workers = [
        emullm_api._provision_on_demand_copilot(f"model-{index}", entry)[0]  # noqa: SLF001
        for index in range(1, 5)
    ]

    assert workers == [f"worker-copilot-{index}" for index in range(5, 9)]
    assert manager.instances["worker-copilot-5"]["config"]["model"] == "model-1"
    assert manager.instances["worker-copilot-5"]["config"]["capabilities"] == [
        "vision_input",
        "!audio_input",
        "file_input",
    ]
    with pytest.raises(HTTPException, match="all 4 elastic Copilot workers are busy"):
        emullm_api._provision_on_demand_copilot("model-5", entry)  # noqa: SLF001


def test_busy_matching_model_launches_an_elastic_replica(monkeypatch) -> None:
    class FakeManager:
        def __init__(self) -> None:
            self.instances = {
                "worker-copilot-5": {
                    "worker_id": "worker-copilot-5",
                    "running": True,
                    "connected": True,
                    "config": {"model": "same-model"},
                }
            }

        def list(self):
            return list(self.instances.values())

        def create(self, config, *, start=True):
            self.instances[config.worker_id] = {
                "worker_id": config.worker_id,
                "running": start,
                "connected": False,
                "config": config.model_dump(mode="json"),
            }

    manager = FakeManager()
    monkeypatch.setattr(emullm_api._copilot_api, "get_manager", lambda: manager)  # noqa: SLF001
    emullm_api._worker_inflight["worker-copilot-5"] = 1  # noqa: SLF001
    entry = {
        "id": "copilot/same-model",
        "on_demand": True,
        "input_modalities": {},
    }

    worker_id, _, switched = emullm_api._provision_on_demand_copilot(  # noqa: SLF001
        "same-model",
        entry,
    )

    assert worker_id == "worker-copilot-6"
    assert switched is False
    assert manager.instances[worker_id]["config"]["model"] == "same-model"
    assert emullm_api._worker_reservations[worker_id] == 1  # noqa: SLF001


def test_full_elastic_pool_reuses_idle_worker_via_runtime_switch(
    monkeypatch,
) -> None:
    class FakeManager:
        @staticmethod
        def list():
            return [
                {
                    "worker_id": f"worker-copilot-{index}",
                    "running": True,
                    "connected": True,
                    "selected_model": f"old-model-{index}",
                    "config": {"model": f"old-model-{index}"},
                }
                for index in range(5, 9)
            ]

    monkeypatch.setattr(
        emullm_api._copilot_api,  # noqa: SLF001
        "get_manager",
        lambda: FakeManager(),
    )
    worker_id, _, switched = emullm_api._provision_on_demand_copilot(  # noqa: SLF001
        "new-model",
        {"id": "copilot/new-model", "on_demand": True, "input_modalities": {}},
    )
    assert worker_id == "worker-copilot-5"
    assert switched is True
    assert emullm_api._worker_reservations[worker_id] == 1  # noqa: SLF001


def test_ensure_on_demand_copilot_switches_with_catalog_entry(monkeypatch) -> None:
    """Regression: the runtime-switch branch passed an undefined name
    (``model_entry``) instead of the resolved catalog ``entry``, so every
    on-demand model that needed a runtime switch (e.g. copilot/gpt-5.3-codex)
    raised NameError -> HTTP 500. This drives the real function body to make
    sure the resolved entry reaches _set_worker_runtime_model."""

    entry = {"id": "copilot/gpt-5.3-codex", "on_demand": True, "input_modalities": {}}

    class FakeManager:
        @staticmethod
        def get(worker_id):
            return {"worker_id": worker_id, "running": True, "connected": True}

    monkeypatch.setattr(
        emullm_api, "_copilot_model_metadata", lambda backing: {"id": backing}  # noqa: SLF001
    )
    monkeypatch.setattr(
        emullm_api, "_copilot_catalog_model_entry", lambda metadata: entry  # noqa: SLF001
    )
    monkeypatch.setattr(
        emullm_api, "_apply_model_catalog_override", lambda value: value  # noqa: SLF001
    )
    monkeypatch.setattr(
        emullm_api,
        "_provision_on_demand_copilot",  # noqa: SLF001
        lambda backing, catalog_entry, **kwargs: ("worker-copilot-5", FakeManager(), True),
    )

    captured: dict = {}

    async def fake_switch(worker_id, backing_model, model_entry):
        captured["worker_id"] = worker_id
        captured["backing_model"] = backing_model
        captured["entry"] = model_entry

    monkeypatch.setattr(emullm_api, "_set_worker_runtime_model", fake_switch)  # noqa: SLF001

    worker_id = asyncio.run(
        emullm_api._ensure_on_demand_copilot(  # noqa: SLF001
            "copilot/gpt-5.3-codex",
            "gpt-5.3-codex",
        )
    )

    assert worker_id == "worker-copilot-5"
    assert captured["entry"] is entry
    assert captured["backing_model"] == "gpt-5.3-codex"


def test_best_match_copilot_backing_prefers_exact_then_family(monkeypatch) -> None:
    """When no worker will accept the requested Copilot model, substitution
    picks the closest available model: an exact id wins outright; otherwise a
    same-family match (e.g. codex -> code) is preferred over mere name
    similarity; the public model id is left for the caller to preserve."""

    catalog = [
        {"id": "gpt-5.6-sol"},
        {"id": "gpt-5.5"},
        {"id": "kimi-k2.7-code"},
        {"id": "mai-code-1.1-flash"},
        {"id": "claude-opus-5"},
        {"id": "gpt-5.3-codex"},
    ]
    monkeypatch.setattr(
        emullm_api._copilot_api,  # noqa: SLF001
        "copilot_models",
        lambda **_kwargs: {"models": catalog, "source": "test"},
    )
    monkeypatch.setattr(emullm_api, "_worker_runtime_models", {}, raising=False)  # noqa: SLF001

    # Exact id present -> returned unchanged.
    assert emullm_api._best_match_copilot_backing("gpt-5.3-codex") == "gpt-5.3-codex"  # noqa: SLF001

    # Exact id absent (e.g. renamed/removed) -> a code-family model is chosen
    # over a merely name-similar gpt-* model.
    substitute = emullm_api._best_match_copilot_backing("gpt-6-codex")  # noqa: SLF001
    assert substitute in {"kimi-k2.7-code", "mai-code-1.1-flash", "gpt-5.3-codex"}
    assert "code" in substitute

    # Excluding the exact model (simulating "this one was rejected") still
    # yields a different, same-family substitute.
    excluded = emullm_api._best_match_copilot_backing(  # noqa: SLF001
        "gpt-5.3-codex", exclude={"gpt-5.3-codex"}
    )
    assert excluded != "gpt-5.3-codex"
    assert "code" in excluded


def test_best_match_copilot_backing_empty_catalog(monkeypatch) -> None:
    monkeypatch.setattr(
        emullm_api._copilot_api,  # noqa: SLF001
        "copilot_models",
        lambda **_kwargs: {"models": [], "source": "test"},
    )
    assert emullm_api._best_match_copilot_backing("gpt-5.3-codex") is None  # noqa: SLF001



    class SwitchPeer:
        def __init__(self) -> None:
            self.sent: list[dict] = []

        async def send_json(self, payload: dict) -> None:
            self.sent.append(payload)
            await emullm_api._handle_worker_message(  # noqa: SLF001
                "worker-copilot-5",
                {
                    "type": "model_changed",
                    "id": payload["id"],
                    "model": payload["model"],
                },
            )

    peer = SwitchPeer()
    emullm_api._connected_workers["worker-copilot-5"] = peer  # noqa: SLF001
    asyncio.run(
        emullm_api._set_worker_runtime_model(  # noqa: SLF001
            "worker-copilot-5",
            "new-model",
            {
                "capabilities": {},
                "input_modalities": {
                    "audio": {"enabled": True},
                },
            },
        )
    )
    assert peer.sent[0]["type"] == "set_model"
    assert peer.sent[0]["model"] == "new-model"
    assert peer.sent[0]["modelmasks"] == ["router/new-model"]
    assert "audio_input" in peer.sent[0]["capabilities"]


def test_worker_registration_tracks_runtime_model_switches() -> None:
    emullm_api._apply_worker_registration(  # noqa: SLF001
        "worker-copilot-5",
        {"runtime_model": "model-a"},
    )
    emullm_api._apply_worker_registration(  # noqa: SLF001
        "worker-copilot-5",
        {"runtime_model": "model-b"},
    )
    emullm_api._apply_worker_registration(  # noqa: SLF001
        "worker-copilot-5",
        {"runtime_model": "model-b"},
    )
    state = emullm_api.admin_state()
    switches = state["worker_model_switches"]["worker-copilot-5"]
    assert switches["count"] == 1
    assert switches["previous_model"] == "model-a"
    assert switches["new_model"] == "model-b"
    assert state["team_model_switches"] == 1


def test_simultaneous_call_limit_defaults_to_fifty(monkeypatch) -> None:
    monkeypatch.setattr(emullm_api, "_max_concurrent_calls", 50)
    for index in range(50):
        emullm_api._reserve_worker(f"worker-{index}")  # noqa: SLF001
    with pytest.raises(HTTPException, match="maximum simultaneous call limit"):
        emullm_api._reserve_worker("worker-51")  # noqa: SLF001


def test_client_capacity_reserves_five_or_thirty_percent(monkeypatch) -> None:
    monkeypatch.setattr(emullm_api, "_max_concurrent_calls", 50)
    emullm_api._connected_workers.update(  # noqa: SLF001
        {f"worker-{index}": FakeWorker() for index in range(21)}
    )
    assert emullm_api._client_worker_capacity() == (14, 7)  # noqa: SLF001
    emullm_api._connected_workers.clear()  # noqa: SLF001
    emullm_api._connected_workers.update(  # noqa: SLF001
        {f"worker-{index}": FakeWorker() for index in range(10)}
    )
    assert emullm_api._client_worker_capacity() == (5, 5)  # noqa: SLF001


def test_admin_state_reports_stuck_workers_and_connection_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        emullm_api,
        "_active_service_requests",
        {
            "request-stuck": {
                "worker_id": "worker-stuck",
                "model": "copilot/gpt-5.3-codex",
                "service_kind": "vision",
                "started_at": "2026-09-01T00:00:00Z",
                "started_monotonic": time.monotonic() - 130,
            }
        },
    )
    monkeypatch.setattr(
        emullm_api._copilot_api,  # noqa: SLF001
        "manager_status",
        lambda: [
            {
                "worker_id": "worker-disconnected",
                "running": True,
                "connected": False,
                "runtime": {
                    "connection_errors": 3,
                    "last_connection_error": "connection reset",
                    "last_disconnected_at": 123.0,
                },
            }
        ],
    )

    state = emullm_api.admin_state()

    assert state["concurrency"]["stuck_workers"] == 1
    assert state["stuck_workers"][0]["worker_id"] == "worker-stuck"
    assert state["stuck_workers"][0]["age_seconds"] >= 120
    assert state["connection_errors"] == [
        {
            "worker_id": "worker-disconnected",
            "connected": False,
            "running": True,
            "connection_errors": 3,
            "last_connection_error": "connection reset",
            "last_disconnected_at": 123.0,
        }
    ]


def test_record_store_count_does_not_deserialize_records(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(emullm_api, "_RUNTIME_DIR", tmp_path)
    store = emullm_api._JsonRecordStore("files")  # noqa: SLF001
    store.save({"id": "one"})
    store.save({"id": "two"})
    (tmp_path / "files" / "broken.json").write_text("not json", encoding="utf-8")
    (tmp_path / "files" / "payload.bin").write_bytes(b"x")

    assert store.count() == 3


def test_concurrency_and_backend_delay_config_apply_live() -> None:
    emullm_api.apply_agent_policies(  # noqa: SLF001
        {
            "max_concurrent_calls": 30,
            "idle_worker_target": 5,
            "idle_grace_seconds": 30,
            "backend_fallback_delay_seconds": 5,
        }
    )
    assert emullm_api._max_concurrent_calls == 30  # noqa: SLF001
    assert emullm_api._on_demand_copilot_limit() == 26  # noqa: SLF001
    assert emullm_api._idle_worker_target == 5  # noqa: SLF001
    assert emullm_api._idle_grace_seconds == 30  # noqa: SLF001
    assert emullm_api._backend_fallback_delay_seconds == 5  # noqa: SLF001


def test_idle_maintainer_restores_five_ready_workers(monkeypatch) -> None:
    monkeypatch.setattr(emullm_api, "_max_concurrent_calls", 50)
    monkeypatch.setattr(emullm_api, "_idle_worker_target", 5)
    emullm_api._connected_workers.update(  # noqa: SLF001
        {f"worker-copilot-{index}": FakeWorker() for index in range(1, 5)}
    )

    async def ensure(
        public_model_id,
        backing_model,
        *,
        require_new=False,
        warmup=False,
    ):
        assert (public_model_id, backing_model, require_new, warmup) == (
            "router/auto",
            "auto",
            True,
            True,
        )
        worker_id = "worker-copilot-5"
        emullm_api._connected_workers[worker_id] = FakeWorker()  # noqa: SLF001
        emullm_api._reserve_worker(worker_id)  # noqa: SLF001
        return worker_id

    monkeypatch.setattr(emullm_api, "_ensure_on_demand_copilot", ensure)

    assert asyncio.run(
        emullm_api.maintain_idle_copilot_workers_once()
    ) == ["worker-copilot-5"]
    assert emullm_api._idle_copilot_worker_count() == 5  # noqa: SLF001


def test_idle_maintainer_waits_for_each_new_worker_to_connect(monkeypatch) -> None:
    monkeypatch.setattr(emullm_api, "_max_concurrent_calls", 50)
    monkeypatch.setattr(emullm_api, "_idle_worker_target", 5)
    emullm_api._connected_workers.update(  # noqa: SLF001
        {f"worker-copilot-{index}": FakeWorker() for index in range(1, 5)}
    )
    calls: list[str] = []

    async def ensure(
        _public_model_id,
        _backing_model,
        *,
        require_new=False,
        warmup=False,
    ):
        assert require_new is True
        assert warmup is True
        calls.append("worker-copilot-5")
        if len(calls) > 1:
            raise AssertionError("maintainer must wait for the first worker to connect")
        emullm_api._reserve_worker("worker-copilot-5")  # noqa: SLF001
        return "worker-copilot-5"

    monkeypatch.setattr(emullm_api, "_ensure_on_demand_copilot", ensure)

    assert asyncio.run(
        emullm_api.maintain_idle_copilot_workers_once()
    ) == ["worker-copilot-5"]
    assert calls == ["worker-copilot-5"]


def test_async_fleet_handlers_offload_manager_listing() -> None:
    source = Path(emullm_api.__file__).read_text(encoding="utf-8")
    assert source.count("instances = await asyncio.to_thread(manager.list)") >= 2
    assert "instances = await asyncio.to_thread(_copilot_api.manager_status)" in source


def test_idle_maintainer_scales_excess_elastic_workers_down(monkeypatch) -> None:
    monkeypatch.setattr(emullm_api, "_max_concurrent_calls", 50)
    monkeypatch.setattr(emullm_api, "_idle_worker_target", 5)
    class ShutdownPeer:
        def __init__(self, worker_id):
            self.worker_id = worker_id

        async def send_json(self, payload):
            assert payload["type"] == "shutdown"
            emullm_api._connected_workers.pop(self.worker_id, None)  # noqa: SLF001

    emullm_api._connected_workers.update(  # noqa: SLF001
        {
            f"worker-copilot-{index}": ShutdownPeer(
                f"worker-copilot-{index}"
            )
            for index in range(1, 7)
        }
    )

    class FakeManager:
        stopped: list[str] = []

        @staticmethod
        def list():
            return [
                {
                    "worker_id": f"worker-copilot-{index}",
                    "running": True,
                    "connected": True,
                }
                for index in range(1, 7)
            ]

        def stop(self, worker_id):
            self.stopped.append(worker_id)

    manager = FakeManager()
    monkeypatch.setattr(
        emullm_api._copilot_api,  # noqa: SLF001
        "get_manager",
        lambda: manager,
    )

    assert asyncio.run(emullm_api.maintain_idle_copilot_workers_once()) == []
    assert manager.stopped == []
    assert emullm_api._idle_copilot_worker_count() == 5  # noqa: SLF001


def test_worker_is_not_idle_during_configured_grace_period(monkeypatch) -> None:
    monkeypatch.setattr(emullm_api, "_idle_grace_seconds", 30)
    now = time.monotonic()
    emullm_api._worker_last_busy_at["worker-copilot-5"] = now  # noqa: SLF001
    assert emullm_api._worker_is_idle("worker-copilot-5") is False  # noqa: SLF001
    emullm_api._worker_last_busy_at["worker-copilot-5"] = now - 31  # noqa: SLF001
    assert emullm_api._worker_is_idle("worker-copilot-5") is True  # noqa: SLF001


def test_model_configurator_lists_only_reserved_on_demand_slots(
    client: TestClient,
    monkeypatch,
) -> None:
    class FakeManager:
        @staticmethod
        def list():
            return [
                {"worker_id": "worker-copilot-1", "role": "headless-copilot"},
                {"worker_id": "worker-copilot-5", "role": "on-demand-copilot"},
            ]

    monkeypatch.setattr(
        emullm_api._copilot_api,  # noqa: SLF001
        "get_manager",
        lambda: FakeManager(),
    )
    response = client.get("/emullm/admin/model-config")

    assert response.status_code == 200
    assert [
        slot["worker_id"] for slot in response.json()["on_demand"]["slots"]
    ] == ["worker-copilot-5"]


def test_copilot_catalog_request_routes_to_ensured_on_demand_worker(
    client: TestClient,
    monkeypatch,
) -> None:
    catalog = [{"id": "demo-model", "name": "Demo Model", "capabilities": {}}]
    monkeypatch.setattr(
        emullm_api._copilot_api,  # noqa: SLF001
        "copilot_models",
        lambda **_kwargs: {"models": catalog, "source": "test"},
    )
    worker = FakeWorker(reply="on-demand answer")
    emullm_api._connected_workers["worker-copilot-5"] = worker  # noqa: SLF001
    calls = []

    async def ensure(public_model_id: str, backing_model: str) -> str:
        calls.append((public_model_id, backing_model))
        return "worker-copilot-5"

    monkeypatch.setattr(emullm_api, "_ensure_on_demand_copilot", ensure)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "copilot/demo-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "on-demand answer"
    assert calls == [("copilot/demo-model", "demo-model")]
    assert worker.sent[0]["model"] == "copilot/demo-model"


def test_loaded_copilot_worker_route_requires_matching_backing_model(
    monkeypatch,
) -> None:
    catalog = [
        {"id": "model-a", "name": "Model A"},
        {"id": "model-b", "name": "Model B"},
    ]
    monkeypatch.setattr(
        emullm_api._copilot_api,  # noqa: SLF001
        "copilot_models",
        lambda **_kwargs: {"models": catalog, "source": "test"},
    )
    emullm_api._connected_workers.update(  # noqa: SLF001
        {
            "worker-copilot-5": FakeWorker(),
            "worker-copilot-6": FakeWorker(),
        }
    )
    emullm_api._worker_runtime_models.update(  # noqa: SLF001
        {
            "worker-copilot-5": "model-a",
            "worker-copilot-6": "model-b",
        }
    )

    assert emullm_api._route_worker_candidates(  # noqa: SLF001
        "copilot/model-b",
        "worker-copilot-*",
    ) == ["worker-copilot-6"]


def test_unknown_worker_glob_still_applies_capability_ordering() -> None:
    emullm_api._connected_workers.update(  # noqa: SLF001
        {
            "worker-unknown-capable": FakeWorker(),
            "worker-unknown-unknown": FakeWorker(),
            "worker-unknown-declined": FakeWorker(),
            "other-worker": FakeWorker(),
        }
    )
    emullm_api._native_worker_ids.update(emullm_api._connected_workers)  # noqa: SLF001
    emullm_api._worker_capabilities.update(  # noqa: SLF001
        {
            "worker-unknown-capable": {"audio_input": True},
            "worker-unknown-declined": {"audio_input": False},
        }
    )

    assert emullm_api._route_worker_candidates(  # noqa: SLF001
        "gpt-4o-audio-preview",
        "worker-unknown-*",
        {"audio_input"},
    ) == ["worker-unknown-capable", "worker-unknown-unknown"]


def test_worker_in_name_route_targets_named_worker(client: TestClient) -> None:
    alice = FakeWorker(reply="alice answered")
    emullm_api._connected_workers["alice"] = alice  # noqa: SLF001
    emullm_api._model_routes["alice/custom"] = ["worker-in-name"]  # noqa: SLF001

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "alice/custom",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "alice answered"
    assert len(alice.sent) == 1


def test_backend_wildcard_route_expands_configured_backends(
    client: TestClient,
    monkeypatch,
) -> None:
    calls = []

    async def proxy(backend, model, prompt, instruction):
        calls.append((backend["name"], model, prompt, instruction))
        return "backend answered"

    monkeypatch.setattr(
        emullm_api,
        "_all_backends",
        lambda: [
            {"name": "one", "base_url": "https://one.example/v1"},
            {"name": "two", "base_url": "https://two.example/v1"},
        ],
    )
    monkeypatch.setattr(emullm_api, "_proxy_chat", proxy)
    emullm_api._model_routes["vendor/model"] = ["backend-*"]  # noqa: SLF001

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "vendor/model",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "backend answered"
    assert calls[0][0] in {"one", "two"}
    assert [
        backend["name"]
        for backend in emullm_api._route_backend_candidates("backend-one")  # noqa: SLF001
    ] == ["one"]


def test_list_models_hides_concrete_backing_alias_but_direct_request_works(
    client: TestClient,
) -> None:
    emullm_api._model_routes["vendor/simulated"] = [  # noqa: SLF001
        "worker-copilot-*",
        "https://llm.example/v1",
    ]
    emullm_api._connected_workers["worker-copilot-1"] = FakeWorker(reply="ok")  # noqa: SLF001
    emullm_api._worker_kinds["worker-copilot-1"] = "headless-copilot"  # noqa: SLF001
    emullm_api._worker_runtime_models["worker-copilot-1"] = "gpt-5.6-sol"  # noqa: SLF001
    emullm_api._worker_descriptions["worker-copilot-1"] = "Resident Copilot worker."  # noqa: SLF001

    models = {entry["id"]: entry for entry in client.get("/v1/models").json()["data"]}

    simulated = models["vendor/simulated"]
    assert simulated["simulated"] is True
    assert simulated["route_targets"] == [
        "worker-copilot-*",
        "https://llm.example/v1",
    ]
    assert simulated["active_workers"] == ["worker-copilot-1"]
    assert simulated["backing_models"] == {"worker-copilot-1": "gpt-5.6-sol"}

    alias_id = "worker-copilot-1/gpt-5.6-sol"
    assert alias_id not in models
    assert client.get(f"/v1/models/{alias_id}").status_code == 404
    reply = client.post(
        "/v1/chat/completions",
        json={
            "model": "worker-copilot-1/gpt-5.6-sol",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert reply.status_code == 200
    assert emullm_api._connected_workers["worker-copilot-1"].sent[-1]["model"] == alias_id  # noqa: SLF001


def test_service_catalog_advertises_one_shared_worker_socket(client: TestClient) -> None:
    response = client.get("/emullm/endpoints")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "emullm"
    worker_sockets = [
        endpoint
        for endpoint in body["endpoints"]
        if endpoint["path"].startswith("/emullm/") and "WS" in endpoint["methods"]
    ]
    assert any(endpoint["path"] == "/emullm/ws" for endpoint in worker_sockets)
    assert all("{worker_id}" not in endpoint["path"] for endpoint in worker_sockets)
    assert all(endpoint["comment"] and "parameters" in endpoint for endpoint in body["endpoints"])

    worker_socket = next(endpoint for endpoint in worker_sockets if endpoint["path"] == "/emullm/ws")
    assert {parameter["name"] for parameter in worker_socket["parameters"]} == {"worker_id", "modelmasks"}
    chat = next(endpoint for endpoint in body["endpoints"] if endpoint["path"] == "/v1/chat/completions")
    assert chat["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith("/ChatRequest")
    interaction_log = next(
        endpoint for endpoint in body["endpoints"] if endpoint["path"] == "/emullm/websock_to_llm_user/events"
    )
    assert {"worker_id", "model", "modelmask", "type", "after", "limit"} == {
        parameter["name"] for parameter in interaction_log["parameters"]
    }
    assert "ChatRequest" in body["schemas"]
    assert client.get("/endpoints").json()["endpoints"] == body["endpoints"]


def test_get_single_model(client: TestClient) -> None:
    response = client.get("/v1/models/yourself/percent25")
    assert response.status_code == 200
    assert response.json()["id"] == "yourself/percent25"
    # A bare/unknown worker_id still resolves (falls back to the default
    # persona menu -- see _models_for), but an unknown SUFFIX never does.
    assert client.get("/v1/models/yourself/no-such-suffix").status_code == 404


@pytest.fixture()
def short_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    # Without this, "no worker connected" would wait the full (900s)
    # production timeout before giving up -- these tests want that to
    # happen almost instantly instead.
    monkeypatch.setattr(emullm_api, "_REQUEST_TIMEOUT_SECONDS", 0.3)


def test_chat_completions_without_worker_waits_then_504(client: TestClient, short_timeout: None) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "yourself/same", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 504


def test_chat_completions_persists_inline_image_as_native_worker_attachment(
    client: TestClient,
) -> None:
    worker = FakeWorker(reply="American flag")
    emullm_api._connected_workers["yourself"] = worker  # noqa: SLF001
    image_bytes = b"\x89PNG\r\n\x1a\nflag-image"
    data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "yourself/same",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the image."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "American flag"
    request = worker.sent[0]
    assert request["kind"] == "vision"
    assert request["images"] == [request["attachments"][0]["url"]]
    assert request["attachments"][0]["mime_type"] == "image/png"
    assert request["attachments"][0]["bytes"] == len(image_bytes)
    assert data_url not in json.dumps(request)
    download = client.get(request["attachments"][0]["url"])
    assert download.status_code == 200
    assert download.content == image_bytes
    assert download.headers["content-type"].startswith("image/png")


def test_chat_completions_rejects_invalid_inline_image_base64(
    client: TestClient,
) -> None:
    emullm_api._connected_workers["yourself"] = FakeWorker(reply="unused")  # noqa: SLF001
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "yourself/same",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the image."},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,not-valid!"},
                        },
                    ],
                }
            ],
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "inline image 1 is not valid base64"


def test_chat_completions_persists_input_audio_as_native_worker_attachment(
    client: TestClient,
) -> None:
    worker = FakeWorker(reply="The tones rise in pitch.")
    emullm_api._connected_workers["yourself"] = worker  # noqa: SLF001
    audio_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt audio-data"

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "yourself/same",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this audio."},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(audio_bytes).decode("ascii"),
                                "format": "wav",
                            },
                        },
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "The tones rise in pitch."
    request = worker.sent[0]
    assert request["kind"] == "audio_attachment"
    assert request["audio"] == request["attachments"][0]["url"]
    assert request["attachments"][0]["mime_type"] == "audio/wav"
    assert request["attachments"][0]["bytes"] == len(audio_bytes)
    assert base64.b64encode(audio_bytes).decode("ascii") not in json.dumps(request)
    download = client.get(request["attachments"][0]["url"])
    assert download.status_code == 200
    assert download.content == audio_bytes
    assert download.headers["content-type"].startswith("audio/wav")


def test_audio_round_robin_prefers_declared_capable_worker(
    client: TestClient,
) -> None:
    unknown = FakeWorker(reply="unknown")
    capable = FakeWorker(reply="capable")
    declined = FakeWorker(reply="declined")
    emullm_api._connected_workers.update(  # noqa: SLF001
        {
            "worker-copilot-1": unknown,
            "worker-copilot-2": capable,
            "worker-copilot-3": declined,
        }
    )
    emullm_api._native_worker_ids.update(emullm_api._connected_workers)  # noqa: SLF001
    emullm_api._worker_capabilities.update(  # noqa: SLF001
        {
            "worker-copilot-2": {"audio_input": True},
            "worker-copilot-3": {"audio_input": False},
        }
    )
    emullm_api._model_routes["gpt-4o-audio-preview"] = ["worker-copilot-*"]  # noqa: SLF001

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o-audio-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this audio."},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(b"RIFF audio").decode("ascii"),
                                "format": "wav",
                            },
                        },
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "capable"
    assert unknown.sent == []
    assert declined.sent == []
    assert capable.sent[0]["required_capabilities"] == ["audio_input"]


def test_audio_round_robin_tries_unknown_worker_after_capable_worker_rejects(
    client: TestClient,
) -> None:
    class RejectingWorker:
        def __init__(self) -> None:
            self.sent: list[dict] = []

        async def send_json(self, payload: dict) -> None:
            self.sent.append(payload)
            await emullm_api._handle_worker_message(  # noqa: SLF001
                "worker-copilot-1",
                {"type": "reject", "id": payload["id"], "reason": "audio unavailable"},
            )

    rejecting = RejectingWorker()
    fallback = FakeWorker(reply="fallback accepted")
    emullm_api._connected_workers.update(  # noqa: SLF001
        {
            "worker-copilot-1": rejecting,
            "worker-copilot-2": fallback,
        }
    )
    emullm_api._native_worker_ids.update(emullm_api._connected_workers)  # noqa: SLF001
    emullm_api._worker_capabilities["worker-copilot-1"] = {"audio_input": True}  # noqa: SLF001
    emullm_api._model_routes["gpt-4o-mini-audio-preview"] = ["worker-copilot-*"]  # noqa: SLF001

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o-mini-audio-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(b"RIFF audio").decode("ascii"),
                                "format": "wav",
                            },
                        }
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "fallback accepted"
    assert len(rejecting.sent) == 1
    assert len(fallback.sent) == 1


def test_not_ready_worker_moves_to_next_route_without_cross_request_cooldown(
    client: TestClient,
) -> None:
    class NotReadyWorker:
        def __init__(self) -> None:
            self.sent: list[dict] = []

        async def send_json(self, payload: dict) -> None:
            self.sent.append(payload)
            await emullm_api._handle_worker_message(  # noqa: SLF001
                "worker-copilot-1",
                {
                    "type": "not_ready",
                    "id": payload["id"],
                    "reason": "runtime temporarily returned no answer",
                    "retry_after": 15,
                },
            )

    deferred = NotReadyWorker()
    fallback = FakeWorker(reply="fallback completed")
    emullm_api._connected_workers.update(  # noqa: SLF001
        {
            "worker-copilot-1": deferred,
            "worker-copilot-2": fallback,
        }
    )
    emullm_api._native_worker_ids.update(emullm_api._connected_workers)  # noqa: SLF001
    emullm_api._worker_capabilities["worker-copilot-1"] = {"audio_input": True}  # noqa: SLF001
    emullm_api._model_routes["audio/model"] = ["worker-copilot-*"]  # noqa: SLF001
    payload = {
        "model": "audio/model",
        "messages": [{"role": "user", "content": "listen"}],
        "required_capabilities": ["audio_input"],
    }

    first = client.post("/v1/chat/completions", json=payload)
    second = client.post("/v1/chat/completions", json=payload)

    assert first.status_code == second.status_code == 200
    assert first.json()["choices"][0]["message"]["content"] == "fallback completed"
    assert len(deferred.sent) == 2
    assert len(fallback.sent) == 2
    assert emullm_api._worker_retry_delay("worker-copilot-1") == 0  # noqa: SLF001
    events = client.get(
        "/emullm/websock_to_llm_user/events",
        params={"model": "audio/model", "type": "LLM_NOT_READY"},
    ).json()["events"]
    assert len(events) == 2
    assert all(event["data"]["reported_retry_after"] == 15 for event in events)
    assert all(event["data"]["cooldown_applied"] is False for event in events)


def test_on_demand_model_moves_to_next_servant_when_primary_is_not_ready(
    monkeypatch,
) -> None:
    calls: list[str] = []

    async def ensure(_public_model_id, _backing_model):
        emullm_api._reserve_worker("worker-copilot-6")  # noqa: SLF001
        return "worker-copilot-6"

    async def relay(worker_id, *_args, **_kwargs):
        calls.append(worker_id)
        if worker_id == "worker-copilot-6":
            emullm_api._begin_worker_request(worker_id, "copilot/test")  # noqa: SLF001
            emullm_api._end_worker_request(worker_id, "copilot/test")  # noqa: SLF001
            raise emullm_api._WorkerNotReady(worker_id, "temporary SDK miss", 15)  # noqa: SLF001
        return {"content": "fallback servant answered"}

    monkeypatch.setattr(emullm_api, "_copilot_backing_model", lambda _model: "test")
    monkeypatch.setattr(emullm_api, "_ensure_on_demand_copilot", ensure)
    monkeypatch.setattr(
        emullm_api,
        "_route_worker_candidates",
        lambda *_args, **_kwargs: ["worker-copilot-6", "worker-copilot-7"],
    )
    monkeypatch.setattr(emullm_api, "_check_and_record_usage", lambda _worker_id: None)
    monkeypatch.setattr(emullm_api, "_relay_to_worker", relay)

    result = asyncio.run(emullm_api._relay_full("copilot/test", "hello"))  # noqa: SLF001

    assert result == {"content": "fallback servant answered"}
    assert calls == ["worker-copilot-6", "worker-copilot-7"]
    assert emullm_api._worker_retry_delay("worker-copilot-6") == 0  # noqa: SLF001


def test_on_demand_model_substitutes_best_match_when_exact_backing_rejected(
    monkeypatch,
) -> None:
    """When the exact requested Copilot backing can't be served (every worker
    rejects it), the relay switches a worker to the best-match substitute and
    reprompts it, while keeping the public model id the caller asked for -- we
    can switch a worker's model at runtime without asking, so an unavailable or
    rejected model is served by the closest available one."""

    ensured_backings: list[str] = []
    relay_calls: list[tuple[str, str, str]] = []  # (worker_id, public_model, instruction)

    async def ensure(_public_model_id, backing_model):
        ensured_backings.append(backing_model)
        return "worker-codex" if backing_model == "gpt-5.3-codex" else "worker-sub"

    async def relay(worker_id, model, _prompt_text, instruction, **_kwargs):
        relay_calls.append((worker_id, model, instruction or ""))
        if worker_id == "worker-codex":
            raise emullm_api._WorkerRejected(worker_id, "model unavailable")  # noqa: SLF001
        return {"content": "substitute answered"}

    monkeypatch.setattr(
        emullm_api, "_copilot_backing_model", lambda _model: "gpt-5.3-codex"
    )
    monkeypatch.setattr(
        emullm_api,
        "_best_match_copilot_backing",
        lambda requested, exclude=None: (
            "kimi-k2.7-code" if "gpt-5.3-codex" in (exclude or set()) else "gpt-5.3-codex"
        ),
    )
    monkeypatch.setattr(emullm_api, "_ensure_on_demand_copilot", ensure)
    monkeypatch.setattr(emullm_api, "_route_worker_candidates", lambda *_a, **_k: [])
    monkeypatch.setattr(emullm_api, "_check_and_record_usage", lambda _worker_id: None)
    monkeypatch.setattr(emullm_api, "_relay_to_worker", relay)

    result = asyncio.run(
        emullm_api._relay_full("copilot/legacy-codex", "hello")  # noqa: SLF001
    )

    assert result == {"content": "substitute answered"}
    # Switched from the exact backing to the best-match substitute at runtime.
    assert ensured_backings == ["gpt-5.3-codex", "kimi-k2.7-code"]
    # Reprompted the substitute worker after the switch.
    assert [call[0] for call in relay_calls] == ["worker-codex", "worker-sub"]
    # Public model id the caller asked for is preserved on every attempt.
    assert all(call[1] == "copilot/legacy-codex" for call in relay_calls)
    # The worker was told to be the substitute backing model.
    assert "kimi-k2.7-code" in relay_calls[1][2]


def test_chat_completion_identifies_selected_worker(client: TestClient) -> None:
    worker = FakeWorker(reply="identified")
    emullm_api._connected_workers["identified-worker"] = worker  # noqa: SLF001
    emullm_api._native_worker_ids.add("identified-worker")  # noqa: SLF001

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "identified-worker/same",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert response.headers["X-EmuLLM-Worker-ID"] == "identified-worker"


def test_required_task_capability_prefers_declared_worker(
    client: TestClient,
) -> None:
    unknown = FakeWorker(reply="unknown")
    coder = FakeWorker(reply="coded")
    emullm_api._connected_workers.update(  # noqa: SLF001
        {"general-worker": unknown, "code-worker": coder}
    )
    emullm_api._native_worker_ids.update(emullm_api._connected_workers)  # noqa: SLF001
    emullm_api._worker_capabilities["code-worker"] = {"code": True}  # noqa: SLF001

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "some-code-model",
            "required_capabilities": ["code"],
            "messages": [{"role": "user", "content": "Write a function."}],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "coded"
    assert unknown.sent == []
    assert coder.sent[0]["required_capabilities"] == ["code"]


def test_worker_model_and_service_timing_are_aggregated(
    client: TestClient,
) -> None:
    worker = FakeWorker(reply="timed answer")
    emullm_api._connected_workers["timed-worker"] = worker  # noqa: SLF001

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "timed-worker/same",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    state = client.get("/emullm/admin/state").json()

    assert response.status_code == 200
    worker_stats = state["worker_service_stats"]["timed-worker"]
    assert worker_stats["services"]["chat"]["attempts"] == 1
    assert worker_stats["services"]["chat"]["served"] == 1
    assert worker_stats["services"]["chat"]["average_seconds"] >= 0
    model_stats = state["team_service_stats"]["models"]["timed-worker/same"]
    assert model_stats["totals"]["attempts"] == 1
    assert model_stats["totals"]["served"] == 1
    assert state["team_service_stats"]["services"]["chat"]["served"] == 1


def test_backend_fallback_delay_is_visible_as_waiting(
    monkeypatch,
) -> None:
    monkeypatch.setattr(emullm_api, "_backend_fallback_delay_seconds", 0.05)

    async def proxy(*_args, **_kwargs):
        return "backend answer"

    monkeypatch.setattr(emullm_api, "_proxy_chat", proxy)
    monkeypatch.setattr(
        emullm_api,
        "_all_backends",
        lambda: [{"name": "test", "base_url": "https://example.test/v1"}],
    )

    async def scenario() -> None:
        task = asyncio.create_task(
            emullm_api._relay_model_route_chain(  # noqa: SLF001
                "vendor/model",
                ["backend-test"],
                "hello",
                {"kind": "chat"},
            )
        )
        await asyncio.sleep(0.01)
        waiting = emullm_api._waiting_for_worker_snapshot()  # noqa: SLF001
        assert len(waiting) == 1
        assert waiting[0]["reason"] == "waiting before last-resort backend fallback"
        assert emullm_api.admin_state()["concurrency"]["waiting_for_worker"] == 1
        assert await task == "backend answer"
        assert emullm_api._waiting_for_worker_snapshot() == []  # noqa: SLF001
        state = emullm_api.admin_state()
        backend = state["worker_service_stats"]["backend-test"]
        assert backend["kind"] == "backend"
        assert backend["services"]["chat"]["served"] == 1
        assert state["team_service_stats"]["models"]["vendor/model"]["totals"][
            "served"
        ] == 1

    asyncio.run(scenario())


def test_configured_backend_is_visible_with_zero_statistics(monkeypatch) -> None:
    monkeypatch.setattr(
        emullm_api,
        "_all_backends",
        lambda: [{"name": "snet", "base_url": "https://example.test/v1"}],
    )
    state = emullm_api.admin_state()
    backend = state["worker_service_stats"]["backend-snet"]
    assert backend == {
        "kind": "backend",
        "active": 0,
        "reserved": 0,
        "services": {},
    }


@pytest.mark.parametrize(
    ("audio_data", "audio_format", "detail"),
    [
        ("not-valid!", "wav", "inline audio 1 is not valid base64"),
        (base64.b64encode(b"audio").decode("ascii"), "aac", "format must be one of"),
    ],
)
def test_chat_completions_rejects_invalid_input_audio(
    client: TestClient,
    audio_data: str,
    audio_format: str,
    detail: str,
) -> None:
    emullm_api._connected_workers["yourself"] = FakeWorker(reply="unused")  # noqa: SLF001
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "yourself/same",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_data,
                                "format": audio_format,
                            },
                        }
                    ],
                }
            ],
        },
    )
    assert response.status_code == 422
    assert detail in response.json()["detail"]


def test_completions_without_worker_waits_then_504(client: TestClient, short_timeout: None) -> None:
    response = client.post("/v1/completions", json={"model": "yourself/same", "prompt": "hi"})
    assert response.status_code == 504


def test_responses_without_worker_waits_then_504(client: TestClient, short_timeout: None) -> None:
    response = client.post("/v1/responses", json={"model": "yourself/same", "input": "hi"})
    assert response.status_code == 504


def test_messages_without_worker_waits_then_504(client: TestClient, short_timeout: None) -> None:
    response = client.post(
        "/v1/messages",
        json={"model": "yourself/same", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
    )
    assert response.status_code == 504


def test_messages_relay_anthropic_content_and_return_message_shape(client: TestClient) -> None:
    worker = FakeWorker(reply="relay reply")
    emullm_api._connected_workers["yourself"] = worker  # noqa: SLF001

    response = client.post(
        "/v1/messages",
        json={
            "model": "yourself/same",
            "system": [{"type": "text", "text": "Be concise."}],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image."},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": "YWJj"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "content": [{"type": "text", "text": "tool output"}]}],
                },
            ],
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200
    message = response.json()
    assert message["id"].startswith("msg-")
    assert message["type"] == "message"
    assert message["role"] == "assistant"
    assert message["content"] == [{"type": "text", "text": "relay reply"}]
    assert message["stop_reason"] == "end_turn"
    assert message["usage"]["input_tokens"] > 0
    assert message["usage"]["output_tokens"] == 2
    assert worker.sent[0]["prompt"] == (
        "[system] Be concise.\n\n[user] Describe this image.\n\n[user] tool output"
    )
    assert worker.sent[0]["images"] == [worker.sent[0]["attachments"][0]["url"]]
    assert worker.sent[0]["attachments"][0]["mime_type"] == "image/png"
    assert client.get(worker.sent[0]["attachments"][0]["url"]).content == b"abc"
    assert worker.sent[0]["kind"] == "vision"


def test_connected_worker_exports_durable_correlated_mailbox_events(client: TestClient, tmp_path) -> None:
    worker = FakeWorker(reply="mailbox reply")
    emullm_api._connected_workers["codex-ide-1"] = worker  # noqa: SLF001

    response = client.post(
        "/v1/messages",
        json={
            "model": "codex-ide-1/same",
            "messages": [{"role": "user", "content": "record this request"}],
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200
    directory = client.get("/ws_collab/v1/mailbox/mailboxes", params={"include_activity": "true"})
    assert directory.status_code == 200
    mailbox = next(entry for entry in directory.json()["mailboxes"] if entry["id"] == "codex-ide-1")
    assert mailbox["source"] == "jsonl"
    assert mailbox["storage"] == "events_logs"
    assert mailbox["transports"] == ["jsonl", "ws"]
    assert mailbox["connected"] is True
    assert mailbox["endpoints"]["ws"] == "/ws_collab/ws"
    assert mailbox["endpoints"]["worker_ws"] == "/emullm/ws?worker_id=codex-ide-1"

    events_response = client.get("/ws_collab/v1/events", params={"stream": "codex-ide-1"})
    assert events_response.status_code == 200
    events = events_response.json()["events"]
    assert [event["type"] for event in events] == ["LLM_REQUEST", "LLM_REPLY"]
    assert [event["seq"] for event in events] == [1, 2]
    assert events[0]["correlation_id"] == events[1]["correlation_id"]
    assert events[0]["data"]["prompt"] == "[user] record this request"
    assert events[1]["data"]["text"] == "mailbox reply"

    messages = client.get("/api/mailbox/messages", params={"mailbox": "codex-ide-1"}).json()["messages"]
    assert [message["text"] for message in messages] == ["[user] record this request", "mailbox reply"]
    assert (tmp_path / "runtime" / "config" / "mailboxes.json").is_file()
    assert (tmp_path / "runtime" / "events_logs" / "codex-ide-1.jsonl").is_file()

    capabilities = client.get("/mailbox_chat/v1/capabilities").json()
    assert capabilities["rest_base"] == "/ws_collab/v1"
    assert "codex-ide-1" in capabilities["streams"]
    assert capabilities["transports"] == ["jsonl", "ws"]


def test_worker_websocket_creates_its_named_mailbox(client: TestClient, tmp_path) -> None:
    with client.websocket_connect("/emullm/ws?worker_id=socket-servant") as websocket:
        assert websocket.receive_json() == {"type": "hello", "worker_id": "socket-servant"}
        directory = client.get("/ws_collab/v1/mailbox/mailboxes", params={"include_activity": "true"}).json()
        mailbox = next(entry for entry in directory["mailboxes"] if entry["id"] == "socket-servant")
        assert mailbox["writable"] is True
        assert mailbox["source"] == "jsonl"
        agents = client.get("/ws_collab/v1/mailbox/agents").json()["agents"]
        assert any(agent["id"] == "socket-servant" and agent["kind"] == "worker" for agent in agents)
        assert (tmp_path / "runtime" / "events_logs").is_dir()
        events = client.get("/ws_collab/v1/events", params={"stream": "socket-servant"}).json()["events"]
        assert events[0]["type"] == "WORKER_CONNECTED"


def test_worker_websocket_generates_an_identity_and_applies_model_masks(client: TestClient) -> None:
    with client.websocket_connect("/emullm/ws?modelmasks=vendor/*,gpt-*") as websocket:
        hello = websocket.receive_json()
        worker_id = hello["worker_id"]
        # An unidentified socket is named after its client ip/port so operators
        # can see where it came from (the in-process test client is testclient:50000).
        assert worker_id == "worker-unknown-testclient-50000"
        assert hello["modelmasks"] == ["vendor/*", "gpt-*"]
        assert emullm_api._worker_model_masks[worker_id] == ("vendor/*", "gpt-*")  # noqa: SLF001


def test_worker_websocket_without_model_masks_accepts_all_models(client: TestClient) -> None:
    with client.websocket_connect("/emullm/ws?worker_id=all-model-servant") as websocket:
        assert websocket.receive_json() == {"type": "hello", "worker_id": "all-model-servant"}
        assert "all-model-servant" not in emullm_api._worker_model_masks  # noqa: SLF001


def test_duplicate_worker_name_joins_as_fallback_team_member(client: TestClient) -> None:
    with client.websocket_connect("/emullm/ws?worker_id=unique-servant") as first:
        assert first.receive_json() == {"type": "hello", "worker_id": "unique-servant"}
        original = emullm_api._connected_workers["unique-servant"]  # noqa: SLF001
        with client.websocket_connect("/emullm/ws?worker_id=unique-servant") as duplicate:
            hello = duplicate.receive_json()
            # A conflicting name is admitted as a fallback under a derived id and
            # joins the same team instead of being shut down.
            assert hello["worker_id"] == "unique-servant-2"
            assert hello["team"] == "unique-servant"
            assert hello["fallback"] is True
            # The original primary keeps its identity and leads the team.
            assert emullm_api._connected_workers["unique-servant"] is original  # noqa: SLF001
            assert emullm_api._worker_teams["unique-servant"] == [  # noqa: SLF001
                "unique-servant",
                "unique-servant-2",
            ]
        assert emullm_api._connected_workers["unique-servant"] is original  # noqa: SLF001


def test_worker_can_rename_itself_at_any_time(client: TestClient) -> None:
    with client.websocket_connect("/emullm/ws?worker_id=nameless") as websocket:
        assert websocket.receive_json() == {"type": "hello", "worker_id": "nameless"}
        websocket.send_json({"type": "identify", "worker_id": "renamed-worker"})
        frame = websocket.receive_json()
        assert frame["type"] == "renamed"
        assert frame["worker_id"] == "renamed-worker"
        assert frame["team"] == "renamed-worker"
        assert frame["fallback"] is False
        assert "renamed-worker" in emullm_api._connected_workers  # noqa: SLF001
        assert "nameless" not in emullm_api._connected_workers  # noqa: SLF001
        assert "nameless" not in emullm_api._worker_teams  # noqa: SLF001


def test_worker_rename_conflict_becomes_team_fallback(client: TestClient) -> None:
    with client.websocket_connect("/emullm/ws?worker_id=leader") as leader:
        assert leader.receive_json() == {"type": "hello", "worker_id": "leader"}
        with client.websocket_connect("/emullm/ws?worker_id=follower") as follower:
            assert follower.receive_json() == {"type": "hello", "worker_id": "follower"}
            follower.send_json({"type": "identify", "worker_id": "leader"})
            frame = follower.receive_json()
            assert frame["type"] == "renamed"
            assert frame["worker_id"] == "leader-2"
            assert frame["team"] == "leader"
            assert frame["fallback"] is True
            assert emullm_api._worker_teams["leader"] == ["leader", "leader-2"]  # noqa: SLF001


def test_worker_in_name_route_reaches_the_whole_team(client: TestClient) -> None:
    with client.websocket_connect("/emullm/ws?worker_id=pool") as primary:
        assert primary.receive_json()["worker_id"] == "pool"
        with client.websocket_connect("/emullm/ws?worker_id=pool") as fallback:
            assert fallback.receive_json()["worker_id"] == "pool-2"
            # Addressing the team name reaches the primary first, then fallbacks.
            candidates = emullm_api._route_worker_candidates(  # noqa: SLF001
                "pool/gpt-5.6-sol", "worker-in-name"
            )
            assert candidates == ["pool", "pool-2"]


def test_websocket_inventory_tracks_worker_frame_counts(client: TestClient) -> None:
    with client.websocket_connect("/emullm/ws?worker_id=tracked-servant") as websocket:
        assert websocket.receive_json()["type"] == "hello"
        websocket.send_json(
            {
                "type": "register",
                "role": "test",
                "worker_kind": "headless-copilot",
                "runtime_model": "gpt-5.6-sol",
                "startup_prompt": "Act as a concise resident worker.",
                "description": "Resident Copilot test servant.",
            }
        )
        rows = []
        for _ in range(20):
            inventory = client.get("/emullm/admin/websockets").json()
            rows = [
                row
                for row in inventory["connections"]
                if row.get("worker_id") == "tracked-servant"
            ]
            if rows and rows[0]["messages_in"] >= 1:
                break
            time.sleep(0.01)
        assert inventory["count"] == 1
        row = rows[0]
        assert row["kind"] == "worker"
        assert row["endpoint"] == "/emullm/ws"
        assert row["messages_out"] >= 1
        assert row["messages_in"] >= 1
        assert row["connected_seconds"] >= 0
        assert row["last_satisfied_at"] is None
        assert row["last_satisfied_seconds"] is None
        assert row["last_client_work_at"] is None
        assert row["last_client_work_seconds"] is None
        assert row["log_url"] == "/emullm/admin/websockets/tracked-servant/log"
        assert row["log_bytes"] > 0
        assert row["log_limit_bytes"] == 6 * 1024 * 1024
        log_response = client.get(row["log_url"])
        assert log_response.status_code == 200
        log_records = [
            json.loads(line)
            for line in log_response.text.splitlines()
            if line
        ]
        assert log_records[0]["record_type"] == "worker_start_prompt"
        assert log_records[0]["worker_id"] == "tracked-servant"
        assert log_records[0]["source"] == "worker-registration"
        assert log_records[0]["from"] == "SYSTEM"
        assert log_records[0]["sender"] == "SYSTEM"
        assert "to" not in log_records[0]
        assert "recipient" not in log_records[0]
        assert log_records[0]["prompt"] == "Act as a concise resident worker."
        assert re.fullmatch(
            r"\d+\.\d{9}",
            log_records[0]["timestamp_epoch_decimal"],
        )
        assert re.fullmatch(
            r"\d+\.\d{9}",
            log_records[0]["precision_clock_decimal"],
        )
        assert isinstance(log_records[0]["precision_clock_ns"], int)
        assert log_records[1]["record_type"] == "segment_boundary"
        assert log_records[1]["segment"] == "first"
        assert {"lifecycle", "outbound", "inbound"}.issubset(
            {
                record["direction"]
                for record in log_records
                if "direction" in record
            }
        )
        outbound = next(
            record for record in log_records if record.get("direction") == "outbound"
        )
        inbound = next(
            record for record in log_records if record.get("direction") == "inbound"
        )
        assert outbound["from"] == outbound["sender"] == "EMULLM"
        assert inbound["from"] == inbound["sender"] == "tracked-servant"
        for record in (outbound, inbound):
            assert "to" not in record
            assert "recipient" not in record

        async def satisfy_request() -> dict:
            request_id = "satisfied-request"
            future = asyncio.get_running_loop().create_future()
            emullm_api._pending[request_id] = future  # noqa: SLF001
            emullm_api._pending_models[request_id] = "copilot/test"  # noqa: SLF001
            await emullm_api._handle_worker_message(  # noqa: SLF001
                "tracked-servant",
                {"type": "reply", "id": request_id, "content": "satisfied"},
            )
            return await future

        assert asyncio.run(satisfy_request()) == {"content": "satisfied"}
        satisfied_row = next(
            row
            for row in client.get("/emullm/admin/websockets").json()["connections"]
            if row.get("worker_id") == "tracked-servant"
        )
        assert satisfied_row["last_satisfied_at"]
        assert 0 <= satisfied_row["last_satisfied_seconds"] < 2
        assert satisfied_row["last_satisfied_kind"] == "client"
        assert satisfied_row["last_client_work_at"]
        assert 0 <= satisfied_row["last_client_work_seconds"] < 2
        assert "last_satisfied_at_epoch" not in satisfied_row
        assert "last_client_work_at_epoch" not in satisfied_row

        last_client_work_at = satisfied_row["last_client_work_at"]
        asyncio.run(
            emullm_api._handle_worker_message(  # noqa: SLF001
                "tracked-servant",
                {
                    "type": "keepalive_reply",
                    "id": "keepalive-1",
                    "prompt_index": 0,
                    "content": "stable",
                },
            )
        )
        keepalive_row = next(
            row
            for row in client.get("/emullm/admin/websockets").json()["connections"]
            if row.get("worker_id") == "tracked-servant"
        )
        assert keepalive_row["last_satisfied_kind"] == "keepalive"
        assert keepalive_row["last_client_work_at"] == last_client_work_at
        caps = client.get("/emullm/caps/tracked-servant").json()
        assert caps["worker_kind"] == "headless-copilot"
        assert caps["backing_model"] == "gpt-5.6-sol"
        assert caps["description"] == "Resident Copilot test servant."

    assert client.get("/emullm/admin/websockets").json()["connections"] == []
    disconnected_log = client.get(
        "/emullm/admin/websockets/tracked-servant/log"
    )
    assert json.loads(disconnected_log.text.splitlines()[-1])["frame"]["type"] == "disconnected"


def test_worker_socket_jsonl_preserves_first_and_rotates_two_tail_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(emullm_api, "_socket_worker_log_segment_bytes", 2_048)
    worker_id = "rotation-worker"
    connection_id = "ws-rotation"
    for index in range(20):
        emullm_api._append_socket_worker_log(  # noqa: SLF001
            worker_id,
            connection_id,
            "outbound",
            {
                "type": "request",
                "id": f"request-{index}",
                "prompt": f"{index}-" + ("x" * 700),
                "image_b64": "A" * 10_000,
            },
        )

    first, current, previous = emullm_api._socket_log_paths(worker_id)  # noqa: SLF001
    assert first.is_file()
    assert current.is_file()
    assert previous.is_file()
    assert first.stat().st_size <= 2_048
    assert current.stat().st_size <= 2_048
    assert previous.stat().st_size <= 2_048
    combined = first.read_bytes() + previous.read_bytes() + current.read_bytes()
    assert len(combined) <= 6_144
    records = [
        json.loads(line)
        for line in combined.decode("utf-8").splitlines()
        if line
    ]
    assert records[0]["frame"]["id"] == "request-0"
    assert records[-1]["frame"]["id"] == "request-19"
    assert records[-1]["frame"]["image_b64"] == {
        "omitted_base64_characters": 10_000
    }
    assert re.fullmatch(
        r"\d+\.\d{9}",
        records[-1]["timestamp_epoch_decimal"],
    )
    assert re.fullmatch(
        r"\d+\.\d{9}",
        records[-1]["precision_clock_decimal"],
    )


def test_worker_socket_log_viewer_uses_chat_bubbles_and_manual_scroll(
    client: TestClient,
    tmp_path,
) -> None:
    image = b"\x89PNG\r\n\x1a\nsocket-image"
    audio = b"RIFF\x00\x00\x00\x00WAVEsocket-audio"
    emullm_api._append_socket_worker_log(  # noqa: SLF001
        "viewer-worker",
        "ws-viewer",
        "outbound",
        {"type": "request", "id": "one", "prompt": "hello"},
    )
    emullm_api._append_socket_worker_log(  # noqa: SLF001
        "viewer-worker",
        "ws-viewer",
        "inbound",
        {
            "type": "reply",
            "id": "one",
            "content": "image",
            "image_b64": base64.b64encode(image).decode("ascii"),
            "mime": "image/png",
        },
    )
    emullm_api._append_socket_worker_log(  # noqa: SLF001
        "viewer-worker",
        "ws-viewer",
        "inbound",
        {
            "type": "reply",
            "id": "two",
            "content": "audio",
            "audio_b64": base64.b64encode(audio).decode("ascii"),
            "mime": "audio/wav",
        },
    )
    raw = client.get("/emullm/admin/websockets/viewer-worker/log")
    records = [
        json.loads(line) for line in raw.text.splitlines() if line
    ]
    media = [
        item
        for record in records
        for item in record.get("media", [])
    ]
    assert [item["kind"] for item in media] == ["image", "audio"]
    assert all(item["available"] is True for item in media)
    image_response = client.get(media[0]["url"])
    audio_response = client.get(media[1]["url"])
    assert image_response.content == image
    assert image_response.headers["content-type"].startswith("image/png")
    assert audio_response.content == audio
    assert audio_response.headers["content-type"].startswith("audio/")

    response = client.get(
        "/emullm/admin/websockets/viewer-worker/log/view"
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "max-width:80%" in response.text
    assert ".bubble.server" in response.text
    assert ".bubble.worker" in response.text
    assert "const speaker = 'from: ' + from" in response.text
    assert "Autoscroll: on" in response.text
    assert "['wheel','touchstart','pointerdown']" in response.text
    assert "setAutoscroll(false)" in response.text
    assert "messages.addEventListener('scroll'" not in response.text
    assert "data:image/svg+xml" in response.text
    assert "Raw JSONL" in response.text
    assert 'class="media"' in response.text
    assert '<img src="' in response.text
    assert "<audio controls preload=" in response.text
    assert "mediaContent(record)" in response.text
    scripts = re.findall(r"<script>(.*?)</script>", response.text, flags=re.DOTALL)
    script_path = tmp_path / "socket-log-viewer.js"
    script_path.write_text("\n".join(scripts), encoding="utf-8")
    result = subprocess.run(
        [shutil.which("node") or "node", "--check", str(script_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_unknown_worker_reply_does_not_mark_websocket_satisfied() -> None:
    connection_id = "ws-test"
    emullm_api._active_websockets[connection_id] = {  # noqa: SLF001
        "connection_id": connection_id,
        "kind": "worker",
        "endpoint": "/emullm/ws",
        "client": "testclient:50000",
        "connected_at": emullm_api._now_iso(),  # noqa: SLF001
        "connected_at_epoch": time.time(),
        "last_satisfied_at": None,
        "last_satisfied_at_epoch": None,
        "last_satisfied_kind": None,
        "last_client_work_at": None,
        "last_client_work_at_epoch": None,
        "messages_in": 1,
        "messages_out": 1,
        "worker_id": "tracked-servant",
    }
    emullm_api._worker_connection_ids["tracked-servant"] = connection_id  # noqa: SLF001

    asyncio.run(
        emullm_api._handle_worker_message(  # noqa: SLF001
            "tracked-servant",
            {"type": "reply", "id": "not-pending", "content": "stale"},
        )
    )

    row = emullm_api._active_websocket_rows()[0]  # noqa: SLF001
    assert row["last_satisfied_at"] is None
    assert row["last_satisfied_seconds"] is None
    assert row["last_client_work_at"] is None
    assert row["last_client_work_seconds"] is None


def test_replaced_worker_socket_cannot_update_new_connection_activity() -> None:
    now = time.time()
    for connection_id in ("ws-old", "ws-new"):
        emullm_api._active_websockets[connection_id] = {  # noqa: SLF001
            "connection_id": connection_id,
            "kind": "worker",
            "endpoint": "/emullm/ws",
            "client": "testclient:50000",
            "connected_at": emullm_api._now_iso(),  # noqa: SLF001
            "connected_at_epoch": now,
            "last_satisfied_at": None,
            "last_satisfied_at_epoch": None,
            "last_satisfied_kind": None,
            "last_client_work_at": None,
            "last_client_work_at_epoch": None,
            "messages_in": 0,
            "messages_out": 0,
            "worker_id": "reconnected-worker",
        }
    emullm_api._worker_connection_ids["reconnected-worker"] = "ws-new"  # noqa: SLF001

    asyncio.run(
        emullm_api._handle_worker_message(  # noqa: SLF001
            "reconnected-worker",
            {"type": "keepalive_reply", "id": "old-reply"},
            connection_id="ws-old",
        )
    )

    assert emullm_api._active_websockets["ws-old"]["last_satisfied_at"]  # noqa: SLF001
    assert emullm_api._active_websockets["ws-new"]["last_satisfied_at"] is None  # noqa: SLF001


def test_openai_http_client_inventory_tracks_logical_sessions(
    client: TestClient,
) -> None:
    headers = {
        "User-Agent": "symbolic-workbench-test/1.0",
        "X-EmuLLM-Client-ID": "video-import",
    }

    assert client.get("/v1/models", headers=headers).status_code == 200
    assert client.get("/v1/models/not-present", headers=headers).status_code == 200
    inventory = client.get("/emullm/admin/clients").json()

    assert inventory["count"] == 1
    assert inventory["active_count"] == 0
    assert inventory["active_requests"] == 0
    assert inventory["request_count"] == 2
    assert len(inventory["requests"]) == 2
    assert all(request["active"] is False for request in inventory["requests"])
    assert all(request["duration_seconds"] >= 0 for request in inventory["requests"])
    logical_client = inventory["clients"][0]
    assert logical_client["declared_id"] == "video-import"
    assert logical_client["host"] == "testclient"
    assert logical_client["user_agent"] == "symbolic-workbench-test/1.0"
    assert logical_client["requests"] == 2
    assert logical_client["connected"] is False
    assert logical_client["last_method"] == "GET"
    assert logical_client["last_endpoint"] == "/v1/models/not-present"
    assert logical_client["last_status"] == 200
    assert logical_client["first_seen_seconds"] >= 0
    assert logical_client["last_seen_seconds"] >= 0
    assert logical_client["last_completed_seconds"] >= 0
    assert not any(key.endswith("_epoch") for key in logical_client)

    assert client.get("/emullm/admin/clients").json()["clients"][0]["requests"] == 2


def test_openai_client_and_request_tracking_caps_are_hard(
    client: TestClient,
) -> None:
    now = time.time()
    emullm_api._openai_clients.update(  # noqa: SLF001
        {
            f"client-{index}": {
                "client_id": f"client-{index}",
                "active_requests": 1,
                "requests": 1,
                "last_seen_at_epoch": now,
            }
            for index in range(emullm_api._MAX_OPENAI_CLIENTS)  # noqa: SLF001
        }
    )
    response = client.get(
        "/v1/models",
        headers={"X-EmuLLM-Client-ID": "overflow-client"},
    )
    assert response.status_code == 200
    assert len(emullm_api._openai_clients) == emullm_api._MAX_OPENAI_CLIENTS  # noqa: SLF001

    emullm_api._openai_clients.clear()  # noqa: SLF001
    headers = {"X-EmuLLM-Client-ID": "known-client"}
    assert client.get("/v1/models", headers=headers).status_code == 200
    emullm_api._openai_requests.clear()  # noqa: SLF001
    emullm_api._openai_requests.update(  # noqa: SLF001
        {
            f"http-{index}": {
                "request_id": f"http-{index}",
                "active": True,
                "started_at_epoch": now,
            }
            for index in range(emullm_api._MAX_OPENAI_REQUESTS)  # noqa: SLF001
        }
    )
    assert client.get("/v1/models", headers=headers).status_code == 200
    assert len(emullm_api._openai_requests) == emullm_api._MAX_OPENAI_REQUESTS  # noqa: SLF001


def test_mailbox_chat_websocket_subscribes_and_publishes_events(client: TestClient) -> None:
    assert client.post("/mailbox/create", json={"id": "ws-servant"}).status_code == 200
    initial = client.post(
        "/events",
        json={
            "stream": "ws-servant",
            "type": "LLM_REQUEST",
            "data": {"text": "catch up"},
            "source_id": "llm-api",
        },
    )
    assert initial.status_code == 200

    with client.websocket_connect("/ws_collab/ws") as websocket:
        websocket.send_json({"type": "subscribe", "streams": ["ws-servant"], "cursors": {}})
        assert websocket.receive_json() == {"type": "subscribed", "streams": ["ws-servant"]}
        caught_up = websocket.receive_json()
        assert caught_up["type"] == "event"
        assert caught_up["event"]["type"] == "LLM_REQUEST"

        websocket.send_json(
            {
                "type": "publish",
                "event": {
                    "stream": "ws-servant",
                    "event_type": "LLM_REPLY",
                    "data": {"text": "live reply"},
                    "source_id": "ws-servant",
                    "source_kind": "worker",
                    "correlation_id": "ws-1",
                },
            }
        )
        published = websocket.receive_json()
        assert published["type"] == "published"
        live = websocket.receive_json()
        assert live["type"] == "event"
        assert live["event"]["type"] == "LLM_REPLY"
        assert live["event"]["correlation_id"] == "ws-1"


def test_mailbox_config_migrates_legacy_transport_metadata(client: TestClient, tmp_path) -> None:
    config_path = tmp_path / "runtime" / "config" / "mailboxes.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mailboxes": {
                    "legacy-servant": {
                        "id": "legacy-servant",
                        "name": "legacy-servant",
                        "source": "events_logs",
                        "transports": ["events_logs", "websocket"],
                    }
                },
                "agents": {},
                "cursors": {},
            }
        ),
        encoding="utf-8",
    )

    directory = client.get("/mailbox/mailboxes", params={"include_activity": "true"}).json()
    mailbox = directory["mailboxes"][0]
    assert mailbox["source"] == "jsonl"
    assert mailbox["transports"] == ["jsonl", "ws"]
    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == 2
    assert persisted["mailboxes"]["legacy-servant"]["source"] == "jsonl"


def test_mailbox_chat_contract_supports_messages_events_and_cursors(client: TestClient) -> None:
    create = client.post(
        "/mailbox_chat/v1/mailbox/create",
        json={"id": "manual-servant", "purpose": "test worker mailbox"},
    )
    assert create.status_code == 200
    assert create.json()["created"] is True

    payload = {
        "to": "manual-servant",
        "text": "hello servant",
        "sender": "operator",
        "correlation_id": "chat-1",
    }
    first = client.post(
        "/mailbox_chat/v1/mailbox/send",
        json=payload,
        headers={"Idempotency-Key": "send-chat-1"},
    )
    assert first.status_code == 200
    assert first.json()["message"]["type"] == "CONVERSATION_MESSAGE"
    duplicate = client.post(
        "/mailbox_chat/v1/mailbox/send",
        json=payload,
        headers={"Idempotency-Key": "send-chat-1"},
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["message"]["id"] == first.json()["message"]["id"]

    posted = client.post(
        "/api/events",
        json={
            "stream": "manual-servant",
            "type": "LLM_REQUEST",
            "data": {"text": "typed request", "from": "llm-api", "to": "manual-servant"},
            "source_id": "llm-api",
            "correlation_id": "request-1",
        },
    )
    assert posted.status_code == 200
    assert posted.json()["seq"] == 2

    events = client.get("/ws_collab/v1/events", params={"stream": "manual-servant"}).json()
    assert len(events["events"]) == 2
    assert events["events"][0]["type"] == "CONVERSATION_MESSAGE"
    assert events["events"][1]["correlation_id"] == "request-1"
    assert client.get(
        "/api/mailbox/messages",
        params={"mailbox": "manual-servant", "from": "operator"},
    ).json()["messages"][0]["text"] == "hello servant"
    assert len(client.get("/mailbox_chat/v1/streams/manual-servant/tail", params={"count": 2}).json()["events"]) == 2

    initial_cursor = client.get(
        "/ws_collab/v1/mailbox/cursor",
        params={"mailbox": "manual-servant", "agent": "operator"},
    ).json()
    assert initial_cursor["initialized"] is False
    beginning = client.post(
        "/ws_collab/v1/mailbox/cursor",
        json={"mailbox": "manual-servant", "agent": "operator", "start": "beginning"},
    ).json()
    assert beginning["initialized"] is True
    assert beginning["offset"] == 0
    now = client.post(
        "/ws_collab/v1/mailbox/cursor",
        json={"mailbox": "manual-servant", "agent": "operator", "start": "now"},
    ).json()
    assert now["offset"] == 2
    cleared = client.delete(
        "/ws_collab/v1/mailbox/cursor",
        params={"mailbox": "manual-servant", "agent": "operator"},
    ).json()
    assert cleared["initialized"] is False


@pytest.mark.parametrize(
    ("path", "body", "expected_marker"),
    [
        (
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hello"}], "stream": True},
            "chat.completion.chunk",
        ),
        ("/v1/completions", {"prompt": "hello", "stream": True}, "text_completion"),
        ("/v1/responses", {"input": "hello", "stream": True}, "response.output_text.delta"),
        (
            "/v1/messages",
            {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 8, "stream": True},
            "event: message_start",
        ),
    ],
)
def test_text_endpoints_support_server_sent_event_streams(
    client: TestClient,
    path: str,
    body: dict[str, object],
    expected_marker: str,
) -> None:
    emullm_api._connected_workers["worker-copilot-1"] = FakeWorker(  # noqa: SLF001
        reply="streamed answer"
    )

    response = client.post(path, json=body)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert expected_marker in response.text
    assert "streamed answer" in response.text


@pytest.mark.parametrize("model", ["unlisted-model", "vendor/unlisted-model", "yourself/unlisted-model"])
def test_unknown_models_are_forwarded_unchanged_to_generic_servant(client: TestClient, model: str) -> None:
    worker = FakeWorker(reply="generic servant reply")
    emullm_api._connected_workers["yourself"] = worker  # noqa: SLF001

    response = client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    assert response.json()["model"] == model
    assert response.json()["choices"][0]["message"]["content"] == "generic servant reply"
    assert worker.sent[0]["model"] == model
    assert worker.sent[0]["worker_id"] == "yourself"
    assert "persona_instruction" not in worker.sent[0]


def test_messages_unknown_model_is_forwarded_unchanged_to_generic_servant(client: TestClient) -> None:
    worker = FakeWorker(reply="generic message reply")
    emullm_api._connected_workers["yourself"] = worker  # noqa: SLF001

    response = client.post(
        "/v1/messages",
        json={"model": "vendor/unlisted-model", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 8},
    )

    assert response.status_code == 200
    assert response.json()["model"] == "vendor/unlisted-model"
    assert response.json()["content"][0]["text"] == "generic message reply"
    assert worker.sent[0]["model"] == "vendor/unlisted-model"
    assert worker.sent[0]["worker_id"] == "yourself"
    assert "persona_instruction" not in worker.sent[0]


def test_mock_mode_answers_without_a_worker(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(emullm_api, "_SERVER_MODE", "mock")
    response = client.post(
        "/v1/chat/completions",
        json={"model": "yourself/same", "messages": [{"role": "user", "content": "hi there"}]},
    )
    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    # default deterministic echo; chat flattens messages to "[role] text"
    assert content == "mock: [user] hi there"


def test_mock_mode_fixed_reply_via_env(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(emullm_api, "_SERVER_MODE", "mock")
    monkeypatch.setenv("EMULLM_MOCK_REPLY", "canned test answer")
    response = client.post(
        "/v1/chat/completions",
        json={"model": "yourself/same", "messages": [{"role": "user", "content": "anything"}]},
    )
    assert response.json()["choices"][0]["message"]["content"] == "canned test answer"


def test_mock_mode_template_from_config(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(emullm_api, "_SERVER_MODE", "mock")
    emullm_api._CONFIG_PATH.write_text(  # noqa: SLF001
        json.dumps({"mock": {"template": "[{model}] echo={prompt}"}}), encoding="utf-8"
    )
    response = client.post(
        "/v1/chat/completions",
        json={"model": "yourself/same", "messages": [{"role": "user", "content": "ping"}]},
    )
    assert response.json()["choices"][0]["message"]["content"] == "[yourself/same] echo=[user] ping"


def test_error_when_empty_mode_fails_fast(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(emullm_api, "_SERVER_MODE", "error-when-empty")
    response = client.post(
        "/v1/chat/completions",
        json={"model": "yourself/same", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 503


def test_error_when_empty_mode_relays_when_worker_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(emullm_api, "_SERVER_MODE", "error-when-empty")
    emullm_api._connected_workers["yourself"] = FakeWorker(reply="hi back")  # noqa: SLF001
    assert asyncio.run(emullm_api._relay("yourself/same", "hi")) == "hi back"


def test_proxy_mode_forwards_to_backend(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(emullm_api, "_SERVER_MODE", "proxy")
    emullm_api._CONFIG_PATH.write_text(  # noqa: SLF001
        json.dumps({"backends": [{"name": "x", "base_url": "http://backend.test/v1", "model": "gpt-x"}]}),
        encoding="utf-8",
    )
    calls: dict = {}

    def fake_post(url, headers, payload, timeout=60.0):
        calls["url"] = url
        calls["payload"] = payload
        return {"choices": [{"message": {"role": "assistant", "content": "backend says hi"}}]}

    monkeypatch.setattr(emullm_api, "_http_post_json", fake_post)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "yourself/same", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "backend says hi"
    assert calls["url"] == "http://backend.test/v1/chat/completions"
    assert calls["payload"]["model"] == "gpt-x"  # backend model overrides the client's model id


def test_direct_backend_model_routes_to_named_backend(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model named ``backend-<name>/<served>`` hits that backend directly and
    forwards <served> upstream -- no proxy mode and no configured route needed."""
    emullm_api._CONFIG_PATH.write_text(  # noqa: SLF001
        json.dumps(
            {
                "backends": [
                    {"name": "snet", "base_url": "http://snet.test/v1"},
                    {"name": "other", "base_url": "http://other.test/v1"},
                ]
            }
        ),
        encoding="utf-8",
    )
    calls: dict = {}

    def fake_post(url, headers, payload, timeout=60.0):
        calls["url"] = url
        calls["payload"] = payload
        return {"choices": [{"message": {"role": "assistant", "content": "snet says hi"}}]}

    monkeypatch.setattr(emullm_api, "_http_post_json", fake_post)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "backend-snet/gpt-4o", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "snet says hi"
    assert calls["url"] == "http://snet.test/v1/chat/completions"
    assert calls["payload"]["model"] == "gpt-4o"  # the served suffix is forwarded upstream


def test_direct_backend_model_forwards_multi_slash_model_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the first slash selects the backend; the rest is the served model id,
    so ``backend-snet/openai/foo`` and ``backend-snet/other/foo`` disambiguate."""
    emullm_api._CONFIG_PATH.write_text(  # noqa: SLF001
        json.dumps({"backends": [{"name": "snet", "base_url": "http://snet.test/v1"}]}),
        encoding="utf-8",
    )
    seen: list[str] = []

    def fake_post(url, headers, payload, timeout=60.0):
        seen.append(payload["model"])
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(emullm_api, "_http_post_json", fake_post)

    for served in ("openai/foo", "other/foo"):
        response = client.post(
            "/v1/chat/completions",
            json={"model": f"backend-snet/{served}", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 200
    assert seen == ["openai/foo", "other/foo"]


def test_direct_backend_model_404_when_backend_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Addressing an unknown backend name fails clearly rather than silently
    falling back to a worker/default. Nothing upstream errored -- the address
    just names no configured backend -- so it is a 404 (not found)."""
    emullm_api._CONFIG_PATH.write_text(  # noqa: SLF001
        json.dumps({"backends": [{"name": "snet", "base_url": "http://snet.test/v1"}]}),
        encoding="utf-8",
    )
    response = client.post(
        "/v1/chat/completions",
        json={"model": "backend-nope/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404


def test_direct_backend_model_slashless_dash_form(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slashless ``backend-<name>-<served>`` model splits at the configured
    backend name, forwarding the remaining dash-joined id upstream."""
    emullm_api._CONFIG_PATH.write_text(  # noqa: SLF001
        json.dumps({"backends": [{"name": "snet", "base_url": "http://snet.test/v1"}]}),
        encoding="utf-8",
    )
    calls: dict = {}

    def fake_post(url, headers, payload, timeout=60.0):
        calls["url"] = url
        calls["payload"] = payload
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(emullm_api, "_http_post_json", fake_post)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "backend-snet-other-foo", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert calls["url"] == "http://snet.test/v1/chat/completions"
    assert calls["payload"]["model"] == "other-foo"


def test_direct_backend_model_slashless_prefers_longest_name(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When two backend names share a prefix, the slashless form matches the
    longest configured name, so 'snet-other' wins over 'snet'."""
    emullm_api._CONFIG_PATH.write_text(  # noqa: SLF001
        json.dumps(
            {
                "backends": [
                    {"name": "snet", "base_url": "http://snet.test/v1"},
                    {"name": "snet-other", "base_url": "http://snet-other.test/v1"},
                ]
            }
        ),
        encoding="utf-8",
    )
    calls: dict = {}

    def fake_post(url, headers, payload, timeout=60.0):
        calls["url"] = url
        calls["payload"] = payload
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(emullm_api, "_http_post_json", fake_post)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "backend-snet-other-foo", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert calls["url"] == "http://snet-other.test/v1/chat/completions"
    assert calls["payload"]["model"] == "foo"


def test_direct_backend_model_dashed_name_with_slashed_served(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dash-joined backend name combined with a slash-separated served id is
    parsed uniformly: the name is stripped, the post-slash tail is forwarded."""
    emullm_api._CONFIG_PATH.write_text(  # noqa: SLF001
        json.dumps(
            {
                "backends": [
                    {"name": "snet", "base_url": "http://snet.test/v1"},
                    {"name": "snet-other", "base_url": "http://snet-other.test/v1"},
                ]
            }
        ),
        encoding="utf-8",
    )
    calls: dict = {}

    def fake_post(url, headers, payload, timeout=60.0):
        calls["url"] = url
        calls["payload"] = payload
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(emullm_api, "_http_post_json", fake_post)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "backend-snet-other/openai/foo", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert calls["url"] == "http://snet-other.test/v1/chat/completions"
    assert calls["payload"]["model"] == "openai/foo"


def test_direct_backend_model_slash_is_opaque_boundary(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Slashes are opaque: the backend/agent name is scanned only within the
    segment before the first '/', and the marker must be dash-joined
    ('backend-<name>'). So 'backend/snet-openai-foo' never resolves to backend
    'snet' -- the slash prevents merging 'backend' + 'snet' into 'backend-snet'."""
    emullm_api._CONFIG_PATH.write_text(  # noqa: SLF001
        json.dumps({"backends": [{"name": "snet", "base_url": "http://snet.test/v1"}]}),
        encoding="utf-8",
    )
    # Dash-joined marker: name is found and stripped from the pre-slash segment.
    assert (
        emullm_api._normalize_direct_backend_model("backend-snet-openai-foo")  # noqa: SLF001
        == "backend-snet/openai-foo"
    )
    # A slash right after "backend" is an opaque boundary; we never see
    # "backend-snet", so the address is left unchanged (-> unresolved downstream).
    assert (
        emullm_api._normalize_direct_backend_model("backend/snet-openai-foo")  # noqa: SLF001
        == "backend/snet-openai-foo"
    )
    # A slash after the name keeps the tail verbatim; the name is not scanned
    # across the slash.
    assert (
        emullm_api._normalize_direct_backend_model("backend-snet/openai/foo")  # noqa: SLF001
        == "backend-snet/openai/foo"
    )


def test_direct_backend_model_token_after_provider_prefix(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 'backend-<name>' token embedded after a provider prefix is stripped,
    the prefix is preserved across the slash, and the request routes to the
    named backend: 'openai/backend-snet-asi1' -> backend snet, served
    'openai/asi1'."""
    emullm_api._CONFIG_PATH.write_text(  # noqa: SLF001
        json.dumps({"backends": [{"name": "snet", "base_url": "http://snet.test/v1"}]}),
        encoding="utf-8",
    )
    calls: dict = {}

    def fake_post(url, headers, payload, timeout=60.0):
        calls["url"] = url
        calls["payload"] = payload
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(emullm_api, "_http_post_json", fake_post)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "openai/backend-snet-asi1", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert calls["url"] == "http://snet.test/v1/chat/completions"
    assert calls["payload"]["model"] == "openai/asi1"
    # The public model id the caller asked for is echoed back unchanged.
    assert response.json()["model"] == "openai/backend-snet-asi1"


# ---------------------------------------------------------------------------
# Resolution-order DSL: error directives, server tail, default route,
# named-route steps, route-list references, and single-visit de-duplication.
# ---------------------------------------------------------------------------


def _configure_routes(config: dict) -> None:
    """Write a config to the isolated config path (so _all_backends() sees the
    backends/proxy-agents) and apply it (so the routing globals populate)."""
    emullm_api._CONFIG_PATH.write_text(json.dumps(config), encoding="utf-8")  # noqa: SLF001
    emullm_api.apply_agent_policies(config)  # noqa: SLF001


def _stub_backends_by_status(
    monkeypatch: pytest.MonkeyPatch,
    statuses: dict[str, int],
    replies: dict[str, str] | None = None,
) -> list[str]:
    """Make each backend name deterministically raise ``HTTPException(<status>)``
    or return a canned reply, and record the order backends are actually tried.

    Patching ``_proxy_chat_with_stats`` (rather than ``_http_post_json``) is what
    lets a test produce a *specific* 4xx/5xx: ``_proxy_chat`` otherwise wraps any
    upstream failure as a 502, which would defeat error-class directives."""
    replies = replies or {}
    attempted: list[str] = []

    async def fake_pcs(backend, model, prompt_text, instruction, service_kind):
        name = str(backend.get("name") or "")
        attempted.append(name)
        if name in statuses:
            raise HTTPException(status_code=statuses[name], detail=f"{name} {statuses[name]}")
        return replies.get(name, f"{name} reply")

    monkeypatch.setattr(emullm_api, "_proxy_chat_with_stats", fake_pcs)  # noqa: SLF001
    return attempted


def test_error_directive_match_unit() -> None:
    m = emullm_api._error_directive_match  # noqa: SLF001
    assert m("backend-x", 500) is None  # not a directive
    assert m("worker-copilot-*", 404) is None
    assert m("error", None) is False  # nothing has failed yet
    assert m("error", 500) is True  # bare error == any prior failure
    assert m("error_any", 400) is True
    assert m("error_4xx", 404) is True
    assert m("error_4xx", 503) is False  # wrong class
    assert m("error_5xx", 502) is True
    assert m("error_502", 502) is True
    assert m("error_502", 503) is False  # exact code only
    assert m("ERROR-5XX", 500) is True  # case- and separator-insensitive


def test_named_route_scope_unit() -> None:
    s = emullm_api._named_route_scope  # noqa: SLF001
    assert s("named_agent_route") == "agent"
    assert s("named-agent-route") == "agent"
    assert s("NAMED_BACKEND_ROUTE") == "backend"
    assert s("named_agent_or_backend_route") is None  # combined token removed
    assert s("backend-snet") is None
    assert s("worker-copilot-*") is None


def test_router_error_target_unit() -> None:
    t = emullm_api._router_error_target  # noqa: SLF001
    assert t("backend-x") is None  # not a synthetic responder
    assert t("error_404") is None  # a directive, not router/error
    assert t("router/error_404") == 404
    assert t("router/error_502") == 502
    assert t("router/error_4xx") == 400  # class -> representative status
    assert t("router/error_5xx") == 500
    assert t("router/error") == 500  # bare -> 500
    assert t("router/error_any") == 500
    assert t("ROUTER-ERROR-404") == 404  # case- and separator-insensitive
    assert t("router_error_503") == 503


def test_expand_route_references_inlines_named_lists() -> None:
    emullm_api._model_routes.clear()  # noqa: SLF001
    emullm_api._model_routes.update(  # noqa: SLF001
        {"leaf": ["backend-a", "backend-b"], "root": ["backend-x", "leaf"]}
    )
    assert emullm_api._expand_route_references(["root"]) == [  # noqa: SLF001
        "backend-x",
        "backend-a",
        "backend-b",
    ]


def test_expand_route_references_mutual_cycle_is_blank() -> None:
    emullm_api._model_routes.clear()  # noqa: SLF001
    emullm_api._model_routes.update({"a": ["b"], "b": ["a"]})  # noqa: SLF001
    # Two lists pointing only at each other flatten to nothing.
    assert emullm_api._expand_route_references(["a"]) == []  # noqa: SLF001
    assert emullm_api._expand_route_references(["b"]) == []  # noqa: SLF001


def test_expand_route_references_cycle_keeps_the_one_real_place() -> None:
    emullm_api._model_routes.clear()  # noqa: SLF001
    # a -> b, b -> [backend-x, a]: the a<->b cycle resolves to that one real
    # place from either entry point (the back-reference is dropped).
    emullm_api._model_routes.update({"a": ["b"], "b": ["backend-x", "a"]})  # noqa: SLF001
    assert emullm_api._expand_route_references(["a"]) == ["backend-x"]  # noqa: SLF001
    assert emullm_api._expand_route_references(["b"]) == ["backend-x"]  # noqa: SLF001


def test_error_directive_4xx_stops_chain_before_backup(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_routes(
        {
            "backends": [
                {"name": "primary", "base_url": "http://primary.test/v1"},
                {"name": "backup", "base_url": "http://backup.test/v1"},
            ],
            "model_routes": {"m": ["backend-primary", "error_4xx", "backend-backup"]},
        }
    )
    attempted = _stub_backends_by_status(monkeypatch, {"primary": 404})
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404
    assert attempted == ["primary"]  # backup never tried on a client error


def test_error_directive_4xx_falls_through_on_5xx(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_routes(
        {
            "backends": [
                {"name": "primary", "base_url": "http://primary.test/v1"},
                {"name": "backup", "base_url": "http://backup.test/v1"},
            ],
            "model_routes": {"m": ["backend-primary", "error_4xx", "backend-backup"]},
        }
    )
    attempted = _stub_backends_by_status(
        monkeypatch, {"primary": 503}, replies={"backup": "backup ok"}
    )
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "backup ok"
    assert attempted == ["primary", "backup"]


def test_bare_error_directive_propagates_last_status(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_routes(
        {
            "backends": [{"name": "primary", "base_url": "http://primary.test/v1"}],
            "model_routes": {"m": ["backend-primary", "error"]},
        }
    )
    _stub_backends_by_status(monkeypatch, {"primary": 429})
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 429


def test_server_routes_tail_serves_after_model_chain(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_routes(
        {
            "backends": [
                {"name": "primary", "base_url": "http://primary.test/v1"},
                {"name": "tail", "base_url": "http://tail.test/v1"},
            ],
            "model_routes": {"m": ["backend-primary"]},
            "server_routes": ["backend-tail"],
        }
    )
    attempted = _stub_backends_by_status(
        monkeypatch, {"primary": 500}, replies={"tail": "tail ok"}
    )
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "tail ok"
    assert attempted == ["primary", "tail"]


def test_error_directive_stops_before_server_tail(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_routes(
        {
            "backends": [
                {"name": "primary", "base_url": "http://primary.test/v1"},
                {"name": "tail", "base_url": "http://tail.test/v1"},
            ],
            "model_routes": {"m": ["backend-primary", "error_5xx"]},
            "server_routes": ["backend-tail"],
        }
    )
    attempted = _stub_backends_by_status(monkeypatch, {"primary": 502})
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 502
    assert attempted == ["primary"]  # server tail not reached


def test_default_model_route_used_for_unspecified_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_routes(
        {
            "backends": [
                {"name": "primary", "base_url": "http://primary.test/v1"},
                {"name": "tail", "base_url": "http://tail.test/v1"},
            ],
            "default_model_route": ["backend-primary"],
            "server_routes": ["backend-tail"],
        }
    )
    attempted = _stub_backends_by_status(
        monkeypatch, {"primary": 500}, replies={"tail": "tail ok"}
    )
    response = client.post(
        "/v1/chat/completions",
        json={"model": "unrouted-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "tail ok"
    assert attempted == ["primary", "tail"]


def test_route_list_reference_is_expanded_inline(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Model "m" references the named list "pool"; the chain flattens to
    # [backend-a, backend-b] and fails over from a to b.
    _configure_routes(
        {
            "backends": [
                {"name": "a", "base_url": "http://a.test/v1"},
                {"name": "b", "base_url": "http://b.test/v1"},
            ],
            "model_routes": {"m": ["backend-a", "pool"], "pool": ["backend-b"]},
        }
    )
    attempted = _stub_backends_by_status(monkeypatch, {"a": 500}, replies={"b": "b ok"})
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "b ok"
    assert attempted == ["a", "b"]


def test_repeated_place_is_visited_only_once(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # backend-a appears in the model chain and again in the server tail; the
    # second occurrence is skipped rather than retried.
    _configure_routes(
        {
            "backends": [
                {"name": "a", "base_url": "http://a.test/v1"},
                {"name": "b", "base_url": "http://b.test/v1"},
            ],
            "model_routes": {"m": ["backend-a"]},
            "server_routes": ["backend-a", "backend-b"],
        }
    )
    attempted = _stub_backends_by_status(monkeypatch, {"a": 500}, replies={"b": "b ok"})
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "b ok"
    assert attempted == ["a", "b"]  # only one attempt at backend "a"


def test_chain_exhaustion_propagates_upstream_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No error directive, but the only place actually returned a 502; when the
    # order is exhausted that real upstream error is propagated (not masked as
    # a 404).
    _configure_routes(
        {
            "backends": [{"name": "primary", "base_url": "http://primary.test/v1"}],
            "model_routes": {"m": ["backend-primary"]},
        }
    )
    _stub_backends_by_status(monkeypatch, {"primary": 502})
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 502


def test_chain_exhaustion_404_when_nothing_matched(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The route names a backend that is not configured, so nothing is ever tried
    # and nothing errors upstream -- the model could not be routed anywhere, a 404.
    _configure_routes(
        {
            "backends": [{"name": "real", "base_url": "http://real.test/v1"}],
            "model_routes": {"m": ["backend-ghost"]},
        }
    )
    monkeypatch.setattr(
        emullm_api,
        "_http_post_json",
        lambda *a, **k: pytest.fail("no backend should be contacted"),  # noqa: SLF001
    )
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404


def test_router_error_target_returns_synthetic_status(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A route pointed at a synthetic error responder reliably answers that
    # status without any backend being contacted.
    _configure_routes({"model_routes": {"probe": ["router/error_404"]}})
    monkeypatch.setattr(
        emullm_api,
        "_http_post_json",
        lambda *a, **k: pytest.fail("no backend should be contacted"),  # noqa: SLF001
    )
    response = client.post(
        "/v1/chat/completions",
        json={"model": "probe", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404


def test_router_error_target_falls_through_to_real_backend(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Without an error directive, the synthetic error is just a failed place; the
    # chain still fails over to a working backend after it.
    _configure_routes(
        {
            "backends": [{"name": "real", "base_url": "http://real.test/v1"}],
            "model_routes": {"m": ["router/error_500", "backend-real"]},
        }
    )
    _stub_backends_by_status(monkeypatch, {}, {"real": "real ok"})
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "real ok"


def test_router_error_target_stops_at_matching_directive(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The synthetic error records a real upstream status, so a following error
    # directive can act on it: a 4xx is caught and propagated, never reaching the
    # backup.
    _configure_routes(
        {
            "backends": [{"name": "backup", "base_url": "http://backup.test/v1"}],
            "model_routes": {"m": ["router/error_404", "error_4xx", "backend-backup"]},
        }
    )
    monkeypatch.setattr(
        emullm_api,
        "_http_post_json",
        lambda *a, **k: pytest.fail("backup must not be contacted"),  # noqa: SLF001
    )
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404


def test_named_agent_route_resolves_proxy_agent(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_routes(
        {
            "agents": [
                {"id": "snet", "name": "snet", "launch": "proxy", "base_url": "http://snet.test/v1"}
            ],
            "backends": [{"name": "vend", "base_url": "http://vend.test/v1"}],
            "model_routes": {"openai/backend-snet-asi1": ["named_agent_route"]},
        }
    )
    calls: dict = {}

    def fake_post(url, headers, payload, timeout=60.0):
        calls["url"] = url
        calls["payload"] = payload
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(emullm_api, "_http_post_json", fake_post)  # noqa: SLF001
    response = client.post(
        "/v1/chat/completions",
        json={"model": "openai/backend-snet-asi1", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert calls["url"] == "http://snet.test/v1/chat/completions"
    assert calls["payload"]["model"] == "openai/asi1"


def test_named_backend_route_skips_agent_scope(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # snet is a proxy *agent*, so a backend-scoped step does not resolve it; the
    # step is a no-op and, with nothing else in the chain and nothing upstream
    # erroring, the request is a 404 (nothing could serve the model).
    _configure_routes(
        {
            "agents": [
                {"id": "snet", "name": "snet", "launch": "proxy", "base_url": "http://snet.test/v1"}
            ],
            "model_routes": {"y/backend-snet-asi1": ["named_backend_route"]},
        }
    )
    monkeypatch.setattr(
        emullm_api,
        "_http_post_json",
        lambda *a, **k: pytest.fail("no backend should be contacted"),  # noqa: SLF001
    )
    response = client.post(
        "/v1/chat/completions",
        json={"model": "y/backend-snet-asi1", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404


def test_named_backend_route_resolves_plain_backend(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_routes(
        {
            "agents": [
                {"id": "snet", "name": "snet", "launch": "proxy", "base_url": "http://snet.test/v1"}
            ],
            "backends": [{"name": "vend", "base_url": "http://vend.test/v1"}],
            "model_routes": {"x/backend-vend-foo": ["named_backend_route"]},
        }
    )
    calls: dict = {}

    def fake_post(url, headers, payload, timeout=60.0):
        calls["url"] = url
        calls["payload"] = payload
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(emullm_api, "_http_post_json", fake_post)  # noqa: SLF001
    response = client.post(
        "/v1/chat/completions",
        json={"model": "x/backend-vend-foo", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert calls["url"] == "http://vend.test/v1/chat/completions"
    assert calls["payload"]["model"] == "x/foo"


def test_proxy_mode_502_without_backend(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(emullm_api, "_SERVER_MODE", "proxy")
    response = client.post(
        "/v1/chat/completions",
        json={"model": "yourself/same", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 502


def test_proxy_forwards_model_and_persona_instruction(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(emullm_api, "_SERVER_MODE", "proxy")
    monkeypatch.setattr(
        emullm_api, "_select_backend", lambda: {"name": "b", "base_url": "http://b/v1", "model": "backend-default"}
    )
    captured = {}

    def fake_post(url, headers, payload, timeout=60.0):
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(emullm_api, "_http_post_json", fake_post)
    # an unknown (real backend) model id is accepted in proxy mode and forwarded as-is
    r = client.post(
        "/v1/chat/completions", json={"model": "real/model-x", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert r.status_code == 200
    assert captured["payload"]["model"] == "real/model-x"
    assert captured["payload"]["messages"][0]["role"] == "user"  # no persona -> no system msg

    # a persona dial (percentNN) -> backend default model + a system instruction
    client.post(
        "/v1/chat/completions", json={"model": "smart/percent10", "messages": [{"role": "user", "content": "hi"}]}
    )
    payload = captured["payload"]
    assert payload["model"] == "backend-default"
    assert payload["messages"][0]["role"] == "system"
    assert "10%" in payload["messages"][0]["content"]


def test_proxy_observe_mirrors_exchange_to_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(emullm_api, "_SERVER_MODE", "proxy-observe")
    monkeypatch.setenv("EMULLM_PROXY_BASE_URL", "http://backend.test/v1")
    monkeypatch.setattr(
        emullm_api,
        "_http_post_json",
        lambda *a, **k: {"choices": [{"message": {"content": "real answer"}}]},
    )

    class ObserverWorker:
        def __init__(self) -> None:
            self.sent: list = []

        async def send_json(self, payload) -> None:
            self.sent.append(payload)

    obs = ObserverWorker()
    emullm_api._connected_workers["yourself"] = obs  # noqa: SLF001

    reply = asyncio.run(emullm_api._relay("yourself/same", "hello"))

    assert reply == "real answer"
    assert obs.sent and obs.sent[0]["type"] == "observe"
    assert obs.sent[0]["reply"] == "real answer"
    assert obs.sent[0]["prompt"] == "hello"


def test_mock_workers_simulate_a_set_of_copilots(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(emullm_api, "_SERVER_MODE", "mock")
    emullm_api.register_mock_workers(  # noqa: SLF001
        [
            {"id": "alice", "reply": "hi from alice", "capabilities": ["images"], "role": "trusted"},
            {"id": "bob", "template": "[bob] {prompt}"},
        ]
    )

    # both pretend copilots show up in /v1/models
    ids = {m["id"] for m in client.get("/v1/models").json()["data"]}
    assert {
        "alice/percent125",
        "alice/percent100",
        "alice/percent25",
        "bob/percent125",
        "bob/percent100",
        "bob/percent25",
    }.issubset(ids)
    assert "alice/same" not in ids
    assert "bob/same" not in ids

    # each is routed to independently
    a = client.post(
        "/v1/chat/completions",
        json={"model": "alice/same", "messages": [{"role": "user", "content": "hey"}]},
    )
    assert a.json()["choices"][0]["message"]["content"] == "hi from alice"
    b = client.post(
        "/v1/chat/completions",
        json={"model": "bob/same", "messages": [{"role": "user", "content": "yo"}]},
    )
    assert b.json()["choices"][0]["message"]["content"] == "[bob] [user] yo"

    # an unregistered worker_id still gets the global mock fallback
    other = client.post(
        "/v1/chat/completions",
        json={"model": "yourself/same", "messages": [{"role": "user", "content": "ping"}]},
    )
    assert other.json()["choices"][0]["message"]["content"] == "mock: [user] ping"

    # admin state reflects the simulated copilots
    state = client.get("/admin/emullm/state").json()
    assert "alice" in state["connected_worker_ids"]
    assert "bob" in state["connected_worker_ids"]
    assert state["worker_roles"]["bob"] == "mock"
    assert state["worker_capabilities"]["alice"] == {"images": True}


def test_specific_worker_routes_to_a_mock_copilot(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(emullm_api, "_SERVER_MODE", "mock")
    emullm_api.register_mock_workers([{"id": "alice", "reply": "alice here"}])  # noqa: SLF001
    response = client.post(
        "/emullm/specific_worker/alice/v1/chat/completions",
        json={"model": "whatever/same", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "alice here"


def test_modes_are_parsed_as_a_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(emullm_api, "_SERVER_MODE", "recruit, proxy , mock")
    assert emullm_api._current_modes() == ["recruit", "proxy", "mock"]
    monkeypatch.setattr(emullm_api, "_SERVER_MODE", ["recruit", "mock"])
    assert emullm_api._current_modes() == ["recruit", "mock"]
    monkeypatch.setattr(emullm_api, "_SERVER_MODE", "")
    assert emullm_api._current_modes() == ["relay"]


def test_chain_recruit_then_mock_falls_back_when_empty(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # No worker connected: recruit passes, mock answers.
    monkeypatch.setattr(emullm_api, "_SERVER_MODE", "recruit,mock")
    response = client.post(
        "/v1/chat/completions",
        json={"model": "yourself/same", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "mock: [user] hi"


def test_chain_recruit_prefers_a_connected_worker_over_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(emullm_api, "_SERVER_MODE", "recruit,mock")
    emullm_api._connected_workers["yourself"] = FakeWorker(reply="real worker answer")  # noqa: SLF001
    # recruit finds the worker, so we never reach the mock fallback.
    assert asyncio.run(emullm_api._relay("yourself/same", "hi")) == "real worker answer"


def test_chain_recruit_then_proxy_falls_back_to_backend(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(emullm_api, "_SERVER_MODE", "recruit,proxy")
    monkeypatch.setenv("EMULLM_PROXY_BASE_URL", "http://backend.test/v1")
    monkeypatch.setattr(
        emullm_api, "_http_post_json",
        lambda *a, **k: {"choices": [{"message": {"content": "from backend"}}]},
    )
    # nobody connected -> recruit passes -> proxy answers
    response = client.post(
        "/v1/chat/completions",
        json={"model": "yourself/same", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "from backend"


def test_recruit_alone_returns_504_when_empty(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # recruit doesn't wait and has no fallback -> chain exhausts -> 504.
    monkeypatch.setattr(emullm_api, "_SERVER_MODE", "recruit")
    response = client.post(
        "/v1/chat/completions",
        json={"model": "yourself/same", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 504


# --- Every run mode is selectable and exercised, using a good-enough mock ---
WORKER_BACKED_MODES = ["relay", "wait", "wait-then-serve", "self", "recruit", "auto", "error-when-empty"]


@pytest.mark.parametrize("mode", WORKER_BACKED_MODES)
def test_every_worker_mode_relays_to_a_connected_mock(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """With a mock agent connected, every worker-backed mode relays to it."""
    monkeypatch.setattr(emullm_api, "_SERVER_MODE", mode)
    emullm_api._connected_workers["yourself"] = FakeWorker(reply=f"ok:{mode}")  # noqa: SLF001
    response = client.post(
        "/v1/chat/completions",
        json={"model": "yourself/same", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200, mode
    assert response.json()["choices"][0]["message"]["content"] == f"ok:{mode}"


@pytest.mark.parametrize(
    "mode,expected_status",
    [
        ("error-when-empty", 503),
        ("recruit", 504),
        ("self", 504),
        ("auto", 504),
        ("wait", 504),
        ("wait-then-serve", 504),
        ("relay", 504),
    ],
)
def test_every_worker_mode_without_a_worker(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, short_timeout: None, mode: str, expected_status: int
) -> None:
    """With no agent connected, each mode fails as designed (fast 503 for
    error-when-empty, otherwise 504)."""
    monkeypatch.setattr(emullm_api, "_SERVER_MODE", mode)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "yourself/same", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == expected_status, mode


def test_mock_mode_is_selectable_and_always_answers(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(emullm_api, "_SERVER_MODE", "mock")
    response = client.post(
        "/v1/chat/completions",
        json={"model": "yourself/same", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "mock: [user] hi"


def test_relay_waits_for_late_worker_instead_of_failing_fast() -> None:
    """Simulates a request landing while no worker is connected: _relay
    must NOT fail fast -- it should wait (like a slow API server) and
    succeed once a worker connects and replies."""

    async def scenario() -> str:
        async def connect_worker_after_delay() -> None:
            await asyncio.sleep(0.2)
            worker = FakeWorker(reply="answered late")
            emullm_api._connected_workers["yourself"] = worker  # noqa: SLF001

        relay_task = asyncio.create_task(emullm_api._relay("yourself/same", "hello"))
        connector_task = asyncio.create_task(connect_worker_after_delay())
        result = await relay_task
        await connector_task
        return result

    original_timeout = emullm_api._REQUEST_TIMEOUT_SECONDS
    emullm_api._REQUEST_TIMEOUT_SECONDS = 5
    try:
        result = asyncio.run(scenario())
    finally:
        emullm_api._REQUEST_TIMEOUT_SECONDS = original_timeout

    assert result == "answered late"


def test_relay_routes_to_the_worker_matching_the_model_prefix() -> None:
    """Two different worker_ids can be "logged in" at once; a request for
    "alice/same" must go to alice's connection, not bob's (or "yourself")."""
    alice = FakeWorker(reply="alice answered")
    bob = FakeWorker(reply="bob answered")
    emullm_api._connected_workers["alice"] = alice  # noqa: SLF001
    emullm_api._connected_workers["bob"] = bob  # noqa: SLF001

    result = asyncio.run(emullm_api._relay("alice/same", "hi"))

    assert result == "alice answered"
    assert len(alice.sent) == 1
    assert not bob.sent


def test_model_masks_prioritize_matches_then_retry_an_all_model_worker() -> None:
    class RejectingWorker:
        def __init__(self) -> None:
            self.sent: list[dict] = []

        async def send_json(self, payload: dict) -> None:
            self.sent.append(payload)
            await emullm_api._handle_worker_message(  # noqa: SLF001
                "masked-servant",
                {"type": "reject", "id": payload["id"], "reason": "outside my live capacity"},
            )

    masked = RejectingWorker()
    fallback = FakeWorker(reply="all-model answer")
    emullm_api._connected_workers["masked-servant"] = masked  # noqa: SLF001
    emullm_api._worker_model_masks["masked-servant"] = ("vendor/*",)  # noqa: SLF001
    emullm_api._connected_workers["all-model-servant"] = fallback  # noqa: SLF001
    emullm_api._native_worker_ids.update({"masked-servant", "all-model-servant"})  # noqa: SLF001

    result = asyncio.run(emullm_api._relay("vendor/any-model", "hello"))

    assert result == "all-model answer"
    assert masked.sent[0]["acceptance_requested"] is True
    assert fallback.sent[0]["model"] == "vendor/any-model"
    assert fallback.sent[0]["worker_id"] == "all-model-servant"


def test_llm_user_interaction_log_is_filterable_over_http_and_websocket(client: TestClient) -> None:
    emullm_api._connected_workers["logged-servant"] = FakeWorker(reply="logged answer")  # noqa: SLF001
    response = client.post(
        "/v1/chat/completions",
        json={"model": "logged-servant/same", "messages": [{"role": "user", "content": "log this"}]},
    )
    assert response.status_code == 200

    log = client.get(
        "/emullm/websock_to_llm_user/events",
        params={"worker_id": "logged-servant", "model": "logged-servant/same"},
    ).json()
    assert log["stream"] == "websock_to_llm_user"
    assert [event["type"] for event in log["events"]] == ["LLM_REQUEST", "LLM_REPLY"]
    assert log["events"][0]["data"]["from"] == "LLM_USER"
    assert log["events"][0]["data"]["to"] == "logged-servant"
    assert log["events"][1]["data"]["from"] == "logged-servant"
    assert log["events"][1]["data"]["to"] == "LLM_USER"

    with client.websocket_connect("/emullm/websock_to_llm_user/ws?worker_id=logged-servant&type=LLM_REPLY") as websocket:
        subscribed = websocket.receive_json()
        assert subscribed["stream"] == "websock_to_llm_user"
        event = websocket.receive_json()
        assert event["type"] == "event"
        assert event["event"]["type"] == "LLM_REPLY"
        assert event["event"]["data"]["worker_id"] == "logged-servant"


def test_list_models_aggregates_every_connected_worker(client: TestClient) -> None:
    emullm_api._connected_workers["alice"] = FakeWorker()  # noqa: SLF001
    emullm_api._worker_models["alice"] = {
        "expert": {"display_name": "(alice)", "instruction": "Be alice."}
    }

    ids = {entry["id"] for entry in client.get("/v1/models").json()["data"]}
    assert "alice/expert" in ids
    assert "alice/same" not in ids
    assert not any(model_id.startswith("yourself/") for model_id in ids)


def test_worker_caps_lookup(client: TestClient) -> None:
    assert client.get("/emullm/caps/yourself").json() == {
        "worker_id": "yourself",
        "connected": False,
        "models": sorted(emullm_api._PERSONA_SUFFIXES.keys()),
        "capabilities": {},
        "modelmasks": None,
        "worker_kind": None,
        "backing_model": None,
        "description": None,
    }

    emullm_api._connected_workers["alice"] = FakeWorker()  # noqa: SLF001
    emullm_api._worker_capabilities["alice"] = {"images": True}

    result = client.get("/emullm/caps/alice").json()
    assert result["connected"] is True
    assert result["capabilities"] == {"images": True}


def test_serve_doc_returns_a_real_markdown_file(client: TestClient) -> None:
    response = client.get("/emullm/docs/EMULLM_RELAY.md")
    assert response.status_code == 200
    assert "text/markdown" in response.headers["content-type"]
    assert "emullm" in response.text.lower()


def test_serve_join_as_worker_doc(client: TestClient) -> None:
    response = client.get("/emullm/docs/EMULLM_ONBOARD.md")
    assert response.status_code == 200
    assert "text/markdown" in response.headers["content-type"]
    assert "worker" in response.text.lower()


def test_serve_doc_404s_for_missing_file(client: TestClient) -> None:
    assert client.get("/emullm/docs/no-such-file.md").status_code == 404


def test_serve_doc_rejects_path_traversal(client: TestClient) -> None:
    assert client.get("/emullm/docs/../server/emullm_api.py").status_code in (400, 404)


def test_serve_doc_file_alias_from_another_directory(client: TestClient, tmp_path) -> None:
    external = tmp_path / "elsewhere" / "external_note.md"
    external.parent.mkdir(parents=True)
    external.write_text("# External\nlives outside the docs tree", encoding="utf-8")
    emullm_api.register_doc_alias("aliased/note.md", external)

    response = client.get("/emullm/docs/aliased/note.md")
    assert response.status_code == 200
    assert "lives outside the docs tree" in response.text
    # a non-aliased missing path still 404s normally
    assert client.get("/emullm/docs/aliased/missing.md").status_code == 404


def test_serve_doc_directory_alias_mounts_a_whole_subtree(client: TestClient, tmp_path) -> None:
    ext_dir = tmp_path / "ext_docs"
    (ext_dir / "sub").mkdir(parents=True)
    (ext_dir / "sub" / "page.md").write_text("subtree page", encoding="utf-8")
    emullm_api.register_doc_alias("mounted", ext_dir)

    assert client.get("/emullm/docs/mounted/sub/page.md").text == "subtree page"
    # traversal out of an aliased directory is refused
    assert client.get("/emullm/docs/mounted/../secret").status_code in (400, 404)


def test_serve_doc_alias_does_not_shadow_real_docs(client: TestClient) -> None:
    # With no alias registered for it, the real on-disk doc still serves.
    assert client.get("/emullm/docs/EMULLM_RELAY.md").status_code == 200


def test_all_docs_served_only_under_the_emullm_docs_prefix(client: TestClient) -> None:
    # Every doc is reachable at the single canonical /emullm/docs/ prefix...
    for name in (
        "EMULLM_ONBOARD.md",
        "EMULLM_RELAY.md",
    ):
        response = client.get(f"/emullm/docs/{name}")
        assert response.status_code == 200, name
        assert "text/markdown" in response.headers["content-type"]

    # ...and the old bare-root / /docs/ / /workbench/docs/ aliases are gone.
    for alias in (
        "/EMULLM_ONBOARD.md",
        "/docs/EMULLM_ONBOARD.md",
        "/workbench/docs/EMULLM_ONBOARD.md",
        "/EMULLM_RELAY.md",
        "/workbench/docs/EMULLM_RELAY.md",
    ):
        assert client.get(alias).status_code == 404, alias

    # The old design/ subpath is gone too; the file lives at the top level.
    assert client.get("/emullm/docs/design/EMULLM_RELAY.md").status_code == 404


def test_static_html_served_at_namespaced_and_root_paths(client: TestClient) -> None:
    namespaced = client.get("/emullm/static/index.html")
    assert namespaced.status_code == 200
    assert "text/html" in namespaced.headers["content-type"]
    assert "emullm" in namespaced.text

    root = client.get("/index.html")
    assert root.status_code == 200
    assert root.text == namespaced.text


def test_static_missing_and_traversal(client: TestClient) -> None:
    assert client.get("/emullm/static/nope.html").status_code == 404
    assert client.get("/does-not-exist.html").status_code == 404
    assert client.get("/emullm/static/../api.py").status_code in (400, 404)


def test_specific_worker_prefix_pins_worker_regardless_of_model_field(client: TestClient, short_timeout: None) -> None:
    """A client hitting /emullm/specific_worker/alice/v1/chat/completions
    must be routed to alice even if it sends a "model" naming someone
    else (or the default) -- only the persona suffix is kept."""
    alice = FakeWorker(reply="alice's real answer")
    emullm_api._connected_workers["alice"] = alice  # noqa: SLF001
    bob = FakeWorker(reply="bob would never see this")
    emullm_api._connected_workers["bob"] = bob  # noqa: SLF001

    response = client.post(
        "/emullm/specific_worker/alice/v1/chat/completions",
        json={"model": "bob/percent25", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "alice's real answer"
    assert not bob.sent
    assert alice.sent[0]["model"] == "alice/percent25"  # worker_id forced, suffix kept


def test_specific_worker_messages_prefix_pins_worker_regardless_of_model_field(client: TestClient) -> None:
    alice = FakeWorker(reply="alice's message answer")
    emullm_api._connected_workers["alice"] = alice  # noqa: SLF001
    bob = FakeWorker(reply="bob would never see this")
    emullm_api._connected_workers["bob"] = bob  # noqa: SLF001

    response = client.post(
        "/emullm/specific_worker/alice/v1/messages",
        json={
            "model": "bob/percent25",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200
    assert response.json()["content"][0]["text"] == "alice's message answer"
    assert not bob.sent
    assert alice.sent[0]["model"] == "alice/percent25"  # worker_id forced, suffix kept


def test_specific_worker_models_listing_is_scoped_to_that_worker(client: TestClient) -> None:
    emullm_api._worker_models["alice"] = {
        "percent100": {"display_name": "(alice)", "instruction": "Be alice."}
    }

    data = client.get("/emullm/specific_worker/alice/v1/models").json()["data"]

    assert {entry["id"] for entry in data} == {"alice/percent100"}


def test_specific_worker_get_model_and_404(client: TestClient) -> None:
    assert client.get(
        "/emullm/specific_worker/alice/v1/models/anything/percent100"
    ).status_code == 200
    assert client.get(
        "/emullm/specific_worker/alice/v1/models/anything/same"
    ).status_code == 404
    assert client.get("/emullm/specific_worker/alice/v1/models/anything/no-such-suffix").status_code == 404


def test_rate_limit_returns_429_with_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once a worker_id has been used up to the configured limit within
    the window, further requests for it fail fast with 429 and a
    Retry-After, instead of queuing more work onto a busy worker."""
    monkeypatch.setattr(emullm_api, "_USAGE_MAX_PER_WINDOW", 2)
    monkeypatch.setattr(emullm_api, "_USAGE_WINDOW_SECONDS", 60.0)
    worker = FakeWorker(reply="ok")
    emullm_api._connected_workers["yourself"] = worker  # noqa: SLF001

    asyncio.run(emullm_api._relay("yourself/same", "one"))
    asyncio.run(emullm_api._relay("yourself/same", "two"))

    with pytest.raises(emullm_api.HTTPException) as excinfo:
        asyncio.run(emullm_api._relay("yourself/same", "three"))
    assert excinfo.value.status_code == 429
    assert "Retry-After" in excinfo.value.headers


def test_rate_limit_is_independent_per_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """A different worker_id isn't affected by another one being maxed
    out -- so an idle worker can pick up slack for a busy one."""
    monkeypatch.setattr(emullm_api, "_USAGE_MAX_PER_WINDOW", 1)
    monkeypatch.setattr(emullm_api, "_USAGE_WINDOW_SECONDS", 60.0)
    emullm_api._connected_workers["alice"] = FakeWorker(reply="a")  # noqa: SLF001
    emullm_api._connected_workers["bob"] = FakeWorker(reply="b")  # noqa: SLF001

    asyncio.run(emullm_api._relay("alice/same", "hi"))
    with pytest.raises(emullm_api.HTTPException):
        asyncio.run(emullm_api._relay("alice/same", "hi again"))

    # bob is untouched by alice's limit
    assert asyncio.run(emullm_api._relay("bob/same", "hi")) == "b"


def test_embeddings_is_deterministic_without_pretend_capability(client: TestClient) -> None:
    first = client.post("/v1/embeddings", json={"input": "hello", "dimensions": 64}).json()
    second = client.post("/v1/embeddings", json={"input": "hello", "dimensions": 64}).json()
    assert first["data"][0]["embedding"] == second["data"][0]["embedding"]
    assert len(first["data"][0]["embedding"]) == 64
    assert first["usage"]["total_tokens"] == 1


def test_embeddings_pretend_mode_routes_to_the_capable_worker(client: TestClient) -> None:
    worker = FakeWorker(reply="a vector about greetings")
    emullm_api._connected_workers["worker-copilot-1"] = worker  # noqa: SLF001
    emullm_api._worker_capabilities["worker-copilot-1"] = {"embeddings": True}

    client.post("/v1/embeddings", json={"input": "hello"})

    assert len(worker.sent) == 1
    assert "pretend-embeddings" in worker.sent[0]["prompt"]


def test_embeddings_declined_capability_stops_before_asking_worker(client: TestClient) -> None:
    worker = FakeWorker(reply="should never be used")
    emullm_api._connected_workers["worker-copilot-1"] = worker  # noqa: SLF001
    emullm_api._worker_capabilities["worker-copilot-1"] = {  # explicit decline
        "embeddings": False
    }

    response = client.post("/v1/embeddings", json={"input": "hello"})

    assert response.status_code == 501
    assert not worker.sent  # never even asked


def test_capability_fallback_error_fails_fast(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # No capable worker + capability_fallback=error -> 503, no stub.
    monkeypatch.setattr(emullm_api, "_CAPABILITY_FALLBACK", "error")
    response = client.post("/v1/embeddings", json={"input": "hello"})
    assert response.status_code == 503


def test_capability_fallback_wait_times_out(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # No capable worker + capability_fallback=wait -> hold, then 504 on timeout.
    monkeypatch.setattr(emullm_api, "_CAPABILITY_FALLBACK", "wait")
    monkeypatch.setattr(emullm_api, "_REQUEST_TIMEOUT_SECONDS", 0.3)
    response = client.post("/v1/embeddings", json={"input": "hello"})
    assert response.status_code == 504


def test_capability_fallback_stub_is_default(client: TestClient) -> None:
    # Default (stub) still answers immediately with the deterministic fake.
    assert emullm_api._capability_fallback() == "stub"
    assert client.post("/v1/embeddings", json={"input": "hello"}).status_code == 200


def test_per_agent_service_behavior_overrides_fallback(client: TestClient) -> None:
    # An agent's per-service behavior wins over the global/server fallback.
    emullm_api.apply_agent_policies(
        {
            "agents": [
                {
                    "kind": "agent",
                    "id": "alice",
                    "launch": "subagent",
                    "services": {"images": "error", "embeddings": "stub", "moderations": "decline"},
                }
            ]
        }
    )
    assert client.post("/v1/images/generations", json={"model": "alice/same", "prompt": "x"}).status_code == 503
    assert client.post("/v1/embeddings", json={"model": "alice/same", "input": "hi"}).status_code == 200
    assert client.post("/v1/moderations", json={"model": "alice/same", "input": "hi"}).status_code == 501


def test_server_level_service_fallback_applies(client: TestClient) -> None:
    # server-level services fallback is used when no agent serves it.
    emullm_api.apply_agent_policies({"services": {"images": {"fallback": "error"}}})
    assert client.post("/v1/images/generations", json={"model": "yourself/same", "prompt": "x"}).status_code == 503


def test_server_fallback_chain(client: TestClient) -> None:
    # A chain "round-robin, error": no agent volunteered for images -> the
    # strategy token passes, then error terminates with 503.
    emullm_api.apply_agent_policies({"services": {"images": {"fallback": "round-robin, error"}}})
    assert emullm_api._service_fallback["images"] == ["round-robin", "error"]
    assert client.post("/v1/images/generations", json={"model": "yourself/same", "prompt": "x"}).status_code == 503
    # "round-robin, stub": strategy passes, stub answers (200)
    emullm_api.apply_agent_policies({"services": {"images": {"fallback": ["round-robin", "stub"]}}})
    assert client.post("/v1/images/generations", json={"model": "yourself/same", "prompt": "x"}).status_code == 200


def test_agent_aggregate_is_a_volunteer(client: TestClient) -> None:
    # an agent-level aggregate service = "I volunteer" -> served (relayed).
    worker = FakeWorker(reply="desc")
    emullm_api._connected_workers["o"] = worker  # noqa: SLF001
    emullm_api.apply_agent_policies(
        {"agents": [{"kind": "agent", "id": "o", "launch": "proxy", "services": {"images": {"behavior": "aggregate"}}}]}
    )
    result = client.post("/v1/images/generations", json={"model": "o/same", "prompt": "a cat"}).json()
    assert result["data"][0].get("pretend_description") == "desc"  # volunteered -> asked the worker


def test_observer_receives_mirrored_exchange(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(emullm_api, "_SERVER_MODE", "mock")
    observer = FakeWorker()
    emullm_api._connected_workers["bob"] = observer  # noqa: SLF001
    emullm_api.apply_agent_policies(
        {"agents": [{"kind": "agent", "id": "bob", "launch": "recruit", "observe": ["chat"]}]}
    )
    client.post("/v1/chat/completions", json={"model": "carol/same", "messages": [{"role": "user", "content": "hi"}]})
    assert any(payload.get("type") == "observe" for payload in observer.sent)


def test_parse_interval() -> None:
    assert emullm_api._parse_interval("1day") == 86400.0
    assert emullm_api._parse_interval(None) is None
    assert emullm_api._parse_interval("never") is None
    assert emullm_api._parse_interval("12h") == 43200.0
    assert emullm_api._parse_interval("30m") == 1800.0
    assert emullm_api._parse_interval("always") == 0.0


def test_advertise_models_aggregates_and_fetches(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        emullm_api, "_http_get_json", lambda url, headers, timeout=15.0: {"data": [{"id": "m1"}, {"id": "m2"}]}
    )
    emullm_api.apply_agent_policies(
        {
            "services": {"model": "base-1", "models": ["base-1"]},
            "agents": [
                {
                    "kind": "agent",
                    "id": "up",
                    "launch": "proxy",
                    "base_url": "http://x/v1",
                    "models": ["cfg-fallback"],
                    "services": {"models": {"behavior": "aggregate", "update_interval": "1day"}},
                }
            ],
        }
    )
    cat = emullm_api.advertised_catalog()
    assert cat["model"] == "base-1"
    assert cat["models"] == ["base-1", "m1", "m2"]  # base + live fetch, deduped in order


def test_update_interval_null_uses_config_models_without_fetching(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}

    def fake_get(url, headers, timeout=15.0):
        calls["n"] += 1
        return {"data": [{"id": "live"}]}

    monkeypatch.setattr(emullm_api, "_http_get_json", fake_get)
    emullm_api.apply_agent_policies(
        {
            "agents": [
                {
                    "kind": "agent",
                    "id": "noref",
                    "launch": "proxy",
                    "base_url": "http://x/v1",
                    "models": ["cfg-a", "cfg-b"],
                    "services": {"models": {"behavior": "aggregate", "update_interval": None}},
                }
            ]
        }
    )
    cat = emullm_api.advertised_catalog()
    assert cat["models"] == ["cfg-a", "cfg-b"]  # null interval -> config models
    assert calls["n"] == 0  # and no live fetch happened


def test_advertise_and_interval_via_services_models_entry(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}

    def fake_get(url, headers, timeout=15.0):
        calls["n"] += 1
        return {"data": [{"id": "live-1"}]}

    monkeypatch.setattr(emullm_api, "_http_get_json", fake_get)
    emullm_api.apply_agent_policies(
        {
            "agents": [
                {
                    "kind": "agent",
                    "id": "o",
                    "launch": "proxy",
                    "base_url": "http://x/v1",
                    "models": ["cfg-1"],
                    "services": {
                        "chat": "serve",
                        "models": {"behavior": "aggregate", "update_interval": None, "description": "catalog"},
                    },
                }
            ]
        }
    )
    cat = emullm_api.advertised_catalog()
    assert cat["models"] == ["cfg-1"]  # advertised via services.models; null interval -> config list
    assert calls["n"] == 0
    # the reserved "models" catalog entry is NOT a routable service behavior
    behaviors = emullm_api._worker_service_behavior.get("o", {})
    assert "models" not in behaviors
    assert behaviors.get("chat") == "serve"


def test_moderations_never_flags_without_pretend_capability(client: TestClient) -> None:
    result = client.post("/v1/moderations", json={"input": "anything"}).json()
    assert result["results"][0]["flagged"] is False


def test_moderations_pretend_mode_uses_worker_verdict(client: TestClient) -> None:
    emullm_api._connected_workers["worker-copilot-1"] = FakeWorker(reply="FLAG")  # noqa: SLF001
    emullm_api._worker_capabilities["worker-copilot-1"] = {"moderations": True}

    result = client.post("/v1/moderations", json={"input": "anything"}).json()

    assert result["results"][0]["flagged"] is True


def test_moderations_declined_capability_stops_before_asking_worker(client: TestClient) -> None:
    worker = FakeWorker(reply="should never be used")
    emullm_api._connected_workers["worker-copilot-1"] = worker  # noqa: SLF001
    emullm_api._worker_capabilities["worker-copilot-1"] = {"moderations": False}

    response = client.post("/v1/moderations", json={"input": "anything"})

    assert response.status_code == 501
    assert not worker.sent


def test_images_generations_returns_stub_url(client: TestClient) -> None:
    result = client.post("/v1/images/generations", json={"prompt": "a cat"}).json()
    assert result["data"][0]["url"].startswith("data:image/png;base64,")
    assert "pretend_description" not in result["data"][0]

    base64_result = client.post(
        "/v1/images/generations",
        json={"prompt": "a cat", "response_format": "b64_json"},
    ).json()
    assert base64_result["data"][0]["b64_json"]
    assert "url" not in base64_result["data"][0]


def test_images_generations_pretend_mode_adds_description(client: TestClient) -> None:
    emullm_api._connected_workers["worker-copilot-1"] = FakeWorker(  # noqa: SLF001
        reply="a fluffy orange cat"
    )
    emullm_api._worker_capabilities["worker-copilot-1"] = {"images": True}

    result = client.post("/v1/images/generations", json={"prompt": "a cat"}).json()

    assert result["data"][0]["pretend_description"] == "a fluffy orange cat"


def test_images_declined_capability_stops_before_asking_worker(client: TestClient) -> None:
    worker = FakeWorker(reply="should never be used")
    emullm_api._connected_workers["worker-copilot-1"] = worker  # noqa: SLF001
    emullm_api._worker_capabilities["worker-copilot-1"] = {"images": False}

    response = client.post("/v1/images/generations", json={"prompt": "a cat"})

    assert response.status_code == 501
    assert not worker.sent


def test_audio_transcriptions_is_stub(client: TestClient) -> None:
    result = client.post(
        "/v1/audio/transcriptions",
        data={"model": "yourself/same"},
        files={"file": ("sample.wav", b"RIFF synthetic audio", "audio/wav")},
    ).json()
    assert "not implemented" in result["text"]
    assert client.post("/v1/audio/transcriptions").status_code == 415


def test_audio_transcriptions_pretend_mode_uses_worker_text(client: TestClient) -> None:
    emullm_api._connected_workers["worker-copilot-1"] = FakeWorker(  # noqa: SLF001
        reply="hello there"
    )
    emullm_api._worker_capabilities["worker-copilot-1"] = {
        "audio_transcription": True
    }

    result = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", b"RIFF synthetic audio", "audio/wav")},
    ).json()

    assert result["text"] == "hello there"


def test_audio_transcriptions_declined_capability_stops_before_asking_worker(client: TestClient) -> None:
    worker = FakeWorker(reply="should never be used")
    emullm_api._connected_workers["worker-copilot-1"] = worker  # noqa: SLF001
    emullm_api._worker_capabilities["worker-copilot-1"] = {
        "audio_transcription": False
    }

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", b"RIFF synthetic audio", "audio/wav")},
    )

    assert response.status_code == 501
    assert not worker.sent


def test_audio_speech_returns_valid_synthetic_wav(client: TestClient) -> None:
    result = client.post("/v1/audio/speech", json={"input": "hi"})
    assert result.status_code == 200
    assert result.headers["content-type"].startswith("audio/wav")
    assert result.headers["x-emullm-synthetic"] == "true"
    assert result.content[:4] == b"RIFF"
    assert result.content[8:12] == b"WAVE"


def test_audio_speech_pretend_mode_adds_description(client: TestClient) -> None:
    emullm_api._connected_workers["worker-copilot-1"] = FakeWorker(  # noqa: SLF001
        reply="said cheerfully"
    )
    emullm_api._worker_capabilities["worker-copilot-1"] = {"audio_speech": True}

    result = client.post("/v1/audio/speech", json={"input": "hi"})

    assert result.headers["x-emullm-description"] == "said cheerfully"


# --- two-way real media pass-through + shared cloud files --------------------
def test_cloud_files_roundtrip_uses_emullm_prefix(client: TestClient) -> None:
    record = emullm_api._store_cloud_bytes(b"hello-cloud", "note.txt", purpose="output")  # noqa: SLF001
    url = emullm_api._cloud_file_url(record["id"])  # noqa: SLF001
    assert url.startswith("/emullm/cloud/files/")
    got = client.get(url)
    assert got.status_code == 200 and got.content == b"hello-cloud"


def test_images_two_way_returns_worker_image_via_cloud_file(client: TestClient) -> None:
    import base64

    raw = b"\x89PNG\r\n\x1a\nFAKE-IMAGE-BYTES"
    b64 = base64.b64encode(raw).decode()
    emullm_api._connected_workers["yourself"] = FakeWorker(reply={"content": "", "image_b64": b64})  # noqa: SLF001
    emullm_api._worker_capabilities["yourself"] = {"images": True}

    entry = client.post("/v1/images/generations", json={"model": "yourself", "prompt": "a red bike"}).json()["data"][0]
    assert entry["source"] == "worker"
    assert entry["url"].startswith("/emullm/cloud/files/")
    assert entry["file_id"]
    got = client.get(entry["url"])
    assert got.status_code == 200 and got.content == raw  # the real bytes came back through the cloud store


def test_images_two_way_b64_json_returns_worker_bytes(client: TestClient) -> None:
    import base64

    raw = b"REAL-IMG"
    b64 = base64.b64encode(raw).decode()
    emullm_api._connected_workers["yourself"] = FakeWorker(reply={"content": "", "image_b64": b64})  # noqa: SLF001
    emullm_api._worker_capabilities["yourself"] = {"images": True}

    entry = client.post(
        "/v1/images/generations",
        json={"model": "yourself", "prompt": "x", "response_format": "b64_json"},
    ).json()["data"][0]
    assert entry["b64_json"] == b64
    assert "url" not in entry


def test_codex_copilot_model_can_generate_real_image_with_tools(
    client: TestClient,
    monkeypatch,
) -> None:
    raw = b"\x89PNG\r\n\x1a\nCODEX-GENERATED"
    metadata = {
        "id": "gpt-5.3-codex",
        "name": "GPT-5.3 Codex",
        "capabilities": {"type": "chat"},
    }
    monkeypatch.setattr(
        emullm_api._copilot_api,  # noqa: SLF001
        "copilot_models",
        lambda **_kwargs: {"models": [metadata], "source": "test"},
    )
    assert emullm_api._copilot_task_capabilities(  # noqa: SLF001
        "gpt-5.3-codex",
        metadata,
    )["image_output"]["enabled"] is True

    async def relay(*_args, **kwargs):
        assert kwargs["kind"] == "image"
        assert kwargs["required_capabilities"] == {"image_output"}
        return {
            "content": "EMULLM_IMAGE_FILE: emullm-generated-image.png",
            "image_b64": base64.b64encode(raw).decode("ascii"),
            "mime": "image/png",
        }

    monkeypatch.setattr(emullm_api, "_relay_full", relay)
    response = client.post(
        "/v1/images/generations",
        json={
            "model": "copilot/gpt-5.3-codex",
            "prompt": "a red circle",
        },
    )

    assert response.status_code == 200, response.text
    entry = response.json()["data"][0]
    assert entry["source"] == "worker"
    assert entry["mime_type"] == "image/png"
    assert client.get(entry["url"]).content == raw


def test_codex_image_edit_accepts_source_mask_and_returns_artifact(
    client: TestClient,
    monkeypatch,
) -> None:
    output = b"\x89PNG\r\n\x1a\nEDITED"
    metadata = {
        "id": "gpt-5.3-codex",
        "name": "GPT-5.3 Codex",
        "capabilities": {"type": "chat"},
    }
    monkeypatch.setattr(
        emullm_api._copilot_api,  # noqa: SLF001
        "copilot_models",
        lambda **_kwargs: {"models": [metadata], "source": "test"},
    )
    captured = {}

    async def relay(*_args, **kwargs):
        captured.update(kwargs)
        return {
            "image_b64": base64.b64encode(output).decode("ascii"),
            "mime": "image/png",
        }

    monkeypatch.setattr(emullm_api, "_relay_full", relay)
    response = client.post(
        "/v1/images/edits",
        data={
            "model": "copilot/gpt-5.3-codex",
            "prompt": "replace the background",
            "size": "512x512",
        },
        files={
            "image": ("private-source.png", b"source", "image/png"),
            "mask": ("private-mask.png", b"mask", "image/png"),
        },
    )

    assert response.status_code == 200, response.text
    entry = response.json()["data"][0]
    assert entry["operation"] == "edit"
    assert entry["source"] == "worker"
    assert entry["artifact"]["mime_type"] == "image/png"
    assert client.get(entry["artifact"]["url"]).content == output
    assert entry["inputs"]["image"]["name"] == "attachment-1"
    assert entry["inputs"]["mask"]["name"] == "attachment-2"
    assert entry["inputs"]["size"] == "512x512"
    assert captured["kind"] == "image_edit"
    assert captured["required_capabilities"] == {
        "vision_input",
        "image_output",
    }
    assert [item["name"] for item in captured["attachments"]] == [
        "attachment-1",
        "attachment-2",
    ]


def test_audio_speech_two_way_returns_worker_audio(client: TestClient) -> None:
    import base64

    wav = b"RIFF____WAVEreal-audio"
    b64 = base64.b64encode(wav).decode()
    emullm_api._connected_workers["yourself"] = FakeWorker(  # noqa: SLF001
        reply={"content": "", "audio_b64": b64, "mime": "audio/wav"}
    )
    emullm_api._worker_capabilities["yourself"] = {"audio_speech": True}

    r = client.post("/v1/audio/speech", json={"model": "yourself", "input": "hello"})
    assert r.status_code == 200
    assert r.content == wav  # the worker's real audio, not the synthetic stub
    assert r.headers["x-emullm-synthetic"] == "false"
    assert r.headers["x-emullm-file"].startswith("/emullm/cloud/files/")


def test_audio_transcription_two_way_references_real_clip(client: TestClient) -> None:
    worker = FakeWorker(reply="the transcript")
    emullm_api._connected_workers["yourself"] = worker  # noqa: SLF001
    emullm_api._worker_capabilities["yourself"] = {"audio_transcription": True}

    r = client.post(
        "/v1/audio/transcriptions",
        data={"model": "yourself/same"},
        files={"file": ("clip.wav", b"RIFF real audio bytes", "audio/wav")},
    ).json()
    assert r["text"] == "the transcript"
    assert r["audio_file"]
    sent = worker.sent[0]
    assert sent["kind"] == "audio_transcription"
    assert sent["audio"].startswith("/emullm/cloud/files/")
    got = client.get(sent["audio"])  # the worker was handed a real, fetchable clip
    assert got.status_code == 200 and got.content == b"RIFF real audio bytes"


def test_fine_tuning_worker_volunteer_trains_and_publishes_result(client: TestClient) -> None:
    upload = client.post(
        "/v1/files",
        data={"purpose": "fine-tune"},
        files={"file": ("t.jsonl", b'{"messages": []}\n', "application/json")},
    ).json()
    worker = FakeWorker(reply="I'll train it with 3 epochs")
    emullm_api._connected_workers["yourself"] = worker  # noqa: SLF001
    emullm_api._worker_capabilities["yourself"] = {"fine_tuning": True}

    job = client.post(
        "/v1/fine_tuning/jobs",
        json={"model": "yourself", "training_file": upload["id"]},
    ).json()
    assert job["status"] == "succeeded"
    assert job["fine_tuned_model"].startswith("ft:yourself:")
    assert job["error"] is None
    assert job["result_files"]
    manifest = client.get(emullm_api._cloud_file_url(job["result_files"][0]))  # noqa: SLF001
    assert manifest.status_code == 200
    assert b"fine_tuned_model" in manifest.content
    # the worker was routed a reference to the real training data
    assert worker.sent[0]["kind"] == "fine_tuning"
    assert worker.sent[0]["files"]["training_file"] == upload["id"]


def test_model_routes_from_config_serves(client: TestClient) -> None:
    emullm_api.apply_agent_policies(
        {"agents": [{"kind": "agent", "id": "srv", "serves": ["google/gemma-4-31b-it", "qwen/qwen3.8-27b"]}]}
    )
    assert emullm_api._model_routes["google/gemma-4-31b-it"] == "srv"  # noqa: SLF001
    wid, _suffix, persona = emullm_api._require_model("qwen/qwen3.8-27b")  # noqa: SLF001
    assert wid == "srv"
    assert persona.get("served_model") == "qwen/qwen3.8-27b"


def test_model_routes_admin_get_set_remove(client: TestClient) -> None:
    assert client.post("/admin/emullm/model_routes", json={"routes": {"a/b": "w1"}}).json()["model_routes"]["a/b"] == "w1"
    assert client.get("/admin/emullm/model_routes").json()["model_routes"]["a/b"] == "w1"
    assert client.get("/admin/emullm/state").json()["model_routes"]["a/b"] == "w1"
    client.post("/admin/emullm/model_routes", json={"routes": {"a/b": ""}})  # empty removes
    assert "a/b" not in client.get("/admin/emullm/model_routes").json()["model_routes"]
    chain = ["worker-copilot-*", "worker-codex-*", "https://llm.example/v1"]
    saved = client.post("/admin/emullm/model_routes", json={"routes": {"a/b": chain}})
    assert saved.json()["model_routes"]["a/b"] == chain
    comma = client.post(
        "/admin/emullm/model_routes",
        json={"routes": {"c/d": "worker-copilot-*, worker-codex-*"}},
    )
    assert comma.json()["model_routes"]["c/d"] == ["worker-copilot-*", "worker-codex-*"]


def test_model_route_relays_catalog_id_to_worker(client: TestClient) -> None:
    # A routed catalog id (with a '/') is served by the mapped worker, and the
    # worker is told which model to emulate.
    emullm_api._connected_workers["srv"] = FakeWorker(reply="served as gemma")  # noqa: SLF001
    client.post("/admin/emullm/model_routes", json={"routes": {"google/gemma-4-31b-it": "srv"}})
    r = client.post(
        "/v1/chat/completions",
        json={"model": "google/gemma-4-31b-it", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "served as gemma"
    sent = emullm_api._connected_workers["srv"].sent[0]  # noqa: SLF001
    assert "google/gemma-4-31b-it" in sent.get("persona_instruction", "")


def test_model_route_chain_prefers_first_matching_worker_glob(client: TestClient) -> None:
    first = FakeWorker(reply="copilot answer")
    second = FakeWorker(reply="codex answer")
    emullm_api._connected_workers.update(  # noqa: SLF001
        {"worker-copilot-2": first, "worker-codex-1": second}
    )
    emullm_api._model_routes["vendor/model"] = [  # noqa: SLF001
        "worker-copilot-*",
        "worker-codex-*",
        "https://llm.example/v1",
    ]

    response = client.post(
        "/v1/chat/completions",
        json={"model": "vendor/model", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "copilot answer"
    assert first.sent
    assert second.sent == []


def test_model_route_chain_retries_next_worker_group_after_rejection(client: TestClient) -> None:
    class RejectingWorker:
        async def send_json(self, payload: dict) -> None:
            future = emullm_api._pending[payload["id"]]  # noqa: SLF001
            future.set_exception(emullm_api._WorkerRejected("worker-copilot-1", "declined"))  # noqa: SLF001

    fallback = FakeWorker(reply="codex fallback")
    emullm_api._connected_workers.update(  # noqa: SLF001
        {"worker-copilot-1": RejectingWorker(), "worker-codex-1": fallback}
    )
    emullm_api._model_routes["vendor/model"] = [  # noqa: SLF001
        "worker-copilot-*",
        "worker-codex-*",
    ]

    response = client.post(
        "/v1/chat/completions",
        json={"model": "vendor/model", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "codex fallback"


def test_model_route_chain_falls_back_to_its_backend_url(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}
    backend_url = "https://llm.a.singularitycompute.com/v1"
    monkeypatch.setattr(
        emullm_api,
        "_all_backends",
        lambda: [{"name": "fallback", "base_url": backend_url, "api_key": "secret"}],
    )

    def fake_post(url, headers, payload, timeout=60.0):
        captured.update({"url": url, "headers": headers, "payload": payload})
        return {"choices": [{"message": {"content": "backend fallback"}}]}

    monkeypatch.setattr(emullm_api, "_http_post_json", fake_post)
    emullm_api._model_routes["vendor/model"] = [  # noqa: SLF001
        "worker-copilot-*",
        "worker-codex-*",
        backend_url,
    ]

    response = client.post(
        "/v1/chat/completions",
        json={"model": "vendor/model", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "backend fallback"
    assert captured["url"] == f"{backend_url}/chat/completions"
    assert captured["payload"]["model"] == "vendor/model"
    assert captured["headers"]["Authorization"] == "Bearer secret"


def test_admin_test_client_cancellation_clears_pending_and_notifies_worker(
    client: TestClient,
) -> None:
    class SlowWorker:
        def __init__(self) -> None:
            self.sent: list[dict] = []

        async def send_json(self, payload: dict) -> None:
            self.sent.append(payload)

    worker = SlowWorker()
    emullm_api._connected_workers["yourself"] = worker  # noqa: SLF001

    async def scenario() -> None:
        request = emullm_api.AdminTestChatRequest(  # noqa: SLF001
            request_id="browser-request",
            model="yourself/same",
            prompt="take your time",
        )
        post_task = asyncio.create_task(emullm_api.admin_test_chat(request))
        for _ in range(100):
            if emullm_api._pending:  # noqa: SLF001
                break
            await asyncio.sleep(0.01)
        cancelled = await emullm_api.admin_cancel_test_chat("browser-request")
        assert cancelled == {"request_id": "browser-request", "cancelled": True}
        with pytest.raises(HTTPException) as exc:
            await post_task
        assert exc.value.status_code == 499

    asyncio.run(scenario())
    assert emullm_api._pending == {}  # noqa: SLF001
    assert emullm_api._admin_test_tasks == {}  # noqa: SLF001
    assert [message["type"] for message in worker.sent] == ["request", "cancel"]


def test_admin_test_client_relays_image_audio_and_general_file_attachments(
    client: TestClient,
) -> None:
    worker = FakeWorker(reply="I received all three attachments.")
    emullm_api._connected_workers["yourself"] = worker  # noqa: SLF001
    attachments = [
        ("pixel.png", "image/png", b"\x89PNG\r\n\x1a\nimage"),
        ("sample.wav", "audio/wav", b"RIFF\x00\x00\x00\x00WAVEaudio"),
        ("notes.txt", "text/plain", b"arbitrary attached text"),
    ]

    response = client.post(
        "/emullm/admin/test-chat",
        json={
            "request_id": "multimodal-test",
            "model": "yourself/same",
            "prompt": "Describe every attachment.",
            "attachments": [
                {
                    "name": name,
                    "mime_type": mime_type,
                    "data_b64": base64.b64encode(data).decode("ascii"),
                }
                for name, mime_type, data in attachments
            ],
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["request_kind"] == "vision"
    assert body["choices"][0]["message"]["content"] == "I received all three attachments."
    assert [entry["name"] for entry in body["attachments"]] == [
        "attachment-1",
        "attachment-2",
        "attachment-3",
    ]
    request = worker.sent[0]
    assert request["kind"] == "vision"
    assert request["images"] == [body["attachments"][0]["url"]]
    assert request["audio"] == body["attachments"][1]["url"]
    assert request["attachments"] == body["attachments"]
    assert request["files"]["attachments"] == body["attachments"]
    serialized_request = json.dumps(request)
    assert all(original_name not in serialized_request for original_name, _, _ in attachments)
    for metadata, (_, mime_type, expected) in zip(body["attachments"], attachments):
        download = client.get(metadata["url"])
        assert download.status_code == 200
        assert download.content == expected
        assert download.headers["content-type"].startswith(mime_type)


def test_admin_test_client_rejects_invalid_attachment_base64(client: TestClient) -> None:
    response = client.post(
        "/emullm/admin/test-chat",
        json={
            "request_id": "bad-attachment",
            "model": "yourself/same",
            "prompt": "Read this.",
            "attachments": [
                {
                    "name": "bad.bin",
                    "mime_type": "application/octet-stream",
                    "data_b64": "not base64!",
                }
            ],
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "attachment 'attachment-1' is not valid base64"


def test_admin_test_client_enforces_per_file_and_total_attachment_limits(
    client: TestClient,
    monkeypatch,
) -> None:
    encoded = base64.b64encode(b"1234").decode("ascii")
    payload = {
        "request_id": "attachment-limit",
        "model": "yourself/same",
        "prompt": "Read this.",
        "attachments": [
            {
                "name": "one.bin",
                "mime_type": "application/octet-stream",
                "data_b64": encoded,
            }
        ],
    }
    monkeypatch.setattr(emullm_api, "_MAX_ADMIN_TEST_ATTACHMENT_BYTES", 3)
    response = client.post("/emullm/admin/test-chat", json=payload)
    assert response.status_code == 413
    assert response.json()["detail"] == "attachment 'attachment-1' exceeds the 3-byte limit"

    monkeypatch.setattr(emullm_api, "_MAX_ADMIN_TEST_ATTACHMENT_BYTES", 10)
    monkeypatch.setattr(emullm_api, "_MAX_ADMIN_TEST_ATTACHMENTS_TOTAL_BYTES", 7)
    payload["attachments"].append(
        {
            "name": "two.bin",
            "mime_type": "application/octet-stream",
            "data_b64": encoded,
        }
    )
    response = client.post("/emullm/admin/test-chat", json=payload)
    assert response.status_code == 413
    assert "exceed the 7-byte total limit" in response.json()["detail"]


def test_admin_attachment_samples_are_listed_and_downloadable(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        emullm_api,
        "test_media_samples",
        lambda: {
            "sample-audio": {
                "id": "sample-audio",
                "name": "sample.wav",
                "mime_type": "audio/wav",
                "description": "Test speech",
                "data": b"RIFF sample",
            }
        },
    )

    listing = client.get("/emullm/admin/test-samples")
    assert listing.status_code == 200
    assert listing.json()["samples"] == [
        {
            "id": "sample-audio",
            "name": "sample.wav",
            "mime_type": "audio/wav",
            "description": "Test speech",
            "bytes": 11,
            "url": "/emullm/admin/test-samples/sample-audio",
        }
    ]
    download = client.get("/emullm/admin/test-samples/sample-audio")
    assert download.status_code == 200
    assert download.content == b"RIFF sample"
    assert download.headers["content-type"].startswith("audio/wav")
    assert client.get("/emullm/admin/test-samples/missing").status_code == 404


def test_bulk_stop_idle_pauses_maintenance_and_skips_recently_busy_workers(
    client: TestClient,
    monkeypatch,
) -> None:
    class ShutdownPeer:
        def __init__(self, worker_id: str) -> None:
            self.worker_id = worker_id

        async def send_json(self, payload):
            assert payload["type"] == "shutdown"
            emullm_api._connected_workers.pop(self.worker_id, None)  # noqa: SLF001

    class FakeManager:
        @staticmethod
        def list():
            return [
                {
                    "worker_id": "worker-copilot-1",
                    "running": True,
                    "connected": True,
                },
                {
                    "worker_id": "worker-copilot-2",
                    "running": True,
                    "connected": True,
                },
            ]

    emullm_api._connected_workers.update(  # noqa: SLF001
        {
            "worker-copilot-1": ShutdownPeer("worker-copilot-1"),
            "worker-copilot-2": ShutdownPeer("worker-copilot-2"),
        }
    )
    emullm_api._worker_last_busy_at["worker-copilot-2"] = time.monotonic()  # noqa: SLF001
    monkeypatch.setattr(
        emullm_api._copilot_api,  # noqa: SLF001
        "get_manager",
        lambda: FakeManager(),
    )

    response = client.post("/emullm/admin/copilots/bulk/stop-idle")

    assert response.status_code == 200
    assert response.json()["affected"] == 1
    assert response.json()["idle_maintenance_paused"] is True
    assert "worker-copilot-1" not in emullm_api._connected_workers  # noqa: SLF001
    assert "worker-copilot-2" in emullm_api._connected_workers  # noqa: SLF001


def test_bulk_restart_runs_workers_in_concurrent_rolling_batches(
    client: TestClient,
    monkeypatch,
) -> None:
    instances = [
        {
            "worker_id": f"worker-copilot-{number}",
            "running": True,
            "connected": True,
        }
        for number in range(1, 10)
    ]

    class FakeManager:
        started: list[str] = []

        @staticmethod
        def list():
            return instances

        @classmethod
        def start(cls, worker_id: str):
            cls.started.append(worker_id)
            return {"worker_id": worker_id, "started": True}

    active_shutdowns = 0
    max_shutdowns = 0

    async def shutdown(worker_id: str, _reason: str) -> bool:
        nonlocal active_shutdowns, max_shutdowns
        active_shutdowns += 1
        max_shutdowns = max(max_shutdowns, active_shutdowns)
        await asyncio.sleep(0.01)
        active_shutdowns -= 1
        return True

    async def offline(_manager, _worker_id: str) -> None:
        return None

    async def connected(_worker_id: str, _timeout_seconds: float = 60.0) -> bool:
        return True

    monkeypatch.setattr(emullm_api._copilot_api, "get_manager", lambda: FakeManager())  # noqa: SLF001
    monkeypatch.setattr(emullm_api, "_shutdown_connected_worker", shutdown)
    monkeypatch.setattr(emullm_api, "_wait_for_managed_worker_offline", offline)
    monkeypatch.setattr(emullm_api, "_wait_for_connected_worker", connected)

    response = client.post(
        "/emullm/admin/copilots/bulk/restart",
        params={"batch_size": 4},
    )

    assert response.status_code == 200
    assert response.json()["affected"] == 9
    assert response.json()["batch_size"] == 4
    assert response.json()["batches"] == 3
    assert max_shutdowns == 4
    assert len(FakeManager.started) == 9


def test_admin_health_is_lightweight_and_state_avoids_full_mailbox_reads(
    client: TestClient,
    monkeypatch,
) -> None:
    health = client.get("/emullm/admin/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ready"
    assert isinstance(health.json()["connected_workers"], int)

    def fail_directory(*_args, **_kwargs):
        raise AssertionError("admin state must not enumerate mailbox histories")

    monkeypatch.setattr(emullm_api, "_mailbox_directory", fail_directory)
    state = client.get("/emullm/admin/state")
    assert state.status_code == 200
    assert isinstance(state.json()["mailboxes"]["count"], int)


def test_mailbox_summary_counts_jsonl_without_deserializing_history(
    client: TestClient,
) -> None:
    path = emullm_api._mailbox_event_log_path("summary-worker")  # noqa: SLF001
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"id":"one","ts":"2026-01-01T00:00:00Z"}\n'
        "\n"
        '{"id":"two","ts":"2026-01-02T00:00:00Z"}\n',
        encoding="utf-8",
    )

    assert emullm_api._mailbox_event_summary("summary-worker") == (  # noqa: SLF001
        2,
        "2026-01-02T00:00:00Z",
    )
    assert emullm_api._mailbox_event_count() >= 2  # noqa: SLF001


@pytest.mark.parametrize("path", ["/v1/files", "/v1/assistants", "/v1/threads"])
def test_durable_resource_list_then_create(client: TestClient, path: str) -> None:
    empty = client.get(path).json()
    assert empty["object"] == "list"
    assert empty["data"] == []
    created = client.post(path, json={"note": "test"}).json()
    assert created["id"]
    listed = client.get(path).json()["data"]
    assert any(item["id"] == created["id"] for item in listed)


@pytest.mark.parametrize(
    ("path", "body", "deleted_object"),
    [
        ("/v1/assistants", {"model": "yourself/same", "name": "helper"}, "assistant.deleted"),
        ("/v1/threads", {"metadata": {"topic": "testing"}}, "thread.deleted"),
    ],
)
def test_platform_resources_support_crud_and_protected_fields(
    client: TestClient,
    path: str,
    body: dict[str, object],
    deleted_object: str,
) -> None:
    created = client.post(path, json={**body, "id": "client-id", "object": "wrong"}).json()
    resource_id = created["id"]
    assert resource_id != "client-id"
    assert client.get(f"{path}/{resource_id}").json() == created

    modified = client.post(
        f"{path}/{resource_id}",
        json={"metadata": {"updated": True}, "id": "replacement", "created_at": 0},
    ).json()
    assert modified["id"] == resource_id
    assert modified["created_at"] == created["created_at"]
    assert modified["metadata"] == {"updated": True}

    deleted = client.delete(f"{path}/{resource_id}")
    assert deleted.json() == {"id": resource_id, "object": deleted_object, "deleted": True}
    assert client.get(f"{path}/{resource_id}").status_code == 404


def test_platform_resource_lists_are_cursor_paginated(client: TestClient) -> None:
    created = [
        client.post("/v1/threads", json={"metadata": {"index": index}}).json()
        for index in range(3)
    ]

    first_page = client.get("/v1/threads", params={"order": "asc", "limit": 2}).json()
    assert [item["id"] for item in first_page["data"]] == [created[0]["id"], created[1]["id"]]
    assert first_page["has_more"] is True
    second_page = client.get(
        "/v1/threads",
        params={"order": "asc", "limit": 2, "after": first_page["last_id"]},
    ).json()
    assert [item["id"] for item in second_page["data"]] == [created[2]["id"]]
    assert second_page["has_more"] is False


def test_fine_tuning_jobs_validate_files_and_expose_terminal_lifecycle(client: TestClient) -> None:
    bad = client.post(
        "/v1/fine_tuning/jobs",
        json={"model": "yourself/same", "training_file": "file-missing"},
    )
    assert bad.status_code == 404

    uploaded = client.post(
        "/v1/files",
        data={"purpose": "fine-tune"},
        files={
            "file": (
                "training.jsonl",
                b'{"messages":[{"role":"user","content":"hello"}]}\n',
                "application/jsonl",
            )
        },
    ).json()
    created = client.post(
        "/v1/fine_tuning/jobs",
        json={"model": "yourself/same", "training_file": uploaded["id"]},
    )

    assert created.status_code == 200
    job = created.json()
    assert job["status"] == "failed"
    assert job["error"]["code"] == "training_not_available"
    assert client.get(f"/v1/fine_tuning/jobs/{job['id']}").json() == job
    events = client.get(f"/v1/fine_tuning/jobs/{job['id']}/events").json()
    assert events["data"][0]["level"] == "error"
    assert client.get(f"/v1/fine_tuning/jobs/{job['id']}/checkpoints").json()["data"] == []
    assert client.post(f"/v1/fine_tuning/jobs/{job['id']}/cancel").status_code == 409


def test_crud_stub_persists_to_a_json_file_per_record(client: TestClient) -> None:
    created = client.post("/v1/files", json={"note": "durable"}).json()
    record_path = emullm_api._files_store._dir / f"{created['id']}.json"
    assert record_path.exists()
    on_disk = json.loads(record_path.read_text(encoding="utf-8"))
    assert on_disk["note"] == "durable"


def test_files_multipart_upload_retrieve_content_and_delete(client: TestClient) -> None:
    content = b'{"messages": [{"role": "user", "content": "hello"}]}\n'
    response = client.post(
        "/v1/files",
        data={"purpose": "fine-tune"},
        files={"file": ("training.jsonl", content, "application/jsonl")},
    )

    assert response.status_code == 200
    created = response.json()
    assert created["object"] == "file"
    assert created["filename"] == "training.jsonl"
    assert created["purpose"] == "fine-tune"
    assert created["bytes"] == len(content)
    assert created["status"] == "processed"

    file_id = created["id"]
    assert client.get(f"/v1/files/{file_id}").json() == created
    assert client.get(f"/v1/files/{file_id}/content").content == content
    assert created in client.get("/v1/files").json()["data"]

    deleted = client.delete(f"/v1/files/{file_id}")
    assert deleted.json() == {"id": file_id, "object": "file", "deleted": True}
    assert client.get(f"/v1/files/{file_id}").status_code == 404
    assert client.get(f"/v1/files/{file_id}/content").status_code == 404


def test_files_upload_validates_required_multipart_fields(client: TestClient) -> None:
    missing_file = client.post(
        "/v1/files",
        data={"purpose": "fine-tune"},
        files={"unused": ("unused.txt", b"x", "text/plain")},
    )
    assert missing_file.status_code == 400

    missing_purpose = client.post(
        "/v1/files",
        files={"file": ("training.jsonl", b"{}\n", "application/jsonl")},
    )
    assert missing_purpose.status_code == 400


def test_files_admin_reset_removes_metadata_and_content(client: TestClient) -> None:
    created = client.post(
        "/v1/files",
        data={"purpose": "user_data"},
        files={"file": ("notes.txt", b"remember me", "text/plain")},
    ).json()
    content_path = emullm_api._files_store.content_path(created["id"])  # noqa: SLF001
    assert content_path.is_file()

    response = client.post("/admin/emullm/reset")

    assert response.status_code == 200
    assert response.json()["removed"]["files"] == 1
    assert not content_path.exists()


def test_files_list_supports_filtering_order_and_cursor_pagination(client: TestClient) -> None:
    first = client.post("/v1/files", json={"filename": "a.txt", "purpose": "assistants"}).json()
    second = client.post("/v1/files", json={"filename": "b.txt", "purpose": "fine-tune"}).json()
    third = client.post("/v1/files", json={"filename": "c.txt", "purpose": "fine-tune"}).json()

    ascending = client.get("/v1/files", params={"order": "asc", "limit": 2}).json()
    assert [record["id"] for record in ascending["data"]] == [first["id"], second["id"]]
    assert ascending["first_id"] == first["id"]
    assert ascending["last_id"] == second["id"]
    assert ascending["has_more"] is True

    next_page = client.get(
        "/v1/files",
        params={"order": "asc", "after": ascending["last_id"], "limit": 2},
    ).json()
    assert [record["id"] for record in next_page["data"]] == [third["id"]]
    assert next_page["has_more"] is False

    filtered = client.get("/v1/files", params={"purpose": "fine-tune", "order": "asc"}).json()
    assert [record["id"] for record in filtered["data"]] == [second["id"], third["id"]]


def test_files_enforce_upload_limit_and_clean_temporary_data(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(emullm_api, "_MAX_FILE_BYTES", 4)

    response = client.post(
        "/v1/files",
        data={"purpose": "user_data"},
        files={"file": ("too-large.txt", b"12345", "text/plain")},
    )

    assert response.status_code == 413
    assert client.get("/v1/files").json()["data"] == []
    assert not list(emullm_api._files_store._dir.glob("*.tmp"))  # noqa: SLF001


def test_files_support_expiration_safe_names_and_range_downloads(client: TestClient) -> None:
    response = client.post(
        "/v1/files",
        data={
            "purpose": "user_data",
            "expires_after[anchor]": "created_at",
            "expires_after[seconds]": "60",
        },
        files={"file": ("../notes.txt", b"abcdef", "text/plain")},
    )
    created = response.json()

    assert response.status_code == 200
    assert created["filename"] == "notes.txt"
    assert created["expires_at"] == created["created_at"] + 60

    partial = client.get(
        f"/v1/files/{created['id']}/content",
        headers={"Range": "bytes=1-3"},
    )
    assert partial.status_code == 206
    assert partial.content == b"bcd"


def test_storage_round_trip_put_get_delete(client: TestClient) -> None:
    assert client.get("/emullm/storage").json() == {"files": []}
    assert client.get("/emullm/storage/notes/todo.txt").status_code == 404

    put_response = client.put("/emullm/storage/notes/todo.txt", content=b"remember this")
    assert put_response.status_code == 200
    assert put_response.json() == {"path": "notes/todo.txt", "bytes": len(b"remember this")}

    assert client.get("/emullm/storage").json() == {"files": ["notes/todo.txt"]}
    get_response = client.get("/emullm/storage/notes/todo.txt")
    assert get_response.status_code == 200
    assert get_response.content == b"remember this"

    delete_response = client.delete("/emullm/storage/notes/todo.txt")
    assert delete_response.status_code == 200
    assert client.get("/emullm/storage/notes/todo.txt").status_code == 404


def test_storage_rejects_path_traversal(client: TestClient) -> None:
    # The HTTP client normalizes ".." before it's even sent in some cases,
    # so either FastAPI's routing 404s on the resulting path, or our own
    # _safe_storage_path guard rejects it with 400 -- both are acceptable,
    # the important thing is neither one ever escapes the storage root.
    assert client.get("/emullm/storage/../../etc/passwd").status_code in (400, 404)
    assert client.put("/emullm/storage/../escape.txt", content=b"x").status_code in (400, 404)


def test_admin_state_reports_connected_workers_and_usage(client: TestClient) -> None:
    emullm_api._connected_workers["alice"] = FakeWorker(reply="ok")  # noqa: SLF001
    asyncio.run(emullm_api._relay("alice/same", "hi"))

    state = client.get("/admin/emullm/state").json()

    assert "alice" in state["connected_worker_ids"]
    assert state["worker_usage"]["alice"]["total_requests"] == 1


def test_admin_state_reports_mode_and_default_role(client: TestClient) -> None:
    emullm_api._connected_workers["alice"] = FakeWorker()  # noqa: SLF001
    state = client.get("/admin/emullm/state").json()
    assert "mode" in state
    assert state["uptime_seconds"] >= 0
    # A connected worker with no declared role reports the default.
    assert state["worker_roles"]["alice"] == "trusted"


def test_admin_state_reports_declared_worker_role(client: TestClient) -> None:
    emullm_api._connected_workers["bob"] = FakeWorker()  # noqa: SLF001
    emullm_api._worker_roles["bob"] = "training"  # noqa: SLF001
    state = client.get("/admin/emullm/state").json()
    assert state["worker_roles"]["bob"] == "training"


def test_status_pages_render_html(client: TestClient) -> None:
    for url in ("/emullm/status", "/emullm/status/detail", "/admin/emullm/status"):
        response = client.get(url)
        assert response.status_code == 200, url
        assert "text/html" in response.headers["content-type"]
        assert "emullm status" in response.text
    # The detail view injects DETAIL = true; the overview injects false.
    detail = client.get("/emullm/status/detail").text
    overview = client.get("/emullm/status").text
    assert "const DETAIL = true" in detail
    assert "const DETAIL = false" in overview
    for html in (detail, overview):
        assert 'id="poll-window"' in html
        assert 'id="poll-hidden"' in html
        assert 'id="poll-wake"' in html
        assert "document.addEventListener('visibilitychange'" in html
        assert "POLL_HIDDEN_MS = 120000" in html
        assert "setInterval(refresh, 3000)" not in html


def test_managed_workers_empty_without_supervisor(client: TestClient) -> None:
    state = client.get("/admin/emullm/state").json()
    assert state["managed_workers"] == []
    listing = client.get("/admin/emullm/workers").json()
    assert listing["supervisor_active"] is False
    assert listing["workers"] == []


def test_worker_control_endpoints_start_and_stop(client: TestClient) -> None:
    from emullm import supervisor as sup

    class FakeProc:
        def __init__(self) -> None:
            self.pid = 777
            self._alive = True

        def poll(self):
            return None if self._alive else 0

        def terminate(self):
            self._alive = False

        def wait(self, timeout=None):
            self._alive = False
            return 0

    supervisor = sup.Supervisor(
        [sup.WorkerSpec(worker_id="emullm_worker_1", argv=["x"], role="training")],
        spawn=lambda spec: FakeProc(),
    )
    emullm_api._sup.set_supervisor(supervisor)  # noqa: SLF001
    try:
        listing = client.get("/admin/emullm/workers").json()
        assert listing["supervisor_active"] is True
        assert listing["workers"][0]["worker_id"] == "emullm_worker_1"
        assert listing["workers"][0]["running"] is False

        started = client.post("/admin/emullm/workers/emullm_worker_1/start").json()
        assert started["started"] is True
        assert started["workers"][0]["running"] is True

        # It also shows up in the aggregate state.
        state = client.get("/admin/emullm/state").json()
        assert state["managed_workers"][0]["running"] is True

        stopped = client.post("/admin/emullm/workers/emullm_worker_1/stop").json()
        assert stopped["stopped"] is True
        assert stopped["workers"][0]["running"] is False

        # Unknown worker -> 404.
        assert client.post("/admin/emullm/workers/nope/start").status_code == 404
    finally:
        emullm_api._sup.set_supervisor(None)  # noqa: SLF001


def test_worker_control_409_without_supervisor(client: TestClient) -> None:
    assert client.post("/admin/emullm/workers/x/start").status_code == 409
    assert client.post("/admin/emullm/workers/x/stop").status_code == 409


def test_config_get_default_is_empty(client: TestClient) -> None:
    result = client.get("/admin/emullm/config").json()
    assert result["config"] == {}
    assert result["path"].endswith("config.json")


def test_config_put_then_get_round_trip(client: TestClient) -> None:
    payload = {
        "config": {"mode": "auto", "workers": [{"id": "w1", "role": "training"}]},
        "expected_revision": client.get("/admin/emullm/config").json()["revision"],
    }
    saved = client.put("/admin/emullm/config", json=payload)
    assert saved.status_code == 200
    assert saved.json()["saved"] is True
    assert client.get("/admin/emullm/config").json()["config"] == payload["config"]
    # alias behaves identically
    assert client.get("/emullm/admin/config").json()["config"] == payload["config"]


def test_config_put_rejects_non_object(client: TestClient) -> None:
    # `config` must be a JSON object; a list is rejected by validation.
    assert client.put("/admin/emullm/config", json={"config": [1, 2, 3]}).status_code == 422


def test_config_put_rejects_unknown_top_level_key(client: TestClient) -> None:
    # Unknown top-level keys are rejected so a hand-edited file catches typos.
    resp = client.put("/admin/emullm/config", json={"config": {"moed": "mock"}})
    assert resp.status_code == 422


def test_config_put_rejects_bad_launch_enum(client: TestClient) -> None:
    payload = {"config": {"agents": [{"kind": "agent", "id": "a", "launch": "nope"}]}}
    assert client.put("/admin/emullm/config", json=payload).status_code == 422


def test_config_put_accepts_unified_agents(client: TestClient) -> None:
    payload = {
        "config": {
            "description": "test cluster",
            "mode": "recruit,mock",
            "capability_fallback": "wait",
            "agents": [
                {
                    "kind": "agent",
                    "id": "alice",
                    "launch": "subagent",
                    "command": "copilot",
                    "description": "spawned copilot",
                    "observe": ["chat"],
                    "services": {
                        "chat": "serve",
                        "images": {"behavior": "error", "description": "not offered"},
                    },
                },
            ],
        },
        "expected_revision": client.get("/admin/emullm/config").json()["revision"],
    }
    saved = client.put("/admin/emullm/config", json=payload)
    assert saved.status_code == 200, saved.text
    assert client.get("/admin/emullm/config").json()["config"] == payload["config"]


def test_config_schema_endpoint(client: TestClient) -> None:
    schema = client.get("/admin/emullm/config/schema").json()
    props = schema["properties"]
    assert "agents" in props and "services" in props and "capability_fallback" in props
    assert "headless_copilots" in props
    assert "codex_suppliers" in props
    assert "anti_idle" in props
    # alias serves the same schema
    assert client.get("/emullm/admin/config/schema").json() == schema


def test_config_section_editor_preserves_unrelated_config_and_validates(
    client: TestClient,
) -> None:
    emullm_api._CONFIG_PATH.write_text(  # noqa: SLF001
        json.dumps(
            {
                "mode": "recruit,mock",
                "services": {"model": "old"},
                "backends": [{"name": "legacy", "base_url": "http://example/v1"}],
            }
        ),
        encoding="utf-8",
    )
    revision = client.get("/emullm/admin/config").json()["revision"]
    saved = client.put(
        "/emullm/admin/config/section/services",
        json={
            "value": {"model": "new", "embeddings": {"fallback": "stub"}},
            "expected_revision": revision,
        },
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["config"]["mode"] == "recruit,mock"
    assert saved.json()["value"]["model"] == "new"

    invalid = client.put(
        "/emullm/admin/config/section/agents",
        json={
            "value": {"not": "a list"},
            "expected_revision": saved.json()["revision"],
        },
    )
    assert invalid.status_code == 422
    assert client.get("/emullm/admin/config").json()["config"]["services"]["model"] == "new"

    deleted = client.put(
        "/admin/emullm/config/section/backends",
        json={"delete": True, "expected_revision": saved.json()["revision"]},
    )
    assert deleted.status_code == 200
    assert "backends" not in deleted.json()["config"]
    assert client.put(
        "/emullm/admin/config/section/nope",
        json={"value": {}, "expected_revision": deleted.json()["revision"]},
    ).status_code == 404


def test_config_section_and_server_settings_updates_are_strict_and_atomic(
    client: TestClient,
) -> None:
    emullm_api._CONFIG_PATH.write_text(  # noqa: SLF001
        json.dumps(
            {
                "mode": "mock",
                "services": {"model": "old"},
                "codex_suppliers": [emullm_api._DEFAULT_CODEX_SUPPLIER],  # noqa: SLF001
            }
        ),
        encoding="utf-8",
    )
    settings = client.put(
        "/emullm/admin/config/server-settings",
        json={
            "description": "patched",
            "mode": "recruit,mock",
            "capability_fallback": "stub",
            "subagent_model": None,
            "max_concurrent_calls": 50,
            "idle_worker_target": 10,
            "idle_grace_seconds": 120,
            "backend_fallback_delay_seconds": 5,
            "validation_interval_default": "never",
            "validation_interval_override": None,
            "expected_revision": client.get("/emullm/admin/config").json()[
                "revision"
            ],
        },
    )
    assert settings.status_code == 200, settings.text
    assert settings.json()["config"]["services"] == {"model": "old"}
    assert settings.json()["config"]["codex_suppliers"][0]["id"] == "copilot"

    malformed = '{"mode": '
    emullm_api._CONFIG_PATH.write_text(malformed, encoding="utf-8")  # noqa: SLF001
    section = client.put(
        "/emullm/admin/config/section/services",
        json={"value": {"model": "new"}, "expected_revision": "stale"},
    )
    assert section.status_code == 409
    assert emullm_api._CONFIG_PATH.read_text(encoding="utf-8") == malformed  # noqa: SLF001


def test_anti_idle_prompt_catalog_can_grow_deprecate_and_aggregate_stats(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anti_idle = emullm_api._copilot_api.AntiIdleConfig()  # noqa: SLF001
    emullm_api._CONFIG_PATH.write_text(  # noqa: SLF001
        json.dumps({"anti_idle": anti_idle.model_dump(mode="json")}),
        encoding="utf-8",
    )

    class Manager:
        @staticmethod
        def list():
            return [
                {
                    "worker_id": "worker-fast",
                    "runtime": {
                        "keepalive_task_stats": {
                            "conversation-01": {
                                "attempts": 2,
                                "completed": 2,
                                "slow": 0,
                                "timeouts": 0,
                                "total_duration_ms": 1_000,
                                "min_duration_ms": 400,
                                "max_duration_ms": 600,
                            }
                        },
                        "retired_keepalive_tasks": [],
                    }
                },
                {
                    "worker_id": "worker-slow",
                    "runtime": {
                        "keepalive_task_stats": {
                            "conversation-01": {
                                "attempts": 1,
                                "completed": 0,
                                "slow": 1,
                                "timeouts": 1,
                                "total_duration_ms": 2_000,
                                "min_duration_ms": 2_000,
                                "max_duration_ms": 2_000,
                            }
                        },
                        "retired_keepalive_tasks": ["conversation-01"],
                    }
                },
            ]

    monkeypatch.setattr(
        emullm_api._copilot_api,  # noqa: SLF001
        "get_manager",
        lambda: Manager(),
    )
    listing = client.get("/emullm/admin/anti-idle").json()
    assert len(listing["prompts"]) == 50
    first = listing["prompts"][0]
    assert first["number"] == 1
    assert first["attempts"] == 3
    assert first["average_duration_ms"] == 1_000
    assert first["min_duration_ms"] == 400
    assert first["shortest_worker_id"] == "worker-fast"
    assert first["max_duration_ms"] == 2_000
    assert first["longest_worker_id"] == "worker-slow"
    assert first["retired_workers"] == 1

    prompts = listing["config"]["prompts"]
    prompts[0]["deprecated"] = True
    prompts.append(
        {
            "id": "conversation-51",
            "prompt": "What kind of request would be fun to answer next?",
            "deprecated": False,
        }
    )
    saved = client.put(
        "/emullm/admin/anti-idle",
        json={
            "expected_revision": listing["revision"],
            "config": {
                "enabled": True,
                "interval_seconds": 60,
                "timeout_seconds": 2.5,
                "slow_budget_seconds": 2,
                "prompts": prompts,
            },
        },
    )
    assert saved.status_code == 200, saved.text
    assert len(saved.json()["config"]["prompts"]) == 51
    assert saved.json()["config"]["prompts"][0]["deprecated"] is True
    assert saved.json()["config"]["interval_seconds"] == 60
    assert saved.json()["config"]["slow_budget_seconds"] == 2

    stale = client.put(
        "/emullm/admin/anti-idle",
        json={
            "expected_revision": listing["revision"],
            "config": saved.json()["config"],
        },
    )
    assert stale.status_code == 409


def test_anti_idle_enabled_toggle_persists_and_updates_connected_workers(
    client: TestClient,
) -> None:
    class ToggleWorker:
        messages: list[dict] = []

        async def send_json(self, payload):
            self.messages.append(payload)
            await emullm_api._handle_worker_message(  # noqa: SLF001
                "toggle-worker",
                {
                    "type": "anti_idle_changed",
                    "id": payload["id"],
                    "enabled": payload["enabled"],
                },
            )

    worker = ToggleWorker()
    emullm_api._connected_workers["toggle-worker"] = worker  # noqa: SLF001
    listing = client.get("/emullm/admin/anti-idle").json()

    response = client.put(
        "/emullm/admin/anti-idle/enabled",
        json={
            "enabled": False,
            "expected_revision": listing["revision"],
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["restart_required"] is False
    assert response.json()["updated_workers"] == 1
    assert response.json()["failed_workers"] == 0
    assert worker.messages[0]["type"] == "set_anti_idle"
    assert worker.messages[0]["enabled"] is False
    assert client.get("/emullm/admin/anti-idle").json()["config"]["enabled"] is False


def test_reset_anti_idle_stats_clears_offline_worker_files(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Manager:
        cleared: list[str] = []

        @staticmethod
        def list():
            return [{"worker_id": "offline-worker", "connected": False}]

        @classmethod
        def clear_keepalive_stats(cls, worker_id):
            cls.cleared.append(worker_id)
            return True

    monkeypatch.setattr(
        emullm_api._copilot_api,  # noqa: SLF001
        "get_manager",
        lambda: Manager(),
    )
    response = client.post("/emullm/admin/anti-idle/reset-stats")
    assert response.status_code == 200
    assert response.json()["reset"] == 1
    assert Manager.cleared == ["offline-worker"]


def test_reset_anti_idle_stats_continues_after_connected_worker_error(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Manager:
        cleared: list[str] = []

        @staticmethod
        def list():
            return [
                {"worker_id": "broken-worker", "connected": True},
                {"worker_id": "offline-worker", "connected": False},
            ]

        @classmethod
        def clear_keepalive_stats(cls, worker_id):
            cls.cleared.append(worker_id)
            return True

    async def fail_send(*_args, **_kwargs):
        raise OSError("disconnected")

    monkeypatch.setattr(
        emullm_api._copilot_api,  # noqa: SLF001
        "get_manager",
        lambda: Manager(),
    )
    monkeypatch.setattr(emullm_api, "_send_worker_json", fail_send)
    emullm_api._connected_workers["broken-worker"] = object()  # noqa: SLF001

    response = client.post("/emullm/admin/anti-idle/reset-stats")

    assert response.status_code == 200
    assert response.json()["reset"] == 1
    assert response.json()["results"][0]["error"] == "disconnected"
    assert Manager.cleared == ["offline-worker"]


def test_backend_config_page_crud_preserves_proxy_agent_metadata(
    client: TestClient,
) -> None:
    emullm_api._CONFIG_PATH.write_text(  # noqa: SLF001
        json.dumps(
            {
                "agents": [
                    {
                        "kind": "agent",
                        "id": "openai",
                        "launch": "proxy",
                        "name": "snet",
                        "base_url": "https://old.example/v1",
                        "api_key": "stored-secret",
                        "default": True,
                        "services": {"chat": {"behavior": "aggregate"}},
                    },
                    {"kind": "agent", "id": "mock", "launch": "mock"},
                ]
            }
        ),
        encoding="utf-8",
    )

    listing = client.get("/emullm/admin/backends/configured").json()
    assert listing["count"] == 1
    assert listing["backends"][0]["source"] == "agents"
    assert listing["backends"][0]["has_api_key"] is True
    assert "api_key" not in listing["backends"][0]

    updated = client.put(
        "/emullm/admin/backends/configured/agents/0",
        json={
            "name": "snet",
            "base_url": "https://new.example/v1/",
            "description": "updated",
            "api_key_env": "SNET_API_KEY",
            "model": "vendor/model",
            "default": True,
            "validation_interval": "1day",
            "expected_revision": listing["backends"][0]["revision"],
        },
    )
    assert updated.status_code == 200, updated.text
    persisted = json.loads(emullm_api._CONFIG_PATH.read_text(encoding="utf-8"))  # noqa: SLF001
    proxy = persisted["agents"][0]
    assert proxy["base_url"] == "https://new.example/v1"
    assert proxy["api_key"] == "stored-secret"
    assert proxy["services"] == {"chat": {"behavior": "aggregate"}}
    assert proxy["id"] == "openai"

    created = client.post(
        "/emullm/admin/backends/configured",
        json={
            "name": "backup",
            "base_url": "http://backup.example/v1",
            "model": "backup/model",
            "default": True,
        },
    )
    assert created.status_code == 201, created.text
    assert created.json()["backend"]["source"] == "backends"
    persisted = json.loads(emullm_api._CONFIG_PATH.read_text(encoding="utf-8"))  # noqa: SLF001
    assert persisted["agents"][0]["default"] is False
    assert persisted["backends"][0]["default"] is True

    duplicate = client.post(
        "/emullm/admin/backends/configured",
        json={"name": "backup", "base_url": "https://duplicate.example/v1"},
    )
    assert duplicate.status_code == 409
    backup = next(
        backend
        for backend in created.json()["backends"]
        if backend["name"] == "backup"
    )
    assert client.delete(
        "/emullm/admin/backends/configured/backends/0",
        params={
            "expected_name": "backup",
            "expected_revision": backup["revision"],
        },
    ).status_code == 200
    assert client.get("/emullm/admin/backends/configured").json()["count"] == 1


def test_backend_config_rejects_stale_edits_and_preserves_malformed_file(
    client: TestClient,
) -> None:
    emullm_api._CONFIG_PATH.write_text(  # noqa: SLF001
        json.dumps(
            {
                "backends": [
                    {"name": "first", "base_url": "https://first.example/v1"},
                    {"name": "second", "base_url": "https://second.example/v1"},
                ]
            }
        ),
        encoding="utf-8",
    )
    stale = client.put(
        "/emullm/admin/backends/configured/backends/0",
        json={
            "name": "renamed",
            "expected_name": "second",
            "expected_revision": client.get(
                "/emullm/admin/backends/configured"
            ).json()["backends"][0]["revision"],
            "base_url": "https://renamed.example/v1",
        },
    )
    assert stale.status_code == 409
    assert json.loads(emullm_api._CONFIG_PATH.read_text(encoding="utf-8"))[  # noqa: SLF001
        "backends"
    ][0]["name"] == "first"

    listing = client.get("/emullm/admin/backends/configured").json()
    first = listing["backends"][0]
    persisted = json.loads(emullm_api._CONFIG_PATH.read_text(encoding="utf-8"))  # noqa: SLF001
    persisted["backends"][0]["model"] = "changed/model"
    emullm_api._CONFIG_PATH.write_text(json.dumps(persisted), encoding="utf-8")  # noqa: SLF001
    same_name_stale = client.put(
        "/emullm/admin/backends/configured/backends/0",
        json={
            "name": "first",
            "expected_name": "first",
            "expected_revision": first["revision"],
            "base_url": "https://replacement.example/v1",
        },
    )
    assert same_name_stale.status_code == 409

    malformed = '{"backends": ['
    emullm_api._CONFIG_PATH.write_text(malformed, encoding="utf-8")  # noqa: SLF001
    response = client.post(
        "/emullm/admin/backends/configured",
        json={"name": "new", "base_url": "https://new.example/v1"},
    )
    assert response.status_code == 409
    assert emullm_api._CONFIG_PATH.read_text(encoding="utf-8") == malformed  # noqa: SLF001


def test_codex_supplier_page_defaults_to_copilot_and_supports_crud(
    client: TestClient,
) -> None:
    initial = client.get("/emullm/admin/codex-suppliers").json()
    assert initial["count"] == 1
    assert initial["suppliers"][0]["id"] == "copilot"
    assert initial["suppliers"][0]["revision"]

    created = client.post(
        "/emullm/admin/codex-suppliers",
        json={
            "id": "remote-codex",
            "name": "Remote Codex",
            "kind": "openai-compatible",
            "enabled": True,
            "worker_pattern": "worker-codex-*",
            "model_prefix": "remote/",
            "model_patterns": ["codex-*"],
            "base_url": "https://codex.example/v1",
            "api_key_env": "CODEX_API_KEY",
            "provider_extension": {"region": "test"},
        },
    )
    assert created.status_code == 201, created.text
    assert [supplier["id"] for supplier in created.json()["suppliers"]] == [
        "copilot",
        "remote-codex",
    ]

    copilot = dict(emullm_api._DEFAULT_CODEX_SUPPLIER)  # noqa: SLF001
    copilot["enabled"] = False
    updated = client.put(
        "/emullm/admin/codex-suppliers/copilot",
        json=copilot,
        params={"expected_revision": initial["suppliers"][0]["revision"]},
    )
    assert updated.status_code == 200
    assert updated.json()["supplier"]["enabled"] is False
    remote = next(
        supplier
        for supplier in updated.json()["suppliers"]
        if supplier["id"] == "remote-codex"
    )
    assert remote["provider_extension"] == {"region": "test"}
    edited_remote = {
        key: value
        for key, value in remote.items()
        if key not in {"revision", "provider_extension"}
    }
    edited_remote["description"] = "edited in known-fields UI"
    preserved = client.put(
        "/emullm/admin/codex-suppliers/remote-codex",
        json=edited_remote,
        params={"expected_revision": remote["revision"]},
    )
    assert preserved.status_code == 200
    assert preserved.json()["supplier"]["provider_extension"] == {"region": "test"}
    assert client.delete(
        "/emullm/admin/codex-suppliers/remote-codex",
        params={
            "expected_revision": preserved.json()["supplier"]["revision"],
        },
    ).status_code == 200
    assert client.get("/emullm/admin/codex-suppliers").json()["count"] == 1


def test_codex_catalog_model_reports_configured_copilot_supplier() -> None:
    entry = emullm_api._copilot_catalog_model_entry(  # noqa: SLF001
        {"id": "gpt-5.3-codex", "name": "GPT-5.3-Codex"}
    )
    assert entry["codex_supplier"] == "copilot"


def test_codex_supplier_prefers_priority_then_specific_pattern() -> None:
    emullm_api._CONFIG_PATH.write_text(  # noqa: SLF001
        json.dumps(
            {
                "codex_suppliers": [
                    {
                        **emullm_api._DEFAULT_CODEX_SUPPLIER,  # noqa: SLF001
                        "priority": 0,
                    },
                    {
                        "id": "specific",
                        "name": "Specific",
                        "kind": "custom",
                        "enabled": True,
                        "priority": 0,
                        "model_patterns": ["gpt-5.3-*"],
                    },
                    {
                        "id": "priority",
                        "name": "Priority",
                        "kind": "custom",
                        "enabled": True,
                        "priority": 10,
                        "model_patterns": ["*codex*"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    assert (
        emullm_api._codex_supplier_for_model("gpt-5.3-codex")["id"]  # noqa: SLF001
        == "priority"
    )


def test_standalone_process_control_endpoints(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(emullm_api._process_control, "schedule_shutdown", lambda: True)  # noqa: SLF001
    monkeypatch.setattr(  # noqa: SLF001
        emullm_api._process_control,
        "schedule_restart",
        lambda host, port: 4242,
    )
    shutdown = client.post("/emullm/admin/shutdown")
    assert shutdown.status_code == 202
    assert shutdown.json()["status"] == "shutting_down"

    restart = client.post("/admin/emullm/restart")
    assert restart.status_code == 202
    assert restart.json()["status"] == "restarting"
    assert restart.json()["helper_pid"] == 4242


def test_emullm_runtime_layout_and_config_seeding(tmp_path, monkeypatch) -> None:
    """The emullm_runtime container is created on first run, holds
    config/logs/metrics/state, seeds server_config.json from the shipped default,
    and honours env overrides regardless of where the package is installed."""
    from emullm import paths

    for var in (
        "EMULLM_RUNTIME_DIR",
        "EMULLM_CONFIG_DIR",
        "EMULLM_CONFIG_FILE",
        "EMULLM_LOG_DIR",
        "EMULLM_DATA_DIR",
        "EMULLM_STATE_DIR",
    ):
        monkeypatch.delenv(var, raising=False)

    root = tmp_path / "emullm_runtime"
    rt = paths.EmullmRuntime(root=root)

    # Everything the server creates lives under the single container.
    assert rt.config_file == root / "config" / "server_config.json"
    assert rt.logs_dir == root / "logs"

    # First run creates the layout and seeds config from the packaged default.
    seeded = rt.ensure_layout()
    assert seeded == root / "config" / "server_config.json"
    assert seeded.is_file()
    assert rt.dist_config_reference.is_file()
    assert rt.logs_dir.is_dir() and rt.metrics_dir.is_dir() and rt.state_dir.is_dir()
    doc = json.loads(seeded.read_text(encoding="utf-8"))
    assert "max_concurrent_calls" in doc  # came from the shipped default

    # Seeding is one-shot: existing settings are preserved on the next run.
    seeded.write_text('{"mode": "mock"}', encoding="utf-8")
    rt.ensure_layout()
    assert json.loads(seeded.read_text(encoding="utf-8")) == {"mode": "mock"}

    # An explicit config-file override always wins, independent of the container.
    override = tmp_path / "override.json"
    monkeypatch.setenv("EMULLM_CONFIG_FILE", str(override))
    assert paths.EmullmRuntime(root=root).config_file == override

    # EMULLM_RUNTIME_DIR relocates the whole container, wherever emullm is installed.
    monkeypatch.delenv("EMULLM_CONFIG_FILE", raising=False)
    monkeypatch.setenv("EMULLM_RUNTIME_DIR", str(tmp_path / "custom"))
    assert paths.EmullmRuntime().root == tmp_path / "custom"


def test_process_controls_reject_embedded_mode(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(emullm_api._process_control, "schedule_shutdown", lambda: False)  # noqa: SLF001
    monkeypatch.setattr(  # noqa: SLF001
        emullm_api._process_control,
        "schedule_restart",
        lambda host, port: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )
    assert client.post("/emullm/admin/shutdown").status_code == 409
    assert client.post("/emullm/admin/restart").status_code == 409


def test_admin_routes_reject_non_loopback_clients() -> None:
    app = FastAPI()
    app.include_router(emullm_api.router)
    remote = TestClient(app, client=("203.0.113.10", 50000))

    assert remote.get("/emullm/admin/config").status_code == 403
    assert remote.get("/admin/emullm/backends/configured").status_code == 403
    assert remote.get("/emullm/admin/copilots").status_code == 403
    assert remote.put(
        "/emullm/admin/anti-idle",
        json={"config": {}, "expected_revision": "irrelevant"},
    ).status_code == 403
    assert remote.get("/v1/models").status_code == 200
def test_carol_enabled_checkbox_api_persists_and_toggles_mock_live(
    client: TestClient,
) -> None:
    emullm_api._CONFIG_PATH.write_text(  # noqa: SLF001
        json.dumps(
            {
                "mode": "mock",
                "agents": [
                    {
                        "kind": "agent",
                        "id": "carol",
                        "enabled": False,
                        "launch": "mock",
                        "reply": "hello from carol",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    initial = client.get("/emullm/admin/agents").json()["agents"][0]
    assert initial["enabled"] is False
    assert initial["mock_registered"] is False

    enabled = client.put(
        "/emullm/admin/agents/carol/enabled", json={"enabled": True}
    )
    assert enabled.status_code == 200
    assert enabled.json()["applied"] is True
    assert enabled.json()["agents"][0]["mock_registered"] is True
    assert "carol" in emullm_api._connected_workers  # noqa: SLF001
    assert json.loads(emullm_api._CONFIG_PATH.read_text(encoding="utf-8"))["agents"][0]["enabled"] is True  # noqa: SLF001

    disabled = client.put(
        "/admin/emullm/agents/carol/enabled", json={"enabled": False}
    )
    assert disabled.status_code == 200
    assert disabled.json()["agents"][0]["mock_registered"] is False
    assert "carol" not in emullm_api._connected_workers  # noqa: SLF001


def test_backends_probe_reports_models(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        emullm_api,
        "_http_get_json",
        lambda url, headers, timeout=15.0: {"data": [{"id": "gpt-4o-mini"}, {"id": "gpt-4o"}]},
    )
    client.put(
        "/admin/emullm/config",
        json={"config": {"agents": [
            {"kind": "agent", "id": "openai", "launch": "proxy",
             "base_url": "http://backend.test/v1", "model": "gpt-4o-mini", "default": True}
        ]}, "expected_revision": client.get("/admin/emullm/config").json()["revision"]},
    )
    result = client.get("/admin/emullm/backends/probe").json()["backends"]
    assert result[0]["ok"] is True
    assert "gpt-4o-mini" in result[0]["models"]


def test_backends_probe_survives_unreachable_backend(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(url, headers, timeout=15.0):
        raise OSError("unreachable")

    monkeypatch.setattr(emullm_api, "_http_get_json", boom)
    client.put(
        "/admin/emullm/config",
        json={
            "config": {"backends": [{"name": "x", "base_url": "http://x/v1"}]},
            "expected_revision": client.get("/admin/emullm/config").json()["revision"],
        },
    )
    result = client.get("/admin/emullm/backends/probe").json()["backends"]
    assert result[0]["ok"] is False
    assert "error" in result[0]


def test_backends_probe_verify_flags_falsely_advertised(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeHTTPError(Exception):
        def __init__(self, code: int) -> None:
            super().__init__(f"HTTP {code}")
            self.code = code

    monkeypatch.setattr(
        emullm_api,
        "_http_get_json",
        lambda url, headers, timeout=15.0: {"data": [{"id": "live-model"}, {"id": "dead-model"}]},
    )

    def fake_post(url, headers, payload, timeout=60.0):
        if payload.get("model") == "live-model" and "chat/completions" in url:
            return {"choices": [{"message": {"content": "ok"}}]}
        raise FakeHTTPError(404)  # dead-model: not loaded

    def fake_raw(url, headers, payload, timeout=60.0):
        raise FakeHTTPError(404)  # audio speech: not offered

    monkeypatch.setattr(emullm_api, "_http_post_json", fake_post)
    monkeypatch.setattr(emullm_api, "_http_post_raw", fake_raw)
    client.put(
        "/admin/emullm/config",
        json={
            "config": {"backends": [{"name": "x", "base_url": "http://x/v1"}]},
            "expected_revision": client.get("/admin/emullm/config").json()["revision"],
        },
    )
    b = client.get("/admin/emullm/backends/probe?verify=true").json()["backends"][0]
    assert b["live"] == ["live-model"]
    assert b["falsely_advertised"] == ["dead-model"]


def test_backends_probe_verify_429_is_inconclusive_not_false(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeHTTPError(Exception):
        def __init__(self, code: int) -> None:
            super().__init__(f"HTTP {code}")
            self.code = code

    monkeypatch.setattr(
        emullm_api, "_http_get_json", lambda url, headers, timeout=15.0: {"data": [{"id": "busy-model"}]}
    )
    monkeypatch.setattr(emullm_api.time, "sleep", lambda *a, **k: None)  # skip real backoff waits

    def always_429(url, headers, payload, timeout=60.0):
        raise FakeHTTPError(429)

    monkeypatch.setattr(emullm_api, "_http_post_json", always_429)
    monkeypatch.setattr(emullm_api, "_http_post_raw", always_429)
    client.put(
        "/admin/emullm/config",
        json={
            "config": {"backends": [{"name": "x", "base_url": "http://x/v1"}]},
            "expected_revision": client.get("/admin/emullm/config").json()["revision"],
        },
    )
    b = client.get("/admin/emullm/backends/probe?verify=true&limit=1").json()["backends"][0]
    # rate-limited is inconclusive, never counted as falsely advertised
    assert b["falsely_advertised"] == []
    assert "busy-model" in b["inconclusive"]


def test_aggregate_validate_drops_dead_models(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeHTTPError(Exception):
        def __init__(self, code: int) -> None:
            super().__init__(f"HTTP {code}")
            self.code = code

    monkeypatch.setattr(emullm_api.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(
        emullm_api, "_http_get_json", lambda url, headers, timeout=15.0: {"data": [{"id": "good"}, {"id": "dead"}]}
    )

    def fake_post(url, headers, payload, timeout=60.0):
        # only "good" answers the text ("what model are you") IQ test
        if payload.get("model") == "good" and "chat/completions" in url:
            if isinstance(payload["messages"][0]["content"], str):
                return {"choices": [{"message": {"content": "I am good"}}]}
        raise FakeHTTPError(404)

    monkeypatch.setattr(emullm_api, "_http_post_json", fake_post)
    monkeypatch.setattr(emullm_api, "_http_post_raw", lambda *a, **k: (_ for _ in ()).throw(FakeHTTPError(404)))
    emullm_api.apply_agent_policies(
        {
            "agents": [
                {
                    "kind": "agent",
                    "id": "o",
                    "launch": "proxy",
                    "base_url": "http://x/v1",
                    "services": {"models": {"behavior": "aggregate", "update_interval": "1day", "validate": True}},
                }
            ]
        }
    )
    cat = emullm_api.advertised_catalog()
    assert cat["models"] == ["good"]  # "dead" (404 everywhere) filtered out by validation


def test_validation_interval_never_and_default(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_get(url, headers, timeout=15.0):
        calls["n"] += 1
        return {"data": [{"id": "live"}]}

    monkeypatch.setattr(emullm_api, "_http_get_json", fake_get)

    # "never" -> no fetch, just the config models
    emullm_api.apply_agent_policies(
        {
            "agents": [
                {
                    "kind": "agent",
                    "id": "n",
                    "launch": "proxy",
                    "base_url": "http://x/v1",
                    "models": ["cfg"],
                    "services": {"models": {"behavior": "aggregate", "validation_interval": "never"}},
                }
            ]
        }
    )
    assert emullm_api.advertised_catalog()["models"] == ["cfg"]
    assert calls["n"] == 0

    # "default" -> inherit the server-level validation_interval (so it fetches)
    emullm_api.apply_agent_policies(
        {
            "validation_interval": "1day",
            "agents": [
                {
                    "kind": "agent",
                    "id": "d",
                    "launch": "proxy",
                    "base_url": "http://x/v1",
                    "models": ["cfg"],
                    "services": {"models": {"behavior": "aggregate", "validation_interval": "default"}},
                }
            ],
        }
    )
    assert "live" in emullm_api.advertised_catalog()["models"]
    assert calls["n"] >= 1


def test_validation_interval_override_forces_all(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_get(url, headers, timeout=15.0):
        calls["n"] += 1
        return {"data": [{"id": "live"}]}

    monkeypatch.setattr(emullm_api, "_http_get_json", fake_get)
    # agent declares its OWN 1day, but the top-level override "never" wins.
    emullm_api.apply_agent_policies(
        {
            "validation_interval_override": "never",
            "agents": [
                {
                    "kind": "agent",
                    "id": "o",
                    "launch": "proxy",
                    "base_url": "http://x/v1",
                    "models": ["cfg"],
                    "services": {"models": {"behavior": "aggregate", "validation_interval": "1day"}},
                }
            ],
        }
    )
    assert emullm_api.advertised_catalog()["models"] == ["cfg"]  # override "never" -> no fetch
    assert calls["n"] == 0


def test_agent_level_validation_interval_is_implicit_for_its_models(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}

    def fake_get(url, headers, timeout=15.0):
        calls["n"] += 1
        return {"data": [{"id": "live"}]}

    monkeypatch.setattr(emullm_api, "_http_get_json", fake_get)
    # plain validation_interval at the agent level applies implicitly to its
    # services.models (which doesn't declare its own).
    emullm_api.apply_agent_policies(
        {
            "agents": [
                {
                    "kind": "agent",
                    "id": "a",
                    "launch": "proxy",
                    "base_url": "http://x/v1",
                    "models": ["cfg"],
                    "validation_interval": "1day",
                    "services": {"models": {"behavior": "aggregate"}},
                }
            ]
        }
    )
    assert "live" in emullm_api.advertised_catalog()["models"]
    assert calls["n"] >= 1


def test_models_can_be_ids_or_nodes(client: TestClient) -> None:
    # A models list may mix bare ids and node objects; the catalog extracts ids.
    emullm_api.apply_agent_policies(
        {
            "services": {"models": ["base-id", {"id": "base-node", "chat": True}]},
            "agents": [
                {
                    "kind": "agent",
                    "id": "a",
                    "launch": "proxy",
                    "base_url": "http://x/v1",
                    "models": ["m-str", {"id": "m-node", "validation_doneAt": "2020-01-01T00:00:00Z"}],
                    "services": {"models": {"behavior": "aggregate", "validation_interval": None}},
                }
            ],
        }
    )
    cat = emullm_api.advertised_catalog()
    assert cat["models"] == ["base-id", "base-node", "m-str", "m-node"]


def test_validate_agent_models_produces_nodes_with_timestamps(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(emullm_api.time, "sleep", lambda *a, **k: None)

    def fake_post(url, headers, payload, timeout=60.0):
        if "chat/completions" in url and isinstance(payload["messages"][0]["content"], str):
            return {"choices": [{"message": {"content": "I am M1"}}]}
        raise RuntimeError("wrong modality")

    monkeypatch.setattr(emullm_api, "_http_post_json", fake_post)
    nodes = emullm_api.validate_agent_models({"id": "o", "base_url": "http://x/v1", "models": ["m1"]})
    node = nodes[0]
    assert node["id"] == "m1"
    assert node["chat"] is True and node["status"] == "live"
    assert node["identity"] == "I am M1"
    assert node["validation_startedAt"] and node["validation_doneAt"]


class _FakeHTTPError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code


def test_probe_status_reachable_when_chat_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    # Endpoint answers a well-formed completion but with no usable content, and
    # serves nothing else -> "reachable" (not dead, but won't chat), not "live".
    monkeypatch.setattr(emullm_api.time, "sleep", lambda *a, **k: None)

    def fake_post(url, headers, payload, timeout=60.0):
        if "chat/completions" in url:
            return {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}
        raise _FakeHTTPError(404)

    monkeypatch.setattr(emullm_api, "_http_post_json", fake_post)
    monkeypatch.setattr(emullm_api, "_http_post_raw", lambda *a, **k: (_ for _ in ()).throw(_FakeHTTPError(404)))
    r = emullm_api._probe_modalities_sync("http://x/v1", {}, "empty-model")
    assert r["status"] == "reachable"
    assert r["live"] is False
    assert r["chat"] is True  # chat capability present; it just won't answer
    assert "won't chat" in r["notes"]


def test_probe_status_embeddings_only(monkeypatch: pytest.MonkeyPatch) -> None:
    # Doesn't chat, but serves embeddings -> usable (live: True) yet not "live"
    # status; reported as "embeddings-only".
    monkeypatch.setattr(emullm_api.time, "sleep", lambda *a, **k: None)

    def fake_post(url, headers, payload, timeout=60.0):
        if "embeddings" in url:
            return {"data": [{"embedding": [0.1, 0.2]}]}
        raise _FakeHTTPError(404)

    monkeypatch.setattr(emullm_api, "_http_post_json", fake_post)
    monkeypatch.setattr(emullm_api, "_http_post_raw", lambda *a, **k: (_ for _ in ()).throw(_FakeHTTPError(404)))
    r = emullm_api._probe_modalities_sync("http://x/v1", {}, "emb-model")
    assert r["status"] == "embeddings-only"
    assert r["live"] is True
    assert r["chat"] is False
    assert "no chat" in r["notes"]


def test_probe_not_loaded_note_confirms_4xx(monkeypatch: pytest.MonkeyPatch) -> None:
    # A definite 4xx on every surface -> not_loaded, with the code confirmed.
    monkeypatch.setattr(emullm_api.time, "sleep", lambda *a, **k: None)

    def fake_post(url, headers, payload, timeout=60.0):
        raise _FakeHTTPError(400)

    monkeypatch.setattr(emullm_api, "_http_post_json", fake_post)
    monkeypatch.setattr(emullm_api, "_http_post_raw", fake_post)
    r = emullm_api._probe_modalities_sync("http://x/v1", {}, "gone-model")
    assert r["status"] == "not_loaded"
    assert r["notes"] == "confirmed not loaded (HTTP 400)"


def test_validate_skips_recently_validated_node(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(emullm_api.time, "sleep", lambda *a, **k: None)

    def fake_post(url, headers, payload, timeout=60.0):
        if "chat/completions" in url and isinstance(payload["messages"][0]["content"], str):
            return {"choices": [{"message": {"content": "ok"}}]}
        raise RuntimeError("wrong modality")

    monkeypatch.setattr(emullm_api, "_http_post_json", fake_post)
    fresh = emullm_api._now_iso()
    agent = {
        "id": "o",
        "base_url": "http://x/v1",
        "services": {
            "models": {
                "behavior": "aggregate",
                "validation_interval": "1day",
                "catalog": [
                    {"id": "recent", "validation_doneAt": fresh},  # fresh -> kept as-is
                    "stale-id",  # bare id -> gets the IQ test
                ],
            }
        },
    }
    nodes = emullm_api.validate_agent_models(agent)
    by_id = {n["id"]: n for n in nodes}
    assert by_id["recent"].get("validation_doneAt") == fresh  # not re-tested
    assert "chat" not in by_id["recent"]  # untouched original node
    assert by_id["stale-id"]["status"] == "live"  # freshly validated


def test_validation_timeout_node(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(emullm_api.time, "sleep", lambda *a, **k: None)

    def hang(base, headers, model_id, **kwargs):
        import threading

        threading.Event().wait(5)  # never returns within the tiny timeout
        return {"id": model_id, "status": "live"}

    monkeypatch.setattr(emullm_api, "_probe_modalities_sync", hang)
    agent = {
        "id": "o",
        "base_url": "http://x/v1",
        "services": {
            "models": {
                "behavior": "aggregate",
                "validation_interval": "1day",
                "validation_timeout": 1,  # 1 second budget
                "catalog": ["slow-model"],
            }
        },
    }
    node = emullm_api.validate_agent_models(agent)[0]
    assert node["status"] == "timeout"
    assert "raise the timeout" in node["description"]


def test_cross_agent_model_failover(client: TestClient) -> None:
    emullm_api.apply_agent_policies(
        {
            "agents": [
                {
                    "kind": "agent",
                    "id": "primary",
                    "launch": "proxy",
                    "base_url": "http://p/v1",
                    "services": {
                        "models": {
                            "behavior": "aggregate",
                            "validation_interval": None,
                            "catalog": [{"id": "shared", "status": "not_loaded", "live": False}, "solo-p"],
                        }
                    },
                },
                {
                    "kind": "agent",
                    "id": "backup",
                    "launch": "proxy",
                    "base_url": "http://b/v1",
                    "services": {
                        "models": {"behavior": "aggregate", "validation_interval": None, "catalog": ["shared", "solo-b"]}
                    },
                },
            ]
        }
    )
    # "shared" is dead on primary, live on backup -> failover routes to backup.
    assert emullm_api.agents_for_model("shared") == ["backup"]
    assert emullm_api.agents_for_model("solo-p") == ["primary"]
    assert emullm_api.model_failover_map()["shared"] == ["backup"]


def test_aggregate_router_strategies(client: TestClient) -> None:
    # failover: skip validated-dead, take first live, rest are the failover order
    chosen, order = emullm_api.select_from_catalog(
        [{"id": "dead", "status": "not_loaded"}, "m1", "m2"], "failover"
    )
    assert chosen == "m1" and order == ["m1", "m2"]
    # round-robin rotates across live entries (keyed)
    picks = [emullm_api.select_from_catalog(["a", "b", "c"], "round-robin", key="k")[0] for _ in range(4)]
    assert picks == ["a", "b", "c", "a"]
    # random returns a live one
    assert emullm_api.select_from_catalog(["x", "y"], "random")[0] in ("x", "y")
    # all dead / empty -> nothing
    assert emullm_api.select_from_catalog([{"id": "d", "live": False}], "failover") == (None, [])


def test_resolve_service_route_aggregate(client: TestClient) -> None:
    agent = {
        "id": "o",
        "services": {
            "images": {
                "behavior": "aggregate",
                "strategy": "failover",
                "catalog": [{"id": "d", "live": False}, "m1", "m2"],
            }
        },
    }
    assert emullm_api.resolve_service_route(agent, "images") == ("m1", ["m1", "m2"])
    # a non-aggregate service has no route
    assert emullm_api.resolve_service_route({"id": "o", "services": {"chat": "serve"}}, "chat") == (None, [])


def test_model_list_lives_on_service_node(client: TestClient) -> None:
    # the model list/cache may live on services.models.catalog (or agent.models)
    emullm_api.apply_agent_policies(
        {
            "agents": [
                {
                    "kind": "agent",
                    "id": "o",
                    "launch": "proxy",
                    "base_url": "http://x/v1",
                    "services": {
                        "models": {
                            "behavior": "aggregate",
                            "validation_interval": None,
                            "catalog": ["svc-a", "svc-b"],
                        }
                    },
                }
            ]
        }
    )
    assert emullm_api.advertised_catalog()["models"] == ["svc-a", "svc-b"]


def test_admin_page_renders_html(client: TestClient) -> None:
    for url in ("/emullm", "/emullm/", "/emullm/admin", "/admin/emullm"):
        response = client.get(url)
        assert response.status_code == 200, url
        assert "text/html" in response.headers["content-type"]
        assert response.headers["cache-control"] == "no-store"
        assert "EMULLM // CONTROL PLANE" in response.text
    # The page resolves its REST calls relative to wherever it's served.
    html = client.get("/emullm/admin").text
    assert "location.pathname" in html
    assert 'id="poll-window"' in html
    assert 'id="poll-hidden"' in html
    assert 'id="poll-wake"' in html
    assert "document.addEventListener('visibilitychange'" in html
    assert "POLL_HIDDEN_MS = 120000" in html
    assert "setInterval(tick, 3000)" not in html
    assert "PAGE_PATH === '/emullm'" in html
    assert "Headless Copilot servants" in html
    assert 'id="copilot-form"' in html
    assert 'id="cp-allow-all"' in html
    assert 'id="cp-warmup"' not in html
    assert 'id="cp-allow-all" type="checkbox" checked' in html
    assert 'id="cp-custom-instructions" type="checkbox" checked' in html
    assert 'id="cp-builtin-mcps" type="checkbox" checked' in html
    assert 'id="refresh-copilot-models"' in html
    assert 'id="cp-model-pool"' in html
    assert 'id="cp-model-picker"' in html
    assert "field('cp-model').value = field('cp-model-picker').value" in html
    assert 'id="cp-model-selector"' in html
    assert "updateReasoningOptions" in html
    assert "most-1" in html and "least-1" in html
    assert 'id="cp-warmup-prompt"' in html
    assert 'id="cp-chunk-prompts"' in html
    assert 'id="cp-chunk-tokens"' in html
    assert 'id="copilot-add-another"' in html
    assert "beginNewCopilot" in html
    assert "Model test client" in html
    assert 'id="model-test-form"' in html
    assert 'id="model-test-model"' in html
    assert 'id="model-test-model-picker"' in html
    assert 'id="api-model-count"' in html
    assert "field('model-test-model').value = field('model-test-model-picker').value" in html
    assert 'id="model-test-cancel"' in html
    assert "new AbortController()" in html
    assert "r.status === 499" in html
    assert "toFixed(1) + 's'" in html
    assert 'id="model-test-files"' in html
    assert 'id="model-test-drop"' in html
    assert 'id="model-test-attachments"' in html
    assert 'id="model-test-uploaded"' in html
    assert 'id="cp-model-capabilities"' in html
    assert 'id="api-model-capabilities"' in html
    assert "model.display_name || model.name || model.id" in html
    assert "<strong>Native audio:</strong>" in html
    assert 'id="model-configurator"' in html
    assert "Export in /v1/models" in html
    assert "getJSON('/v1/models?hidden=true'" in html
    assert "[unexported]" in html
    assert 'id="model-config-list" size="18" multiple' in html
    assert 'id="model-config-json"' in html
    assert 'id="model-config-route"' in html
    assert 'id="model-route-order"' in html
    assert 'id="model-route-specific-backends"' in html
    assert "data-route-move" in html
    assert 'id="model-config-audio"' in html
    assert 'id="model-config-code"' in html
    assert 'id="model-config-image-output"' in html
    assert 'id="model-config-summary"' in html
    assert 'id="model-config-save"' in html
    assert 'id="model-config-load"' in html
    assert 'id="model-test-configure"' in html
    assert 'id="model-test-capabilities"' in html
    assert 'id="image-generation-model"' in html
    assert 'value="router/gpt-5.3-codex"' in html
    assert 'id="image-generation-run"' in html
    assert 'id="image-generation-preview"' in html
    assert 'id="model-test-samples"' in html
    assert "refreshModelTestSamples" in html
    assert 'href="#backend-config"' in html
    assert 'id="backend-form"' in html
    assert 'id="backend-config-rows"' in html
    assert 'href="#codex-suppliers"' in html
    assert 'id="supplier-form"' in html
    assert 'id="supplier-rows"' in html
    assert "refreshBackendConfigs" in html
    assert "refreshCodexSuppliers" in html
    assert "revision: record.revision" in html
    assert "body.expected_revision = editingBackend.revision" in html
    assert "editingSupplier = { id: supplier.id, revision: supplier.revision }" in html
    assert "if (r.ok) await loadConfig();" in html
    assert 'href="#anti-idle"' in html
    assert 'id="anti-idle-list"' in html
    assert 'id="anti-idle-interval"' in html
    assert 'id="anti-idle-timeout"' in html
    assert 'id="anti-idle-slow-budget"' in html
    assert 'id="anti-idle-reset-stats"' in html
    assert "/anti-idle/enabled" in html
    assert "applies immediately" in html
    assert html.count("data-anti-sort=") == 10
    assert "Shortest / worker" in html
    assert "Longest / worker" in html
    assert "Over budget" in html
    assert "data-anti-deprecated" in html
    assert "#anti-idle-table { width: max-content; min-width: 100%; }" in html
    assert "white-space: nowrap" in html
    assert "antiIdleDuration(prompt.min_duration_ms) + ' · <code>'" in html
    assert "average_duration_ms" in html
    assert "deprecated (scheduler skips it)" in html
    assert "refreshAntiIdle" in html


def test_admin_page_javascript_has_valid_syntax(
    client: TestClient,
    tmp_path,
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    html = client.get("/emullm").text
    scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.DOTALL)
    assert scripts
    script_path = tmp_path / "admin-page.js"
    script_path.write_text("\n".join(scripts), encoding="utf-8")
    result = subprocess.run(
        [node, "--check", str(script_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "'/test-chat'" in html
    assert 'id="refresh-websockets"' in html
    assert 'id="websocket-connections"' in html
    assert 'id="carol-enabled"' in html
    assert 'id="server-settings"' in html
    assert 'id="server-settings-save"' in html
    assert 'id="server-proxy-url"' not in html
    assert "'/config/server-settings'" in html
    assert 'id="server-max-concurrent"' in html
    assert 'id="server-idle-workers"' in html
    assert 'id="server-idle-grace"' in html
    assert 'id="server-backend-delay"' in html
    assert 'id="top-uptime"' in html
    assert 'id="top-active-count"' in html
    assert 'id="top-waiting-count"' in html
    assert 'id="top-stuck-count"' in html
    assert 'id="top-served-count"' in html
    assert 'id="telemetry-alerts"' in html
    assert "client max " in html
    assert 'id="telemetry-services"' in html
    assert 'id="telemetry-workers"' in html
    assert 'id="telemetry-models"' in html
    assert 'id="telemetry-footer-mode"' in html
    assert 'id="telemetry-workers-toggle"' in html
    assert 'id="telemetry-models-toggle"' in html
    assert 'id="telemetry-workers-total"' in html
    assert 'id="telemetry-models-total"' in html
    assert 'data-stats-table="workers"' in html
    assert 'data-stats-table="models"' in html
    assert 'id="top-switch-count"' in html
    assert 'id="copilot-start-all"' in html
    assert 'id="copilot-stop-all"' in html
    assert 'id="copilot-stop-idle"' in html
    assert "formatDuration" in html
    assert 'id="config-section-tabs"' in html
    assert 'id="config-section-editor"' in html
    assert "'/config/section/'" in html
    assert 'id="server-restart"' in html
    assert 'id="server-shutdown"' in html


def test_admin_rest_works_under_both_prefixes(client: TestClient) -> None:
    # config PUT/GET reachable under both admin namespaces
    revision = client.get("/emullm/admin/config").json()["revision"]
    first = client.put(
        "/emullm/admin/config",
        json={"config": {"mode": "mock"}, "expected_revision": revision},
    )
    assert first.status_code == 200
    assert client.get("/admin/emullm/config").json()["config"] == {"mode": "mock"}
    assert client.put(
        "/admin/emullm/config",
        json={
            "config": {"mode": "auto"},
            "expected_revision": first.json()["revision"],
        },
    ).status_code == 200
    assert client.get("/emullm/admin/config").json()["config"] == {"mode": "auto"}
    # workers listing reachable under both
    assert client.get("/emullm/admin/workers").status_code == 200
    assert client.get("/admin/emullm/workers").status_code == 200


def test_admin_runtime_dir_and_reset(client: TestClient, tmp_path) -> None:
    new_dir = tmp_path / "moved"
    client.post("/admin/emullm/runtime_dir", json={"path": str(new_dir)})
    client.post("/v1/files", json={"note": "x"})
    assert client.get("/v1/files").json()["data"]

    client.post("/admin/emullm/reset")

    assert client.get("/v1/files").json()["data"] == []


def test_admin_delete_record(client: TestClient) -> None:
    created = client.post("/v1/files", json={"note": "x"}).json()
    assert client.delete(f"/admin/emullm/records/files/{created['id']}").status_code == 200
    assert client.get("/v1/files").json()["data"] == []
    assert client.delete(f"/admin/emullm/records/files/{created['id']}").status_code == 404
    assert client.delete("/admin/emullm/records/no-such-kind/x").status_code == 404


def test_admin_routes_have_an_emullm_admin_alias(client: TestClient) -> None:
    """/emullm/admin/* must behave identically to /admin/emullm/*."""
    a = client.get("/emullm/admin/state").json()
    b = client.get("/admin/emullm/state").json()
    # uptime_seconds is time-varying between the two calls; compare the rest.
    a.pop("uptime_seconds", None)
    b.pop("uptime_seconds", None)
    assert a == b

    created = client.post("/v1/files", json={"note": "via alias"}).json()
    reset_via_alias = client.post("/emullm/admin/reset").json()
    assert reset_via_alias["removed"]["files"] == 1
    assert client.get("/v1/files").json()["data"] == []

    created = client.post("/v1/files", json={"note": "delete via alias"}).json()
    assert client.delete(f"/emullm/admin/records/files/{created['id']}").status_code == 200
    assert client.get("/v1/files").json()["data"] == []

    emullm_api._connected_workers["yourself"] = FakeWorker(reply="ok")  # noqa: SLF001
    asyncio.run(emullm_api._relay("yourself/same", "hi"))
    client.post("/emullm/admin/usage/reset")
    assert client.get("/admin/emullm/state").json()["worker_usage"] == {}


def test_clients_never_need_auth_tokens(client: TestClient) -> None:
    """OpenAI-compatible clients must work keyless: no Authorization header
    required, and a bogus Bearer token must not be treated as required/valid
    auth (it is simply ignored)."""
    bare = client.get("/v1/models")
    assert bare.status_code == 200
    assert "data" in bare.json()

    with_bogus = client.get(
        "/v1/models",
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert with_bogus.status_code == 200
    assert with_bogus.json() == bare.json()

    emullm_api._connected_workers["yourself"] = FakeWorker(reply="ok")  # noqa: SLF001
    chat = client.post(
        "/v1/chat/completions",
        json={
            "model": "yourself/same",
            "messages": [{"role": "user", "content": "hi"}],
        },
        # deliberately no Authorization header
    )
    assert chat.status_code == 200
    assert chat.json()["choices"][0]["message"]["content"] == "ok"

    chat_bogus = client.post(
        "/v1/chat/completions",
        json={
            "model": "yourself/same",
            "messages": [{"role": "user", "content": "hi again"}],
        },
        headers={"Authorization": "Bearer sk-no-key-required"},
    )
    assert chat_bogus.status_code == 200
    assert chat_bogus.json()["choices"][0]["message"]["content"] == "ok"


def test_tokens_new_page_is_html(client: TestClient) -> None:
    response = client.get("/emullm/tokens/new")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "You do not need a token" in response.text
    assert "Optional token" in response.text


def test_create_token_requires_email(client: TestClient) -> None:
    assert client.post("/emullm/tokens", json={}).status_code == 422  # missing required field
    assert client.post("/emullm/tokens", json={"email": "  "}).status_code == 400


def test_create_token_generates_one_by_default(client: TestClient) -> None:
    result = client.post("/emullm/tokens", json={"email": "a@example.com"}).json()
    assert result["id"]
    assert result["email"] == "a@example.com"
    assert emullm_api.is_valid_token(result["id"]) is True
    assert emullm_api.is_valid_token("not-a-real-token") is False


def test_create_token_accepts_a_bring_your_own_token(client: TestClient) -> None:
    result = client.post("/emullm/tokens", json={"email": "a@example.com", "token": "my-own-token"}).json()
    assert result["id"] == "my-own-token"
    assert emullm_api.is_valid_token("my-own-token") is True


def test_create_token_can_register_a_public_key(client: TestClient) -> None:
    pubkey = "ssh-ed25519 AAAAtest a@example.com"
    client.post("/emullm/tokens", json={"email": "a@example.com", "public_key": pubkey})
    assert emullm_api.is_registered_public_key(pubkey) is True
    assert emullm_api.is_registered_public_key("ssh-ed25519 AAAAnope") is False

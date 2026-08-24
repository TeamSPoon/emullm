"""Unit tests for the auto-mode worker supervisor.

Uses an injected fake launcher so no real subprocesses are started.
"""
from __future__ import annotations

import asyncio
import json
import sys

from emullm import supervisor as sup
from emullm import worker as worker_mod


def test_worker_wait_for_reply_forwards_full_media_dict(tmp_path):
    # A student worker returns real media beside "content"; the worker client
    # must forward the whole object (not just the text) so image_b64/audio_b64
    # reach the relay's two-way media path.
    reply_file = tmp_path / "reply.json"
    reply_file.write_text(
        json.dumps({"id": "abc", "content": "here it is", "image_b64": "QUJD", "mime": "image/png"}),
        encoding="utf-8",
    )
    got = asyncio.run(worker_mod._wait_for_reply("abc", reply_file, timeout=2.0))
    assert got["content"] == "here it is"
    assert got["image_b64"] == "QUJD"
    assert got["mime"] == "image/png"
    assert not reply_file.exists()  # consumed after use


class FakeProc:
    """Minimal stand-in for subprocess.Popen."""

    def __init__(self, pid: int = 4321) -> None:
        self.pid = pid
        self._alive = True
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self.terminated = True
        self._alive = False

    def wait(self, timeout=None):
        self._alive = False
        return 0

    def kill(self):
        self.killed = True
        self._alive = False


def _fake_spawn():
    launched = []

    def spawn(spec):
        proc = FakeProc(pid=1000 + len(launched))
        launched.append(spec)
        return proc

    return spawn, launched


def test_discover_worker_specs_finds_subagent_folders(tmp_path):
    for i in (1, 2, 3):
        (tmp_path / "subagents" / f"emullm_worker_{i}").mkdir(parents=True)
    (tmp_path / "subagents" / "not_a_worker").mkdir()

    specs = sup.discover_worker_specs(tmp_path, "ws://127.0.0.1:9999")

    ids = [s.worker_id for s in specs]
    assert ids == ["emullm_worker_1", "emullm_worker_2", "emullm_worker_3"]
    first = specs[0]
    assert first.cwd == tmp_path / "subagents" / "emullm_worker_1"
    # by default a discovered subagent launches the Copilot CLI (reads AGENTS.md)
    assert first.argv[0] == "copilot"
    # unattended workers get broad permissions (--allow-all = tools+paths+urls)
    assert "--allow-all" in first.argv
    assert "--no-ask-user" in first.argv


def test_discover_worker_specs_launch_override(tmp_path):
    (tmp_path / "subagents" / "emullm_worker_1").mkdir(parents=True)
    specs = sup.discover_worker_specs(
        tmp_path, "ws://127.0.0.1:8801", launch=["python", "-m", "emullm.worker"]
    )
    assert specs[0].argv == ["python", "-m", "emullm.worker"]


def test_discover_worker_specs_worker_kind(tmp_path):
    (tmp_path / "subagents" / "emullm_worker_1").mkdir(parents=True)
    specs = sup.discover_worker_specs(tmp_path, "ws://127.0.0.1:8801", launch="worker")
    assert specs[0].argv[0] == sys.executable
    assert "emullm.worker" in specs[0].argv


def test_discover_worker_specs_recruit_kind_not_spawned(tmp_path):
    # interactive recruits connect themselves -> nothing to spawn
    for i in (1, 2):
        (tmp_path / "subagents" / f"emullm_worker_{i}").mkdir(parents=True)
    for kind in ("recruit", "interactive"):
        assert sup.discover_worker_specs(tmp_path, launch=kind) == []


def test_specs_from_config_resolves_launch_kinds(tmp_path):
    config = {"workers": [
        {"id": "w1", "launch": "copilot"},
        {"id": "w2", "launch": "recruit"},  # connects itself -> skipped
        {"id": "w3"},                        # no launch -> default worker loop
    ]}
    specs = sup.specs_from_config(config, tmp_path)
    ids = [s.worker_id for s in specs]
    assert ids == ["w1", "w3"]
    assert specs[0].argv[0] == "copilot"
    assert "emullm.worker" in specs[1].argv


def test_expand_agents_maps_launch_types(tmp_path):
    config = {"agents": [
        {"kind": "agent", "id": "a", "launch": "subagent", "command": "copilot"},
        {"kind": "agent", "id": "b", "launch": "recruit"},
        {"kind": "agent", "id": "c", "launch": "mock", "reply": "hi"},
        {"kind": "agent", "id": "d", "launch": "proxy", "base_url": "http://x", "default": True},
    ]}
    out = sup.expand_agents(config)
    assert [w["id"] for w in out["workers"]] == ["a"]
    assert out["workers"][0]["launch"] == "copilot"
    assert out["workers"][0]["cwd"] == "subagents/a"
    assert [m["id"] for m in out["mock_workers"]] == ["c"]
    assert [b["base_url"] for b in out["backends"]] == ["http://x"]
    # recruit 'b' connects itself -> not spawned/registered/proxied
    assert all(entry.get("id") != "b" for entry in out["workers"] + out["mock_workers"])


def test_copilot_launch_argv_includes_model_when_set():
    with_model = sup.copilot_launch_argv(model="claude-sonnet-4.5")
    assert "--model" in with_model and "claude-sonnet-4.5" in with_model
    assert with_model[with_model.index("--model") + 1] == "claude-sonnet-4.5"
    # omitted -> no --model (Copilot picks 'auto')
    assert "--model" not in sup.copilot_launch_argv()


def test_config_worker_model_flows_to_copilot_launch(tmp_path):
    config = {"workers": [{"id": "w1", "launch": "copilot", "model": "gpt-5.4"}]}
    spec = sup.specs_from_config(config, tmp_path)[0]
    assert spec.argv[0] == "copilot"
    assert "--model" in spec.argv and "gpt-5.4" in spec.argv


def test_subagent_model_default_flows_to_discovered_workers(tmp_path):
    (tmp_path / "subagents" / "emullm_worker_1").mkdir(parents=True)
    specs = sup.build_specs(tmp_path, config={"subagent_model": "claude-opus-4.5"})
    assert "--model" in specs[0].argv and "claude-opus-4.5" in specs[0].argv


def test_expand_agents_subagent_carries_model(tmp_path):
    config = {"agents": [{"kind": "agent", "id": "w1", "launch": "subagent", "model": "gpt-5.4"}]}
    out = sup.expand_agents(config)
    assert out["workers"][0]["model"] == "gpt-5.4"
    # and that model reaches the actual launch argv
    spec = sup.specs_from_config(out, tmp_path)[0]
    assert "--model" in spec.argv and "gpt-5.4" in spec.argv


def test_expand_agents_passthrough_without_agents(tmp_path):
    config = {"mode": "mock", "workers": [{"id": "w1"}]}
    assert sup.expand_agents(config) == config


def test_supervisor_start_stop_and_status():
    spawn, launched = _fake_spawn()
    spec = sup.WorkerSpec(worker_id="w1", argv=["x"], role="training")
    s = sup.Supervisor([spec], spawn=spawn)

    assert s.start("w1") is True
    assert len(launched) == 1
    # already running -> no second launch
    assert s.start("w1") is False
    assert len(launched) == 1

    row = s.status()[0]
    assert row["worker_id"] == "w1"
    assert row["running"] is True
    assert row["managed"] is True
    assert row["role"] == "training"
    assert row["pid"] is not None

    assert s.stop("w1") is True
    assert s.status()[0]["running"] is False
    # stopping an already-stopped worker is a no-op
    assert s.stop("w1") is False


def test_supervisor_start_autostart_respects_flag():
    spawn, launched = _fake_spawn()
    specs = [
        sup.WorkerSpec(worker_id="a", argv=["x"], autostart=True),
        sup.WorkerSpec(worker_id="b", argv=["x"], autostart=False),
    ]
    s = sup.Supervisor(specs, spawn=spawn)

    started = s.start_autostart()

    assert started == ["a"]
    running = {r["worker_id"]: r["running"] for r in s.status()}
    assert running == {"a": True, "b": False}

    s.stop_all()
    assert all(not r["running"] for r in s.status())


def test_supervisor_start_unknown_worker_is_false():
    s = sup.Supervisor([], spawn=lambda spec: FakeProc())
    assert s.start("nope") is False


def test_remove_spec_stops_and_forgets():
    spawn, _ = _fake_spawn()
    s = sup.Supervisor([sup.WorkerSpec(worker_id="w", argv=["x"])], spawn=spawn)
    s.start("w")
    s.remove_spec("w")
    assert s.status() == []


def test_specs_from_config_builds_workers(tmp_path):
    config = {
        "workers": [
            {"id": "w1", "role": "training", "cwd": "subagents/w1", "launch": ["copilot"]},
            {"id": "w2"},  # defaults: role trusted, autostart true, default argv
            {"role": "no-id-skip-me"},  # skipped (no id)
        ]
    }
    specs = sup.specs_from_config(config, tmp_path, "ws://127.0.0.1:8801")

    assert [s.worker_id for s in specs] == ["w1", "w2"]
    w1, w2 = specs
    assert w1.role == "training"
    assert w1.cwd == tmp_path / "subagents" / "w1"
    assert w1.argv == ["copilot"]
    # w2 uses the default python worker launch
    assert "emullm.worker" in w2.argv and "w2" in w2.argv


def test_build_specs_prefers_config_then_discovery(tmp_path):
    # With config workers -> those win.
    cfg = {"workers": [{"id": "cfgworker"}]}
    specs = sup.build_specs(tmp_path, "ws://127.0.0.1:8801", cfg)
    assert [s.worker_id for s in specs] == ["cfgworker"]

    # Without config workers -> fall back to subagents/ discovery.
    (tmp_path / "subagents" / "emullm_worker_1").mkdir(parents=True)
    specs = sup.build_specs(tmp_path, "ws://127.0.0.1:8801", {})
    assert [s.worker_id for s in specs] == ["emullm_worker_1"]


def test_load_config_reads_object_or_empty(tmp_path):
    path = tmp_path / "config.json"
    assert sup.load_config(path) == {}
    path.write_text('{"mode": "auto"}', encoding="utf-8")
    assert sup.load_config(path) == {"mode": "auto"}
    path.write_text("not json", encoding="utf-8")
    assert sup.load_config(path) == {}


def test_app_lifespan_starts_workers_from_config(tmp_path, monkeypatch):
    """End-to-end: EMULLM auto mode + config.json workers -> the app spawns
    them on startup and clears the supervisor on shutdown. subprocess.Popen
    is faked so no real processes are launched."""
    import json

    from fastapi.testclient import TestClient

    from emullm import api as api_mod
    from emullm import app as app_mod

    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps({"mode": "auto", "workers": [{"id": "cfgw", "role": "training"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(api_mod, "_CONFIG_PATH", cfg)
    monkeypatch.setattr(app_mod, "_BASE_DIR", tmp_path)
    monkeypatch.setattr(sup.subprocess, "Popen", lambda *a, **k: FakeProc())

    with TestClient(app_mod.app) as client:
        listing = client.get("/admin/emullm/workers").json()
        assert listing["supervisor_active"] is True
        rows = {w["worker_id"]: w for w in listing["workers"]}
        assert "cfgw" in rows
        assert rows["cfgw"]["running"] is True
        assert rows["cfgw"]["role"] == "training"

    # lifespan shutdown clears the supervisor singleton
    assert sup.get_supervisor() is None


def test_app_lifespan_registers_mock_copilots_from_config(tmp_path, monkeypatch):
    """config `mock_workers` -> the app registers pretend copilots on startup
    (visible as connected workers) and removes them on shutdown."""
    import json

    from fastapi.testclient import TestClient

    from emullm import api as api_mod
    from emullm import app as app_mod

    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps({"mode": "mock", "mock_workers": [{"id": "carol", "reply": "c"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(api_mod, "_CONFIG_PATH", cfg)
    monkeypatch.setattr(app_mod, "_BASE_DIR", tmp_path)

    with TestClient(app_mod.app) as client:
        state = client.get("/admin/emullm/state").json()
        assert "carol" in state["connected_worker_ids"]

    # cleaned up on shutdown
    assert "carol" not in api_mod._connected_workers

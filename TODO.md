# TODO — you & copilot

Our casual shared scratchpad. Not formal issue tracking — just what we're
poking at. Anyone can add a line; check things off with `[x]`.

Tags: **(you)** = for Doug, **(cop)** = for copilot, **(both)** = together.

## Now / in progress

- [x] Model "IQ test" validation: text ("what model are you" + identity), then
      vision (10x10 green square -> expect "green"); text-first gates vision,
      embeddings fallback. Shared `_probe_modalities_sync` (async probe wraps
      it); same battery for real backends and recruits — *uncommitted*
- [x] `validate: true` on an aggregate entry filters dead models (keeps
      live/inconclusive; drops definite 404s) — *uncommitted*
- [x] Full-modality IQ test: text (identity) + vision (green square) + embeddings
      + image-gen + audio(TTS); per-model timeout (`validation_timeout`, default
      120s -> `status: timeout` w/ note) + `notes` (HTTP codes seen); catalog
      nodes carry start/done timestamps. Live-probed all 15 SNET models; wrote
      accurate per-service sections into config.json — *uncommitted*

## Up next (menu — tell me a number to prioritize)

1. [x] **Aggregate router + server fallback chain + live proxy** — `fallback`
       is now a chain (`"round-robin, error"`); agents just volunteer/reject
       (agent `aggregate`=volunteer). Proxy accepts real backend model ids
       (passthrough) and forwards the requested model; persona `percentNN`
       dials apparent capability via a system message. **Verified live on
       SNET**: split into 3 proxy agents, chat+embeddings per model, cross-agent
       failover (`asi1-mini` on 2 agents), and percent125 vs percent10 capability.

1. [x] **config.json layer** — typed schema + validation + **runtime wiring**
       done. Unified `agents` (kind/launch/services/observe/description) now
       drives spawn/mock/proxy + per-service behavior + observers; JSON Schema
       at `GET /admin/emullm/config/schema`; live `config.json` exercises it.
2. [x] **static admin page** — edit config + start/stop workers (`/emullm/admin`)
3. [x] **backends** — `proxy` + `proxy-observe` forward to a real
       OpenAI-compatible endpoint (config `backends` or `EMULLM_PROXY_*`)
4. [x] **run modes** — every mode selectable + tested against a mock agent;
       modes compose into an ordered fallback chain (`recruit,proxy,mock`).
5. [x] `httpx2` added to the `[test]` extra — fresh `pip install -e ".[test]"`
       runs tests out of the box (starlette 1.6's TestClient needs it).
6. [x] bogus root `package.json` — **deleted** (accidental npm install; kept
       gitignored). Re-add a real one only if a JS admin UI ever needs it.

## Ideas / someday

- [ ] Proxy the non-text `/v1` too (embeddings/images/audio) — today `proxy`
      only forwards text; non-text falls back to the local stub
- [ ] Surface the per-type `/v1` capability matrix live (status/admin page)
- [ ] Config checker: warn when a config key is set but unused/ineffective (next month)
- [ ] Record + replay mode: capture real backend responses, replay as `mock`
- [ ] Discovery-based worker connect when embedded in a host app (no port)
- [ ] More mock types (for automated tests):
  - [ ] A. scripted/sequenced replies (in-order per request)
  - [ ] B. rule/keyword matching (prompt → reply)
  - [ ] C. per-persona replies (percent level → terse/verbose)
  - [ ] D. latency injection (artificial delay)
  - [ ] E. error injection (429/500/... on demand)
  - [ ] F. streaming mock (token-by-token SSE)
  - [ ] G. usage/token shaping (specific token counts)
  - [ ] H. non-text mocks (images/embeddings/moderations/audio)

## Done

- [x] Rename project `emulllm` → `emullm` (package, script, env prefix)
- [x] Fix `emullm-serve` entry point (`run:main` → `emullm.cli:main`)
- [x] Commit the canonical working tree on `master`
- [x] Serve all docs under one `/emullm/docs/` prefix — committed `2f76bc3`
- [x] Reorganize the README — committed `e335188`
- [x] Document configurable constants in the README — committed `a9445de`
- [x] Status pages (overview + detail) + server `mode` + per-worker `role` — committed `8f4c648`
- [x] Supervisor: `auto` mode spawns worker subprocesses + start/stop endpoints — committed `3afb2ff`
- [x] Static admin page (`/emullm/admin`) + minimal `config.json` read/write — committed `81a8927`
- [x] Wire `config.json` workers into the supervisor + admin page works under both prefixes — committed `bc65447`
- [x] Run modes: `mock` + `error-when-empty` behavior in `_relay` (others use the wait path) — committed `7197ce8`
- [x] Backends: `proxy` + `proxy-observe` forward to a real OpenAI-compatible endpoint — committed `1f7eda4`
- [x] Configurable `mock` reply (fixed / template) for automated tests — committed `603dc0a`
- [x] Mock a set of pretend peers (`config.mock_workers` / `register_mock_workers`) — committed `8f5f665`
- [x] Deleted the bogus root `package.json`/lock (accidental npm; stays gitignored) — committed `05eb01f`
- [x] Run modes as a composable fallback chain; every mode selectable + tested vs a mock — *uncommitted*
- [x] Reframe `mock` as a transport-level success (pretend the websocket peer answered) — committed `3af2795`
- [x] `httpx2` in the `[test]` extra so fresh installs run tests OOTB — *uncommitted*
- [x] Worker types: `auto` subagents default to launching **Copilot** (auto-configured
      agent); `subagent_launch` = `copilot`/`worker`/`recruit`/argv (env `EMULLM_SUBAGENT_LAUNCH`).
      An **interactive recruit** (IDE copilot) connects itself — not spawned. — *uncommitted*
- [x] Rewrote the 3 `subagents/*/AGENTS.md` to a known-good, consistent state
      (fixed stale `worker.py` path → `python -m emullm.worker`; restored worker_3's guides) — committed `c0ba15b`
- [x] Named the proxied-backend worker type + per-type `/v1` capability matrix in the README — committed `7177308`
- [x] Capability fallback policy for non-text `/v1` (`stub`/`wait`/`error`) via
      `EMULLM_CAPABILITY_FALLBACK` / config `capability_fallback` — *uncommitted*
- [x] Typed config schema + validation: unified `agents` (kind/launch/services/
      observe/description), 422 on bad/unknown keys, JSON Schema endpoint — committed `058640a`
- [x] Wire the unified `agents` into the runtime: `expand_agents`
      (subagent→spawn, mock→register, proxy→backend, recruit→self-connect),
      unified launch-kind resolution in `specs_from_config`, per-agent
      `services` behavior + server-level fallback in `_capable_or_policy`,
      `observe` mirroring, descriptions surfaced in `admin_state`; live
      `config.json` exercises the combos offline — committed `678ad40`
- [x] Backend capability probe: `GET /admin/emullm/backends/probe` calls each
      proxy backend's `/v1/models` (reference capability set); on-demand,
      failure-tolerant; configured backends listed in `admin_state` — committed `30181af`
- [x] Probe `?verify=true`: actually call each model (chat, then embeddings) to
      catch *falsely advertised* ones; sequential + 429 backoff; classifies
      live / falsely_advertised / inconclusive — *uncommitted*
- [x] Full-key playground `config.json`: every server-level key + all launch
      types; live SNET model list + per-service breakdown written from the
      probe (13 chat, 2 embeddings) — committed `9cda53a`
- [x] Server-level `services` catalog (`ServicesConfig`: `model`/`models` +
      per-service entries). `advertise_models` publishes a proxy agent's models
      into the user-facing catalog; `update_interval` (`null`/`"1day"`/…)
      refreshes them live with a cache + offline fallback
      (`advertised_catalog`, surfaced in `admin_state`) — *uncommitted*

<!-- Design principle (Doug): prefer REAL workers over emulating them. `auto`
     spawns real agents (Copilot by default, or the worker.py loop); an
     interactive recruit is a real IDE copilot that connects itself. `mock`
     is the one deliberate fake and stays a pure transport-success simulation
     (no agent emulation cruft). -->


<!--
How we use this file:
- Keep it casual. Short lines. No ceremony.
- Doug queues ideas in the intake block at the bottom; copilot triages them
  into "Up next" (a numbered menu) and works top-down unless told otherwise.
- Copilot: read this at the start of a work session and keep it updated
  as things change; ask before deleting anyone's item.
-->


Douglas wishes Copilot to add the note below to this TODO.md file from this casual text and or put questions for me here to
```text


```
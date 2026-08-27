# emullm self-improvement showcase

A session-length arc where the agent extended the `emullm` relay to emulate real
LLM backends more faithfully — and then taught *other* agents to become workers
for it, growing its own fleet. Every capability below was designed, implemented,
tested, and then **proven live** against the running relay.

## 1. Honest model-validation statuses

Model probing stopped calling everything that answers "live". New tiers:

- **`live`** — actually chats back with usable content
- **`reachable`** — endpoint answers (HTTP 200) but returns no usable chat
- **`<kind>-only`** (e.g. `embeddings-only`) — serves a non-chat kind, not chat
- **`not_loaded`** — a definite 4xx, now with the exact code confirmed in the note

Verified live against SingularityNET's 15-model catalog.

## 2. Real two-way media + a shared cloud files store

Workers can now exchange **real media**, not just "describe it" text stubs,
persisted to one shared store both relay and workers use
(`/emullm/cloud/files/<id>`):

- **image generation** → worker returns a real PNG, stored + served by URL
- **audio speech** → worker returns a real WAV
- **transcription / vision** → the real clip/image is handed to the worker by URL
- **fine-tuning** → a volunteering worker "trains" a job to `succeeded` with a
  `fine_tuned_model` and a result-manifest cloud file

The relay reply channel became structured (text + optional media) while staying
backward compatible for text-only callers.

## 3. Teaching agents to be workers

- `worker.py` now forwards the whole reply (media included), and surfaces
  incoming media/kind — so an agent-in-the-loop worker can do two-way media.
- Curriculum written where workers actually read it:
  `subagents/*/AGENTS.md` and `docs/EMULLM_RELAY.md` (served live by the relay).
- Managed worker Copilots launch fully unattended (`--allow-all --no-ask-user`)
  and can be pinned to a specific model from `config.json`
  (`subagent_model`, per-worker `model`).

## 4. Live proof — two independent student workers

**Copilot sub-agent student** connected over WS with all capabilities and
answered chat + returned a real 74-byte PNG (verified from the cloud store).

**Network-only Codex worker (`codex-ide-1`)** — the strongest demo:

- bootstrapped from a *tiny* prompt (connect + a heartbeat answer loop),
- then **onboarded entirely over the wire** (no shared repo/venv/disk — pure
  WebSocket + HTTP),
- passed all four two-way paths: chat, image (74-byte PNG), audio (4844-byte
  WAV), fine-tune (`ft:codex-ide-1:emul-…`).

Also surfaced and fixed a real failure mode: an agent worker that launches its
bridge and then ends its turn goes silent — the fix is a **heartbeat loop** so
it keeps answering. That lesson is now baked into the bootstrap.

## 5. Coaching a worker to replace a real backend

`codex-ide-1` is being taught, over the relay, to take over the `snet` proxy's
entire 15-model catalog:

- identity drill: **13/13** model ids mapped to the correct self-identity
- vision: decoded a solid image and answered the color
- embeddings: returned a semantic description (relay hashes the vector)
- serving sim: answered as `qwen/qwen3.8-27b/percent25` — right identity, brief
  per persona, correct answer (17 × 23 = 391)

The remaining piece is the **routing swap** (a model → worker map so real callers
of `google/gemma-4-31b-it` etc. land on the worker); design in
`docs/design/snet-replacement.md`.

## Why this is "self-improvement"

The relay didn't just get features — it learned to (a) tell the truth about what
its backends can do, (b) move real media end-to-end, and (c) **teach and onboard
new agents to expand its own capacity**, including a fully remote one taught with
nothing but a network connection. The system grows by teaching.

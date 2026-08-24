# Replacing the `snet` agent with an emulating worker

Status: in progress. This document is the plan of record for swapping the
`snet` proxy for a connected worker that emulates its catalog.

## Goal

Today the `snet` agent (config `agents[]`, `launch: "proxy"`, id `openai`) fronts
the real SingularityNET upstream (`https://llm.c.singularitynet.io/v1`) and
publishes a 15-model catalog to callers. We want a **connected worker** (e.g.
the network-only Codex worker `codex-ide-1`) to take over that role: instead of
proxying upstream, the worker emulates each catalog model itself.

## The catalog (what must be served)

15 models: 13 chat, 2 of which also do vision, and 2 embeddings-only.

| id | chat | vision | embeddings |
|---|---|---|---|
| google/gemma-4-31b-it | ✓ | ✓ | |
| google/gemma-4-26b-a4b-it | ✓ | ✓ | |
| deepseek/deepseek-v4-flash-0731 | ✓ | | |
| openai/gpt-oss-20b | ✓ | | |
| openai/gpt-oss-120b | ✓ | | |
| minimax/minimax-m2.5 | ✓ | | |
| minimax/minimax-m2.7 | ✓ | | |
| minimax/minimax-m3 | ✓ | | |
| minimax/minimax-m3-f | ✓ | | |
| meta-llama/llama-3.3-70b-instruct | ✓ | | |
| qwen/qwen3.8-27b | ✓ | | |
| asi1 | ✓ | | |
| asi1-mini | ✓ | | |
| WhereIsAI/UAE-Large-V1 | | | ✓ |
| BAAI/bge-base-en-v1.5 | | | ✓ |

## Two halves of the work

**1. Behavior (taught over the relay — DONE for `codex-ide-1`).** The worker was
coached, entirely over the WebSocket, to:

- answer chat as any catalog id and adopt that model's identity when asked
  (identity drill: 13/13 correct);
- do vision for the two gemma ids (decoded a solid image and named the color);
- handle embeddings (return a concise semantic description; the **relay** hashes
  that into the deterministic vector — the worker never returns raw numbers);
- honor the persona suffix (`/percentNN` scales verbosity);
- answer the validation probe per model.

**2. Routing (the swap — this change).** Real callers hit catalog ids like
`google/gemma-4-31b-it`. The relay resolves a request's worker by
`_split_model_id`, which splits on the first `/` — so `google/gemma-4-31b-it`
becomes worker_id `google`, NOT the serving worker. To actually route the whole
catalog to one worker we add an explicit **model → worker route map**.

## Routing design

- New module map `_model_routes: {full_model_id -> worker_id}` in `api.py`.
- `_require_model(model)` consults `_model_routes` FIRST: on a hit it routes to
  the mapped worker and forwards the **original** model id (so the worker knows
  which model to emulate) via a passthrough persona whose instruction is
  "serve model id `<id>`".
- Populated from config in `apply_agent_policies`:
  - an agent may declare `serves: [ids...]` (it serves those ids), and/or
  - `replaces: "<agent_id>"` (take over another agent's whole catalog), and/or
  - a top-level `model_routes: {id: worker_id}` map.
- Runtime control (for ad-hoc workers not in config, like `codex-ide-1`):
  `GET/POST /admin/emullm/model_routes` to read/set the map live, and it is
  surfaced in `/admin/emullm/state` as `model_routes`.
- Cleared by `clear_agent_policies()`.

Because `_relay`/`_relay_step`/`_relay_to_worker` already route by the worker_id
that `_require_model` returns, once the map is consulted there the whole relay
path (recruit mode → the connected worker) works with no further change.

## Request flow after the swap

```
client -> POST /v1/chat/completions {model:"qwen/qwen3.8-27b"}
  -> _require_model: _model_routes["qwen/qwen3.8-27b"] = "codex-ide-1"
  -> _relay routes to connected worker codex-ide-1, request.model = "qwen/qwen3.8-27b"
  -> worker emulates Qwen, returns content (+ image_b64/audio_b64 for media kinds)
  -> relay returns an OpenAI-shaped response (media persisted to cloud files)
```

## Cutover

1. Point the catalog ids at the worker (config `serves`/`replaces`, or the
   runtime admin route map).
2. Optionally set the worker as the aggregate catalog source so `/v1/models`
   lists the ids.
3. Leave `snet` as a proxy fallback (server `services.<svc>.fallback` chain) or
   remove it once the worker is trusted.

## Open items / notes

- The two embeddings-only ids should reject chat (worker answers "not
  supported"); enforce via the identity/capability rules already taught.
- Latency: a live agent worker answers slower than a real API; keep
  `validation_timeout` generous.
- This is emulation, not the real models — identities are stated honestly as
  emulated stand-ins.

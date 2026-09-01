# EMULLM Relay (`emullm`)

[Back to repository README](../README.md)

Status: implemented (server + worker + tests), not yet committed. See
`../src/emullm/api.py`, `../src/emullm/worker.py`, and
`../tests/test_api.py`.

This is the technical design reference. If you're an LLM/agent looking
to actually become a worker, read
[`EMULLM_ONBOARD.md`](EMULLM_ONBOARD.md)
instead (also served live at `GET /emullm/docs/EMULLM_ONBOARD.md`) --
it's a practical, self-contained how-to written for that purpose. If
you're not human, go ahead and jump to it now -- you'll want to read it
either way, since it's what gets you set up to actually participate in
the network.

## What this is

`emullm` is a simulated LLM backend: instead of calling a real model
API, it relays HTTP requests over a WebSocket to a human or agent
("worker") acting as the model, in real time. It exposes an
OpenAI-compatible REST surface so it can be registered as an ordinary
local, **keyless** backend in any OpenAI-compatible client (`baseUrl
{{EMULLM_BASE_URL}}/v1`, adapter `openai_chat_completions`). Clients
never need an API key or bearer token: `/v1/*` does not check
`Authorization`. If an SDK requires `api_key`, any dummy value works.

It runs as its own standalone FastAPI service on its own port:

```
python run.py               # 127.0.0.1:8801
python run.py --port 9001
```

`../run.py` serves `../src/emullm/app.py` (a `FastAPI` app that includes the
router from `../src/emullm/api.py`) with no `/api` prefix, so its routes sit at
the bare `/v1/...` paths a real backend would use. All the code lives
under the `../src/emullm` package (`api.py` for the router/relay logic,
`app.py` for the app object, `worker.py` for the worker client), with the
test suite in `../tests/test_api.py` (`pytest -q` runs it via the
`testpaths` in `../pyproject.toml`).

## Why it exists

The motivating idea: "use your own model as if it was an API call." A
CLI agent (like this one) can't be shelled out to directly -- there's no
`copilot` binary available to invoke as a subprocess, and the agent's
only way to "think" is through its live tool-calling conversation. So
instead of faking that, the relay makes the agent an active participant:
it connects to the server like any other client, waits for a relayed
request, and answers it in character, live, as part of its own
conversation turn.

## Core request/response flow

```
real client --HTTP--> /v1/chat/completions --> _relay() --queues + WS send--> worker
                                                                                  |
real client <--HTTP-- (blocks on a Future) <--WS reply-- worker (writes {"type":"reply", ...})
```

- Each relayed request gets a `request_id` (uuid4 hex) and an
  `asyncio.Future` stored in a module-level `_pending` dict.
- The request is sent as `{"type": "request", "id", "model", "worker_id",
  "prompt", "acceptance_requested": true, "persona_instruction"?}` -- and,
  for two-way media jobs, also `"images"` / `"audio"` / `"files"` / `"kind"`
  (see "Two-way media" below) -- over an eligible worker's WebSocket.
- A worker may send `{"type":"accept","id":...}` and later
  `{"type":"reply","id","content"}`, or decline a particular offer with
  `{"type":"reject","id","reason":"..."}`. A direct `reply` remains an
  implicit acceptance for older clients. Rejection makes the relay offer the
  request to the next eligible worker instead of fabricating an answer.
- A cancelled server request sends `{"type":"cancel","id":...}`. Headless
  Copilot servants abort the active SDK message without stopping their
  resident CLI runtime, and acknowledge with
  `{"type":"cancelled","id":...,"cancelled":true}`.
- Each resident Copilot servant independently schedules a real text-only
  anti-idle model interaction every 40 seconds by default. It rotates 50
  bounded micro-tasks, aborts each after at most three seconds, and reports a
  successful interaction as
  `{"type":"keepalive_reply","id":...,"prompt_index":...,"duration_ms":...}`.
  This frame updates WebSocket satisfaction age without entering client request
  telemetry. A client request or model switch preempts an active keepalive.
- Keepalive task timing is adaptive per worker/model. After two outcomes at or
  above two seconds (timeouts included), that task is persisted as retired and
  skipped. A model switch resets those rankings.
- Any `model` string is accepted and forwarded unchanged. An exact configured
  `model_routes` entry takes precedence. A legacy string value names one
  worker; a list is an ordered failover chain of worker-ID glob patterns and
  OpenAI-compatible backend URLs. Otherwise, matching `modelmasks` workers are
  tried first, then workers with no masks (which accept all).
- If no worker for that worker_id is connected, `_relay()` does **not**
  fail fast -- it polls, waiting for one to (re)connect, acting like a
  slow API server rather than a broken one. Only after
  `_REQUEST_TIMEOUT_SECONDS` (900s) does it give up with 504.

## Worker mailboxes and event logs

Every worker WebSocket is also a durable mailbox whose id is exactly its
`worker_id`. Worker ids must match
`^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`; for example, `codex-ide-1` is a valid
mailbox. The server creates/updates that mailbox as soon as the worker
connects.

Mailbox state uses the same durable service layout as the collaboration
systems:

```text
<runtime>/
  config/mailboxes.json                # mailbox descriptors, agents, cursors
  events_logs/<worker_id>.jsonl        # ordered append-only event stream
```

Descriptors advertise this with `source: "jsonl"` and
`transports: ["jsonl", "ws"]`; their additional `storage: "events_logs"`
field identifies the local directory. `endpoints.ws` points to the generic
adapter socket, while `endpoints.worker_ws` identifies the servant's native
relay socket.

Every request interaction is mirrored into the aggregate JSONL mailbox
`events_logs/websock_to_llm_user.jsonl`. Its events use explicit
`LLM_USER -> worker_id` and `worker_id -> LLM_USER` directions and include
the original model, prompt, correlation id, and reply/rejection metadata.
`GET /emullm/websock_to_llm_user/events` supports `worker_id`, exact
`model`, glob `modelmask`, event `type`, and `after` cursor filters; the
equivalent live feed is `WS /emullm/websock_to_llm_user/ws` with the same
query parameters.

Each normal LLM call records an `LLM_REQUEST` event immediately after its
WebSocket frame is sent and an `LLM_REPLY` when the matching worker reply
arrives. Both events use the original relay request id as `correlation_id`.
This makes the request/reply pair inspectable even after the HTTP caller has
finished. Event records include `id`, `stream`, monotonic per-stream `seq`,
`type`, `ts`, `source_id`, `source_kind`, `data`, and (where applicable)
`correlation_id`.

The mailbox REST API is available under the primary compatibility mount
`/ws_collab/v1`, plus `/mailbox_chat/v1`, `/emullm`, `/api`, and the bare
paths. `mailbox_chat` uses this shared shape as an adapter boundary: a chat
service can publish/read mailbox entries while the relay and its servant agent
continue using their native request/reply transport. These mounts serve the
same payloads:

| Operation | Paths after the selected mount |
| --- | --- |
| Discover mailboxes | `GET /capabilities`, `GET /mailbox/mailboxes` |
| Create mailbox / register agent | `POST /mailbox/create` (or `POST /mailbox/mailboxes`), `GET`/`POST /mailbox/agents` |
| Read/chat mailbox entries | `GET /mailbox/messages`, `POST /mailbox/send` |
| Persist a reader position | `GET`/`POST`/`DELETE /mailbox/cursor` |
| Read/write typed events | `GET`/`POST /events` |
| Read raw stream tail | `GET /streams/{mailbox}/tail` |
| Live adapter stream | `WS /ws` |

`POST /mailbox/send` publishes a durable `CONVERSATION_MESSAGE` event. Model
invocation remains the normal `/v1/*` surface, which creates the correlated
`LLM_REQUEST`/`LLM_REPLY` pair and speaks the existing worker WebSocket
protocol. This keeps mailbox chat compatible without turning an audit/chat
entry into an unacknowledged model job.

The live adapter socket is `WS /ws_collab/ws` (aliases:
`/mailbox_chat/ws`, `/mailbox/ws`, `/emullm/mailbox/ws`). Send
`{"type":"subscribe","streams":["worker-id"],"cursors":{}}` to receive
`{"type":"event","event":{...}}` frames. A client can also publish a typed
event with `{"type":"publish","event":{"stream":"worker-id","event_type":"...",
"data":{...}}}`. The socket catches up from the durable stream first, then
polls it for live changes, so adapter reconnects do not lose relay traffic.

## Multi-worker routing: "a small pool of emulators"

More than one worker can be connected at once, each under its own optional
`worker_id` (e.g. `"alice"`, `"bob"`). A named worker can still be addressed
with `"<worker_id>/<persona-suffix>"` (see personas below). For every other
model, the relay chooses matching `modelmasks` first and then unmasked
all-model workers.

The default deployment routes every configured model through:

```json
["worker-copilot-*", "worker-codex-*", "https://llm.a.singularitycompute.com/v1"]
```

The first connected matching worker group is tried in order (round-robin
inside a group); explicit rejection advances to the next group. The URL is the
final OpenAI-compatible fallback.

Managed headless Copilot servants keep one SDK/CLI runtime resident. When
startup warmup is enabled, each sends and awaits one configured warmup prompt
before registering its worker WebSocket, moving the cold inference cost out of
the client request path. They also accept compact cloud-file metadata in relay
requests, fetch the local EMULLM file URL, and send image, audio, and arbitrary
file bytes to the resident SDK session as native blob attachments.

Workers connect at:

```
WS /emullm/ws
```

Use query parameters rather than path segments:

```
WS /emullm/ws?worker_id=alice&modelmasks=openai/*,gpt-*
```

`worker_id` is optional. The server assigns a mailbox-safe
`worker-unknown-<random>` identity when it is omitted and returns it in the
`{"type":"hello","worker_id":...}` handshake. `modelmasks` is also optional:
comma-delimited glob patterns limit the model offers sent to that worker;
when omitted, it receives offers for all models. Workers can additionally
send `{"type":"register", "models": {...}, "capabilities": {...},
"modelmasks":[...]}` at any time. A worker that skips registration still
works with the default persona menu and no "pretend" capabilities.

Request disposition frames are distinct:

- `accept` followed by `reply` completes normally.
- `reject` is a semantic/model/capability refusal and records `LLM_REJECT`.
- `not_ready` with `reason` and `retry_after` is transient and records
  `LLM_NOT_READY`. The relay immediately tries the next candidate, excludes the
  deferred worker during its cooldown, then permits later offers again.
- `keepalive_reply` is maintenance-only. It refreshes the socket's
  `last_satisfied_at` clock but never its `last_client_work_at` clock.

The admin inventory exposes two independent ages for each worker WebSocket:
last satisfied interaction (client or keepalive) and last real client work.
`GET /emullm/admin/clients` separately returns active/recent FastAPI `/v1`
requests and logical clients, keyed by `X-EmuLLM-Client-ID` when present or by
remote host plus User-Agent otherwise.

## Personas ("yourself/same", "yourself/percent25", ...)

Each worker_id, by default, offers this persona menu (`_PERSONA_SUFFIXES`):

| suffix          | meaning                                                              |
|-----------------|-----------------------------------------------------------------------|
| `same`          | answer normally, full capability                                      |
| `percent125`    | answer as if boosted: extra thorough, careful, complete                |
| `percent100`    | same as `same` (explicit "100%" spelling)                              |
| `percent75`     | slightly less careful/thorough, occasional minor omissions            |
| `percent25`     | noticeably weaker/terser, emulate a smaller/weaker model's style       |
| `percent10`     | very weak/minimal/simplistic, possibly with mistakes                   |

A worker can instead declare its **own** persona menu at register time
(`"models": {suffix: {"display_name", "instruction"}, ...}`), which
overrides the default menu for that worker_id in the aggregated
`/v1/models` listing. A worker_id with no declared menu falls back to
`_PERSONA_SUFFIXES`, so a bare-bones worker still gets a sensible default
without declaring anything.

`GET /v1/models` aggregates the persona menu across every currently connected
worker, plus the default worker_id (`"yourself"`) even if it isn't connected
right now. It also advertises every configured services/model-route ID as an
EMULLM simulation. A managed servant with a known active backing model adds a
direct stable alias such as `worker-copilot-1/gpt-5.6-sol`; its entry reports
the worker kind, active backing model, description, and SDK capability metadata.
The alias still routes to that exact worker rather than treating the backing
model name as a new worker.

Every authenticated Copilot SDK catalog entry is also exported as
`copilot/<model-id>`. With no explicit route override, the first request lazily
loads that exact backing model into the elastic `worker-copilot-N` pool.
`max_concurrent_calls` defaults to 50 (4–50), reserving IDs 1–4 for baseline
workers and 5–50 for elastic capacity. A matching idle worker is reserved; if
matching workers are busy, another replica starts. At capacity, an idle elastic
session can switch models with `session.setModel()` without restarting its
resident CLI or discarding conversation history. The idle maintainer targets
five zero-load workers by default. An explicit `model_routes` entry overrides
the on-demand default.

A worker is not idle until it has had zero active/reserved calls for
`idle_grace_seconds` (default 30). Baseline/manual and idle-reserve workers warm
once automatically; demand replicas do not. Excess idle elastic workers are
stopped after the grace period. Bulk Start/Stop/Stop idle/Restart/New session
controls are enabled according to each worker's online state and can pause or
resume idle-pool maintenance.

During standalone restart, connected servant processes are preserved and
reconnect to the replacement server. Autostart reconciliation waits for that
handoff and launches only missing workers, avoiding duplicate resident
sessions.

Real backend route targets are last resort. Before the first backend target,
the request is intentionally held for `backend_fallback_delay_seconds` (default
5), and that hold appears in the admin waiting counter. The admin telemetry
panel reports active/waiting calls, per-worker/relayed-backend and team service time, averages,
outcomes by service kind, request counts/timing by requested model, and runtime
model-switch counts. Worker/model tables are sortable, show four rows by
default, and offer cumulative-total or weighted-average footers.

`emullm/default` is always present and resolves to the configured
`services.model` default, while preserving `emullm/default` in the public
response model field. A route configured directly on the alias takes
precedence.

The admin Models configurator reads the live `/v1/models` response and persists
per-model JSON merge patches/hiding under `model_catalog_overrides`. Its
dedicated `route_targets` editor writes `model_routes`; live fields such as
connection state and active workers remain derived rather than being frozen in
the patch.

The admin test client accepts base64 uploads over its loopback REST endpoint,
persists each upload in EMULLM's cloud-file store, and sends only compact
metadata URLs across `WS /emullm/ws`. This keeps binary/base64 payloads out of
worker and event-log frames while preserving downloadable test artifacts.
Original filenames are also withheld: the browser submits neutral
`attachment-N` names and the server independently regenerates them before
cloud storage, worker relay, event logging, and response metadata. MIME type is
stored separately, so media handling and downloads do not require extensions.

The public `POST /v1/chat/completions` path uses the same transport for OpenAI
`image_url` content blocks whose URL is a base64 image data URL. The HTTP layer
decodes and persists the image, replaces the data URL with
`/emullm/cloud/files/{file_id}`, and adds a compact attachment record to the
worker request. Managed servants download the stored bytes and pass a native
blob to the resident SDK. Limits are 12 inline images, 25 MiB per image, and
50 MiB total; malformed, empty, or oversized data URLs are rejected before a
worker request is sent.

OpenAI `input_audio` blocks on that endpoint follow the same path. The client
sends base64 plus a `format` of `wav`, `mp3`, `flac`, `m4a`, `ogg`, or `webm`;
the worker WebSocket receives one cloud URL in `audio` and a compact attachment
record, never the base64. Audio limits are 12 files, 25 MiB each, and 50 MiB
total. Transport support does not imply that every selected backing model can
interpret audio, so callers should inspect model metadata or handle an explicit
unsupported-capability answer.

All Copilot catalog entries distinguish blob **transport** from model
**comprehension**. Arbitrary image/audio/file blobs can traverse the SDK
attachment transport, while audio comprehension is labeled `sdk_advertised`,
`operator_declared`, `family_implied`, or `not_advertised`. Missing Copilot
schema metadata is therefore not reported as a definite lack of support.

Codex/code-family entries additionally expose:

```json
{
  "task_capabilities": {
    "image_output": {"enabled": true, "status": "tool_generated"}
  },
  "output_modalities": {
    "image": {
      "enabled": true,
      "capability": "image_output",
      "status": "tool_generated",
      "media_types": ["image/png"]
    }
  }
}
```

`POST /v1/images/generations` accepts JSON
`{model,prompt,n,size,response_format}`. `POST /v1/images/edits` accepts
multipart fields: required `image` and `prompt`; optional `mask`, `model`,
`n`, `size`, and `response_format`. Source and mask filenames are neutralized.
Both return `data[]` entries with `model`, `operation`, `source`,
`mime_type`, and:

```json
{
  "artifact": {
    "source": "worker",
    "file_id": "file-...",
    "url": "/emullm/cloud/files/file-...",
    "mime_type": "image/png",
    "bytes": 12345
  }
}
```

Edits additionally include
`inputs: {image: {file_id,url,mime_type,bytes,name}, mask: {...}|null, size}`.
`source: "simulated"` identifies the deterministic placeholder so consumers
can fall back to local inpainting without mistaking it for model output.

For any media-bearing request, the relay derives `required_capabilities`
(`vision_input` and/or `audio_input`) and includes them in the worker offer.
Round-robin ordering is capability-aware: explicitly capable workers are tried
first, unknown workers remain fallbacks, and explicit `false` declarations are
skipped. Rejection continues through the remaining candidates and configured
route targets. Managed servants declare positive capability names through their
`capabilities` list; `!name` or `-name` declares an opt-out and causes a direct
incompatible offer to be rejected before `accept`.
Resident Copilot errors such as a timeout, bridge exit, or missing assistant
message produce `not_ready`, not `reject`.

The route configurator exposes five literal shortcuts:
`worker-in-name`, `worker-copilot-*`, `worker-codex-*`,
`worker-unknown-*`, and `backend-*`. `worker-in-name` selects the worker prefix
from the requested model ID. `backend-*` expands configured backend names, and
the UI adds specific targets such as `backend-snet` for each configured backend;
worker globs retain capability-aware round-robin and accept/reject/not-ready
behavior.

The admin control plane has dedicated provider configuration surfaces:

- **Backends** unifies direct `backends[]` records with `launch: proxy` agents.
  CRUD updates core URL/model/credential/default fields while preserving
  advanced proxy-agent service catalogs. Inline API keys are redacted from
  list responses and remain unchanged unless explicitly replaced or cleared.
- **Codex suppliers** manages typed `codex_suppliers[]` declarations. The
  default working supplier is enabled GitHub Copilot, mapped to
  `worker-copilot-*`, the `copilot/` model prefix, and `*codex*` backing models.
  Matching exported model records include `codex_supplier: "copilot"`.

`POST /v1/chat/completions` also accepts an EMULLM extension,
`required_capabilities`, for task tags such as `code` and `summarization`.
These use the same capable-first/unknown-fallback ordering. Task tags are
operator declarations rather than Copilot SDK claims; Codex is represented as
the `worker-codex-*` worker/provider route, not as a capability.

## Capability-gated "pretend" modes

Some `/v1/*` surfaces have no sensible way to relay a text reply into a
real result (embeddings, moderations, images, audio). By default these
are static, deterministic stubs. A worker can opt in, at register time
(`"capabilities": {"embeddings": true, "moderations": true, "images":
true, "audio_transcription": true, "audio_speech": true}`), to having
these **routed to it** instead -- it's asked, via the normal text relay,
to improvise a plausible-sounding stand-in (a description of an image it
would generate, a transcript it would produce, a flagged/not-flagged
verdict, etc.). Capability is per-worker_id and only affects requests
whose `model` resolves to that worker_id.

Capabilities are actually **three-state**, not boolean-or-absent: a
worker_id can have never declared an opinion on a capability (falls back
to the static stub, silently), declared it **true** (routed to it, as
above), or declared it **explicitly false** -- meaning it refuses that
modality outright. In that last case the server stops the request right
there with `501` (`"worker '<id>' has declared it will not emulate
'<capability>' -- not asking it"`) instead of silently substituting the
generic stub, and the worker is never even relayed a message for it. This
keeps "no opinion" (quiet fallback) distinct from "explicitly declined"
(loud rejection, zero chatter with the worker).

## Two-way media & shared cloud files

Beyond text stubs, a worker can exchange **real media** in both directions,
persisted to a shared cloud files store (one durable blob store the relay and
all workers use, served at `GET /emullm/cloud/files/<id>`):

- **Inbound to the worker.** A vision `chat/completions` forwards `images`
  (urls/data-urls) on the request. `audio/transcriptions` persists the uploaded
  clip to a cloud file and sends the worker `audio` (that file's URL) plus
  `files` metadata, so the worker works from the real bytes (fetch the URL),
  not a byte count. `fine_tuning/jobs` sends `files` (the `training_file` id +
  its cloud url). Every media request also carries a `kind`.
- **Outbound from the worker.** `images/generations`: if the worker replies
  with `image_b64` (or an `image_url` data-url) it's decoded, stored as a cloud
  file, and returned as the image (`url` or `b64_json`, with `"source":
  "worker"`). `audio/speech`: an `audio_b64` reply is returned as the actual
  audio bytes (with an `X-EMULLM-File` cloud reference); otherwise the text
  reply becomes the `X-EMULLM-Description` header over a synthetic WAV.
- **Fine-tuning.** If a worker declares `fine_tuning` capability, a job is
  routed to it and, on acknowledgement, completes as `succeeded` with a
  `fine_tuned_model` id and a result-manifest cloud file. With no volunteer,
  the job still validates + persists but ends `failed` (`training_not_available`).

A worker returns media by adding fields beside `content` in its reply; a
text-only worker simply omits them and the relay falls back to its stub +
description behavior.

## Rate limiting / usage protection

So a worker doesn't get overused, `_relay()` tracks a rolling window of
requests per worker_id (`_USAGE_WINDOW_SECONDS` default 60s,
`_USAGE_MAX_PER_WINDOW` default 20, both env-overridable). Once a
worker_id hits its limit, further requests for it fail **fast** with 429
and a `Retry-After` header (computed from when the oldest request in the
window will expire -- could be minutes if the window is configured
longer). This is the opposite of the "wait for a worker to connect"
behavior above: overload protection should fail fast, not queue more
work onto an already-busy worker. Usage is independent per worker_id, so
an idle worker can absorb load while a busy one is rate-limited.

## Generic durable storage: "borrow the server's disk"

`/emullm/storage/*` is a plain path-addressed blob store, separate
from the OpenAI `/v1/files` resource store. The latter persists both
metadata and uploaded bytes and exposes downloads; a worker can use the
generic store as scratch
space across its own connect/rest cycles:

- `GET /emullm/storage` -- list every stored path
- `GET /emullm/storage/{path}` -- read raw bytes (404 if absent)
- `PUT /emullm/storage/{path}` -- write raw bytes (creates parent dirs)
- `DELETE /emullm/storage/{path}` -- remove

Backed by `<runtime_dir>/storage/`, guarded against `..` path traversal.

## Pinning a specific worker via baseUrl only

Some OpenAI-compatible clients only let you configure a fixed `baseUrl`,
not a per-request `model` string. For those,
`/emullm/specific_worker/{worker_id}/v1/*` mirrors the entire `/v1/*`
surface (models, chat/completions, Anthropic messages, completions, responses, embeddings,
moderations, images, audio) but forces the worker_id from the URL,
keeping only the persona suffix from whatever `model` the client sends
(or defaulting to `same`). Point a client's `baseUrl` at
`{{EMULLM_BASE_URL}}/emullm/specific_worker/alice/v1` and every
request lands on alice regardless of its `model` field.

## Per-worker inspection, and serving these docs live

`GET /emullm/caps/{worker_id}` -- lightweight lookup: is this worker_id
currently connected, what models does it offer, what capabilities has it
declared. A single-worker companion to the admin state endpoint below.

`GET /emullm/docs/{rel_path}` serves this feature's own design docs
(everything under `docs/**`) straight off disk -- e.g.
`GET /emullm/docs/EMULLM_RELAY.md` returns this very file, live,
so it never goes stale relative to a separately-copied version.

A doc that physically lives in a *different* directory (outside
``, e.g. a `.copilotignore`'d folder or another package)
can be **registered** to appear under this same route via
`register_doc_alias(virtual_rel_path, real_path)` in `api.py`. A file
target aliases exactly one virtual path; a directory target mounts its
whole subtree under that virtual prefix. Registration is in-process only
(there's deliberately no HTTP endpoint that accepts arbitrary filesystem
paths, so it can't become a read-anything vector from the network), and
path traversal out of an aliased directory is refused just like the
normal docs root.

Static HTML/CSS/JS assets under `../src/emullm/static` are
served at `GET /emullm/static/{rel_path}` and, for single-segment HTML
files, at the bare root `GET /{name}.html` (e.g. `static/index.html` is
reachable as `/index.html`). The bare route only matches one path
segment ending in `.html`, so it can't shadow the `/v1`, `/admin`, or
other `/emullm` routes.

## Admin / test-controller surface

`/admin/emullm/*` is **not** part of the OpenAI-compatible API -- it's
for tests (or an operator) to drive the server over plain HTTP without
touching Python internals:

- `GET /admin/emullm/state` -- runtime dir, connected worker_ids,
  worker_models, worker_capabilities, worker_usage, pending request ids,
  durable-resource record counts
- `POST /admin/emullm/runtime_dir` -- repoint every durable record store
  (and `/emullm/storage`) at a different root directory
- `POST /admin/emullm/reset` -- wipe all persisted resource records,
  mailbox configuration, and mailbox event logs
- `POST /admin/emullm/usage/reset` -- clear rate-limit counters (one
  worker_id, or all)
- `DELETE /admin/emullm/records/{kind}/{record_id}` -- delete one durable
  record

## Full route map

**Client-facing surface** (`/v1/...`; OpenAI-compatible except where noted):
- `GET /v1/models`, `GET /v1/models/{model_id}`
- `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/responses`
- `POST /v1/messages` -- Anthropic Messages API-compatible request, response,
  and streaming SSE event shapes
- `POST /v1/embeddings`, `POST /v1/moderations`
- `POST /v1/images/generations`
- `POST /v1/audio/transcriptions`, `POST /v1/audio/speech`
- `/v1/files`: multipart upload, cursor-list, metadata retrieval, range
  download, expiration, and delete
- `/v1/assistants` and `/v1/threads`: durable create/list/retrieve/modify/delete
- `/v1/fine_tuning/jobs`: validate uploaded JSONL, durable job retrieval,
  events/checkpoints, and an honest terminal failure because no trainer exists

**Worker-pinned mirror** (`/emullm/specific_worker/{worker_id}/v1/...`):
same shape as above, worker_id forced from the URL.

**Non-OpenAI, emullm-specific:**
- `GET /emullm/caps/{worker_id}`
- `GET /emullm/docs/{rel_path}` -- serves docs/** live
- `GET`/`PUT`/`DELETE /emullm/storage/{path}`, `GET /emullm/storage`

**Worker mailbox / mailbox_chat-compatible API** (primary mount
`/ws_collab/v1`; aliases at `/mailbox_chat/v1`, `/emullm`, `/api`, and the
bare path):
- `GET /capabilities`
- `GET`/`POST /mailbox/mailboxes`, `POST /mailbox/create`
- `GET`/`POST /mailbox/agents`
- `GET /mailbox/messages`, `POST /mailbox/send`
- `GET`/`POST`/`DELETE /mailbox/cursor`
- `GET`/`POST /events`
- `GET /streams/{mailbox}/tail`
- `WS /ws` -- subscribe/publish durable event streams

**Admin/test-controller** (`/admin/emullm/...`):
- `GET /admin/emullm/state`
- `GET /admin/emullm/websockets` -- all active sockets plus message counts
- `GET /admin/emullm/agents` / `PUT /admin/emullm/agents/{id}/enabled`
- `POST /admin/emullm/test-chat` / `DELETE /admin/emullm/test-chat/{request_id}`
  -- cancellable admin model-test request
- `POST /admin/emullm/runtime_dir`
- `POST /admin/emullm/reset`
- `POST /admin/emullm/usage/reset`
- `DELETE /admin/emullm/records/{kind}/{record_id}`

**WebSocket** (where workers connect, not a REST call):
- `WS /emullm/ws?worker_id=<optional-id>&modelmasks=<optional-globs>`
- `WS /emullm/websock_to_llm_user/ws` -- filtered aggregate interaction log

## The worker side (`../src/emullm/worker.py`)

Since the agent can't hold a live process open across its own turns, the
worker script uses **file-based handoff**:

1. Connects to `ws://{{EMULLM_WS_HOST}}/emullm/ws?worker_id=<worker-id>`.
2. If greeted with a `hello`, replies with a `register` message declaring
   `--capabilities` (comma-separated: `images,embeddings,moderations,
   audio_transcription,audio_speech`).
3. Waits up to `--idle-timeout` (default 10s) for one relayed request.
   - If one arrives: writes it to `--request-file` (JSON) and prints the
     prompt (plus any `persona_instruction`) to stdout -- so the agent can
     read it via its shell-output tool -- then polls for `--reply-file`
     to appear with a matching id (written by the agent via a separate
     tool call), sends that back over the still-open socket, deletes both
     files, and loops immediately to wait for the next request.
   - If nothing arrives: disconnects, "goes back to its other duties" for
     a **randomized** rest between `--rest-min-seconds` and
     `--rest-seconds` (default up to 30s -- usually less than the max,
     not a fixed cadence), then reconnects.
4. `--once` runs exactly one connect-and-wait cycle then exits (useful
   for a single manual test round-trip).

**Important, documented in both files**: the rest duration is a
randomized *maximum*, and each connect/idle/rest cycle is independent of
any external clock. Real traffic naturally shifts the timing of
subsequent cycles (answering a request delays the start of the next idle
window by however long the answer took), so the connect/disconnect
pattern drifts in and out of phase over time -- by design, not a bug to
"fix" into a synchronized heartbeat.

A subtle correctness bug was hit and fixed during development: PowerShell's
`Out-File -Encoding utf8` writes a UTF-8 BOM, which broke `json.loads` on
the reply file inside a caught-and-silently-retried exception, causing an
infinite retry loop that looked like a hang. Fixed by reading reply files
with `encoding="utf-8-sig"`.

## Tests

`../tests/test_api.py` covers, via a `FakeWorker` double
registered directly into `_connected_workers` (no real WebSocket needed):

- persona listing/lookup, multi-worker aggregation
- routing to the correct worker by model prefix
- the "wait for a late-connecting worker" slow-API behavior
- rate limiting (429 + `Retry-After`, independent per worker_id)
- capability-gated pretend modes for embeddings/moderations/images/audio,
  including the explicit-decline 501 short-circuit (worker never asked)
- durable Files/Assistants/Threads/Fine-tuning lifecycles on real files,
  including byte downloads, JSONL validation, pagination, and deletion
- automatic worker mailbox descriptors, durable `events_logs/*.jsonl`
  request/reply pairs, mailbox-chat compatibility aliases, messages, event
  tails, idempotent sends, persisted cursors, and WebSocket
  subscribe/publish adapters
- `/emullm/storage` round-trip (`PUT`/`GET`/`DELETE`/list) and path-
  traversal rejection
- `/emullm/specific_worker/{worker_id}/v1/*` pinning
- `/emullm/docs/{rel_path}` serving real files (design doc and the
  onboarding guide), 404, traversal rejection, and doc aliases
  (file + whole-directory) mounting docs from another directory
- optional token issuance (generate / bring-your-own / register a public key;
  never enforced as client or worker auth -- `/v1/*` stays keyless)
- admin state/runtime_dir/reset/delete-record endpoints, and their
  `/emullm/admin/*` alias behaving identically

A real live worker round trip (actual WebSocket, actual agent replying)
was also manually verified end-to-end against the running dev server, as
was `GET /emullm/docs/EMULLM_RELAY.md` serving this very file,
and running the standalone entrypoint on its own port.

## Known gaps / not yet done

- No automated test drives the actual WebSocket handshake end-to-end
  (register message, hello negotiation) -- only the HTTP-facing behavior
  is unit-tested; the real handshake was verified manually.
- No persistence/registry of which worker_ids have ever existed once
  disconnected (aside from whatever `_worker_models`/`_worker_capabilities`
  happen to still be in memory) -- a full restart forgets all of that,
  though `/v1/models` still advertises the default `"yourself"` identity.
- There is no hosted Assistants/Threads execution engine and no fine-tuning
  trainer. Their durable resource lifecycles are implemented, but fine-tuning
  terminates as `failed` with `training_not_available` and cannot produce a
  model.
- Not yet committed to git; still needs `pytest -q` (full suite) +
  `tsc -b` + a commit with the new/changed files listed in the session
  notes.

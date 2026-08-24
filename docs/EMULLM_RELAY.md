# EMULLM Relay (`emullm`)

[Back to repository README](../README.md)

Status: implemented (server + worker + tests), not yet committed. See
`../emullm/api.py`, `../emullm/worker.py`, and
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

`../run.py` serves `../emullm/app.py` (a `FastAPI` app that includes the
router from `../emullm/api.py`) with no `/api` prefix, so its routes sit at
the bare `/v1/...` paths a real backend would use. All the code lives
under the `../emullm` package (`api.py` for the router/relay logic,
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
  "prompt", "persona_instruction"?}` -- and, for two-way media jobs, also
  `"images"` / `"audio"` / `"files"` / `"kind"` (see "Two-way media" below)
  -- over whichever worker's WebSocket matches the model's worker_id.
- The worker's `{"type": "reply", "id", "content"}` message resolves the
  matching future, which becomes the HTTP response. A worker may also
  include real media on the reply (`"image_b64"`/`"image_url"`/`"mime"`,
  `"audio_b64"`/`"audio_url"`); the relay persists it to the shared cloud
  files store and hands the caller a stable URL.
- If no worker for that worker_id is connected, `_relay()` does **not**
  fail fast -- it polls, waiting for one to (re)connect, acting like a
  slow API server rather than a broken one. Only after
  `_REQUEST_TIMEOUT_SECONDS` (900s) does it give up with 504.

## Multi-worker routing: "a small pool of emulators"

More than one worker can be connected at once, each under its own
`worker_id` (e.g. `"yourself"`, `"alice"`, `"bob"`). A model id is
`"<worker_id>/<persona-suffix>"` (see personas below), and a request for
that model is routed to whichever worker is currently registered under
that worker_id -- not just "whoever happens to be connected."

Workers connect at:

```
WS /emullm/{worker_id}/ws
```

worker_id comes straight from the URL path. On connect, the server
sends a `{"type": "hello", "worker_id": ...}` handshake and waits
(briefly, ~10s) for an optional
`{"type": "register", "models": {...}, "capabilities": {...}}` reply (see
below). A worker that skips this (or an older/simpler client that never
registers) still works fine, just with the default persona menu and no
"pretend" capabilities.

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

`GET /v1/models` aggregates the persona menu across every currently
connected worker, plus the default worker_id (`"yourself"`) even if it
isn't connected right now, so the primary identity is always
discoverable.

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
all workers use, served at `GET /emullm/cloud/files/<id>` -- the swapped
spelling `/emullm/cloud/files/<id>` is accepted too):

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
surface (models, chat/completions, completions, responses, embeddings,
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

Static HTML/CSS/JS assets under `../emullm/static` are
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
- `POST /admin/emullm/reset` -- wipe all persisted resource records
- `POST /admin/emullm/usage/reset` -- clear rate-limit counters (one
  worker_id, or all)
- `DELETE /admin/emullm/records/{kind}/{record_id}` -- delete one durable
  record

## Full route map

**OpenAI-compatible client-facing surface** (`/v1/...`):
- `GET /v1/models`, `GET /v1/models/{model_id}`
- `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/responses`
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

**Admin/test-controller** (`/admin/emullm/...`):
- `GET /admin/emullm/state`
- `POST /admin/emullm/runtime_dir`
- `POST /admin/emullm/reset`
- `POST /admin/emullm/usage/reset`
- `DELETE /admin/emullm/records/{kind}/{record_id}`

**WebSocket** (where workers connect, not a REST call):
- `WS /emullm/{worker_id}/ws`

## The worker side (`../emullm/worker.py`)

Since the agent can't hold a live process open across its own turns, the
worker script uses **file-based handoff**:

1. Connects to `ws://{{EMULLM_WS_HOST}}/emullm/<worker-id>/ws`.
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

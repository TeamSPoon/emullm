"""emullm worker: the client side of workbench/server/emullm_api.py.

Connects to ws://<host>/emullm/<worker-id>/ws (worker-id is a plain
identity in the URL, e.g. "yourself", "alice", "bob" -- a small pool of
these can be connected at once, each independently routed to) and waits
up to --idle-timeout seconds for one relayed request. If one arrives, it
is written to --request-file as JSON and this script then waits
(polling) for --reply-file to appear with a matching id, sends that
reply back over the socket, deletes both files, and immediately loops to
wait for the next request (no rest in between while there's active
traffic). If nothing arrives within --idle-timeout seconds, the socket is
closed, the worker "goes back to its other duties" for a randomized rest
(between --rest-min-seconds and --rest-seconds, so it's usually less
than the max), then reconnects and waits again -- so an idle worker
isn't permanently tying up a connection.

On connect, the server may send a {"type":"hello",...} handshake message
asking this worker to register; if it arrives within a couple seconds,
this script replies with {"type":"register","capabilities":{...}}
declaring which non-text stub surfaces (--capabilities) it's willing to
"pretend" at, e.g. describing an image it would have generated. Requests
for a modality this worker didn't declare are simply never routed here
(the server uses its own static stub instead).

The request's requested "model" (see /v1/models on the server -- e.g.
"yourself/same", "yourself/percent25") is included in the request file
and printed alongside the prompt, so the worker (a human or an agent)
knows which persona/capability level to answer as.

Two-way media: a relayed request may carry real media for you to work
from -- "images" (urls/data-urls for a vision chat), "audio" (a cloud
file URL of a clip to transcribe), "files" (e.g. a fine-tune training
file + its cloud url), and "kind" (chat/vision/image/audio_speech/
audio_transcription/fine_tuning). Fetch any URL from your host (prefix a
relative /emullm/cloud/files/... path with the server's http base) to
get the actual bytes. You may likewise RETURN real media in your reply:
add "image_b64" (base64 image, with "mime") for image generation, or
"audio_b64" for speech. The relay persists whatever you return to the
shared cloud files store and hands the caller a stable URL.

NOTE FOR FUTURE MAINTAINERS / OTHER LLM WORKERS: the --rest-seconds
value (default 30) is intentionally a randomized MAX, not a fixed
cadence, and each connect/idle/rest cycle is independent of any external
clock. Do not "fix" this into a synchronized fixed-interval heartbeat.
The reasoning: if a request lands partway through an idle-timeout
window, handling it (however long that takes) shifts the START of the
worker's NEXT idle window later by exactly that amount -- there is no
attempt to stay aligned to wall-clock minute/30s boundaries. So real
traffic naturally drifts the connect/rest cycle in and out of phase over
time; that drift is expected and harmless, not a bug to "correct" by
re-synchronizing to a fixed schedule.

Usage:
    python -m emullm.worker --once   # one connect-and-wait cycle, then exit
    python -m emullm.worker          # loop forever (Ctrl+C to stop)
    python -m emullm.worker --worker-id alice --capabilities images,embeddings

While a request is pending (request file written, reply file not yet
present), answer it by writing --reply-file as JSON:
    {"id": "<same id as the request>", "content": "<your reply text>"}
To return real media two-way, add e.g. "image_b64"/"mime" (image gen)
or "audio_b64" (speech) alongside "content".
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

import websockets

DEFAULT_HOST_WS_URL = "ws://127.0.0.1:8801"
DEFAULT_WORKER_ID = "yourself"
DEFAULT_REQUEST_FILE = Path(__file__).resolve().parents[1] / "runtime" / "emullm_request.json"
DEFAULT_REPLY_FILE = Path(__file__).resolve().parents[1] / "runtime" / "emullm_reply.json"


async def _wait_for_reply(request_id: str, reply_file: Path, timeout: float) -> dict:
    """Poll for the reply file and return the whole reply object (not just the
    text). A worker may include real media alongside ``content`` -- e.g.
    ``image_b64``/``image_url``/``mime`` for image generation, or
    ``audio_b64``/``audio_url`` for speech -- and those are forwarded to the
    relay, which persists them to the shared cloud files store."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if reply_file.exists():
            try:
                # utf-8-sig tolerates a leading BOM (e.g. PowerShell's
                # `Out-File -Encoding utf8` writes one) that would otherwise
                # make json.loads fail forever and silently retry-loop here.
                data = json.loads(reply_file.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                await asyncio.sleep(0.5)
                continue
            if isinstance(data, dict) and str(data.get("id")) == request_id:
                reply_file.unlink(missing_ok=True)
                return data
        await asyncio.sleep(0.5)
    raise TimeoutError(f"no reply written to {reply_file} within {timeout}s")


# Optional media/reference fields a worker may return two-way beside "content".
_REPLY_MEDIA_KEYS = ("image_b64", "image_url", "audio_b64", "audio_url", "mime", "file_id", "file_url", "images")


async def _maybe_register(websocket, capabilities: dict[str, bool], role: str = "") -> None:
    """If the server greets us with a "hello" handshake within a couple
    seconds, declare our capabilities (and optional role) in reply. An
    older/simpler server that doesn't send a hello is tolerated -- we just
    skip registering."""
    try:
        hello = await asyncio.wait_for(websocket.recv(), timeout=3.0)
        data = json.loads(hello)
    except (asyncio.TimeoutError, json.JSONDecodeError):
        return
    if isinstance(data, dict) and data.get("type") == "hello":
        message: dict[str, object] = {"type": "register", "capabilities": capabilities}
        if role:
            message["role"] = role
        await websocket.send(json.dumps(message))


async def _run_once(
    ws_url: str,
    request_file: Path,
    reply_file: Path,
    idle_timeout: float,
    reply_timeout: float,
    capabilities: dict[str, bool],
    role: str = "",
) -> bool:
    """One connect. Returns True if a request was handled (caller should
    reconnect immediately), False if idle (caller should rest first)."""
    async with websockets.connect(ws_url) as websocket:
        await _maybe_register(websocket, capabilities, role)

        try:
            raw = await asyncio.wait_for(websocket.recv(), timeout=idle_timeout)
        except asyncio.TimeoutError:
            print(f"IDLE: no request within {idle_timeout}s, disconnecting", flush=True)
            return False

        data = json.loads(raw)
        if data.get("type") != "request":
            return True

        request_id = str(data["id"])
        request_file.parent.mkdir(parents=True, exist_ok=True)
        request_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        persona_note = ""
        if data.get("persona_instruction"):
            persona_note = f"\nPERSONA INSTRUCTION: {data['persona_instruction']}"
        media_note = ""
        incoming = [k for k in ("images", "audio", "files") if data.get(k)]
        if data.get("kind") not in (None, "chat") or incoming:
            media_note = f"\nMEDIA: kind={data.get('kind')} attached={incoming} (fetch any URL from your host to get the bytes)"
        print(
            f"REQUEST {request_id} (model={data.get('model', '?')}):{persona_note}{media_note}\n{data.get('prompt', '')}\n---",
            flush=True,
        )

        reply = await _wait_for_reply(request_id, reply_file, reply_timeout)
        message: dict[str, object] = {"type": "reply", "id": request_id, "content": str(reply.get("content") or "")}
        for key in _REPLY_MEDIA_KEYS:
            if reply.get(key) is not None:
                message[key] = reply[key]
        await websocket.send(json.dumps(message))
        returned = [k for k in _REPLY_MEDIA_KEYS if k in message]
        print(f"REPLIED to {request_id}" + (f" (+media: {returned})" if returned else ""), flush=True)
        return True


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker-id", default=DEFAULT_WORKER_ID, help="identity this worker connects/registers as (used in the URL path)")
    parser.add_argument("--host-ws-url", default=DEFAULT_HOST_WS_URL, help="server base, e.g. ws://127.0.0.1:8801 -- worker-id is appended as /emullm/<worker-id>/ws")
    parser.add_argument("--ws-url", default=None, help="full websocket URL override; if given, --worker-id/--host-ws-url are ignored")
    parser.add_argument(
        "--capabilities",
        default="",
        help="comma-separated list of non-text stub surfaces this worker is willing to 'pretend' at, "
        "e.g. 'images,embeddings,moderations,audio_transcription,audio_speech,fine_tuning'",
    )
    parser.add_argument("--request-file", type=Path, default=DEFAULT_REQUEST_FILE)
    parser.add_argument("--reply-file", type=Path, default=DEFAULT_REPLY_FILE)
    parser.add_argument("--idle-timeout", type=float, default=10.0, help="seconds to wait for a request per connection")
    parser.add_argument("--rest-seconds", type=float, default=30.0, help="max seconds to rest after an idle cycle before reconnecting (actual rest is randomized between --rest-min-seconds and this)")
    parser.add_argument("--rest-min-seconds", type=float, default=1.0, help="min seconds to rest after an idle cycle before reconnecting")
    parser.add_argument("--reply-timeout", type=float, default=3600.0, help="max seconds to wait for the reply file once a request arrives")
    parser.add_argument("--once", action="store_true", help="run exactly one connect-and-wait cycle, then exit")
    parser.add_argument("--role", default="", help="self-declared role/phase shown on the status page, e.g. 'trusted' or 'training'")
    args = parser.parse_args()

    ws_url = args.ws_url or f"{args.host_ws_url.rstrip('/')}/emullm/{args.worker_id}/ws"
    capabilities = {name.strip(): True for name in args.capabilities.split(",") if name.strip()}

    while True:
        try:
            handled = await _run_once(
                ws_url, args.request_file, args.reply_file, args.idle_timeout, args.reply_timeout, capabilities, args.role
            )
        except (ConnectionRefusedError, OSError) as error:
            print(f"connect failed: {error}", flush=True)
            handled = False
        except Exception as error:  # noqa: BLE001 -- keep the loop alive regardless
            print(f"error: {error}", flush=True)
            handled = False

        if args.once:
            return
        if not handled:
            rest_for = random.uniform(min(args.rest_min_seconds, args.rest_seconds), args.rest_seconds)
            print(f"resting {rest_for:.1f}s (up to {args.rest_seconds:.0f}s) before reconnecting", flush=True)
            await asyncio.sleep(rest_for)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)

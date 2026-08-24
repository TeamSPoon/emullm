# Running the worker in the cloud, keeping the two-way link

Research + design for moving a worker (e.g. the Codex worker) off the local
machine onto OpenAI Codex cloud / GitHub Copilot cloud agent, while preserving
two-way "socketing" back to the relay.

## Egress reality (why localhost won't work)

- **OpenAI Codex cloud:** outbound network is **off by default**. It can be
  enabled per environment, with a **domain allowlist**. The *setup* phase may
  have network (installing deps); the *task* phase runs offline unless internet
  is explicitly enabled for that environment.
- **GitHub Copilot cloud agent:** outbound is restricted to an allowlist;
  org/repo admins can add custom domains. Allowlisting is by host/domain, not by
  protocol — so an allowlisted host permits HTTP(S) **and** `ws://`/`wss://`.
- Therefore a cloud worker: (a) cannot reach `127.0.0.1:8801`; (b) can only
  reach allowlisted hosts; (c) cannot host an inbound listener we can dial.

## Path A — direct socket (preferred)

1. Expose the relay on a public host: a tunnel (`cloudflared`, `ngrok`) or a
   real deploy. Prefer `wss://` on 443.
2. Allowlist that host in the cloud env (Codex environment internet settings;
   Copilot org/repo firewall allowlist).
3. The worker connects **out** to `wss://<relay-host>/emullm/<id>/ws`. The
   socket is full-duplex, so both directions work: jobs down, worker messages up
   (relay inbox/back-channel). Media travels by cloud-file URL on the same
   allowlisted host.
4. Drop the worker-hosted listener (no inbound in cloud). Two-way is preserved
   via the single outbound socket + the relay as rendezvous. "Find each other
   again" = reconnect loop + stable `worker_id` (the relay is the directory).

## Path B — dead-drop transport (when you can't allowlist a custom host)

If the only reachable hosts are fixed allowlisted ones (e.g. `github.com`),
tunnel the same protocol over a shared read/write medium both sides can reach —
a store-and-forward "dead drop":

- **Rendezvous options:** a GitHub gist (github.com is commonly allowlisted for
  the Copilot cloud agent), a pastebin, **email mailboxes** (see below), or any
  HTTPS key/value the env allows.
- **Protocol (polling):** the coach writes a job JSON to a known slot; the worker
  polls, reads, executes, writes a reply JSON to a reply slot; the coach polls
  the reply. Same `{id, prompt, kind, ...}` schema as the socket path.
- **Media:** pass by URL (cloud files on an allowlisted host), or inline base64
  in the paste when small.
- **Trade-off:** higher latency (polling interval) but works through the
  strictest HTTP-only allowlist — no custom domain or open port required.

### Path B variant — email as the transport

Even simpler to obtain in a locked-down box: **register a mailbox** for each side
and carry jobs/replies as messages.

- Each side has an address (`worker@…`, `coach@…`), which is the durable
  "identity" — the rediscovery handle (like `worker_id`).
- **Send** a job/reply as one email whose body (or attachment) is the job JSON;
  the `Subject`/a header carries the request `id` and `kind` for correlation.
- **Receive** by polling the inbox (IMAP/POP, or a mail API) for new messages
  addressed to you; process, then reply — threading (In-Reply-To) keeps the
  request/reply paired.
- **Media** rides as an attachment or a cloud-file URL.
- Works anywhere SMTP/IMAP (or a mail HTTP API) is reachable — often true even
  when arbitrary sockets/domains are blocked. Latency is mailbox-poll bound.

## What carries over unchanged

- The job/reply schema, personas, model-route map, and media-by-cloud-file all
  work over either transport.
- The two-way message channel becomes: worker→coach via the relay inbox (Path A)
  or the reply slot (Path B); coach→worker via jobs / the job slot.
- Resilience: a cloud env's own re-run is the outer watchdog; a reconnect/poll
  loop + stable `worker_id` is the rediscovery.

## Recommendation

Path A (tunnel + allowlist) for a real deployment. Keep Path B (gist/paste
dead-drop) as the universal fallback for locked-down environments — it needs
nothing but one shared HTTPS endpoint both ends can already reach.

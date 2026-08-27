# Remote MAF Handoff Bridge (test app)

Standalone FastAPI/WebSocket server that wraps a Microsoft Agent Framework (MAF) handoff
`Workflow` and exposes it to OpenCode's `/agent` remote picker. This is the WS1 workstream from
`opecode-sprints/remote-maf-handoff-agents/design-proposal.md`.

**This is test-only scaffolding.** It lives under `opencode/test/` for now so it can be
iterated on alongside the OpenCode changes that consume it, and is expected to move out into
its own repository once the pattern is proven (per the original PoC request) — do not treat
its location as permanent product structure.

## What it is

- `GET /agents/manifest` — lists configured orchestrators and their participants.
- `WS /agents/{id}/session` — one Workflow instance per connection; speaks the wire protocol
  defined in `app/protocol.py` (`user_message`, `tool_result`, `steer_to_agent` in;
  `assistant_delta`, `handoff`, `tool_call`, `turn_complete`, `error` out).
- `app/orchestrator.py` — the sample "support" handoff group (triage/billing/refunds) used for
  manual demos (see `../../opecode-sprints/remote-maf-handoff-agents/test-requirements.md`).
- `app/telemetry.py` — exports OTel spans to Phoenix when `PHOENIX_COLLECTOR_ENDPOINT` is set;
  no-ops otherwise (Phoenix stays a passive telemetry consumer, never a control-plane
  dependency).

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export OPENAI_API_KEY=...          # required for the real sample agents in app/orchestrator.py
uvicorn app.main:app --reload --port 8000
curl localhost:8000/agents/manifest
```

## Test

```bash
source .venv/bin/activate
python -m pytest tests/ -q
```

Tests use deterministic fake `Agent`/`BaseChatClient` fixtures (no `OPENAI_API_KEY`, no network
egress) modeled on `agent-framework/python/packages/orchestrations/tests/test_handoff.py`'s
`MockChatClient`/`MockHandoffAgent` pattern.

## Docker

```bash
docker build -t remote-maf-handoff-bridge .
docker run -p 8000:8000 -e OPENAI_API_KEY=... remote-maf-handoff-bridge
curl localhost:8000/agents/manifest
```

> Note: the Dockerfile's packaging path (`pip install .` picking up the top-level `app`
> package) was verified locally via a non-editable `pip install .` into a fresh venv plus a
> successful `GET /agents/manifest` — the `docker build` itself could not be run in the
> environment this was authored in (no Docker daemon available). Re-verify `docker build`
> once a daemon is available, before relying on the image for the Layer 4 manual demos.

See WS5 (`compose-dev-env`) for the full `docker-compose.yml` wiring this up alongside Phoenix.

"""FastAPI/WebSocket server bridging a MAF handoff `Workflow` to OpenCode's remote-agent
wire protocol (see protocol.py). One Workflow instance is created per WebSocket connection.

Endpoints:
  GET  /agents/manifest        -> Manifest of configured orchestrators + participants.
  WS   /agents/{id}/session    -> Chat + tool-bridge + handoff-status channel for one session.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import AsyncIterable

from agent_framework import AgentResponse, AgentResponseUpdate, AgentSession, WorkflowEvent
from agent_framework.orchestrations import HandoffAgentUserRequest, HandoffSentEvent
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter, ValidationError

from opentelemetry import trace

from .orchestrator import PATTERN_SEMANTICS, OrchestratorSpec, default_orchestrators
from . import checkpoints, sessions
from .protocol import (
    AssistantDeltaFrame,
    ClientFrame,
    HandoffFrame,
    Manifest,
    SessionResumedFrame,
    SteerToAgentFrame,
    ToolCallFrame,
    ToolResultFrame,
    TurnCompleteFrame,
    UserMessageFrame,
)
from .telemetry import configure_telemetry

logger = logging.getLogger("remote_maf_handoff_bridge")
# Resolved lazily per span, so it picks up whichever TracerProvider `configure_telemetry()`
# installed regardless of import order; a no-op provider when telemetry is unconfigured.
_tracer = trace.get_tracer("remote_maf_handoff_bridge")

# Timeout for a single client-executed tool call round trip. Chosen generously since it covers
# real shell commands/file edits a human may need to approve locally, not just fast lookups.
TOOL_CALL_TIMEOUT_SECONDS = 300


def create_app(orchestrators: list[OrchestratorSpec] | None = None) -> FastAPI:
    """Build the FastAPI app. `orchestrators` is injectable so tests can supply fake-agent
    orchestrators instead of the real OpenAI-backed sample registry in `orchestrator.py`.
    """

    registry = {spec.id: spec for spec in (orchestrators if orchestrators is not None else default_orchestrators())}
    configure_telemetry()
    app = FastAPI(title="opencode-remote-maf-handoff-bridge")

    @app.get("/agents/manifest", response_model=Manifest)
    async def get_manifest() -> Manifest:
        return Manifest(orchestrators=[spec.manifest_entry() for spec in registry.values()])

    @app.websocket("/agents/{orchestrator_id}/session")
    async def session(
        websocket: WebSocket,
        orchestrator_id: str,
        start_agent: str | None = None,
        session_id: str | None = None,
        solo: bool = False,
    ) -> None:
        spec = registry.get(orchestrator_id)
        if spec is None:
            await websocket.close(code=4404, reason=f"unknown orchestrator '{orchestrator_id}'")
            return
        # `start_agent` lets a client address any participant in the network directly instead of
        # always entering through the orchestrator's default start agent. Handoffs still work
        # normally from wherever the conversation starts.
        if start_agent is not None and start_agent not in spec.participant_ids:
            await websocket.close(code=4404, reason=f"unknown participant '{start_agent}' in '{orchestrator_id}'")
            return
        # Only patterns whose entry point is a client decision can honour `start_agent`. In group
        # chat, magentic, sequential and concurrent workflows the pattern itself decides who
        # speaks, so silently ignoring the request would leave the client believing it addressed
        # an agent it did not.
        if start_agent is not None and not PATTERN_SEMANTICS[spec.pattern][2]:
            await websocket.close(
                code=4400,
                reason=f"'{orchestrator_id}' is a {spec.pattern} workflow; its participants are not directly addressable",
            )
            return
        # A solo session is a conversation with one named agent and no workflow around it, so it
        # needs to know which agent, and the orchestrator has to be able to hand one out.
        if solo and start_agent is None:
            await websocket.close(code=4400, reason="a solo session must name a participant via start_agent")
            return
        if solo and spec.build_agent is None:
            await websocket.close(code=4400, reason=f"'{orchestrator_id}' does not support solo sessions")
            return

        await websocket.accept()
        pending_tool_calls: dict[str, asyncio.Future[str]] = {}
        # Turns are processed one at a time in the order received (steer/queue semantics), but
        # frames must still be *read* concurrently with turn processing — a tool call raised
        # mid-turn blocks on a `tool_result` frame arriving on this same connection, so a
        # single sequential read-then-process loop would deadlock (the read would never
        # happen because it's waiting on the very stream that's waiting on it). A background
        # reader task decouples "receive frames" from "process one turn at a time".
        turn_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def run_local_command(command: str) -> str:
            call_id = str(uuid.uuid4())
            future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
            pending_tool_calls[call_id] = future
            await websocket.send_json(ToolCallFrame(call_id=call_id, name="run_local_command", arguments={"command": command}).model_dump())
            try:
                return await asyncio.wait_for(future, timeout=TOOL_CALL_TIMEOUT_SECONDS)
            finally:
                pending_tool_calls.pop(call_id, None)

        # A solo session skips the workflow entirely: one agent, no handoffs, and its conversation
        # is carried by an AgentSession rather than a workflow checkpoint.
        workflow = None
        agent = None
        agent_session = None
        resume_from = None
        resumed_solo_session = False
        if solo:
            agent = spec.build_agent(run_local_command, start_agent)
            agent_session = sessions.load(orchestrator_id, start_agent, session_id)
            resumed_solo_session = agent_session is not None
            if agent_session is not None:
                await websocket.send_json(
                    SessionResumedFrame(session_id=session_id or "", checkpoint_id=agent_session.session_id).model_dump()
                )
            else:
                agent_session = AgentSession(session_id=session_id or str(uuid.uuid4()))
        else:
            name = checkpoints.workflow_name(orchestrator_id, session_id)
            workflow = spec.build(run_local_command, start_agent, name)

            # Rejoin rather than restart. The Workflow object is per-connection, but its
            # conversation is durable, so a reconnect carrying the same session id picks the
            # conversation back up. Only a checkpoint taken while the workflow was idle awaiting a
            # user turn can be continued, which is what `latest_resumable` selects for.
            resume_from = await checkpoints.latest_resumable(name) if session_id else None
            if resume_from:
                await websocket.send_json(
                    SessionResumedFrame(session_id=session_id or "", checkpoint_id=resume_from.checkpoint_id).model_dump()
                )

        async def read_frames() -> None:
            try:
                while True:
                    raw = await websocket.receive_json()
                    try:
                        frame: ClientFrame = _parse_client_frame(raw)
                    except ValidationError as exc:
                        await websocket.send_json({"type": "error", "message": f"invalid frame: {exc}"})
                        continue

                    if isinstance(frame, ToolResultFrame):
                        future = pending_tool_calls.get(frame.call_id)
                        if future is not None and not future.done():
                            future.set_result(frame.output)
                        continue

                    text = _resolve_turn_text(frame)
                    if text is None:
                        await websocket.send_json({"type": "error", "message": f"unexpected frame type: {frame.type}"})
                        continue
                    await turn_queue.put(text)
            except WebSocketDisconnect:
                logger.info("session for orchestrator=%s disconnected; cancelling workflow", orchestrator_id)
                await turn_queue.put(None)

        reader_task = asyncio.create_task(read_frames())
        pending_request_id: str | None = None
        try:
            while True:
                text = await turn_queue.get()
                if text is None:
                    break
                resumed_this_turn = pending_request_id is None and resume_from is not None
                if solo:
                    # No workflow to drive, so the turn is just this one agent answering with its
                    # own conversation attached. Nothing else observes it.
                    with _tracer.start_as_current_span(f"turn {orchestrator_id}/{start_agent} (solo)") as span:
                        span.set_attribute("maf.orchestrator.id", orchestrator_id)
                        span.set_attribute("maf.orchestrator.pattern", spec.pattern)
                        span.set_attribute("maf.start_agent", start_agent or "")
                        span.set_attribute("maf.session.solo", True)
                        if session_id:
                            span.set_attribute("session.id", session_id)
                        span.set_attribute("maf.session.resumed", resumed_solo_session)
                        span.set_attribute("maf.agents.engaged", [start_agent or ""])
                        span.set_attribute("maf.agent.responding", start_agent or "")
                        await _consume_agent_updates(
                            agent.run(text, stream=True, session=agent_session), websocket, start_agent or ""
                        )
                    # Saved every turn rather than on disconnect: a dropped socket never runs a
                    # clean shutdown, which is exactly the case rejoining has to survive.
                    sessions.save(orchestrator_id, start_agent, session_id, agent_session)
                    resumed_solo_session = False
                    continue
                if pending_request_id is not None:
                    stream = workflow.run(
                        stream=True,
                        responses={pending_request_id: HandoffAgentUserRequest.create_response(text)},
                    )
                elif resume_from is not None:
                    # "Restore then send": the checkpoint's pending request keeps its original id
                    # across restore, so the turn is delivered as a continuation of the restored
                    # conversation rather than as a fresh one.
                    stream = workflow.run(
                        stream=True,
                        checkpoint_id=resume_from.checkpoint_id,
                        responses={
                            request_id: HandoffAgentUserRequest.create_response(text)
                            for request_id in resume_from.pending_request_info_events
                        },
                    )
                    resume_from = None
                else:
                    stream = workflow.run(text, stream=True)
                # A bridge-owned span per user turn. Without it the Phoenix trace list shows only
                # `workflow.run` / `workflow.build`, and identifying which agent actually answered
                # means drilling into the executor tree. This span names the engaged agents up
                # front and carries the session correlation id.
                with _tracer.start_as_current_span(f"turn {orchestrator_id}") as span:
                    span.set_attribute("maf.orchestrator.id", orchestrator_id)
                    span.set_attribute("maf.orchestrator.pattern", spec.pattern)
                    if start_agent:
                        span.set_attribute("maf.start_agent", start_agent)
                    if session_id:
                        span.set_attribute("session.id", session_id)
                    span.set_attribute("maf.session.resumed", resumed_this_turn)
                    engaged: list[str] = []
                    pending_request_id = await _consume_workflow_events(stream, websocket, engaged)
                    if engaged:
                        span.set_attribute("maf.agents.engaged", engaged)
                        span.set_attribute("maf.agent.responding", engaged[-1])
                        span.update_name(f"turn {orchestrator_id}/{'→'.join(engaged)}")
        finally:
            reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader_task

    return app


def _resolve_turn_text(frame: ClientFrame) -> str | None:
    """Translate a client frame into the text fed as the next turn's input, or None if the
    frame isn't turn-initiating (i.e. `ToolResultFrame`, handled separately by the caller).
    """

    if isinstance(frame, UserMessageFrame):
        return frame.text
    if isinstance(frame, SteerToAgentFrame):
        # Advisory nudge — see protocol.py's SteerToAgentFrame docstring and
        # design-proposal.md WS1 for why this cannot be a hard override.
        return f"The user has requested you hand off to '{frame.agent_id}' now."
    return None


def _parse_client_frame(raw: dict) -> ClientFrame:
    return TypeAdapter(ClientFrame).validate_python(raw)


async def _consume_workflow_events(
    stream: AsyncIterable[WorkflowEvent], websocket: WebSocket, engaged: list[str] | None = None
) -> str | None:
    """Consume workflow events, translate to wire frames, send over `websocket`.

    Returns the `request_id` of a pending `request_info` event if the workflow is now waiting
    for the next user turn, or `None` if the workflow reached idle/terminated on its own.
    """

    pending_request_id: str | None = None
    async for event in stream:
        if event.type == "output":
            text = _extract_text(event.data)
            if text:
                agent_id = event.executor_id or "unknown"
                # Ordered, de-duplicated: the turn's handoff chain, e.g. ["triage", "refunds"].
                if engaged is not None and (not engaged or engaged[-1] != agent_id):
                    engaged.append(agent_id)
                await websocket.send_json(AssistantDeltaFrame(agent_id=agent_id, text=text).model_dump())
        elif event.type == "handoff_sent" and isinstance(event.data, HandoffSentEvent):
            # Handoff edges, not just responders: an agent that hands off without emitting output
            # still participated, and the target is the one that will answer next. Without this
            # the chain collapses to whoever happened to speak last.
            if engaged is not None:
                for agent_id in (event.data.source, event.data.target):
                    if not engaged or engaged[-1] != agent_id:
                        engaged.append(agent_id)
            await websocket.send_json(HandoffFrame(source=event.data.source, target=event.data.target).model_dump())
        elif event.type == "request_info":
            pending_request_id = event.request_id

    await websocket.send_json(TurnCompleteFrame().model_dump())
    return pending_request_id


async def _consume_agent_updates(stream: AsyncIterable[object], websocket: WebSocket, agent_id: str) -> None:
    """Relay one solo agent's streamed reply. The single-agent counterpart of
    `_consume_workflow_events`: there is no handoff or request-info event to translate, so this
    only forwards text and closes the turn.
    """

    async for update in stream:
        text = _extract_text(update)
        if text:
            await websocket.send_json(AssistantDeltaFrame(agent_id=agent_id, text=text).model_dump())
    await websocket.send_json(TurnCompleteFrame().model_dump())


def _extract_text(data: object) -> str:
    if isinstance(data, (AgentResponse, AgentResponseUpdate)):
        return data.text or ""
    return ""

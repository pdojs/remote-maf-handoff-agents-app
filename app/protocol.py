"""Wire protocol for the OpenCode <-> MAF handoff bridge WebSocket channel.

This is the Python side of the contract; `packages/core/src/session/execution/remote-protocol.ts`
in the opencode repo is a manually-kept-in-sync TypeScript mirror (see design-proposal.md WS2
step 2 for the cross-repo drift note). Keep both in sync whenever a frame shape changes here.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

# region client -> server frames


class UserMessageFrame(BaseModel):
    """A new user turn. Sent for the first message and for follow-up steer/queue messages."""

    type: Literal["user_message"] = "user_message"
    text: str


class ToolResultFrame(BaseModel):
    """Result of a tool call the client executed locally, in response to a `ToolCallFrame`."""

    type: Literal["tool_result"] = "tool_result"
    call_id: str
    output: str


class SteerToAgentFrame(BaseModel):
    """Advisory nudge asking the currently active participant to hand off to `agent_id`.

    Best-effort only: MAF's handoff routing is entirely decided by the active agent's own
    `handoff_to_<target_id>` tool call (see `_handoff.py`). There is no force-override API, so
    this frame is implemented as a synthetic instruction injected as the active agent's next
    input, not a hard redirect.
    """

    type: Literal["steer_to_agent"] = "steer_to_agent"
    agent_id: str


ClientFrame = Annotated[
    Union[UserMessageFrame, ToolResultFrame, SteerToAgentFrame],
    Field(discriminator="type"),
]

# endregion

# region server -> client frames


class AssistantDeltaFrame(BaseModel):
    """An incremental text chunk from whichever agent currently holds the conversation."""

    type: Literal["assistant_delta"] = "assistant_delta"
    agent_id: str
    text: str


class HandoffFrame(BaseModel):
    """Emitted whenever the workflow's `HandoffSentEvent` fires."""

    type: Literal["handoff"] = "handoff"
    source: str
    target: str


class ToolCallFrame(BaseModel):
    """The active agent invoked a tool the client must execute locally (bash/edit-style)."""

    type: Literal["tool_call"] = "tool_call"
    call_id: str
    name: str
    arguments: dict


class TurnCompleteFrame(BaseModel):
    """The current turn has finished (workflow is idle or awaiting the next user input)."""

    type: Literal["turn_complete"] = "turn_complete"


class SessionResumedFrame(BaseModel):
    """Sent on connect when a prior conversation for this session was restored from a checkpoint.

    Informational: the client already has its own durable transcript, so this exists to tell the
    user they rejoined an existing conversation rather than started a new one.
    """

    type: Literal["session_resumed"] = "session_resumed"
    session_id: str
    checkpoint_id: str


class ErrorFrame(BaseModel):
    """An unrecognized frame type or a server-side failure, sent back to the client."""

    type: Literal["error"] = "error"
    message: str


ServerFrame = Annotated[
    Union[AssistantDeltaFrame, HandoffFrame, ToolCallFrame, TurnCompleteFrame, SessionResumedFrame, ErrorFrame],
    Field(discriminator="type"),
]

# endregion


class Participant(BaseModel):
    id: str
    name: str
    description: str = ""


class OrchestratorManifestEntry(BaseModel):
    """Manifest metadata for one orchestrator, including the interaction semantics a client
    needs in order to model a session correctly.

    The capability fields exist because MAF's five orchestration patterns do NOT share a
    session model, and a client that assumes handoff semantics everywhere will silently
    corrupt or discard conversation state on the other four. See `orchestrator.py`'s
    `PATTERN_SEMANTICS` for the per-pattern derivation and its source citations.
    """

    id: str
    name: str
    description: str
    participants: list[Participant]
    # One of the five MAF orchestration patterns: sequential | concurrent | handoff |
    # group_chat | magentic.
    pattern: str = "handoff"
    # How much conversation each participant sees:
    #   shared   — every agent sees the full conversation (handoff, group_chat, magentic,
    #              sequential by default)
    #   scoped   — an agent sees only what the previous one passed down (sequential with
    #              chain_only_agent_responses=True)
    #   isolated — each agent sees only the original user input, never each other's
    #              (concurrent)
    context_scope: str = "shared"
    # Whether the workflow natively continues across user turns inside one run (handoff's
    # request_info loop). When false, each user turn is a fresh single-shot run and the
    # client must not assume the agents remember anything from the previous turn.
    multi_turn: bool = True
    # Whether a client may direct a turn at a named participant (handoff's start agent).
    # When false, the pattern itself decides who speaks and any such request is refused.
    addressable: bool = True


class Manifest(BaseModel):
    orchestrators: list[OrchestratorManifestEntry]

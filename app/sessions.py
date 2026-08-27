"""Durable storage for solo (single-agent) conversations.

A solo session talks to one participant directly, with no Workflow and therefore no workflow
checkpoint to resume from. `AgentSession` is the framework's own conversation handle, so
persisting it here is what lets a solo conversation be left and rejoined.

Note what actually gets stored. With the OpenAI chat client the transcript lives service-side and
`AgentSession.to_dict()` yields only a `service_session_id` pointer to it -- around 140 bytes, with
an empty `state`. Rejoining works, but the conversation content is held by the model provider
rather than by this container, unlike a handoff conversation whose full content is checkpointed
onto our own volume (see checkpoints.py). A provider that keeps history client-side would instead
populate `state`, and this module stores that just as well.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from agent_framework import AgentSession

# Shares the checkpoint volume so solo and handoff conversations have the same durability.
SESSION_DIR = Path(os.environ.get("MAF_SESSION_DIR", ".sessions"))


def _path(orchestrator_id: str, participant_id: str, session_id: str) -> Path:
    # Session ids come from OpenCode and are opaque here, so flatten anything path-like rather
    # than trusting them to stay within the directory.
    key = f"{orchestrator_id}::{participant_id}::{session_id}".replace("/", "_")
    return SESSION_DIR / f"{key}.json"


def load(orchestrator_id: str, participant_id: str, session_id: str | None) -> AgentSession | None:
    """The stored conversation for this agent and session, or None to start a new one."""
    if not session_id:
        return None
    path = _path(orchestrator_id, participant_id, session_id)
    if not path.is_file():
        return None
    return AgentSession.from_dict(json.loads(path.read_text()))


def save(orchestrator_id: str, participant_id: str, session_id: str | None, session: AgentSession) -> None:
    if not session_id:
        return
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    path = _path(orchestrator_id, participant_id, session_id)
    # Written via a temp file then renamed so a crash mid-write cannot leave a half-written
    # session that fails to parse on the next rejoin.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(session.to_dict()))
    tmp.replace(path)

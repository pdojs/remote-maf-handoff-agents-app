"""Durable MAF workflow checkpoints, so a conversation outlives its WebSocket.

A `Workflow` is built per connection and discarded when the socket closes, which used to end the
conversation with it. Checkpointing lets a reconnect restore the same conversation instead.

Checkpoints can only be filtered by `workflow_name` — MAF deliberately stores no per-instance id
(`_checkpoint.py:35-40`). Rather than keep a side map, every session gets its own workflow name,
so the checkpoint namespace *is* the session. That is only sound because the name does not
participate in the graph signature that restore validates against, which was verified by building
two identically-shaped workflows under different names and comparing `graph_signature_hash`.
"""

import os
from pathlib import Path

from agent_framework._workflows._checkpoint import FileCheckpointStorage, WorkflowCheckpoint

# Relative by default so tests and local runs work anywhere; the container sets an absolute
# path onto a mounted volume so conversations also survive a container restart.
CHECKPOINT_DIR = Path(os.environ.get("MAF_CHECKPOINT_DIR", ".checkpoints"))

_storage: FileCheckpointStorage | None = None


def storage() -> FileCheckpointStorage:
    """Built on first use, not at import: `FileCheckpointStorage` creates its directory in the
    constructor, which would make merely importing this module fail wherever the path is not
    writable."""
    global _storage
    if _storage is None:
        # Checkpoint payloads are pickled and reads are blocked unless the type is allowlisted.
        # The framework auto-allows the `agent_framework.` prefix, but the orchestrations package
        # is `agent_framework_orchestrations.` and so falls outside it: a handoff workflow parks on
        # a `HandoffAgentUserRequest` (plus the `GenericAlias` in its type annotation) at exactly
        # the point we want to resume from, so without these two every checkpoint we write back is
        # unreadable. This list was derived by decoding real checkpoints, not guessed -- keep it
        # minimal, since it widens what unpickling may instantiate.
        _storage = FileCheckpointStorage(
            CHECKPOINT_DIR,
            allowed_checkpoint_types=[
                "agent_framework_orchestrations._handoff:HandoffAgentUserRequest",
                "types:GenericAlias",
            ],
        )
    return _storage


def workflow_name(orchestrator_id: str, session_id: str | None) -> str:
    """Checkpoint namespace for a session. Sessionless connections share a scratch namespace
    they never resume from, keeping the no-session path free of special cases elsewhere."""
    if not session_id:
        return f"{orchestrator_id}::ephemeral"
    return f"{orchestrator_id}::{session_id}"


async def latest_resumable(name: str) -> WorkflowCheckpoint | None:
    """The newest checkpoint that a user turn can actually be delivered to.

    Not simply `get_latest`: a run records a checkpoint per superstep, and only the one taken
    while the workflow sits idle awaiting the next user turn carries a pending request to answer.
    `iteration_count` is explicitly not unique across those (`_checkpoint.py:60-71`), so select on
    the pending requests themselves and break ties by timestamp.
    """

    candidates = [
        checkpoint
        for checkpoint in await _storage.list_checkpoints(workflow_name=name)
        if checkpoint.pending_request_info_events
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda checkpoint: checkpoint.timestamp)

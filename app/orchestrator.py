"""Sample MAF handoff orchestrator used for manual demos (test-requirements.md) and as the
production code path exercised by the automated Layer 1 tests (see tests/test_server.py,
which builds workflows from fake agents instead of this module's real OpenAI-backed agents).

Kept deliberately small: three cooperative participants (triage / billing / refunds) whose
system prompts explicitly instruct them to comply with an explicit user handoff request, so
that Demo 3's `steer_to_agent` nudge (test-requirements.md row 12) is reliably observable.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Annotated

from agent_framework import tool
from agent_framework.openai import OpenAIChatClient

from . import checkpoints
from agent_framework import Workflow
from agent_framework_orchestrations import HandoffBuilder

from .protocol import OrchestratorManifestEntry, Participant

RunLocalCommand = Callable[[str], Awaitable[str]]

# Explicit compliance instruction appended to every sample agent's prompt so that an advisory
# `steer_to_agent` nudge (design-proposal.md WS1) is honored whenever it's on-topic for the
# agent to comply with. This does not make steering a hard override — see test-requirements.md
# Demo 3 row 14 for the documented negative case.
# Unconditional on purpose. The client turns a participant switch in its agent picker into an
# explicit named handoff request in the user turn, so this is a direct user directive about who
# they want to talk to — not a hint the agent should weigh against its own view of the request.
# An earlier "unless clearly irrelevant" escape hatch caused agents to keep answering questions
# they judged to be in their own lane, so a picked agent never became the responder.
_HANDOFF_COMPLIANCE_INSTRUCTION = (
    "If the user explicitly asks to be transferred or handed off to a specific colleague by "
    "name, hand off to them immediately, before answering anything else. Do not answer the "
    "request yourself first, and do not decide on the user's behalf that you are the better "
    "fit. The user chooses who they talk to."
)

# Every sample agent is given the workspace tool, so state plainly that it exists and is real —
# otherwise the model refuses local-command requests on the (normally correct) grounds that it
# has no access to the user's machine.
_LOCAL_WORKSPACE_INSTRUCTION = (
    " You have a `run_local_command` tool that really does execute shell commands in the user's "
    "local workspace on their machine. Use it whenever the user asks you to inspect or change "
    "files there, and never claim you are unable to run local commands."
)

_TRIAGE_INSTRUCTIONS = (
    "You are a front-line support triage agent. Greet the user, understand their need, and "
    "hand off to 'billing' for billing/invoice questions or 'refunds' for refund requests. "
    "Handle anything else yourself. " + _HANDOFF_COMPLIANCE_INSTRUCTION + _LOCAL_WORKSPACE_INSTRUCTION
)
_BILLING_INSTRUCTIONS = (
    "You are a billing support agent. Help with invoices, charges, and payment methods. "
    "Hand off to 'refunds' if the user asks about a refund instead. " + _HANDOFF_COMPLIANCE_INSTRUCTION + _LOCAL_WORKSPACE_INSTRUCTION
)
_REFUNDS_INSTRUCTIONS = (
    "You are a refunds agent. Help process refund requests and explain refund policy. "
    "Hand off to 'billing' if the user asks a billing question instead. " + _HANDOFF_COMPLIANCE_INSTRUCTION + _LOCAL_WORKSPACE_INSTRUCTION
)


# Interaction semantics per MAF orchestration pattern, derived by reading the orchestration
# builders in agent_framework rather than by assumption — the five patterns genuinely differ,
# and the differences are not the ones intuition suggests.
#
# The notable surprise: **handoff broadcasts the full conversation to every participant**. It is
# not a narrow pass-down. Each agent keeps a synchronized replica of the whole conversation, so
# after `triage` hands off to `refunds`, `refunds` already knows everything the user told
# `triage`. Handoff is the *only* pattern with first-class multi-turn support: it emits a
# `request_info` event after each turn, and `workflow.run(responses={id: ...})` continues the
# same run indefinitely with that shared context intact.
#
# Conversely **concurrent is the narrow one**: every agent receives only the original user input
# and never sees any other agent's output.
#
# Sequential defaults to forwarding the full conversation down the chain, but narrows to just
# the preceding agent's reply when built with `chain_only_agent_responses=True` — so its scope is
# a property of how the workflow was built, which is why `OrchestratorSpec` declares it per
# orchestrator rather than deriving it from the pattern name alone.
#
# Group chat and magentic share the full conversation but are single-shot: the run terminates
# when the orchestrator decides it is done, so a second user turn is necessarily a new run with
# no memory of the first. A client must not present those as continuous conversations.
PATTERN_SEMANTICS: Mapping[str, tuple[str, bool, bool]] = {
    # pattern: (context_scope, multi_turn, addressable)
    "handoff": ("shared", True, True),
    "group_chat": ("shared", False, False),
    "magentic": ("shared", False, False),
    "sequential": ("shared", False, False),
    "concurrent": ("isolated", False, False),
}


@dataclass(frozen=True)
class OrchestratorSpec:
    """Manifest metadata plus a factory for the underlying Workflow.

    Manifest fields are tracked explicitly here rather than introspected from
    `Workflow.executors`, since that internal dict is not a stable/public participant list.
    """

    id: str
    name: str
    description: str
    participant_ids: tuple[str, ...]
    # `start_agent` selects which participant the conversation begins with; None uses the
    # workflow's own default. Lets a client address any agent in the network directly.
    # `workflow_name` namespaces this session's checkpoints; see app/checkpoints.py.
    build: Callable[[RunLocalCommand, str | None, str | None], Workflow]
    # Builds one participant as a standalone agent for a solo session: no workflow, no handoffs,
    # and no other agent sees the conversation. None means this orchestrator's participants can
    # only be talked to as part of the network.
    build_agent: Callable[[RunLocalCommand, str], object] | None = None
    # Human-readable name/description per participant id, surfaced in the manifest so a client
    # can list and address every agent in the network individually. Ids missing here fall back
    # to the id itself as the name.
    participant_details: Mapping[str, tuple[str, str]] = field(default_factory=dict)
    # Which MAF orchestration pattern this workflow is built with. Drives the interaction
    # semantics advertised in the manifest; see PATTERN_SEMANTICS above.
    pattern: str = "handoff"
    # Overrides the pattern's default context scope. Needed for sequential workflows built with
    # `chain_only_agent_responses=True`, whose scope is "scoped" rather than the pattern default.
    context_scope_override: str | None = None

    def manifest_entry(self) -> OrchestratorManifestEntry:
        return OrchestratorManifestEntry(
            id=self.id,
            name=self.name,
            description=self.description,
            pattern=self.pattern,
            context_scope=self.context_scope_override or PATTERN_SEMANTICS[self.pattern][0],
            multi_turn=PATTERN_SEMANTICS[self.pattern][1],
            addressable=PATTERN_SEMANTICS[self.pattern][2],
            participants=[
                Participant(
                    id=pid,
                    name=self.participant_details.get(pid, (pid, ""))[0],
                    description=self.participant_details.get(pid, (pid, ""))[1],
                )
                for pid in self.participant_ids
            ],
        )


def make_run_local_command_tool(run_local_command: RunLocalCommand):
    """Wrap a per-connection `run_local_command` bridge callable as a `FunctionTool` agents can
    call. Factored out so tests can attach the exact same tool implementation to fake agents
    without duplicating the wrapping logic (see tests/test_server.py's tool-bridge fixture).
    """

    async def run_local_command_tool(
        command: Annotated[str, "The shell command to run in the user's local workspace."],
    ) -> str:
        """Run a shell command in the user's local workspace and return its combined output."""
        return await run_local_command(command)

    return tool(
        run_local_command_tool,
        name="run_local_command",
        description="Run a shell command in the user's local workspace and return its combined output.",
    )


def _sample_support_agents(run_local_command: RunLocalCommand):
    """The three support agents, built the same way whether they are wired into a handoff
    workflow or addressed one-to-one in a solo session -- so a solo conversation gets an agent
    with identical instructions and the same local-workspace tool.
    """

    local_command_tool = make_run_local_command_tool(run_local_command)
    # Defaults to a low-cost chat model for manual/demo runs (overridable via OPENAI_CHAT_MODEL)
    # — OpenAIChatClient() has no built-in default and raises SettingNotFoundError without one.
    client = OpenAIChatClient(model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    # HandoffBuilder.build() requires every participant agent to opt in to
    # per-service-call history persistence, so that local history stays consistent
    # with the service across handoff tool-call short-circuits.
    return {
        name: client.as_agent(
            instructions=instructions,
            name=name,
            tools=[local_command_tool],
            require_per_service_call_history_persistence=True,
        )
        for name, instructions in (
            ("triage", _TRIAGE_INSTRUCTIONS),
            ("billing", _BILLING_INSTRUCTIONS),
            ("refunds", _REFUNDS_INSTRUCTIONS),
        )
    }


def _build_sample_support_workflow(
    run_local_command: RunLocalCommand,
    start_agent: str | None = None,
    workflow_name: str | None = None,
) -> Workflow:
    agents = _sample_support_agents(run_local_command)
    triage, billing, refunds = agents["triage"], agents["billing"], agents["refunds"]
    # The name namespaces checkpoints per session. It does not affect the graph signature that
    # restore validates against, so a workflow rebuilt for a different session still restores.
    return (
        HandoffBuilder(
            name=workflow_name or "support",
            description="Support triage handoff group: triage, billing, refunds",
            participants=[triage, billing, refunds],
        )
        .with_start_agent(agents.get(start_agent or "triage", triage))
        .with_checkpointing(checkpoints.storage())
        .build()
    )


def _build_sample_support_agent(run_local_command: RunLocalCommand, participant_id: str):
    return _sample_support_agents(run_local_command)[participant_id]


SAMPLE_SUPPORT_ORCHESTRATOR = OrchestratorSpec(
    id="support",
    name="Support Triage",
    description="Support triage handoff group: triage, billing, refunds",
    participant_ids=("triage", "billing", "refunds"),
    participant_details={
        "triage": ("Triage", "Front-line support: understands the need and hands off"),
        "billing": ("Billing", "Invoices, charges, and payment methods"),
        "refunds": ("Refunds", "Refund requests and refund policy"),
    },
    build=_build_sample_support_workflow,
    build_agent=_build_sample_support_agent,
)


def default_orchestrators() -> list[OrchestratorSpec]:
    """The orchestrator registry served by `GET /agents/manifest`.

    A single sample orchestrator today; add entries here as more demo workflows are needed.
    """

    return [SAMPLE_SUPPORT_ORCHESTRATOR]

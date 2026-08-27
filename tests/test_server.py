"""Layer 1 contract tests for the FastAPI/WebSocket bridge (design-proposal.md WS1, Testing
Strategy Layer 1). Uses deterministic fake `Agent`/`BaseChatClient` fixtures modeled directly
on `agent-framework/python/packages/orchestrations/tests/test_handoff.py`'s `MockChatClient`/
`MockHandoffAgent` pattern — no real LLM calls, no network egress.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Mapping, Sequence
from typing import Any

from agent_framework import (
    Agent,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    Message,
    ResponseStream,
)
from agent_framework._clients import BaseChatClient
from agent_framework._middleware import ChatMiddlewareLayer
from agent_framework._tools import FunctionInvocationLayer
import pytest
from dataclasses import replace

from agent_framework_orchestrations import HandoffBuilder
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app import sessions
from app.orchestrator import OrchestratorSpec, make_run_local_command_tool
from app.server import create_app


class MockChatClient(FunctionInvocationLayer[Any], ChatMiddlewareLayer[Any], BaseChatClient[Any]):
    """Deterministic fake chat client: replies with a fixed handoff (or plain text) each call,
    unless the latest user message contains an explicit "hand off to '<target>'" instruction
    (as sent by our server's `steer_to_agent` handling), in which case it hands off to that
    target instead — mirroring the `_HANDOFF_COMPLIANCE_INSTRUCTION` behavior real sample
    agents are prompted to exhibit (see orchestrator.py).
    """

    def __init__(self, *, name: str, handoff_to: str | None = None) -> None:
        ChatMiddlewareLayer.__init__(self)
        FunctionInvocationLayer.__init__(self)
        BaseChatClient.__init__(self)
        self._name = name
        self._handoff_to = handoff_to
        self._call_index = 0

    def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]:
        handoff_to = self._resolve_handoff_target(messages)

        if stream:
            return self._build_streaming_response(handoff_to=handoff_to)

        async def _get() -> ChatResponse:
            contents = _build_reply_contents(self._name, handoff_to, self._next_call_id(handoff_to))
            return ChatResponse(messages=Message(role="assistant", contents=contents), response_id="mock_response")

        return _get()

    def _build_streaming_response(self, *, handoff_to: str | None) -> ResponseStream[ChatResponseUpdate, ChatResponse]:
        async def _stream() -> AsyncIterable[ChatResponseUpdate]:
            contents = _build_reply_contents(self._name, handoff_to, self._next_call_id(handoff_to))
            yield ChatResponseUpdate(contents=contents, role="assistant", finish_reason="stop")

        def _finalize(updates: Sequence[ChatResponseUpdate]) -> ChatResponse:
            return ChatResponse.from_updates(updates)

        return ResponseStream(_stream(), finalizer=_finalize)

    def _resolve_handoff_target(self, messages: Sequence[Message]) -> str | None:
        for message in reversed(messages):
            if message.role != "user":
                continue
            text = message.text or ""
            if "hand off to '" in text:
                return text.split("hand off to '", 1)[1].split("'", 1)[0]
            break
        return self._handoff_to

    def _next_call_id(self, handoff_to: str | None) -> str | None:
        if not handoff_to:
            return None
        call_id = f"{self._name}-handoff-{self._call_index}"
        self._call_index += 1
        return call_id


def _build_reply_contents(agent_name: str, handoff_to: str | None, call_id: str | None) -> list[Content]:
    contents: list[Content] = []
    if handoff_to and call_id:
        contents.append(
            Content.from_function_call(call_id=call_id, name=f"handoff_to_{handoff_to}", arguments={"handoff_to": handoff_to})
        )
    contents.append(Content.from_text(text=f"{agent_name} reply"))
    return contents


def _mock_agent(name: str, handoff_to: str | None = None) -> Agent:
    return Agent(
        client=MockChatClient(name=name, handoff_to=handoff_to),
        name=name,
        id=name,
        require_per_service_call_history_persistence=True,
    )


def _noop_run_local_command(_command: str) -> Any:
    raise AssertionError("run_local_command should not be invoked by these fixtures")


def _two_agent_handoff_spec() -> OrchestratorSpec:
    # `alpha` always hands off to `beta`; `beta` never hands off, so after its reply the
    # workflow requests the next user input (human-in-loop default) — this proves the
    # assistant_delta* -> handoff -> assistant_delta* -> turn_complete frame sequence.
    def build(run_local_command: Any, start_agent: str | None = None, workflow_name: str | None = None) -> Any:
        del run_local_command, start_agent, workflow_name
        alpha = _mock_agent("alpha", handoff_to="beta")
        beta = _mock_agent("beta")
        return HandoffBuilder(name="demo", participants=[alpha, beta]).with_start_agent(alpha).build()

    return OrchestratorSpec(
        id="demo", name="Demo", description="two-agent handoff fixture", participant_ids=("alpha", "beta"), build=build
    )


def _three_agent_spec() -> OrchestratorSpec:
    # None of the three hand off on their own; a `steer_to_agent` frame is the only thing that
    # causes a handoff, proving the advisory-nudge mechanism in isolation from any "agent
    # decided to hand off anyway" ambiguity.
    def build(run_local_command: Any, start_agent: str | None = None, workflow_name: str | None = None) -> Any:
        del run_local_command, workflow_name
        alpha = _mock_agent("alpha")
        beta = _mock_agent("beta")
        gamma = _mock_agent("gamma")
        agents = {"alpha": alpha, "beta": beta, "gamma": gamma}
        return (
            HandoffBuilder(name="demo3", participants=[alpha, beta, gamma])
            .with_start_agent(agents.get(start_agent or "alpha", alpha))
            .build()
        )

    return OrchestratorSpec(
        id="demo3",
        name="Demo3",
        description="three-agent steer fixture",
        participant_ids=("alpha", "beta", "gamma"),
        build=build,
    )


def test_manifest_shape() -> None:
    app = create_app(orchestrators=[_two_agent_handoff_spec()])
    client = TestClient(app)

    response = client.get("/agents/manifest")

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "orchestrators": [
            {
                "id": "demo",
                "name": "Demo",
                "description": "two-agent handoff fixture",
                "pattern": "handoff",
                "context_scope": "shared",
                "multi_turn": True,
                "addressable": True,
                "participants": [
                    {"id": "alpha", "name": "alpha", "description": ""},
                    {"id": "beta", "name": "beta", "description": ""},
                ],
            }
        ]
    }


def test_handoff_sequence_streams_expected_frames() -> None:
    app = create_app(orchestrators=[_two_agent_handoff_spec()])
    client = TestClient(app)

    with client.websocket_connect("/agents/demo/session") as ws:
        ws.send_json({"type": "user_message", "text": "I need help"})

        frames = []
        while True:
            frame = ws.receive_json()
            frames.append(frame)
            if frame["type"] == "turn_complete":
                break

        types = [frame["type"] for frame in frames]
        assert types == ["assistant_delta", "handoff", "assistant_delta", "turn_complete"]
        assert frames[0]["agent_id"] == "alpha"
        assert frames[1] == {"type": "handoff", "source": "alpha", "target": "beta"}
        assert frames[2]["agent_id"] == "beta"


def test_unrecognized_frame_type_is_rejected() -> None:
    app = create_app(orchestrators=[_two_agent_handoff_spec()])
    client = TestClient(app)

    with client.websocket_connect("/agents/demo/session") as ws:
        ws.send_json({"type": "not_a_real_frame"})
        frame = ws.receive_json()
        assert frame["type"] == "error"


def test_steer_to_agent_targets_participant() -> None:
    app = create_app(orchestrators=[_three_agent_spec()])
    client = TestClient(app)

    with client.websocket_connect("/agents/demo3/session") as ws:
        ws.send_json({"type": "user_message", "text": "hello"})
        frames = _drain_until_turn_complete(ws)
        assert [f["type"] for f in frames] == ["assistant_delta", "turn_complete"]
        assert frames[0]["agent_id"] == "alpha"

        ws.send_json({"type": "steer_to_agent", "agent_id": "gamma"})
        frames = _drain_until_turn_complete(ws)
        types = [f["type"] for f in frames]
        assert "handoff" in types
        handoff_frame = next(f for f in frames if f["type"] == "handoff")
        assert handoff_frame == {"type": "handoff", "source": "alpha", "target": "gamma"}


def test_start_agent_addresses_a_participant_directly() -> None:
    """A client can open a session on any agent in the network, not just the default start
    agent, so `/agents` can list and address every participant individually."""
    app = create_app(orchestrators=[_three_agent_spec()])
    client = TestClient(app)

    with client.websocket_connect("/agents/demo3/session?start_agent=gamma") as ws:
        ws.send_json({"type": "user_message", "text": "hello"})
        frames = _drain_until_turn_complete(ws)
        assert [f["type"] for f in frames] == ["assistant_delta", "turn_complete"]
        # Without start_agent this would be "alpha" (see test_steer_to_agent_targets_participant).
        assert frames[0]["agent_id"] == "gamma"


def test_unknown_start_agent_is_rejected() -> None:
    app = create_app(orchestrators=[_three_agent_spec()])
    client = TestClient(app)

    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/agents/demo3/session?start_agent=nobody") as ws:
            ws.receive_json()
    assert excinfo.value.code == 4404


def test_non_addressable_pattern_refuses_start_agent() -> None:
    """Group chat, magentic, sequential and concurrent workflows decide internally who speaks,
    so a client cannot address a participant. Refusing loudly beats silently ignoring the
    request and letting the client believe it reached the agent it named.
    """
    spec = replace(_three_agent_spec(), id="chat", pattern="group_chat")
    client = TestClient(create_app(orchestrators=[spec]))

    # The participant exists; it is the pattern that makes it unaddressable.
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/agents/chat/session?start_agent=gamma") as ws:
            ws.receive_json()
    assert excinfo.value.code == 4400

    # ...and the same orchestrator still accepts a plain, unaddressed session.
    with client.websocket_connect("/agents/chat/session") as ws:
        ws.send_json({"type": "user_message", "text": "hello"})
        assert [f["type"] for f in _drain_until_turn_complete(ws)] == ["assistant_delta", "turn_complete"]


def test_manifest_advertises_pattern_semantics() -> None:
    """A client needs the pattern's interaction semantics to model a session correctly: a
    single-shot pattern must not be presented as a continuous conversation.
    """
    specs = [
        replace(_two_agent_handoff_spec(), id="h", pattern="handoff"),
        replace(_two_agent_handoff_spec(), id="c", pattern="concurrent"),
        replace(_two_agent_handoff_spec(), id="s", pattern="sequential", context_scope_override="scoped"),
    ]
    body = TestClient(create_app(orchestrators=specs)).get("/agents/manifest").json()
    got = {o["id"]: (o["pattern"], o["context_scope"], o["multi_turn"], o["addressable"]) for o in body["orchestrators"]}

    assert got["h"] == ("handoff", "shared", True, True)
    # Concurrent fans the same user input out to every agent and none of them see each other.
    assert got["c"] == ("concurrent", "isolated", False, False)
    # Sequential defaults to forwarding the whole conversation; this one was built to narrow it.
    assert got["s"] == ("sequential", "scoped", False, False)


def test_tool_call_bridge_round_trip() -> None:
    def build(run_local_command: Any, start_agent: str | None = None, workflow_name: str | None = None) -> Any:
        del start_agent, workflow_name
        agent = Agent(
            client=_ToolCallingMockChatClient(),
            name="assistant",
            id="assistant",
            tools=[make_run_local_command_tool(run_local_command)],
            require_per_service_call_history_persistence=True,
        )
        return HandoffBuilder(name="tool-demo", participants=[agent]).with_start_agent(agent).build()

    spec = OrchestratorSpec(id="tool-demo", name="Tool Demo", description="tool bridge fixture", participant_ids=("assistant",), build=build)
    app = create_app(orchestrators=[spec])
    client = TestClient(app)

    with client.websocket_connect("/agents/tool-demo/session") as ws:
        ws.send_json({"type": "user_message", "text": "run echo hi"})

        tool_call = ws.receive_json()
        assert tool_call["type"] == "tool_call"
        assert tool_call["name"] == "run_local_command"
        assert tool_call["arguments"] == {"command": "echo hi"}

        ws.send_json({"type": "tool_result", "call_id": tool_call["call_id"], "output": "ran: echo hi"})

        frames = _drain_until_turn_complete(ws)
        assert any(f["type"] == "assistant_delta" and "ran: echo hi" in f["text"] for f in frames)


class _ToolCallingMockChatClient(FunctionInvocationLayer[Any], ChatMiddlewareLayer[Any], BaseChatClient[Any]):
    """First call: emit a `run_local_command` function call. Second call (after the function
    result is fed back by `FunctionInvocationLayer`): echo the tool result in plain text so the
    round trip is observable end-to-end.
    """

    def __init__(self) -> None:
        ChatMiddlewareLayer.__init__(self)
        FunctionInvocationLayer.__init__(self)
        BaseChatClient.__init__(self)
        self._call_index = 0

    def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]:
        tool_result_text = self._find_tool_result(messages)

        if tool_result_text is None:
            contents: list[Content] = [
                Content.from_function_call(call_id="call-1", name="run_local_command", arguments={"command": "echo hi"})
            ]
        else:
            contents = [Content.from_text(text=f"done: {tool_result_text}")]

        async def _get() -> ChatResponse:
            return ChatResponse(messages=Message(role="assistant", contents=contents), response_id="mock_response")

        if stream:
            async def _stream() -> AsyncIterable[ChatResponseUpdate]:
                yield ChatResponseUpdate(contents=contents, role="assistant", finish_reason="stop")

            return ResponseStream(_stream(), finalizer=lambda updates: ChatResponse.from_updates(updates))

        return _get()

    def _find_tool_result(self, messages: Sequence[Message]) -> str | None:
        for message in messages:
            for content in message.contents:
                if getattr(content, "type", None) == "function_result" and getattr(content, "call_id", None) == "call-1":
                    return str(content.result)
        return None


def _solo_capable_spec() -> OrchestratorSpec:
    """Same participants as the handoff fixture, but also able to hand one out standalone."""

    def build(run_local_command: Any, start_agent: str | None = None, workflow_name: str | None = None) -> Any:
        del run_local_command, start_agent, workflow_name
        alpha = _mock_agent("alpha", handoff_to="beta")
        beta = _mock_agent("beta")
        return HandoffBuilder(name="solo-demo", participants=[alpha, beta]).with_start_agent(alpha).build()

    def build_agent(run_local_command: Any, participant_id: str) -> Any:
        del run_local_command
        return _mock_agent(participant_id)

    return OrchestratorSpec(
        id="solo",
        name="Solo",
        description="solo-capable fixture",
        participant_ids=("alpha", "beta"),
        build=build,
        build_agent=build_agent,
    )


def test_solo_session_answers_from_the_named_agent_without_handoff(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(sessions, "SESSION_DIR", tmp_path)
    client = TestClient(create_app([_solo_capable_spec()]))

    with client.websocket_connect("/agents/solo/session?solo=true&start_agent=beta&session_id=s1") as ws:
        ws.send_json({"type": "user_message", "text": "hello"})
        frames = _drain_until_turn_complete(ws)

    # `alpha` is the workflow's start agent and hands off; addressing `beta` solo must bypass both.
    assert not any(f["type"] == "handoff" for f in frames)
    assert [f["agent_id"] for f in frames if f["type"] == "assistant_delta"] == ["beta"]


def test_solo_session_is_persisted_and_resumed(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(sessions, "SESSION_DIR", tmp_path)
    client = TestClient(create_app([_solo_capable_spec()]))

    with client.websocket_connect("/agents/solo/session?solo=true&start_agent=beta&session_id=s2") as ws:
        ws.send_json({"type": "user_message", "text": "hello"})
        first = _drain_until_turn_complete(ws)
    assert not any(f["type"] == "session_resumed" for f in first)

    # Reconnecting with the same session id must pick the stored conversation back up; a
    # different id must not see it.
    with client.websocket_connect("/agents/solo/session?solo=true&start_agent=beta&session_id=s2") as ws:
        ws.send_json({"type": "user_message", "text": "again"})
        second = _drain_until_turn_complete(ws)
    assert any(f["type"] == "session_resumed" for f in second)

    with client.websocket_connect("/agents/solo/session?solo=true&start_agent=beta&session_id=other") as ws:
        ws.send_json({"type": "user_message", "text": "again"})
        third = _drain_until_turn_complete(ws)
    assert not any(f["type"] == "session_resumed" for f in third)


def test_solo_session_requires_a_named_participant() -> None:
    client = TestClient(create_app([_solo_capable_spec()]))
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/agents/solo/session?solo=true") as ws:
            ws.receive_json()


def test_solo_session_rejected_when_orchestrator_cannot_build_one() -> None:
    client = TestClient(create_app([_two_agent_handoff_spec()]))
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/agents/demo/session?solo=true&start_agent=alpha") as ws:
            ws.receive_json()


def _drain_until_turn_complete(ws: Any) -> list[dict]:
    frames = []
    while True:
        frame = ws.receive_json()
        frames.append(frame)
        if frame["type"] == "turn_complete":
            break
    return frames

"""Pytest suite for the Synth ReAct agent loop (src/synth/agent.py).

Only the LLM is faked — it is the sole external dependency. The ToolRegistry,
SessionStore, tool functions and prompt builder are all the real production
code, exercised against a tmp_path working directory and a tmp_path SQLite DB.
"""

from __future__ import annotations

import json
import os

import pytest

from synth.agent import Agent, AgentConfig, AgentError, RunResult
from synth.llm import LLMResponse, ToolCall
from synth.prompts import build_system_prompt
from synth.session import Message, SessionStore
from synth.tools import ToolRegistry, default_registry


class FakeLLM:
    """Scripted LLM: returns queued responses in order.

    A ``default`` factory can be supplied to generate responses forever
    (used to drive the max-iterations path). The queue is deliberately
    unforgiving: running out of scripted responses fails the test loudly
    instead of silently inventing an answer.
    """

    def __init__(self, *responses: LLMResponse, default=None) -> None:
        self.responses = list(responses)
        self.default = default
        self.calls: list[dict] = []
        self._n = 0

    def chat(self, messages, tools=None, stream=True) -> LLMResponse:
        # Copy the message list: the agent mutates it in place across
        # iterations, and each recorded call must snapshot what the model
        # actually saw at that moment.
        self.calls.append({"messages": list(messages), "tools": tools, "stream": stream})
        if self.responses:
            return self.responses.pop(0)
        if self.default is not None:
            self._n += 1
            return self.default(self._n)
        raise AssertionError("FakeLLM scripted responses exhausted")

    @property
    def exhausted(self) -> bool:
        return not self.responses


def _tool_call(name: str, args: dict, call_id: str | None = None) -> ToolCall:
    # Unique IDs per call: the LLM API requires distinct tool_call ids, and
    # the agent appends one tool-result message per call.
    _tool_call._counter += 1
    return ToolCall(
        id=call_id if call_id is not None else f"call_{_tool_call._counter}",
        name=name,
        args=args,
    )


_tool_call._counter = 0


# --- fixtures -------------------------------------------------------------


@pytest.fixture
def registry() -> ToolRegistry:
    return default_registry()


@pytest.fixture
def store(tmp_path):
    with SessionStore(db_path=tmp_path / "agent.db") as session_store:
        yield session_store


@pytest.fixture
def session_id(store) -> str:
    return store.create_session(title="agent-tests")


@pytest.fixture(autouse=True)
def cwd(tmp_path, monkeypatch) -> str:
    """Run every test inside tmp_path so the real tools can touch the FS."""
    monkeypatch.chdir(tmp_path)
    return os.getcwd()


@pytest.fixture
def make_agent(registry, store, session_id):
    def _make(llm, config: AgentConfig | None = None, output_handler=None) -> Agent:
        return Agent(
            llm,
            registry,
            store,
            session_id,
            config=config,
            output_handler=output_handler,
        )

    return _make


# --- run(): the happy paths ------------------------------------------------


def test_run_immediate_text_answer_ends_after_one_iteration(make_agent):
    llm = FakeLLM(LLMResponse(text="42"))
    agent = make_agent(llm)

    result = agent.run("What is the meaning of life?")

    assert isinstance(result, RunResult)
    assert result.text == "42"
    assert result.iterations == 1
    assert result.tool_calls == 0
    assert result.error is None
    assert len(llm.calls) == 1
    assert llm.exhausted


def test_run_single_tool_call_then_text_answer(make_agent, tmp_path):
    llm = FakeLLM(
        LLMResponse(
            tool_calls=[
                _tool_call("write_file", {"path": "greeting.txt", "content": "hello"})
            ]
        ),
        LLMResponse(text="Wrote the file."),
    )
    agent = make_agent(llm)

    result = agent.run("Write 'hello' to greeting.txt")

    assert result.text == "Wrote the file."
    assert result.iterations == 2
    assert result.tool_calls == 1
    assert result.error is None
    # The real write_file tool ran in the real cwd.
    assert (tmp_path / "greeting.txt").read_text() == "hello"
    assert llm.exhausted


def test_run_two_tool_calls_then_answer(make_agent, tmp_path):
    llm = FakeLLM(
        LLMResponse(
            tool_calls=[
                _tool_call("write_file", {"path": "greeting.txt", "content": "hello"})
            ]
        ),
        LLMResponse(tool_calls=[_tool_call("read_file", {"path": "greeting.txt"})]),
        LLMResponse(text="The file says hello."),
    )
    agent = make_agent(llm)

    result = agent.run("Write 'hello', read it back, then report.")

    assert result.text == "The file says hello."
    assert result.iterations == 3
    assert result.tool_calls == 2
    assert result.error is None
    # Observation of the first tool is visible to the second LLM call.
    second_call_messages = llm.calls[1]["messages"]
    tool_results = [m for m in second_call_messages if m["role"] == "tool"]
    assert len(tool_results) == 1
    assert "wrote 5 bytes" in tool_results[0]["content"]
    # The full conversation the model saw on the final call.
    roles = [m["role"] for m in llm.calls[-1]["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool"]
    assert llm.exhausted


# --- run(): the error paths ------------------------------------------------


def test_run_max_iterations_returns_error_result(make_agent):
    looping = FakeLLM(
        default=lambda n: LLMResponse(
            tool_calls=[_tool_call("bash", {"command": "echo loop"}, call_id=f"call_{n}")]
        )
    )
    agent = make_agent(looping, config=AgentConfig(max_iterations=3, stream=False))

    result = agent.run("Loop forever.")

    assert result.text == ""
    assert result.error == "max iterations reached"
    assert result.iterations == 3
    assert result.tool_calls == 3
    assert len(looping.calls) == 3
    # stream=False propagates to the LLM client.
    assert looping.calls[0]["stream"] is False


def test_run_empty_prompt_raises_agent_error(make_agent, store, session_id):
    # Whitespace-only prompts are rejected the same way as empty ones.
    for empty in ["", "   ", "\n\t  "]:
        llm = FakeLLM(LLMResponse(text="must not be reached"))

        with pytest.raises(AgentError):
            make_agent(llm).run(empty)

        # The loop never calls the LLM and persists nothing.
        assert llm.calls == []
        assert store.load_session(session_id) == []


def test_run_unknown_tool_error_result_keeps_loop_alive(make_agent):
    llm = FakeLLM(
        LLMResponse(tool_calls=[_tool_call("not_a_real_tool", {})]),
        LLMResponse(text="I recovered."),
    )
    agent = make_agent(llm)

    result = agent.run("Call a tool that does not exist.")

    # A broken tool call is observed, not raised: the loop continues.
    assert result.text == "I recovered."
    assert result.iterations == 2
    assert result.tool_calls == 1
    assert result.error is None
    tool_messages = [m for m in agent.messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert "unknown tool" in tool_messages[0]["content"]


# --- run(): session persistence -------------------------------------------


def test_run_persists_messages_to_session(make_agent, store, session_id):
    llm = FakeLLM(
        LLMResponse(
            tool_calls=[
                _tool_call("write_file", {"path": "out.txt", "content": "data"})
            ]
        ),
        LLMResponse(text="Done."),
    )
    agent = make_agent(llm)

    agent.run("Write data to out.txt.")

    saved = store.load_session(session_id)
    roles = [m.role for m in saved]
    assert roles == ["user", "assistant", "tool", "assistant"]

    assert saved[0].content == "Write data to out.txt."
    # Assistant turn that requested the tool keeps the JSON tool_calls payload.
    assert saved[1].content is None
    assert saved[1].tool_calls is not None
    decoded = json.loads(saved[1].tool_calls)
    assert decoded[0]["function"]["name"] == "write_file"
    # Real tool result was stored verbatim.
    assert "OK: wrote 4 bytes" in saved[2].content
    assert saved[3].content == "Done."
    assert saved[3].tool_calls is None


# --- run(): history resume -------------------------------------------------


def test_run_prepends_history_before_user_prompt(make_agent, cwd):
    prior_tool_calls = json.dumps(
        [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps({"path": "greeting.txt", "content": "hello"}),
                },
            }
        ]
    )
    history = [
        Message(role="user", content="Write 'hello' to greeting.txt"),
        Message(role="assistant", content="", tool_calls=prior_tool_calls),
        Message(role="tool", content="OK: wrote 5 bytes", tool_call_id="call_1"),
    ]
    llm = FakeLLM(LLMResponse(text="resumed"))
    agent = make_agent(llm)

    result = agent.run("Now read it back.", history)

    assert result.text == "resumed"
    assert result.iterations == 1

    # History sits between the system prompt and the new user message.
    sent = llm.calls[0]["messages"]
    roles = [m["role"] for m in sent]
    assert roles == ["system", "user", "assistant", "tool", "user"]
    assert sent[1]["content"] == "Write 'hello' to greeting.txt"
    assert sent[-1]["content"] == "Now read it back."

    # History rows are rebuilt into API shape: tool_calls JSON is decoded...
    assert sent[2]["tool_calls"] == json.loads(prior_tool_calls)
    # ...and the tool row keeps its tool_call_id linkage.
    assert sent[3]["tool_call_id"] == "call_1"
    assert sent[3]["content"] == "OK: wrote 5 bytes"

    # The system prompt carries the real cwd.
    assert cwd in sent[0]["content"]


# --- output handler --------------------------------------------------------


def test_output_handler_receives_call_and_result_events(make_agent):
    events = []
    llm = FakeLLM(
        LLMResponse(tool_calls=[_tool_call("bash", {"command": "echo hi"})]),
        LLMResponse(text="Done."),
    )
    agent = make_agent(llm, output_handler=lambda *args: events.append(args))

    agent.run("Run echo hi.")

    assert len(events) == 2
    assert events[0][0] == "call"
    assert events[0][1] == "bash"
    assert events[0][2] == {"command": "echo hi"}
    assert events[1][0] == "result"
    assert events[1][1] == "bash"
    assert "hi" in events[1][2]


# --- LLM wiring -----------------------------------------------------------


def test_run_passes_tool_schemas_and_stream_flag_to_llm(make_agent, registry):
    llm = FakeLLM(LLMResponse(text="ok"))

    make_agent(llm).run("Anything.")

    call = llm.calls[0]
    assert call["tools"] == registry.schemas()
    assert [t["function"]["name"] for t in call["tools"]] == [
        "bash",
        "read_file",
        "write_file",
    ]
    assert call["stream"] is True


# --- build_system_prompt ---------------------------------------------------


def test_build_system_prompt_includes_cwd(monkeypatch):
    # Mock soul.md tidak ada agar pakai default prompt
    monkeypatch.setattr("synth.prompts._load_soul_md", lambda: None)
    prompt = build_system_prompt("/tmp/synth-workspace")

    assert "Current working directory: /tmp/synth-workspace" in prompt
    assert "You are Synth, a CLI AI agent." in prompt


def test_build_system_prompt_empty_cwd_still_labels_directory():
    prompt = build_system_prompt("")

    assert "Current working directory:" in prompt


def test_build_system_prompt_in_agent_message_uses_real_cwd(make_agent, cwd):
    agent = make_agent(FakeLLM(LLMResponse(text="ok")))

    agent.run("hi")

    system = agent.messages[0]
    assert system["role"] == "system"
    assert cwd in system["content"]

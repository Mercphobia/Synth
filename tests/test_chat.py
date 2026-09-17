"""Tests for the interactive REPL (synth chat)."""

from __future__ import annotations

import io

import pytest

from synth.chat import Repl
from synth.session import SessionStore


class FakeResult:
    def __init__(self, text: str, error: str | None = None) -> None:
        self.text = text
        self.error = error


class FakeAgent:
    """Minimal agent stand-in recording prompts."""

    def __init__(self, session_id: str, replies: list[str] | None = None) -> None:
        self.session_id = session_id
        self.replies = replies or ["ok"]
        self.prompts: list[str] = []

    def run(self, prompt, history=None):
        self.prompts.append(prompt)
        idx = min(len(self.prompts) - 1, len(self.replies) - 1)
        return FakeResult(self.replies[idx])


@pytest.fixture
def store(tmp_path):
    s = SessionStore(tmp_path / "sessions.db")
    yield s
    s.close()


def make_repl(store, stdin_text: str, replies=None) -> Repl:
    agents: list[FakeAgent] = []

    def factory(session_id):
        agent = FakeAgent(session_id, replies)
        agents.append(agent)
        return agent

    repl = Repl(agent_factory=factory, store=store, stdin=io.StringIO(stdin_text))
    repl.agents = agents  # test hook
    return repl


def test_session_created_on_start(store):
    repl = make_repl(store, "/exit\n")
    assert repl.session_id
    assert store.load_session(repl.session_id) == []


def test_exit_command_stops(store):
    repl = make_repl(store, "/exit\nhello\n")
    assert repl.run() == 0
    assert repl.agents[0].prompts == []


def test_quit_alias(store):
    repl = make_repl(store, "quit\n")
    assert repl.run() == 0


def test_eof_stops_loop(store):
    repl = make_repl(store, "")  # immediate EOF
    assert repl.run() == 0


def test_prompt_runs_agent(store):
    repl = make_repl(store, "do it\n/exit\n", replies=["done"])
    repl.run()
    assert repl.agents[0].prompts == ["do it"]


def test_blank_lines_ignored(store):
    repl = make_repl(store, "\n\n/exit\n")
    repl.run()
    assert repl.agents[0].prompts == []


def test_help_does_not_hit_agent(store, capsys):
    repl = make_repl(store, "/help\n/exit\n")
    repl.run()
    assert repl.agents[0].prompts == []


def test_new_resets_session(store):
    repl = make_repl(store, "a\n/new\nb\n/exit\n")
    old_id = repl.session_id
    repl.run()
    assert repl.session_id != old_id
    # 'b' went to the new agent, not the first.
    assert repl.agents[0].prompts == ["a"]
    assert repl.agents[1].prompts == ["b"]


def test_history_carries_across_turns(store):
    repl = make_repl(store, "one\ntwo\n/exit\n")
    repl.run()
    assert repl.agents[0].prompts == ["one", "two"]

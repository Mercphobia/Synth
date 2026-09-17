"""End-to-end integration test: CLI runs a real task with a mocked LLM.

Covers the acceptance criterion: 'synth "create hello.txt" creates the file'.

The mock is injected into the in-process CLI entry point (monkeypatching the
LLM client class), then cli.main() runs with argv set — the real argparse,
config bootstrap, session store, tools and agent loop all execute.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synth.cli import EXIT_OK
from synth.llm import LLMClient, LLMResponse, ToolCall


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ~/.synth into tmp_path so the test never touches real config."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def scripted_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace LLMClient.chat with a two-step scripted response."""
    responses = [
        LLMResponse(
            text=None,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="write_file",
                    args={"path": "hello.txt", "content": "hi"},
                )
            ],
        ),
        LLMResponse(text="Created hello.txt with content 'hi'."),
    ]

    def fake_chat(self, messages, tools=None, stream=True):  # type: ignore[no-untyped-def]
        return responses.pop(0) if responses else LLMResponse(text="done")

    monkeypatch.setattr(LLMClient, "chat", fake_chat)


def test_cli_create_file_end_to_end(
    isolated_home: Path, scripted_llm: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """synth 'create hello.txt' — config bootstraps and the file is written."""
    import synth.cli as cli_mod

    monkeypatch.setattr("sys.argv", ["synth", "create hello.txt with content 'hi'"])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    exit_code = cli_mod.main()

    assert exit_code == EXIT_OK
    created = isolated_home / "hello.txt"
    assert created.exists(), "write_file tool did not create the file"
    assert created.read_text() == "hi"

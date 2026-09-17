"""Interactive REPL for Synth (spec 6.12 #118: synth chat).

A plain input loop around the agent: type a prompt, get a turn, repeat.
Slash commands keep the session alive across turns (shared history).
On exit, a session summary is printed (spec 6.6 #78).
"""

from __future__ import annotations

import sys
from typing import Callable, TextIO

from rich.console import Console
from rich.markdown import Markdown

from synth.agent import Agent, AgentConfig, AgentError
from synth.llm import LLMError
from synth.session import Message, SessionStore
from synth.session_plus import build_exit_summary

console = Console()

# Slash commands recognized by the REPL (spec 11: prompt syntax /help).
REPL_COMMANDS = {
    "/help": "Show this help.",
    "/exit": "End the session (aliases: /quit, :q).",
    "/new": "Start a fresh session id.",
    "/session": "Print the current session id.",
}

EXIT_ALIASES = {"/exit", "/quit", ":q", "exit", "quit"}


def _print_help() -> None:
    console.print("[bold]REPL commands[/bold]")
    for cmd, desc in REPL_COMMANDS.items():
        console.print(f"  {cmd:<9} {desc}")
    console.print("  Any other text is sent to the agent.")


class Repl:
    """Chat loop with a shared agent and session."""

    def __init__(
        self,
        agent_factory: Callable[[str], Agent],
        store: SessionStore,
        stdin: TextIO | None = None,
    ) -> None:
        """Args:
            agent_factory: callable(session_id) -> configured Agent.
            store: session store used to mint/resume session ids.
            stdin: input stream for tests; defaults to sys.stdin.
        """
        self.agent_factory = agent_factory
        self.store = store
        self.stdin = stdin if stdin is not None else sys.stdin
        self.session_id = store.create_session()
        self.agent = agent_factory(self.session_id)
        self.history: list[Message] = []

    def _read_line(self, prompt: str) -> str | None:
        """Read one line from stdin. Returns None on EOF (Ctrl-D)."""
        try:
            console.print(prompt, end="")
            line = self.stdin.readline()
        except (EOFError, KeyboardInterrupt, OSError):
            return None
        if line == "":  # EOF
            return None
        return line.rstrip("\n")

    def run(self) -> int:
        """Run the loop until exit/EOF. Returns the process exit code."""
        console.print("[bold]Synth chat[/bold] — type a task; /help for commands.")
        turns = 0
        while True:
            line = self._read_line("[cyan]>[/cyan] ")
            if line is None:
                console.print("\nBye.")
                break
            line = line.strip()
            if not line:
                continue

            if line in EXIT_ALIASES:
                console.print("Bye.")
                break
            if line == "/help":
                _print_help()
                continue
            if line == "/session":
                console.print(f"session: {self.session_id}")
                continue
            if line == "/new":
                self.session_id = self.store.create_session()
                self.agent = self.agent_factory(self.session_id)
                self.history = []
                console.print(f"new session: {self.session_id}")
                continue

            try:
                result = self.agent.run(line, self.history or None)
            except (LLMError, AgentError) as exc:
                console.print(f"[red]Error:[/red] {exc}")
                continue

            turns += 1
            if result.error:
                console.print(f"[yellow]{result.error}[/yellow]")
            if result.text:
                console.print(Markdown(result.text))
            # Reload persisted history so the next turn carries full context.
            self.history = self.store.load_session(self.session_id)

        # Summary on exit (spec 6.6 #78).
        if turns:
            summary = build_exit_summary(self.history)
            console.print(f"[dim]{summary}[/dim]")
        return 0

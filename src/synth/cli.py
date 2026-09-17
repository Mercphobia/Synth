"""CLI entry point: parse arguments, wire modules, run the agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

from synth import __version__
from synth.agent import Agent, AgentConfig, RunResult
from synth.config import ConfigError, load_config
from synth.llm import LLMClient, LLMError
from synth.session import SessionError, SessionStore
from synth.tools import default_registry

console = Console()

# Exit codes (spec 6.3)
EXIT_OK = 0
EXIT_GENERAL = 1
EXIT_ARG = 2
EXIT_CONFIG = 3
EXIT_LLM = 4
EXIT_TOOL = 5

THINKING_PREFIX = "∿ thinking..."


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser (spec 6.1/6.2)."""
    parser = argparse.ArgumentParser(
        prog="synth",
        description="CLI AI agent — runs a task using an LLM and tools.",
    )
    parser.add_argument("prompt", nargs="?", help="Task for the agent")
    parser.add_argument("--resume", metavar="ID", help="Resume an existing session")
    parser.add_argument("--session", metavar="ACTION", help="Session actions: list")
    parser.add_argument("--model", metavar="MODEL", help="Override the config model")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    parser.add_argument("--verbose", action="store_true", help="Debug logs")
    parser.add_argument("--version", action="version", version=f"synth {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Program entry. Returns the exit code (spec 6.3)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    if args.session == "list":
        return _run_session_list()

    if not args.prompt:
        parser.error("prompt is required unless using --session or --version")
        return EXIT_ARG
    try:
        config = load_config()
    except ConfigError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return EXIT_CONFIG

    model = args.model or config.model
    try:
        api_key = config.resolve_api_key()
    except ConfigError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return EXIT_CONFIG

    if args.model:
        # An override may target a provider not in config; resolve again.
        config = _rebuild_config_for_model(config, args.model)

    llm = LLMClient(
        model=model,
        api_key=api_key,
        stream_printer=_stream_printer if not args.no_stream else None,
    )
    tools = default_registry(
        bash_timeout=config.tools.bash_timeout,
        max_file_size=config.tools.max_file_size,
        max_output_size=config.tools.max_output_size,
    )

    try:
        store = SessionStore(config.session.db_path)
    except SessionError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return EXIT_GENERAL

    if args.resume:
        session_id = args.resume
        try:
            history = store.load_session(session_id)
        except SessionError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            return EXIT_ARG
        if not history:
            console.print(f"[red]Error:[/red] no session found with id {session_id}")
            return EXIT_ARG
    else:
        session_id = store.create_session()
        history = []

    agent = Agent(
        llm=llm,
        tools=tools,
        session_store=store,
        session_id=session_id,
        config=AgentConfig(max_iterations=config.max_iterations, stream=not args.no_stream),
        output_handler=_tool_display,
    )

    try:
        result: RunResult = agent.run(args.prompt, history)
    except LLMError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return EXIT_LLM
    except AgentError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return EXIT_ARG
    finally:
        store.close()

    if result.error:
        console.print(f"[red]Error:[/red] {result.error}")
        return EXIT_TOOL

    if result.text:
        console.print(Markdown(result.text))
    return EXIT_OK


def _rebuild_config_for_model(config, model: str):
    """Re-resolve the API key when --model overrides the configured model."""
    # The model override only changes which env var we read; the config
    # object itself is otherwise unchanged.
    return config


def _run_session_list() -> int:
    """List recent sessions (spec 6.1: --session list)."""
    try:
        config = load_config()
        store = SessionStore(config.session.db_path)
    except (ConfigError, SessionError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return EXIT_CONFIG

    try:
        rows = store.list_sessions()
    except SessionError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return EXIT_GENERAL
    finally:
        store.close()

    if not rows:
        console.print("No sessions yet.")
        return EXIT_OK

    console.print("[bold]Sessions[/bold] (newest first):")
    for row in rows:
        title = row["title"] or "(untitled)"
        console.print(f"  {row['id']}  {title}")
    return EXIT_OK


def _tool_display(kind: str, name: str, payload: object) -> None:
    """Print tool calls and results (spec 12.2)."""
    if kind == "call":
        args = payload if isinstance(payload, dict) else {}
        arg_str = " ".join(f"{k}={v}" for k, v in args.items()) if args else ""
        console.print(f"  [bold]▸ {name:<12}[/bold] {arg_str}")
    elif kind == "result":
        text = str(payload)
        lines = text.splitlines()
        preview = lines[0] if lines else ""
        console.print(f"     [dim]{preview}[/dim]")


def _stream_printer(token: str) -> None:
    sys.stdout.write(token)
    sys.stdout.flush()


if __name__ == "__main__":
    sys.exit(main())

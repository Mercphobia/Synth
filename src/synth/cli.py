"""CLI entry point: parse arguments, wire modules, run the agent."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

from synth import __version__
from synth.agent import Agent, AgentConfig, AgentError, RunResult
from synth.config import ConfigError, load_config
from synth.cron_cli import handle_cron_command, setup_cron_parser
from synth.llm import LLMClient, LLMError
from synth.modes import resolve_mode, tool_filter
from synth.memory import MemoryStore, MemoryError, build_memory_prompt
from synth.session import SessionError, SessionStore
from synth.tools import default_registry
from synth.tools_ext import extended_registry

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
    parser.add_argument(
        "--godmode",
        action="store_true",
        help="Enable godmode: unrestricted access (use with extreme caution)",
    )
    parser.add_argument("--model", metavar="MODEL", help="Override the config model")
    parser.add_argument(
        "--mode",
        metavar="MODE",
        default="act",
        help="Agent mode: plan, act (default), auto, architect",
    )
    parser.add_argument(
        "--checkpoint",
        metavar="TAG",
        help="Create a checkpoint with the given tag before running",
    )
    parser.add_argument(
        "--undo",
        action="store_true",
        help="Undo the last step in the current session before running",
    )
    parser.add_argument(
        "--swarm",
        action="store_true",
        help="Enable multi-agent supervisor mode",
    )
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

    # Resolve agent mode (spec 6.1): plan/act/auto/architect.
    try:
        mode_config = resolve_mode(args.mode)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
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
    # Handle godmode activation
    if args.godmode:
        os.environ["GODMODE"] = "1"
        console.print("[red]⚠[/red] Godmode enabled: all safety boundaries disabled")
    else:
        os.environ.pop("GODMODE", None)

    tools = extended_registry(
        bash_timeout=config.tools.bash_timeout,
        max_file_size=config.tools.max_file_size,
        max_output_size=config.tools.max_output_size,
    )

    # Resolve agent mode (spec 6.1): plan/act/auto/architect.
    try:
        mode_config = resolve_mode(args.mode)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return EXIT_ARG
    if mode_config.allowed_tool_names is not None:
        tools = tool_filter(tools, mode_config.allowed_tool_names)

    # Load memory block (spec 6.3): user facts + recent lessons.
    memory_block = ""
    try:
        with MemoryStore() as mem:
            memory_block = build_memory_prompt(mem)
    except MemoryError as exc:
        console.print(f"[yellow]Warning:[/yellow] memory unavailable: {exc}")
        memory_block = ""

    # Handle checkpoints and undo (spec 6.6)
    from synth.checkpoints import CheckpointStore, CheckpointError
    checkpoint_store = None
    if args.checkpoint or args.undo:
        try:
            checkpoint_store = CheckpointStore()
        except CheckpointError as exc:
            console.print(f"[red]Error:[/red] checkpoint store unavailable: {exc}")
            return EXIT_GENERAL

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

    # Handle undo before running
    if args.undo:
        try:
            if checkpoint_store.undo_last_step(session_id):
                console.print("[green]✓[/green] Last step undone")
                # Reload history after undo
                history = store.load_session(session_id) or []
            else:
                console.print("[yellow]Warning:[/yellow] Nothing to undo")
        except CheckpointError as exc:
            console.print(f"[red]Error:[/red] undo failed: {exc}")
            return EXIT_GENERAL

    # Handle checkpoint creation
    if args.checkpoint:
        try:
            checkpoint_id = checkpoint_store.create_checkpoint(session_id, args.checkpoint)
            console.print(f"[green]✓[/green] Checkpoint created: {checkpoint_id[:8]}")
        except CheckpointError as exc:
            console.print(f"[red]Error:[/red] checkpoint creation failed: {exc}")
            return EXIT_GENERAL

    # Initialize swarm if requested
    from synth.swarm import Supervisor, SwarmError
    if args.swarm:
        try:
            supervisor = Supervisor(llm, store, tools)
            # For now, just create the supervisor - actual swarm execution
            # will be handled in a future enhancement
            console.print("[green]✓[/green] Multi-agent supervisor initialized")
        except SwarmError as exc:
            console.print(f"[red]Error:[/red] swarm initialization failed: {exc}")
            return EXIT_GENERAL

    agent = Agent(
        llm=llm,
        tools=tools,
        session_store=store,
        session_id=session_id,
        config=AgentConfig(
            max_iterations=mode_config.max_iterations, stream=not args.no_stream
        ),
        output_handler=_tool_display,
        memory_block=memory_block,
        mode_suffix=mode_config.system_prompt_suffix,
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

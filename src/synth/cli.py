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
from synth.daemon import main_daemon
from synth.dx import run_config, run_doctor, run_init
from synth.llm import LLMClient, LLMError
from synth.longtask import LongTaskStore, TaskPlanner
from synth.modes import resolve_mode, tool_filter
from synth.memory import MemoryStore, MemoryError, build_memory_prompt
from synth.model_manager import ManagerError, ModelManager
from synth.security import full_scan, report_to_markdown
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
# Spec 11 exit codes 6-9 (added with the v1.0 security/sandbox features).
EXIT_SECURITY = 6      # security violation (e.g. godmode denied, audit block)
EXIT_SANDBOX = 7       # sandbox error
EXIT_PERMISSION = 8    # permission denied
EXIT_CANCELLED = 9     # cancelled by user (SIGINT)

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


def _dispatch_command(argv: list[str]) -> int:
    """Dispatch subcommand families that don't fit the prompt positional.

    Each command builds its own private parser, so the main parser keeps a
    single positional 'prompt' argument (a shared argparse subparser would
    consume every prompt that starts with a reserved word).
    """
    command = argv[0]
    rest = argv[1:]

    if command == "cron":
        parser = argparse.ArgumentParser(prog="synth cron")
        cron_sub = parser.add_subparsers(dest="action", required=True)
        add_p = cron_sub.add_parser("add")
        add_p.add_argument("expression")
        add_p.add_argument("command", nargs=argparse.REMAINDER)
        rm_p = cron_sub.add_parser("remove")
        rm_p.add_argument("job_id")
        cron_sub.add_parser("list")
        en_p = cron_sub.add_parser("enable")
        en_p.add_argument("job_id")
        di_p = cron_sub.add_parser("disable")
        di_p.add_argument("job_id")
        cron_sub.add_parser("run")
        args = parser.parse_args(rest)
        if args.action == "add":
            args.command = " ".join(args.command)
        return handle_cron_command(args)

    if command == "daemon":
        return main_daemon(rest)

    if command == "scan":
        parser = argparse.ArgumentParser(prog="synth scan")
        parser.add_argument("path", nargs="?", default=".")
        parser.add_argument("--markdown", action="store_true", help="Output markdown")
        args = parser.parse_args(rest)
        return _run_security_scan(Path(args.path), args.markdown)

    if command == "models":
        parser = argparse.ArgumentParser(prog="synth models")
        parser.add_argument("action", choices=["list", "available", "install", "uninstall", "status"])
        parser.add_argument("name", nargs="?", default=None)
        parser.add_argument("--tier", default=None)
        args = parser.parse_args(rest)
        return _run_models_command(args)

    if command == "task":
        parser = argparse.ArgumentParser(prog="synth task")
        parser.add_argument("action", choices=["list", "create", "status", "delete"])
        parser.add_argument("arg", nargs="?", default=None)
        parser.add_argument("--goal", default=None)
        parser.add_argument("--steps", default=None, help="Comma-separated step descriptions")
        args = parser.parse_args(rest)
        return _run_task_command(args)

    if command == "doctor":
        return run_doctor(console)

    if command == "init":
        parser = argparse.ArgumentParser(prog="synth init")
        parser.add_argument("--force", action="store_true", help="Overwrite AGENTS.md")
        args = parser.parse_args(rest)
        return run_init(console, force=args.force)

    if command == "config":
        parser = argparse.ArgumentParser(prog="synth config")
        parser.add_argument("action", choices=["path", "get", "set"])
        parser.add_argument("key", nargs="?", default=None)
        parser.add_argument("value", nargs="?", default=None)
        args = parser.parse_args(rest)
        return run_config(console, args.action, args.key, args.value)

    console.print(f"[red]Error:[/red] unknown command: {command}")
    return EXIT_ARG


def _run_security_scan(root: Path, as_markdown: bool) -> int:
    """Run the security auditor over a repo path."""
    try:
        report = full_scan(root)
    except Exception as exc:  # noqa: BLE001 — boundary
        console.print(f"[red]Error:[/red] scan failed: {exc}")
        return EXIT_GENERAL

    if as_markdown:
        console.print(report_to_markdown(report))
        return EXIT_OK

    summary = report.get("summary", {})
    console.print(
        f"[bold]Security scan:[/bold] {report.get('total', 0)} findings "
        f"(CRITICAL={summary.get('CRITICAL', 0)} HIGH={summary.get('HIGH', 0)} "
        f"MEDIUM={summary.get('MEDIUM', 0)} LOW={summary.get('LOW', 0)})"
    )
    for f in report.get("findings", [])[:20]:
        console.print(f"  [{f.severity}] {f.category}: {f.path}:{f.line}")
    return EXIT_OK


def _run_models_command(args: argparse.Namespace) -> int:
    """Handle 'synth models ...' commands."""
    manager = ModelManager()
    if args.action == "available":
        for model in manager.recommend(tier=args.tier):
            console.print(f"  {model.name:<20} {model.tier:<9} ~{model.size_hint_gb} GB  {model.description}")
        return EXIT_OK
    if args.action == "list":
        for name in manager.installed():
            console.print(f"  {name}")
        return EXIT_OK
    if args.action == "status":
        status = manager.status()
        console.print(f"Ollama available: {status['available']}")
        console.print(f"Installed models: {status['models']}")
        if status["free_gb"] is not None:
            console.print(f"Free disk: {status['free_gb']:.1f} GB")
        return EXIT_OK
    if not args.name:
        console.print("[red]Error:[/red] model name required")
        return EXIT_ARG
    try:
        if args.action == "install":
            result = manager.install(args.name, callback=lambda s: console.print(f"  {s}"))
            console.print(f"{args.name}: {result}")
        elif args.action == "uninstall":
            ok = manager.uninstall(args.name)
            console.print(f"{args.name}: {'removed' if ok else 'not installed'}")
    except ManagerError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return EXIT_GENERAL
    return EXIT_OK


def _run_task_command(args: argparse.Namespace) -> int:
    """Handle 'synth task ...' commands (long-horizon tasks)."""
    store = LongTaskStore()
    try:
        if args.action == "list":
            for task in store.list_tasks(include_completed=True):
                pct = task.progress_percent()
                console.print(f"  #{task.id} [{pct:.0f}%] {task.goal}")
            return EXIT_OK
        if args.action == "create":
            goal = args.arg or ""
            if not goal:
                console.print("[red]Error:[/red] goal text required")
                return EXIT_ARG
            steps = [s.strip() for s in args.steps.split(",")] if args.steps else TaskPlanner.decompose(goal)
            task = store.create_task(goal, steps)
            console.print(f"Created task #{task.id} with {len(task.steps)} steps")
            for i, s in enumerate(task.steps):
                console.print(f"  {i}. {s.description}")
            return EXIT_OK
        if args.action == "status":
            if args.arg is None:
                console.print("[red]Error:[/red] task id required")
                return EXIT_ARG
            task = store.get_task(int(args.arg))
            if task is None:
                console.print(f"[red]Error:[/red] no task #{args.arg}")
                return EXIT_ARG
            console.print(f"Task #{task.id}: {task.goal} ({task.progress_percent():.0f}%)")
            for i, s in enumerate(task.steps):
                mark = "x" if s.completed else " "
                console.print(f"  [{mark}] {i}. {s.description}")
            return EXIT_OK
        if args.action == "delete":
            if args.arg is None or not store.delete_task(int(args.arg)):
                console.print("[red]Error:[/red] task not found")
                return EXIT_ARG
            console.print("Task deleted")
            return EXIT_OK
    finally:
        store.close()
    return EXIT_ARG


def main(argv: list[str] | None = None) -> int:
    """Program entry. Returns the exit code (spec 6.3)."""
    # Manual subcommand dispatch: keeps the prompt positional argument clean
    # (a shared argparse subparser conflicts with 'prompt', so commands are
    # detected as the first argv token instead).
    argv = list(argv) if argv is not None else sys.argv[1:]
    if argv and argv[0] in {"cron", "scan", "daemon", "models", "task", "doctor",
                            "init", "config"}:
        return _dispatch_command(argv)

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
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user.[/yellow]")
        store.close()
        return EXIT_CANCELLED
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

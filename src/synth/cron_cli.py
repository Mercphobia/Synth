"""CLI integration for cron scheduler."""

from __future__ import annotations

import argparse
from typing import List

from rich.console import Console
from rich.table import Table

from synth.cron import CronScheduler

console = Console()


def setup_cron_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add cron subcommands to CLI parser."""
    cron_parser = subparsers.add_parser(
        "cron", help="Schedule and manage cron jobs"
    )
    cron_subparsers = cron_parser.add_subparsers(
        dest="cron_command", required=True, metavar="COMMAND"
    )

    # Add job
    add_parser = cron_subparsers.add_parser(
        "add", help="Add a new cron job"
    )
    add_parser.add_argument(
        "expression", help="Cron expression (e.g., '*/5 * * * *')"
    )
    add_parser.add_argument(
        "command", help="Command to execute"
    )

    # Remove job
    remove_parser = cron_subparsers.add_parser(
        "remove", help="Remove a cron job"
    )
    remove_parser.add_argument(
        "job_id", type=int, help="Job ID to remove"
    )

    # List jobs
    list_parser = cron_subparsers.add_parser(
        "list", help="List all cron jobs"
    )
    list_parser.add_argument(
        "--show-disabled", action="store_true",
        help="Show disabled jobs (default: show only enabled)"
    )

    # Enable job
    enable_parser = cron_subparsers.add_parser(
        "enable", help="Enable a disabled job"
    )
    enable_parser.add_argument(
        "job_id", type=int, help="Job ID to enable"
    )

    # Disable job
    disable_parser = cron_subparsers.add_parser(
        "disable", help="Disable a job"
    )
    disable_parser.add_argument(
        "job_id", type=int, help="Job ID to disable"
    )

    # Run pending jobs
    run_parser = cron_subparsers.add_parser(
        "run", help="Run pending jobs"
    )


def handle_cron_command(args: argparse.Namespace) -> int:
    """Handle cron subcommands."""
    with CronScheduler() as scheduler:
        if args.cron_command == "add":
            return _handle_cron_add(scheduler, args)
        elif args.cron_command == "remove":
            return _handle_cron_remove(scheduler, args)
        elif args.cron_command == "list":
            return _handle_cron_list(scheduler, args)
        elif args.cron_command == "enable":
            return _handle_cron_enable(scheduler, args)
        elif args.cron_command == "disable":
            return _handle_cron_disable(scheduler, args)
        elif args.cron_command == "run":
            return _handle_cron_run(scheduler, args)
    return 0


def _handle_cron_add(scheduler: CronScheduler, args: argparse.Namespace) -> int:
    """Handle cron add command."""
    try:
        job_id = scheduler.add_job(args.expression, args.command)
        console.print(f"[green]✓[/green] Added job {job_id}: {args.expression}")
        return 0
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1


def _handle_cron_remove(scheduler: CronScheduler, args: argparse.Namespace) -> int:
    """Handle cron remove command."""
    try:
        scheduler.remove_job(args.job_id)
        console.print(f"[green]✓[/green] Removed job {args.job_id}")
        return 0
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1


def _handle_cron_list(scheduler: CronScheduler, args: argparse.Namespace) -> int:
    """Handle cron list command."""
    jobs = scheduler.list_jobs()
    
    if not jobs:
        console.print("No cron jobs")
        return 0
    
    # Filter if needed
    if not args.show_disabled:
        jobs = [j for j in jobs if j.enabled]
    
    if not jobs:
        console.print("No enabled cron jobs (use --show-disabled to see all)")
        return 0
    
    table = Table(title="Cron Jobs")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Expression", style="yellow")
    table.add_column("Command", style="white")
    table.add_column("Enabled", style="green")
    table.add_column("Last Run", style="dim")
    
    for job in jobs:
        enabled_str = "✓" if job.enabled else "✗"
        
        if job.last_run:
            import datetime
            last_run_dt = datetime.datetime.fromtimestamp(job.last_run)
            last_run_str = last_run_dt.strftime("%Y-%m-%d %H:%M")
        else:
            last_run_str = "Never"
        
        table.add_row(
            str(job.id),
            job.expression,
            job.command[:50] + ("..." if len(job.command) > 50 else ""),
            enabled_str,
            last_run_str,
        )
    
    console.print(table)
    return 0


def _handle_cron_enable(scheduler: CronScheduler, args: argparse.Namespace) -> int:
    """Handle cron enable command."""
    try:
        scheduler.enable_job(args.job_id)
        console.print(f"[green]✓[/green] Enabled job {args.job_id}")
        return 0
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1


def _handle_cron_disable(scheduler: CronScheduler, args: argparse.Namespace) -> int:
    """Handle cron disable command."""
    try:
        scheduler.disable_job(args.job_id)
        console.print(f"[green]✓[/green] Disabled job {args.job_id}")
        return 0
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1


def _handle_cron_run(scheduler: CronScheduler, args: argparse.Namespace) -> int:
    """Handle cron run command."""
    try:
        scheduler.run_pending_jobs()
        console.print("[green]✓[/green] Ran pending jobs")
        return 0
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1


# Integration with main CLI
def integrate_cron_cli(parser: argparse.ArgumentParser) -> None:
    """Integrate cron commands into the main CLI parser."""
    # Add cron subcommand if subparsers exist
    subparsers = next(
        (action for action in parser._actions if isinstance(action, argparse._SubParsersAction)),
        None
    )
    if subparsers:
        setup_cron_parser(subparsers)
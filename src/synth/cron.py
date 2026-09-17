"""Cron scheduler for Synth with SQLite persistence.

Supports standard cron syntax for minute, hour, day, month, weekday fields.
Jobs are executed as subprocesses with agent prompts.
"""

from __future__ import annotations

import datetime
import logging
import re
import shlex
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from synth.constants import CONFIG_DIR

logger = logging.getLogger(__name__)
DEFAULT_CRON_DB_PATH = Path.home() / CONFIG_DIR / "cron.db"

# Cron field positions
MINUTE, HOUR, DAY, MONTH, WEEKDAY = range(5)


@dataclass(frozen=True)
class CronJob:
    """Represents a scheduled cron job."""

    id: int
    expression: str
    command: str
    enabled: bool
    created_at: int
    last_run: Optional[int] = None


class CronScheduler:
    """Manages cron jobs with SQLite persistence."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialize the cron scheduler.

        Args:
            db_path: Path to cron database. None uses default location.
        """
        if db_path is None:
            self.db_path = DEFAULT_CRON_DB_PATH
        else:
            self.db_path = Path(str(db_path)).expanduser()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        """Create cron tables if they don't exist."""
        schema = """
        CREATE TABLE IF NOT EXISTS cron_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            expression TEXT NOT NULL,
            command TEXT NOT NULL,
            enabled BOOLEAN NOT NULL DEFAULT 1,
            created_at INTEGER NOT NULL,
            last_run INTEGER
        );

        CREATE TABLE IF NOT EXISTS job_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            run_at INTEGER NOT NULL,
            exit_code INTEGER,
            stdout TEXT,
            stderr TEXT,
            FOREIGN KEY (job_id) REFERENCES cron_jobs(id) ON DELETE CASCADE
        );

        -- Keep only last 100 executions per job
        CREATE TRIGGER IF NOT EXISTS trim_history 
        AFTER INSERT ON job_history
        BEGIN
            DELETE FROM job_history 
            WHERE job_id = NEW.job_id 
            AND id NOT IN (
                SELECT id FROM job_history 
                WHERE job_id = NEW.job_id 
                ORDER BY run_at DESC 
                LIMIT 100
            );
        END;
        """
        self._conn.executescript(schema)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    def __enter__(self) -> "CronScheduler":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def add_job(self, expression: str, command: str) -> int:
        """Add a new cron job.

        Args:
            expression: Cron expression (e.g., "*/5 * * * *")
            command: Command to execute

        Returns:
            Job ID

        Raises:
            ValueError: If expression is invalid
        """
        self._parse_cron_expression(expression)  # Validate expression
        now = int(time.time())
        cursor = self._conn.execute(
            "INSERT INTO cron_jobs (expression, command, enabled, created_at) VALUES (?, ?, ?, ?)",
            (expression, command, True, now),
        )
        self._conn.commit()
        return cursor.lastrowid

    def remove_job(self, job_id: int) -> None:
        """Remove a cron job by ID.

        Args:
            job_id: Job ID to remove

        Raises:
            ValueError: If job doesn't exist
        """
        cursor = self._conn.execute("DELETE FROM cron_jobs WHERE id = ?", (job_id,))
        if cursor.rowcount == 0:
            raise ValueError(f"Job not found: {job_id}")
        self._conn.commit()

    def list_jobs(self) -> List[CronJob]:
        """List all cron jobs."""
        cursor = self._conn.execute(
            "SELECT id, expression, command, enabled, created_at, last_run FROM cron_jobs"
        )
        return [
            CronJob(
                id=row["id"],
                expression=row["expression"],
                command=row["command"],
                enabled=bool(row["enabled"]),
                created_at=row["created_at"],
                last_run=row["last_run"],
            )
            for row in cursor.fetchall()
        ]

    def enable_job(self, job_id: int) -> None:
        """Enable a cron job.

        Args:
            job_id: Job ID to enable

        Raises:
            ValueError: If job doesn't exist
        """
        cursor = self._conn.execute(
            "UPDATE cron_jobs SET enabled = 1 WHERE id = ?", (job_id,)
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Job not found: {job_id}")
        self._conn.commit()

    def disable_job(self, job_id: int) -> None:
        """Disable a cron job.

        Args:
            job_id: Job ID to disable

        Raises:
            ValueError: If job doesn't exist
        """
        cursor = self._conn.execute(
            "UPDATE cron_jobs SET enabled = 0 WHERE id = ?", (job_id,)
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Job not found: {job_id}")
        self._conn.commit()

    def _parse_cron_expression(self, expression: str) -> List[List[int]]:
        """Parse a cron expression into lists of allowed values for each field.

        Args:
            expression: Cron expression (e.g., "*/5 * * * *")

        Returns:
            List of 5 lists containing allowed values for each field

        Raises:
            ValueError: If expression is invalid
        """
        fields = expression.strip().split()
        if len(fields) != 5:
            raise ValueError(f"Invalid cron expression: {expression}")

        # Field ranges
        ranges = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 6)]  # minute, hour, day, month, weekday

        parsed_fields = []
        for i, field in enumerate(fields):
            min_val, max_val = ranges[i]
            values = self._parse_field(field, min_val, max_val)
            parsed_fields.append(values)

        return parsed_fields

    def _parse_field(self, field: str, min_val: int, max_val: int) -> List[int]:
        """Parse a single cron field.

        Supports:
        - * (all values)
        - */step (every step values)
        - value (single value)
        - value1,value2 (multiple values)
        - value1-value2 (range)
        - value1-value2/step (range with step)

        Args:
            field: Field string
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            List of allowed values

        Raises:
            ValueError: If field is invalid
        """
        if field == "*":
            return list(range(min_val, max_val + 1))

        # Check for step syntax
        step_match = re.match(r"^(\*|(\d+)(?:-(\d+))?)\/(\d+)$", field)
        if step_match:
            base, start, end, step_str = step_match.groups()
            step = int(step_str)
            if step <= 0:
                raise ValueError(f"Invalid step in field: {field}")

            if base == "*":
                # */step
                return list(range(min_val, max_val + 1, step))
            else:
                # start-end/step or value/step
                start_val = int(start)
                end_val = int(end) if end else start_val
                if start_val < min_val or end_val > max_val or start_val > end_val:
                    raise ValueError(f"Invalid range in field: {field}")
                return list(range(start_val, end_val + 1, step))

        # Handle comma-separated values and ranges
        values = set()
        for part in field.split(","):
            if "-" in part:
                # Range
                range_parts = part.split("-")
                if len(range_parts) != 2:
                    raise ValueError(f"Invalid range in field: {field}")
                start_val = int(range_parts[0])
                end_val = int(range_parts[1])
                if start_val < min_val or end_val > max_val or start_val > end_val:
                    raise ValueError(f"Invalid range in field: {field}")
                values.update(range(start_val, end_val + 1))
            else:
                # Single value
                try:
                    val = int(part)
                except ValueError:
                    raise ValueError(f"Invalid value in field: {field}")
                if val < min_val or val > max_val:
                    raise ValueError(f"Invalid value in field: {field}")
                values.add(val)

        return sorted(values)

    def _should_run(self, job: CronJob, now: datetime.datetime) -> bool:
        """Check if a job should run at the given time.

        Args:
            job: Cron job to check
            now: Current time

        Returns:
            True if job should run
        """
        if not job.enabled:
            return False

        try:
            parsed = self._parse_cron_expression(job.expression)
        except ValueError:
            logger.error(f"Invalid cron expression for job {job.id}: {job.expression}")
            return False

        # Convert weekday to match cron (0=Sunday, 6=Saturday)
        # Python's weekday(): Monday=0, Sunday=6
        # So we need to convert: Python weekday -> cron weekday
        python_weekday = now.weekday()
        cron_weekday = (python_weekday + 1) % 7  # Monday=1, Sunday=0

        # Check each field
        checks = [
            now.minute in parsed[MINUTE],
            now.hour in parsed[HOUR],
            now.day in parsed[DAY],
            now.month in parsed[MONTH],
            cron_weekday in parsed[WEEKDAY],
        ]

        return all(checks)

    def get_next_run_time(self, expression: str, from_time: Optional[datetime.datetime] = None) -> Optional[datetime.datetime]:
        """Get the next run time for a cron expression.

        Args:
            expression: Cron expression
            from_time: Time to start searching from (default: now)

        Returns:
            Next run time, or None if invalid expression
        """
        if from_time is None:
            from_time = datetime.datetime.now()

        try:
            parsed = self._parse_cron_expression(expression)
        except ValueError:
            return None

        current = from_time.replace(second=0, microsecond=0)
        
        # Try up to 366 days (account for leap years)
        for _ in range(366 * 24 * 60):  # 366 days in minutes
            current += datetime.timedelta(minutes=1)
            
            # Convert weekday
            python_weekday = current.weekday()
            cron_weekday = (python_weekday + 1) % 7
            
            if (current.minute in parsed[MINUTE] and
                current.hour in parsed[HOUR] and
                current.day in parsed[DAY] and
                current.month in parsed[MONTH] and
                cron_weekday in parsed[WEEKDAY]):
                return current
        
        return None

    def run_pending_jobs(self) -> None:
        """Run all pending jobs that should have run since last check.

        This handles missed jobs when the system was down.
        """
        now = datetime.datetime.now()
        jobs = self.list_jobs()
        
        for job in jobs:
            if not job.enabled:
                continue
                
            # Always run if never run before
            if job.last_run is None:
                if self._should_run(job, now):
                    self._execute_job(job)
                continue
                
            # Check for missed runs
            last_run_time = datetime.datetime.fromtimestamp(job.last_run)
            current_check = last_run_time
            
            # Find all times between last_run and now where job should have run
            while current_check < now:
                current_check += datetime.timedelta(minutes=1)
                if self._should_run(job, current_check):
                    # Don't run if it's in the future
                    if current_check <= now:
                        self._execute_job(job, run_time=current_check)
                    break  # Only run once per cycle

    def _execute_job(self, job: CronJob, run_time: Optional[datetime.datetime] = None) -> None:
        """Execute a cron job.

        Args:
            job: Job to execute
            run_time: Time to record for execution (default: now)
        """
        if run_time is None:
            run_time = datetime.datetime.now()
            
        run_timestamp = int(run_time.timestamp())
        logger.info(f"Executing cron job {job.id}: {job.command}")
        
        try:
            # Parse command safely
            cmd_parts = shlex.split(job.command)
            if not cmd_parts:
                raise ValueError("Empty command")
                
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=str(Path.home()),
            )
            
            exit_code = result.returncode
            stdout = result.stdout[:10000]  # Limit to 10KB
            stderr = result.stderr[:10000]
            
        except Exception as exc:
            exit_code = -1
            stdout = ""
            stderr = str(exc)[:10000]
            
        # Record execution
        try:
            self._conn.execute(
                "INSERT INTO job_history (job_id, run_at, exit_code, stdout, stderr) VALUES (?, ?, ?, ?, ?)",
                (job.id, run_timestamp, exit_code, stdout, stderr),
            )
            self._conn.execute(
                "UPDATE cron_jobs SET last_run = ? WHERE id = ?",
                (run_timestamp, job.id),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            logger.error(f"Failed to record job execution: {exc}")
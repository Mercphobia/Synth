"""Tests for cron scheduler."""

import datetime
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Add src to path so we can import synth modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from synth.cron import CronJob, CronScheduler


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test_cron.db"
    yield db_path
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def scheduler(temp_db):
    """Create a cron scheduler with temporary database."""
    with CronScheduler(db_path=temp_db) as scheduler:
        yield scheduler


def test_add_job(scheduler):
    """Test adding a cron job."""
    job_id = scheduler.add_job("*/5 * * * *", "echo hello")
    assert job_id == 1
    
    jobs = scheduler.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].id == 1
    assert jobs[0].expression == "*/5 * * * *"
    assert jobs[0].command == "echo hello"
    assert jobs[0].enabled is True


def test_remove_job(scheduler):
    """Test removing a cron job."""
    job_id = scheduler.add_job("0 * * * *", "test")
    assert len(scheduler.list_jobs()) == 1
    
    scheduler.remove_job(job_id)
    assert len(scheduler.list_jobs()) == 0


def test_remove_nonexistent_job(scheduler):
    """Test removing a non-existent job raises ValueError."""
    with pytest.raises(ValueError, match="Job not found"):
        scheduler.remove_job(999)


def test_list_jobs(scheduler):
    """Test listing cron jobs."""
    scheduler.add_job("0 9 * * 1", "morning_task")
    scheduler.add_job("*/10 * * * *", "frequent_task")
    
    jobs = scheduler.list_jobs()
    assert len(jobs) == 2
    assert isinstance(jobs[0], CronJob)
    assert jobs[0].expression in ["0 9 * * 1", "*/10 * * * *"]


def test_enable_disable_job(scheduler):
    """Test enabling and disabling jobs."""
    job_id = scheduler.add_job("* * * * *", "test")
    
    # Initially enabled
    jobs = scheduler.list_jobs()
    assert jobs[0].enabled is True
    
    # Disable
    scheduler.disable_job(job_id)
    jobs = scheduler.list_jobs()
    assert jobs[0].enabled is False
    
    # Enable
    scheduler.enable_job(job_id)
    jobs = scheduler.list_jobs()
    assert jobs[0].enabled is True


def test_enable_nonexistent_job(scheduler):
    """Test enabling non-existent job raises ValueError."""
    with pytest.raises(ValueError, match="Job not found"):
        scheduler.enable_job(999)


def test_disable_nonexistent_job(scheduler):
    """Test disabling non-existent job raises ValueError."""
    with pytest.raises(ValueError, match="Job not found"):
        scheduler.disable_job(999)


def test_parse_cron_expression_valid():
    """Test parsing valid cron expressions."""
    scheduler = CronScheduler(":memory:")
    
    # Every 5 minutes
    parsed = scheduler._parse_cron_expression("*/5 * * * *")
    assert parsed[0] == list(range(0, 60, 5))  # minutes
    assert parsed[1] == list(range(0, 24))     # hours
    assert parsed[2] == list(range(1, 32))     # days
    assert parsed[3] == list(range(1, 13))     # months
    assert parsed[4] == list(range(0, 7))      # weekdays
    
    # Monday 9am
    parsed = scheduler._parse_cron_expression("0 9 * * 1")
    assert parsed[0] == [0]
    assert parsed[1] == [9]
    assert parsed[2] == list(range(1, 32))
    assert parsed[3] == list(range(1, 13))
    assert parsed[4] == [1]
    
    # Multiple values
    parsed = scheduler._parse_cron_expression("0,30 8,17 * * 1-5")
    assert parsed[0] == [0, 30]
    assert parsed[1] == [8, 17]
    assert parsed[4] == [1, 2, 3, 4, 5]


def test_parse_cron_expression_invalid():
    """Test parsing invalid cron expressions raises ValueError."""
    scheduler = CronScheduler(":memory:")
    
    invalid_expressions = [
        "*/5 * * *",      # Too few fields
        "*/5 * * * * *",  # Too many fields
        "*/0 * * * *",    # Invalid step
        "60 * * * *",     # Invalid minute
        "25 24 * * *",    # Invalid hour
        "0 0 0 * *",      # Invalid day
        "0 0 * 0 *",      # Invalid month
        "0 0 * * 7",      # Invalid weekday
        "*/5a * * * *",   # Invalid step syntax
    ]
    
    for expr in invalid_expressions:
        with pytest.raises(ValueError):
            scheduler._parse_cron_expression(expr)


def test_should_run(scheduler):
    """Test job execution timing logic."""
    job = CronJob(
        id=1,
        expression="0 9 * * 1",  # Monday 9am
        command="test",
        enabled=True,
        created_at=0,
    )
    
    # Monday 9am should run
    monday_9am = datetime.datetime(2023, 10, 2, 9, 0)  # Monday
    assert scheduler._should_run(job, monday_9am) is True
    
    # Monday 10am should not run
    monday_10am = datetime.datetime(2023, 10, 2, 10, 0)
    assert scheduler._should_run(job, monday_10am) is False
    
    # Tuesday 9am should not run
    tuesday_9am = datetime.datetime(2023, 10, 3, 9, 0)  # Tuesday
    assert scheduler._should_run(job, tuesday_9am) is False
    
    # Disabled job should not run
    disabled_job = CronJob(
        id=2,
        expression="* * * * *",
        command="test",
        enabled=False,
        created_at=0,
    )
    now = datetime.datetime.now()
    assert scheduler._should_run(disabled_job, now) is False


def test_get_next_run_time(scheduler):
    """Test calculating next run time."""
    # Every 5 minutes
    next_run = scheduler.get_next_run_time("*/5 * * * *")
    assert next_run is not None
    assert next_run.minute % 5 == 0
    
    # Monday 9am
    next_monday = scheduler.get_next_run_time("0 9 * * 1")
    assert next_monday is not None
    assert next_monday.weekday() == 0  # Monday
    assert next_monday.hour == 9
    assert next_monday.minute == 0
    
    # Invalid expression returns None
    assert scheduler.get_next_run_time("invalid") is None


def test_weekday_conversion():
    """Test weekday conversion between Python and cron formats."""
    scheduler = CronScheduler(":memory:")
    
    # Test all days of the week
    for python_weekday in range(7):  # Monday=0, Sunday=6
        test_time = datetime.datetime(2023, 10, 2 + python_weekday, 9, 0)
        cron_weekday = (python_weekday + 1) % 7  # Convert to cron format
        
        # Create a job that only runs on this specific weekday
        job = CronJob(
            id=1,
            expression=f"0 9 * * {cron_weekday}",
            command="test",
            enabled=True,
            created_at=0,
        )
        
        assert scheduler._should_run(job, test_time) is True


def test_run_pending_jobs_new_job(scheduler):
    """Test running pending jobs for newly added jobs."""
    job_id = scheduler.add_job("* * * * *", "echo test")
    
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["echo", "test"], returncode=0, stdout="test\n", stderr=""
        )
        
        scheduler.run_pending_jobs()
        
        # Should have executed once
        mock_run.assert_called_once()
        
        # Check history
        cursor = scheduler._conn.execute(
            "SELECT * FROM job_history WHERE job_id = ?", (job_id,)
        )
        history = cursor.fetchall()
        assert len(history) == 1
        assert history[0]["exit_code"] == 0


def test_run_pending_jobs_missed_execution(scheduler):
    """Test running missed executions when system was down."""
    job_id = scheduler.add_job("0 9 * * *", "morning_task")  # Daily at 9am
    
    # Simulate last run was yesterday at 9am
    yesterday_9am = int((datetime.datetime.now() - datetime.timedelta(days=1)).replace(
        hour=9, minute=0, second=0, microsecond=0
    ).timestamp())
    
    scheduler._conn.execute(
        "UPDATE cron_jobs SET last_run = ? WHERE id = ?", (yesterday_9am, job_id)
    )
    scheduler._conn.commit()
    
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["morning_task"], returncode=0, stdout="", stderr=""
        )
        
        scheduler.run_pending_jobs()
        
        # Should execute today's 9am job if we're past 9am
        now = datetime.datetime.now()
        if now.hour >= 9:
            mock_run.assert_called_once()
        else:
            mock_run.assert_not_called()


def test_job_history_limit(scheduler):
    """Test job history is limited to 100 entries per job."""
    job_id = scheduler.add_job("* * * * *", "test")
    
    # Add 150 history entries
    for i in range(150):
        scheduler._conn.execute(
            "INSERT INTO job_history (job_id, run_at, exit_code) VALUES (?, ?, ?)",
            (job_id, int(datetime.datetime.now().timestamp()) - i, 0)
        )
    
    scheduler._conn.commit()
    
    # Trigger the trim trigger by adding one more
    scheduler._conn.execute(
        "INSERT INTO job_history (job_id, run_at, exit_code) VALUES (?, ?, ?)",
        (job_id, int(datetime.datetime.now().timestamp()), 0)
    )
    scheduler._conn.commit()
    
    # Should only have 100 entries
    cursor = scheduler._conn.execute(
        "SELECT COUNT(*) FROM job_history WHERE job_id = ?", (job_id,)
    )
    count = cursor.fetchone()[0]
    assert count == 100


def test_execute_job_error_handling(scheduler):
    """Test error handling during job execution."""
    job_id = scheduler.add_job("* * * * *", "nonexistent_command_12345")
    
    scheduler._execute_job(scheduler.list_jobs()[0])
    
    # Should have recorded failure
    cursor = scheduler._conn.execute(
        "SELECT exit_code, stderr FROM job_history WHERE job_id = ?", (job_id,)
    )
    result = cursor.fetchone()
    assert result["exit_code"] == -1
    assert "nonexistent_command_12345" in result["stderr"]


def test_empty_command_handling(scheduler):
    """Test handling of empty commands."""
    job_id = scheduler.add_job("* * * * *", "")
    
    scheduler._execute_job(scheduler.list_jobs()[0])
    
    cursor = scheduler._conn.execute(
        "SELECT exit_code, stderr FROM job_history WHERE job_id = ?", (job_id,)
    )
    result = cursor.fetchone()
    assert result["exit_code"] == -1
    assert "Empty command" in result["stderr"]


def test_database_initialization(temp_db):
    """Test database initialization creates correct schema."""
    with CronScheduler(db_path=temp_db) as scheduler:
        # Check tables exist
        cursor = scheduler._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert "cron_jobs" in tables
        assert "job_history" in tables
        
        # Check trigger exists
        cursor = scheduler._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        )
        triggers = {row[0] for row in cursor.fetchall()}
        assert "trim_history" in triggers


def test_context_manager(temp_db):
    """Test context manager functionality."""
    # Connection should be closed after context
    with CronScheduler(db_path=temp_db) as s:
        assert s._conn is not None
    
    # Create a new scheduler to test the original connection is still open
    with CronScheduler(db_path=temp_db) as s2:
        assert s2._conn is not None
"""Coverage tests for synth.daemon — PID management, health checks, signals.

Exercises the Daemon class, DaemonConfig, HealthCheckHandler, and
_process_task.  Source is left untouched; all paths are isolated via tmp_path.
"""

from __future__ import annotations

import io
import json
import os
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest

from synth.daemon import Daemon, DaemonConfig, HealthCheckHandler


# --- DaemonConfig defaults ---------------------------------------------------


def test_daemon_config_defaults():
    """DaemonConfig exposes health_port, sleep_interval, and Path defaults."""
    cfg = DaemonConfig()
    assert cfg.health_port == 8765
    assert cfg.sleep_interval == 5.0
    assert isinstance(cfg.pid_file, Path)
    assert isinstance(cfg.tasks_dir, Path)
    assert isinstance(cfg.log_file, Path)


# --- _write_pid / _remove_pid round-trip -------------------------------------


def test_write_and_remove_pid_round_trip(tmp_path):
    """_write_pid writes os.getpid(); _remove_pid deletes the file."""
    cfg = DaemonConfig(
        pid_file=tmp_path / "synth.pid",
        log_file=tmp_path / "daemon.log",
        tasks_dir=tmp_path / "tasks",
    )
    d = Daemon(cfg)
    d._write_pid()
    assert cfg.pid_file.exists()
    assert cfg.pid_file.read_text() == str(os.getpid())
    d._remove_pid()
    assert not cfg.pid_file.exists()


def test_remove_pid_missing_is_silent(tmp_path):
    """_remove_pid must not raise when the pid file is already gone."""
    cfg = DaemonConfig(
        pid_file=tmp_path / "missing.pid",
        log_file=tmp_path / "daemon.log",
        tasks_dir=tmp_path / "tasks",
    )
    d = Daemon(cfg)
    # Should not raise.
    d._remove_pid()


# --- _is_running -------------------------------------------------------------


def test_is_running_false_when_no_pid_file(tmp_path):
    cfg = DaemonConfig(
        pid_file=tmp_path / "nope.pid",
        log_file=tmp_path / "daemon.log",
        tasks_dir=tmp_path / "tasks",
    )
    d = Daemon(cfg)
    assert d._is_running() is False


def test_is_running_false_when_dead_pid(tmp_path):
    """A pid that does not exist (no process) → False, and file is cleaned."""
    cfg = DaemonConfig(
        pid_file=tmp_path / "dead.pid",
        log_file=tmp_path / "daemon.log",
        tasks_dir=tmp_path / "tasks",
    )
    # Pick a pid guaranteed not to exist (very high number on Linux/Android).
    cfg.pid_file.write_text("999999999")
    d = Daemon(cfg)
    assert d._is_running() is False
    # Stale pid file should have been removed.
    assert not cfg.pid_file.exists()


def test_is_running_true_when_live_pid(tmp_path):
    """A pid equal to os.getpid() (this test process) → True."""
    cfg = DaemonConfig(
        pid_file=tmp_path / "live.pid",
        log_file=tmp_path / "daemon.log",
        tasks_dir=tmp_path / "tasks",
    )
    cfg.pid_file.write_text(str(os.getpid()))
    d = Daemon(cfg)
    assert d._is_running() is True


def test_is_running_false_on_invalid_pid(tmp_path):
    """Garbage in the pid file → ValueError path → False."""
    cfg = DaemonConfig(
        pid_file=tmp_path / "bad.pid",
        log_file=tmp_path / "daemon.log",
        tasks_dir=tmp_path / "tasks",
    )
    cfg.pid_file.write_text("not-a-number")
    d = Daemon(cfg)
    assert d._is_running() is False


# --- HealthCheckHandler ------------------------------------------------------


class _StubServer:
    """Minimal server with wfile/rfile for BaseHTTPRequestHandler tests."""

    def __init__(self, path: str):
        self.path = path
        self._wfile = io.BytesIO()
        self.rfile = io.BytesIO(b"")

    @property
    def wfile(self):
        return self._wfile

    def get_output(self) -> bytes:
        return self._wfile.getvalue()


def _make_handler(path: str) -> HealthCheckHandler:
    server = _StubServer(path)
    handler = HealthCheckHandler.__new__(HealthCheckHandler)
    handler.path = path
    # Wire up send_response/send_header/end_headers/wfile without socket.
    sent = {"status": None, "headers": []}

    def send_response(code, *args, **kwargs):
        sent["status"] = code

    def send_header(key, val):
        sent["headers"].append((key, val))

    def end_headers():
        pass

    handler.send_response = send_response
    handler.send_header = send_header
    handler.end_headers = end_headers
    handler.wfile = server.wfile
    handler._sent = sent  # type: ignore[attr-defined]
    return handler


def test_health_handler_do_get_health_returns_200():
    h = _make_handler("/health")
    h.do_GET()
    assert h._sent["status"] == 200
    body = json.loads(h.wfile.getvalue().decode())
    assert body["status"] == "healthy"
    assert "timestamp" in body


def test_health_handler_do_get_other_returns_404():
    h = _make_handler("/other")
    h.do_GET()
    assert h._sent["status"] == 404


# --- _signal_handler ---------------------------------------------------------


def test_signal_handler_sets_running_false(tmp_path):
    cfg = DaemonConfig(
        pid_file=tmp_path / "synth.pid",
        log_file=tmp_path / "daemon.log",
        tasks_dir=tmp_path / "tasks",
    )
    d = Daemon(cfg)
    d._running = True
    d._signal_handler(signal.Signals.SIGTERM, None)  # type: ignore[name-defined]
    assert d._running is False


# --- _process_task -----------------------------------------------------------


def test_process_task_skips_empty_file(tmp_path):
    """Empty task file → logged warning, returns True, file NOT deleted."""
    cfg = DaemonConfig(
        pid_file=tmp_path / "synth.pid",
        log_file=tmp_path / "daemon.log",
        tasks_dir=tmp_path / "tasks",
    )
    d = Daemon(cfg)
    task = tmp_path / "empty.task"
    task.write_text("   ")
    result = d._process_task(task)
    assert result is True
    # Empty file is skipped, not removed.
    assert task.exists()


def test_process_task_runs_subprocess(tmp_path):
    """Non-empty task → subprocess.run called → returns True, file deleted."""
    cfg = DaemonConfig(
        pid_file=tmp_path / "synth.pid",
        log_file=tmp_path / "daemon.log",
        tasks_dir=tmp_path / "tasks",
    )
    d = Daemon(cfg)
    task = tmp_path / "job.task"
    task.write_text("hello world")

    class _FakeResult:
        returncode = 0
        stdout = "ok"
        stderr = ""

    with patch("synth.daemon.subprocess.run", return_value=_FakeResult()) as m:
        result = d._process_task(task)
    assert result is True
    m.assert_called_once()
    # Successful task → file deleted.
    assert not task.exists()


def test_process_task_subprocess_failure(tmp_path):
    """Subprocess returns non-zero → False, file kept."""
    cfg = DaemonConfig(
        pid_file=tmp_path / "synth.pid",
        log_file=tmp_path / "daemon.log",
        tasks_dir=tmp_path / "tasks",
    )
    d = Daemon(cfg)
    task = tmp_path / "fail.task"
    task.write_text("boom")

    class _FakeResult:
        returncode = 1
        stdout = ""
        stderr = "error here"

    with patch("synth.daemon.subprocess.run", return_value=_FakeResult()):
        result = d._process_task(task)
    assert result is False
    assert task.exists()


def test_process_task_timeout(tmp_path):
    """subprocess.TimeoutExpired → False."""
    cfg = DaemonConfig(
        pid_file=tmp_path / "synth.pid",
        log_file=tmp_path / "daemon.log",
        tasks_dir=tmp_path / "tasks",
    )
    d = Daemon(cfg)
    task = tmp_path / "slow.task"
    task.write_text("takes long")

    import subprocess

    def _raise(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="synth", timeout=300)

    with patch("synth.daemon.subprocess.run", side_effect=_raise):
        result = d._process_task(task)
    assert result is False


def test_process_task_generic_exception(tmp_path):
    """Any other exception in processing → False."""
    cfg = DaemonConfig(
        pid_file=tmp_path / "synth.pid",
        log_file=tmp_path / "daemon.log",
        tasks_dir=tmp_path / "tasks",
    )
    d = Daemon(cfg)
    task = tmp_path / "x.task"
    task.write_text("payload")

    with patch("synth.daemon.subprocess.run", side_effect=RuntimeError("boom")):
        result = d._process_task(task)
    assert result is False


# --- Daemon.__init__ ---------------------------------------------------------


def test_daemon_init_creates_tasks_dir(tmp_path):
    """__init__ must create the tasks_dir and pid_file parent."""
    tasks = tmp_path / "deep" / "tasks"
    cfg = DaemonConfig(
        pid_file=tmp_path / "deep" / "sub" / "synth.pid",
        log_file=tmp_path / "daemon.log",
        tasks_dir=tasks,
    )
    assert not tasks.exists()
    Daemon(cfg)
    assert tasks.is_dir()
    assert cfg.pid_file.parent.is_dir()


# --- start() / stop() / status() --------------------------------------------


def test_start_returns_1_when_already_running(tmp_path):
    """start() with a live pid file already present → returns 1."""
    cfg = DaemonConfig(
        pid_file=tmp_path / "synth.pid",
        log_file=tmp_path / "daemon.log",
        tasks_dir=tmp_path / "tasks",
    )
    # Pretend a daemon is already running with our own pid.
    cfg.pid_file.write_text(str(os.getpid()))
    d = Daemon(cfg)
    rc = d.start()
    assert rc == 1


def test_stop_returns_0_when_not_running(tmp_path):
    """stop() with no pid file → returns 0 (nothing to stop)."""
    cfg = DaemonConfig(
        pid_file=tmp_path / "synth.pid",
        log_file=tmp_path / "daemon.log",
        tasks_dir=tmp_path / "tasks",
    )
    d = Daemon(cfg)
    rc = d.stop()
    assert rc == 0


def test_status_not_running_prints_stopped(tmp_path, capsys):
    """status() when not running → prints 'stopped', returns 1."""
    cfg = DaemonConfig(
        pid_file=tmp_path / "synth.pid",
        log_file=tmp_path / "daemon.log",
        tasks_dir=tmp_path / "tasks",
    )
    d = Daemon(cfg)
    rc = d.status()
    out = capsys.readouterr().out
    assert rc == 1
    assert "stopped" in out.lower()


def test_status_running_prints_running(tmp_path, capsys):
    """status() when daemon is running (live pid) → prints 'running', 0."""
    cfg = DaemonConfig(
        pid_file=tmp_path / "synth.pid",
        log_file=tmp_path / "daemon.log",
        tasks_dir=tmp_path / "tasks",
    )
    cfg.pid_file.write_text(str(os.getpid()))
    d = Daemon(cfg)
    rc = d.status()
    out = capsys.readouterr().out
    assert rc == 0
    assert "running" in out.lower()


def test_stop_kills_running_process(tmp_path):
    """stop() sends SIGTERM then removes the pid file.

    We can't fork a child here without extra machinery, so we simulate a
    live daemon by writing a pid for a long-lived child thread/process we
    spawn via os.fork-free approach: spawn a subprocess that sleeps.
    """
    import subprocess

    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        cfg = DaemonConfig(
            pid_file=tmp_path / "synth.pid",
            log_file=tmp_path / "daemon.log",
            tasks_dir=tmp_path / "tasks",
        )
        cfg.pid_file.write_text(str(sleeper.pid))
        d = Daemon(cfg)
        rc = d.stop()
        assert rc == 0
        # pid file removed by stop().
        assert not cfg.pid_file.exists()
        # Sleeper should have been terminated.
        sleeper.wait(timeout=5)
    finally:
        if sleeper.poll() is None:
            sleeper.kill()


# --- _start_health_server / _stop_health_server (integration) ---------------


def test_health_server_lifecycle(tmp_path):
    """Start + stop the real HTTP health server on a free port."""
    # Find a free port up front to avoid clashes.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("localhost", 0))
    port = sock.getsockname()[1]
    sock.close()

    cfg = DaemonConfig(
        pid_file=tmp_path / "synth.pid",
        log_file=tmp_path / "daemon.log",
        tasks_dir=tmp_path / "tasks",
        health_port=port,
    )
    d = Daemon(cfg)
    d._start_health_server()
    try:
        # Probe /health.
        with urllib.request.urlopen(
            f"http://localhost:{port}/health", timeout=3
        ) as resp:
            assert resp.status == 200
            body = json.loads(resp.read().decode())
            assert body["status"] == "healthy"
        # Probe /other → 404.
        try:
            urllib.request.urlopen(
                f"http://localhost:{port}/other", timeout=3
            )
            assert False, "expected 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        d._stop_health_server()


# --- _process_tasks loop ----------------------------------------------------


def test_process_tasks_processes_all_files(tmp_path):
    """_process_tasks iterates *.task files in sorted order."""
    cfg = DaemonConfig(
        pid_file=tmp_path / "synth.pid",
        log_file=tmp_path / "daemon.log",
        tasks_dir=tmp_path / "tasks",
    )
    cfg.tasks_dir.mkdir(parents=True, exist_ok=True)
    (cfg.tasks_dir / "a.task").write_text("prompt-a")
    (cfg.tasks_dir / "b.task").write_text("prompt-b")
    d = Daemon(cfg)
    d._running = True

    class _FakeResult:
        returncode = 0
        stdout = ""
        stderr = ""

    with patch("synth.daemon.subprocess.run", return_value=_FakeResult()) as m:
        d._process_tasks()
    assert m.call_count == 2
    # All successful → all deleted.
    assert not list(cfg.tasks_dir.glob("*.task"))


def test_process_tasks_stops_when_not_running(tmp_path):
    """_process_tasks breaks out of the loop when _running flips to False."""
    cfg = DaemonConfig(
        pid_file=tmp_path / "synth.pid",
        log_file=tmp_path / "daemon.log",
        tasks_dir=tmp_path / "tasks",
    )
    cfg.tasks_dir.mkdir(parents=True, exist_ok=True)
    (cfg.tasks_dir / "a.task").write_text("prompt-a")
    (cfg.tasks_dir / "b.task").write_text("prompt-b")
    d = Daemon(cfg)
    d._running = False  # not running → loop should break immediately

    with patch("synth.daemon.subprocess.run") as m:
        d._process_tasks()
    assert m.call_count == 0


# --- main_daemon -------------------------------------------------------------


def test_main_daemon_no_args_returns_1(capsys):
    from synth.daemon import main_daemon

    rc = main_daemon([])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Usage" in out


def test_main_daemon_unknown_command_returns_1(tmp_path, capsys, monkeypatch):
    from synth.daemon import main_daemon, Daemon

    # Redirect config to tmp so we don't touch real ~/.synth.
    def _fake_init(self, config=None):
        config = config or DaemonConfig(
            pid_file=tmp_path / "synth.pid",
            log_file=tmp_path / "daemon.log",
            tasks_dir=tmp_path / "tasks",
        )
        self.config = config
        self.config.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.config.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self._running = False
        self._httpd = None
        import logging
        logging.basicConfig(handlers=[])  # silence
        self.logger = logging.getLogger("synth.daemon")

    monkeypatch.setattr(Daemon, "__init__", _fake_init)
    rc = main_daemon(["bogus"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Unknown daemon command" in out


def test_main_daemon_status_when_stopped(tmp_path, capsys, monkeypatch):
    from synth.daemon import main_daemon, Daemon

    def _fake_init(self, config=None):
        config = config or DaemonConfig(
            pid_file=tmp_path / "synth.pid",
            log_file=tmp_path / "daemon.log",
            tasks_dir=tmp_path / "tasks",
        )
        self.config = config
        self.config.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.config.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self._running = False
        self._httpd = None
        import logging
        logging.basicConfig(handlers=[])
        self.logger = logging.getLogger("synth.daemon")

    monkeypatch.setattr(Daemon, "__init__", _fake_init)
    rc = main_daemon(["status"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "stopped" in out.lower()


# Late import keeps the test module importable on older Pythons.
import signal
import urllib.error

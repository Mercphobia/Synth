"""Daemon mode for Synth: run agent in background with task queue.

Implements a long-running daemon that processes prompts from a directory,
with PID management, graceful shutdown, and health checks.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from synth.config import load_config
from synth.constants import CONFIG_DIR


@dataclass
class DaemonConfig:
    """Configuration for the daemon process."""
    
    pid_file: Path = Path.home() / CONFIG_DIR / "synth.pid"
    log_file: Path = Path.home() / CONFIG_DIR / "daemon.log"
    tasks_dir: Path = Path.home() / CONFIG_DIR / "tasks"
    health_port: int = 8765
    sleep_interval: float = 5.0


class HealthCheckHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for health checks."""
    
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response = {"status": "healthy", "timestamp": datetime.now().isoformat()}
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()


class Daemon:
    """Background daemon that processes Synth tasks."""
    
    def __init__(self, config: DaemonConfig | None = None) -> None:
        self.config = config or DaemonConfig()
        self.config.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.config.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self._running = False
        self._httpd: HTTPServer | None = None
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Configure logging to file."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.config.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _write_pid(self) -> None:
        """Write PID to file."""
        try:
            self.config.pid_file.write_text(str(os.getpid()))
            atexit.register(self._remove_pid)
        except OSError as exc:
            self.logger.error(f"Failed to write PID file: {exc}")
            raise
            
    def _remove_pid(self) -> None:
        """Remove PID file on exit."""
        try:
            self.config.pid_file.unlink(missing_ok=True)
        except OSError:
            pass
            
    def _is_running(self) -> bool:
        """Check if daemon is already running."""
        if not self.config.pid_file.exists():
            return False
            
        try:
            pid = int(self.config.pid_file.read_text().strip())
            # Check if process exists by sending signal 0
            os.kill(pid, 0)
            return True
        except (OSError, ValueError):
            # Process doesn't exist or invalid PID
            self.config.pid_file.unlink(missing_ok=True)
            return False
            
    def _start_health_server(self) -> None:
        """Start health check HTTP server in background thread."""
        def run_server():
            try:
                self._httpd = HTTPServer(("localhost", self.config.health_port), HealthCheckHandler)
                self._httpd.serve_forever()
            except Exception as exc:
                self.logger.error(f"Health server error: {exc}")
                
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        # Give server time to start
        time.sleep(0.1)
        
    def _stop_health_server(self) -> None:
        """Stop health check HTTP server."""
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
            
    def _process_task(self, task_file: Path) -> bool:
        """Process a single task file."""
        try:
            prompt = task_file.read_text().strip()
            if not prompt:
                self.logger.warning(f"Skipping empty task file: {task_file}")
                return True
                
            # Run synth with the prompt
            result = subprocess.run(
                [sys.executable, "-m", "synth", prompt],
                capture_output=True,
                text=True,
                cwd=str(Path.cwd()),
                timeout=300  # 5 minute timeout per task
            )
            
            if result.returncode == 0:
                self.logger.info(f"Task completed successfully: {task_file.name}")
                task_file.unlink(missing_ok=True)
                return True
            else:
                self.logger.error(f"Task failed ({result.returncode}): {task_file.name}")
                self.logger.error(f"stderr: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Task timeout: {task_file.name}")
            return False
        except Exception as exc:
            self.logger.error(f"Task processing error: {exc}")
            return False
            
    def _process_tasks(self) -> None:
        """Process all pending tasks."""
        try:
            task_files = sorted(self.config.tasks_dir.glob("*.task"))
            for task_file in task_files:
                if not self._running:
                    break
                self._process_task(task_file)
        except Exception as exc:
            self.logger.error(f"Task processing loop error: {exc}")
            
    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self._running = False
        
    def start(self) -> int:
        """Start the daemon process."""
        if self._is_running():
            self.logger.error("Daemon is already running")
            return 1
            
        self._write_pid()
        self._start_health_server()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Daemon started")
        self._running = True
        
        try:
            while self._running:
                self._process_tasks()
                time.sleep(self.config.sleep_interval)
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        finally:
            self._running = False
            self._stop_health_server()
            self.logger.info("Daemon stopped")
            
        return 0
        
    def stop(self) -> int:
        """Stop the running daemon."""
        if not self._is_running():
            self.logger.warning("Daemon is not running")
            return 0
            
        try:
            pid = int(self.config.pid_file.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            self.logger.info("Stop signal sent to daemon")
            
            # Wait for process to terminate
            for _ in range(10):  # 10 seconds max
                try:
                    os.kill(pid, 0)  # Check if process exists
                    time.sleep(1)
                except OSError:
                    break
            else:
                # Force kill if still running
                try:
                    os.kill(pid, signal.SIGKILL)
                    self.logger.warning("Force killed unresponsive daemon")
                except OSError:
                    pass
                    
            self.config.pid_file.unlink(missing_ok=True)
            return 0
        except (OSError, ValueError) as exc:
            self.logger.error(f"Failed to stop daemon: {exc}")
            return 1
            
    def status(self) -> int:
        """Check daemon status."""
        if self._is_running():
            print("Daemon is running")
            return 0
        else:
            print("Daemon is stopped")
            return 1


def main_daemon(args: list[str]) -> int:
    """CLI entry point for daemon commands."""
    if len(args) < 1:
        print("Usage: synth --daemon [start|stop|status]")
        return 1
        
    command = args[0].lower()
    daemon = Daemon()
    
    if command == "start":
        return daemon.start()
    elif command == "stop":
        return daemon.stop()
    elif command == "status":
        return daemon.status()
    else:
        print(f"Unknown daemon command: {command}")
        return 1
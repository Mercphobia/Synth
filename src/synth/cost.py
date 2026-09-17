"""Cost tracking for LLM usage.

Tracks token usage and calculates costs per model, with SQLite persistence
and budget alerts.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synth.constants import CONFIG_DIR


# Model pricing in USD per 1M tokens (input, output)
_MODEL_PRICING = {
    # OpenAI models
    "gpt-4": (30.0, 60.0),
    "gpt-3.5-turbo": (0.5, 1.5),
    # Anthropic models
    "claude-3-sonnet": (3.0, 15.0),
    # Meta models
    "llama-3.1-70b": (0.59, 0.79),
    # Ollama models (free)
    "ollama": (0.0, 0.0),
}


def _get_model_pricing(model_name: str) -> tuple[float, float]:
    """Get input/output pricing for a model."""
    # Handle model name variations
    if model_name.startswith("gpt-4"):
        return _MODEL_PRICING["gpt-4"]
    elif model_name.startswith("gpt-3.5"):
        return _MODEL_PRICING["gpt-3.5-turbo"]
    elif "claude-3-sonnet" in model_name:
        return _MODEL_PRICING["claude-3-sonnet"]
    elif "llama-3.1-70b" in model_name:
        return _MODEL_PRICING["llama-3.1-70b"]
    elif model_name.startswith("ollama"):
        return _MODEL_PRICING["ollama"]
    else:
        # Default to free for unknown models
        return (0.0, 0.0)


@dataclass(frozen=True)
class CostRecord:
    """A single cost tracking record."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    session_id: str
    timestamp: int


class CostError(Exception):
    """Raised when cost tracking fails."""


class CostTracker:
    """Tracks LLM usage costs with SQLite persistence."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        daily_budget: float | None = None,
        monthly_budget: float | None = None,
    ) -> None:
        """Initialize the cost tracker.

        Args:
            db_path: Path to cost database. None uses default location.
            daily_budget: Daily spending limit in USD.
            monthly_budget: Monthly spending limit in USD.
        """
        if db_path is None:
            self.db_path = Path.home() / CONFIG_DIR / "cost.db"
        else:
            self.db_path = Path(str(db_path)).expanduser()

        self.daily_budget = daily_budget
        self.monthly_budget = monthly_budget

        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._migrate()
        except sqlite3.Error as exc:
            raise CostError(f"Cannot open cost DB {self.db_path}: {exc}") from exc

    def _migrate(self) -> None:
        """Create or migrate the database schema."""
        schema = """
        CREATE TABLE IF NOT EXISTS costs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            input_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            cost_usd REAL NOT NULL,
            session_id TEXT NOT NULL,
            timestamp INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_costs_timestamp ON costs(timestamp);
        CREATE INDEX IF NOT EXISTS idx_costs_session ON costs(session_id);
        """
        try:
            self._conn.executescript(schema)
            self._conn.commit()
        except sqlite3.Error as exc:
            raise CostError(f"Schema migration failed: {exc}") from exc

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    def __enter__(self) -> "CostTracker":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def record_cost(
        self, model: str, usage: dict[str, int], session_id: str
    ) -> CostRecord:
        """Record a cost entry and return the record.

        Args:
            model: The model name used.
            usage: Dictionary with 'input_tokens' and 'output_tokens'.
            session_id: The session identifier.

        Returns:
            CostRecord with calculated cost.

        Raises:
            CostError: If recording fails or budget exceeded.
        """
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        input_price, output_price = _get_model_pricing(model)
        cost_usd = (input_tokens * input_price + output_tokens * output_price) / 1_000_000

        timestamp = int(time.time())
        record = CostRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            session_id=session_id,
            timestamp=timestamp,
        )

        # Check budget before recording
        self._check_budget_limits(cost_usd)

        try:
            self._conn.execute(
                "INSERT INTO costs (model, input_tokens, output_tokens, cost_usd, session_id, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record.model,
                    record.input_tokens,
                    record.output_tokens,
                    record.cost_usd,
                    record.session_id,
                    record.timestamp,
                ),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise CostError(f"Failed to record cost: {exc}") from exc

        return record

    def _check_budget_limits(self, additional_cost: float) -> None:
        """Check if adding cost would exceed budget limits."""
        if self.daily_budget is not None:
            daily_total = self.get_daily_cost()
            projected_total = daily_total + additional_cost
            
            # Check if would exceed 100% of budget (raise error)
            if projected_total >= self.daily_budget:
                raise CostError(
                    f"Daily budget exceeded: ${projected_total:.4f} >= ${self.daily_budget:.4f}"
                )
            # Check if would reach 80% warning threshold
            elif projected_total >= self.daily_budget * 0.8:
                import sys
                sys.stdout.write(
                    f"WARNING: Daily budget at {projected_total / self.daily_budget * 100:.1f}%\n"
                )
                sys.stdout.flush()

        if self.monthly_budget is not None:
            monthly_total = self.get_monthly_cost()
            projected_total = monthly_total + additional_cost
            
            # Check if would exceed 100% of budget (raise error)
            if projected_total >= self.monthly_budget:
                raise CostError(
                    f"Monthly budget exceeded: ${projected_total:.4f} >= ${self.monthly_budget:.4f}"
                )
            # Check if would reach 80% warning threshold
            elif projected_total >= self.monthly_budget * 0.8:
                import sys
                sys.stdout.write(
                    f"WARNING: Monthly budget at {projected_total / self.monthly_budget * 100:.1f}%\n"
                )
                sys.stdout.flush()

    def get_session_cost(self, session_id: str) -> float:
        """Get total cost for a session."""
        try:
            cursor = self._conn.execute(
                "SELECT SUM(cost_usd) FROM costs WHERE session_id = ?", (session_id,)
            )
            result = cursor.fetchone()[0]
            return float(result) if result is not None else 0.0
        except sqlite3.Error as exc:
            raise CostError(f"Failed to get session cost: {exc}") from exc

    def get_daily_cost(self, timestamp: int | None = None) -> float:
        """Get total cost for today (or specified day)."""
        if timestamp is None:
            timestamp = int(time.time())
        
        # Get start of day (midnight)
        start_of_day = timestamp - (timestamp % 86400)
        end_of_day = start_of_day + 86400

        try:
            cursor = self._conn.execute(
                "SELECT SUM(cost_usd) FROM costs WHERE timestamp >= ? AND timestamp < ?",
                (start_of_day, end_of_day),
            )
            result = cursor.fetchone()[0]
            return float(result) if result is not None else 0.0
        except sqlite3.Error as exc:
            raise CostError(f"Failed to get daily cost: {exc}") from exc

    def get_monthly_cost(self, timestamp: int | None = None) -> float:
        """Get total cost for current month (or specified month)."""
        if timestamp is None:
            timestamp = int(time.time())
        
        # Calculate start of month
        import datetime
        dt = datetime.datetime.fromtimestamp(timestamp)
        start_of_month = int(datetime.datetime(dt.year, dt.month, 1).timestamp())
        
        # Calculate start of next month
        if dt.month == 12:
            next_month = datetime.datetime(dt.year + 1, 1, 1)
        else:
            next_month = datetime.datetime(dt.year, dt.month + 1, 1)
        end_of_month = int(next_month.timestamp())

        try:
            cursor = self._conn.execute(
                "SELECT SUM(cost_usd) FROM costs WHERE timestamp >= ? AND timestamp < ?",
                (start_of_month, end_of_month),
            )
            result = cursor.fetchone()[0]
            return float(result) if result is not None else 0.0
        except sqlite3.Error as exc:
            raise CostError(f"Failed to get monthly cost: {exc}") from exc

    def get_total_cost(self) -> float:
        """Get total cost across all time."""
        try:
            cursor = self._conn.execute("SELECT SUM(cost_usd) FROM costs")
            result = cursor.fetchone()[0]
            return float(result) if result is not None else 0.0
        except sqlite3.Error as exc:
            raise CostError(f"Failed to get total cost: {exc}") from exc
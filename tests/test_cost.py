"""Tests for cost tracking functionality."""

import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from synth.cost import CostError, CostRecord, CostTracker, _get_model_pricing


def test_get_model_pricing():
    """Test model pricing lookup."""
    # OpenAI models
    assert _get_model_pricing("gpt-4") == (30.0, 60.0)
    assert _get_model_pricing("gpt-4-0613") == (30.0, 60.0)
    assert _get_model_pricing("gpt-3.5-turbo") == (0.5, 1.5)
    assert _get_model_pricing("gpt-3.5-turbo-0125") == (0.5, 1.5)
    
    # Anthropic models
    assert _get_model_pricing("claude-3-sonnet-20240229") == (3.0, 15.0)
    assert _get_model_pricing("claude-3-sonnet") == (3.0, 15.0)
    
    # Meta models
    assert _get_model_pricing("llama-3.1-70b") == (0.59, 0.79)
    assert _get_model_pricing("meta-llama-3.1-70b-instruct") == (0.59, 0.79)
    
    # Ollama models
    assert _get_model_pricing("ollama/llama3") == (0.0, 0.0)
    assert _get_model_pricing("ollama/mistral") == (0.0, 0.0)
    
    # Unknown models default to free
    assert _get_model_pricing("unknown-model") == (0.0, 0.0)


def test_cost_record_creation():
    """Test CostRecord dataclass."""
    record = CostRecord(
        model="gpt-4",
        input_tokens=1000,
        output_tokens=500,
        cost_usd=0.06,
        session_id="test-session",
        timestamp=1234567890,
    )
    assert record.model == "gpt-4"
    assert record.input_tokens == 1000
    assert record.output_tokens == 500
    assert record.cost_usd == 0.06
    assert record.session_id == "test-session"
    assert record.timestamp == 1234567890


def test_cost_tracker_initialization(tmp_path):
    """Test CostTracker initialization."""
    db_path = tmp_path / "test_cost.db"
    tracker = CostTracker(db_path=db_path)
    assert tracker.db_path == db_path
    assert tracker._conn is not None
    tracker.close()


def test_cost_tracker_context_manager(tmp_path):
    """Test CostTracker context manager."""
    db_path = tmp_path / "test_cost.db"
    with CostTracker(db_path=db_path) as tracker:
        assert tracker.db_path == db_path
        assert tracker._conn is not None


def test_record_cost_basic(tmp_path):
    """Test basic cost recording."""
    db_path = tmp_path / "test_cost.db"
    with CostTracker(db_path=db_path) as tracker:
        usage = {"input_tokens": 1000, "output_tokens": 500}
        record = tracker.record_cost("gpt-4", usage, "test-session")
        
        assert record.model == "gpt-4"
        assert record.input_tokens == 1000
        assert record.output_tokens == 500
        assert abs(record.cost_usd - 0.06) < 0.0001  # 1000*30 + 500*60 = 60000 tokens = $0.06
        assert record.session_id == "test-session"
        assert record.timestamp > 0


def test_record_cost_gpt35(tmp_path):
    """Test GPT-3.5 cost calculation."""
    db_path = tmp_path / "test_cost.db"
    with CostTracker(db_path=db_path) as tracker:
        usage = {"input_tokens": 2000, "output_tokens": 1000}
        record = tracker.record_cost("gpt-3.5-turbo", usage, "test-session")
        
        expected_cost = (2000 * 0.5 + 1000 * 1.5) / 1_000_000  # $0.0025
        assert abs(record.cost_usd - expected_cost) < 0.0001


def test_record_cost_claude(tmp_path):
    """Test Claude cost calculation."""
    db_path = tmp_path / "test_cost.db"
    with CostTracker(db_path=db_path) as tracker:
        usage = {"input_tokens": 1000, "output_tokens": 2000}
        record = tracker.record_cost("claude-3-sonnet", usage, "test-session")
        
        expected_cost = (1000 * 3.0 + 2000 * 15.0) / 1_000_000  # $0.033
        assert abs(record.cost_usd - expected_cost) < 0.0001


def test_record_cost_llama(tmp_path):
    """Test Llama cost calculation."""
    db_path = tmp_path / "test_cost.db"
    with CostTracker(db_path=db_path) as tracker:
        usage = {"input_tokens": 1000, "output_tokens": 1000}
        record = tracker.record_cost("llama-3.1-70b", usage, "test-session")
        
        expected_cost = (1000 * 0.59 + 1000 * 0.79) / 1_000_000  # $0.00138
        assert abs(record.cost_usd - expected_cost) < 0.0001


def test_record_cost_ollama(tmp_path):
    """Test Ollama (free) cost calculation."""
    db_path = tmp_path / "test_cost.db"
    with CostTracker(db_path=db_path) as tracker:
        usage = {"input_tokens": 10000, "output_tokens": 5000}
        record = tracker.record_cost("ollama/llama3", usage, "test-session")
        
        assert record.cost_usd == 0.0


def test_get_session_cost(tmp_path):
    """Test session cost aggregation."""
    db_path = tmp_path / "test_cost.db"
    with CostTracker(db_path=db_path) as tracker:
        # Record multiple costs for same session
        usage1 = {"input_tokens": 1000, "output_tokens": 500}
        usage2 = {"input_tokens": 2000, "output_tokens": 1000}
        
        tracker.record_cost("gpt-4", usage1, "test-session")
        tracker.record_cost("gpt-4", usage2, "test-session")
        
        total_cost = tracker.get_session_cost("test-session")
        expected_cost = (3000 * 30.0 + 1500 * 60.0) / 1_000_000  # $0.18
        assert abs(total_cost - expected_cost) < 0.0001


def test_get_session_cost_empty(tmp_path):
    """Test session cost for non-existent session."""
    db_path = tmp_path / "test_cost.db"
    with CostTracker(db_path=db_path) as tracker:
        total_cost = tracker.get_session_cost("non-existent-session")
        assert total_cost == 0.0


def test_get_total_cost(tmp_path):
    """Test total cost aggregation."""
    db_path = tmp_path / "test_cost.db"
    with CostTracker(db_path=db_path) as tracker:
        tracker.record_cost("gpt-4", {"input_tokens": 1000, "output_tokens": 500}, "session1")
        tracker.record_cost("gpt-3.5-turbo", {"input_tokens": 2000, "output_tokens": 1000}, "session2")
        
        total_cost = tracker.get_total_cost()
        expected_cost = (1000 * 30.0 + 500 * 60.0 + 2000 * 0.5 + 1000 * 1.5) / 1_000_000
        assert abs(total_cost - expected_cost) < 0.0001


def test_get_total_cost_empty(tmp_path):
    """Test total cost when no records exist."""
    db_path = tmp_path / "test_cost.db"
    with CostTracker(db_path=db_path) as tracker:
        total_cost = tracker.get_total_cost()
        assert total_cost == 0.0


@patch("time.time")
def test_get_daily_cost(mock_time, tmp_path):
    """Test daily cost aggregation."""
    db_path = tmp_path / "test_cost.db"
    mock_time.return_value = 1700000000  # Fixed timestamp
    
    with CostTracker(db_path=db_path) as tracker:
        # Record costs on the same day
        tracker.record_cost("gpt-4", {"input_tokens": 1000, "output_tokens": 500}, "session1")
        tracker.record_cost("gpt-4", {"input_tokens": 2000, "output_tokens": 1000}, "session2")
        
        daily_cost = tracker.get_daily_cost()
        expected_cost = (3000 * 30.0 + 1500 * 60.0) / 1_000_000  # $0.18
        assert abs(daily_cost - expected_cost) < 0.0001


@patch("time.time")
def test_get_daily_cost_different_days(mock_time, tmp_path):
    """Test daily cost excludes other days."""
    db_path = tmp_path / "test_cost.db"
    
    with CostTracker(db_path=db_path) as tracker:
        # Record cost on day 1
        mock_time.return_value = 1700000000
        tracker.record_cost("gpt-4", {"input_tokens": 1000, "output_tokens": 500}, "session1")
        
        # Record cost on day 2
        mock_time.return_value = 1700000000 + 86400  # Next day
        tracker.record_cost("gpt-4", {"input_tokens": 2000, "output_tokens": 1000}, "session2")
        
        # Get cost for day 1 only
        daily_cost = tracker.get_daily_cost(1700000000)
        expected_cost = (1000 * 30.0 + 500 * 60.0) / 1_000_000  # $0.06
        assert abs(daily_cost - expected_cost) < 0.0001


@patch("time.time")
def test_get_monthly_cost(mock_time, tmp_path):
    """Test monthly cost aggregation."""
    db_path = tmp_path / "test_cost.db"
    mock_time.return_value = 1700000000  # November 2023
    
    with CostTracker(db_path=db_path) as tracker:
        tracker.record_cost("gpt-4", {"input_tokens": 1000, "output_tokens": 500}, "session1")
        tracker.record_cost("gpt-4", {"input_tokens": 2000, "output_tokens": 1000}, "session2")
        
        monthly_cost = tracker.get_monthly_cost()
        expected_cost = (3000 * 30.0 + 1500 * 60.0) / 1_000_000  # $0.18
        assert abs(monthly_cost - expected_cost) < 0.0001


def test_daily_budget_warning(tmp_path, capsys):
    """Test daily budget warning at 80%."""
    db_path = tmp_path / "test_cost.db"
    with CostTracker(db_path=db_path, daily_budget=0.10) as tracker:
        # Record cost that brings us to 81% of budget (to ensure trigger)
        usage = {"input_tokens": 2700, "output_tokens": 0}  # ~$0.081
        tracker.record_cost("gpt-4", usage, "test-session")
        
        captured = capsys.readouterr()
        assert "WARNING: Daily budget" in captured.out


def test_daily_budget_exceeded(tmp_path):
    """Test daily budget exceeded error."""
    db_path = tmp_path / "test_cost.db"
    with CostTracker(db_path=db_path, daily_budget=0.05) as tracker:
        # Record cost within budget
        tracker.record_cost("gpt-4", {"input_tokens": 1000, "output_tokens": 0}, "session1")  # $0.03
        
        # Try to record cost that exceeds budget
        with pytest.raises(CostError, match="Daily budget exceeded"):
            tracker.record_cost("gpt-4", {"input_tokens": 1000, "output_tokens": 0}, "session2")  # Would be $0.06 total


def test_monthly_budget_warning(tmp_path, capsys):
    """Test monthly budget warning at 80%."""
    db_path = tmp_path / "test_cost.db"
    with CostTracker(db_path=db_path, monthly_budget=0.10) as tracker:
        # Record cost that brings us to 81% of budget (to ensure trigger)
        usage = {"input_tokens": 2700, "output_tokens": 0}  # ~$0.081
        tracker.record_cost("gpt-4", usage, "test-session")
        
        captured = capsys.readouterr()
        assert "WARNING: Monthly budget" in captured.out


def test_monthly_budget_exceeded(tmp_path):
    """Test monthly budget exceeded error."""
    db_path = tmp_path / "test_cost.db"
    with CostTracker(db_path=db_path, monthly_budget=0.05) as tracker:
        # Record cost within budget
        tracker.record_cost("gpt-4", {"input_tokens": 1000, "output_tokens": 0}, "session1")  # $0.03
        
        # Try to record cost that exceeds budget
        with pytest.raises(CostError, match="Monthly budget exceeded"):
            tracker.record_cost("gpt-4", {"input_tokens": 1000, "output_tokens": 0}, "session2")  # Would be $0.06 total


def test_database_persistence(tmp_path):
    """Test that costs persist across tracker instances."""
    db_path = tmp_path / "test_cost.db"
    
    # First tracker instance
    with CostTracker(db_path=db_path) as tracker1:
        tracker1.record_cost("gpt-4", {"input_tokens": 1000, "output_tokens": 500}, "test-session")
    
    # Second tracker instance
    with CostTracker(db_path=db_path) as tracker2:
        total_cost = tracker2.get_total_cost()
        expected_cost = (1000 * 30.0 + 500 * 60.0) / 1_000_000  # $0.06
        assert abs(total_cost - expected_cost) < 0.0001


def test_error_handling_invalid_usage(tmp_path):
    """Test error handling for invalid usage data."""
    db_path = tmp_path / "test_cost.db"
    with CostTracker(db_path=db_path) as tracker:
        # Test with missing keys
        usage = {}
        record = tracker.record_cost("gpt-4", usage, "test-session")
        assert record.input_tokens == 0
        assert record.output_tokens == 0
        assert record.cost_usd == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
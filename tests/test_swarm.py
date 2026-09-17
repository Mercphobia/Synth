"""Tests for multi-agent supervisor (swarm module)."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock

import pytest

from synth.agent import Agent, RunResult
from synth.session import SessionStore
from synth.swarm import SubAgentConfig, Supervisor, SwarmError
from synth.tools import ToolRegistry


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    llm = Mock()
    return llm


@pytest.fixture
def mock_session_store():
    """Create a mock session store."""
    store = Mock(spec=SessionStore)
    store.create_session.return_value = "test-session-id"
    return store


@pytest.fixture
def mock_tool_registry():
    """Create a mock tool registry with sample tools."""
    registry = Mock(spec=ToolRegistry)
    registry.names.return_value = ["read_file", "write_file", "bash"]
    registry.get.return_value = None
    return registry


@pytest.fixture
def supervisor(mock_llm, mock_session_store, mock_tool_registry):
    """Create a supervisor instance with mocked dependencies."""
    return Supervisor(mock_llm, mock_session_store, mock_tool_registry)


# --- SubAgentConfig tests ---


def test_subagent_config_creation():
    """Test SubAgentConfig dataclass creation."""
    config = SubAgentConfig(
        name="test_agent",
        role="test role",
        tools=["read_file", "write_file"],
        max_iterations=15,
    )
    assert config.name == "test_agent"
    assert config.role == "test role"
    assert config.tools == ["read_file", "write_file"]
    assert config.max_iterations == 15


def test_subagent_config_defaults():
    """Test SubAgentConfig default values."""
    config = SubAgentConfig(name="agent", role="role", tools=["tool"])
    assert config.max_iterations == 20


# --- Supervisor initialization tests ---


def test_supervisor_init(supervisor, mock_llm, mock_session_store, mock_tool_registry):
    """Test Supervisor initialization."""
    assert supervisor.llm is mock_llm
    assert supervisor.session_store is mock_session_store
    assert supervisor.base_tools is mock_tool_registry


# --- spawn_subagents tests ---


def test_spawn_subagents_creates_agents(supervisor, mock_session_store, mock_tool_registry):
    """Test spawn_subagents creates correct number of agents."""
    configs = [
        SubAgentConfig(name="agent1", role="role1", tools=["read_file"]),
        SubAgentConfig(name="agent2", role="role2", tools=["write_file"]),
    ]
    agents = supervisor.spawn_subagents(configs)
    
    assert len(agents) == 2
    assert all(isinstance(agent, Agent) for agent in agents)
    assert mock_session_store.create_session.call_count == 2


def test_spawn_subagents_empty_list_raises(supervisor):
    """Test spawn_subagents raises error for empty config list."""
    with pytest.raises(SwarmError, match="At least one subagent configuration"):
        supervisor.spawn_subagents([])


def test_spawn_subagents_invalid_config_raises(supervisor):
    """Test spawn_subagents raises error for invalid config."""
    with pytest.raises(SwarmError, match="Invalid subagent config"):
        supervisor.spawn_subagents([{"name": "invalid"}])


def test_spawn_subagents_empty_name_raises(supervisor):
    """Test spawn_subagents raises error for empty name."""
    config = SubAgentConfig(name="", role="role", tools=["tool"])
    with pytest.raises(SwarmError, match="Subagent name must be non-empty"):
        supervisor.spawn_subagents([config])


def test_spawn_subagents_empty_tools_raises(supervisor):
    """Test spawn_subagents raises error for empty tools list."""
    config = SubAgentConfig(name="agent", role="role", tools=[])
    with pytest.raises(SwarmError, match="Subagent tools must be non-empty"):
        supervisor.spawn_subagents([config])


# --- run_parallel tests ---


def test_run_parallel_executes_all_agents(supervisor, mock_llm, mock_session_store, mock_tool_registry):
    """Test run_parallel executes all agents and aggregates results."""
    # Create mock agents
    agent1 = Mock(spec=Agent)
    agent1.run.return_value = RunResult(text="Result 1", iterations=5, tool_calls=2)
    
    agent2 = Mock(spec=Agent)
    agent2.run.return_value = RunResult(text="Result 2", iterations=3, tool_calls=1)
    
    result = supervisor.run_parallel("Test task", [agent1, agent2])
    
    assert "Result 1" in result.text
    assert "Result 2" in result.text
    assert result.iterations == 8
    assert result.tool_calls == 3
    assert result.error is None
    
    agent1.run.assert_called_once_with("Test task")
    agent2.run.assert_called_once_with("Test task")


def test_run_parallel_empty_agents_raises(supervisor):
    """Test run_parallel raises error for empty agent list."""
    with pytest.raises(SwarmError, match="At least one subagent"):
        supervisor.run_parallel("task", [])


def test_run_parallel_empty_task_raises(supervisor):
    """Test run_parallel raises error for empty task."""
    agent = Mock(spec=Agent)
    with pytest.raises(SwarmError, match="Task must be a non-empty string"):
        supervisor.run_parallel("", [agent])


def test_run_parallel_handles_agent_errors(supervisor):
    """Test run_parallel handles errors from individual agents."""
    agent1 = Mock(spec=Agent)
    agent1.run.side_effect = Exception("Agent 1 failed")
    
    agent2 = Mock(spec=Agent)
    agent2.run.return_value = RunResult(text="Success", iterations=2, tool_calls=1)
    
    result = supervisor.run_parallel("task", [agent1, agent2])
    
    assert result.error is not None
    assert "Parallel execution failed" in result.error
    assert "Agent 1 failed" in result.error


# --- orchestrate_dynamic_workflow tests ---


def test_orchestrate_dynamic_workflow_sequential(supervisor, mock_llm, mock_session_store, mock_tool_registry):
    """Test orchestrate_dynamic_workflow executes steps sequentially."""
    # Mock spawn_subagents to return a mock agent
    mock_agent = Mock(spec=Agent)
    mock_agent.run.return_value = RunResult(text="Step result", iterations=3, tool_calls=1)
    
    supervisor.spawn_subagents = Mock(return_value=[mock_agent])
    
    steps = [
        {"agent_name": "agent1", "prompt": "First step"},
        {"agent_name": "agent2", "prompt": "Second step"},
    ]
    
    result = supervisor.orchestrate_dynamic_workflow("Base task", steps)
    
    assert result.text == "Step result"
    assert mock_agent.run.call_count == 2
    assert supervisor.spawn_subagents.call_count == 2


def test_orchestrate_dynamic_workflow_empty_steps_raises(supervisor):
    """Test orchestrate_dynamic_workflow raises error for empty steps."""
    with pytest.raises(SwarmError, match="At least one step"):
        supervisor.orchestrate_dynamic_workflow("task", [])


def test_orchestrate_dynamic_workflow_empty_task_raises(supervisor):
    """Test orchestrate_dynamic_workflow raises error for empty task."""
    steps = [{"agent_name": "agent", "prompt": "step"}]
    with pytest.raises(SwarmError, match="Task must be a non-empty string"):
        supervisor.orchestrate_dynamic_workflow("", steps)


def test_orchestrate_dynamic_workflow_missing_agent_name_raises(supervisor):
    """Test orchestrate_dynamic_workflow raises error for missing agent_name."""
    steps = [{"prompt": "step without agent"}]
    with pytest.raises(SwarmError, match="missing valid 'agent_name'"):
        supervisor.orchestrate_dynamic_workflow("task", steps)


def test_orchestrate_dynamic_workflow_missing_prompt_raises(supervisor):
    """Test orchestrate_dynamic_workflow raises error for missing prompt."""
    steps = [{"agent_name": "agent"}]
    with pytest.raises(SwarmError, match="missing valid 'prompt'"):
        supervisor.orchestrate_dynamic_workflow("task", steps)


def test_orchestrate_dynamic_workflow_step_failure_raises(supervisor, mock_llm, mock_session_store, mock_tool_registry):
    """Test orchestrate_dynamic_workflow propagates step failures."""
    mock_agent = Mock(spec=Agent)
    mock_agent.run.return_value = RunResult(text="", iterations=1, tool_calls=0, error="Step failed")
    
    supervisor.spawn_subagents = Mock(return_value=[mock_agent])
    
    steps = [{"agent_name": "agent1", "prompt": "Failing step"}]
    
    with pytest.raises(SwarmError, match="Dynamic workflow failed"):
        supervisor.orchestrate_dynamic_workflow("task", steps)


# --- SwarmError tests ---


def test_swarm_error_is_exception():
    """Test SwarmError is an Exception subclass."""
    error = SwarmError("test error")
    assert isinstance(error, Exception)
    assert str(error) == "test error"

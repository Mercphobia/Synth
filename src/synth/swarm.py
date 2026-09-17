"""Multi-agent supervisor: subagents, parallel execution, dynamic workflows.

Spec 6.5 Multi-Agent: Supervisor + subagents. Parallel. Dynamic workflows.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from synth.agent import Agent, AgentConfig, RunResult
from synth.modes import tool_filter
from synth.session import SessionStore
from synth.tools import ToolRegistry


class SwarmError(Exception):
    """Raised when the swarm supervisor encounters an error."""


@dataclass
class SubAgentConfig:
    """Configuration for a subagent in the swarm."""

    name: str
    role: str
    tools: list[str]
    max_iterations: int = 20


class Supervisor:
    """Supervisor that manages multiple subagents for parallel and dynamic workflows."""

    def __init__(
        self,
        llm: Any,
        session_store: SessionStore,
        base_tools: ToolRegistry,
    ) -> None:
        """Initialize the supervisor with required dependencies.

        Args:
            llm: LLMClient instance for making LLM calls.
            session_store: Session storage for managing agent sessions.
            base_tools: Base tool registry containing all available tools.
        """
        self.llm = llm
        self.session_store = session_store
        self.base_tools = base_tools

    def spawn_subagents(self, configs: list[SubAgentConfig]) -> list[Agent]:
        """Create Agent instances with filtered tools based on configurations.

        Args:
            configs: List of SubAgentConfig defining each subagent's properties.

        Returns:
            List of initialized Agent instances.

        Raises:
            SwarmError: If configuration is invalid or tool filtering fails.
        """
        if not configs:
            raise SwarmError("At least one subagent configuration is required")

        agents = []
        for config in configs:
            if not isinstance(config, SubAgentConfig):
                raise SwarmError(f"Invalid subagent config: {config}")
            if not config.name or not isinstance(config.name, str):
                raise SwarmError(f"Subagent name must be non-empty string: {config.name}")
            if not config.tools or not isinstance(config.tools, list):
                raise SwarmError(f"Subagent tools must be non-empty list: {config.tools}")

            # Create isolated session for each subagent
            session_id = self.session_store.create_session(title=f"subagent-{config.name}")

            # Filter tools to only those allowed for this subagent
            filtered_tools = tool_filter(self.base_tools, config.tools)

            # Create agent config with specified max_iterations
            agent_config = AgentConfig(max_iterations=config.max_iterations)

            # Create and store the agent
            agent = Agent(
                llm=self.llm,
                tools=filtered_tools,
                session_store=self.session_store,
                session_id=session_id,
                config=agent_config,
                memory_block=config.role,
            )
            agents.append(agent)

        return agents

    def run_parallel(self, task: str, subagents: list[Agent]) -> RunResult:
        """Execute multiple agents concurrently and aggregate results.

        Args:
            task: The task prompt to give to all subagents.
            subagents: List of Agent instances to run in parallel.

        Returns:
            Aggregated RunResult containing combined outputs.

        Raises:
            SwarmError: If no subagents provided or execution fails.
        """
        if not subagents:
            raise SwarmError("At least one subagent is required for parallel execution")

        if not isinstance(task, str) or not task.strip():
            raise SwarmError("Task must be a non-empty string")

        results: list[RunResult] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def _run_agent(agent: Agent) -> None:
            """Run a single agent and collect its result."""
            try:
                result = agent.run(task)
                with lock:
                    results.append(result)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        # Create and start threads for each agent
        threads = []
        for agent in subagents:
            thread = threading.Thread(target=_run_agent, args=(agent,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Handle any errors that occurred during execution
        if errors:
            error_messages = [str(err) for err in errors]
            combined_error = "; ".join(error_messages)
            return RunResult(
                text="",
                iterations=0,
                tool_calls=0,
                error=f"Parallel execution failed: {combined_error}",
            )

        # Aggregate successful results
        if not results:
            return RunResult(
                text="",
                iterations=0,
                tool_calls=0,
                error="No results returned from parallel execution",
            )

        # Combine all text results
        combined_text = "\n\n".join(
            f"[{i+1}] {result.text}" for i, result in enumerate(results) if result.text
        )

        # Sum up iterations and tool calls
        total_iterations = sum(result.iterations for result in results)
        total_tool_calls = sum(result.tool_calls for result in results)

        return RunResult(
            text=combined_text or "No output from subagents",
            iterations=total_iterations,
            tool_calls=total_tool_calls,
            error=None,
        )

    def orchestrate_dynamic_workflow(self, task: str, steps: list[dict]) -> RunResult:
        """Execute a dynamic workflow defined by sequential steps.

        Args:
            task: Base task context for the workflow.
            steps: List of step dictionaries, each containing:
                   - 'agent_name': Name of the agent to use
                   - 'prompt': Specific prompt for this step
                   - Optional: other step-specific parameters

        Returns:
            Final RunResult from the last step in the workflow.

        Raises:
            SwarmError: If steps are invalid or workflow execution fails.
        """
        if not steps:
            raise SwarmError("At least one step is required for dynamic workflow")

        if not isinstance(task, str) or not task.strip():
            raise SwarmError("Task must be a non-empty string")

        current_context = task
        final_result = None

        for step_idx, step in enumerate(steps):
            if not isinstance(step, dict):
                raise SwarmError(f"Step {step_idx} must be a dictionary")

            agent_name = step.get("agent_name")
            step_prompt = step.get("prompt")

            if not agent_name or not isinstance(agent_name, str):
                raise SwarmError(f"Step {step_idx} missing valid 'agent_name'")

            if not step_prompt or not isinstance(step_prompt, str):
                raise SwarmError(f"Step {step_idx} missing valid 'prompt'")

            # Create a temporary subagent config for this step
            # Since we don't have pre-configured agents, we'll create one dynamically
            # In a real scenario, this would map to existing subagent configurations
            temp_config = SubAgentConfig(
                name=agent_name,
                role=f"Step {step_idx + 1} executor",
                tools=[],  # Will be populated based on available tools
                max_iterations=10,
            )

            # For dynamic workflow, we need to determine which tools to give
            # Since we don't have a mapping, we'll use all base tools
            # In practice, this would be more sophisticated
            temp_config.tools = self.base_tools.names() or ["read_file", "write_file", "bash"]

            try:
                # Spawn a temporary agent for this step
                agents = self.spawn_subagents([temp_config])
                agent = agents[0]

                # Execute the step with the current context
                full_prompt = f"Context: {current_context}\n\nTask: {step_prompt}"
                result = agent.run(full_prompt)

                if result.error:
                    raise SwarmError(f"Step {step_idx} failed: {result.error}")

                # Update context for next step
                current_context = result.text
                final_result = result

            except Exception as exc:
                raise SwarmError(f"Dynamic workflow failed at step {step_idx}: {exc}") from exc

        if final_result is None:
            return RunResult(
                text="",
                iterations=0,
                tool_calls=0,
                error="Dynamic workflow produced no result",
            )

        return final_result
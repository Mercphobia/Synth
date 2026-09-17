"""The ReAct loop: LLM call -> tool execution -> observe -> repeat.

The agent stops when the LLM returns plain text (no tool calls) or when
max_iterations is exceeded (spec 8.2).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from synth.llm import LLMClient, LLMResponse, ToolCall
from synth.prompts import build_system_prompt
from synth.session import Message, SessionStore
from synth.tools import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


class AgentError(Exception):
    """Raised when the agent cannot complete the task."""


@dataclass
class AgentConfig:
    """Knobs for one agent run."""

    max_iterations: int = 20
    stream: bool = True


@dataclass
class RunResult:
    """Final outcome of running the agent."""

    text: str
    iterations: int = 0
    tool_calls: int = 0
    error: str | None = None


class Agent:
    """Runs the ReAct loop against an LLMClient and ToolRegistry."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        session_store: SessionStore,
        session_id: str,
        config: AgentConfig | None = None,
        output_handler: Any = None,
    ) -> None:
        """Wire the agent to its collaborators.

        Args:
            output_handler: Optional callable(str) for tool-call display.
        """
        self.llm = llm
        self.tools = tools
        self.session = session_store
        self.session_id = session_id
        self.config = config or AgentConfig()
        self.output_handler = output_handler
        self.messages: list[dict[str, Any]] = []

    def run(self, user_prompt: str, history: list[Message] | None = None) -> RunResult:
        """Run the agent until it answers or hits the iteration limit.

        Args:
            user_prompt: The user's task.
            history: Optional prior messages to prepend.

        Returns:
            RunResult with the final text or an error.

        Raises:
            AgentError: If the LLM client fails mid-loop.
        """
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise AgentError("prompt must be a non-empty string")

        import os

        self.messages = [self._system_message(os.getcwd())]
        if history:
            for msg in history:
                self.messages.append(self._history_to_dict(msg))
        self.messages.append({"role": "user", "content": user_prompt})

        self.session.append_message(
            self.session_id, Message(role="user", content=user_prompt)
        )

        tool_call_count = 0
        for iteration in range(1, self.config.max_iterations + 1):
            logger.debug("agent iteration %d/%d", iteration, self.config.max_iterations)
            response = self.llm.chat(
                self.messages,
                tools=self.tools.schemas(),
                stream=self.config.stream and True,
            )

            if response.tool_calls:
                tool_call_count += len(response.tool_calls)
                self.messages.append(self._assistant_message(response))
                self._save_assistant(response)
                for call in response.tool_calls:
                    result = self._execute_tool(call)
                    self.messages.append(self._tool_result_message(call, result))
                    self._save_tool_result(call, result)
                continue

            # Plain text = final answer.
            text = response.text or ""
            self.session.append_message(
                self.session_id, Message(role="assistant", content=text)
            )
            return RunResult(text=text, iterations=iteration, tool_calls=tool_call_count)

        return RunResult(
            text="",
            iterations=self.config.max_iterations,
            tool_calls=tool_call_count,
            error="max iterations reached",
        )

    # --- message helpers ---

    def _system_message(self, cwd: str) -> dict[str, Any]:
        return {"role": "system", "content": build_system_prompt(cwd)}

    def _history_to_dict(self, msg: Message) -> dict[str, Any]:
        """Rebuild one history row into an API message."""
        if msg.tool_call_id is not None:
            return {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content or "",
            }
        if msg.tool_calls:
            return {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": json.loads(msg.tool_calls),
            }
        return {"role": msg.role, "content": msg.content or ""}

    def _assistant_message(self, response: LLMResponse) -> dict[str, Any]:
        """Build the assistant message including raw tool_calls for the API."""
        tool_calls_raw = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.args, ensure_ascii=False),
                },
            }
            for call in (response.tool_calls or [])
        ]
        return {
            "role": "assistant",
            "content": response.text,
            "tool_calls": tool_calls_raw,
        }

    def _tool_result_message(self, call: ToolCall, result: ToolResult) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": call.id,
            "content": result.text,
        }

    def _execute_tool(self, call: ToolCall) -> ToolResult:
        """Run one tool call and display it."""
        if self.output_handler is not None:
            self.output_handler("call", call.name, call.args)
        result = self.tools.execute(call.name, call.args)
        if self.output_handler is not None:
            self.output_handler("result", call.name, result.text)
        return result

    # --- persistence helpers ---

    def _save_assistant(self, response: LLMResponse) -> None:
        tool_calls_json = None
        if response.tool_calls:
            tool_calls_json = json.dumps(
                [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.args, ensure_ascii=False),
                        },
                    }
                    for call in response.tool_calls
                ],
                ensure_ascii=False,
            )
        self.session.append_message(
            self.session_id,
            Message(
                role="assistant",
                content=response.text,
                tool_calls=tool_calls_json,
            ),
        )

    def _save_tool_result(self, call: ToolCall, result: ToolResult) -> None:
        self.session.append_message(
            self.session_id,
            Message(
                role="tool",
                content=result.text,
                tool_call_id=call.id,
            ),
        )

    # unused placeholder kept for API symmetry
    _ = field(default=None)

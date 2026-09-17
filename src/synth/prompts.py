"""System prompt templates for the Synth agent.

The prompt is assembled from three dynamic parts:
1. The base instructions (reasoning rules + tool list derived from the registry).
2. The active agent mode suffix (see modes.py, spec 6.1).
3. The memory block injected from the memory store (see memory.py, spec 6.3).
"""

from __future__ import annotations

from synth.tools import ToolRegistry

BASE_PROMPT = """You are Synth, a CLI AI agent.

You help users with software tasks by reasoning step-by-step and using tools.

Rules:
1. Always think before calling a tool.
2. Call one tool at a time.
3. Read files before writing to them.
4. If a tool fails, analyze the error and try a different approach.
5. When the task is complete, respond with a final text answer (no tool call).
6. Be concise. Show only what matters.
7. Respond in the same language the user's task was written in.

Current working directory: {cwd}
"""


def _format_tool_list(tools: ToolRegistry | None) -> str:
    """Render the registry's tools as a bulleted list.

    Args:
        tools: The tool registry whose schemas describe the available tools.

    Returns:
        A newline-joined bullet list like "- read_file(path): Read a file" plus a
        blank line, or an empty string when no registry is provided.
    """
    if tools is None:
        return ""
    lines = []
    for schema in tools.schemas():
        if schema.get("type") == "function":
            schema = schema.get("function", {})
        name = schema.get("name", "")
        description = schema.get("description", "").splitlines()[0:1]
        description_text = description[0] if description else ""
        params = schema.get("parameters", {}).get("properties", {})
        param_names = ", ".join(params.keys())
        lines.append(f"- {name}({param_names}): {description_text}")
    return "\n".join(lines) + "\n" if lines else ""


def build_system_prompt(
    cwd: str,
    tools: ToolRegistry | None = None,
    mode_suffix: str = "",
    memory_block: str = "",
) -> str:
    """Assemble the full system prompt (spec 8.1).

    Args:
        cwd: The current working directory shown to the model.
        tools: Optional registry used to build a dynamic tool list. When None,
            the legacy static tool list is omitted.
        mode_suffix: Extra instructions appended by the active agent mode.
        memory_block: Formatted memory text appended after the base prompt.

    Returns:
        The fully rendered system prompt.
    """
    parts = [BASE_PROMPT.format(cwd=cwd)]
    if tools is not None:
        parts.append("You have access to these tools:\n")
        parts.append(_format_tool_list(tools))
    if memory_block:
        parts.append(memory_block.rstrip() + "\n")
    if mode_suffix:
        parts.append(mode_suffix.rstrip() + "\n")
    return "\n".join(parts).rstrip() + "\n"

"""Agent modes (spec 6.1): PLAN, ACT, AUTO, ARCHITECT.

A mode is a bundle of the knobs that change how the agent approaches a task:
the extra instructions appended to the system prompt, the iteration budget,
and which tools the agent is allowed to call.

This module is intentionally side-effect free: it produces configs and
filtered registries, it never runs the agent.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from synth.agent import AgentConfig
from synth.constants import MAX_ITERATIONS
from synth.tools import ToolRegistry

# --- Mode names ---
# The four core modes of spec 6.1. Values are the CLI/config strings users
# type, so they stay lowercase and hyphen-free.


class Mode(str, Enum):
    """The four core agent modes."""

    PLAN = "plan"
    ACT = "act"
    AUTO = "auto"
    ARCHITECT = "architect"


# --- Iteration budgets per mode ---
# ACT is the default work mode: it inherits the project-wide default from
# constants.MAX_ITERATIONS (20) rather than restating the number, so a
# single-source change propagates everywhere.
ACT_MAX_ITERATIONS = MAX_ITERATIONS

# PLAN spends its iterations on reasoning, not tool calls, so a step is
# cheap — give it twice the default budget for deep decomposition.
PLAN_MAX_ITERATIONS = 2 * MAX_ITERATIONS

# AUTO runs long multi-step tasks unattended without asking for
# confirmation; 50 steps covers build→test→fix cycles that would otherwise
# stall out on the default budget.
AUTO_MAX_ITERATIONS = 50

# ARCHITECT only reads code and writes prose; design work converges faster
# than implementation, so it gets less than the default.
ARCHITECT_MAX_ITERATIONS = 15

# --- Tool access policies ---
# None means "every registered tool is allowed". An empty list means "no
# tools at all" — PLAN mode, which must reason without acting.
ALL_TOOLS: list[str] | None = None
NO_TOOLS: list[str] = []

# The read-only tools ARCHITECT may use. grep and glob live in the extended
# registry (tools_ext.extended_registry), not the base one; they are
# referenced by name so ARCHITECT works whether or not that extension is
# loaded — tool_filter simply skips names that are not registered.
ARCHITECT_TOOLS: list[str] = ["read_file", "glob", "grep"]

# --- System-prompt suffixes ---
# Each suffix is appended to the base system prompt by
# build_mode_system_prompt to steer the model's behavior for that mode.

PLAN_SUFFIX = """You are running in PLAN mode.

Reason carefully about the task and produce a detailed, step-by-step plan
before anything else. Consider:
- what the desired end state is,
- which steps are needed, in what order,
- what could go wrong, and how you would handle it.

You may NOT call any tools in this mode — planning only. When the plan is
complete, reply with it as your final text answer and stop. Do not start
executing the work yourself."""

ACT_SUFFIX = """You are running in ACT mode, the default work mode.

Execute the task with the tools available to you. Think one step ahead, call
one tool at a time, and read each result before continuing. When the task is
done, reply with a concise summary of what you changed."""

AUTO_SUFFIX = """You are running in AUTO mode, maximum autonomy.

Work through the entire task end to end without asking the user for
confirmation. Make reasonable decisions on your own and keep going until the
task is complete, then report what you did. Stop early only if you hit a
blocker you cannot resolve."""

ARCHITECT_SUFFIX = """You are running in ARCHITECT mode, big-picture design.

Explore the codebase with the read-only tools you have been given and reason
about its structure: components, data flow, interfaces, and trade-offs.
Produce a design document as your final text answer. Do not modify any
files."""


@dataclass(frozen=True)
class ModeConfig:
    """Every knob a mode controls.

    Attributes:
        name: The mode this config belongs to.
        system_prompt_suffix: Extra instructions appended to the base prompt.
        max_iterations: Iteration budget for the ReAct loop.
        allowed_tool_names: Names of tools the agent may call. None means all
            registered tools; an empty list means none.
        stream: Whether LLM responses are streamed.
        verbose: Whether to surface extra reasoning detail to the user.
    """

    name: Mode
    system_prompt_suffix: str
    max_iterations: int
    allowed_tool_names: list[str] | None
    stream: bool = True
    verbose: bool = False


# --- Mode presets ---
# One frozen ModeConfig per mode; resolve_mode() is the only supported way
# to look one up.

PLAN_MODE = ModeConfig(
    name=Mode.PLAN,
    system_prompt_suffix=PLAN_SUFFIX,
    max_iterations=PLAN_MAX_ITERATIONS,
    allowed_tool_names=NO_TOOLS,
    stream=True,
    verbose=True,
)

ACT_MODE = ModeConfig(
    name=Mode.ACT,
    system_prompt_suffix=ACT_SUFFIX,
    max_iterations=ACT_MAX_ITERATIONS,
    allowed_tool_names=ALL_TOOLS,
    stream=True,
    verbose=False,
)

AUTO_MODE = ModeConfig(
    name=Mode.AUTO,
    system_prompt_suffix=AUTO_SUFFIX,
    max_iterations=AUTO_MAX_ITERATIONS,
    allowed_tool_names=ALL_TOOLS,
    stream=True,
    verbose=True,
)

ARCHITECT_MODE = ModeConfig(
    name=Mode.ARCHITECT,
    system_prompt_suffix=ARCHITECT_SUFFIX,
    max_iterations=ARCHITECT_MAX_ITERATIONS,
    allowed_tool_names=ARCHITECT_TOOLS,
    stream=True,
    verbose=True,
)

# name -> preset. Keys are the lowercased Mode values.
MODES: dict[str, ModeConfig] = {
    Mode.PLAN.value: PLAN_MODE,
    Mode.ACT.value: ACT_MODE,
    Mode.AUTO.value: AUTO_MODE,
    Mode.ARCHITECT.value: ARCHITECT_MODE,
}


def resolve_mode(name: str) -> ModeConfig:
    """Look up the preset for a mode name.

    Args:
        name: Mode name — case-insensitive ("plan", "PLAN", "Plan" all work).

    Returns:
        The ModeConfig preset for that mode.

    Raises:
        ValueError: If name is not one of the four known modes, or not a
            string.
    """
    if not isinstance(name, str):
        raise ValueError(f"mode must be a string, got {type(name).__name__}")

    key = name.strip().lower()
    if key not in MODES:
        known = ", ".join(sorted(mode.value for mode in Mode))
        raise ValueError(f"unknown mode: {name!r} (expected one of: {known})")

    return MODES[key]


def build_mode_system_prompt(base_prompt: str, mode_config: ModeConfig) -> str:
    """Append a mode's instructions to a base system prompt.

    Args:
        base_prompt: The base system prompt, e.g. from
            synth.prompts.build_system_prompt().
        mode_config: The mode whose suffix is appended.

    Returns:
        The base prompt, a separator, and the mode-specific instructions.
    """
    return f"{base_prompt.rstrip()}\n\n---\n{mode_config.system_prompt_suffix.strip()}\n"


def apply_mode_to_agent_config(
    agent_config: AgentConfig, mode_config: ModeConfig
) -> AgentConfig:
    """Return a new AgentConfig carrying the mode's budget and stream setting.

    The mode's knobs win over the caller's config; every other field is kept.
    The original config is not modified.

    Args:
        agent_config: The base agent configuration.
        mode_config: The mode to apply.

    Returns:
        A new AgentConfig with max_iterations and stream taken from the mode.
    """
    return replace(
        agent_config,
        max_iterations=mode_config.max_iterations,
        stream=mode_config.stream,
    )


def tool_filter(registry: ToolRegistry, allowed_names: list[str] | None) -> ToolRegistry:
    """Return a registry that exposes only the allowed tools.

    Args:
        registry: The full tool registry.
        allowed_names: Tool names to keep. None means "no restriction" — the
            input registry is returned unchanged. An empty list means "no
            tools at all" (PLAN mode).

    Returns:
        A filtered copy of the registry. Names in allowed_names that are not
        registered are skipped: extended tools such as grep and glob may
        legitimately be absent when only the base registry is loaded.
    """
    if allowed_names is None:
        return registry

    allowed = set(allowed_names)
    filtered = ToolRegistry()
    for name in registry.names():
        spec = registry.get(name)
        if spec is not None and spec.name in allowed:
            filtered.register(spec)
    return filtered

"""Tests for modes.py: the four core modes, prompt suffixes, config
application, and tool filtering."""

from __future__ import annotations

import pytest

from synth.agent import AgentConfig
from synth.constants import MAX_ITERATIONS
from synth.modes import (
    ACT_MAX_ITERATIONS,
    ACT_MODE,
    ARCHITECT_MODE,
    ARCHITECT_TOOLS,
    AUTO_MAX_ITERATIONS,
    AUTO_MODE,
    PLAN_MAX_ITERATIONS,
    PLAN_MODE,
    Mode,
    apply_mode_to_agent_config,
    build_mode_system_prompt,
    resolve_mode,
    tool_filter,
)
from synth.tools import ToolRegistry, ToolResult, ToolSpec

# Tools present in the fake registries used by these tests: the three base
# tools plus the two extended (read-only) ones ARCHITECT references.
TEST_TOOL_NAMES = ["bash", "glob", "grep", "read_file", "write_file"]


def _dummy_spec(name: str) -> ToolSpec:
    """Build a tool spec that never runs — only its name matters here."""

    def _func(args: dict) -> ToolResult:  # pragma: no cover - never called
        return ToolResult(f"dummy {name}")

    return ToolSpec(
        name=name,
        description=f"dummy {name}",
        parameters={"type": "object", "properties": {}},
        func=_func,
    )


def _make_registry(names: list[str] = TEST_TOOL_NAMES) -> ToolRegistry:
    """Build a small registry of dummy tools, independent of tools_ext."""
    registry = ToolRegistry()
    for name in names:
        registry.register(_dummy_spec(name))
    return registry


# --- resolve_mode ---


@pytest.mark.parametrize(
    "name,expected",
    [
        ("plan", PLAN_MODE),
        ("act", ACT_MODE),
        ("auto", AUTO_MODE),
        ("architect", ARCHITECT_MODE),
    ],
)
def test_known_modes(name: str, expected: ModeConfig) -> None:
    """Each of the four modes resolves to its preset."""
    assert resolve_mode(name) is expected


def test_case_insensitive_and_trimmed() -> None:
    """Mode names are normalised before lookup, so casing is cosmetic."""
    assert resolve_mode("PLAN") is resolve_mode("plan")
    assert resolve_mode("  Auto ") is AUTO_MODE


def test_unknown_mode_raises() -> None:
    """An unknown mode is rejected with the list of valid modes."""
    with pytest.raises(ValueError, match="unknown mode"):
        resolve_mode("ninja")


def test_non_string_mode_raises() -> None:
    """Anything that is not a string is rejected rather than coerced."""
    with pytest.raises(ValueError, match="must be a string"):
        resolve_mode(42)  # type: ignore[arg-type]


# --- build_mode_system_prompt ---


def test_suffix_appended_after_base() -> None:
    """The mode suffix is appended after the base prompt, which stays intact."""
    prompt = build_mode_system_prompt("You are Synth.", ACT_MODE)
    assert prompt.startswith("You are Synth.")
    assert ACT_MODE.system_prompt_suffix in prompt


def test_each_mode_suffix_present() -> None:
    """Every mode's prompt carries a distinctive marker of that mode."""
    for mode in Mode:
        config = resolve_mode(mode.value)
        prompt = build_mode_system_prompt("base", config)
        assert f"{mode.value.upper()} mode" in prompt


# --- apply_mode_to_agent_config ---


def test_apply_overrides_budget_and_stream() -> None:
    """AUTO raises the iteration budget above the caller's config."""
    base = AgentConfig(max_iterations=MAX_ITERATIONS, stream=False)
    applied = apply_mode_to_agent_config(base, AUTO_MODE)

    assert applied.max_iterations == AUTO_MAX_ITERATIONS
    assert applied.stream is True


def test_apply_does_not_mutate_original() -> None:
    """The caller's config is untouched; a new config is returned."""
    base = AgentConfig(max_iterations=5, stream=False)
    applied = apply_mode_to_agent_config(base, PLAN_MODE)

    assert base.max_iterations == 5
    assert base.stream is False
    assert applied.max_iterations == PLAN_MAX_ITERATIONS
    assert applied is not base


# --- tool_filter ---


def test_none_returns_same_registry() -> None:
    """allowed_names=None means "no restriction": the registry comes back as-is."""
    registry = _make_registry()
    assert tool_filter(registry, None) is registry


def test_restricts_to_allowed_names() -> None:
    """Only the requested tools survive; the rest are hidden."""
    filtered = tool_filter(_make_registry(), ["read_file", "glob"])

    assert filtered.names() == ["glob", "read_file"]
    assert filtered.get("bash") is None


def test_empty_list_removes_every_tool() -> None:
    """An empty allowed list means no tools — PLAN's policy."""
    assert tool_filter(_make_registry(), []).names() == []


def test_unregistered_names_are_skipped() -> None:
    """Names the registry doesn't have (e.g. unloaded extensions) are dropped."""
    filtered = tool_filter(_make_registry(), ["read_file", "grep", "nonexistent"])

    assert filtered.names() == ["grep", "read_file"]


# --- per-mode tool policies ---


def test_plan_mode_disallows_all_tools() -> None:
    """PLAN reasons only: empty allowed list, and filtering yields no tools."""
    assert PLAN_MODE.allowed_tool_names == []
    assert tool_filter(_make_registry(), PLAN_MODE.allowed_tool_names).names() == []


def test_act_mode_allows_all_tools() -> None:
    """ACT is the default work mode: no tool restriction."""
    registry = _make_registry()
    assert ACT_MODE.allowed_tool_names is None
    assert tool_filter(registry, ACT_MODE.allowed_tool_names).names() == registry.names()


def test_architect_mode_is_read_only() -> None:
    """ARCHITECT sees only read_file/glob/grep — never write_file or bash."""
    assert ARCHITECT_MODE.allowed_tool_names == ARCHITECT_TOOLS
    filtered = tool_filter(_make_registry(), ARCHITECT_MODE.allowed_tool_names)

    assert filtered.names() == ["glob", "grep", "read_file"]
    assert "write_file" not in filtered.names()
    assert "bash" not in filtered.names()


# --- end-to-end wiring of the four functions ---


def test_mode_pipeline_resolves_and_applies() -> None:
    """resolve_mode -> build -> apply compose into one configured run."""
    mode = resolve_mode("architect")
    prompt = build_mode_system_prompt("BASE", mode)
    config = apply_mode_to_agent_config(AgentConfig(), mode)

    assert "ARCHITECT mode" in prompt
    assert config.max_iterations == ARCHITECT_MODE.max_iterations
    assert tool_filter(_make_registry(), mode.allowed_tool_names).names() == [
        "glob",
        "grep",
        "read_file",
    ]

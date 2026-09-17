"""Tests for the token efficiency layer (spec section 8)."""

from __future__ import annotations

import pytest

from synth.token_efficiency import (
    AutoCompressor,
    Caveman,
    RtkFilter,
    TokenDashboard,
    apply_effort,
    ponytail_prompt,
)


# --- RTK: generic cleanup ---------------------------------------------------

class TestRtkGeneric:
    def test_strips_ansi_codes(self):
        r = RtkFilter()
        assert r.filter_output("x", "\x1b[31mred\x1b[0m text") == "red text"
        assert r.filter_output("x", "\x1b[31mred\x1b[0m text\n") == "red text\n"

    def test_collapses_repeated_lines(self):
        r = RtkFilter()
        out = r.filter_output("x", "a\na\na\nb\n")
        assert "[3 repeated]" in out
        assert "b" in out

    def test_keeps_error_lines_verbatim(self):
        r = RtkFilter()
        out = r.filter_output("x", "ok\nERROR: boom\nfailed at step 2\nok\n")
        assert "ERROR: boom" in out
        assert "failed at step 2" in out

    def test_blank_runs_collapsed(self):
        r = RtkFilter()
        out = r.filter_output("x", "a\n\n\n\nb\n")
        assert "\n\n\n" not in out

    def test_empty_output_passthrough(self):
        r = RtkFilter()
        assert r.filter_output("x", "") == ""
        assert r.filter_output("x", "   ") == "   "

    def test_max_length_enforced(self):
        r = RtkFilter(max_output_tokens=50)
        out = r.filter_output("x", "word " * 100)
        assert len(out) <= 50
        assert out.endswith("...")

    def test_unknown_command_passthrough_after_cleanup(self):
        r = RtkFilter()
        out = r.filter_output("weirdtool --flag", "\x1b[1mbold\x1b[0m data")
        assert out == "bold data"


# --- RTK: per-command filters -----------------------------------------------

class TestRtkCommands:
    def test_git_status_keeps_file_lines(self):
        r = RtkFilter()
        output = (
            "On branch main\n"
            "Your branch is up to date.\n"
            "\n"
            "Changes not staged for commit:\n"
            "  (use \"git add\" to update)\n"
            " M src/app.py\n"
            "?? new.py\n"
        )
        out = r.filter_output("git status", output)
        assert " M src/app.py" in out
        assert "?? new.py" in out
        assert "Your branch is up to date" not in out

    def test_git_status_keeps_errors(self):
        r = RtkFilter()
        out = r.filter_output("git status", "fatal error: not a repo\n")
        assert "fatal error" in out

    def test_git_log_keeps_subjects(self):
        r = RtkFilter()
        output = (
            "commit abc123\n"
            "Author: Bob <b@x>\n"
            "Date: Mon\n"
            "\n"
            "    fix the thing\n"
            "commit def456\n"
            "Author: Alice <a@x>\n"
            "Date: Tue\n"
            "\n"
            "    add feature\n"
        )
        out = r.filter_output("git log", output)
        assert "fix the thing" in out
        assert "add feature" in out
        assert "Author: Bob" not in out

    def test_pytest_drops_passing_dots(self):
        r = RtkFilter()
        output = ".................\nFAILED tests/test_x.py::test_y - assert 0\n1 failed\n"
        out = r.filter_output("pytest", output)
        assert "FAILED" in out
        assert "1 failed" in out

    def test_ls_collapses_long_listings(self):
        r = RtkFilter()
        rows = [f"-rw-r--r-- 1 user group 100 Jan  1 file{i}.py" for i in range(12)]
        out = r.filter_output("ls -la", "\n".join(rows))
        assert "rw-r--r--" not in out
        assert "file7.py" in out

    def test_ls_short_listing_untouched(self):
        r = RtkFilter()
        out = r.filter_output("ls", "-rw x.py\n-rw y.py\n")
        assert "x.py" in out

    def test_docker_ps_columns(self):
        r = RtkFilter()
        output = (
            "CONTAINER ID   IMAGE     STATUS          NAMES\n"
            "abc123         nginx     Up 2 minutes    web\n"
        )
        out = r.filter_output("docker ps", output)
        assert "CONTAINER ID" in out
        assert "abc123" in out

    def test_disabled_filter_passthrough(self):
        r = RtkFilter(enabled_filters=["pytest"])
        output = "On branch main\n M src/app.py\n"
        out = r.filter_output("git status", output)
        # git filter disabled -> generic cleanup only, hint survives.
        assert "On branch main" in out


# --- Ponytail ---------------------------------------------------------------

class TestPonytail:
    def test_ladder_has_seven_steps(self):
        out = ponytail_prompt("build api")
        for n in range(1, 8):
            assert f"{n}." in out

    def test_goal_included(self):
        assert "build api" in ponytail_prompt("build api")

    def test_aggressiveness_changes_wording(self):
        minimal = ponytail_prompt("g", aggressiveness="minimal")
        aggressive = ponytail_prompt("g", aggressiveness="aggressive")
        assert "Consider" in minimal
        assert "Demand" in aggressive

    def test_preserve_items_appear(self):
        out = ponytail_prompt("g", preserve=["security", "error_handling"])
        assert "security" in out.lower()
        assert "error handling" in out.lower()

    def test_existing_symbols_does_not_crash(self):
        out = ponytail_prompt("g", existing_symbols=["foo", "bar"])
        assert "decision ladder" in out

    def test_unknown_aggressiveness_raises(self):
        with pytest.raises(ValueError):
            ponytail_prompt("g", aggressiveness="turbo")


# --- Caveman ----------------------------------------------------------------

class TestCaveman:
    def test_strips_filler(self):
        c = Caveman()
        out = c.terse("I have analyzed the code. Let me fix it.")
        assert "I have" not in out
        assert "Let me" not in out
        assert "analyzed" in out

    def test_strips_pleasantry_lines(self):
        c = Caveman()
        out = c.terse("The fix is X.\nHope this helps!\n")
        assert "Hope this helps" not in out
        assert "fix is X" in out

    def test_replaces_utilize(self):
        c = Caveman()
        out = c.terse("We utilize the tool in order to work.")
        assert "use" in out
        assert "utilize" not in out
        assert "in order to" not in out

    def test_code_block_untouched(self):
        c = Caveman()
        text = "```py\n# I have basically analyzed this\nx = 1\n```\n"
        out = c.terse(text)
        assert "# I have basically analyzed this" in out

    def test_bullets_kept(self):
        c = Caveman()
        out = c.terse("- one thing\n- another thing\n")
        assert out.count("- ") == 2

    def test_idempotent(self):
        c = Caveman()
        once = c.terse("I will basically check the following items now.")
        assert c.terse(once) == once

    def test_empty_passthrough(self):
        c = Caveman()
        assert c.terse("") == ""


# --- AutoCompressor ----------------------------------------------------------

def big_msg(role: str, size: int = 20000) -> dict:
    return {"role": role, "content": "x" * size}


class TestAutoCompressor:
    def test_below_threshold_untouched(self):
        ac = AutoCompressor(context_window=200_000)
        msgs = [big_msg("user", 100)]
        out, changed = ac.compress(msgs)
        assert changed is False
        assert out == msgs

    def test_above_threshold_compresses(self):
        ac = AutoCompressor(threshold=0.01, keep_recent=2, context_window=10_000)
        msgs = [big_msg("user") for _ in range(10)]
        out, changed = ac.compress(msgs)
        assert changed is True
        assert len(out) < len(msgs)

    def test_system_message_survives(self):
        ac = AutoCompressor(threshold=0.01, keep_recent=2, context_window=10_000)
        msgs = [{"role": "system", "content": "be nice"}] + [big_msg("user") for _ in range(10)]
        out, _ = ac.compress(msgs)
        assert any(m["role"] == "system" and m["content"] == "be nice" for m in out)

    def test_tool_messages_survive(self):
        ac = AutoCompressor(threshold=0.01, keep_recent=2, context_window=10_000)
        msgs = [
            {"role": "tool", "content": "tool result", "tool_call_id": "t1"},
            {"role": "assistant", "content": "x" * 20000},
        ] + [big_msg("user") for _ in range(10)]
        out, _ = ac.compress(msgs)
        assert any(m.get("tool_call_id") == "t1" for m in out)

    def test_summary_marker_present(self):
        ac = AutoCompressor(threshold=0.01, keep_recent=2, context_window=10_000)
        msgs = [{"role": "user", "content": ("word " * 5000)}, big_msg("user"), big_msg("user"), big_msg("user")]
        out, changed = ac.compress(msgs)
        assert changed
        assert any("[compressed]" in m.get("content", "") for m in out)

    def test_empty_history(self):
        ac = AutoCompressor()
        out, changed = ac.compress([])
        assert out == []
        assert changed is False


# --- TokenDashboard -----------------------------------------------------------

class TestTokenDashboard:
    def test_accumulates(self):
        d = TokenDashboard()
        d.record("gpt-4", 100, 50)
        d.record("gpt-4", 200, 100)
        s = d.stats()
        assert s["total_calls"] == 2
        assert s["prompt_chars"] == 300
        assert s["completion_chars"] == 150

    def test_est_tokens_is_quarter(self):
        d = TokenDashboard()
        d.record("m", 400, 400)
        assert d.stats()["est_tokens"] == 200

    def test_by_model_split(self):
        d = TokenDashboard()
        d.record("a", 10, 10)
        d.record("b", 20, 20)
        s = d.stats()
        assert s["by_model"]["a"]["calls"] == 1
        assert s["by_model"]["b"]["prompt_chars"] == 20

    def test_stats_returns_copy(self):
        d = TokenDashboard()
        d.record("a", 1, 1)
        s = d.stats()
        s["by_model"]["a"]["calls"] = 999
        assert d.stats()["by_model"]["a"]["calls"] == 1


# --- Effort levels -------------------------------------------------------------

class TestEffort:
    def test_levels(self):
        assert apply_effort(20, "low") == 10
        assert apply_effort(20, "medium") == 20
        assert apply_effort(20, "high") == 30
        assert apply_effort(20, "xhigh") == 40

    def test_minimum_one(self):
        assert apply_effort(1, "low") == 1

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            apply_effort(20, "ultra")

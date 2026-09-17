"""Tests for best-of-N selection (spec 6.1 #7)."""

from __future__ import annotations

import pytest

from synth.best_of_n import BestOfN, Candidate, score_answer


class FakeResponse:
    def __init__(self, text: str | None) -> None:
        self.text = text


class FakeClient:
    def __init__(self, text: str | None = None, error: Exception | None = None) -> None:
        self._text = text
        self._error = error

    def chat(self, messages, tools=None, stream=False):
        if self._error:
            raise self._error
        return FakeResponse(self._text)


# --- score_answer -----------------------------------------------------------

def test_score_empty_is_zero():
    assert score_answer("") == 0.0
    assert score_answer("   ") == 0.0


def test_score_plain_text_positive():
    assert 0 < score_answer("Some reasonable answer here.") <= 1.0


def test_score_prefers_code_block():
    with_code = score_answer("Here:\n```py\nprint(1)\n```\n")
    without = score_answer("Here is a plain paragraph with similar length.")
    assert with_code > without


def test_score_prefers_structure():
    structured = score_answer("- one\n- two\n- three\n- four\n- five\n")
    flat = score_answer("one item with words")
    assert structured > flat


def test_score_penalizes_hedging():
    confident = score_answer("The fix is to add an index on user_id.")
    hedgy = score_answer("I'm not sure, perhaps maybe it might be an index.")
    assert confident > hedgy


def test_score_is_capped_for_rambles():
    short = score_answer("x" * 5000)
    long_ = score_answer("x" * 50000)
    # Same length credit once past LENGTH_FULL_MARKS.
    assert short == long_


# --- BestOfN ----------------------------------------------------------------

def test_best_of_n_requires_clients():
    with pytest.raises(ValueError):
        BestOfN({})


def test_best_of_n_picks_highest_score():
    boa = BestOfN({
        "weak": FakeClient("not sure, maybe"),
        "strong": FakeClient("Plan:\n- step 1\n- step 2\n```sh\nls\n```"),
    })
    result = boa.run("do a thing")
    assert result.best.model == "strong"
    assert [c.model for c in result.candidates] == ["strong", "weak"]


def test_best_of_n_records_empty_answer_as_zero():
    boa = BestOfN({"empty": FakeClient(""), "ok": FakeClient("real answer")})
    result = boa.run("x")
    empty = next(c for c in result.candidates if c.model == "empty")
    assert empty.score == 0.0
    assert result.best.model == "ok"


def test_best_of_n_all_empty_raises():
    boa = BestOfN({"m1": FakeClient("")})
    with pytest.raises(RuntimeError, match="all models failed"):
        boa.run("x")


def test_best_of_n_all_failures_raise():
    boa = BestOfN({
        "a": FakeClient(error=RuntimeError("boom")),
        "b": FakeClient(error=RuntimeError("boom")),
    })
    with pytest.raises(RuntimeError, match="all models failed"):
        boa.run("x")


def test_best_of_n_partial_failure_uses_survivor():
    boa = BestOfN({
        "dead": FakeClient(error=RuntimeError("boom")),
        "alive": FakeClient("survivor answer"),
    })
    result = boa.run("x")
    assert result.best.model == "alive"
    dead = next(c for c in result.candidates if c.model == "dead")
    assert dead.score == 0.0


def test_candidate_dataclass_defaults():
    c = Candidate(model="m", text="t")
    assert c.score == 0.0

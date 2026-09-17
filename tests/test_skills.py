"""Tests for the skills engine (src/synth/skills.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from synth.skills import (
    Skill,
    SkillError,
    SkillExecutor,
    SkillLoader,
    SkillRefiner,
    SkillStore,
)


@pytest.fixture
def store(tmp_path: Path) -> SkillStore:
    return SkillStore(tmp_path / "skills.db")


def make_skill(name: str = "hello", code: str = 'result = "hi"') -> Skill:
    return Skill(name=name, description="a test skill", code=code)


# --- store CRUD -------------------------------------------------------------

def test_save_and_load_round_trip(store: SkillStore):
    store.save_skill(make_skill())
    got = store.load_skill("hello")
    assert got is not None
    assert got.name == "hello"
    assert got.code == 'result = "hi"'


def test_load_missing_returns_none(store: SkillStore):
    assert store.load_skill("nope") is None


def test_list_skills(store: SkillStore):
    store.save_skill(make_skill("alpha"))
    store.save_skill(make_skill("beta"))
    names = [s.name for s in store.list_skills()]
    assert names == ["alpha", "beta"]


def test_delete_skill(store: SkillStore):
    store.save_skill(make_skill())
    assert store.delete_skill("hello") is True
    assert store.delete_skill("hello") is False


def test_search_skills(store: SkillStore):
    store.save_skill(make_skill("code-review", code="x=1"))
    store.save_skill(make_skill("deploy-bot", code="y=2"))
    hits = store.search_skills("review")
    assert [s.name for s in hits] == ["code-review"]


def test_search_empty_query_raises(store: SkillStore):
    with pytest.raises(SkillError):
        store.search_skills("  ")


def test_save_rejects_bad_name(store: SkillStore):
    with pytest.raises(SkillError):
        store.save_skill(Skill(name="Bad Name!!", description="d", code="x=1"))


def test_save_rejects_empty_code(store: SkillStore):
    with pytest.raises(SkillError):
        store.save_skill(Skill(name="ok", description="d", code="   "))


def test_save_upserts(store: SkillStore):
    store.save_skill(make_skill(code="a=1"))
    store.save_skill(make_skill(code="a=2"))
    assert store.load_skill("hello").code == "a=2"
    assert len(store.list_skills()) == 1


def test_context_manager(tmp_path: Path):
    with SkillStore(tmp_path / "s.db") as store:
        store.save_skill(make_skill())
    # Reopen: persisted.
    with SkillStore(tmp_path / "s.db") as store2:
        assert store2.load_skill("hello") is not None


# --- executor ---------------------------------------------------------------

def test_validate_ok():
    assert SkillExecutor().validate_code("x = 1 + 1\nresult = x") is True


def test_validate_blocks_import():
    assert SkillExecutor().validate_code("import os\nresult = os") is False


def test_validate_blocks_eval():
    assert SkillExecutor().validate_code("result = eval('1')") is False


def test_validate_blocks_syntax_error():
    assert SkillExecutor().validate_code("def (") is False


def test_execute_returns_result(store: SkillStore):
    ex = SkillExecutor(store)
    result = ex.execute(make_skill(code='result = 6 * 7'), {})
    assert not result.is_error
    assert "42" in result.text


def test_execute_captures_print(store: SkillStore):
    result = SkillExecutor().execute(make_skill(code='print("out")\nresult = 1'), {})
    assert "out" in result.text
    assert "1" in result.text


def test_execute_uses_context(store: SkillStore):
    result = SkillExecutor().execute(
        make_skill(code="result = context['n'] + 1"), {"n": 41}
    )
    assert "42" in result.text


def test_execute_catches_runtime_error():
    result = SkillExecutor().execute(make_skill(code="result = 1/0"), {})
    assert result.is_error
    assert "ZeroDivisionError" in result.text


def test_execute_rejects_invalid_code():
    result = SkillExecutor().execute(make_skill(code="import os"), {})
    assert result.is_error
    assert "validation" in result.text


def test_usage_count_increments(store: SkillStore):
    store.save_skill(make_skill())
    SkillExecutor(store).execute(store.load_skill("hello"), {})
    assert store.load_skill("hello").usage_count == 1


# --- loader -------------------------------------------------------------------

def test_from_github_fetches_raw_file(store: SkillStore, monkeypatch):
    class FakeResp:
        status = 200

        def read(self):
            return b"result = 1\n"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    seen: dict = {}

    def fake_urlopen(url, timeout=None):
        seen["url"] = str(url)
        return FakeResp()

    monkeypatch.setattr("synth.skills.urllib.request.urlopen", fake_urlopen)
    skill = SkillLoader().from_github("Mercphobia/Synth", "skills/foo.py")
    assert skill.name == "foo"
    assert skill.code == "result = 1\n"
    assert seen["url"].startswith("https://raw.githubusercontent.com/")


def test_from_github_rejects_bad_repo():
    with pytest.raises(SkillError):
        SkillLoader().from_github("noslug", "a/b.py")


def test_from_playbook_steps():
    yaml = (
        "name: greet\n"
        "description: write and read a file\n"
        "steps:\n"
        "  - {action: write_file, args: {path: a.txt, content: hi}}\n"
        "  - {action: read_file, args: {path: a.txt}}\n"
    )
    skill = SkillLoader().from_playbook(yaml)
    assert skill.name == "greet"
    assert "write_file" in skill.code
    assert "read_file" in skill.code


def test_from_playbook_rejects_empty():
    with pytest.raises(SkillError):
        SkillLoader().from_playbook("")


def test_from_playbook_requires_steps():
    with pytest.raises(SkillError):
        SkillLoader().from_playbook("name: x\ndescription: y\n")


# --- refiner --------------------------------------------------------------------

class FakeLLM:
    def __init__(self, text: str) -> None:
        self._text = text

    def chat(self, messages, tools=None, stream=False):
        class R:
            pass
        r = R()
        r.text = self._text
        return r


def test_refiner_returns_new_code():
    skill = make_skill(code="result = 1")
    refined = SkillRefiner(FakeLLM("```python\nresult = 2\n```")).refine(skill, "double it")
    assert "result = 2" in refined.code
    assert refined.name == "hello"


def test_refiner_rejects_unsafe_output():
    skill = make_skill(code="result = 1")
    with pytest.raises(SkillError):
        SkillRefiner(FakeLLM("import os")).refine(skill, "do bad")


def test_refiner_requires_feedback():
    with pytest.raises(SkillError):
        SkillRefiner(FakeLLM("x=1")).refine(make_skill(), "   ")

"""Tests for synth.advanced (v2.0 advanced features)."""

from __future__ import annotations

from synth.best_of_n import Candidate
from synth.session import Message, SessionStore
from synth.advanced import (
    COMPRESS_DEFAULT_RATIO,
    DEDUP_THRESHOLD,
    GRAPH_MAX_DEPTH,
    AgentCanvas,
    BrowserAutomation,
    CodeGraph,
    CommandPalette,
    CrossSessionMessaging,
    GPUAcceleration,
    HandoffProtocol,
    ImageInput,
    MemoryGraph,
    MultiBackend,
    Notebook,
    PDFExcelParser,
    PromptCompressor,
    PRCreation,
    Provenance,
    SemanticDedup,
    SessionBranch,
    TokenSaviour,
    VoiceInput,
    VoiceOutput,
    VotingSystem,
    WebDashboard,
)


# === Constants =============================================================

def test_constants_exist():
    assert GRAPH_MAX_DEPTH == 3
    assert DEDUP_THRESHOLD == 0.8
    assert COMPRESS_DEFAULT_RATIO == 0.5


# === MemoryGraph ===========================================================

def test_memory_graph_add_node():
    g = MemoryGraph()
    g.add_node("a", "concept", "alpha")
    assert "a" in g.nodes
    assert g.nodes["a"]["type"] == "concept"
    assert g.nodes["a"]["content"] == "alpha"
    assert g.nodes["a"]["links"] == []


def test_memory_graph_link():
    g = MemoryGraph()
    g.add_node("a", "concept", "alpha")
    g.add_node("b", "concept", "beta")
    g.link("a", "b", "related")
    assert len(g.nodes["a"]["links"]) == 1
    assert g.nodes["a"]["links"][0]["target"] == "b"
    assert g.nodes["a"]["links"][0]["label"] == "related"


def test_memory_graph_link_unknown_source():
    import pytest
    g = MemoryGraph()
    g.add_node("b", "c", "beta")
    with pytest.raises(KeyError):
        g.link("a", "b")


def test_memory_graph_link_unknown_target():
    import pytest
    g = MemoryGraph()
    g.add_node("a", "c", "alpha")
    with pytest.raises(KeyError):
        g.link("a", "b")


def test_memory_graph_traverse_bfs():
    g = MemoryGraph()
    for nid in ("a", "b", "c", "d"):
        g.add_node(nid, "node", nid)
    g.link("a", "b")
    g.link("b", "c")
    g.link("c", "d")
    # depth 1: start only; depth 2: start + neighbors; depth 3: + 2-hop
    assert g.traverse("a", max_depth=1) == ["a"]
    assert g.traverse("a", max_depth=2) == ["a", "b"]
    assert g.traverse("a", max_depth=3) == ["a", "b", "c"]
    # default depth (GRAPH_MAX_DEPTH=3) should not reach d
    assert "d" not in g.traverse("a")


def test_memory_graph_traverse_unknown_start():
    g = MemoryGraph()
    assert g.traverse("missing") == []


def test_memory_graph_traverse_cycle_safe():
    g = MemoryGraph()
    g.add_node("a", "n", "a")
    g.add_node("b", "n", "b")
    g.link("a", "b")
    g.link("b", "a")  # cycle
    result = g.traverse("a", max_depth=5)
    assert result == ["a", "b"]


def test_memory_graph_to_dot():
    g = MemoryGraph()
    g.add_node("a", "concept", "alpha")
    g.add_node("b", "concept", "beta")
    g.link("a", "b", "related")
    dot = g.to_dot()
    assert dot.startswith("digraph memory {")
    assert '"a"' in dot
    assert '"b"' in dot
    assert "->" in dot
    assert "label=\"related\"" in dot
    assert dot.rstrip().endswith("}")


def test_memory_graph_save_load(tmp_path):
    g = MemoryGraph()
    g.add_node("a", "concept", "alpha")
    g.add_node("b", "concept", "beta")
    g.link("a", "b", "rel")
    db = tmp_path / "graph.db"
    g.save(db)
    g2 = MemoryGraph()
    g2.load(db)
    assert set(g2.nodes.keys()) == {"a", "b"}
    assert g2.nodes["a"]["content"] == "alpha"
    assert g2.nodes["a"]["links"][0]["target"] == "b"
    assert g2.nodes["a"]["links"][0]["label"] == "rel"


# === VotingSystem ==========================================================

def _cand(text: str, score: float = 0.5, model: str = "m") -> Candidate:
    return Candidate(model=model, text=text, score=score)


def test_voting_majority_prefix():
    vs = VotingSystem()
    cands = [
        _cand("The answer is 42 because physics.", 0.6),
        _cand("The answer is 42 because physics.", 0.4),
        _cand("The answer is 42 because physics.", 0.5),
    ]
    winner = vs.vote(cands)
    assert winner.text.startswith("The answer is 42")


def test_voting_no_majority_picks_highest_score():
    vs = VotingSystem()
    cands = [
        _cand("Completely different answer one.", 0.3),
        _cand("Totally unrelated answer two.", 0.8),
        _cand("Yet another distinct answer three.", 0.5),
    ]
    winner = vs.vote(cands)
    assert winner.score == 0.8


def test_voting_requires_three():
    import pytest
    vs = VotingSystem()
    with pytest.raises(ValueError):
        vs.vote([_cand("a"), _cand("b")])


# === HandoffProtocol =======================================================

def test_handoff_round_trip():
    hp = HandoffProtocol()
    hid = hp.create_handoff("agent-1", {"task": "summarize", "data": [1, 2]})
    result = hp.receive_handoff(hid)
    assert result["agent_name"] == "agent-1"
    assert result["context"]["task"] == "summarize"
    assert result["context"]["data"] == [1, 2]


def test_handoff_unknown_id():
    import pytest
    hp = HandoffProtocol()
    with pytest.raises(KeyError):
        hp.receive_handoff("nope")


def test_handoff_multiple():
    hp = HandoffProtocol()
    h1 = hp.create_handoff("a", "ctx-a")
    h2 = hp.create_handoff("b", "ctx-b")
    assert hp.receive_handoff(h1)["context"] == "ctx-a"
    assert hp.receive_handoff(h2)["context"] == "ctx-b"


# === CrossSessionMessaging =================================================

def test_cross_session_send_receive():
    csm = CrossSessionMessaging()
    csm.send("sess-1", "sess-2", "hello")
    csm.send("sess-1", "sess-2", "world")
    msgs = csm.receive("sess-2")
    assert len(msgs) == 2
    assert msgs[0]["message"] == "hello"
    assert msgs[1]["message"] == "world"
    assert msgs[0]["from"] == "sess-1"


def test_cross_session_receive_empty():
    csm = CrossSessionMessaging()
    assert csm.receive("no-messages") == []


def test_cross_session_receive_clears_queue():
    csm = CrossSessionMessaging()
    csm.send("a", "b", "msg")
    first = csm.receive("b")
    second = csm.receive("b")
    assert len(first) == 1
    assert second == []


# === SessionBranch =========================================================

def test_session_branch(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    sid = store.create_session(title="orig")
    store.append_message(sid, Message(role="user", content="hello"))
    store.append_message(sid, Message(role="assistant", content="hi"))

    sb = SessionBranch(store)
    bid = sb.branch(sid, "branch-1")
    msgs = store.load_session(bid)
    assert len(msgs) == 2
    assert msgs[0].content == "hello"
    assert msgs[1].content == "hi"
    store.close()


def test_session_merge(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    target = store.create_session(title="target")
    source = store.create_session(title="source")
    store.append_message(source, Message(role="user", content="src-msg"))

    sb = SessionBranch(store)
    sb.merge(target, source)
    msgs = store.load_session(target)
    assert len(msgs) == 1
    assert msgs[0].content == "src-msg"
    store.close()


def test_session_compare(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    a = store.create_session(title="a")
    b = store.create_session(title="b")
    store.append_message(a, Message(role="user", content="common"))
    store.append_message(a, Message(role="user", content="only-in-a"))
    store.append_message(b, Message(role="user", content="common"))
    store.append_message(b, Message(role="user", content="only-in-b"))

    sb = SessionBranch(store)
    result = sb.compare(a, b)
    assert "common" in result["common"][0]
    assert len(result["only_a"]) == 1
    assert len(result["only_b"]) == 1
    assert result["only_a"][0] == "only-in-a"
    assert result["only_b"][0] == "only-in-b"
    store.close()


# === CodeGraph ============================================================

def test_code_graph_extracts_from_py_files(tmp_path):
    (tmp_path / "mod.py").write_text(
        "import os\n"
        "from typing import List\n"
        "\n"
        "class Foo:\n"
        "    def bar(self):\n"
        "        pass\n"
        "\n"
        "def baz():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "other.py").write_text(
        "import json\n"
        "\n"
        "class Bar:\n"
        "    pass\n",
        encoding="utf-8",
    )
    # Non-py file should be ignored
    (tmp_path / "readme.md").write_text("not python", encoding="utf-8")

    cg = CodeGraph()
    graph = cg.build_graph(tmp_path)
    assert "mod.py" in graph["files"]
    assert "sub/other.py" in graph["files"]
    assert "readme.md" not in graph["files"]
    assert "os" in graph["imports"]["mod.py"]
    assert "typing" in graph["imports"]["mod.py"]
    assert "Foo" in graph["classes"]["mod.py"]
    assert "baz" in graph["functions"]["mod.py"]
    assert "Bar" in graph["classes"]["sub/other.py"]
    assert "json" in graph["imports"]["sub/other.py"]


def test_code_graph_skips_venv(tmp_path):
    (tmp_path / "real.py").write_text("def f(): pass\n", encoding="utf-8")
    venv_dir = tmp_path / ".venv"
    venv_dir.mkdir()
    (venv_dir / "vendor.py").write_text(
        "def vendor(): pass\n", encoding="utf-8"
    )
    cg = CodeGraph()
    graph = cg.build_graph(tmp_path)
    assert "real.py" in graph["files"]
    assert ".venv/vendor.py" not in graph["files"]


def test_code_graph_handles_syntax_error(tmp_path):
    (tmp_path / "bad.py").write_text("def (:\n", encoding="utf-8")
    (tmp_path / "good.py").write_text("def ok(): pass\n", encoding="utf-8")
    cg = CodeGraph()
    graph = cg.build_graph(tmp_path)
    assert "good.py" in graph["files"]
    # bad.py is skipped but good.py still processed
    assert "ok" in graph["functions"]["good.py"]


def test_code_graph_empty_dir(tmp_path):
    cg = CodeGraph()
    graph = cg.build_graph(tmp_path)
    assert graph["files"] == []
    assert graph["imports"] == {}


# === CommandPalette =======================================================

def test_command_palette_register_search_execute():
    cp = CommandPalette()
    results = []

    cp.register("run-tests", "Execute the test suite", results.append)
    matches = cp.search("test")
    assert ("run-tests", "Execute the test suite") in matches

    cp.execute("run-tests", "done")
    assert results == ["done"]


def test_command_palette_search_no_match():
    cp = CommandPalette()
    cp.register("run-tests", "Execute tests", lambda: None)
    assert cp.search("nonexistent") == []


def test_command_palette_search_empty_query_returns_all():
    cp = CommandPalette()
    cp.register("cmd-a", "desc a", lambda: None)
    cp.register("cmd-b", "desc b", lambda: None)
    assert len(cp.search("")) == 2


def test_command_palette_execute_unknown():
    import pytest
    cp = CommandPalette()
    with pytest.raises(KeyError):
        cp.execute("nope")


def test_command_palette_fuzzy_multi_word():
    cp = CommandPalette()
    cp.register("git-commit", "Create a new git commit", lambda: None)
    cp.register("git-push", "Push commits to remote", lambda: None)
    matches = cp.search("git commit")
    assert ("git-commit", "Create a new git commit") in matches
    # git-push should rank lower (no 'commit' in name/desc) but may still match
    # on the shared 'git' prefix — verify it's ranked after git-commit.
    assert matches[0] == ("git-commit", "Create a new git commit")


# === TokenSaviour ==========================================================

def test_token_saviour_picks_grep_over_read():
    ts = TokenSaviour()
    tools = [
        {"name": "read_file"},
        {"name": "grep"},
        {"name": "bash"},
    ]
    assert ts.pick_cheapest_tool("find the function", tools) == "grep"


def test_token_saviour_picks_bash_for_run():
    ts = TokenSaviour()
    tools = [
        {"name": "read_file"},
        {"name": "bash"},
        {"name": "grep"},
    ]
    assert ts.pick_cheapest_tool("run the tests", tools) == "bash"


def test_token_saviour_default_first():
    ts = TokenSaviour()
    tools = [
        {"name": "summarize"},
        {"name": "translate"},
    ]
    assert ts.pick_cheapest_tool("translate this", tools) == "summarize"


def test_token_saviour_empty_tools():
    ts = TokenSaviour()
    assert ts.pick_cheapest_tool("do something", []) == ""


# === SemanticDedup ========================================================

def test_semantic_dedup_removes_near_duplicates():
    sd = SemanticDedup()
    msgs = [
        {"content": "the quick brown fox jumps over the lazy dog"},
        {"content": "the quick brown fox jumps over the lazy dog"},
        {"content": "completely different sentence about apples"},
    ]
    result = sd.dedup(msgs)
    assert len(result) == 2
    contents = [m["content"] for m in result]
    assert "completely different sentence about apples" in contents


def test_semantic_dedup_keeps_longer():
    sd = SemanticDedup()
    short = {"content": "the quick brown fox jumps over the lazy dog"}
    long = {
        "content": "the quick brown fox jumps over the lazy dog "
        "the end"
    }
    # 8 common words, long has 10 total → Jaccard = 8/10 = 0.8 (threshold)
    result = sd.dedup([short, long])
    assert len(result) == 1
    assert "the end" in result[0]["content"]


def test_semantic_dedup_keeps_dissimilar():
    sd = SemanticDedup()
    msgs = [
        {"content": "alpha beta gamma"},
        {"content": "delta epsilon zeta eta theta"},
    ]
    result = sd.dedup(msgs)
    assert len(result) == 2


def test_semantic_dedup_empty_list():
    sd = SemanticDedup()
    assert sd.dedup([]) == []


# === PromptCompressor =====================================================

def test_prompt_compressor_reduces_length():
    pc = PromptCompressor()
    text = (
        "The system processes data efficiently. "
        "Data processing is the core function. "
        "Efficiency matters for performance. "
        "Performance impacts user experience. "
        "User experience drives retention."
    )
    compressed = pc.compress(text, ratio=0.3)
    assert len(compressed) <= len(text)
    assert len(compressed) > 0


def test_prompt_compressor_preserves_key_sentences():
    pc = PromptCompressor()
    text = (
        "The system uses a database for storage. "
        "The database stores all records. "
        "Storage is important for persistence. "
        "Records contain user data. "
        "Persistence ensures data survives restarts."
    )
    compressed = pc.compress(text, ratio=0.4)
    # compressed text should contain at least one full sentence from original
    assert any(
        sent in text for sent in compressed.split(". ")
        if sent.strip()
    )


def test_prompt_compressor_short_text_returns_as_is():
    pc = PromptCompressor()
    assert pc.compress("Short.") == "Short."


def test_prompt_compressor_empty_text():
    pc = PromptCompressor()
    assert pc.compress("") == ""


def test_prompt_compressor_default_ratio():
    pc = PromptCompressor()
    text = ". ".join(
        f"Sentence number {i} about topic {i}" for i in range(10)
    )
    compressed = pc.compress(text)
    assert len(compressed) <= len(text)


# === Stubs ================================================================

def test_agent_canvas_stub():
    import pytest
    with pytest.raises(NotImplementedError):
        AgentCanvas()


def test_browser_automation_stub():
    import pytest
    with pytest.raises(NotImplementedError):
        BrowserAutomation()


def test_image_input_stub():
    import pytest
    with pytest.raises(NotImplementedError):
        ImageInput()


def test_pdf_excel_parser_stub():
    import pytest
    with pytest.raises(NotImplementedError):
        PDFExcelParser()


def test_voice_input_stub():
    import pytest
    with pytest.raises(NotImplementedError):
        VoiceInput()


def test_voice_output_stub():
    import pytest
    with pytest.raises(NotImplementedError):
        VoiceOutput()


def test_web_dashboard_stub():
    import pytest
    with pytest.raises(NotImplementedError):
        WebDashboard()


def test_multi_backend_stub():
    import pytest
    with pytest.raises(NotImplementedError):
        MultiBackend()


def test_pr_creation_stub():
    import pytest
    with pytest.raises(NotImplementedError):
        PRCreation()


def test_provenance_stub():
    import pytest
    with pytest.raises(NotImplementedError):
        Provenance()


def test_notebook_stub():
    import pytest
    with pytest.raises(NotImplementedError):
        Notebook()


def test_gpu_acceleration_stub():
    import pytest
    with pytest.raises(NotImplementedError):
        GPUAcceleration()

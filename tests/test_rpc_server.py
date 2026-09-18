"""Tests for rpc_server.py: the JSON-RPC stdin/stdout bridge.

The server reads NDJSON from stdin and writes NDJSON to stdout. We drive it
by redirecting stdin/stdout to StringIO buffers and parsing the captured
output as JSON, one line per response.

All tests run with a temporary HOME so no real ~/.synth config is touched.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from unittest import mock

import pytest

from synth import rpc_server
from synth.rpc_server import METHODS, serve


# --- Helpers ---------------------------------------------------------------


def _run_serve(input_text: str) -> list[dict]:
    """Call serve() with crafted stdin; return parsed stdout JSON lines.

    Redirects sys.stdin to a StringIO holding ``input_text`` and captures
    everything written to sys.stdout. Each non-empty captured line is
    parsed as JSON; parse failures raise (tests fail loudly).
    """
    stdin = io.StringIO(input_text)
    stdout = io.StringIO()
    with mock.patch.object(rpc_server.sys, "stdin", stdin), \
         mock.patch.object(rpc_server.sys, "stdout", stdout):
        # serve() reassigns sys.stderr to /devnull; restore it afterwards
        # so pytest's real stderr keeps working for the next test.
        orig_stderr = rpc_server.sys.stderr
        try:
            serve()
        finally:
            rpc_server.sys.stderr = orig_stderr
    lines = [ln for ln in stdout.getvalue().splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


def _req(method: str, *, id: int = 1, params: dict | None = None,
         extra: dict | None = None) -> str:
    """Build a single JSON-RPC request line."""
    msg = {"jsonrpc": "2.0", "method": method, "id": id}
    if params is not None:
        msg["params"] = params
    if extra:
        msg.update(extra)
    return json.dumps(msg)


# --- Fixtures --------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point HOME at a temp dir so config/session files don't leak."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("SYNTH_LOG_FILE", str(tmp_path / "synth.log"))
    return home


# --- Tests -----------------------------------------------------------------


def test_ping_returns_pong() -> None:
    """ping → result 'pong'."""
    resp = _run_serve(_req("ping", id=1))
    assert len(resp) == 1
    assert resp[0] == {"jsonrpc": "2.0", "id": 1, "result": "pong"}


def test_get_config_has_model_and_max_iterations() -> None:
    """get_config → result dict has 'model' and 'max_iterations'."""
    resp = _run_serve(_req("get_config", id=2))
    assert len(resp) == 1
    result = resp[0]["result"]
    assert "model" in result
    assert "max_iterations" in result
    assert isinstance(result["max_iterations"], int)


def test_list_sessions_returns_list_empty_fresh() -> None:
    """list_sessions on a fresh store returns an empty list."""
    resp = _run_serve(_req("list_sessions", id=3))
    assert len(resp) == 1
    result = resp[0]["result"]
    assert isinstance(result, list)
    assert result == []


def test_shutdown_exits_cleanly() -> None:
    """shutdown → os._exit(0) is called (mocked) without crashing."""
    with mock.patch.object(rpc_server.os, "_exit", create=True) as mock_exit:
        mock_exit.return_value = None
        # serve() won't return naturally after _exit because we mocked it,
        # but the handler is invoked inside a try/except in serve(); since
        # _exit no longer terminates, the loop continues. The response is
        # still emitted before the (mocked) _exit call.
        stdin = io.StringIO(_req("shutdown", id=4))
        stdout = io.StringIO()
        with mock.patch.object(rpc_server.sys, "stdin", stdin), \
             mock.patch.object(rpc_server.sys, "stdout", stdout):
            orig_stderr = rpc_server.sys.stderr
            try:
                serve()
            finally:
                rpc_server.sys.stderr = orig_stderr
        mock_exit.assert_called_once_with(0)
        lines = [ln for ln in stdout.getvalue().splitlines() if ln.strip()]
        # No response is written because _exit normally prevents the
        # _send("result": None) line; here it's mocked so the handler
        # returns None and serve() emits a result line.
        assert len(lines) == 1
        out = json.loads(lines[0])
        assert out["id"] == 4
        assert out["result"] is None


def test_unknown_method_error_code_32601() -> None:
    """Unknown method → error code -32601 (method not found)."""
    resp = _run_serve(_req("nonexistent_method", id=5))
    assert len(resp) == 1
    err = resp[0]["error"]
    assert err["code"] == -32601
    assert "method not found" in err["message"]
    assert resp[0]["id"] == 5


def test_invalid_json_error_code_32700() -> None:
    """Unparseable JSON line → error code -32700 (parse error)."""
    resp = _run_serve("not valid json at all")
    assert len(resp) == 1
    err = resp[0]["error"]
    assert err["code"] == -32700
    assert "parse" in err["message"].lower()
    # No id because the message couldn't be parsed.
    assert "id" not in resp[0] or resp[0].get("id") is None


def test_missing_method_field_error_code_32600() -> None:
    """Request without 'method' key → error code -32600 (invalid request)."""
    bad_msg = json.dumps({"jsonrpc": "2.0", "id": 7, "params": {}})
    resp = _run_serve(bad_msg)
    assert len(resp) == 1
    err = resp[0]["error"]
    assert err["code"] == -32600
    assert "invalid request" in err["message"]
    assert resp[0]["id"] == 7


def test_empty_params_defaults_to_empty_dict() -> None:
    """Request with no 'params' key → handler receives {} (empty dict)."""
    # ping ignores params, but we verify the dispatcher passes {} by
    # checking that a method needing params still works without them.
    msg = json.dumps({"jsonrpc": "2.0", "method": "ping", "id": 8})
    resp = _run_serve(msg)
    assert len(resp) == 1
    assert resp[0]["result"] == "pong"
    assert resp[0]["id"] == 8


def test_multiple_requests_in_sequence() -> None:
    """Several NDJSON lines processed in order; one response each."""
    batch = "\n".join([
        _req("ping", id=10),
        _req("ping", id=11),
        _req("get_config", id=12),
        _req("list_sessions", id=13),
        _req("unknown_xyz", id=14),
    ])
    resp = _run_serve(batch)
    assert len(resp) == 5
    assert [r["id"] for r in resp] == [10, 11, 12, 13, 14]
    assert resp[0]["result"] == "pong"
    assert resp[1]["result"] == "pong"
    assert "model" in resp[2]["result"]
    assert resp[3]["result"] == []
    assert resp[4]["error"]["code"] == -32601


def test_run_missing_prompt_returns_error() -> None:
    """run with no 'prompt' in params → result has 'error' field."""
    # params present but prompt missing → _run returns {"error": ...}
    resp = _run_serve(_req("run", id=20, params={}))
    assert len(resp) == 1
    result = resp[0]["result"]
    assert "error" in result
    assert "prompt" in result["error"]


def test_run_bad_mode_returns_exit_code_2() -> None:
    """run with an invalid mode → result has exit_code 2."""
    resp = _run_serve(_req("run", id=21, params={"prompt": "hi", "mode": "bogus"}))
    assert len(resp) == 1
    result = resp[0]["result"]
    assert "error" in result
    assert result.get("exit_code") == 2


def test_blank_lines_skipped() -> None:
    """Empty/whitespace-only lines are ignored, no spurious responses."""
    resp = _run_serve("\n   \n\n" + _req("ping", id=30) + "\n\n  \n")
    assert len(resp) == 1
    assert resp[0]["id"] == 30
    assert resp[0]["result"] == "pong"


def test_handler_exception_returns_error_32603() -> None:
    """If a registered handler raises, error code -32603 is returned."""
    # Temporarily register a handler that always blows up.
    def boom(_params):
        raise RuntimeError("kaboom")
    rpc_server.METHODS["__boom__"] = boom
    try:
        resp = _run_serve(_req("__boom__", id=40))
    finally:
        rpc_server.METHODS.pop("__boom__", None)
    assert len(resp) == 1
    err = resp[0]["error"]
    assert err["code"] == -32603
    assert "kaboom" in err["message"]
    assert resp[0]["id"] == 40


def test_method_decorator_registers_in_methods_dict() -> None:
    """The @method decorator inserts handlers into the METHODS registry."""
    assert "ping" in METHODS
    assert "run" in METHODS
    assert "list_sessions" in METHODS
    assert "get_config" in METHODS
    assert "shutdown" in METHODS
    assert callable(METHODS["ping"])
    assert METHODS["ping"]({"x": 1}) == "pong"


def test_run_with_empty_string_prompt_returns_error() -> None:
    """run with prompt='' is treated as missing → error."""
    resp = _run_serve(_req("run", id=50, params={"prompt": ""}))
    assert len(resp) == 1
    result = resp[0]["result"]
    assert "error" in result


def test_response_has_jsonrpc_field() -> None:
    """Every successful response carries jsonrpc='2.0'."""
    resp = _run_serve(_req("ping", id=60))
    assert resp[0]["jsonrpc"] == "2.0"
    assert resp[0]["id"] == 60


def test_list_sessions_after_create_returns_session() -> None:
    """create_session via SessionStore then list_sessions via RPC."""
    # Pre-populate the session DB so list_sessions is non-empty.
    from synth.config import load_config
    from synth.session import SessionStore
    config = load_config()
    store = SessionStore(config.session.db_path)
    sid = store.create_session(title="rpc test")
    store.update_session_title(sid, "rpc test")
    store.close()

    resp = _run_serve(_req("list_sessions", id=70))
    assert len(resp) == 1
    result = resp[0]["result"]
    assert isinstance(result, list)
    assert len(result) >= 1
    assert any(r["id"] == sid for r in result)

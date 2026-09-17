"""Tests for llm.py: LLMClient chat, retries, streaming, and response parsing.

Zero network: every test monkeypatches litellm.completion with a fake that
records kwargs and returns canned responses/chunks or raises canned errors.
No test ever calls the real litellm, and no test mocks the code under test.

litellm is an optional import at collection time: if the package is not
installed in the running interpreter, a minimal stub (with the exception
classes llm.py catches) is registered in sys.modules first. The stub is only
a stand-in for importability -- the fakes still drive every behavior.
"""

from __future__ import annotations

import sys
import types

import pytest

# --- importability guard: register a stub litellm only if it is absent ------

try:
    import litellm  # noqa: F401  (real package present; nothing to stub)
except ImportError:  # pragma: no cover - depends on the environment

    class _AuthenticationError(Exception):
        pass

    class _RateLimitError(Exception):
        pass

    class _APIConnectionError(Exception):
        pass

    class _Timeout(Exception):
        pass

    _exceptions = types.ModuleType("litellm.exceptions")
    _exceptions.AuthenticationError = _AuthenticationError
    _exceptions.RateLimitError = _RateLimitError
    _exceptions.APIConnectionError = _APIConnectionError
    _exceptions.Timeout = _Timeout

    _litellm = types.ModuleType("litellm")
    _litellm.completion = None  # placeholder; every test monkeypatches it
    _litellm.exceptions = _exceptions
    sys.modules["litellm"] = _litellm
    sys.modules["litellm.exceptions"] = _exceptions
    litellm = _litellm

from synth.constants import NETWORK_RETRIES, RATE_LIMIT_BACKOFF_BASE, RATE_LIMIT_RETRIES
from synth.llm import (
    LLMClient,
    LLMError,
    LLMResponse,
    ToolCall,
    _extract_delta,
    _extract_tool_calls,
    _extract_usage,
    _safe_json,
)


# --- fake litellm.completion factory ----------------------------------------


def _make_error(kind: str) -> Exception:
    """Build the SDK exception llm.py catches, or a plain error for 'generic'."""
    classes = {
        "auth": "AuthenticationError",
        "rate_limit": "RateLimitError",
        "connection": "APIConnectionError",
        "timeout": "Timeout",
    }
    if kind == "generic":
        return ValueError("generic boom")
    cls = getattr(litellm.exceptions, classes[kind])
    # Real litellm exception signatures require llm_provider and model
    # positionally; the stub in self-healing mode does not.
    try:
        return cls(f"fake {kind} error", "provider", "model")
    except TypeError:
        return cls(f"fake {kind} error")


def fake_completion(response=None, chunks=None, errors=(), stream_error=None):
    """Return (fake, calls) where calls records the kwargs of every call.

    errors is a sequence raised one per attempt (by position); once exhausted
    the fake returns `response`. stream_error is raised on the streaming path.
    """
    calls: list[dict] = []

    def _complete(**kwargs):
        calls.append(kwargs)
        if kwargs.get("stream"):
            if stream_error is not None:
                raise _make_error(stream_error)
            return iter(chunks if chunks is not None else [])
        attempt = len(calls)
        if attempt <= len(errors):
            raise _make_error(errors[attempt - 1])
        return response

    return _complete, calls


def text_response(text: str = "Hello", usage: dict | None = None) -> dict:
    """A well-formed non-streaming response carrying plain text."""
    resp = {"choices": [{"message": {"content": text}}]}
    if usage is not None:
        resp["usage"] = usage
    return resp


def tool_call_response() -> dict:
    """A well-formed response asking for one tool call with JSON arguments."""
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "/etc/hosts"}',
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Patch time.sleep so retry backoffs are instant and observable."""
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", lambda seconds: sleeps.append(seconds))
    return sleeps


@pytest.fixture
def patch_completion(monkeypatch: pytest.MonkeyPatch):
    """Monkeypatch litellm.completion and yield (fake, calls) per test."""

    def _install(**factory_kwargs):
        fake, calls = fake_completion(**factory_kwargs)
        monkeypatch.setattr("litellm.completion", fake)
        return fake, calls

    return _install


# --- LLMClient.__init__ -----------------------------------------------------


def test_init_defaults_use_constants():
    client = LLMClient(model="claude-sonnet-4-5", api_key="secret")
    assert client.model == "claude-sonnet-4-5"
    assert client.api_key == "secret"
    assert client._print is None
    assert client.max_retries == RATE_LIMIT_RETRIES
    assert client.backoff_base == RATE_LIMIT_BACKOFF_BASE
    assert client.network_retries == NETWORK_RETRIES


def test_init_custom_params_stored():
    def printer(token: str) -> None:
        pass

    client = LLMClient(
        model="gpt-4o",
        api_key="key",
        stream_printer=printer,
        max_retries=5,
        backoff_base=2.0,
        network_retries=1,
    )
    assert client._print is printer
    assert client.max_retries == 5
    assert client.backoff_base == 2.0
    assert client.network_retries == 1


# --- LLMClient.chat ---------------------------------------------------------


def test_chat_text_only_returns_response(patch_completion):
    _, calls = patch_completion(response=text_response("Hi", usage={
        "prompt_tokens": 5, "completion_tokens": 2}))
    client = LLMClient(model="m", api_key="k")

    result = client.chat(messages=[{"role": "user", "content": "hi"}], stream=False)

    assert isinstance(result, LLMResponse)
    assert result.text == "Hi"
    assert result.tool_calls is None
    assert result.usage == {"input_tokens": 5, "output_tokens": 2}
    assert calls[0] == {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "api_key": "k",
        "stream": False,
    }


def test_chat_tool_call_returns_tool_calls(patch_completion):
    _, calls = patch_completion(response=tool_call_response())
    client = LLMClient(model="m", api_key="k")
    tools = [{"type": "function", "function": {"name": "read_file"}}]

    result = client.chat(
        messages=[{"role": "user", "content": "read it"}], tools=tools, stream=False
    )

    assert result.text is None
    assert result.tool_calls == [
        ToolCall(id="call_1", name="read_file", args={"path": "/etc/hosts"})
    ]
    assert calls[0]["tools"] == tools


def test_chat_without_tools_omits_tools_kwarg(patch_completion):
    _, calls = patch_completion(response=text_response("ok"))
    client = LLMClient(model="m", api_key="k")

    client.chat(messages=[{"role": "user", "content": "go"}], stream=False)

    assert "tools" not in calls[0]


def test_chat_streaming_delegates_to_streaming_path(patch_completion):
    _, calls = patch_completion(chunks=[
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [{"delta": {"content": None}}]},
        {"choices": [{"finish_reason": "stop", "delta": {}}]},
    ])
    tokens: list[str] = []
    client = LLMClient(model="m", api_key="k", stream_printer=tokens.append)

    result = client.chat(messages=[{"role": "user", "content": "hi"}], stream=True)

    assert result.text == "Hello"
    assert tokens == ["Hel", "lo"]
    assert calls[0]["stream"] is True
    assert "tools" not in calls[0]


def test_chat_tools_disables_streaming(patch_completion):
    _, calls = patch_completion(response=tool_call_response())
    tokens: list[str] = []
    client = LLMClient(model="m", api_key="k", stream_printer=tokens.append)
    tools = [{"type": "function", "function": {"name": "read_file"}}]

    result = client.chat(
        messages=[{"role": "user", "content": "read"}], tools=tools, stream=True
    )

    # Tool calls need the full response, so streaming is bypassed.
    assert calls[0]["stream"] is False
    assert tokens == []
    assert result.tool_calls is not None


def test_chat_stream_true_without_printer_is_non_streaming(patch_completion):
    _, calls = patch_completion(response=text_response("ok"))
    client = LLMClient(model="m", api_key="k")

    client.chat(messages=[{"role": "user", "content": "hi"}], stream=True)

    assert calls[0]["stream"] is False


def test_chat_auth_error_raises_llm_error(patch_completion, no_sleep):
    patch_completion(response=None, errors=("auth",))
    client = LLMClient(model="m", api_key="bad-key")

    with pytest.raises(LLMError, match="Authentication failed"):
        client.chat(messages=[{"role": "user", "content": "hi"}], stream=False)

    assert no_sleep == []


# --- LLMClient._chat_streaming (via chat) -----------------------------------


def test_chat_streaming_success_prints_and_accumulates_tokens(patch_completion):
    patch_completion(chunks=[
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
    ])
    tokens: list[str] = []
    client = LLMClient(model="m", api_key="k", stream_printer=tokens.append)

    result = client.chat(messages=[{"role": "user", "content": "hi"}])

    assert result.text == "Hello world"
    assert tokens == ["Hel", "lo", " world"]
    assert result.usage == {}


def test_chat_streaming_only_empty_deltas_returns_none_text(patch_completion):
    patch_completion(chunks=[
        {"choices": [{"delta": {"content": None}}]},
        {"choices": [{"delta": {}}]},
        {"choices": [{"finish_reason": "stop"}]},
    ])
    tokens: list[str] = []
    client = LLMClient(model="m", api_key="k", stream_printer=tokens.append)

    result = client.chat(messages=[{"role": "user", "content": "hi"}])

    assert result.text is None
    assert tokens == []


def test_chat_streaming_sdk_error_raises_llm_error(patch_completion):
    patch_completion(chunks=[], stream_error="generic")
    client = LLMClient(model="m", api_key="k", stream_printer=lambda _t: None)

    with pytest.raises(LLMError, match="Streaming failed"):
        client.chat(messages=[{"role": "user", "content": "hi"}])


def test_chat_streaming_rate_limit_error_raises_llm_error(patch_completion, no_sleep):
    patch_completion(chunks=[], stream_error="rate_limit")
    client = LLMClient(model="m", api_key="k", stream_printer=lambda _t: None)

    with pytest.raises(LLMError, match="Streaming failed"):
        client.chat(messages=[{"role": "user", "content": "hi"}])

    # Streaming has no retry loop; the failure is immediate.
    assert no_sleep == []


# --- LLMClient._call_with_retries (via chat, stream=False) ------------------


def test_call_with_retries_success_returns_response(patch_completion, no_sleep):
    _, calls = patch_completion(response=text_response("ok"))
    client = LLMClient(model="m", api_key="k")

    result = client.chat(messages=[{"role": "user", "content": "hi"}], stream=False)

    assert result.text == "ok"
    assert len(calls) == 1
    assert no_sleep == []


def test_call_with_retries_auth_error_raises_llm_error_immediately(
    patch_completion, no_sleep
):
    _, calls = patch_completion(response=None, errors=("auth",))
    client = LLMClient(model="m", api_key="k")

    with pytest.raises(LLMError, match="Authentication failed"):
        client.chat(messages=[{"role": "user", "content": "hi"}], stream=False)

    # A wrong key never fixes itself: exactly one attempt, no backoff.
    assert len(calls) == 1
    assert no_sleep == []


def test_call_with_retries_rate_limit_retries_then_succeeds(patch_completion, no_sleep):
    _, calls = patch_completion(response=text_response("ok"), errors=("rate_limit",))
    client = LLMClient(model="m", api_key="k")

    result = client.chat(messages=[{"role": "user", "content": "hi"}], stream=False)

    assert result.text == "ok"
    assert len(calls) == 2
    assert no_sleep == [1.0]  # backoff_base * 2**0


def test_call_with_retries_rate_limit_persistent_raises_llm_error(
    patch_completion, no_sleep
):
    _, calls = patch_completion(response=None, errors=("rate_limit",) * 4)
    client = LLMClient(model="m", api_key="k")

    with pytest.raises(LLMError, match="failed after 4 attempts"):
        client.chat(messages=[{"role": "user", "content": "hi"}], stream=False)

    # First attempt + max_retries=3 retries -> 4 calls, 3 backoffs (1s, 2s, 4s).
    assert len(calls) == 4
    assert no_sleep == [1.0, 2.0, 4.0]


def test_call_with_retries_connection_error_retries_then_succeeds(
    patch_completion, no_sleep
):
    _, calls = patch_completion(response=text_response("ok"), errors=("connection",))
    client = LLMClient(model="m", api_key="k")

    result = client.chat(messages=[{"role": "user", "content": "hi"}], stream=False)

    assert result.text == "ok"
    assert len(calls) == 2
    assert no_sleep == [1.0]


def test_call_with_retries_timeout_retries_then_succeeds(patch_completion, no_sleep):
    _, calls = patch_completion(response=text_response("ok"), errors=("timeout",))
    client = LLMClient(model="m", api_key="k")

    result = client.chat(messages=[{"role": "user", "content": "hi"}], stream=False)

    assert result.text == "ok"
    assert len(calls) == 2
    assert no_sleep == [1.0]


def test_call_with_retries_generic_error_raises_llm_error_immediately(
    patch_completion, no_sleep
):
    _, calls = patch_completion(response=None, errors=("generic",))
    client = LLMClient(model="m", api_key="k")

    with pytest.raises(LLMError, match="LLM call failed"):
        client.chat(messages=[{"role": "user", "content": "hi"}], stream=False)

    assert len(calls) == 1
    assert no_sleep == []


# --- LLMClient._parse_response ----------------------------------------------


def test_parse_response_text_only_returns_text_and_usage():
    client = LLMClient(model="m", api_key="k")
    resp = text_response("Hello", usage={"prompt_tokens": 8, "completion_tokens": 2})

    result = client._parse_response(resp, messages=[])

    assert result.text == "Hello"
    assert result.tool_calls is None
    assert result.usage == {"input_tokens": 8, "output_tokens": 2}


def test_parse_response_tool_calls_parsed():
    client = LLMClient(model="m", api_key="k")

    result = client._parse_response(tool_call_response(), messages=[])

    assert result.text is None
    assert result.tool_calls == [
        ToolCall(id="call_1", name="read_file", args={"path": "/etc/hosts"})
    ]
    assert result.usage == {"input_tokens": 12, "output_tokens": 3}


def test_parse_response_null_content_with_tool_calls_succeeds():
    client = LLMClient(model="m", api_key="k")
    resp = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_9",
                            "function": {"name": "bash", "arguments": "{}"},
                        }
                    ],
                }
            }
        ]
    }

    result = client._parse_response(resp, messages=[])

    assert result.tool_calls == [ToolCall(id="call_9", name="bash", args={})]


def test_parse_response_missing_choices_raises_llm_error():
    client = LLMClient(model="m", api_key="k")

    with pytest.raises(LLMError, match="Malformed LLM response"):
        client._parse_response({"id": "x"}, messages=[])


def test_parse_response_empty_message_raises_llm_error():
    client = LLMClient(model="m", api_key="k")
    resp = {"choices": [{"message": {"content": None}}]}

    with pytest.raises(LLMError, match="neither text nor tool calls"):
        client._parse_response(resp, messages=[])


def test_parse_response_missing_usage_returns_empty_dict():
    client = LLMClient(model="m", api_key="k")

    result = client._parse_response(text_response("ok"), messages=[])

    assert result.usage == {}


# --- _extract_delta ---------------------------------------------------------


def test_extract_delta_returns_content():
    assert _extract_delta({"choices": [{"delta": {"content": "abc"}}]}) == "abc"


def test_extract_delta_malformed_chunk_returns_empty():
    assert _extract_delta({}) == ""
    assert _extract_delta({"choices": []}) == ""
    assert _extract_delta({"choices": [{"delta": {}}]}) == ""
    assert _extract_delta(None) == ""


def test_extract_delta_none_content_returns_empty():
    assert _extract_delta({"choices": [{"delta": {"content": None}}]}) == ""


# --- _extract_tool_calls ----------------------------------------------------


def test_extract_tool_calls_parses_json_arguments():
    message = {
        "tool_calls": [
            {
                "id": "call_1",
                "function": {"name": "write_file", "arguments": '{"path": "a.txt"}'},
            }
        ]
    }

    assert _extract_tool_calls(message) == [
        ToolCall(id="call_1", name="write_file", args={"path": "a.txt"})
    ]


def test_extract_tool_calls_accepts_dict_arguments():
    message = {
        "tool_calls": [
            {"id": "call_2", "function": {"name": "bash", "arguments": {"cmd": "ls"}}}
        ]
    }

    assert _extract_tool_calls(message) == [
        ToolCall(id="call_2", name="bash", args={"cmd": "ls"})
    ]


def test_extract_tool_calls_multiple_calls_preserve_order():
    message = {
        "tool_calls": [
            {"id": "a", "function": {"name": "f1", "arguments": "{}"}},
            {"id": "b", "function": {"name": "f2", "arguments": '{"x": 1}'}},
        ]
    }

    calls = _extract_tool_calls(message)

    assert [c.id for c in calls] == ["a", "b"]
    assert [c.name for c in calls] == ["f1", "f2"]
    assert calls[1].args == {"x": 1}


def test_extract_tool_calls_absent_returns_none():
    assert _extract_tool_calls({"content": "hi"}) is None


def test_extract_tool_calls_empty_list_returns_none():
    assert _extract_tool_calls({"tool_calls": []}) is None


def test_extract_tool_calls_malformed_json_raises_llm_error():
    message = {
        "tool_calls": [
            {"id": "call_1", "function": {"name": "f", "arguments": "{oops"}}
        ]
    }

    with pytest.raises(LLMError, match="Failed to parse tool call"):
        _extract_tool_calls(message)


def test_extract_tool_calls_missing_function_raises_llm_error():
    message = {"tool_calls": [{"id": "call_1"}]}

    with pytest.raises(LLMError, match="Failed to parse tool call"):
        _extract_tool_calls(message)


# --- _extract_usage ---------------------------------------------------------


def test_extract_usage_returns_token_counts():
    resp = {"usage": {"prompt_tokens": 10, "completion_tokens": 4}}

    assert _extract_usage(resp) == {"input_tokens": 10, "output_tokens": 4}


def test_extract_usage_missing_usage_returns_empty_dict():
    assert _extract_usage({"choices": []}) == {}


def test_extract_usage_non_numeric_tokens_returns_empty_dict():
    resp = {"usage": {"prompt_tokens": "many", "completion_tokens": 2}}

    assert _extract_usage(resp) == {}


# --- _safe_json -------------------------------------------------------------


def test_safe_json_serializes_response():
    assert _safe_json({"a": 1}) == '{"a": 1}'


def test_safe_json_unserializable_falls_back_to_repr():
    obj = {object(): 1}  # unserializable key -> json.dumps raises

    serialized = _safe_json(obj)

    assert isinstance(serialized, str)
    assert serialized.startswith("{")

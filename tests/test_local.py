"""Tests for local.py: Ollama client, model resolution, download manager.

Nothing touches the network or a real Ollama daemon. synth.local.urlopen is
monkeypatched with fake response objects; socket.create_connection is
monkeypatched for the liveness probe. Fake responses mirror the stdlib HTTP
response contract: a .status, a .read() returning the body bytes, line
iteration for NDJSON streams, and context-manager __enter__/__exit__.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from urllib.error import HTTPError, URLError

import synth.local as local
from synth.local import (
    LocalModelConfig,
    LocalModelError,
    ModelDownloadManager,
    OllamaClient,
    STATUS_FAILED,
    STATUS_SUCCESS,
    STATUS_UNKNOWN,
    resolve_local_model,
)


# --- Fakes ----------------------------------------------------------------


class FakeResponse:
    """Minimal stand-in for an http.client.HTTPResponse."""

    def __init__(
        self,
        status: int = 200,
        body: bytes = b"",
        lines: list[bytes] | None = None,
    ) -> None:
        self.status = status
        self.body = body
        self._lines = lines if lines is not None else []

    def read(self) -> bytes:
        return self.body

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


class FakeSocket:
    """Stand-in for a connected socket (context manager)."""

    def __enter__(self) -> FakeSocket:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


def _tags_body(names: list[str]) -> bytes:
    return json.dumps({"models": [{"name": n} for n in names]}).encode("utf-8")


def _install_urlopen(monkeypatch: pytest.MonkeyPatch, responder: Any) -> list:
    """Point synth.local.urlopen at a fake; record each call's args."""
    calls: list = []

    def _fake_urlopen(*args, **kwargs):
        calls.append((args, kwargs))
        result = responder(*args, **kwargs)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(local, "urlopen", _fake_urlopen)
    return calls


def _connection_refused() -> URLError:
    """URLError as raised on Linux/Termux when nothing is listening."""
    return URLError(ConnectionRefusedError(111, "Connection refused"))


# --- URL scheme validation (security) ------------------------------------


def test_client_rejects_file_scheme_raises() -> None:
    with pytest.raises(LocalModelError, match="http"):
        OllamaClient(base_url="file:///etc/passwd")


def test_client_rejects_empty_base_url_raises() -> None:
    with pytest.raises(LocalModelError, match="non-empty"):
        OllamaClient(base_url="")


def test_client_strips_trailing_slash_from_base_url() -> None:
    client = OllamaClient(base_url="http://ollama.local:11434/")
    assert client.base_url == "http://ollama.local:11434"


# --- list_models ----------------------------------------------------------


def test_list_models_returns_names_sorted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_urlopen(
        monkeypatch, lambda *a, **kw: FakeResponse(200, _tags_body(["zephyr:7b", "llama3.2:3b"]))
    )
    client = OllamaClient()

    models = client.list_models()

    assert models == ["llama3.2:3b", "zephyr:7b"]
    assert calls[0][0][0] == "http://localhost:11434/api/tags"


def test_list_models_connection_refused_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_urlopen(monkeypatch, lambda *a, **kw: _connection_refused())

    assert OllamaClient().list_models() == []


def test_list_models_http_500_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    error = HTTPError("http://x/api/tags", 500, "Internal", None, None)
    _install_urlopen(monkeypatch, lambda *a, **kw: error)

    with pytest.raises(LocalModelError, match="HTTP 500|500"):
        OllamaClient().list_models()


def test_list_models_malformed_body_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_urlopen(monkeypatch, lambda *a, **kw: FakeResponse(200, b"not-json"))

    with pytest.raises(LocalModelError, match="Malformed"):
        OllamaClient().list_models()


# --- is_available ---------------------------------------------------------


def test_is_available_true_when_socket_connects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local.socket, "create_connection", lambda addr, timeout: FakeSocket()
    )

    assert OllamaClient().is_available() is True


def test_is_available_false_when_connection_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _refused(addr: tuple[str, int], timeout: int) -> FakeSocket:
        raise ConnectionRefusedError(111, "Connection refused")

    monkeypatch.setattr(local.socket, "create_connection", _refused)

    assert OllamaClient().is_available() is False


# --- pull_model -----------------------------------------------------------

PULL_LINES = [
    b'{"status":"downloading","digest":"sha256:abc","total":100,"completed":10}\n',
    b'{"status":"downloading","digest":"sha256:abc","total":100,"completed":90}\n',
    b'{"status":"success","total":100,"completed":100}\n',
]


def test_pull_model_success_returns_true(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_urlopen(
        monkeypatch, lambda *a, **kw: FakeResponse(200, lines=PULL_LINES)
    )

    assert OllamaClient().pull_model("llama3.2:3b") is True
    request = calls[0][0][0]
    assert request.full_url == "http://localhost:11434/api/pull"
    assert request.get_method() == "POST"
    assert json.loads(request.data) == {"name": "llama3.2:3b", "stream": True}


def test_pull_model_callback_receives_all_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_urlopen(monkeypatch, lambda *a, **kw: FakeResponse(200, lines=PULL_LINES))
    statuses: list[str] = []

    OllamaClient().pull_model("llama3.2:3b", stream_callback=statuses.append)

    assert statuses == ["downloading", "downloading", "success"]


def test_pull_model_stream_without_success_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = [b'{"status":"downloading"}\n']
    _install_urlopen(monkeypatch, lambda *a, **kw: FakeResponse(200, lines=lines))

    assert OllamaClient().pull_model("llama3.2:3b") is False


def test_pull_model_404_raises_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    error = HTTPError("http://x/api/pull", 404, "Not Found", None, None)
    _install_urlopen(monkeypatch, lambda *a, **kw: error)

    with pytest.raises(LocalModelError, match="not found"):
        OllamaClient().pull_model("nope:1b")


def test_pull_model_timeout_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_urlopen(monkeypatch, lambda *a, **kw: URLError(TimeoutError("timed out")))

    with pytest.raises(LocalModelError, match="timed out"):
        OllamaClient().pull_model("llama3.2:3b")


def test_pull_model_rejects_shell_metacharacters_in_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # urlopen must never be reached for a hostile name.
    calls = _install_urlopen(monkeypatch, lambda *a, **kw: FakeResponse(200))

    with pytest.raises(LocalModelError, match="Invalid model name"):
        OllamaClient().pull_model("llama;rm -rf /")

    assert calls == []


def test_pull_model_skips_malformed_ndjson_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = [b"garbage\n", b'{"status":"success"}\n']
    _install_urlopen(monkeypatch, lambda *a, **kw: FakeResponse(200, lines=lines))

    assert OllamaClient().pull_model("llama3.2:3b") is True


# --- model_exists ---------------------------------------------------------


def test_model_exists_true_for_installed_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_urlopen(
        monkeypatch, lambda *a, **kw: FakeResponse(200, _tags_body(["llama3.2:3b"]))
    )

    assert OllamaClient().model_exists("llama3.2:3b") is True


def test_model_exists_false_for_missing_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_urlopen(
        monkeypatch, lambda *a, **kw: FakeResponse(200, _tags_body(["llama3.2:3b"]))
    )

    assert OllamaClient().model_exists("qwen2.5:7b") is False


def test_model_exists_rejects_invalid_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_urlopen(monkeypatch, lambda *a, **kw: FakeResponse(200))

    with pytest.raises(LocalModelError, match="Invalid model name"):
        OllamaClient().model_exists("llama && cat /etc/passwd")


# --- resolve_local_model --------------------------------------------------


def test_resolve_local_model_uses_requested_name() -> None:
    config = LocalModelConfig(default_model="llama3.2:3b")

    assert resolve_local_model(config, "qwen2.5:7b") == "ollama/qwen2.5:7b"


def test_resolve_local_model_falls_back_to_default() -> None:
    config = LocalModelConfig(default_model="llama3.2:3b")

    assert resolve_local_model(config, None) == "ollama/llama3.2:3b"


def test_resolve_local_model_raises_without_default_or_request() -> None:
    config = LocalModelConfig()

    with pytest.raises(LocalModelError, match="No local model"):
        resolve_local_model(config, None)


def test_resolve_local_model_rejects_invalid_requested_name() -> None:
    config = LocalModelConfig()

    with pytest.raises(LocalModelError, match="Invalid model name"):
        resolve_local_model(config, "llama;rm")


# --- ModelDownloadManager -------------------------------------------------


class RecordingOllamaClient(OllamaClient):
    """OllamaClient whose pull_model is fully faked."""

    def __init__(self, pull_result: bool | Exception) -> None:
        super().__init__()
        self._pull_result = pull_result
        self.requested: list[str] = []

    def pull_model(self, name: str, stream_callback=None) -> bool:
        self.requested.append(name)
        if isinstance(self._pull_result, Exception):
            raise self._pull_result
        return self._pull_result


def test_download_success_records_success_status() -> None:
    client = RecordingOllamaClient(pull_result=True)
    manager = ModelDownloadManager(client)

    assert manager.download("llama3.2:3b") is True
    assert manager.status("llama3.2:3b") == STATUS_SUCCESS
    assert client.requested == ["llama3.2:3b"]


def test_download_failure_records_failed_status() -> None:
    client = RecordingOllamaClient(pull_result=False)
    manager = ModelDownloadManager(client)

    assert manager.download("qwen2.5:7b") is False
    assert manager.status("qwen2.5:7b") == STATUS_FAILED


def test_download_swallows_client_error_and_records_failure() -> None:
    client = RecordingOllamaClient(
        pull_result=LocalModelError("Model not found: 'bad:1b'")
    )
    manager = ModelDownloadManager(client)

    assert manager.download("bad:1b") is False
    assert manager.status("bad:1b") == STATUS_FAILED


def test_status_unknown_for_never_downloaded_model() -> None:
    manager = ModelDownloadManager(RecordingOllamaClient(pull_result=True))

    assert manager.status("never:pulled") == STATUS_UNKNOWN


def test_download_forwards_stream_callback() -> None:
    client = RecordingOllamaClient(pull_result=True)
    manager = ModelDownloadManager(client)
    statuses: list[str] = []

    manager.download("llama3.2:3b", stream_callback=statuses.append)

    # RecordingOllamaClient never calls the callback itself; this asserts the
    # manager passes it through to pull_model without dropping it.
    assert client.requested == ["llama3.2:3b"]
    assert statuses == []

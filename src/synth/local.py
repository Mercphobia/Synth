"""Local model support: Ollama client, model resolution, download manager.

Spec 6.18: talk to a locally running Ollama daemon over HTTP, resolve a
user-supplied model name to a LiteLLM 'ollama/<name>' identifier, and
stream model pulls through a download manager.

Uses only the standard library (urllib + socket) so the project does not
gain another HTTP-client dependency (rules 17). Ollama listens on plain
HTTP on localhost by design; http is allowed for that local case only and
every other scheme is rejected.
"""

from __future__ import annotations

import errno
import json
import logging
import re
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)

# --- Constants ------------------------------------------------------------

# Ollama's default bind address (daemon flag OLLAMA_HOST). Overridable via
# LocalModelConfig / OllamaClient(base_url=...) so tests and remote daemons
# never need to touch this default.
OLLAMA_DEFAULT_URL = "http://localhost:11434"

# Default request timeout: local daemon, so 10s is ample for /api/tags; a
# pull uses its own long-lived stream and is not bound by this short window.
OLLAMA_DEFAULT_TIMEOUT_SECONDS = 10

# HTTP status Ollama returns for an unknown model in /api/pull.
HTTP_NOT_FOUND = 404

# Model names may contain letters, digits and ':', '.', '_', '-' only
# (e.g. 'llama3.2:3b', 'qwen2.5-coder:7b'). Whitelisting keeps shell
# metacharacters and path traversal out of the request body and out of
# the LiteLLM model string.
MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._:-]+$")

# Errnos that mean 'nothing is listening on that host:port' — the daemon
# is simply not running, which is a normal state, not an error.
CONNECTION_REFUSED_ERRNOS = frozenset({errno.ECONNREFUSED, errno.EADDRNOTAVAIL})

# Download-manager lifecycle states (ModelDownloadManager.status()).
STATUS_PENDING = "pending"
STATUS_DOWNLOADING = "downloading"
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_UNKNOWN = "unknown"

# Scheme allow-list for the daemon URL. http is permitted because Ollama
# listens on plain HTTP on localhost; anything else (file://, ftp://, ...)
# is refused so a configured URL can never be pointed at a local file or
# an unexpected transport.
ALLOWED_URL_SCHEMES = frozenset({"http", "https"})

StreamCallback = Callable[[str], None]


class LocalModelError(Exception):
    """Raised when the local model setup is invalid or Ollama fails."""


def _validate_base_url(base_url: str) -> str:
    """Return the base_url minus any trailing slash, or raise.

    Args:
        base_url: Ollama daemon URL, e.g. 'http://localhost:11434'.

    Returns:
        Normalized base URL (no trailing slash).

    Raises:
        LocalModelError: If the URL is empty or its scheme is not http/https.
    """
    if not base_url or not isinstance(base_url, str):
        raise LocalModelError(
            f"Ollama base_url must be a non-empty string, got {base_url!r}"
        )
    parsed = urlparse(base_url)
    if parsed.scheme not in ALLOWED_URL_SCHEMES or not parsed.netloc:
        raise LocalModelError(
            f"Ollama base_url must be http(s)://host:port, got {base_url!r} "
            f"(scheme {parsed.scheme!r} is not allowed)"
        )
    return base_url.rstrip("/")


def _validate_model_name(name: str) -> str:
    """Return the model name if it matches the safe-name whitelist.

    Args:
        name: Ollama model tag, e.g. 'llama3.2:3b'.

    Returns:
        The validated name.

    Raises:
        LocalModelError: If the name is empty or contains characters outside
            [A-Za-z0-9._:-].
    """
    if not name or not isinstance(name, str):
        raise LocalModelError(f"Model name must be a non-empty string, got {name!r}")
    if not MODEL_NAME_PATTERN.match(name):
        raise LocalModelError(
            f"Invalid model name {name!r}: only letters, digits and "
            f"':', '.', '_', '-' are allowed"
        )
    return name


def _is_connection_refused(exc: URLError) -> bool:
    """True when a URLError means 'no daemon is listening'."""
    reason = exc.reason
    if isinstance(reason, OSError):
        return reason.errno in CONNECTION_REFUSED_ERRNOS
    # Some platforms report the errno as a string instead of an OSError.
    return isinstance(reason, str) and "refused" in reason.lower()


def _extract_model_names(body: object) -> list[str]:
    """Pull model names out of a parsed GET /api/tags response.

    Args:
        body: Decoded JSON from /api/tags, shaped like
            {'models': [{'name': 'llama3.2:3b'}, ...]}.

    Returns:
        Sorted, de-duplicated model names; [] if the payload is missing the
        'models' array (older or newer daemon shapes degrade to 'empty').
    """
    if not isinstance(body, dict):
        return []
    models = body.get("models")
    if not isinstance(models, list):
        return []
    names = {
        entry["name"]
        for entry in models
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    }
    return sorted(names)


class OllamaClient:
    """ Talks to a local Ollama daemon over its HTTP API. """

    def __init__(
        self,
        base_url: str = OLLAMA_DEFAULT_URL,
        timeout: int = OLLAMA_DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Store the daemon URL and request timeout.

        Args:
            base_url: Ollama daemon URL (default OLLAMA_DEFAULT_URL).
            timeout: Per-request connect/read timeout in seconds.

        Raises:
            LocalModelError: If base_url is not an http(s):// URL.
        """
        self.base_url = _validate_base_url(base_url)
        self.timeout = timeout

    def list_models(self) -> list[str]:
        """List installed models from GET /api/tags.

        Returns:
            Sorted list of model names, or [] when the daemon is not running.

        Raises:
            LocalModelError: On any non-connection-refused HTTP or network
                failure (a down daemon is expected; a 500 is not).
        """
        url = f"{self.base_url}/api/tags"
        try:
            with urlopen(url, timeout=self.timeout) as response:
                payload = response.read()
        except URLError as exc:
            if _is_connection_refused(exc):
                # Boundary: 'Ollama not installed/running' is a normal state,
                # so warn and return an empty list rather than raising.
                logger.warning(
                    "Ollama not running at %s (connection refused); "
                    "no local models available",
                    self.base_url,
                )
                return []
            code = getattr(exc, "code", None)
            status = f" HTTP {code}" if code is not None else ""
            raise LocalModelError(
                f"Failed to reach Ollama at {url}:{status} {exc.reason}"
            ) from exc
        except OSError as exc:
            raise LocalModelError(f"Failed to reach Ollama at {url}: {exc}") from exc

        try:
            body = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LocalModelError(f"Malformed response from {url}: {exc}") from exc

        models = _extract_model_names(body)
        logger.debug("Ollama at %s has %d model(s)", self.base_url, len(models))
        return models

    def is_available(self) -> bool:
        """Quick TCP probe of the daemon without a full HTTP request.

        Returns:
            True if a TCP connection to the daemon succeeds, False otherwise
            (never raises — this is a liveness check).
        """
        parsed = urlparse(self.base_url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not host:
            return False
        try:
            with socket.create_connection((host, port), timeout=self.timeout):
                return True
        except OSError as exc:
            logger.debug("Ollama at %s:%s unreachable: %s", host, port, exc)
            return False

    def pull_model(
        self, name: str, stream_callback: StreamCallback | None = None
    ) -> bool:
        """Pull (download) a model via streamed POST /api/pull.

    Args:
        name: Ollama model tag, e.g. 'llama3.2:3b'.
        stream_callback: Optional callable invoked once per NDJSON status
            line, e.g. for a progress bar.

    Returns:
        True if the daemon reports a 'success' status.

    Raises:
        LocalModelError: If the name is invalid, the model does not exist
            (HTTP 404), the daemon returns another HTTP error, or the
            request times out.
    """
        _validate_model_name(name)
        url = f"{self.base_url}/api/pull"
        payload = json.dumps({"name": name, "stream": True}).encode("utf-8")
        request = Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return self._consume_pull_stream(response, name, stream_callback)
        except HTTPError as exc:
            if exc.code == HTTP_NOT_FOUND:
                raise LocalModelError(
                    f"Model not found: {name!r} (pull failed with HTTP 404 "
                    f"from {url})"
                ) from exc
            raise LocalModelError(
                f"Ollama returned HTTP {exc.code} pulling {name!r}: "
                f"{_truncate(exc.reason)}"
            ) from exc
        except URLError as exc:
            reason = exc.reason
            if isinstance(reason, TimeoutError) or (
                isinstance(reason, OSError) and reason.errno == errno.ETIMEDOUT
            ):
                raise LocalModelError(
                    f"Pull of {name!r} timed out after {self.timeout}s"
                ) from exc
            raise LocalModelError(
                f"Failed to pull {name!r} from {url}: {reason}"
            ) from exc

    def _consume_pull_stream(
        self,
        response: object,
        name: str,
        stream_callback: StreamCallback | None,
    ) -> bool:
        """Read NDJSON status lines, forwarding each to stream_callback."""
        succeeded = False
        for raw_line in response:  # type: ignore[attr-defined]
            text = raw_line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                line = json.loads(text)
            except json.JSONDecodeError as exc:
                # One bad line shouldn't kill a multi-GB pull; warn and move on.
                logger.warning("Skipping malformed pull status line: %s", exc)
                continue
            status = line.get("status")
            if not isinstance(status, str):
                continue
            if stream_callback is not None:
                stream_callback(status)
            if status == "success":
                succeeded = True
        if not succeeded:
            # Stream ended without 'success': treat as failure, never as
            # silent success (rule 4).
            logger.warning("Pull of %r ended without a success status", name)
        return succeeded

    def model_exists(self, name: str) -> bool:
        """Return True if the model is already installed locally.

        Args:
            name: Ollama model tag, e.g. 'llama3.2:3b'.

        Raises:
            LocalModelError: If the name is invalid or the daemon cannot be
                reached for a reason other than 'not running'.
        """
        _validate_model_name(name)
        return name in self.list_models()


@dataclass(frozen=True)
class LocalModelConfig:
    """Local-model settings (e.g. the [local] section of config.toml)."""

    base_url: str = OLLAMA_DEFAULT_URL
    default_model: str | None = None
    auto_pull: bool = False


def resolve_local_model(
    config: LocalModelConfig, requested: str | None
) -> str:
    """Resolve a model name to a LiteLLM 'ollama/<name>' identifier.

    An explicit request always wins; without one the configured default is
    used. LiteLLM routes 'ollama/...' to the local daemon, so this string is
    what LLMClient consumes.

    Args:
        config: Local-model configuration supplying the fallback default.
        requested: Model name asked for on the command line, or None.

    Returns:
        LiteLLM model string, e.g. 'ollama/llama3.2:3b'.

    Raises:
        LocalModelError: If neither requested nor a default is available, or
            if either contains disallowed characters.
    """
    name = requested if requested else config.default_model
    if not name:
        raise LocalModelError(
            "No local model specified: pass a model name or set "
            "local.default_model in config"
        )
    _validate_model_name(name)
    return f"ollama/{name}"


class ModelDownloadManager:
    """Tracks pull progress for local models by model name."""

    def __init__(self, client: OllamaClient) -> None:
        """Store the client used to perform pulls.

        Args:
            client: OllamaClient the downloads run through.
        """
        self._client = client
        self._downloads: dict[str, str] = {}

    def download(
        self, name: str, stream_callback: StreamCallback | None = None
    ) -> bool:
        """Pull a model, recording its lifecycle status throughout.

    Args:
        name: Ollama model tag, e.g. 'llama3.2:3b'.
        stream_callback: Optional callable invoked per NDJSON status line.

    Returns:
        True if the pull succeeded. A validation failure or pull error is
        recorded as STATUS_FAILED and False is returned (never raised), so
        a batch of downloads survives one bad model.
    """
        self._downloads[name] = STATUS_PENDING
        try:
            self._downloads[name] = STATUS_DOWNLOADING
            ok = self._client.pull_model(name, stream_callback)
        except LocalModelError as exc:
            # Boundary: record the failure and keep going rather than
            # aborting the caller's whole download batch.
            self._downloads[name] = STATUS_FAILED
            logger.warning("Download of %r failed: %s", name, exc)
            return False
        self._downloads[name] = STATUS_SUCCESS if ok else STATUS_FAILED
        return ok

    def status(self, name: str) -> str:
        """Return the last known status for a model, or STATUS_UNKNOWN."""
        return self._downloads.get(name, STATUS_UNKNOWN)


def _truncate(text: object, limit: int = 200) -> str:
    """Truncate error bodies so logs stay readable (rule 16)."""
    value = str(text)
    return value if len(value) <= limit else value[:limit] + "…"

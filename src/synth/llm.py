"""LLM client wrapping LiteLLM.

Boundary module: network failures become typed errors or return values,
never silent. Streaming prints tokens as they arrive (spec 11.2).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

# --- Platform compatibility shims (Termux / Android, Python 3.14) ---
# The tokenizers and fastuuid wheels build against a CPython ABI that is
# missing symbols on Android, so litellm fails to import. Both are used only
# for token counting and UUID generation, neither of which our code path
# needs; we stub them with stdlib equivalents before litellm imports.
import uuid as _uuid

if "tokenizers" not in sys.modules:
    _tok = ModuleType("tokenizers")

    class _StubTokenizer:
        """Minimal stand-in: token counting falls back to estimation."""

        @staticmethod
        def from_pretrained(name: str) -> "_StubTokenizer":
            raise NotImplementedError("tokenizers stub on Android")

    _tok.Tokenizer = _StubTokenizer  # type: ignore[attr-defined]
    sys.modules["tokenizers"] = _tok

if "fastuuid" not in sys.modules:
    _fu = ModuleType("fastuuid")
    _fu.uuid4 = _uuid.uuid4  # type: ignore[attr-defined]
    _fu.uuid7 = lambda *a, **k: _uuid.uuid4()  # type: ignore[attr-defined]
    sys.modules["fastuuid"] = _fu

import litellm

from synth.constants import NETWORK_RETRIES, RATE_LIMIT_BACKOFF_BASE, RATE_LIMIT_RETRIES

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the LLM call fails after retries."""


@dataclass
class ToolCall:
    """One tool invocation requested by the LLM."""

    id: str
    name: str
    args: dict[str, Any]


@dataclass
class LLMResponse:
    """Result of one LLM call."""

    text: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: dict[str, int] = field(default_factory=dict)


class LLMClient:
    """Sends chat completion requests through LiteLLM."""

    def __init__(
        self,
        model: str,
        api_key: str,
        stream_printer: Any = None,
        max_retries: int = RATE_LIMIT_RETRIES,
        backoff_base: float = RATE_LIMIT_BACKOFF_BASE,
        network_retries: int = NETWORK_RETRIES,
    ) -> None:
        """Store credentials and retry policy.

        Args:
            model: LiteLLM model ID (e.g. 'claude-sonnet-4-5').
            api_key: Provider API key.
            stream_printer: Optional callable(str) invoked per streamed token.
            max_retries: Rate-limit retry count (spec 11.3: 3).
            backoff_base: First backoff seconds; doubles per retry (1s, 2s, 4s).
            network_retries: Network-error retry count (spec 11.3: 2).
        """
        self.model = model
        self.api_key = api_key
        self._print = stream_printer
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.network_retries = network_retries

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
    ) -> LLMResponse:
        """Send a chat request, optionally streaming and with tools.

        Args:
            messages: Full conversation as OpenAI-style dicts.
            tools: Tool schemas the model may call.
            stream: If True, print tokens as they arrive.

        Returns:
            LLMResponse with text and/or tool_calls.

        Raises:
            LLMError: On auth failure, rate limit past retries, or an
                unusable response.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "api_key": self.api_key,
        }
        if tools:
            kwargs["tools"] = tools

        if stream and self._print is not None and not tools:
            return self._chat_streaming(messages, kwargs)

        response = self._call_with_retries(messages, kwargs)
        return self._parse_response(response, messages)

    def _chat_streaming(
        self, messages: list[dict[str, Any]], kwargs: dict[str, Any]
    ) -> LLMResponse:
        """Stream a response, accumulating text and printing tokens."""
        chunks: list[str] = []
        try:
            stream = litellm.completion(stream=True, **kwargs)
            for chunk in stream:
                delta = _extract_delta(chunk)
                if delta:
                    chunks.append(delta)
                    self._print(delta)
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 — wrap anything from the SDK
            raise LLMError(f"Streaming failed: {exc}") from exc

        text = "".join(chunks)
        return LLMResponse(text=text or None, usage={})

    def _call_with_retries(
        self, messages: list[dict[str, Any]], kwargs: dict[str, Any]
    ) -> Any:
        """Call the LLM with rate-limit and network retries (spec 11.3)."""
        attempt = 0
        last_error: Exception | None = None

        # Spec 11.3: first attempt, then up to max_retries retries with
        # exponential backoff (1s, 2s, 4s for max_retries=3).
        attempts_left = self.max_retries + 1
        while attempts_left > 0:
            attempts_left -= 1
            try:
                return litellm.completion(stream=False, **kwargs)
            except litellm.exceptions.AuthenticationError as exc:
                # No retry: a wrong key never fixes itself.
                raise LLMError(f"Authentication failed for {self.model}: {exc}") from exc
            except (
                litellm.exceptions.RateLimitError,
                litellm.exceptions.APIConnectionError,
                litellm.exceptions.Timeout,
            ) as exc:
                last_error = exc
                if attempts_left <= 0:
                    break  # last attempt already consumed; don't sleep
                wait = self.backoff_base * (2**attempt)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                    wait,
                )
                time.sleep(wait)
                attempt += 1
            except Exception as exc:  # noqa: BLE001 — anything else is fatal here
                raise LLMError(f"LLM call failed: {exc}") from exc

        raise LLMError(
            f"LLM call failed after {self.max_retries + 1} attempts: {last_error}"
        )

    def _parse_response(
        self, response: Any, messages: list[dict[str, Any]]
    ) -> LLMResponse:
        """Convert a LiteLLM response into LLMResponse, or raise LLMError."""
        try:
            choice = response["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raw = _safe_json(response)
            raise LLMError(f"Malformed LLM response: {raw[:500]}") from exc

        text = message.get("content")
        tool_calls = _extract_tool_calls(message)
        usage = _extract_usage(response)

        if not text and not tool_calls:
            raw = _safe_json(response)
            raise LLMError(f"LLM returned neither text nor tool calls: {raw[:500]}")

        return LLMResponse(text=text, tool_calls=tool_calls, usage=usage)


def _extract_delta(chunk: Any) -> str:
    """Pull the text delta out of one streamed chunk."""
    try:
        return chunk["choices"][0]["delta"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


def _extract_tool_calls(message: Any) -> list[ToolCall] | None:
    """Convert OpenAI-style tool_calls into our ToolCall list."""
    raw_calls = message.get("tool_calls")
    if not raw_calls:
        return None

    calls: list[ToolCall] = []
    for raw in raw_calls:
        try:
            fn = raw["function"]
            args_raw = fn.get("arguments") or "{}"
            args = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
            calls.append(ToolCall(id=raw["id"], name=fn["name"], args=args))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise LLMError(f"Failed to parse tool call {raw}: {exc}") from exc

    return calls or None


def _extract_usage(response: Any) -> dict[str, int]:
    """Best-effort token usage extraction."""
    try:
        raw = response["usage"]
        return {
            "input_tokens": int(raw.get("prompt_tokens", 0)),
            "output_tokens": int(raw.get("completion_tokens", 0)),
        }
    except (KeyError, TypeError, ValueError):
        return {}


def _safe_json(obj: Any) -> str:
    """Serialize a response for error logs without crashing."""
    try:
        return json.dumps(obj, default=str)
    except Exception:  # noqa: BLE001 — logging fallback only
        return repr(obj)[:500]

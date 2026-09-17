"""JSON-RPC server for the Synth Python runtime (spec 6.16 bridge).

Listens on stdin/stdout for NDJSON JSON-RPC 2.0 messages, dispatches
to the appropriate runtime function, and writes responses as NDJSON.

The TS CLI spawns this process with ``python -m synth serve`` and
exchanges messages over its stdin/stdout pipes.

Supported methods:
  - ping() -> "pong"
  - run(prompt, session_id?, mode?) -> {text, iterations, tool_calls, error?}
  - list_sessions() -> [{id, title, ...}]
  - load_session(session_id) -> [messages]
  - get_config() -> {model, max_iterations, ...}
  - shutdown() -> exit
"""

from __future__ import annotations

import json
import sys
import os
from typing import Any

# Methods that the runtime knows how to dispatch. Adding a new method is a
# one-function change: define it in METHODS below and implement it.
METHODS: dict[str, Any] = {}


def method(name: str):
    """Decorator: register a JSON-RPC method."""
    def wrapper(fn):
        METHODS[name] = fn
        return fn
    return wrapper


@method("ping")
def _ping(_params: Any) -> str:
    return "pong"


@method("run")
def _run(params: Any) -> dict[str, Any]:
    """Run the agent loop. params: {prompt, session_id?, mode?}."""
    prompt = params.get("prompt", "") if isinstance(params, dict) else ""
    if not prompt:
        return {"error": "prompt is required"}

    # Lazy import: the agent stack is heavy (litellm etc.) and we don't want
    # to pay that cost just for a ping.
    from synth.config import ConfigError, load_config
    from synth.agent import Agent, AgentConfig, AgentError
    from synth.llm import LLMClient, LLMError
    from synth.modes import resolve_mode, tool_filter
    from synth.memory import MemoryStore, MemoryError, build_memory_prompt
    from synth.session import SessionStore, SessionError
    from synth.tools_ext import extended_registry

    try:
        config = load_config()
        api_key = config.resolve_api_key()
    except ConfigError as exc:
        return {"error": str(exc), "exit_code": 3}

    mode_name = params.get("mode", "act") if isinstance(params, dict) else "act"
    try:
        mode_config = resolve_mode(mode_name)
    except ValueError as exc:
        return {"error": str(exc), "exit_code": 2}

    llm = LLMClient(model=config.model, api_key=api_key)
    tools = extended_registry(
        bash_timeout=config.tools.bash_timeout,
        max_file_size=config.tools.max_file_size,
        max_output_size=config.tools.max_output_size,
    )
    if mode_config.allowed_tool_names is not None:
        tools = tool_filter(tools, mode_config.allowed_tool_names)

    memory_block = ""
    try:
        with MemoryStore() as mem:
            memory_block = build_memory_prompt(mem)
    except MemoryError:
        pass

    try:
        store = SessionStore(config.session.db_path)
    except SessionError as exc:
        return {"error": str(exc), "exit_code": 1}

    session_id = params.get("session_id") if isinstance(params, dict) else None
    if session_id:
        history = store.load_session(session_id)
    else:
        session_id = store.create_session()
        history = []

    agent = Agent(
        llm=llm, tools=tools, session_store=store, session_id=session_id,
        config=AgentConfig(max_iterations=mode_config.max_iterations),
        memory_block=memory_block,
        mode_suffix=mode_config.system_prompt_suffix,
    )

    try:
        result = agent.run(prompt, history)
    except (LLMError, AgentError) as exc:
        store.close()
        return {"error": str(exc), "exit_code": 4}
    store.close()
    return {
        "text": result.text, "iterations": result.iterations,
        "tool_calls": result.tool_calls, "error": result.error,
        "session_id": session_id,
    }


@method("list_sessions")
def _list_sessions(_params: Any) -> list[dict[str, Any]]:
    from synth.config import load_config
    from synth.session import SessionStore
    try:
        config = load_config()
        store = SessionStore(config.session.db_path)
        rows = store.list_sessions()
        store.close()
        return [{"id": r["id"], "title": r["title"] or ""} for r in rows]
    except Exception as exc:  # noqa: BLE001
        return [{"error": str(exc)}]


@method("get_config")
def _get_config(_params: Any) -> dict[str, Any]:
    from synth.config import load_config
    try:
        config = load_config()
        return {"model": config.model, "max_iterations": config.max_iterations}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


@method("shutdown")
def _shutdown(_params: Any) -> None:
    os._exit(0)


# --- protocol loop -------------------------------------------------------

def _send(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _send_notification(method: str, params: dict[str, Any]) -> None:
    _send({"jsonrpc": "2.0", "method": method, "params": params})


def serve() -> int:
    """Read NDJSON lines from stdin, dispatch, write responses to stdout.

    Returns exit code 0 on clean shutdown, 1 on fatal error.
    """
    # Silence stderr from the runtime so it doesn't corrupt the JSON stream.
    # Callers who want debug logs should set SYNTH_LOG_FILE.
    log_file = os.environ.get("SYNTH_LOG_FILE")
    if log_file:
        sys.stderr = open(log_file, "a")  # noqa: SIM115
    else:
        sys.stderr = open(os.devnull, "w")  # noqa: SIM115

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            _send({"jsonrpc": "2.0", "error": {"code": -32700, "message": "parse error"}})
            continue

        msg_id = msg.get("id")
        method_name = msg.get("method")
        params = msg.get("params", {})

        if not method_name:
            _send({"jsonrpc": "2.0", "id": msg_id,
                   "error": {"code": -32600, "message": "invalid request"}})
            continue

        handler = METHODS.get(method_name)
        if handler is None:
            _send({"jsonrpc": "2.0", "id": msg_id,
                   "error": {"code": -32601, "message": f"method not found: {method_name}"}})
            continue

        try:
            result = handler(params)
            _send({"jsonrpc": "2.0", "id": msg_id, "result": result})
        except Exception as exc:  # noqa: BLE001
            _send({"jsonrpc": "2.0", "id": msg_id,
                   "error": {"code": -32603, "message": str(exc)}})

    return 0

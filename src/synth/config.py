"""Config loading for Synth.

Reads ~/.synth/config.toml. On first run the file is created from defaults.
API keys are never stored in config — they come from environment variables.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from synth import constants
from synth.constants import (
    BASH_TIMEOUT,
    CONFIG_DIR,
    CONFIG_FILENAME,
    DEFAULT_API_KEY_ENV_ANTHROPIC,
    DEFAULT_API_KEY_ENV_OPENAI,
    DEFAULT_MODEL,
    MAX_FILE_SIZE,
    MAX_ITERATIONS,
    MAX_OUTPUT_SIZE,
    SESSION_DB_FILENAME,
)
from synth.session import DEFAULT_SESSION_DIR


class ConfigError(Exception):
    """Raised when configuration is invalid or a required value is missing."""


DEFAULT_CONFIG_TOML = f"""[default]
model = "{DEFAULT_MODEL}"
max_iterations = {MAX_ITERATIONS}
stream = true

[providers.anthropic]
api_key_env = "{DEFAULT_API_KEY_ENV_ANTHROPIC}"

[providers.openai]
api_key_env = "{DEFAULT_API_KEY_ENV_OPENAI}"

[session]
db_path = "~/{CONFIG_DIR}/{SESSION_DB_FILENAME}"

[tools]
bash_timeout = {BASH_TIMEOUT}
max_file_size = {MAX_FILE_SIZE}
max_output_size = {MAX_OUTPUT_SIZE}
"""


@dataclass(frozen=True)
class ProviderConfig:
    """Maps a provider name to the env var holding its API key."""

    api_key_env: str


@dataclass(frozen=True)
class ToolsConfig:
    """Limits applied to tool execution."""

    bash_timeout: int = BASH_TIMEOUT
    max_file_size: int = MAX_FILE_SIZE
    max_output_size: int = MAX_OUTPUT_SIZE


@dataclass(frozen=True)
class SessionConfig:
    """Where session history lives."""

    db_path: str = f"~/{CONFIG_DIR}/{SESSION_DB_FILENAME}"


@dataclass(frozen=True)
class Config:
    """Fully resolved configuration for one Synth run."""

    model: str = DEFAULT_MODEL
    max_iterations: int = MAX_ITERATIONS
    stream: bool = True
    providers: dict[str, ProviderConfig] = field(
        default_factory=lambda: {
            "anthropic": ProviderConfig(DEFAULT_API_KEY_ENV_ANTHROPIC),
            "openai": ProviderConfig(DEFAULT_API_KEY_ENV_OPENAI),
        }
    )
    session: SessionConfig = field(default_factory=SessionConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    config_path: Path = Path("~/.synth/config.toml")

    def resolve_api_key(self) -> str:
        """Return the API key for the configured model's provider.

        LiteLLM routes by model name; we map the model prefix to the provider
        entry in config, then read the key from its env var.

        Raises:
            ConfigError: If the env var is unset or empty.
        """
        provider = _provider_for_model(self.model)
        if provider not in self.providers:
            # Unknown provider: fall back to OPENAI_API_KEY, which LiteLLM
            # also uses as a generic default for OpenAI-compatible endpoints.
            provider = "openai"
        env_var = self.providers[provider].api_key_env
        key = os.environ.get(env_var, "")
        if not key:
            raise ConfigError(
                f"Missing API key: set {env_var} in environment "
                f"(model {self.model} expects provider {provider})"
            )
        return key


def _provider_for_model(model: str) -> str:
    """Map a LiteLLM model name to a provider key used in config."""
    # LiteLLM prefixes: anthropic/, openai/, or bare OpenAI-style names.
    if model.startswith("anthropic/") or model.startswith("claude"):
        return "anthropic"
    return "openai"


def config_path() -> Path:
    """Return the path to the config file (~/.synth/config.toml)."""
    return Path(os.path.expanduser(f"~/{CONFIG_DIR}/{CONFIG_FILENAME}"))


def load_config(path: Path | None = None) -> Config:
    """Load configuration, creating it from defaults on first run.

    Args:
        path: Override config location (tests use temp files).

    Returns:
        Fully resolved Config.

    Raises:
        ConfigError: If the file is unreadable, malformed, or a required
            key is missing.
    """
    resolved_path = path if path is not None else config_path()
    if not resolved_path.exists():
        _write_default_config(resolved_path)
    raw = _read_toml(resolved_path)
    return _parse_config(raw, resolved_path)


def _write_default_config(path: Path) -> None:
    """Create the config directory and write the default config file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Failed to write default config to {path}: {exc}") from exc


def _read_toml(path: Path) -> dict[str, Any]:
    """Parse the TOML file, wrapping parse errors with a helpful message."""
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Cannot read config file {path}: {exc}") from exc


def _parse_config(raw: dict[str, Any], source_path: Path) -> Config:
    """Convert the raw TOML dict into a validated Config object."""
    default_section = raw.get("default", {})
    if not isinstance(default_section, dict):
        raise ConfigError("'default' section must be a table")

    model = default_section.get("model", DEFAULT_MODEL)
    if not isinstance(model, str) or not model:
        raise ConfigError("'default.model' must be a non-empty string")

    max_iterations = default_section.get("max_iterations", MAX_ITERATIONS)
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ConfigError("'default.max_iterations' must be a positive integer")

    stream = default_section.get("stream", True)
    if not isinstance(stream, bool):
        raise ConfigError("'default.stream' must be a boolean")

    # A top-level 'providers' key that isn't a table (e.g. providers = "x")
    # is invalid: the real table would live under [providers.*].
    if "providers" in raw and not isinstance(raw["providers"], dict):
        raise ConfigError("'providers' section must be a table")

    providers = _parse_providers(raw.get("providers", {}))
    session = _parse_session(raw.get("session", {}))
    tools = _parse_tools(raw.get("tools", {}))

    return Config(
        model=model,
        max_iterations=max_iterations,
        stream=stream,
        providers=providers,
        session=session,
        tools=tools,
        config_path=source_path,
    )


def _parse_providers(raw: Any) -> dict[str, ProviderConfig]:
    if not isinstance(raw, dict):
        raise ConfigError("'providers' section must be a table")
    providers: dict[str, ProviderConfig] = {}
    for name, section in raw.items():
        if not isinstance(section, dict):
            raise ConfigError(f"'providers.{name}' must be a table")
        env_var = section.get("api_key_env")
        if not isinstance(env_var, str) or not env_var:
            raise ConfigError(f"'providers.{name}.api_key_env' must be a non-empty string")
        providers[name] = ProviderConfig(api_key_env=env_var)
    return providers


def _parse_session(raw: Any) -> SessionConfig:
    if not isinstance(raw, dict):
        raise ConfigError("'session' section must be a table")
    db_path = raw.get("db_path")
    if db_path is not None and (not isinstance(db_path, str) or not db_path):
        raise ConfigError("'session.db_path' must be a non-empty string")
    if db_path is None:
        return SessionConfig()
    return SessionConfig(db_path=db_path)

def _parse_tools(raw: Any) -> ToolsConfig:
    if not isinstance(raw, dict):
        raise ConfigError("'tools' section must be a table")

    def _positive_int(value: Any, field_name: str, default: int) -> int:
        if value is None:
            return default
        if not isinstance(value, int) or value <= 0:
            raise ConfigError(f"'tools.{field_name}' must be a positive integer")
        return value

    return ToolsConfig(
        bash_timeout=_positive_int(raw.get("bash_timeout"), "bash_timeout", BASH_TIMEOUT),
        max_file_size=_positive_int(raw.get("max_file_size"), "max_file_size", MAX_FILE_SIZE),
        max_output_size=_positive_int(raw.get("max_output_size"), "max_output_size", MAX_OUTPUT_SIZE),
    )


# Kept for symmetry with spec; session module owns the real default dir.
_ = constants  # noqa: F841  (avoid unused-import lint noise if refactored)
_ = DEFAULT_SESSION_DIR

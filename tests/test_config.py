"""Tests for config.py: first-run bootstrap, TOML parsing, validation, API keys.

No test ever touches the real ~/.synth directory: every load_config() call
passes an explicit path under tmp_path, and resolve_api_key() tests build
Config objects directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synth.config import (
    Config,
    ConfigError,
    ProviderConfig,
    SessionConfig,
    ToolsConfig,
    config_path,
    load_config,
)
from synth.constants import (
    BASH_TIMEOUT,
    DEFAULT_API_KEY_ENV_ANTHROPIC,
    DEFAULT_API_KEY_ENV_OPENAI,
    DEFAULT_MODEL,
    MAX_FILE_SIZE,
    MAX_ITERATIONS,
    MAX_OUTPUT_SIZE,
)


# --- Fixtures -------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep tests out of the real home dir and real environment.

    Removes any API keys that happen to be set in the ambient environment so
    the "missing key" tests are deterministic; monkeypatch restores them.
    """
    monkeypatch.chdir(tmp_path)
    for var in (DEFAULT_API_KEY_ENV_ANTHROPIC, DEFAULT_API_KEY_ENV_OPENAI):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


@pytest.fixture
def write_config(tmp_path: Path):
    """Return a helper that writes TOML text to a temp config path."""

    def _write(text: str, name: str = "config.toml") -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    return _write


MINIMAL_TOML = """[default]
model = "openai/gpt-4o"
max_iterations = 5
stream = false
"""


# --- load_config: success cases ------------------------------------------


def test_load_config_existing_file_returns_parsed_values(write_config) -> None:
    path = write_config(
        """[default]
model = "anthropic/claude-opus-4"
max_iterations = 7
stream = false

[providers.openai]
api_key_env = "MY_OPENAI_KEY"

[session]
db_path = "/tmp/custom-sessions.db"

[tools]
bash_timeout = 60
max_file_size = 500
max_output_size = 2000
"""
    )

    config = load_config(path)

    assert config.model == "anthropic/claude-opus-4"
    assert config.max_iterations == 7
    assert config.stream is False
    assert config.providers == {"openai": ProviderConfig(api_key_env="MY_OPENAI_KEY")}
    assert config.session == SessionConfig(db_path="/tmp/custom-sessions.db")
    assert config.tools == ToolsConfig(
        bash_timeout=60, max_file_size=500, max_output_size=2000
    )
    assert config.config_path == path


def test_load_config_first_run_writes_default_config(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "config.toml"
    assert not path.exists()

    config = load_config(path)

    # The file is created from defaults, not silently absent.
    assert path.exists()
    assert path.read_text(encoding="utf-8").startswith("[default]")
    # And the returned config mirrors those defaults.
    assert config.model == DEFAULT_MODEL
    assert config.max_iterations == MAX_ITERATIONS
    assert config.stream is True
    assert config.providers == {
        "anthropic": ProviderConfig(api_key_env=DEFAULT_API_KEY_ENV_ANTHROPIC),
        "openai": ProviderConfig(api_key_env=DEFAULT_API_KEY_ENV_OPENAI),
    }
    assert config.tools == ToolsConfig(
        bash_timeout=BASH_TIMEOUT,
        max_file_size=MAX_FILE_SIZE,
        max_output_size=MAX_OUTPUT_SIZE,
    )
    assert config.config_path == path


def test_load_config_first_run_is_idempotent(tmp_path: Path) -> None:
    """A second load of the generated file yields the same defaults."""
    path = tmp_path / "config.toml"
    first = load_config(path)
    second = load_config(path)
    assert first == second


def test_load_config_custom_db_path_parsed(write_config) -> None:
    path = write_config(
        MINIMAL_TOML
        + """
[session]
db_path = "~/.synth/elsewhere.db"
"""
    )
    assert load_config(path).session.db_path == "~/.synth/elsewhere.db"


def test_load_config_custom_tools_values_parsed(write_config) -> None:
    path = write_config(
        MINIMAL_TOML
        + """
[tools]
bash_timeout = 120
max_file_size = 2_000_000
max_output_size = 50_000
"""
    )
    tools = load_config(path).tools
    assert tools.bash_timeout == 120
    assert tools.max_file_size == 2_000_000
    assert tools.max_output_size == 50_000


# --- load_config: error cases --------------------------------------------


def test_load_config_invalid_toml_raises_config_error(write_config) -> None:
    path = write_config("[default\nmodel = ")  # malformed TOML
    with pytest.raises(ConfigError, match="Invalid TOML"):
        load_config(path)


def test_load_config_unreadable_path_raises_config_error(tmp_path: Path) -> None:
    """A path that exists but cannot be read as a file raises ConfigError."""
    path = tmp_path / "config.toml"
    path.mkdir()  # a directory, not a file
    with pytest.raises(ConfigError):
        load_config(path)


def test_load_config_invalid_model_raises_config_error(write_config) -> None:
    empty = write_config('[default]\nmodel = ""\nmax_iterations = 5\n')
    with pytest.raises(ConfigError, match=r"default.model"):
        load_config(empty)

    wrong_type = write_config("[default]\nmodel = 42\nmax_iterations = 5\n")
    with pytest.raises(ConfigError, match=r"default.model"):
        load_config(wrong_type)


def test_load_config_invalid_max_iterations_raises_config_error(write_config) -> None:
    zero = write_config('[default]\nmodel = "x"\nmax_iterations = 0\n')
    with pytest.raises(ConfigError, match="max_iterations"):
        load_config(zero)

    negative = write_config('[default]\nmodel = "x"\nmax_iterations = -3\n')
    with pytest.raises(ConfigError, match="max_iterations"):
        load_config(negative)

    wrong_type = write_config('[default]\nmodel = "x"\nmax_iterations = "many"\n')
    with pytest.raises(ConfigError, match="max_iterations"):
        load_config(wrong_type)


def test_load_config_invalid_stream_raises_config_error(write_config) -> None:
    text_stream = write_config(
        '[default]\nmodel = "x"\nmax_iterations = 5\nstream = "yes"\n'
    )
    with pytest.raises(ConfigError, match="stream"):
        load_config(text_stream)

    int_stream = write_config(
        '[default]\nmodel = "x"\nmax_iterations = 5\nstream = 1\n'
    )
    with pytest.raises(ConfigError, match="stream"):
        load_config(int_stream)


def test_load_config_invalid_tools_values_raise_config_error(write_config) -> None:
    zero_timeout = write_config(MINIMAL_TOML + "[tools]\nbash_timeout = 0\n")
    with pytest.raises(ConfigError, match=r"tools.bash_timeout"):
        load_config(zero_timeout)

    negative_size = write_config(MINIMAL_TOML + "[tools]\nmax_file_size = -10\n")
    with pytest.raises(ConfigError, match=r"tools.max_file_size"):
        load_config(negative_size)

    wrong_type = write_config(MINIMAL_TOML + '[tools]\nmax_output_size = "lots"\n')
    with pytest.raises(ConfigError, match=r"tools.max_output_size"):
        load_config(wrong_type)


def test_load_config_invalid_providers_raise_config_error(write_config) -> None:
    empty_env = write_config(
        MINIMAL_TOML + '[providers.openai]\napi_key_env = ""\n'
    )
    with pytest.raises(ConfigError, match="api_key_env"):
        load_config(empty_env)

    # Top-level 'providers' must be a table. Placed before [default] so it
    # lands at the root, not inside the default section.
    not_a_table = write_config('providers = "nope"\n' + MINIMAL_TOML)
    with pytest.raises(ConfigError, match="providers"):
        load_config(not_a_table)


def test_load_config_invalid_session_db_path_raises_config_error(write_config) -> None:
    empty = write_config(MINIMAL_TOML + '[session]\ndb_path = ""\n')
    with pytest.raises(ConfigError, match="db_path"):
        load_config(empty)

    wrong_type = write_config(MINIMAL_TOML + "[session]\ndb_path = 3\n")
    with pytest.raises(ConfigError, match="db_path"):
        load_config(wrong_type)


# --- resolve_api_key -----------------------------------------------------


def test_resolve_api_key_anthropic_env_returns_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(DEFAULT_API_KEY_ENV_ANTHROPIC, "sk-anthropic-123")
    config = Config(model="claude-sonnet-4-5")
    assert config.resolve_api_key() == "sk-anthropic-123"


def test_resolve_api_key_openai_env_returns_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(DEFAULT_API_KEY_ENV_OPENAI, "sk-openai-abc")
    config = Config(model="openai/gpt-4o")
    assert config.resolve_api_key() == "sk-openai-abc"


def test_resolve_api_key_unknown_provider_falls_back_to_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bare/unknown model names resolve via the OpenAI env var."""
    monkeypatch.setenv(DEFAULT_API_KEY_ENV_OPENAI, "sk-openai-fallback")
    config = Config(model="some-unknown-model")
    assert config.resolve_api_key() == "sk-openai-fallback"


def test_resolve_api_key_missing_env_raises_config_error() -> None:
    # isolated_env removed both keys; claude model wants ANTHROPIC_API_KEY.
    config = Config(model="claude-sonnet-4-5")
    with pytest.raises(ConfigError, match="Missing API key"):
        config.resolve_api_key()


def test_resolve_api_key_empty_env_raises_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(DEFAULT_API_KEY_ENV_OPENAI, "")
    config = Config(model="gpt-4o")
    with pytest.raises(ConfigError, match="Missing API key"):
        config.resolve_api_key()


def test_resolve_api_key_custom_env_var_returns_key(
    write_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider entry pointing at a custom env var is honored."""
    monkeypatch.setenv("CUSTOM_LITELLM_KEY", "sk-custom")
    path = write_config(
        """[default]
model = "openai/gpt-4o"
max_iterations = 5
stream = true

[providers.openai]
api_key_env = "CUSTOM_LITELLM_KEY"
"""
    )
    assert load_config(path).resolve_api_key() == "sk-custom"


# --- config_path ---------------------------------------------------------


def test_config_path_points_into_synth_dir() -> None:
    """Shapes the path without writing to (or reading) the real home dir."""
    path = config_path()
    assert path.name == "config.toml"
    assert path.parent.name == ".synth"

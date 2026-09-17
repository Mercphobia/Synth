# Synth

**A CLI AI agent framework that runs tasks with an LLM and a small set of file/shell tools.**

Synth reads a task from the command line, reasons about it with a ReAct loop
(think → call a tool → observe → repeat), and prints the final answer. It
remembers every session in a SQLite database so you can resume work later.

This is the **MVP v1**: one agent, three tools, one config file. The full
spec (multi-agent swarms, memory, skills, sandbox, token-efficiency layers)
lives in `Spec.md`; see [Limitations & Security](#limitations--security) for
what is deliberately not here yet.

---

## Requirements

- Python ≥ 3.12 (developed on 3.14)
- [uv](https://docs.astral.sh/uv/) — the package manager
- An API key from Anthropic **or** OpenAI, exported in your environment

## Installation

```bash
git clone <repo-url> synth
cd synth
uv sync
```

`uv sync` creates the `.venv/` and installs `synth` plus dev dependencies
(`pytest`, `pytest-cov`) from `pyproject.toml`.

Run the installed console script:

```bash
synth "read pyproject.toml and summarize it"
```

…or run it through uv without activating anything:

```bash
uv run synth "read pyproject.toml and summarize it"
```

You can also invoke the module directly:

```bash
uv run python -m synth --version
```

On first run Synth creates `~/.synth/config.toml` (see
[Configuration](#configuration)). If the required API key is missing, Synth
exits with code `3` and tells you which variable to set — it never prompts
and never stores keys on disk.

---

## Usage

```bash
synth [PROMPT] [--resume ID] [--session ACTION] [--model MODEL]
      [--mode MODE] [--godmode] [--caveman] [--no-rtk] [--no-stream]
      [--checkpoint TAG] [--undo] [--swarm] [--verbose] [--version]
```

The `PROMPT` is the task. It is required unless you pass a subcommand,
`--session`, `--version`, or `-h`.

### Subcommands

| Command | Description |
|---|---|
| `synth chat` | Interactive REPL with `/help`, `/new`, `/session`, `/exit`. Prints a session summary on exit. |
| `synth cost [--today\|--month]` | Token/cost report from the cost tracker. |
| `synth best-of-n "task" --models a,b,c` | Run a prompt across N models, score, pick best. |
| `synth doctor` | Environment health check (Python, config, API key, DB, git, Ollama, disk). |
| `synth init [--force]` | Generate `AGENTS.md` project guide. |
| `synth config path\|get\|set` | Edit `~/.synth/config.toml` by dotted key. |
| `synth scan [PATH] [--markdown]` | Security audit: secrets, SAST, CVE, OWASP. |
| `synth session search "query"` | FTS5 full-text search across session history. |
| `synth session export <id> --format json\|md\|html [--out FILE]` | Export a session. |
| `synth snapshot create <session> [--tag LABEL]` | Create a checkpoint snapshot. |
| `synth snapshot list <session>` | List snapshots for a session. |
| `synth snapshot restore <checkpoint>` | Restore a snapshot into the live session. |
| `synth git status\|diff\|commit\|log\|review` | Git integration (auto-commit, diff, secret-warn). |
| `synth models list\|available\|install\|uninstall\|status` | Ollama model management. |
| `synth task list\|create\|status\|delete` | Long-horizon task tracking. |
| `synth cron add\|remove\|list\|enable\|disable\|run` | Cron job scheduler. |
| `synth daemon start\|stop\|status` | Background daemon with task queue + health check. |

### Examples for every flag

Run a task (the positional argument):

```bash
synth "baca pyproject.toml dan jelaskan isinya"
synth "create a file notes.txt with today's date"
synth "run the test suite and report which tests fail"
```

`--resume ID` — continue an existing session by its ID (see
[Sessions](#sessions)):

```bash
synth --resume 3f5a1c2b-... "now add the same thing for README.md"
```

`--session ACTION` — session management. The only action in the MVP is
`list`, which prints the 20 most recent sessions, newest first:

```bash
synth --session list
```

`--model MODEL` — override the model from config for this one run. The value
is any LiteLLM model ID:

```bash
synth --model claude-sonnet-4-5 "explain this function"
synth --model openai/gpt-4o "explain this function"
```

The API key is resolved from the **configured default model's** provider env
var (see [Configuration](#configuration)), so if you point `--model` at a
provider different from the default, make sure that provider's variable is
set too.

`--no-stream` — disable token streaming and wait for the full response
before printing it:

```bash
synth --no-stream "write a haiku about sqlite"
```

`--verbose` — emit DEBUG-level logs (LLM calls, retries, iteration numbers)
to stderr:

```bash
synth --verbose "find every TODO in this repo"
```

`--version` — print the version and exit:

```bash
synth --version
# synth 0.1.0
```

`-h` / `--help` — the full usage reference:

```bash
synth --help
```

### What a run looks like

Tool calls and their results are printed inline as the agent works; the
final answer is rendered as Markdown:

```
  ▸ read_file     path=pyproject.toml
     [default] model = "synth"
  ▸ bash          command=python -c "print(2+2)"
     exit_code=0
     ...
```

The loop stops when the model answers with plain text (no tool call) or
after 20 iterations (configurable, see below).

---

## Configuration

Synth reads `~/.synth/config.toml`. The file is created from defaults on
first run; edit it freely. Every value below is the default.

```toml
[default]
model = "claude-sonnet-4-5"
max_iterations = 20
stream = true

[providers.anthropic]
api_key_env = "ANTHROPIC_API_KEY"

[providers.openai]
api_key_env = "OPENAI_API_KEY"

[session]
db_path = "~/.synth/sessions.db"

[tools]
bash_timeout = 30
max_file_size = 1000000
max_output_size = 10000
```

### Sections

| Section        | Key              | Meaning                                                            |
|----------------|------------------|--------------------------------------------------------------------|
| `[default]`    | `model`          | LiteLLM model ID used when `--model` is not given.                 |
|                | `max_iterations` | Hard cap on ReAct loop iterations.                                 |
|                | `stream`         | Stream tokens as they arrive (also toggleable with `--no-stream`). |
| `[providers.*]`| `api_key_env`    | Name of the environment variable holding that provider's key.      |
| `[session]`    | `db_path`        | Where the SQLite session database lives.                           |
| `[tools]`      | `bash_timeout`   | Seconds before a `bash` command is killed.                         |
|                | `max_file_size`  | Max bytes for `read_file` input and `write_file` output.           |
|                | `max_output_size`| Max bytes returned by any single tool call.                        |

### API keys: environment variables only

Synth **never writes or reads API keys in the config file**. Each provider
entry only names the environment variable to read:

```bash
# Anthropic models (default)
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI models, or any LiteLLM route that falls back to OpenAI
export OPENAI_API_KEY="sk-..."
```

Resolution rules:

- A model starting with `anthropic/` or `claude` maps to the `anthropic`
  provider; anything else maps to `openai`.
- An unknown provider falls back to `openai` / `OPENAI_API_KEY`, which
  LiteLLM also uses for OpenAI-compatible endpoints.
- If the variable is unset or empty, Synth prints which variable it wanted
  and exits with code **3**.

---

## Sessions

Every run is stored in `~/.synth/sessions.db` (SQLite, WAL mode): the
session row plus every message — user, assistant, tool calls, and tool
results — so a session can be replayed exactly.

List recent sessions (20 newest first):

```bash
synth --session list
```

```
Sessions (newest first):
  3f5a1c2b-...  (untitled)
  9d8e7f6a-...  (untitled)
```

Resume one — its history is loaded and sent back to the model, then the new
prompt continues the same conversation:

```bash
synth --resume 3f5a1c2b-... "same thing, but for the tests directory"
```

Without `--resume`, each run starts a fresh session. The database is created
automatically on first use; the schema is migrated on open.

---

## Tools

Synth ships **8 tools**. Every tool returns errors as text instead of
raising — the model sees the failure and can change approach.

| Tool        | Arguments                          | Notes                                                    |
|-------------|------------------------------------|----------------------------------------------------------|
| `read_file` | `path`                             | Reads text (UTF-8, invalid bytes replaced).              |
| `write_file`| `path`, `content`                  | Creates parent dirs; **overwrites** existing files.     |
| `bash`      | `command`                          | Runs via `shell=True` in the current working directory.  |
| `edit_file` | `path`, `old_string`, `new_string` | Patch-style edit; old_string must be unique in file.     |
| `glob`      | `pattern`, `path?`                 | Find files matching a glob pattern (supports `**`).       |
| `grep`      | `pattern`, `path?`, `include?`     | Regex search inside files, ripgrep-style output.         |
| `web_fetch` | `url`, `method?`                   | HTTP GET/POST with SSRF guards (no private IPs by default). |
| `web_search`| `query`, `limit?`                  | DuckDuckGo HTML search; no API keys needed.              |

### Modes

| Mode | Behavior | Flag |
|---|---|---|
| `plan` | Reason only, no tool execution (40 iterations) | `--mode plan` |
| `act` | Default: read, write, edit, run commands (20 iter) | `--mode act` |
| `auto` | Maximum autonomy, finish end-to-end (50 iter) | `--mode auto` |
| `architect` | Design docs only, read-only tools (15 iter) | `--mode architect` |

### Godmode

`--godmode` disables all safety boundaries: path traversal guard, file size
limits, output truncation, and bash timeout. Use with **extreme caution** and
only with trusted models in isolated environments. With `GODMODE=1` set,
every tool operates unrestricted.

### Limits

| Limit                     | Default      | Config key                |
|---------------------------|--------------|---------------------------|
| ReAct loop iterations     | 20           | `default.max_iterations`  |
| `bash` timeout            | 30 s         | `tools.bash_timeout`      |
| Tool output size          | 10,000 bytes | `tools.max_output_size`   |
| File size (read and write)| 1,000,000 B  | `tools.max_file_size`     |
| Sessions listed           | 20           | hardcoded (`--session list`) |

Guards that are present in the MVP:

- **Path traversal blocked** — any path containing `..`, or resolving
  outside the current working directory, is rejected.
- **Command timeout** — `bash` is killed after `bash_timeout` seconds.
- **Output truncation** — output past `max_output_size` is cut with a
  `...[truncated at N bytes]` marker so one verbose command can't eat the
  context window.
- **Size caps** — files above `max_file_size` are refused for reading, and
  writes of more than `max_file_size` bytes are refused.
- **Typed failures** — unknown tool, missing arguments, and non-string
  arguments all become error messages the model can read, not crashes.

---

## Exit codes

| Code | Meaning                                        |
|------|------------------------------------------------|
| 0    | Success — the agent produced a final answer.   |
| 1    | General error (e.g. the session DB can't open).|
| 2    | Invalid argument (e.g. no prompt given).       |
| 3    | Config error — including a **missing API key**.|
| 4    | LLM error (auth failure, or retries exhausted).|
| 5    | Tool error (a tool failed terminally).         |
| 6    | Security violation (e.g. audit block).          |
| 7    | Sandbox error.                                  |
| 8    | Permission denied.                              |
| 9    | Cancelled by user (Ctrl-C / SIGINT).           |

---

## Testing

Tests live in `tests/` and use `pytest` with `pytest-cov`; the coverage
defaults are already wired into `pyproject.toml` (`--cov=src/synth
--cov-report=term-missing`), so a plain run reports coverage:

```bash
uv run pytest
```

To get the coverage report explicitly:

```bash
uv run pytest --cov
```

Coverage targets (per spec §17): **minimum 80%, target 90%**.

Run a single test file or test:

```bash
uv run pytest tests/test_tools.py
uv run pytest tests/test_tools.py::TestBash::test_timeout_returns_error
```

---

## Limitations & Security

This is MVP v1. The features below are **known gaps, not oversights** — they
are scheduled for v1.0 (see `Spec.md` §3.19 and the v1.0 acceptance
criteria). Until then, treat Synth as trusted-person tooling and run it in a
working directory you can afford to lose.

**Missing safeguards (the gaps)**

- **No sandbox for `bash`.** Commands run directly on your host with
  `shell=True` — no Docker, no nsjail, no filesystem isolation. Anything the
  model can say, it can do.
- **No permission prompts.** There is no "allow / deny / always" prompt
  before a tool runs. `bash` and `write_file` execute immediately and
  silently.
- **No network policy.** No egress filtering — `bash` can reach the network,
  and nothing in the MVP fetches URLs but nothing blocks it either.
- **Files are overwritten without confirmation.** `write_file` clobbers
  existing content by design; there is no backup, diff preview, or undo.
- **The user prompt is trusted.** There are no prompt-injection defenses.
  Content the model reads via `read_file` is data, but nothing stops it from
  being treated as instructions — don't point Synth at untrusted files.

**What is already in place (defensive, per spec §2.11)**

- API keys come **only** from environment variables and are never written to
  the config file or logs.
- Path-traversal check on every file tool.
- Parameterized SQL everywhere — user input never touches SQL as text.
- `bash` timeout, output truncation, and file-size caps (see [Tools](#tools)).
- A bounded loop (20 iterations) so a confused model can't run forever.
- Tool failures return error strings instead of raising, and LLM auth
  failures never retry (a bad key does not fix itself).

**Other MVP boundaries worth knowing**

- **Streaming is not wired to the terminal.** The CLI builds the LLM client
  without a stream printer, so responses arrive as one completion and are
  rendered with `rich` Markdown at the end. `--no-stream` is still accepted
  and passed through; full token-by-token printing arrives with the v1.0
  streaming work.
- **`--model` does not re-resolve the API key.** The key is read from the
  *configured default model's* provider variable, even when you override the
  model on the command line.
- **Resuming an unknown session ID does not fail.** It starts a fresh
  history under that ID rather than erroring.
- **Single agent, no long-term memory.** No multi-agent swarms, skills,
  memory layers, plan/act modes, checkpoints, git auto-commit, or repo maps
  — those are v1.0 scope. Sessions are the only persistence.

**Not included (by design, per spec §3.19)**: exploit chaining, zero-day
discovery, autonomous exploitation, malware generation, and auth bypass.
Synth's security features are defensive only.

---

## Project layout

```
src/synth/
  cli.py        Argument parsing, wiring, exit codes
  config.py     ~/.synth/config.toml loading + env-var key resolution
  constants.py  Every hardcoded value, documented
  llm.py        LiteLLM client: retries, streaming, tool-call parsing
  agent.py      The ReAct loop
  tools.py      read_file / write_file / bash + registry + path guard
  session.py    SQLite session store
  prompts.py    System prompt
tests/
  test_tools.py Tool and registry tests
```

## License

MIT (see `LICENSE`) — or whatever the repo's license file says.

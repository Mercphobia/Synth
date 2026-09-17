"""Named constants for Synth.

Every value that could be hardcoded lives here. Each constant documents why
it has that value so future readers don't have to guess.
"""

from __future__ import annotations

# --- File tool limits ---

# 1 MB safety limit: matches spec 7 [tools] max_file_size. Stops the agent
# from loading binary blobs or huge logs into context.
MAX_FILE_SIZE = 1_000_000

# 10 KB tool output cap: matches spec 7 [tools] max_output_size. Keeps one
# verbose command from eating the whole context window.
MAX_OUTPUT_SIZE = 10_000

# --- ReAct loop ---

# spec 8.2: max_iterations default. 20 steps is enough for read→edit→test
# cycles without letting a confused model loop forever.
MAX_ITERATIONS = 20

# --- bash tool ---

# spec 7.3: 30s timeout. Long enough for test suites, short enough that a
# hung command doesn't freeze the agent indefinitely.
BASH_TIMEOUT = 30

# --- LLM retries (spec 11.3) ---

# Rate-limit retries with exponential backoff: 1s, 2s, 4s.
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_BACKOFF_BASE = 1.0  # seconds; backoff = base * 2**attempt

# Network-error retries (fewer than rate-limit: transient nets recover fast).
NETWORK_RETRIES = 2

# --- Session store ---

# spec 10.2: list_sessions default. 20 recent sessions fits one terminal page.
SESSION_LIST_LIMIT = 20

# --- Config bootstrap ---

# First-run config path. Written when no config exists (spec section 7).
CONFIG_DIR = ".synth"
CONFIG_FILENAME = "config.toml"
SESSION_DB_FILENAME = "sessions.db"

# Model chosen for the default config. Must exist in the provider's live
# catalog — if the ID goes stale, the LLM error path (llm.py) reports it
# instead of silently failing.
DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_API_KEY_ENV_ANTHROPIC = "ANTHROPIC_API_KEY"
DEFAULT_API_KEY_ENV_OPENAI = "OPENAI_API_KEY"

# --- Session extras (session_extras.py) ---

# spec #80: default watch polling interval in seconds. 2s balances
# responsiveness against CPU cost on battery-constrained devices.
WATCH_INTERVAL_DEFAULT = 2.0

# spec #127: default cap on fuzzy-search result count. 10 fits one screen.
FUZZY_LIMIT_DEFAULT = 10

# spec #73: character cap for timeline message previews. 80 chars is one
# terminal line at standard width.
TIMELINE_PREVIEW_CHARS = 80

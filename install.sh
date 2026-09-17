#!/bin/bash
# Synth CLI installer
# Usage: curl -fsSL https://raw.githubusercontent.com/Mercphobia/Synth/main/install.sh | sh

set -euo pipefail

SYNTH_HOME="${HOME}/.synth-dist"
SYNTH_BIN="${SYNTH_HOME}/bin"

# Check Python version
if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 not found" >&2
    exit 1
fi

if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)' >/dev/null 2>&1; then
    echo "Error: Python 3.12+ required" >&2
    exit 1
fi

# Create directories
mkdir -p "${SYNTH_HOME}" "${SYNTH_BIN}"

# Clone or use local repo
if [ -f "pyproject.toml" ] && [ -d "src/synth" ]; then
    # Running from inside repo
    cp -r . "${SYNTH_HOME}/synth"
else
    # Clone from GitHub
    if command -v git >/dev/null 2>&1; then
        git clone --depth 1 https://github.com/Mercphobia/Synth.git "${SYNTH_HOME}/synth"
    else
        echo "Error: git not found, and not running from repo directory" >&2
        exit 1
    fi
fi

# Create virtual environment
cd "${SYNTH_HOME}/synth"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install .

# Create symlink
ln -sf "${SYNTH_HOME}/synth/.venv/bin/synth" "${SYNTH_BIN}/synth"

# Add to PATH if possible
if [[ ":${PATH}:" != *":${SYNTH_BIN}:"* ]]; then
    echo "Add to PATH: export PATH=\"\${PATH}:${SYNTH_BIN}\"" >&2
    echo "Then run: synth --help" >&2
else
    echo "Synth installed! Run: synth --help" >&2
fi
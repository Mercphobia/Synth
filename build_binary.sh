#!/bin/bash
# Build synth as a single binary using PyInstaller
# Usage: ./build_binary.sh [output_name]

set -euo pipefail

OUTPUT_NAME="${1:-synth}"

# Check if PyInstaller is installed
if ! command -v pyinstaller >/dev/null 2>&1; then
    echo "Error: PyInstaller not found"
    echo "Install with: pip install pyinstaller"
    exit 1
fi

# Build the binary
echo "Building ${OUTPUT_NAME}..."
pyinstaller \
    --onefile \
    --name "${OUTPUT_NAME}" \
    --add-data "src/synth:src/synth" \
    --hidden-import synth.cli \
    --hidden-import synth.agent \
    --hidden-import synth.llm \
    --hidden-import synth.tools \
    --hidden-import synth.session \
    --hidden-import synth.config \
    --hidden-import synth.prompts \
    --hidden-import synth.constants \
    --clean \
    src/synth/cli.py

echo "Build complete: dist/${OUTPUT_NAME}"
echo "Binary size: $(du -h "dist/${OUTPUT_NAME}" | cut -f1)"
echo "Test with: ./dist/${OUTPUT_NAME} --version"

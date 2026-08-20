#!/usr/bin/env bash
# Builds the environment inside this folder: a standalone Python plus every
# library, in ./.venv. Nothing is installed system-wide.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv python install 3.12
uv sync --python 3.12

echo
echo "Done. Opening the worker dashboard..."
uv run peerpixel dashboard

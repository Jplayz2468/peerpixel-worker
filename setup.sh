#!/usr/bin/env bash
# Build the environment. Everything ends up inside this folder: the Python
# interpreter, the libraries, and the source you are reading. Nothing touches
# the system Python and nothing is hidden in a bundle.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (fetches a standalone Python, no system changes)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# uv downloads a relocatable CPython rather than using whatever is on the box,
# so every install is the same interpreter and the same wheels.
uv python install 3.12
uv sync --python 3.12

echo
echo "Done. The environment is in ./.venv — interpreter and all."
echo
echo "  node bin/peerpixel.mjs pair <CODE>"
echo "  node bin/peerpixel.mjs bench"
echo "  node bin/peerpixel.mjs run"

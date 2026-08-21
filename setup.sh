#!/usr/bin/env bash
# Builds the environment inside this folder: a standalone Python plus every
# library, in ./.venv. Nothing is installed system-wide.
set -euo pipefail
cd "$(dirname "$0")"

# See the note in update.sh: uv's installer does not reliably land on PATH for a
# shell that is already running, so look where it actually puts itself.
ensure_uv() {
  UV="$(command -v uv 2>/dev/null || true)"
  [ -n "$UV" ] && return 0
  for candidate in \
    "${UV_INSTALL_DIR:-}/uv" \
    "${XDG_BIN_HOME:-$HOME/.local/bin}/uv" \
    "$HOME/.local/bin/uv" \
    "$HOME/.cargo/bin/uv" \
    /opt/homebrew/bin/uv \
    /usr/local/bin/uv
  do
    if [ -x "$candidate" ]; then UV="$candidate"; return 0; fi
  done
  return 1
}

if ! ensure_uv; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  if ! ensure_uv; then
    echo
    echo "uv installed but could not be found afterwards."
    echo "Install it yourself and run this again: https://docs.astral.sh/uv/"
    exit 1
  fi
fi

"$UV" python install 3.12
"$UV" sync --python 3.12

echo
echo "Done. Opening the worker dashboard..."
exec "$UV" run peerpixel dashboard

#!/usr/bin/env bash
# Pulls the latest worker and restarts it. Safe to run any time.
set -euo pipefail
cd "$(dirname "$0")"

# uv installs itself to ~/.local/bin, which is not on PATH for a shell that was
# already open when setup ran -- and never is for a fresh login shell unless the
# installer got to edit a profile it could find. Looking in the handful of
# places it actually lives is the difference between this working and a bare
# "uv: command not found" that reads like the update itself is broken.
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
  echo "uv is not installed. Installing it now..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  if ! ensure_uv; then
    echo
    echo "uv still could not be found after installing it."
    echo "Run ./setup.sh instead, or install uv yourself: https://docs.astral.sh/uv/"
    exit 1
  fi
fi

git pull --ff-only
"$UV" sync --python 3.12
exec "$UV" run peerpixel dashboard

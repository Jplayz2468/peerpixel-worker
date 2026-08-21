#!/usr/bin/env bash
# Everything that has to happen before the app can draw its own window.
#
# There is exactly one thing the app cannot install for you, and it is the
# interpreter the app runs on. So this script fetches uv, has uv fetch a
# standalone Python, and hands over. It is the only part of PeerPixel that
# happens in a console, it usually takes half a minute, and after the first run
# it takes none at all because both are already there.
#
# It still draws a bar while it waits. The rule does not have an exception for
# the bit before the app starts -- that is the bit somebody is most likely to
# think has hung.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
LOG="$ROOT/.peerpixel-setup.log"
: > "$LOG"

BOLD=$'\033[1m'; DIM=$'\033[2m'; OFF=$'\033[0m'; AMBER=$'\033[38;5;179m'
[ -t 1 ] || { BOLD=""; DIM=""; OFF=""; AMBER=""; }

banner() {
  printf '\n  %s%sPeerPixel%s\n' "$BOLD" "$AMBER" "$OFF"
  printf '  %sgetting this machine ready. it only happens once.%s\n\n' "$DIM" "$OFF"
}

# A bar with the same manners as the ones in the app: it moves from the first
# tick, it keeps moving when the estimate turns out wrong, and it never claims
# to be finished before the thing it is measuring is.
draw() { # fraction label eta
  local width=32 filled
  filled=$(awk -v f="$1" -v w="$width" 'BEGIN{printf "%d", (f*w)+0.5}')
  local done_bar empty_bar
  done_bar=$(printf '%*s' "$filled" '' | tr ' ' '#')
  empty_bar=$(printf '%*s' "$((width - filled))" '' | tr ' ' '.')
  printf '\r  %-26.26s [%s%s] %3d%%  %-14s' "$2" "$done_bar" "$empty_bar" \
    "$(awk -v f="$1" 'BEGIN{printf "%d", f*100}')" "$3"
}

step() { # label estimate_seconds -- command...
  local label="$1" estimate="$2"; shift 3
  if [ ! -t 1 ]; then
    printf '  %s...\n' "$label"
    "$@" >>"$LOG" 2>&1
    return $?
  fi
  "$@" >>"$LOG" 2>&1 &
  local pid=$! start=$SECONDS elapsed fraction eta
  while kill -0 "$pid" 2>/dev/null; do
    elapsed=$((SECONDS - start))
    # See creep() in peerpixel/progress.py: constant speed to the estimate,
    # then ever slower, approaching but never reaching the end.
    read -r fraction eta < <(awk -v t="$elapsed" -v e="$estimate" 'BEGIN{
      if (t <= e) { p = 0.9 * t / e; r = e - t }
      else { o = (t - e) / e; p = 0.995 - 0.095 / (1 + o); r = t * (0.995 - p) / p }
      printf "%.4f %d", p, r }')
    draw "$fraction" "$label" "$([ "$eta" -gt 0 ] && echo "${eta}s left" || echo "almost there")"
    sleep 0.25
  done
  wait "$pid"; local code=$?
  draw 1 "$label" "done"
  printf '\n'
  return $code
}

give_up() {
  printf '\n\n  %sThat did not work.%s\n' "$BOLD" "$OFF"
  printf '  %s\n\n' "$1"
  printf '  The last few lines of %s:\n\n' "$LOG"
  tail -n 12 "$LOG" | sed 's/^/    /'
  printf '\n'
  [ -t 0 ] && { printf '  Press return to close.'; read -r _; }
  exit 1
}

# uv installs itself to ~/.local/bin, which is not on the PATH of a shell that
# was already open -- and never is for a fresh login shell unless its installer
# found a profile to edit. Looking where it actually lives is the difference
# between this working and a bare "uv: command not found" that reads like
# PeerPixel itself is broken.
find_uv() {
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
    [ -x "$candidate" ] && { UV="$candidate"; return 0; }
  done
  return 1
}

banner

if ! find_uv; then
  step "Fetching the installer" 25 -- \
    bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh' \
    || give_up "Could not download uv. Check the connection and try again."
  find_uv || give_up "uv installed but could not be found. Install it yourself: https://docs.astral.sh/uv/"
fi

if ! "$UV" python find 3.12 >/dev/null 2>&1; then
  step "Fetching Python" 35 -- "$UV" python install 3.12 \
    || give_up "Could not install Python 3.12."
fi

if [ "${PEERPIXEL_COMMAND:-}" = "" ]; then
  printf '\n'
fi

# --no-project on purpose. This interpreter only has to be able to *start*
# PeerPixel; the rendering libraries go into .venv, installed by PeerPixel
# itself with a progress bar on it, and it moves onto that interpreter the
# moment they are there. See runtime.use_venv.
export PEERPIXEL_UV="$UV"
exec "$UV" run --no-project --python 3.12 python -m peerpixel ${PEERPIXEL_COMMAND:+"$PEERPIXEL_COMMAND"} "$@"

#!/usr/bin/env bash
# Kept for muscle memory. Updating is a button in the app; this is the same
# thing from a terminal: fetch the newer worker and install it.
set -euo pipefail
cd "$(dirname "$0")"
PEERPIXEL_COMMAND=update exec ./launch/bootstrap.sh "$@"

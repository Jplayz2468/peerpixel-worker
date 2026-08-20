#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
git pull --ff-only
uv sync --python 3.12
uv run peerpixel dashboard

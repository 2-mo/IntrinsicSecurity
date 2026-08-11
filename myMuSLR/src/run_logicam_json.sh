#!/usr/bin/env bash

set -Eeuo pipefail

WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$WORKSPACE/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing Python environment: $PYTHON" >&2
  exit 1
fi

exec "$PYTHON" "$WORKSPACE/src/run_logicam_json.py" "$@"

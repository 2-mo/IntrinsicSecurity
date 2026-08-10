#!/usr/bin/env bash

set -Eeuo pipefail

WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$WORKSPACE/.env"
PYTHON="$WORKSPACE/.venv/bin/python"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing configuration: $ENV_FILE" >&2
  echo "Create it with: cp $WORKSPACE/.env.example $ENV_FILE" >&2
  exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing Python environment: $PYTHON" >&2
  echo "See $WORKSPACE/README.md for environment setup." >&2
  exit 1
fi

# Export .env values so they take precedence when run.py loads its defaults.
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

OUTPUT_DIR="${MUSLR_OUTPUT_DIR:-/data1/nemo/my_projects/IntrinsicSecurity/outputs}"
BASELINE_MAX_TOKENS="${MUSLR_BASELINE_MAX_TOKENS:-2048}"
LOGICAM_MAX_TOKENS="${MUSLR_LOGICAM_MAX_TOKENS:-4096}"
mkdir -p "$OUTPUT_DIR"

run_method() {
  local method="$1"
  local max_tokens="$2"
  local output="$OUTPUT_DIR/muslr_${method}.jsonl"
  local log="$OUTPUT_DIR/muslr_${method}.log"

  echo "[$(date '+%F %T')] Starting $method"
  echo "Output: $output"
  echo "Log:    $log"
  echo "Max tokens: $max_tokens"

  PYTHONUNBUFFERED=1 "$PYTHON" -u "$WORKSPACE/run.py" \
    --method "$method" \
    --max-tokens "$max_tokens" \
    --output "$output" 2>&1 | tee -a "$log"

  echo "[$(date '+%F %T')] Finished $method"
}

echo "MuSLR sequential run: baseline -> LogiCAM"
echo "Workspace: $WORKSPACE"
echo "Outputs:   $OUTPUT_DIR"

run_method baseline "$BASELINE_MAX_TOKENS"
run_method logicam "$LOGICAM_MAX_TOKENS"

echo "[$(date '+%F %T')] All MuSLR runs finished."

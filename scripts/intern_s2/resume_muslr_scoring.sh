#!/usr/bin/env bash
# Score saved predictions without starting the model service or rerunning inference.
set -euo pipefail
cd /data4/r1/workspace/cwork/projects/IntrinsicSecurity
output=outputs/intern_s2
mkdir -p "$output"
exec 9>"$output/muslr_scoring.lock"
flock -n 9 || { echo 'MuSLR scoring is already running'; exit 1; }
exec > >(tee -a "$output/muslr_scoring.log") 2>&1

unset ALL_PROXY all_proxy
export NO_PROXY=127.0.0.1,localhost
export no_proxy=$NO_PROXY
export CUDA_VISIBLE_DEVICES=""
# Both scoring models are already cached locally on this host.
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export PYTHONUNBUFFERED=1
export TTY_COMPATIBLE=1

phase_file="$output/muslr.phase"
trap 'code=$?; printf "exit_code=%s at %s\n" "$code" "$(date -Is)" >> "$phase_file"' EXIT
echo "scoring CPU $(date -Is)" > "$phase_file"
args=(--data /data4/r1/workspace/cwork/datasets/MuSLR
      --predictions "$output/muslr/predictions.jsonl" --model Intern-S2-Preview
      --output "$output/muslr/summary.json" --report "$output/muslr/report.md")

# Save answer metrics first so a later semantic metric failure cannot hide them.
.venv/bin/python myMuSLR/evaluate.py "${args[@]}"
.venv/bin/python scripts/intern_s2/collect_results.py
.venv/bin/python myMuSLR/evaluate.py "${args[@]}" --reasoning-metrics --reasoning-device cpu
.venv/bin/python scripts/intern_s2/collect_results.py
echo "complete $(date -Is)" > "$phase_file"
echo "[$(date -Is)] MuSLR scoring complete; no GPU service was started"

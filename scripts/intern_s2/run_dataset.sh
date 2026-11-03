#!/usr/bin/env bash
set -euo pipefail
cd /data4/r1/workspace/cwork/projects/IntrinsicSecurity
bench=${1:?muslr, cot-control or processbench required}
workers=${INTERN_S2_WORKERS:-6}
mkdir -p outputs/intern_s2
exec > >(tee -a "outputs/intern_s2/${bench}.log") 2>&1
export PYTHONUNBUFFERED=1
export NO_PROXY=127.0.0.1,localhost
export no_proxy=$NO_PROXY
phase_file="outputs/intern_s2/${bench}.phase"
trap 'code=$?; printf "exit_code=%s at %s\n" "$code" "$(date -Is)" >> "$phase_file"' EXIT
echo "Waiting for model server on 127.0.0.1:22100" | tee "$phase_file"
.venv/bin/python - <<'PY'
import time, urllib.request
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
start=time.monotonic()
while time.monotonic()-start < 7200:
    try:
        with opener.open('http://127.0.0.1:22100/health',timeout=3) as r:
            if r.status==200: break
    except Exception: pass
    print('Waiting for model server...',flush=True)
    time.sleep(15)
else: raise RuntimeError('Model server was not ready within 2 hours')
PY
echo "smoke $(date -Is)" > "$phase_file"
case "$bench" in
  muslr)
    common=(--data /data4/r1/workspace/cwork/datasets/MuSLR --method baseline
      --batch-backend openai --base-url http://127.0.0.1:22100/v1
      --model Intern-S2-Preview --media-mode data-url --workers "$workers" --max-tokens 8192 --timeout 1800)
    .venv/bin/python myMuSLR/run.py "${common[@]}" --limit 6 --output outputs/intern_s2/muslr_smoke/predictions.jsonl
    .venv/bin/python myMuSLR/evaluate.py --data /data4/r1/workspace/cwork/datasets/MuSLR \
      --predictions outputs/intern_s2/muslr_smoke/predictions.jsonl --selected-only \
      --model Intern-S2-Preview --output outputs/intern_s2/muslr_smoke/summary.json
    echo "full $(date -Is)" > "$phase_file"
    .venv/bin/python myMuSLR/run.py "${common[@]}" --output outputs/intern_s2/muslr/predictions.jsonl
    .venv/bin/python myMuSLR/evaluate.py --data /data4/r1/workspace/cwork/datasets/MuSLR \
      --predictions outputs/intern_s2/muslr/predictions.jsonl --model Intern-S2-Preview \
      --output outputs/intern_s2/muslr/summary.json --report outputs/intern_s2/muslr/report.md \
      --reasoning-metrics --reasoning-device cpu
    ;;
  cot-control)
    .venv/bin/python myCoTControl/run.py --endpoints http://127.0.0.1:22100/v1 --workers "$workers" \
      --output outputs/intern_s2/cot_control_smoke
    echo "full $(date -Is)" > "$phase_file"
    .venv/bin/python myCoTControl/run.py --endpoints http://127.0.0.1:22100/v1 --workers "$workers" --limit 0 \
      --output outputs/intern_s2/cot_control
    ;;
  processbench)
    .venv/bin/python myProcessBench/run.py --limit 4 --workers "$workers" --output outputs/intern_s2/processbench_smoke
    echo "full $(date -Is)" > "$phase_file"
    .venv/bin/python myProcessBench/run.py --workers "$workers" --output outputs/intern_s2/processbench
    ;;
  *) echo "Unknown benchmark: $bench"; exit 2 ;;
esac
.venv/bin/python scripts/intern_s2/collect_results.py
echo "complete $(date -Is)" > "$phase_file"
echo "[$(date -Is)] $bench INFERENCE AND EVALUATION COMPLETE"

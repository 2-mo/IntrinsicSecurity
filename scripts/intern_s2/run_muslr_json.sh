#!/usr/bin/env bash
# The dedicated two-GPU service must already be running in server-json.
set -euo pipefail
cd /data4/r1/workspace/cwork/projects/IntrinsicSecurity
output=outputs/intern_s2/muslr_json_tp2
mkdir -p "$output"
exec 9>"$output/run.lock"
flock -n 9 || { echo 'MuSLR JSON is already running'; exit 1; }
exec > >(tee -a "$output/run.log") 2>&1
unset ALL_PROXY all_proxy
export NO_PROXY=127.0.0.1,localhost
export no_proxy=$NO_PROXY
export PYTHONPATH="$PWD/myMuSLR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TTY_COMPATIBLE=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export INTERN_S2_BASE_URL=http://127.0.0.1:22104/v1
phase_file=outputs/intern_s2/muslr-json.phase
service_stopped=0

stop_service() {
    if [[ $service_stopped == 0 ]]; then
        .venv/bin/python - <<'PY'
import runpy
ns = runpy.run_path('scripts/intern_s2/after_muslr.py')
pane = 'intern-s2-eval:server-json'
if ns['tmux']('display-message', '-p', '-t', pane, '#{pane_dead}') != '1':
    ns['stop_server'](22104, pane=pane)
print('Dedicated TP2 JSON service stopped; its GPUs are released', flush=True)
PY
        service_stopped=1
    fi
}

finish() {
    code=$?
    trap - EXIT
    stop_service || { echo 'Service cleanup failed; inspect server-json'; code=1; }
    printf 'exit_code=%s at %s\n' "$code" "$(date -Is)" >> "$phase_file"
    exit "$code"
}
trap finish EXIT
echo "Waiting for JSON TP2 service $(date -Is)" > "$phase_file"
.venv/bin/python - <<'PY'
import subprocess, time, urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for _ in range(120):
    try:
        with opener.open('http://127.0.0.1:22104/health', timeout=3) as r:
            if r.status == 200:
                break
    except OSError:
        pass
    dead = subprocess.check_output(['pimux', 'tmux', 'display-message', '-p', '-t',
                                    'intern-s2-eval:server-json', '#{pane_dead}'], text=True).strip()
    if dead == '1':
        raise RuntimeError('JSON TP2 service exited; see server_json_tp2.log')
    time.sleep(5)
else:
    raise RuntimeError('JSON TP2 service did not become ready within 10 minutes')
PY

common=(--data /data4/r1/workspace/cwork/datasets/MuSLR
        --template /data4/r1/workspace/cwork/projects/MuSLR/prompts/baseline/baseline_CoT.txt
        --media-root "$output/media" --endpoints "$INTERN_S2_BASE_URL"
        --model Intern-S2-Preview --max-tokens 8192 --workers 4 --pool-concurrency 4
        --timeout 1800 --image-url-mode data)
echo "smoke $(date -Is)" > "$phase_file"
.venv/bin/python -m baseline_json.run "${common[@]}" --ids flickr30k_7760,coco_2031 --limit 2 --output "$output/smoke.jsonl"
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('outputs/intern_s2/muslr_json_tp2')
rows = [json.loads(line) for line in (p/'smoke.jsonl').read_text().splitlines()]
if len(rows) != 2 or any(r['status'] != 'completed' for r in rows):
    raise RuntimeError('JSON smoke test failed; see smoke.jsonl before starting the full run')
for r in rows:
    assert json.loads(r['output']) == r['structured_output']
print('JSON smoke passed for a multiple-choice and a truth-evaluation sample', flush=True)
if not (p/'predictions.jsonl').exists():
    (p/'predictions.jsonl').write_text((p/'smoke.jsonl').read_text())
PY
echo "full $(date -Is)" > "$phase_file"
.venv/bin/python -m baseline_json.run "${common[@]}" --output "$output/predictions.jsonl"
stop_service

echo "scoring CPU $(date -Is)" > "$phase_file"
eval_args=(--data /data4/r1/workspace/cwork/datasets/MuSLR
           --predictions "$output/predictions.jsonl" --model 'Intern-S2-Preview JSON TP2'
           --output "$output/summary.json" --report "$output/report.md")
.venv/bin/python myMuSLR/evaluate.py "${eval_args[@]}"
.venv/bin/python myMuSLR/evaluate.py "${eval_args[@]}" --reasoning-metrics --reasoning-device cpu
echo "complete $(date -Is)" > "$phase_file"
echo "[$(date -Is)] MuSLR JSON inference and evaluation complete"

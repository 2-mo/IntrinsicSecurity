"""Start the JSON experiment when two GPUs are unused and have enough memory."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT / 'outputs/intern_s2/muslr_json_tp2'
PHASE = OUTPUT.parent / 'muslr-json.phase'
SERVER = 'intern-s2-eval:server-json'


def available_gpus():
    busy = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader,nounits'],
        text=True, timeout=10,
    ).splitlines()
    rows = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'],
        text=True, timeout=10,
    ).splitlines()
    available = []
    for row in rows:
        index, uuid, free = [part.strip() for part in row.split(',')]
        if uuid not in busy and int(free) >= 39000:
            available.append(int(index))
    return available


def tmux(*args):
    return subprocess.check_output(['pimux', 'tmux', *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    os.chdir(PROJECT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.plan:
        print(json.dumps({'currently_available_gpus': available_gpus(), 'required_gpus': 2,
                          'min_free_mib_per_gpu': 39000, 'model_dtype': 'bfloat16',
                          'samples': 1093, 'max_tokens': 8192, 'workers': 4,
                          'response_format': 'json_schema',
                          'structured_output_backend': 'outlines',
                          'on_inference_complete': 'stop dedicated service, score on CPU'}, indent=2))
        return
    with (OUTPUT / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if tmux('display-message', '-p', '-t', SERVER, '#{pane_dead}') != '1':
                raise RuntimeError('Dedicated JSON server pane is already active')
            PHASE.write_text(f'queued: waiting for two unused GPUs, each with 39000 MiB free; {datetime.now().astimezone().isoformat()}\n')
            print(PHASE.read_text().strip(), flush=True)
            while True:
                free = available_gpus()
                if len(free) >= 2:
                    devices = free[:2]
                    break
                time.sleep(15)
            deployment = {
                'time': datetime.now().astimezone().isoformat(), 'gpu_ids': devices,
                'tensor_parallel_size': 2, 'dtype': 'bfloat16', 'thinking_enabled': True,
                'endpoint': 'http://127.0.0.1:22104/v1', 'workers': 4,
                'max_tokens': 8192, 'max_model_len': 32768, 'temperature': 0,
                'response_format': 'json_schema', 'schema_version': 'baseline-json-v1',
                'structured_output_backend': 'outlines',
                'fields': ['reasoning', 'final_answer'], 'samples': 1093,
                'on_inference_complete': 'stop service and release GPUs, then evaluate on CPU',
            }
            (OUTPUT / 'deployment.json').write_text(json.dumps(deployment, indent=2) + '\n')
            command = shlex.join([
                'env', '-u', 'ALL_PROXY', '-u', 'all_proxy',
                f'INTERN_S2_GPUS={",".join(map(str, devices))}', 'INTERN_S2_TP=2',
                'INTERN_S2_PORT=22104', 'INTERN_S2_MAX_SEQS=4', 'INTERN_S2_MEMORY=0.95',
                'INTERN_S2_MAX_BATCHED_TOKENS=2048', 'INTERN_S2_SERVER_LOG=server_json_tp2.log',
                'INTERN_S2_STRUCTURED_BACKEND=outlines',
                'bash', 'scripts/intern_s2/serve.sh',
            ])
            print(f'Starting dedicated BF16 service on GPUs {devices}', flush=True)
            tmux('respawn-pane', '-k', '-t', SERVER, command)
            subprocess.run(['bash', 'scripts/intern_s2/run_muslr_json.sh'], check=True)
        except Exception as error:
            with PHASE.open('a') as stream:
                stream.write(f'exit_code=1 at {datetime.now().astimezone().isoformat()}: {type(error).__name__}: {error}\n')
            raise


if __name__ == '__main__':
    main()

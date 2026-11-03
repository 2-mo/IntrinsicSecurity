"""Live progress for the PiMux monitor window; no network or model calls."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2] / 'outputs/intern_s2'


def count(path):
    if not path.exists():
        return 0, 0
    rows = {}
    with path.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
                rows[row['id']] = row
            except (json.JSONDecodeError, KeyError):
                continue
    return len(rows), sum(r.get('status') in ('failed', 'skipped_missing_image') for r in rows.values())


while True:
    print('\033[2J\033[H', end='')
    print(time.strftime('%Y-%m-%d %H:%M:%S'), 'Intern-S2-Preview / all benchmarks\n')
    for bench, directory, total in [('muslr', 'muslr', 1093), ('cot-control', 'cot_control', 2383),
                                     ('processbench', 'processbench', 3400)]:
        phase = ROOT / f'{bench}.phase'
        current = phase.read_text().strip() if phase.exists() else 'not started'
        done, failed = count(ROOT / directory / 'predictions.jsonl')
        smoke, smoke_failed = count(ROOT / f'{directory}_smoke' / 'predictions.jsonl')
        print(f'{bench:14} {done:4}/{total} ({100*done/total:5.1f}%) failures={failed}; smoke={smoke} failures={smoke_failed}')
        print(' ', current)
    print('\nGPU status:')
    subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv'])
    print('\nOpen with: pimux -n intern-s2-eval', flush=True)
    time.sleep(10)

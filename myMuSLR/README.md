# MuSLR local inference

Qwen3-VL-8B experiments for MuSLR using the RIPO OpenAI-compatible endpoint.
The active experiments use JSON Schema structured output; older free-text code is
kept separately for reproducibility.

## Project layout

```text
myMuSLR/
├── .env.example             versioned path and runtime configuration
├── requirements.txt         local `.venv` dependencies
├── README.md
└── src/
    ├── baseline_json/       official baseline prompt + minimal JSON wrapper
    ├── logicam_json/        official LogiCAM prompts + minimal JSON wrapper
    ├── muslr_core/          dataset, images, parsing, checkpoints, RIPO adapter
    ├── evaluation/          metrics, reports, comparisons, paper tables
    ├── result_dashboard/    interactive result browser
    ├── legacy/              archived free-text runners and old dashboard
    ├── scripts/             supported shell entry points
    └── tests/               all project tests
```

Generated predictions and reports are stored outside this repository at
`/data1/nemo/my_projects/IntrinsicSecurity/outputs`.

## Environment and configuration

```bash
cd /data1/nemo/my_projects/IntrinsicSecurity/myMuSLR
source .venv/bin/activate
```

Edit `.env` to change dataset, model endpoint, output paths, token limits, or
worker counts. `.env.example` documents every active option.

## Active JSON experiments

Run or resume Baseline-JSON:

```bash
src/scripts/run_baseline_json.sh
src/scripts/evaluate_baseline_json.sh
```

Run or resume the audited LogiCAM-JSON v2:

```bash
src/scripts/run_logicam_json.sh
src/scripts/evaluate_logicam_json.sh
```

Each runner uses Rich completed/total progress, temperature 0, one request
attempt, incremental JSONL checkpoints, and automatic endpoint discovery.

Start the result dashboard:

```bash
src/scripts/run_dashboard.sh
# http://127.0.0.1:5060
```

Detailed experiment contracts and smoke-test examples are in
[`src/README.md`](src/README.md). The LogiCAM reproduction audit is in
[`src/logicam_json/AUDIT.md`](src/logicam_json/AUDIT.md).

## Evaluation modules

The shell scripts above are the normal entry points. Modules can also be called
directly when custom paths are needed:

```bash
PYTHONPATH=src:. .venv/bin/python -m evaluation.evaluate --help
PYTHONPATH=src:. .venv/bin/python -m evaluation.compare --help
PYTHONPATH=src:. .venv/bin/python -m evaluation.paper_comparison --help
```

## Legacy free-text reproduction

The previous free-text baseline and LogiCAM code remains under `src/legacy/`.
It is not used by the active JSON experiments.

```bash
# Validate both legacy methods
src/scripts/run_legacy_all.sh --dry-run

# Run one legacy method directly
PYTHONPATH=src:. .venv/bin/python -m legacy.free_text_run --method baseline
```

`src/legacy/dashboard.py` is the superseded, path-specific comparison dashboard;
use `src/scripts/run_dashboard.sh` for current results.

## Tests

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest discover -s src/tests -v
```

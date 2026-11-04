# MuSLR local inference

MuSLR experiments through any OpenAI-compatible vision-language endpoint
with JSON Schema support, including vLLM. No RIPO installation is required.
The current runners use JSON Schema responses. Older free-text code is kept in
`src/legacy/` for reproduction.

## What this package contains

`src/baseline_json/` and `src/logicam_json/` contain the two active runners.
`src/muslr_core/` contains shared dataset and inference code. `src/evaluation/`
produces metrics and reports. `src/result_dashboard/` browses results.
`src/scripts/` contains shell entry points; `src/tests/` contains local tests.

This directory is a standalone client/evaluation package. Model serving is a
separate deployment: the host-specific Intern-S2 setup, GPU selection, PiMux
queues, and service shutdown scripts stay in the parent repository's
`scripts/intern_s2/` and are not part of the shareable package.

The package contains no MuSLR data, official prompt files, model weights,
predictions, or Python environment. Obtain the dataset and official prompts
from the [MuSLR project](https://github.com/Aiden0526/MuSLR).

## Setup

From the extracted `myMuSLR` directory, with Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` to set `MUSLR_DATA`, `MUSLR_PROMPT_DIR`, the explicit endpoint URL
(including `/v1`), and served model name. Use absolute paths. `MUSLR_DATA` must
point to a downloaded MuSLR directory containing `metadata.csv` and `images/`.
`MUSLR_PROMPT_DIR` must contain `baseline/baseline_CoT.txt`,
`logicam/prompt.txt`, and `logicam/prompt_fol.txt`. API keys belong in `.env`;
the file is ignored by the parent repository's Git configuration and excluded
from the source export.

The endpoint must support image input and `json_schema` response format.
The client sends the schema unchanged, uses one request attempt, and does
not silently fall back if the endpoint rejects structured output. It accepts
comma-separated endpoints and caps concurrent requests at `pool-concurrency`
per endpoint. `auto` discovery is unsupported. Images use data URLs by default,
so the server needs no access to your local files. `--image-url-mode file`
requires a shared filesystem and a server configured to allow those paths.
Output defaults to `myMuSLR/outputs/`.

## Run

Check local inputs without a model request:

```bash
src/scripts/run_baseline_json.sh --dry-run --limit 1
src/scripts/run_logicam_json.sh --dry-run --limit 1
```

Run and evaluate each method:

```bash
src/scripts/run_baseline_json.sh
src/scripts/evaluate_baseline_json.sh
src/scripts/run_logicam_json.sh
src/scripts/evaluate_logicam_json.sh
```

The runners use temperature 0, one request attempt, incremental JSONL
checkpoints, and resume by default. Set `--output` for a separate experiment;
use `--no-resume` to restart one. Evaluation may download pretrained models
for optional reasoning metrics on first use.

Browse LogiCAM results after running inference:

```bash
src/scripts/run_dashboard.sh
# http://127.0.0.1:5060
```

More details: [experiment contracts](src/README.md) and
[LogiCAM audit](src/logicam_json/AUDIT.md).

## Tests

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest discover -s src/tests -v
```

## Share this package

From the parent IntrinsicSecurity repository:

```bash
python scripts/package_muslr.py
# outputs/share/myMuSLR-standalone.zip
```

The export uses a source-file allowlist. It excludes `.env`, virtual environments,
model weights, datasets, results, and this host's deployment scripts. Recipients
configure their own endpoint and data paths; they do not need access to your
RIPO backend. Inference does not start or stop their model service.

## Historical RIPO experiments

`run.py` runs the archived free-text prompt using OpenAI by default, with data
URLs. It accepts `OPENAI_BASE_URL`, `OPENAI_MODEL`, and `OPENAI_API_KEY`, or
explicit `--base-url`, `--model`, and `--api-key` arguments.
Only the optional `--batch-backend ripo --media-mode local` path loads the
separate adapter in `src/muslr_core/ripo_batch.py`. Historical reproduction
requires a compatible external [RIPO](https://github.com/2-mo/RIPO) checkout
(commit `ed37a8770c522cee0a2d8a4b8b65f0ba6cd3b8bc`) and `--ripo-root`.
The JSON runners, evaluation, dashboard, and default tests do not import RIPO.

# IntrinsicSecurity

Local inference and evaluation for MuSLR, CoT-Control QA, and ProcessBench using
Intern-S2-Preview and an OpenAI-compatible model endpoint.

## Data and runners

- `myMuSLR/`: MuSLR baseline/LogiCAM runner, answer and reasoning metrics, and review dashboard.
- `myCoTControl/`: QA runner using the official prompts and rule graders; baseline and word suppression by default.
- `myProcessBench/`: critic evaluation using the official prompt and harmonic-mean metric.
- `datasets/CoT-Control/`: official public data, evaluation code, license, and download manifest.
- `datasets/ProcessBench/`: all four official JSON splits and download manifest.

MuSLR images and metadata are loaded from `/data4/r1/workspace/cwork/datasets/MuSLR`.
The model weights are read from `/data3/Intern-S2-Preview`.

## Running on this host

The serving environment is separate from the evaluation environment:

```bash
bash scripts/intern_s2/setup.sh
bash scripts/intern_s2/serve.sh
```

`setup.sh` installs vLLM 0.22.1 and Transformers 5.8.1 in `.venv-intern-s2`.
`serve.sh` checks four-GPU communication and serves the BF16 model on port 22100.
Its NCCL settings bypass the PCIe P2P path that stalled initialization on this host.

Use the project `.venv` for evaluation. Each runner's dependencies are listed
in its own `requirements.txt`: `myCoTControl/`, `myProcessBench/`, and `myMuSLR/`.

Run one benchmark, with checkpoint resume and evaluation:

```bash
INTERN_S2_WORKERS=16 bash scripts/intern_s2/run_dataset.sh muslr
# Other supported arguments: processbench, cot-control
```

For a PiMux window:

```bash
pimux tmux new-window -t intern-s2-eval -n muslr \
  'INTERN_S2_WORKERS=16 bash scripts/intern_s2/run_dataset.sh muslr'
pimux -n intern-s2-eval
```

The wrapper runs a small smoke test before full inference. Results and logs go
to `outputs/intern_s2/`, which is ignored by Git. The progress display is
`scripts/intern_s2/status.py`; `collect_results.py` combines completed summaries.

To score existing MuSLR predictions on CPU without starting a GPU service, run
`bash scripts/intern_s2/resume_muslr_scoring.sh`. It uses the cached scoring models
and saves answer metrics before computing the full reasoning metrics.

The separate two-GPU JSON MuSLR experiment uses `baseline_json.run`, with a
`json_schema` response containing `reasoning` and `final_answer`. Its 1,093
predictions, logs, and reports go to `outputs/intern_s2/muslr_json_tp2/`; the old
free-text results are preserved. `scripts/intern_s2/queue_muslr_json.py` waits for
two unused GPUs with at least 39,000 MiB free each, then starts the dedicated
`intern-s2-eval:server-json` service and runs `run_muslr_json.sh`. The job uses
BF16, TP2, four workers, and an 8,192-token output limit. It validates JSON on a
multiple-choice and a truth-evaluation sample using the Outlines backend, which
supports the model's custom slow tokenizer, before running all samples, stops
the service after inference, and scores on CPU. Check availability without
starting the job with `.venv/bin/python scripts/intern_s2/queue_muslr_json.py --plan`.
For vLLM 0.22.1, the JSON service applies `patch_outlines_eos.py` to accept model
stop tokens after a complete JSON value; incomplete JSON remains invalid. Check
this fix with:

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' .venv-intern-s2/bin/python scripts/intern_s2/test_outlines_eos.py -v
```

On this four-GPU host, `scripts/intern_s2/after_muslr.py` waits for the four-GPU
MuSLR job and its evaluation to finish, then starts a text-only service on GPUs
0 and 1. It validates that deployment before resuming CoT-Control and ProcessBench
in sequence with four concurrent requests. Completed checkpoints are preserved,
and a deployment history records which predictions came from the prior TP4 run.
The TP2 service stops after both text benchmarks finish. This queue is visible in
the `queue` PiMux window; it does not interrupt the active MuSLR job.

CoT-Control reports QA accuracy and rule compliance. Agent tasks and external
meta-discussion judging are not included. ProcessBench uses exact first-error
accuracy and no-error accuracy, with their harmonic mean as the official F1.
MuSLR reports answer accuracy, coverage, truncations, domain/symbol/depth groups,
and optional ROUGE-L/BERTScore/ROSCOE reasoning metrics.

## Checks

MuSLR JSON runners and tests use a standalone OpenAI-compatible client and do
not require RIPO. Model deployment remains separate in `scripts/intern_s2/`.
To share the portable MuSLR client/evaluation package, run
`python scripts/package_muslr.py`; it writes `outputs/share/myMuSLR-standalone.zip`.
Recipients set their own endpoint, model, data, and prompt paths following
`myMuSLR/README.md`. The export excludes this host's deployment scripts and results.

CoT-Control QA and ProcessBench also use direct OpenAI-compatible HTTP requests
and require no RIPO. `python scripts/package_text_benchmarks.py` exports two
independent bundles in `outputs/share/`, including their public data and required
official prompts/code. ProcessBench does not import CoT-Control; both bundles
include the small `benchmark_common/` utilities. Setup instructions are in
`myCoTControl/README.md` and `myProcessBench/README.md`. The bundles exclude
credentials, model weights, results, and host deployment scripts. Configure
`OPENAI_BASE_URL`, `OPENAI_MODEL`, and `OPENAI_API_KEY` in the extracted root's
`.env`, or pass explicit CLI arguments. Local model paths are optional provenance.

```bash
PYTHONPATH=myMuSLR/src .venv/bin/python -m unittest discover -s myMuSLR/src/tests -v
.venv/bin/python -m unittest discover -s myCoTControl -p test_run.py -v
.venv/bin/python -m unittest discover -s myProcessBench -p test_run.py -v
```

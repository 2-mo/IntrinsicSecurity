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

Use the project `.venv` for evaluation. CoT-Control and ProcessBench dependencies
are listed in `myCoTControl/requirements.txt`; MuSLR dependencies are in
`myMuSLR/requirements.txt`.

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

CoT-Control reports QA accuracy and rule compliance. Agent tasks and external
meta-discussion judging are not included. ProcessBench uses exact first-error
accuracy and no-error accuracy, with their harmonic mean as the official F1.
MuSLR reports answer accuracy, coverage, truncations, domain/symbol/depth groups,
and optional ROUGE-L/BERTScore/ROSCOE reasoning metrics.

## Checks

MuSLR JSON tests also require the RIPO revision documented in `myMuSLR/README.md`.
Set `RIPO_ROOT` to that checkout before running the test suite.

```bash
PYTHONPATH=myMuSLR/src .venv/bin/python -m unittest discover -s myMuSLR/src/tests -v
.venv/bin/python -m unittest discover -s myCoTControl -p test_run.py -v
.venv/bin/python -m unittest discover -s myProcessBench -p test_run.py -v
```

# MuSLR structured-output experiments

This source tree contains two Qwen3-VL-8B experiments using the RIPO structured
JSON interface demonstrated in `02_pi_engine_text_inference.ipynb`, section 5.

## Layout

```text
src/
├── baseline_json/       official few-shot baseline + minimal JSON contract
├── logicam_json/        LogiCAM formal reasoning + structured JSON contract
├── result_dashboard/    interactive result inspection page
├── scripts/             run, evaluate, and dashboard shell entry points
└── tests/               schema and prompt contract tests
```

`baseline_json` deliberately constrains only `reasoning` and `final_answer`. It
continues to use the official three-example baseline CoT prompt and introduces no
formal notation, rule catalog, candidate consistency gate, or LogiCAM behavior.

`logicam_json` contains visual facts, context formalization, derivation steps,
candidate evaluations, and a schema-constrained final answer. Cross-field checks
are diagnostic warnings only; every parseable `final_answer` is scored.

## Baseline-JSON

Validate configuration without contacting the model:

```bash
cd /data1/nemo/my_projects/IntrinsicSecurity/myMuSLR
src/scripts/run_baseline_json.sh --dry-run
```

Run a separate 10-sample smoke test:

```bash
src/scripts/run_baseline_json.sh \
  --limit 10 \
  --no-resume \
  --output /data1/nemo/my_projects/IntrinsicSecurity/outputs/muslr_baseline_json_smoke.jsonl
```

Run all 1,093 samples with checkpoint resume, then evaluate:

```bash
src/scripts/run_baseline_json.sh
src/scripts/evaluate_baseline_json.sh
```

Each evaluator also writes a compact publication table with `Method`, `Overall`,
`Traffic`, `Science`, `Finance`, and `Others`. All cells use strict E2E accuracy;
`Others` pools every remaining sample before computing the percentage.

## LogiCAM-JSON

```bash
src/scripts/run_logicam_json.sh --dry-run
src/scripts/run_logicam_json.sh
src/scripts/evaluate_logicam_json.sh
```

Recover legacy results that were rejected by the former strict consistency gate:

```bash
PYTHONPATH=src:. .venv/bin/python -m logicam_json.recover_results \
  /absolute/path/to/predictions.jsonl
```

## Dashboard

```bash
src/scripts/run_dashboard.sh
# http://127.0.0.1:5060
```

Choose another result with
`src/scripts/run_dashboard.sh --predictions /absolute/path/to/result.jsonl`.

Both runners use one model request attempt, Rich progress with completed/total,
temperature 0, checkpoint resume, and path/configuration values from `.env`.

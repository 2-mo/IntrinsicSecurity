# LogiCAM-JSON experiment

This directory contains a JSON-constrained LogiCAM variant for MuSLR. It follows RIPO
`02_pi_engine_text_inference.ipynb`, section 5: each request supplies an OpenAI-compatible
`json_schema` response format and parses the response with `PIVLM.parse`.

The experiment is separate from the existing free-text baseline and LogiCAM outputs.

## Structured reasoning contract

Every response contains:

1. `visual_facts`: relevant image observations and formal atoms.
2. `context_formalization`: context premises translated into PL, FOL, or NM notation.
3. `derivation_steps`: at most 12 steps using a predefined rule enum.
4. `candidate_evaluations`: every option evaluated exactly once.
5. `final_answer`: schema-constrained to A-D or True/False/Unknown.

PL, FOL, and NM use separate notation instructions and rule catalogs. The schema bounds all
arrays, which prevents unbounded free-text reconsideration loops.

Cross-field consistency checks are diagnostic only. To match the original LogiCAM evaluation,
every parseable schema-valid `final_answer` is scored; disagreements with intermediate candidate
verdicts are retained in `consistency_warnings` and never suppress the prediction.

## Commands

Run static validation without contacting the model:

```bash
cd /data1/nemo/my_projects/IntrinsicSecurity/myMuSLR
src/run_logicam_json.sh --dry-run
```

Run a separate 10-sample smoke test:

```bash
src/run_logicam_json.sh \
  --limit 10 \
  --no-resume \
  --output /data1/nemo/my_projects/IntrinsicSecurity/outputs/muslr_logicam_json_smoke.jsonl
```

Run all 1,093 samples with checkpoint resume:

```bash
src/run_logicam_json.sh
```

Evaluate the full output:

```bash
src/evaluate_logicam_json.sh
```

Open the live result dashboard (auto-refresh the browser to read the current JSONL):

```bash
src/run_dashboard.sh
# http://127.0.0.1:5060
```

Choose another smoke or full result file with
`src/run_dashboard.sh --predictions /absolute/path/to/result.jsonl`.

The runner defaults to RIPO endpoint auto-discovery, one request attempt, two workers, data-URL
images, temperature 0, and 2,048 output tokens. Configuration lives in the parent `.env`.

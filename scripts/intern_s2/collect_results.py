"""Collect benchmark summaries into one report; explicitly retain pending runs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "outputs/intern_s2"
summaries = {}
for key in ("muslr", "cot_control", "processbench"):
    path = ROOT / key / "summary.json"
    summaries[key] = json.loads(path.read_text()) if path.exists() else None

lines = ["# Intern-S2-Preview benchmark results", "",
         "Model: `/data3/Intern-S2-Preview`. Serving: BF16, tensor parallel 4, thinking enabled.", "",
         "| Benchmark | Evaluation scope | Status |", "| :--- | :--- | :--- |"]
for key, count in [("muslr", 1093), ("cot_control", 2383), ("processbench", 3400)]:
    lines.append(f"| {key} | {count} requests | {'Summary available' if summaries[key] else 'Pending'} |")

if summaries["muslr"]:
    s = summaries["muslr"]
    m = s["overall"]
    lines += ["", "## MuSLR baseline", "",
              f"End-to-end accuracy: {m['accuracy_end_to_end_percent']:.2f}%; answer coverage: {m['coverage_percent']:.2f}%.",
              f"Correct: {m['correct']}/{m['records']}; missing: {m['missing']}; unparsed: {m['unparsed']}.",
              "", "Reasoning quality details and domain/symbol/depth breakdowns are in `muslr/summary.json`."]
if summaries["cot_control"]:
    lines += ["", "## CoT-Control QA", "",
              "| Dataset/mode | N | Accuracy | Reasoning coverage | Rule compliance (all requests) |",
              "| :--- | ---: | ---: | ---: | ---: |"]
    for name, m in summaries["cot_control"]["groups"].items():
        rate = m["rule_compliance_all_requests"]
        rate = f"{100*rate:.2f}%" if rate is not None else "N/A"
        lines.append(f"| {name} | {m['completed']} | {100*m['accuracy']:.2f}% | {100*m['reasoning_coverage']:.2f}% | {rate} |")
    lines += ["", "Rule compliance uses official rule graders. Meta-discussion LLM judging and agent tasks were not run; these are not the paper's full judged scores."]
if summaries["processbench"]:
    lines += ["", "## ProcessBench", "",
              "| Split | N | First-error accuracy | No-error accuracy | F1 |",
              "| :--- | ---: | ---: | ---: | ---: |"]
    for split, m in summaries["processbench"]["groups"].items():
        rate = lambda v: f"{100*v:.2f}%" if v is not None else "N/A"
        lines.append(f"| {split} | {m['completed']} | {rate(m['error_accuracy'])} | {rate(m['correct_accuracy'])} | {rate(m['f1'])} |")
    lines += ["", "F1 is the official harmonic mean of the two accuracies; failed/unparsed requests remain in the denominator."]
lines += ["", "## Generation limits", "",
          "Greedy decoding (temperature 0). MuSLR: 8,192 output tokens; CoT-Control: 25,000; ProcessBench: 16,384 (seed 42).",
          "Requests and raw reasoning are checkpointed. Truncations and failures are reported in each summary.", ""]
ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
(ROOT / "report.md").write_text("\n".join(lines))
print("\n".join(lines))

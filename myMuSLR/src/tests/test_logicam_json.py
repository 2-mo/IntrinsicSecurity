from __future__ import annotations

import unittest

from logicam_json import (
    LOGIC_SPECS,
    SCHEMA_VERSION,
    response_format_for,
    system_prompt_for,
    validate_structured_result,
)
from muslr import Sample


def sample(symbol: str = "pl", choices: bool = True) -> Sample:
    return Sample(
        id="test",
        source_id="test",
        row_index=0,
        image="images/test.jpg",
        domain="Science",
        symbol=symbol,
        depth=3,
        context="If P then Q.",
        question="What follows?",
        choices=["A. P", "B. Q", "C. R", "D. S"] if choices else [],
        answer="B" if choices else "True",
    )


def valid_result(item: Sample) -> dict:
    family = LOGIC_SPECS[item.symbol]["label"]
    candidates = ["A", "B", "C", "D"] if item.choices else ["ARGUMENT"]
    return {
        "schema_version": SCHEMA_VERSION,
        "logic_family": family,
        "task_type": item.task_type,
        "visual_facts": [
            {
                "id": "V1",
                "description": "visible fact",
                "atom": "P",
                "polarity": "positive",
            }
        ],
        "context_formalization": [
            {"id": "C1", "natural_language": "If P then Q", "formula": "P -> Q"}
        ],
        "derivation_steps": [
            {
                "id": "D1",
                "rule": "MODUS_PONENS",
                "premises": ["V1", "C1"],
                "conclusion": "Q",
                "justification": "P and P -> Q entail Q.",
            }
        ],
        "candidate_evaluations": [
            {
                "candidate": candidate,
                "formal_statement": "Q",
                "verdict": (
                    "true"
                    if candidate == "ARGUMENT"
                    else "entailed"
                    if candidate == "B"
                    else "unsupported"
                ),
                "evidence": ["D1"] if candidate in {"B", "ARGUMENT"} else [],
            }
            for candidate in candidates
        ],
        "final_answer": "B" if item.choices else "True",
        "concise_reason": "Q follows by modus ponens.",
    }


class LogiCAMJSONTests(unittest.TestCase):
    def test_multiple_choice_schema_constrains_four_candidates_and_answer(self) -> None:
        item = sample()
        schema = response_format_for(item)["json_schema"]["schema"]
        candidates = schema["properties"]["candidate_evaluations"]
        self.assertEqual(candidates["minItems"], 4)
        self.assertEqual(candidates["maxItems"], 4)
        self.assertEqual(
            schema["properties"]["final_answer"]["enum"], ["A", "B", "C", "D"]
        )

    def test_truth_schema_constrains_truth_labels(self) -> None:
        item = sample(choices=False)
        schema = response_format_for(item)["json_schema"]["schema"]
        self.assertEqual(
            schema["properties"]["final_answer"]["enum"],
            ["True", "False", "Unknown"],
        )
        verdict = schema["properties"]["candidate_evaluations"]["items"]["properties"][
            "verdict"
        ]
        self.assertEqual(verdict["enum"], ["true", "false", "unknown"])

    def test_logic_families_have_distinct_templates(self) -> None:
        self.assertIn("UNIVERSAL_INSTANTIATION", system_prompt_for(sample("fol")))
        self.assertIn("DEFAULT_INFERENCE", system_prompt_for(sample("nm")))
        self.assertNotIn("DEFAULT_INFERENCE", system_prompt_for(sample("pl")))

    def test_valid_result_passes_semantic_validation(self) -> None:
        item = sample()
        validate_structured_result(valid_result(item), item)

    def test_duplicate_candidate_is_rejected(self) -> None:
        item = sample()
        result = valid_result(item)
        result["candidate_evaluations"][3]["candidate"] = "A"
        with self.assertRaisesRegex(ValueError, "candidate_evaluations mismatch"):
            validate_structured_result(result, item)

    def test_final_answer_must_match_entailed_candidate(self) -> None:
        item = sample()
        result = valid_result(item)
        result["final_answer"] = "A"
        with self.assertRaisesRegex(ValueError, "exactly one entailed candidate"):
            validate_structured_result(result, item)


if __name__ == "__main__":
    unittest.main()

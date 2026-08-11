"""Structured LogiCAM prompts, JSON schemas, and result validation."""

from __future__ import annotations

from typing import Any

from muslr import Sample

SCHEMA_VERSION = "logicam-json-v1"


class StructuredConsistencyError(ValueError):
    """The JSON is schema-valid but its reasoning fields disagree."""


COMMON_RULES = (
    "VISUAL_GROUNDING",
    "COMMONSENSE_BRIDGE",
    "MODUS_PONENS",
    "MODUS_TOLLENS",
    "HYPOTHETICAL_SYLLOGISM",
    "DISJUNCTIVE_SYLLOGISM",
    "CONJUNCTION_INTRODUCTION",
    "CONJUNCTION_ELIMINATION",
    "CONTRADICTION_CHECK",
    "OPTION_EVALUATION",
)

LOGIC_SPECS: dict[str, dict[str, Any]] = {
    "pl": {
        "label": "PL",
        "notation": (
            "Use uppercase atoms P1, P2, ... with NOT, AND, OR, and ->. "
            "Example: (P1 AND P2) -> P3. Never treat missing evidence as NOT P."
        ),
        "rules": COMMON_RULES,
    },
    "fol": {
        "label": "FOL",
        "notation": (
            "Use Predicate(entity), FORALL x: expression, EXISTS x: expression, "
            "NOT, AND, OR, and ->. Keep entity names stable and do not invent witnesses."
        ),
        "rules": COMMON_RULES
        + (
            "UNIVERSAL_INSTANTIATION",
            "EXISTENTIAL_INSTANTIATION",
            "EXISTENTIAL_GENERALIZATION",
            "QUANTIFIER_NEGATION",
        ),
    },
    "nm": {
        "label": "NM",
        "notation": (
            "Use FACT P, DEFAULT P => Q, EXCEPTION E BLOCKS (P => Q), and NOT. "
            "Apply a default only when its prerequisite holds and no explicit exception holds. "
            "Never use closed-world negation."
        ),
        "rules": COMMON_RULES
        + (
            "DEFAULT_INFERENCE",
            "EXCEPTION_CHECK",
            "DEFAULT_BLOCKING",
            "SPECIFICITY_RESOLUTION",
        ),
    },
}

RETURN_EXAMPLE = {
    "schema_version": "",
    "logic_family": "",
    "task_type": "",
    "visual_facts": [],
    "context_formalization": [],
    "derivation_steps": [],
    "candidate_evaluations": [],
    "final_answer": "",
    "concise_reason": "",
}


def _spec(symbol: str) -> dict[str, Any]:
    try:
        return LOGIC_SPECS[symbol.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported MuSLR logic family: {symbol!r}") from error


def answer_values(sample: Sample) -> list[str]:
    if sample.task_type == "multiple_choice":
        return ["A", "B", "C", "D"]
    return ["True", "False", "Unknown"]


def candidate_values(sample: Sample) -> list[str]:
    if sample.task_type == "multiple_choice":
        return ["A", "B", "C", "D"]
    return ["ARGUMENT"]


def verdict_values(sample: Sample) -> list[str]:
    if sample.task_type == "multiple_choice":
        return ["entailed", "contradicted", "unsupported"]
    return ["true", "false", "unknown"]


def response_format_for(sample: Sample) -> dict[str, Any]:
    """Build the per-task JSON Schema passed to RIPO/vLLM."""
    spec = _spec(sample.symbol)
    candidates = candidate_values(sample)
    candidate_count = len(candidates)
    schema = {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string", "enum": [SCHEMA_VERSION]},
            "logic_family": {"type": "string", "enum": [spec["label"]]},
            "task_type": {"type": "string", "enum": [sample.task_type]},
            "visual_facts": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "description": {"type": "string"},
                        "atom": {"type": "string"},
                        "polarity": {
                            "type": "string",
                            "enum": ["positive", "negative"],
                        },
                    },
                    "required": ["id", "description", "atom", "polarity"],
                    "additionalProperties": False,
                },
            },
            "context_formalization": {
                "type": "array",
                "minItems": 1,
                "maxItems": 12,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "natural_language": {"type": "string"},
                        "formula": {"type": "string"},
                    },
                    "required": ["id", "natural_language", "formula"],
                    "additionalProperties": False,
                },
            },
            "derivation_steps": {
                "type": "array",
                "minItems": 1,
                "maxItems": 12,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "rule": {"type": "string", "enum": list(spec["rules"])},
                        "premises": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 6,
                            "items": {"type": "string"},
                        },
                        "conclusion": {"type": "string"},
                        "justification": {"type": "string"},
                    },
                    "required": [
                        "id",
                        "rule",
                        "premises",
                        "conclusion",
                        "justification",
                    ],
                    "additionalProperties": False,
                },
            },
            "candidate_evaluations": {
                "type": "array",
                "minItems": candidate_count,
                "maxItems": candidate_count,
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate": {"type": "string", "enum": candidates},
                        "formal_statement": {"type": "string"},
                        "verdict": {
                            "type": "string",
                            "enum": verdict_values(sample),
                        },
                        "evidence": {
                            "type": "array",
                            "maxItems": 6,
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "candidate",
                        "formal_statement",
                        "verdict",
                        "evidence",
                    ],
                    "additionalProperties": False,
                },
            },
            "final_answer": {"type": "string", "enum": answer_values(sample)},
            "concise_reason": {"type": "string"},
        },
        "required": list(RETURN_EXAMPLE),
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"muslr_{sample.symbol}_{sample.task_type}",
            "schema": schema,
        },
    }


def system_prompt_for(sample: Sample) -> str:
    spec = _spec(sample.symbol)
    rules = ", ".join(spec["rules"])
    return f"""You are LogiCAM-JSON, a disciplined multimodal symbolic reasoner.

Your response is constrained by a JSON Schema. Return only the JSON object.

Logic family: {spec["label"]}
Formal notation: {spec["notation"]}
Allowed derivation rules: {rules}

Reasoning policy:
1. Record only visible facts that matter to the question. A missing visual detail is not a negative fact.
2. Formalize the supplied context without strengthening, reversing, or inventing premises.
3. Derive each new conclusion with one allowed rule and cite premise IDs.
4. Keep at most 12 derivation steps. Do not restart, reconsider, or repeat a step.
5. For multiple choice, evaluate A-D exactly once. Mark exactly one option entailed and use that same letter as final_answer.
6. For truth evaluation, candidate_evaluations has one ARGUMENT item. Its verdict and final_answer must match exactly: true/True when the argument is derived, false/False when its explicit negation is derived, otherwise unknown/Unknown.
7. Keep every description and justification to one concise sentence.
"""


def user_prompt_for(sample: Sample) -> str:
    if sample.choices:
        candidates = "\n".join(sample.choices)
    else:
        candidates = "ARGUMENT TO EVALUATE:\n" + sample.question
    return f"""Analyze this MuSLR instance with the attached image.

CONTEXT PREMISES:
{sample.context}

QUESTION:
{sample.question}

CANDIDATES:
{candidates}

Build the JSON result in this order: visual facts, context formalization, derivation steps, candidate evaluations, final answer. Do not include Markdown or text outside JSON.
"""


def validate_structured_result(result: dict[str, Any], sample: Sample) -> None:
    """Perform semantic checks beyond provider-side JSON Schema decoding."""
    spec = _spec(sample.symbol)
    if set(result) != set(RETURN_EXAMPLE):
        missing = sorted(set(RETURN_EXAMPLE) - set(result))
        extra = sorted(set(result) - set(RETURN_EXAMPLE))
        raise StructuredConsistencyError(
            f"Top-level JSON keys mismatch; missing={missing}, extra={extra}"
        )
    if result["schema_version"] != SCHEMA_VERSION:
        raise StructuredConsistencyError(
            f"Unexpected schema_version: {result['schema_version']!r}"
        )
    if result["logic_family"] != spec["label"]:
        raise StructuredConsistencyError(
            f"Unexpected logic_family: {result['logic_family']!r}"
        )
    if result["task_type"] != sample.task_type:
        raise StructuredConsistencyError(
            f"Unexpected task_type: {result['task_type']!r}"
        )
    if result["final_answer"] not in answer_values(sample):
        raise StructuredConsistencyError(
            f"Invalid final_answer: {result['final_answer']!r}"
        )

    expected_candidates = set(candidate_values(sample))
    evaluations = result["candidate_evaluations"]
    actual_candidates = [item.get("candidate") for item in evaluations]
    if len(actual_candidates) != len(expected_candidates):
        raise StructuredConsistencyError("candidate_evaluations has the wrong length")
    if set(actual_candidates) != expected_candidates:
        raise StructuredConsistencyError(
            f"candidate_evaluations mismatch: expected={sorted(expected_candidates)}, "
            f"actual={actual_candidates}"
        )

    if sample.task_type == "multiple_choice":
        entailed = [
            str(item["candidate"])
            for item in evaluations
            if item.get("verdict") == "entailed"
        ]
        if entailed != [result["final_answer"]]:
            raise StructuredConsistencyError(
                "Multiple-choice result must have exactly one entailed candidate "
                "matching final_answer"
            )
    else:
        expected_final = {
            "true": "True",
            "false": "False",
            "unknown": "Unknown",
        }.get(str(evaluations[0].get("verdict")))
        if expected_final != result["final_answer"]:
            raise StructuredConsistencyError(
                "Truth-evaluation verdict must agree with final_answer"
            )

    steps = result["derivation_steps"]
    if not 1 <= len(steps) <= 12:
        raise StructuredConsistencyError("derivation_steps must contain 1-12 items")
    invalid_rules = sorted(
        {str(step.get("rule")) for step in steps} - set(spec["rules"])
    )
    if invalid_rules:
        raise StructuredConsistencyError(f"Invalid derivation rules: {invalid_rules}")

    for field, minimum, maximum in (
        ("visual_facts", 1, 8),
        ("context_formalization", 1, 12),
    ):
        value = result[field]
        if not isinstance(value, list) or not minimum <= len(value) <= maximum:
            raise StructuredConsistencyError(
                f"{field} must contain {minimum}-{maximum} items"
            )


__all__ = [
    "LOGIC_SPECS",
    "RETURN_EXAMPLE",
    "SCHEMA_VERSION",
    "StructuredConsistencyError",
    "answer_values",
    "candidate_values",
    "response_format_for",
    "system_prompt_for",
    "user_prompt_for",
    "validate_structured_result",
    "verdict_values",
]

"""MuSLR reference-based ROSCOE functions, preserved from the workspace scorer."""
from __future__ import annotations
import gc
import re
from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Iterable
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

@dataclass
class Sample:
    id: str
    context: str
    reasoning: dict[str, str]


FINAL_ANSWER_LINE_RE = re.compile(
    r"(?im)^\s*(?:final\s+)?answer\s*:\s*\{?\s*"
    r"(?:true|false|unknown|[a-d])\s*\}?\s*\.?\s*$"
)


NUMBERED_STEP_RE = re.compile(r"^\s*(?:step\s*)?\d+\s*[.):\-]\s*", re.IGNORECASE)


CONTEXT_SPLIT_RE = re.compile(r",\s+(?=(?:if|either|a|an|the|there|no)\b)", re.IGNORECASE)


def reference_steps(sample: Sample) -> list[str]:
    return [step for step in sample.reasoning.values() if step.strip()]


def generated_steps(output: str) -> list[str]:
    """Remove the final answer and split a free-form CoT into stable paragraph steps."""
    text = FINAL_ANSWER_LINE_RE.sub("", output).strip()
    if not text:
        return []

    steps: list[str] = []
    block: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if block:
                steps.append(" ".join(block))
                block = []
            continue
        numbered = bool(NUMBERED_STEP_RE.match(line))
        line = NUMBERED_STEP_RE.sub("", line).strip()
        if numbered and block:
            steps.append(" ".join(block))
            block = []
        if line:
            block.append(line)
    if block:
        steps.append(" ".join(block))
    return steps


def context_steps(sample: Sample) -> list[str]:
    steps = [part.strip() for part in CONTEXT_SPLIT_RE.split(sample.context) if part.strip()]
    return steps or [sample.context or "No context provided."]


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    free_memory = [torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count())]
    return f"cuda:{max(range(len(free_memory)), key=free_memory.__getitem__)}"


def reasoning_pairs(
    samples_by_id: dict[str, Sample],
    predictions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for sample in samples_by_id.values():
        record = predictions.get(sample.id)
        output = (record.get("reasoning") or record.get("output")) if record else None
        references = reference_steps(sample)
        hypotheses = generated_steps(output) if isinstance(output, str) else []
        if not references or not hypotheses:
            continue
        pairs.append(
            {
                "id": sample.id,
                "hypothesis_steps": hypotheses,
                "reference_steps": references,
                "context_steps": context_steps(sample),
                "hypothesis": "\n".join(hypotheses),
                "reference": "\n".join(references),
            }
        )
    return pairs


def _scaled_cosine(left: Any, right: Any) -> float:
    return (1.0 + float(left @ right)) / 2.0


def _alignment(reference: list[Any], hypothesis: list[Any]) -> list[float]:
    if not reference:
        return []
    return [max(_scaled_cosine(item, candidate) for item in reference) for candidate in hypothesis]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return fmean(values) if values else 0.0


def _embed_texts(
    texts: list[str],
    *,
    model_name: str,
    device: str,
    batch_size: int,
    console: Console,
) -> list[Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    embeddings: list[Any] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("ROSCOE embeddings", total=len(texts))
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                hidden = model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.extend(pooled.float().cpu().numpy())
            progress.advance(task, len(batch))
    del model
    gc.collect()
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.empty_cache()
    return embeddings


def compute_roscoe(
    pairs: list[dict[str, Any]],
    *,
    model_name: str,
    device: str,
    batch_size: int,
    console: Console,
) -> dict[str, Any]:
    """Compute the reference-based ROSCOE score and its semantic components."""
    locations: dict[tuple[int, str], tuple[int, int]] = {}
    texts: list[str] = []
    for pair_index, pair in enumerate(pairs):
        for key in ("hypothesis_steps", "reference_steps", "context_steps"):
            start = len(texts)
            texts.extend(pair[key])
            locations[(pair_index, key)] = (start, len(texts))
        pair["context"] = "\n".join(pair["context_steps"])
        for key in ("hypothesis", "reference", "context"):
            start = len(texts)
            texts.append(pair[key])
            locations[(pair_index, key)] = (start, len(texts))

    embedded = _embed_texts(
        texts,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        console=console,
    )
    components: dict[str, list[float]] = defaultdict(list)
    for pair_index, _ in enumerate(pairs):
        vectors: dict[str, list[Any]] = {}
        for key in (
            "hypothesis_steps",
            "reference_steps",
            "context_steps",
            "hypothesis",
            "reference",
            "context",
        ):
            start, end = locations[(pair_index, key)]
            vectors[key] = embedded[start:end]

        hypothesis = vectors["hypothesis_steps"]
        reference = vectors["reference_steps"]
        context = vectors["context_steps"]
        hypothesis_to_reference = _alignment(reference, hypothesis)
        reference_to_hypothesis = _alignment(hypothesis, reference)
        hypothesis_to_context = _alignment(context, hypothesis)
        context_to_hypothesis = _alignment(hypothesis, context)
        reference_to_context = _alignment(context, reference)

        components["reasoning_alignment"].append(_mean(hypothesis_to_reference))
        components["faithfulness"].append(_mean(hypothesis_to_context))
        components["informativeness_step"].append(
            (_mean(hypothesis_to_context) + _mean(context_to_hypothesis)) / 2.0
        )
        components["informativeness_chain"].append(
            _scaled_cosine(vectors["hypothesis"][0], vectors["context"][0])
        )
        components["external_hallucination"].append(
            1.0
            - max(
                (1.0 - ref_score) * (1.0 - context_score)
                for ref_score, context_score in zip(
                    hypothesis_to_reference, hypothesis_to_context, strict=True
                )
            )
        )
        components["redundancy"].append(min(hypothesis_to_reference))
        components["common_sense_error"].append(
            1.0
            - max(
                (1.0 - hypothesis_score) * (1.0 - context_score)
                for hypothesis_score, context_score in zip(
                    reference_to_hypothesis, reference_to_context, strict=True
                )
            )
        )
        components["missing_step"].append(min(reference_to_hypothesis))
        components["semantic_coverage_step"].append(
            1.0 - abs(_mean(reference_to_context) - _mean(hypothesis_to_context))
        )
        components["semantic_coverage_chain"].append(
            _scaled_cosine(vectors["hypothesis"][0], vectors["reference"][0])
        )

    means = {key: fmean(values) for key, values in components.items()}
    return {
        "score": means["reasoning_alignment"],
        "paper_score_mapping": "reasoning_alignment",
        "model": model_name,
        "pooling": "attention-mask mean pooling; L2 normalized",
        "samples": len(pairs),
        "components": means,
        "reproduction_note": (
            "MuSLR publishes one ROSCOE value but no scorer configuration. "
            "This report maps it to the official reference-based reasoning_alignment score."
        ),
    }

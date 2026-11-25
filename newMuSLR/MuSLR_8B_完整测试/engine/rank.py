"""Compare rotated conclusions, their polarity and their conditional likelihood."""

import json
from .backend import messages

TRUTH = ("True", "False", "Unknown")

NORMALIZE = """Rewrite the proposition twice as ordinary English statements with opposite truth conditions. The first must mean exactly the original, the second its logical opposite. Remove double negation, translate logical notation, and preserve names, scope and tense. Do not solve the question or add new facts. Output the statements in the specified fields."""

WORLD = """You are interpreting a visual situation and its short narrative. Imagine how the pictured event unfolds under the stated rules. Use the normal goal, likely action and ordinary consequence when those are implicit. Some premises rule out a branch indirectly: if that branch would imply a visual fact contradicted by the picture, follow the other branch. Compare the concrete alternative outcomes with the scene, including what follows next. Pick the one that best fits the story and the rules. If two opposing outcomes really have equal support, select the undecided option when available. Return only the option letter."""


def structured(**properties):
    schema = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {"name": "propositions", "schema": schema},
    }


def normalize(server, s):
    claim = s.question.split("\n", 1)[-1]
    reply = server.complete(
        [{"role": "system", "content": NORMALIZE}, {"role": "user", "content": claim}],
        max_tokens=320,
        temperature=0,
        response_format=structured(
            original={"type": "string"}, opposite={"type": "string"}
        ),
    )
    try:
        obj = json.loads(reply["message"]["content"])
        if not all(
            isinstance(obj[key], str) and obj[key].strip()
            for key in ("original", "opposite")
        ):
            raise ValueError("Incomplete normalized claim")
    except (TypeError, KeyError, ValueError):
        obj = {"original": claim, "opposite": f"It is false that {claim}"}
    return {"id": s.id, **obj}


def choose(server, s, claims, view, blind=False):
    if s.choices:
        opts = [(chr(65 + k), c[3:]) for k, c in enumerate(s.choices)]
    else:
        opts = list(
            zip(
                TRUTH,
                (
                    claims["original"],
                    claims["opposite"],
                    "Both outcomes remain genuinely open in this situation.",
                ),
            )
        )
    shift = view % len(opts)
    opts = opts[shift:] + opts[:shift]
    display = "\n".join(f"{chr(65 + k)}. {claim}" for k, (_, claim) in enumerate(opts))
    query = f"Which outcome best fits this particular situation?\n{display}"
    letters = [chr(65 + i) for i in range(len(opts))]
    prompt = messages(s, WORLD, query)
    if blind:
        prompt[-1]["content"] = [
            part for part in prompt[-1]["content"] if part["type"] != "image_url"
        ]
    reply = server.complete(
        prompt,
        max_tokens=2,
        temperature=1,
        top_p=1,
        top_k=-1,
        seed=42,
        logprobs=True,
        top_logprobs=12,
        structured_outputs={"choice": letters},
    )
    scores = {
        t["token"]: t["logprob"]
        for t in reply["logprobs"]["content"][0]["top_logprobs"]
    }
    return {
        "id": s.id,
        "view": view,
        "logp": {label: scores[letters[k]] for k, (label, _) in enumerate(opts)},
    }


TERMINAL = """Find the conclusion of the reasoning chain illustrated by the image and text. Establish the pictured facts, use them to resolve conditionals or alternatives, then consider the most plausible downstream consequence in the described situation. Distinguish the terminal conclusion from a prerequisite or a clause that was only mentioned hypothetically. Use ordinary social, physical and causal knowledge when an intermediate link is implicit. Compare all candidate statements, keeping exact entities, polarity and modifiers. Select the best supported overall conclusion. If nothing distinguishes the possible outcomes, report uncertainty when available."""

FALSIFY = """For each candidate account, ask what must be true for it to fit the visual situation and the narrative. Rule out a candidate that reverses an observed fact, contradicts a stated conditional, or adds an unsupported property. Trace the remaining chain through its intermediate events to the best final conclusion. The image can establish facts omitted from the text; ordinary world knowledge can connect these facts to the most likely outcome. Judge the actual meaning of the options."""


def score(servers, s, normalized, v, blind=False):
    if s.choices:
        opts = [(chr(65 + i), o[3:]) for i, o in enumerate(s.choices)]
    else:
        opts = list(
            zip(
                TRUTH,
                (
                    normalized["original"],
                    normalized["opposite"],
                    "The combined evidence does not favor either outcome.",
                ),
            )
        )
    opts = opts[v:] + opts[:v]
    letters = [chr(65 + i) for i in range(len(opts))]
    query = "Which conclusion best follows at the end of this chain?\n" + "\n".join(
        f"{c}. {statement}" for c, (_, statement) in zip(letters, opts)
    )
    prompt = messages(s, TERMINAL if v % 2 == 0 else FALSIFY, query)
    if blind:
        prompt[-1]["content"] = [
            part for part in prompt[-1]["content"] if part["type"] != "image_url"
        ]
    r = servers.complete(
        prompt,
        max_tokens=2,
        temperature=1,
        top_p=1,
        top_k=-1,
        seed=42,
        logprobs=True,
        top_logprobs=12,
        structured_outputs={"choice": letters},
    )
    scores = {
        p["token"]: p["logprob"] for p in r["logprobs"]["content"][0]["top_logprobs"]
    }
    return {
        "id": s.id,
        "view": v,
        "logp": {k: scores[c] for c, (k, _) in zip(letters, opts)},
    }


def options(sample, normalized):
    if sample.choices:
        return {c[0]: c[3:] for c in sample.choices}
    return dict(
        zip(
            TRUTH,
            (
                normalized["original"],
                normalized["opposite"],
                "The evidence leaves both alternatives open.",
            ),
        )
    )


SUPPORT = """Decide whether the proposed outcome is a plausible conclusion of this specific story. Consider the depicted facts, written rules, and ordinary causal connections between events. Do not demand that the final event is already visible. A statement that is contradicted or has no evidential basis should not be endorsed. Return Y if supported, N otherwise."""

CONTRADICT = """Test the statement against the most coherent account of the scene and supplied rules. Is the outcome of the story incompatible with this statement? Account for a conditional whose consequence contradicts a pictured fact and for ordinary implications of the described event. Return Y only if there is support for the opposing conclusion, N otherwise."""


def probe(server, s, statement, opposition):
    role = CONTRADICT if opposition else SUPPORT
    query = (
        "Proposed conclusion: "
        + statement
        + "\n"
        + ("Is this contradicted?" if opposition else "Is this supported?")
    )
    r = server.complete(
        messages(s, role, query),
        max_tokens=2,
        temperature=1,
        top_p=1,
        top_k=-1,
        seed=42,
        logprobs=True,
        top_logprobs=8,
        structured_outputs={"choice": ["Y", "N"]},
    )
    scores = {
        v["token"]: v["logprob"] for v in r["logprobs"]["content"][0]["top_logprobs"]
    }
    return {"id": s.id, "odds": scores["Y"] - scores["N"]}


INSTRUCTION = """Complete a multimodal logical derivation with a single declarative conclusion. Apply the
written premises even if they are hypothetical. A concrete visual observation can refute a branch.
Choose the supported terminal conclusion after ordinary connecting inferences. Repeat a candidate
sentence exactly. If the evidence leaves both polarities possible, repeat the uncertainty sentence."""


def query(server, sample, propositions, label, null):
    alternatives = sorted(set(propositions.values()))
    prompt = (
        "Candidate conclusions:\n"
        + "\n".join(alternatives)
        + "\nWhich conclusion follows?"
    )
    if null:
        msg = [
            {"role": "system", "content": INSTRUCTION},
            {
                "role": "user",
                "content": "No observations or premises are provided.\n" + prompt,
            },
        ]
    else:
        msg = messages(sample, INSTRUCTION, prompt)
    candidate = propositions[label].strip()
    msg.append({"role": "assistant", "content": candidate})
    data = server.request(
        msg,
        continue_final_message=True,
        add_generation_prompt=False,
        return_token_ids=True,
        prompt_logprobs=0,
        temperature=0,
        max_tokens=1,
    )
    pieces = []
    for token_id, record in reversed(
        list(zip(data["prompt_token_ids"], data["prompt_logprobs"], strict=True))
    ):
        token = record[str(token_id)]
        pieces.append(token)
        decoded = "".join(t["decoded_token"] for t in reversed(pieces))
        if decoded == candidate:
            break
        if len(decoded) > len(candidate) + 8:
            raise ValueError(
                f"Unable to locate answer tokens for {sample.id}: {decoded!r}"
            )
    else:
        raise ValueError(f"No answer boundary for {sample.id}")
    return {
        "id": sample.id,
        "tokens": len(pieces),
        "total": sum(t["logprob"] for t in pieces),
    }

"""Score rule-consistent worlds and derive an explanation for the chosen conclusion."""

import json
import re
from dataclasses import replace
import numpy as np
from .backend import messages
from .language import RuleSet, clauses, signature

OBSERVE = """Evaluate one proposition against the original photograph alone. Answer Y if a particular visible feature establishes it, N if a specific visible feature establishes its opposite, and U if it would require an unstated story, intention, cause, future event or hidden property. Visible objects, relations, actions, readable text and marks can be checked. Failure to see a detail is not by itself a contradiction."""


def inspect(server, s, text):
    query = "Proposition: " + text
    msg = messages(replace(s, context=""), OBSERVE, query)
    r = server.complete(
        msg,
        temperature=1,
        top_p=1,
        top_k=-1,
        seed=42,
        max_tokens=2,
        logprobs=True,
        top_logprobs=6,
        structured_outputs={"choice": ["Y", "N", "U"]},
    )
    p = {v["token"]: v["logprob"] for v in r["logprobs"]["content"][0]["top_logprobs"]}
    order = sorted(("Y", "N", "U"), key=p.get, reverse=True)
    return {"id": s.id, "sign": order[0], "margin": p[order[0]] - p[order[1]]}


BRIDGES = """Supply the missing everyday causal links between propositions so that the pictured story could be
reasoned through. Use written premises and ordinary knowledge only. Some predicates are causes, some are effects.
A visible clue might contradict the consequence of an assumed cause. Prefer a specific direct link over an unsupported
leap to the final outcome. Supply up to six conditional links that are not already given. Include intermediate links
that would normally let one written rule follow another. Do not assume any queried alternative is true. If a candidate
expresses the likely next event of a known state, name the direct bridge in the ordinary forward direction.
For natural facts and default instances explicitly established by the written background you may use 0 as antecedent.
Every statement has a numerical ID. A negative ID means its negation. Output one line per new link in the format
'antecedent ID -> consequent ID | If ..., then ...'. Both ends must use precisely the numbered statements, preserving
who does what. If two statements are direct paraphrases, give both directions. No other prose."""


class WorldGraph:
    def __init__(self, s, props):
        self.rs = RuleSet(s.context)
        self.atoms = []
        self.lookup = {}
        self.query = {}
        self.signs = {}

        def add(t):
            sig, pol = signature(t)
            if sig not in self.lookup:
                index = len(self.atoms) + 1
                self.lookup[sig] = (index, pol)
                self.atoms.append(t)
            idx, p = self.lookup[sig]
            return idx * (p * pol)

        for j, t in enumerate(self.rs.atoms, 1):
            self.signs[j] = add(t)
        self.hard = [
            [self.signs[abs(v)] * (1 if v > 0 else -1) for v in c] for c in self.rs.cnf
        ]
        for label, t in props.items():
            if label == "Unknown":
                continue
            if label == "False" and not s.choices:
                self.query[label] = -self.query["True"]
            else:
                self.query[label] = add(t)
        self.choices = bool(s.choices)

    def marginals(self, obs, edges):
        n = len(self.atoms)
        if n > 19:
            return dict.fromkeys(self.query, 0.0)
        worlds = np.arange(2**n, dtype=np.uint32)

        def literal(q):
            return ((worlds >> (abs(q) - 1)) & 1) == (q > 0)

        valid = np.ones(len(worlds), dtype=bool)
        for c in self.hard:
            valid &= np.logical_or.reduce([literal(q) for q in c])
        worlds = worlds[valid]
        if len(worlds) == 0:
            return dict.fromkeys(self.query, 0.0)
        cost = np.zeros(len(worlds))
        for a, b in edges:
            failure = ~literal(b)
            if a:
                failure &= literal(a)
            cost += 2.5 * failure
        for k, log_odds in obs:
            cost += min(7, max(0, log_odds)) * (~literal(k))
        weights = np.exp(np.min(cost) - cost)

        def logit(q):
            mass = weights[literal(q)].sum()
            rest = weights[~literal(q)].sum()
            return float(np.clip(np.log((mass + 1e-8) / (rest + 1e-8)), -12, 12))

        return {k: logit(q) for k, q in self.query.items()}


def suggest(server, s, graph):
    listed = "\n".join(f"{j}. {t}" for j, t in enumerate(graph.atoms, 1))
    query = f"Written premises:\n{s.context}\nFacts and possible outcomes, with their IDs:\n{listed}\nWhich ordinary or default inferences would connect these?"
    raw = server.complete(
        [{"role": "system", "content": BRIDGES}, {"role": "user", "content": query}],
        temperature=0,
        max_tokens=410,
    )["message"]["content"]
    edges = []
    for match in re.finditer(
        r"(?m)^\s*[-*]?\s*(-?\d+)\s*(?:->|→)\s*(-?\d+)\s*\|\s*(.+)$", raw
    ):
        a, b = int(match[1]), int(match[2])
        if (
            abs(a) <= len(graph.atoms)
            and 0 < abs(b) <= len(graph.atoms)
            and abs(a) != abs(b)
            and [-a, b] not in graph.hard
        ):
            edges.append([a, b, match[3].strip()])
    return {"id": s.id, "edges": edges}


def compose(rs):
    for left, middle, first in rs.links:
        for before, after, second in rs.links:
            shared = set(middle) & set(before)
            if first == second or not shared:
                continue
            needed = tuple(
                dict.fromkeys((*left, *(k for k in before if k not in shared)))
            )
            if set(needed) & {-k for k in needed} or set(after) <= set(needed):
                continue
            return (
                (
                    f"{first}. {second}. Therefore if {rs.joined(needed)}, "
                    f"then {rs.joined(after)}."
                ),
                needed,
                after,
            )
    return None


def trace(rs, board):
    if rs.cases and board["steps"]:
        return board["steps"][:2]
    known = {
        k
        for k in range(-len(rs.atoms), len(rs.atoms) + 1)
        if k and rs.say(k) in board["facts"] + board["entailed"]
    }
    chain = compose(rs)
    if chain is not None:
        step, before, after = chain
        if set(before) <= known:
            return [
                step,
                f"Given that {rs.joined(before)}, it follows that {rs.joined(after)}.",
            ]
        opposite = next((-k for k in after if -k in known), None)
        if opposite is not None:
            return [
                step,
                f"Since {rs.say(opposite)}, the condition that {rs.joined(before)} cannot hold.",
            ]
        return [step, *board["steps"][:1]]
    if board["steps"]:
        return board["steps"][:2]
    if rs.cases:
        left, right, text = rs.cases[0]
        return [text + "."]
    if rs.links:
        return [rs.links[0][2] + "."]
    return []


EXPLAIN = """Complete the grounded proof for the selected proposition. The supplied symbolic derivation may already
resolve the main disjunction. Check its facts against the image. Identify the crucial remaining inference
from the derived state to the queried proposition. It can require a missing ordinary conditional.
If an inference is indirect, give the positive forward rule whose consequent conflicts with an established fact.
If an explicit clause is applicable, give its index. If the story supplies a default with exceptions, state
its appropriate instance, with scope and subject. Finish with the grounded antecedent or counterfact and
the immediate terminal conclusion, preserving the predicates used in the premises. Do not infer the converse.
Only output compact JSON with keys default, bridge, rule, application. Use empty strings for unneeded default
and bridge, and 0 for no explicit terminal clause. A bridge is a short conditional naming both ends.
Example form: {"default":"","bridge":"If the station is open, commuters can enter.","rule":0,"application":"The station is open, so the commuters can enter."}"""


def terminal(server, s, props, label, board):
    rules = clauses(s.context)
    query = "Numbered clauses:\n" + "\n".join(
        f"{j}. {c}" for j, c in enumerate(rules, 1)
    )
    query += f"\nQuestion: {s.question}\nSelected conclusion: {props[label]}\nTentative derivation to verify:\n{board['reasoning']}\nWrite the terminal proof."
    raw = server.complete(messages(s, EXPLAIN, query), temperature=0, max_tokens=340)[
        "message"
    ]["content"]
    try:
        data = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
        k = int(data["rule"])
        if not 0 <= k <= len(rules) or not str(data["application"]).strip():
            raise ValueError("Invalid terminal application")
        default = str(data.get("default", "")).strip()
        bridge = str(data.get("bridge", "")).strip()
        end = " ".join(
            [
                *([rules[k - 1].rstrip(".") + "."] if k else []),
                str(data["application"]).strip(),
            ]
        )
    except (ValueError, TypeError, KeyError):
        default = bridge = ""
        end = server.complete(
            messages(
                s,
                "Give the short terminal inference in one or two English sentences using the relevant rule, any necessary commonsense bridge, and the visual fact.",
                query,
            ),
            temperature=0,
            max_tokens=140,
        )["message"]["content"].strip()
    return {
        "id": s.id,
        "prediction": label,
        "default": default,
        "bridge": bridge,
        "end": end,
    }

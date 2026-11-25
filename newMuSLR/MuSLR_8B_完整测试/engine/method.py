"""Run the fixed 8B method on one original image and question."""

import re
from dataclasses import replace

import numpy as np

from . import proof, rank
from .certificate import run_item
from .language import RuleSet
from .zoom import zoom_view


def mean_scores(*rows):
    return {k: sum(row[k] for row in rows) / len(rows) for k in rows[0]}


def winner(scores):
    return max(scores, key=scores.get)


def candidate_scores(server, sample, props, claims, evidence):
    rotations = range(len(props))

    def rotated(fn, service=server, scene=sample, blind=False):
        return mean_scores(
            *(fn(service, scene, claims, v, blind)["logp"] for v in rotations)
        )

    world, outcome = rotated(rank.choose), rotated(rank.score)
    full = mean_scores(world, outcome)
    text = mean_scores(
        rotated(rank.choose, blind=True), rotated(rank.score, blind=True)
    )
    zoom = mean_scores(full, text)
    view = server
    if winner(full) != winner(text) or winner(world) != winner(outcome):
        view = zoom_view(server, sample)
        if view is not server:
            zoom = mean_scores(zoom, rotated(rank.score, view))

    binary = {}
    for label, statement in props.items():
        if label == "Unknown":
            binary[label] = 0.0
            continue
        positive = rank.probe(view, sample, statement, False)["odds"]
        negative = rank.probe(view, sample, statement, True)["odds"]
        binary[label] = (positive - negative) / 2
    if sample.choices:
        graph = proof.WorldGraph(sample, props)
        hints = proof.suggest(server, sample, graph)["edges"]
        edges = list(dict.fromkeys((a, b) for a, b, _ in hints))
        observations = [
            (graph.signs[abs(lit)] * (1 if lit > 0 else -1), margin)
            for lit, margin in evidence
        ]
        return {
            "proposal": binary,
            "zoom": zoom,
            "worlds8": graph.marginals(observations, edges),
        }

    blank = replace(
        sample, context="No information about this particular scene is provided."
    )
    null = mean_scores(
        rotated(rank.choose, scene=blank, blind=True),
        rotated(rank.score, scene=blank, blind=True),
    )
    proposal = {k: (7 * zoom[k] + binary[k] - null[k]) / 8 for k in props}
    sentence = {}
    for label in props:
        scene_score = rank.query(server, sample, props, label, False)
        prior = rank.query(server, sample, props, label, True)
        if scene_score["tokens"] != prior["tokens"]:
            raise ValueError(f"Inconsistent tokenization for {sample.id}, {label}")
        sentence[label] = (scene_score["total"] - prior["total"]) / scene_score[
            "tokens"
        ] ** 0.5
    return {"proposal": proposal, "sentence": sentence}


def feature_matrix(labels, channels, names):
    matrix = np.array(
        [[channels[name][label] for name in names] for label in labels], dtype=float
    )
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite candidate scores")
    matrix -= matrix.mean(axis=0)
    return matrix / np.sqrt(1 + matrix.var(axis=0))


def infer(server, sample, weights, calibration=False):
    claims = None if sample.choices else rank.normalize(server, sample)
    props = rank.options(sample, claims)
    rules = RuleSet(sample.context)
    observations = {}

    def observe(index):
        if index not in observations:
            observations[index] = proof.inspect(server, sample, rules.atoms[index - 1])
        return observations[index]["sign"]

    item = {"id": sample.id, "full_context": sample.context, "choices": sample.choices}
    certificate = run_item(item, props, observe, "active")
    for index in range(1, len(rules.atoms) + 1):
        observe(index)
    evidence = [
        (i if obs["sign"] == "Y" else -i, obs["margin"])
        for i, obs in sorted(observations.items())
        if obs["sign"] != "U"
    ]
    label = certificate.get("prediction")
    if label is None or calibration:
        channels = candidate_scores(server, sample, props, claims, evidence)
        names = list(weights[sample.task])
        labels = list(props)
        matrix = feature_matrix(labels, channels, names)
        if calibration:
            return dict(
                id=sample.id,
                task=sample.task,
                labels=labels,
                names=names,
                features=matrix.tolist(),
                certified=label,
            )
        label = labels[(matrix @ list(weights[sample.task].values())).argmax()]

    board = rules.outline([(k, margin) for k, margin in evidence if margin >= 1])
    steps = list(board["steps"])
    if board["facts"]:
        steps.insert(
            0, "Reported visual facts to verify: " + "; ".join(board["facts"]) + "."
        )
    board["reasoning"] = "\n".join(steps)
    end = proof.terminal(server, sample, props, label, board)
    parts = [end["default"], *proof.trace(rules, board), end["bridge"], end["end"]]
    sentences = dict.fromkeys(
        t
        for part in parts
        for t in re.split(r"(?<=[.!?])\s+(?=[A-Z])", part.strip())
        if t
    )
    if not end["end"].strip():
        raise ValueError(f"Empty terminal explanation for {sample.id}")
    return {
        "id": sample.id,
        "prediction": label,
        "reasoning": " ".join(sentences),
        "route": certificate["route"],
    }

"""Fit fixed nonnegative weights using only the supplied external labels."""

import itertools
import json

import numpy as np

from .dataset import references


def select_weights(dataset, predictions, destination):
    rows = [json.loads(s) for s in predictions.read_text().splitlines()]
    refs = references(dataset)
    weights = {}
    for task in ("multiple_choice", "truth_evaluation"):
        subset = [row for row in rows if row["task"] == task]
        if not subset:
            raise ValueError(f"External calibration requires examples of {task}")
        names = subset[0]["names"]
        grid = [
            w for w in itertools.product(range(9), repeat=len(names)) if sum(w) == 8
        ]
        candidates = np.array(grid).T
        correct = np.zeros(len(grid), dtype=int)
        for row in subset:
            label = refs[row["id"]][0]
            target = [k.upper() for k in row["labels"]].index(label.upper())
            if row["certified"] is None:
                votes = np.array(row["features"]) @ candidates
                correct += votes.argmax(axis=0) == target
            else:
                correct += row["certified"] == row["labels"][target]
        best = max(
            range(len(grid)), key=lambda i: (correct[i], grid[i].count(0), grid[i])
        )
        weights[task] = dict(zip(names, grid[best], strict=True))
    temp = destination.with_suffix(".tmp")
    temp.write_text(json.dumps(weights, indent=2) + "\n")
    temp.replace(destination)

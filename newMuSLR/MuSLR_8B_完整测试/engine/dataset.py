"""Read original questions separately from the evaluation references."""

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Question:
    id: str
    image: Path
    context: str
    question: str
    choices: tuple[str, ...]

    @property
    def task(self):
        return "multiple_choice" if self.choices else "truth_evaluation"


def rows(directory):
    counts = Counter()
    with (directory / "metadata.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            source_id = row["id"]
            counts[source_id] += 1
            index = counts[source_id]
            yield source_id if index == 1 else f"{source_id}#{index}", row


def questions(directory):
    return [
        Question(
            uid,
            directory / r["file_name"],
            r["full_context"].strip(),
            r["question"].strip(),
            tuple(json.loads(r["choices"] or "[]")),
        )
        for uid, r in rows(directory)
    ]


def references(directory):
    return {
        uid: (r["answer"].strip(), json.loads(r["reasoning"]))
        for uid, r in rows(directory)
    }

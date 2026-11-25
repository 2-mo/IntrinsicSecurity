"""Build image-disjoint calibration and confirmation questions from A-OKVQA validation."""

import argparse
import csv
import hashlib
import io
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

HERE = Path(__file__).resolve().parent
PARQUET = "data/validation-00000-of-00001-b2bd0de231b6326a.parquet"
SOURCE_SHA = "419b38d94b1529b00d4aa0da0e945b946d8f028514183a267f87e25297ea404d"
ACTIONS = [
    (
        "the print is framed",
        "the print is mounted on a board",
        "the framing team receives the picture",
        "the mounting team receives the picture",
    ),
    (
        "the photograph goes into a slide show",
        "the photograph goes into a book",
        "the projectionist receives a slide",
        "the publisher receives a print",
    ),
    (
        "the editor schedules a follow-up interview",
        "the editor schedules a site visit",
        "a reporter prepares interview questions",
        "a reporter prepares a field notebook",
    ),
    (
        "the class prepares a pencil sketch",
        "the class prepares a watercolor",
        "the teacher provides pencils",
        "the teacher provides paints",
    ),
]


def build(source, dest):
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("A-OKVQA validation parquet checksum mismatch")
    records = {r["question_id"]: r for r in pq.read_table(source).to_pylist()}
    rng = random.Random(11913)
    used_images = set()
    for split in ("calibration", "confirmation"):
        folder = dest / split
        pictures = folder / "images"
        pictures.mkdir(parents=True, exist_ok=True)
        tmp = folder / "metadata.csv.tmp"
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "id",
                    "file_name",
                    "full_context",
                    "question",
                    "choices",
                    "answer",
                    "reasoning",
                ]
            )
            for j, uid in enumerate(
                (HERE / f"{split}_ids.txt").read_text().splitlines()
            ):
                row = records[uid]
                img = row["image"]["bytes"]
                fingerprint = hashlib.sha256(img).digest()
                if fingerprint in used_images:
                    raise ValueError(f"Repeated photograph {uid}")
                used_images.add(fingerprint)
                image_name = f"images/{uid}.jpg"
                with Image.open(io.BytesIO(img)) as im:
                    im.convert("RGB").save(folder / image_name, quality=95)
                q, opts, right = (
                    row["question"],
                    row["choices"],
                    row["correct_choice_idx"],
                )
                ctx = f'A photographer is asked this question about the picture: "{q}"'
                choices = [f"{c}. {s}" for c, s in zip("ABCD", opts, strict=True)]
                reasons = {f"R{k}": text for k, text in enumerate(row["rationales"], 1)}
                writer.writerow(
                    [
                        uid + "-choice",
                        image_name,
                        ctx,
                        q,
                        json.dumps(choices),
                        "ABCD"[right],
                        json.dumps(reasons),
                    ]
                )

                m, n, t, u = rng.choice(ACTIONS)
                order = list(range(4))
                rng.shuffle(order)
                index = order.index(right)
                conditional = [
                    f'If the answer to the photographer’s question is "{opts[k]}", then {m if a < 2 else n}.'
                    for a, k in enumerate(order)
                ]
                gate = j % 3 == 0
                cause = f"{m} and the editor signs the release" if gate else m
                outcomes = [f"If {cause}, then {t}.", f"If {n}, then {u}."]
                premises = (
                    conditional
                    + outcomes
                    + [
                        f"Exactly one of these arrangements applies: {m} or {n}.",
                        f"It is impossible for both of these outcomes to occur: {t} and {u}.",
                    ]
                )
                rng.shuffle(premises)
                claim = rng.randrange(2)
                positive = (t, u)[claim]
                flip = bool(rng.getrandbits(1))
                assertion = (
                    "It is false that " + positive
                    if flip
                    else positive[0].upper() + positive[1:]
                ) + "."
                answer = (
                    "Unknown"
                    if gate and index < 2
                    else str(bool(index // 2 == claim) != flip)
                )
                derivation = dict(
                    reasons, branch=conditional[index], consequence=outcomes[index // 2]
                )
                if answer == "Unknown":
                    derivation["gap"] = (
                        "The photograph does not establish whether the editor signs the release."
                    )
                else:
                    derivation["result"] = (
                        f"The active branch establishes {(t, u)[index // 2]}; the two outcomes cannot both occur."
                    )
                writer.writerow(
                    [
                        uid + "-truth",
                        image_name,
                        ctx + "\n" + "\n".join(premises),
                        "Is this conclusion True, False, or Unknown?\n" + assertion,
                        "[]",
                        answer,
                        json.dumps(derivation),
                    ]
                )
        tmp.replace(folder / "metadata.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--parquet", type=Path, help="Local copy of the pinned A-OKVQA validation file"
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.parquet is None:
        from huggingface_hub import hf_hub_download

        args.parquet = Path(
            hf_hub_download(
                "HuggingFaceM4/A-OKVQA",
                PARQUET,
                repo_type="dataset",
                revision="d1b0efa3a436e9101dfbde3752db7607da696c35",
            )
        )
    build(args.parquet, args.output)

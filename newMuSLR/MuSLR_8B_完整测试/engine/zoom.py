"""Request a closer view when image and text rankings disagree."""

import base64
import io
import json
from PIL import Image
from .backend import messages

SEEK = """Decide whether a closer look at a specific region could help distinguish the proposed outcomes by checking a concrete visual premise. Small objects, actions, relations, printed words and marks can be checked. A future event or a mental state cannot itself be read from pixels, but its prerequisite sometimes can. If a localized check is useful, specify a short answer-neutral observation question and its bounding box [left,top,right,bottom] on a 0 to 999 coordinate scale. Otherwise set box to an empty array. Do not decide the final answer."""

BOX_SCHEMA = {
    "type": "object",
    "properties": {
        "check": {"type": "string"},
        "box": {
            "type": "array",
            "maxItems": 4,
            "items": {"type": "integer", "minimum": 0, "maximum": 999},
        },
    },
    "required": ["check", "box"],
    "additionalProperties": False,
}


def structured(name, schema):
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}


def zoom_view(server, sample):
    query = sample.question + "\n" + "\n".join(sample.choices)
    reply = server.complete(
        messages(sample, SEEK, query),
        max_tokens=180,
        temperature=0,
        response_format=structured("optional_inspection", BOX_SCHEMA),
    )
    try:
        data = json.loads(reply["message"]["content"])
        x1, y1, x2, y2 = data["box"]
        check = data["check"].strip()
        if not (0 <= x1 < x2 <= 999 and 0 <= y1 < y2 <= 999 and check):
            return server
    except (ValueError, KeyError, TypeError):
        return server

    with Image.open(sample.image) as source:
        w, h = source.size
        left = min(w - 1, round(x1 * w / 1000))
        top = min(h - 1, round(y1 * h / 1000))
        right = max(left + 1, round(x2 * w / 1000))
        bottom = max(top + 1, round(y2 * h / 1000))
        cx, cy = (left + right) / 2, (top + bottom) / 2
        dx, dy = (right - left) * 0.75, (bottom - top) * 0.75
        box = (
            max(0, round(cx - dx)),
            max(0, round(cy - dy)),
            min(w, round(cx + dx)),
            min(h, round(cy + dy)),
        )
        crop = source.convert("RGB").crop(box)
        if min(crop.size) < 336:
            scale = min(4, 336 / min(crop.size))
            crop = crop.resize(
                tuple(round(n * scale) for n in crop.size), Image.Resampling.LANCZOS
            )
        buffer = io.BytesIO()
        crop.save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    return SecondView(server, check, url)


class SecondView:
    def __init__(self, server, check, url):
        self.server = server
        self.blocks = [
            {
                "type": "text",
                "text": "Second view of the same scene. Check this visual premise: "
                + check,
            },
            {"type": "image_url", "image_url": {"url": url}},
        ]

    def complete(self, prompt, **options):
        user = prompt[-1]
        first, *rest = user["content"]
        return self.server.complete(
            [*prompt[:-1], {**user, "content": [first, *self.blocks, *rest]}], **options
        )

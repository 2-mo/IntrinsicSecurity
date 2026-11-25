"""Compile explicit English clauses and propagate grounded visual facts."""

import re
from nltk.stem import PorterStemmer

CLAUSE_BOUNDARY = re.compile(
    r"(?:\n+|,\s+(?=(?:If|Either|Whenever|Unless)\b)|(?:^|(?<=\s))\d+[.)]\s+)"
)


def clauses(context):
    return [
        c.strip(" ,\t\r\n")
        for c in CLAUSE_BOUNDARY.split(context)
        if c.strip(" ,\t\r\n") not in ("", "-")
    ]


def unit_closure(clauses, assumptions):
    """Unit propagation returns guaranteed consequences; contradictions invalidate that assignment."""
    known = set(assumptions)
    if any(-k in known for k in known):
        return None
    while True:
        before = len(known)
        for clause in clauses:
            if any(k in known for k in clause):
                continue
            remaining = {k for k in clause if -k not in known}
            if not remaining:
                return None
            if len(remaining) == 1:
                known.update(remaining)
        if len(known) == before:
            return known


VERB = re.compile(
    r"\b(?:is|are|was|were|has|have|had|will|can|would|could|should|must|might|may|does|do|did|wears|holds|shows|contains|means|sees|uses|gets|wants|makes|feels|decides|starts|needs|takes|goes|becomes|looks|seems|indicates|suggests|helps|leads|involves|attracts|features|remains|receives)\b",
    re.I,
)


def fragments(text):
    chunks = re.split(r"\s+and\s+", text.strip(), flags=re.I)
    return chunks if len(chunks) > 1 and all(VERB.search(x) for x in chunks) else [text]


class RuleSet:
    def __init__(self, context):
        self.atoms = []
        self.lookup = {}
        self.links = []
        self.cases = []
        self.cnf = []
        for text in clauses(context):
            raw = text.strip().rstrip(".")
            match = re.fullmatch(r"If\s+(.+?),\s*then\s+(.+)", raw, re.I)
            if match:
                left, right = [
                    tuple(self.literal(x) for x in fragments(y)) for y in match.groups()
                ]
                self.links.append((left, right, raw))
                self.cnf.extend([[-a for a in left] + [b] for b in right])
            match = re.fullmatch(r"Either\s+(.+?),\s*or\s+(.+)", raw, re.I)
            if match:
                left, right = [
                    tuple(self.literal(x) for x in fragments(y)) for y in match.groups()
                ]
                self.cases.append((left, right, raw))
                self.cnf.extend([[a, b] for a in left for b in right])

    def literal(self, text):
        text = text.strip(" ,.\n")
        sign = 1
        tail = re.search(r"\s+(?:is false|is not true|is untrue)$", text, re.I)
        if tail:
            text = text[: tail.start()].strip()
            sign = -1
        negative = re.search(
            r"\b(is|are|was|were|has|have|had|will|can|would|could|should|may|must)\s+not\s+",
            text,
            re.I,
        )
        if negative:
            text = text[: negative.start()] + negative[1] + " " + text[negative.end() :]
            sign *= -1
        key = re.sub(r"^(?:the|a|an)\s+", "", text.casefold()).strip(" .")
        if key not in self.lookup:
            self.atoms.append(text)
            self.lookup[key] = len(self.atoms)
        return sign * self.lookup[key]

    def say(self, literal):
        text = self.atoms[abs(literal) - 1]
        return text if literal > 0 else "it is not the case that " + text

    def joined(self, lits):
        return " and ".join(self.say(k) for k in lits)

    def implications(self):
        for left, right, text in self.links:
            yield left, right, text
            if len(left) == 1:
                for v in right:
                    yield (-v,), (-left[0],), text

    def project(self, branch):
        for before, after, rule in self.implications():
            if set(before) <= set(branch) and not set(after) <= set(branch):
                return after, rule
        return branch, ""

    def outline(self, observations):
        accepted = []
        for k, _ in sorted(observations, key=lambda t: -t[1]):
            if unit_closure(self.cnf, accepted + [k]) is not None:
                accepted.append(k)
        known = unit_closure(self.cnf, accepted) or set()
        steps = []
        for a, b, source in self.cases:
            x, rule_x = self.project(a)
            y, rule_y = self.project(b)
            premises = list(
                dict.fromkeys(t + "." for t in (rule_x, rule_y, source) if t)
            )
            if rule_x or rule_y:
                steps.append(
                    " ".join(premises)
                    + " Therefore either "
                    + self.joined(x)
                    + " or "
                    + self.joined(y)
                    + "."
                )
            for branch, other in ((x, y), (y, x)):
                failed = next((t for t in branch if -t in known), None)
                if failed is not None and set(other) <= known:
                    steps.append(
                        "Either "
                        + self.joined(x)
                        + " or "
                        + self.joined(y)
                        + ". Since "
                        + self.say(-failed)
                        + ", "
                        + self.joined(other)
                        + "."
                    )
        for left, right, rule in self.implications():
            if set(left) <= known and not set(right) <= set(accepted):
                steps.append(
                    rule
                    + ". Given that "
                    + self.joined(left)
                    + ", it follows that "
                    + self.joined(right)
                    + "."
                )
        return {
            "facts": [self.say(k) for k in accepted],
            "entailed": [self.say(k) for k in sorted(known - set(accepted))],
            "steps": list(dict.fromkeys(steps)),
        }


stem = PorterStemmer()


def signature(text):
    s = text.lower().strip().rstrip(".")
    sign = 1
    for fmt in (
        r"^it is (?:false|not true) that\s+",
        r"\s+(?:is false|is not true|is untrue)$",
        r"(?<!\w)not(?!\w)",
    ):
        repl, n = re.subn(fmt, " ", s)
        if n:
            if n % 2:
                sign *= -1
            s = repl
    s = re.sub(r"\b(?:the|a|an|did|does|do)\b", " ", s)
    return " ".join(stem.stem(w) for w in re.findall(r"[\w]+", s)), sign

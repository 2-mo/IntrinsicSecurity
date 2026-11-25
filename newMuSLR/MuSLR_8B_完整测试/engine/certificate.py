"""Select the next visual question from the effects of both possible answers."""
from functools import lru_cache

from .language import RuleSet, signature


def satisfiable(clauses, assumptions=()):
    rows = tuple(map(frozenset, clauses)) + tuple(frozenset([x]) for x in assumptions)
    return solve(rows)


@lru_cache(maxsize=32768)
def solve(rows):
    if not rows:
        return True
    pivot = min(rows, key=len)
    if not pivot:
        return False
    literal = min(pivot)
    for value in ([literal] if len(pivot) == 1 else [literal, -literal]):
        reduced = tuple(c - {-value} for c in rows if value not in c)
        if solve(reduced):
            return True
    return False


def compile_item(item, props):
    rules = RuleSet(item['full_context'])
    dictionary = {}

    def bind(text):
        name, polarity = signature(text)
        if name not in dictionary:
            dictionary[name] = len(dictionary) + 1
        return polarity * dictionary[name]

    ids = {i: bind(text) for i, text in enumerate(rules.atoms, 1)}

    def convert(lit):
        return ids[abs(lit)] * (1 if lit > 0 else -1)

    blocks = []
    for left, right, text in rules.links:
        blocks.append((text, [[-convert(x) for x in left] + [convert(y)] for y in right]))
    for left, right, text in rules.cases:
        blocks.append((text, [[convert(x), convert(y)] for x in left for y in right]))
    goals = {k: bind(t) for k, t in props.items() if k not in ('Unknown', 'False')}
    if not item['choices']:
        goals['False'] = -goals['True']
    return blocks, goals, [(i, ids[i], t) for i, t in enumerate(rules.atoms, 1)]


def undecided(theory, facts, goals):
    return sum(satisfiable(theory, [*facts, g]) and satisfiable(theory, [*facts, -g]) for g in goals)


def certificate(blocks, facts, goals):
    theory = [clause for _, cs in blocks for clause in cs]
    literals = [v for v, _ in facts]
    forced = [(k, v) for k, v in goals.items() if not satisfiable(theory, [*literals, -v])]
    if len(forced) != 1:
        return None
    label, goal = forced[0]
    if not satisfiable(theory, [-goal]) or not satisfiable([], [*literals, -goal]):
        return None
    support = [('rule', t, cs) for t, cs in blocks] + [('image', t, [[v]]) for v, t in facts]
    # Delete redundant premises while keeping entailment. The result is inclusion-minimal.
    for entry in support.copy():
        trial = support.copy()
        trial.remove(entry)
        if not satisfiable([c for _, _, cs in trial for c in cs], [-goal]):
            support = trial
    return label, [{'source': kind, 'text': text, 'clauses': cs} for kind, text, cs in support]


BUDGET = 3


def run_item(item, props, observe, mode):
    blocks, goals, probes = compile_item(item, props)
    theory = [clause for _, cs in blocks for clause in cs]
    # A candidate itself cannot be the visual premise for a cross-modal proof.
    targets = {abs(v) for v in goals.values()}
    probes = [p for p in probes if abs(p[1]) not in targets]
    facts, trace = [], []
    answer = dict(id=item['id'], route='rank', trace=trace)
    if not satisfiable(theory):
        return answer
    for _ in range(BUDGET):
        literals = [f[0] for f in facts]
        available = [p for p in probes if all(satisfiable(theory, [*literals, v]) for v in (p[1], -p[1]))]
        if not available:
            break
        if mode == 'active':
            def priority(p):
                remaining = [undecided(theory, [*literals, v], targets) for v in (p[1], -p[1])]
                return max(remaining), sum(remaining), p[0]
            chosen = min(available, key=priority)
        elif mode == 'sequential':
            chosen = available[0]
        else:
            raise ValueError(mode)
        index, atom, text = chosen
        probes = [p for p in probes if abs(p[1]) != abs(atom)]
        sign = observe(index)
        trace.append(dict(index=index, atom=atom, question=text, sign=sign))
        if sign == 'U':
            continue
        if sign not in ('Y', 'N'):
            raise ValueError(sign)
        facts.append((atom if sign == 'Y' else -atom, text if sign == 'Y' else 'it is false that ' + text))
        proof = certificate(blocks, facts, goals)
        if proof:
            label, core = proof
            visual = ' and '.join(v['text'] for v in core if v['source'] == 'image')
            rules = '. '.join(v['text'].rstrip('.') for v in core if v['source'] == 'rule')
            answer.update(prediction=label, route='verified', support=core, goal=goals[label],
                          reasoning=f"The image shows that {visual}. {rules}. It follows that {props[label].rstrip('.')}.")
            break
    return answer

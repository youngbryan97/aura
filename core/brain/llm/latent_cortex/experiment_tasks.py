"""Self-verifying task families for the latent-cortex experiments.

A task carries its own answer, so an arm can be graded without a human and
without a second model. Split out of ``experiments.py`` when that module
crossed the 2,000-line ceiling; nothing here grades anything.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "TASK_FAMILIES",
    "Task",
    "khop_reachability",
    "modular_chain",
    "nested_boolean",
    "task_battery",
]

# Workload bounds. Boolean depth expands exponentially and chain families
# grow linearly in prompt size, so an unvalidated caller dimension is a
# denial-of-service on the experiment runner.
MAX_TASK_DEPTH = 64
MAX_PER_CELL = 512


#: Spacing a writer puts between thousands groups: "476 583", "476\u202f583".
_GROUPED_DIGITS = re.compile(r"(?<![\d.,])\d{1,3}(?:[ \u00a0\u2009\u202f]\d{3})+(?!\d)")
_GROUP_SPACES = re.compile(r"[ \u00a0\u2009\u202f]")
#: LaTeX that wraps a final answer: a box around it, or a thin space or a
#: braced comma between its thousands groups.
_LATEX_BOX = re.compile(r"\\(?:boxed|fbox)\s*\{([^{}]*)\}")
_LATEX_GROUP = re.compile(r"(?<=\d)(?:\\,|\{,\})(?=\d{3})")
#: Punctuation, emphasis and math delimiters around a number, never part of it.
_ANSWER_EDGE = ".,:;!?()[]{}*_`|$\\"


def answer_tokens(text: str) -> list[str]:
    """A reply's whitespace tokens, each a candidate for its final answer.

    Emphasis and punctuation come off a token's edges, a thousands group
    written with spaces is one number, and a typographic minus is a minus.
    LIVE 2026-10-06 her correct answers "**2,104,802,751,450**." and
    "**476 583**" were read as 4 and 9: the bold kept the final number from
    looking like one, and the last plain number in her working table won.
    G05 2026-10-07: "$$\\boxed{888211}$$" was read as 20, the remainder on the
    line above it, for the same reason.
    """
    unwrapped = _LATEX_GROUP.sub("", _LATEX_BOX.sub(r" \1 ", str(text or "")))
    rejoined = _GROUPED_DIGITS.sub(
        lambda match: _GROUP_SPACES.sub("", match.group(0)), unwrapped.replace("\u2212", "-")
    )
    return [token.strip(_ANSWER_EDGE) for token in rejoined.split()]


def is_answer_shaped(token: str) -> bool:
    """True for any token that could be a final numeric answer.

    Integers, decimals, signed values, fractions, scientific notation, and
    thousands-separated numbers all count — anything a model might end on.
    """
    body = token.strip().lstrip("+-")
    if not body:
        return False
    if "/" in body:
        parts = body.split("/")
        return len(parts) == 2 and all(
            part.strip().replace(",", "").replace(".", "", 1).isdigit()
            for part in parts
            if part.strip()
        ) and all(part.strip() for part in parts)
    normalized = body.replace(",", "")
    if normalized.replace(".", "", 1).isdigit():
        return True
    # Scientific notation: 1.5e-3
    lowered = normalized.lower()
    if "e" in lowered:
        mantissa, _, exponent = lowered.partition("e")
        exponent = exponent.lstrip("+-")
        return bool(
            mantissa
            and exponent
            and mantissa.replace(".", "", 1).isdigit()
            and exponent.isdigit()
        )
    return False



# ── Self-verifying task generators ──────────────────────────────────────


@dataclass
class Task:
    prompt: str
    answer: str
    depth: int
    family: str
    seed: int

    def verify(self, text: str) -> bool:
        """Exact-answer check on the FINAL claim in the output.

        The last answer-shaped token wins — chain-of-thought before it is
        fine; hedging two different answers is not.

        "Answer-shaped" must cover EVERY numeric form the model can end on,
        not just integers. The old filter kept a token only when it equalled
        the ground truth or was an integer, so a wrong final answer written
        as a decimal, fraction, or signed value was filtered out and an
        earlier correct token became the "final" claim — scoring a wrong
        answer as correct.
        """
        tokens = answer_tokens(text)
        candidates = [
            token
            for token in tokens
            if token and (token == self.answer or is_answer_shaped(token))
        ]
        return bool(candidates) and candidates[-1] == self.answer


def khop_reachability(depth: int, seed: int, n_nodes: int = 12) -> Task:
    """Follow a functional graph for ``depth`` hops; answer the landing node."""
    rng = random.Random(seed * 1_000_003 + depth)
    successor = {i: rng.randrange(n_nodes) for i in range(n_nodes)}
    start = rng.randrange(n_nodes)
    node = start
    for _ in range(depth):
        node = successor[node]
    edges = ", ".join(f"{a}->{b}" for a, b in sorted(successor.items()))
    prompt = (
        f"A directed graph has exactly one outgoing edge per node: {edges}. "
        f"Start at node {start} and follow exactly {depth} edges. "
        "Answer with the final node number only."
    )
    return Task(prompt=prompt, answer=str(node), depth=depth, family="khop", seed=seed)


def nested_boolean(depth: int, seed: int) -> Task:
    """Evaluate a nested and/or/not expression; answer 1 (true) or 0 (false)."""
    rng = random.Random(seed * 2_000_003 + depth)

    def build(d: int) -> tuple[str, bool]:
        if d <= 0:
            v = rng.random() < 0.5
            return ("1" if v else "0"), v
        op = rng.choice(("and", "or", "not"))
        if op == "not":
            s, v = build(d - 1)
            return f"(not {s})", (not v)
        left, lv = build(d - 1)
        right, rv = build(max(0, d - 1 - rng.randrange(2)))
        value = (lv and rv) if op == "and" else (lv or rv)
        return f"({left} {op} {right})", value

    expr, value = build(depth)
    prompt = (
        f"Evaluate this boolean expression where 1=true and 0=false: {expr}. "
        "Answer with a single digit, 1 or 0."
    )
    return Task(prompt=prompt, answer="1" if value else "0", depth=depth, family="boolean", seed=seed)


def modular_chain(depth: int, seed: int, mod: int = 17) -> Task:
    """Apply ``depth`` sequential +/× operations mod m; answer the result."""
    rng = random.Random(seed * 3_000_017 + depth)
    value = rng.randrange(mod)
    steps = [f"start with {value}"]
    for _ in range(depth):
        op, operand = rng.choice(("+", "*")), rng.randrange(1, mod)
        value = (value + operand) % mod if op == "+" else (value * operand) % mod
        steps.append(f"{op} {operand}, then take mod {mod}")
    prompt = (
        "Compute step by step: " + "; ".join(steps) + ". "
        f"All arithmetic is modulo {mod}. Answer with the final number only."
    )
    return Task(prompt=prompt, answer=str(value), depth=depth, family="modular", seed=seed)


TASK_FAMILIES: dict[str, Callable[[int, int], Task]] = {
    "khop": khop_reachability,
    "boolean": nested_boolean,
    "modular": modular_chain,
}


def task_battery(families: list[str], depths: list[int], per_cell: int, seed: int = 0) -> list[Task]:
    """Generate the requested battery, with BOUNDED workload dimensions.

    depth and per_cell were used unvalidated: a large boolean depth expands
    exponentially, a long chain consumes unbounded time, and a zero or
    negative per_cell silently produced an empty battery that later read as
    a legitimately-run experiment.
    """
    if not isinstance(families, list) or not families:
        raise ValueError("task_battery requires a non-empty family list")
    unknown = [family for family in families if family not in TASK_FAMILIES]
    if unknown:
        raise ValueError(f"unknown task families: {sorted(unknown)}")
    if not isinstance(depths, list) or not depths:
        raise ValueError("task_battery requires a non-empty depth list")
    for depth in depths:
        if type(depth) is not int or not 1 <= depth <= MAX_TASK_DEPTH:
            raise ValueError(
                f"task depth must be an int in [1, {MAX_TASK_DEPTH}]: {depth!r}"
            )
    if type(per_cell) is not int or not 1 <= per_cell <= MAX_PER_CELL:
        raise ValueError(
            f"per_cell must be an int in [1, {MAX_PER_CELL}]: {per_cell!r}"
        )
    if type(seed) is not int:
        raise ValueError("task_battery seed must be an int")

    tasks: list[Task] = []
    for family in families:
        gen = TASK_FAMILIES[family]
        for depth in depths:
            for i in range(per_cell):
                tasks.append(gen(depth, seed * 7919 + i))
    return tasks



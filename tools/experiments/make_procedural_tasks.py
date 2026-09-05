#!/usr/bin/env python3
"""Sample fresh task instances for the matched experiment.

The generator was written here, so this establishes that each INSTANCE is
unseen — drawn from a space too large to have been memorised — and not that
the task type is novel. The harness records the difference and the verdict
says so, because a run made of these alone is weaker evidence than one made of
externally authored problems and a reader should not have to ask which it was.

Every family here is checkable by rule, so the grader is exact rather than a
judgement about an answer.

    python tools/experiments/make_procedural_tasks.py --count 200 > tasks.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys


def multiplication(rng: random.Random) -> tuple[str, str]:
    left, right = rng.randint(101, 999), rng.randint(101, 999)
    return (
        f"What is {left} multiplied by {right}? Reply with the number only.",
        str(left * right),
    )


def modular(rng: random.Random) -> tuple[str, str]:
    base, mod = rng.randint(1000, 99999), rng.randint(7, 97)
    return (
        f"What is {base} modulo {mod}? Reply with the number only.",
        str(base % mod),
    )


def ordering(rng: random.Random) -> tuple[str, str]:
    values = rng.sample(range(100, 999), 6)
    return (
        "Sort these into ascending order, separated by single spaces, and "
        f"reply with nothing else: {' '.join(str(v) for v in values)}",
        " ".join(str(v) for v in sorted(values)),
    )


def counting(rng: random.Random) -> tuple[str, str]:
    letter = rng.choice("abcdefg")
    length = rng.randint(20, 40)
    text = "".join(rng.choice("abcdefg") for _ in range(length))
    return (
        f"How many times does the letter {letter} appear in this string? "
        f"Reply with the number only: {text}",
        str(text.count(letter)),
    )


def arithmetic_chain(rng: random.Random) -> tuple[str, str]:
    start = rng.randint(10, 99)
    steps = []
    value = start
    for _ in range(4):
        amount = rng.randint(2, 19)
        if rng.random() < 0.5:
            steps.append(f"add {amount}")
            value += amount
        else:
            steps.append(f"subtract {amount}")
            value -= amount
    return (
        f"Start with {start}, then {', then '.join(steps)}. "
        "What is the result? Reply with the number only.",
        str(value),
    )


FAMILIES = (multiplication, modular, ordering, counting, arithmetic_chain)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # No default seed: instances drawn from the same seed every time would be
    # the same instances every time, which is the memorisation this exists to
    # avoid. A seed is available for reproducing one particular run.
    rng = random.Random(args.seed)
    for index in range(args.count):
        family = FAMILIES[index % len(FAMILIES)]
        prompt, answer = family(rng)
        print(
            json.dumps(
                {
                    "id": f"{family.__name__}-{index}",
                    "prompt": prompt,
                    "answer": answer,
                    "author": "procedural_unseen",
                    "source": f"tools/experiments/make_procedural_tasks.py::{family.__name__}",
                }
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

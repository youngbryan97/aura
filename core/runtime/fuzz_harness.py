"""Deterministic fuzz harness for tools / skills / governance / state.

The audit lists fuzz targets: message router, tool schema parser, memory
record deserializer, state snapshot loader, event bus payload parser,
governance decision inputs, capability token parser, self-mod patch parser,
config loader, migration loader. This module gives a tiny but *real* fuzzer
that drives those targets with deterministic inputs (no Hypothesis dependency
required) so the contract is exercised in CI even without the full chaos run.
"""
from __future__ import annotations


import random
import string
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


def random_string(rng: random.Random, *, max_len: int = 64) -> str:
    n = rng.randint(0, max_len)
    return "".join(rng.choice(string.printable) for _ in range(n))


def random_dict(rng: random.Random, *, depth: int = 2) -> Dict[str, Any]:
    if depth <= 0:
        return {}
    out: Dict[str, Any] = {}
    for _ in range(rng.randint(0, 4)):
        key = random_string(rng, max_len=12)
        roll = rng.random()
        if roll < 0.25:
            out[key] = random_string(rng)
        elif roll < 0.5:
            out[key] = rng.randint(-1_000_000, 1_000_000)
        elif roll < 0.75:
            out[key] = random_dict(rng, depth=depth - 1)
        else:
            out[key] = [random_string(rng) for _ in range(rng.randint(0, 3))]
    return out


@dataclass
class FuzzReport:
    target: str
    iterations: int
    failures: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


def fuzz_target(
    name: str,
    parse: Callable[[Any], Any],
    *,
    iterations: int = 500,
    seed: int = 12345,
    input_fn: Optional[Callable[[random.Random], Any]] = None,
    forbidden_exceptions: Optional[List[type]] = None,
) -> FuzzReport:
    """Drive ``parse`` with ``iterations`` random inputs.

    A failure is recorded when the parser raises an exception that is not
    a subclass of any expected exception. For governance/parsers the
    expectation is that a malformed input either parses or raises a
    declared validation error — never crashes with TypeError, AttributeError,
    or KeyError.
    """
    rng = random.Random(seed)
    forbidden = forbidden_exceptions or [TypeError, AttributeError, KeyError, IndexError, AssertionError]
    report = FuzzReport(target=name, iterations=iterations)
    fn = input_fn or random_dict
    for i in range(iterations):
        candidate = fn(rng)
        try:
            parse(candidate)
        except BaseException as exc:
            if isinstance(exc, tuple(forbidden)):
                report.failures.append({"i": i, "input": repr(candidate)[:200], "exc": repr(exc)})
    return report

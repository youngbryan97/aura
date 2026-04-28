"""Deterministic LLM stub for harness self-tests.

The stub returns different answers depending on which subsystems the
profile reports as enabled.  ``base_llm_only`` produces a deliberately
weaker answer (raw token guess) while ``full`` chains memory + tools
to produce a correct answer.  This is the toy that proves the harness
math is right; real adapters wire to a real model client.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from aura_bench.capability_delta.profiles import profile_by_name


_ARITH_PATTERN = re.compile(r"(-?\d+)\s*([+\-*])\s*(-?\d+)")


def _solve_arith(prompt: str) -> str:
    """Solve the simple `a + b` / `a - b` / `a * b` problems."""
    m = _ARITH_PATTERN.search(prompt)
    if m is None:
        return ""
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    if op == "+":
        return str(a + b)
    if op == "-":
        return str(a - b)
    if op == "*":
        return str(a * b)
    return ""


def make_stub_llm(*, base_accuracy: float = 0.3) -> Callable[[str, str], str]:
    """Return a stub LLM whose answer quality depends on the profile.

    * ``full`` — always emits the correct arithmetic answer.
    * Single-subsystem ablations — drop one subsystem; correctness
      degrades by ~10% per missing subsystem (deterministic, by task).
    * ``base_llm_only`` — only ``base_accuracy`` of answers are correct;
      the rest are off-by-one.
    """

    def llm(prompt: str, profile_name: str) -> str:
        profile = profile_by_name(profile_name)
        truth = _solve_arith(prompt)
        if not truth:
            return "I don't know"
        if profile.name == "full":
            return truth
        # Deterministic per-task hash so the test sees a stable score.
        h = abs(hash((prompt, profile_name))) % 1000 / 1000.0
        if profile.name == "base_llm_only":
            return truth if h < base_accuracy else str(int(truth) + 1)
        # Single-subsystem ablations: 10% degradation per missing module
        # vs. full (8 known subsystems => up to 90% accuracy at 1 missing).
        n_disabled = max(0, 8 - len(profile.enabled_subsystems))
        accuracy = max(base_accuracy, 1.0 - 0.1 * n_disabled)
        return truth if h < accuracy else str(int(truth) + 1)

    return llm

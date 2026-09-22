"""Math truth engine — exact arithmetic / sympy / z3 checks on a candidate.

Wraps :class:`core.reasoning.symbolic_bridge.SymbolicBridge`. It does not ask the
LLM to be right — it re-checks every stated ``expr = value`` claim with exact
arithmetic and flags the wrong ones. This is the cheapest, highest-yield verifier:
a confident calculation error becomes a hard fail.
"""
from __future__ import annotations

import math
import re
from typing import Any

from core.runtime.errors import record_degradation

from .base import VerificationResult

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_ANSWER_TAG_RE = re.compile(r"<answer\b[^>]*>(.*?)</answer>", re.I | re.S)
_FINAL_MARKER_RE = re.compile(
    r"(?:^|\n)\s*(?:FINAL_ANSWER|final\s+answer|answer)\s*(?::|=|\bis\b)\s*(.+)",
    re.I,
)
# Bounds keep derivation exact and cheap (no giant powers/factorials).
_MAX_POW_EXP = 64
_MAX_FACT_N = 50


def _i(s: str) -> int:
    return int(s.replace(",", ""))


def _derive_exact_answer(question: str) -> tuple[str, str] | None:
    """Derive a canonical, exactly-computable answer from the QUESTION text.

    Returns (label, exact_value_as_str) for operation classes models reliably fumble
    — modulo, power, gcd, factorial, and binary arithmetic — or None when the question
    isn't one of these. This is what makes the verifier *sound* for these classes: a
    wrong final number becomes a hard fail the amplifier can filter out.
    """
    q = str(question or "").lower()
    try:
        m = re.search(r"(\d[\d,]*)\s*(?:mod(?:ulo)?|%)\s*(\d[\d,]*)", q)
        if m and _i(m.group(2)) != 0:
            return f"{_i(m.group(1))} mod {_i(m.group(2))}", str(_i(m.group(1)) % _i(m.group(2)))

        m = re.search(r"(\d[\d,]*)\s*(?:\^|\*\*|to the power of|raised to(?: the power of)?)\s*(\d+)", q)
        if m and _i(m.group(2)) <= _MAX_POW_EXP:
            return f"{_i(m.group(1))}^{_i(m.group(2))}", str(_i(m.group(1)) ** _i(m.group(2)))

        m = re.search(r"(?:gcd|greatest common divisor)\D*(\d[\d,]*)\D+(\d[\d,]*)", q)
        if m:
            return f"gcd({_i(m.group(1))},{_i(m.group(2))})", str(math.gcd(_i(m.group(1)), _i(m.group(2))))

        m = re.search(r"(\d+)\s*(?:!|factorial)", q)
        if m and _i(m.group(1)) <= _MAX_FACT_N and "trailing" not in q and "zero" not in q:
            return f"{_i(m.group(1))}!", str(math.factorial(_i(m.group(1))))

        m = re.search(r"(\d[\d,]*)\s*(?:times|multiplied by|\*)\s*(\d[\d,]*)", q)
        if m:
            return f"{_i(m.group(1))}*{_i(m.group(2))}", str(_i(m.group(1)) * _i(m.group(2)))
    # not a failure: a number this cannot read is not one this engine answers.
    except (ValueError, OverflowError):
        return None
    return None


def _final_answer_surface(text: str) -> str:
    """Return the candidate's final asserted answer, never an intermediate step."""

    rendered = str(text or "").strip()
    tagged = _ANSWER_TAG_RE.findall(rendered)
    if tagged:
        return tagged[-1].strip()
    marked = _FINAL_MARKER_RE.findall(rendered)
    if marked:
        return marked[-1].strip()
    # Natural prose without an explicit envelope remains supported, but only
    # its last numeric conclusion is eligible for an exact numeric target.
    numbers = _NUM.findall(rendered)
    return numbers[-1] if numbers else rendered


def _final_answer_matches(text: str, exact: str) -> bool:
    """Compare the exact target to the final asserted answer only."""

    surface = _final_answer_surface(text)
    try:
        target = float(exact)
    except ValueError:
        return exact.strip().casefold() == surface.strip().casefold()
    numbers = _NUM.findall(surface)
    if not numbers:
        return False
    try:
        final_value = float(numbers[-1].replace(",", ""))
    # not a failure: a value that is not a number is not one this can read.
    except ValueError:
        return False
    return math.isclose(final_value, target, rel_tol=0.0, abs_tol=1e-6)


class MathTruthEngine:
    name = "math"
    domains = ("math", "arithmetic", "calculation", "logic_math")

    def handles(self, task_type: str) -> bool:
        return task_type in self.domains

    async def verify(self, candidate: str, *, context: dict[str, Any] | None = None) -> VerificationResult:
        text = str(candidate or "")
        try:
            from core.reasoning.symbolic_bridge import SymbolicBridge

            bridge = SymbolicBridge()
        except (ImportError, RuntimeError) as exc:  # pragma: no cover - import guard
            record_degradation("math_truth_engine", exc)
            return VerificationResult(domain="math", ok=True, checked=False, engine=self.name)

        arithmetic_errors = bridge.check_arithmetic_claims(text)
        # An explicit verification target lets us actually solve & compare.
        target = (context or {}).get("verify_expression") if context else None
        target_ok = None
        target_value: Any = None
        target_label: Any = target
        if target:
            res = bridge.evaluate(str(target))
            if res.ok:
                target_value = res.result
                target_ok = _final_answer_matches(text, str(res.result))

        # No explicit target — derive a canonical exact answer from the QUESTION so the
        # verifier is SOUND for modulo/power/gcd/factorial/arithmetic (the classes models
        # fumble). A wrong final number is now a hard fail amplification can filter out.
        if target_ok is None:
            derived = _derive_exact_answer(str((context or {}).get("objective", "")))
            if derived is not None:
                target_label, target_value = derived
                target_ok = _final_answer_matches(text, str(target_value))

        # If there were no numeric claims and nothing to compare against, nothing to check.
        if not arithmetic_errors and target_ok is None and "=" not in text:
            return VerificationResult(domain="math", ok=True, checked=False, engine=self.name)

        issues = [f"arithmetic error: {e['claim']} (correct: {e['correct']})" for e in arithmetic_errors]
        if target_ok is False:
            issues.append(f"answer does not match exact value {target_value} of {target_label}")
        ok = not arithmetic_errors and target_ok is not False
        score = 0.95 if ok else max(0.05, 0.5 - 0.2 * len(arithmetic_errors))
        evidence = [f"exact({target_label}) = {target_value}"] if target_value is not None else []
        return VerificationResult(
            domain="math",
            ok=ok,
            checked=True,
            score=round(score, 4),
            engine=self.name,
            issues=issues,
            evidence=evidence,
            detail={"arithmetic_errors": len(arithmetic_errors)},
        )

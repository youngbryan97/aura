"""Math truth engine — exact arithmetic / sympy / z3 checks on a candidate.

Wraps :class:`core.reasoning.symbolic_bridge.SymbolicBridge`. It does not ask the
LLM to be right — it re-checks every stated ``expr = value`` claim with exact
arithmetic and flags the wrong ones. This is the cheapest, highest-yield verifier:
a confident calculation error becomes a hard fail.
"""
from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Any

from core.runtime.errors import record_degradation

from .base import VerificationResult

_NUM = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?(?:/[-+]?\d+)?")
_ANSWER_TAG_RE = re.compile(r"<answer\b[^>]*>(.*?)</answer>", re.I | re.S)
_FINAL_MARKER_RE = re.compile(
    r"(?:^|\n)\s*(?:FINAL_ANSWER|final\s+answer|answer)\s*(?::|=|\bis\b)\s*(.+)",
    re.I,
)
# Bounds keep derivation exact and cheap (no giant powers/factorials).
_MAX_POW_EXP = 64
_MAX_FACT_N = 50


def _i(s: str) -> int:
    if not re.fullmatch(r"[-+]?(?:\d+|\d{1,3}(?:,\d{3})+)", s):
        raise ValueError("invalid integer grouping")
    return int(s.replace(",", ""))


def _derive_exact_answer(question: str) -> tuple[str, str] | None:
    """Derive a canonical, exactly-computable answer from the QUESTION text.

    Returns (label, exact_value_as_str) for operation classes models reliably fumble
    — modulo, power, gcd, factorial, and binary arithmetic — or None when the question
    isn't one of these. This is what makes the verifier *sound* for these classes: a
    wrong final number becomes a hard fail the amplifier can filter out.
    """
    q = str(question or "").lower().strip().rstrip("?.").strip()
    q = re.sub(r"^(?:what is|calculate|compute|evaluate|find)\s+", "", q)
    try:
        m = re.fullmatch(r"(?:gcd|greatest common divisor)\s*(?:of\s*)?\(?\s*([-+]?\d[\d,]*)\s*(?:and|,)\s*([-+]?\d[\d,]*)\s*\)?", q)
        if m:
            return f"gcd({_i(m.group(1))},{_i(m.group(2))})", str(math.gcd(_i(m.group(1)), _i(m.group(2))))

        m = re.fullmatch(r"(\d+)\s*(?:!|factorial)", q)
        if m and _i(m.group(1)) <= _MAX_FACT_N and "trailing" not in q and "zero" not in q:
            return f"{_i(m.group(1))}!", str(math.factorial(_i(m.group(1))))

        from core.reasoning.arithmetic_on_the_floor import integer_expression

        for token in re.findall(r"\d[\d,]*", q):
            if "," in token:
                _i(token)
        expression = q.replace(",", "").replace("^", "**")
        for phrase, operator in (("raised to the power of", "**"), ("to the power of", "**"),
                                 ("raised to", "**"), ("multiplied by", "*"), ("times", "*"),
                                 ("modulo", "%"), ("mod", "%")):
            expression = re.sub(rf"\b{phrase}\b", operator, expression)
        value = integer_expression(expression.strip())
        if value is not None:
            return q, str(value)
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
        target = _exact_number(exact)
    except (ValueError, ZeroDivisionError):
        return exact.strip().casefold() == surface.strip().casefold()
    numbers = _NUM.findall(surface)
    if len(numbers) != 1:
        return False
    try:
        final_value = _exact_number(numbers[-1])
    # not a failure: a value that is not a number is not one this can read.
    except (ValueError, ZeroDivisionError):
        return False
    return final_value == target


def _exact_number(text: str) -> Fraction:
    for token in re.findall(r"\d[\d,]*", text):
        if "," in token:
            _i(token)
    cleaned = text.replace(",", "").strip()
    if len(cleaned) > 4096:
        raise ValueError("numeric comparison exceeds its budget")
    exponent = re.search(r"[eE]([-+]?\d+)", cleaned)
    if exponent and (len(exponent[1]) > 5 or abs(int(exponent[1])) > 4096):
        raise ValueError("numeric exponent exceeds its budget")
    return Fraction(cleaned)


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

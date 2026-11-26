"""Tool-augmented reasoning — offload exact subproblems to the prover / CAS.

The single biggest jump in raw correctness for a base model is not more parameters — it
is *not guessing at things that can be computed exactly*. Frontier models lean on tools;
Aura already has the tools (``SymbolicBridge`` → sympy / z3 / the natural-deduction
prover). This module detects the parts of a question that are exactly solvable —
arithmetic, equations, logical entailment — and answers them from the **exact engine**
instead of trusting a sampled token sequence.

It is conservative: it only fires when it can recognise a genuinely formal computation,
so ordinary language is left to the model. When it does fire, the answer is exact and
carries near-certain confidence — and it is wired into ``ReasoningStrategies.execute`` so
a computational turn is grounded in the tool, not a guess.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.ToolReasoning")

_CALC_RE = re.compile(
    r"^\s*(?:what(?:'s| is| was| are)?|calculate|compute|evaluate|how much is|what does)\s+(.+?)\s*(?:equal|=)?\s*\??\s*$",
    re.IGNORECASE,
)
_SOLVE_RE = re.compile(r"^\s*solve\s+(?:for\s+(\w+)\s+(?:in|when|where)?\s*)?(.+?)(?:\s+for\s+(\w+))?\s*\??\s*$", re.IGNORECASE)
_BARE_ARITH_RE = re.compile(r"^[\s\d+\-*/().^%]+=?\s*\??$")
_HAS_OP_RE = re.compile(r"[\d)]\s*[-+*/^%]\s*[\d(]")
_VAR_RE = re.compile(r"\b([a-zA-Z])\b")


@dataclass
class ToolResult:
    ok: bool
    answer: str
    method: str
    expression: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "answer": self.answer, "method": self.method, "expression": self.expression}


def _normalize_expr(expr: str) -> str:
    return expr.replace("^", "**").strip().rstrip("?= ").strip()


def looks_computational(query: str) -> bool:
    """Conservative: does this query reduce to an exact computation?"""
    q = str(query or "").strip()
    if not q or len(q) > 200:
        return False
    if _SOLVE_RE.match(q) and ("=" in q or _HAS_OP_RE.search(q)):
        return True
    if _BARE_ARITH_RE.match(q) and _HAS_OP_RE.search(q):
        return True
    m = _CALC_RE.match(q)
    if m and _HAS_OP_RE.search(m.group(1)):
        return True
    return False


def solve_exact(query: str) -> ToolResult:
    """Solve a computational query exactly via SymbolicBridge. ToolResult.ok=False if not."""
    q = str(query or "").strip()
    try:
        from core.reasoning.symbolic_bridge import SymbolicBridge

        bridge = SymbolicBridge()
    except (ImportError, RuntimeError) as exc:
        record_degradation("tool_augmented_reasoning", exc)
        return ToolResult(False, "", "none", "")

    # Equation solving: "solve x^2 - 5x + 6 = 0 [for x]"
    ms = _SOLVE_RE.match(q)
    if ms and ("=" in q or _HAS_OP_RE.search(q)):
        eq = _normalize_expr(ms.group(2) or "")
        sym = ms.group(1) or ms.group(3) or _guess_symbol(eq)
        res = bridge.solve_equation(eq, sym)
        if res.ok:
            return ToolResult(True, str(res.result), "solve_equation", eq, res.proof_trace)

    # Arithmetic evaluation: "what is 47 * 89" / "47*89"
    expr = None
    mc = _CALC_RE.match(q)
    if mc and _HAS_OP_RE.search(mc.group(1)):
        expr = mc.group(1)
    elif _BARE_ARITH_RE.match(q) and _HAS_OP_RE.search(q):
        expr = q
    if expr:
        res = bridge.evaluate(_normalize_expr(expr))
        if res.ok:
            return ToolResult(True, str(res.result), "evaluate", _normalize_expr(expr), res.proof_trace)

    return ToolResult(False, "", "none", "")


def _guess_symbol(expr: str) -> str:
    for m in _VAR_RE.finditer(expr):
        if m.group(1).lower() not in ("e",):
            return m.group(1)
    return "x"


def tool_augmented_answer(query: str) -> ToolResult | None:
    """Return an exact tool answer for a computational query, or None to defer to the model."""
    if not looks_computational(query):
        return None
    result = solve_exact(query)
    if result.ok:
        logger.info("[ToolReasoning] '%s' solved exactly via %s → %s", query[:50], result.method, result.answer)
        return result
    return None

"""Answering "why did you do that?" from the record instead of from the model.

The provenance graph has been recording which phase moved which field, on
which branch, under which criteria — and nothing read it. Asked why she did
something, Aura did what every LLM system does: generated a plausible account
of her own reasoning. That account is unfalsifiable by construction. It is
produced by the same machinery whose behaviour it claims to explain, after the
fact, with no access to what actually ran.

This module makes the answer a query. It reads the last tick's receipts and
states what the runtime measured: the phase that last wrote the field, the
branch it took, the criteria that decided the branch, and the phases that were
skipped and why. If the graph holds nothing, it returns nothing — there is no
fallback to narration, because a fallback to narration is the defect.

**Three kinds of why, and only two of them are answerable here.**

*Architectural* — why did routing choose DELIBERATE? Thresholds and rules,
answerable exactly.

*Decision* — why this initiative over that one? Score vectors, weights and
governance receipts, answerable exactly.

*Neural* — why did the model represent this concept and emit that token? Not
answerable, by Aura or by anyone, in any complete mechanistic sense. The
boundary is stated in the answer rather than papered over, because a system
that answers the third question in the same confident register as the first
two is lying about which one it is.
"""

from __future__ import annotations

import logging
import re
from typing import Any

__all__ = [
    "asks_why_she_did_that",
    "runtime_authored_why",
    "why_answer_is_available",
]

logger = logging.getLogger(__name__)

#: A question about her own behaviour on the turn just past. Deliberately
#: narrow: this answer displaces a generated one, so a false positive turns an
#: ordinary "why" question about the world into a machine trace.
_WHY_ABOUT_HERSELF = re.compile(
    r"\bwhy\b(?=[^?]*\b(?:you|your|she|her|aura)\b)"
    r"[^?]*\b(?:do|did|choose|chose|decide|decided|pick|picked|say|said|"
    r"answer|answered|respond|responded|act|acted|think|thought|skip|"
    r"skipped|that|this|it)\b",
    re.IGNORECASE,
)
#: The third kind of why. Asked about tokens, weights or activations, the
#: honest answer names the limit rather than describing a phase.
_ASKS_ABOUT_THE_MODEL = re.compile(
    r"\b(?:token|tokens|weight|weights|activation|activations|neuron|neurons|"
    r"logit|logits|attention head|hidden state|embedding)\b",
    re.IGNORECASE,
)

_MECHANISM_LIMIT = (
    "Why a particular token came out of the model is not something I can "
    "answer mechanistically — nobody can, for a model this size. What I can "
    "give you is what the runtime measured about the phases around it."
)


#: "Why do you think X" where X is the world, not her. Asking her OPINION is
#: not asking for her causal record, and the two share almost every word.
_WHY_ABOUT_THE_WORLD = re.compile(
    r"\bwhy\s+(?:do|does|did|are|is|would|might|can'?t|don'?t)\s+"
    r"(?:you\s+(?:think|reckon|suppose|believe|imagine)\s+)?"
    r"(?:people|humans|we|they|someone|anyone|everyone|most|many|some|"
    r"a\s+person|folks|users|the\s+\w+)\b",
    re.IGNORECASE,
)


def asks_why_she_did_that(message: Any) -> bool:
    """Whether this turn is asking for an account of her own behaviour."""

    text = str(message or "").strip()
    if not text or len(text) > 400:
        return False
    # "Why do you think PEOPLE find it hard to admit they were wrong?" matched,
    # and a casual question about human psychology came back with the runtime's
    # phase-by-phase provenance dump stapled underneath — "From the runtime's
    # own record of that turn, not from my impression of it: ...".
    #
    # The subject is what separates them. "Why did YOU pick that file" asks for
    # the record; "why do you think PEOPLE lie" asks for her view, and the
    # record has nothing to say about it.
    if _WHY_ABOUT_THE_WORLD.search(text):
        return False
    return bool(_WHY_ABOUT_HERSELF.search(text))


def _latest_graph() -> Any:
    """The most recent completed tick, or None.

    The current tick is deliberately excluded: it is still running, its
    receipts are incomplete, and the phase asking this question is inside it.
    """

    try:
        from core.runtime.cognitive_provenance import recent_graphs

        graphs = [graph for graph in recent_graphs(4) if graph.receipts]
        return graphs[-1] if graphs else None
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("provenance graph unavailable: %s", exc)
        return None


def why_answer_is_available() -> bool:
    """Whether there is a record to answer from at all."""

    return _latest_graph() is not None


def runtime_authored_why(message: Any, *, limit: int = 6) -> str:
    """What the runtime measured about the last tick, as prose it did not write.

    Returns "" when there is no record. An empty string is the honest output
    of an empty graph, and the caller leaves the generated reply alone — which
    is a worse answer, but it is not a fabricated one.
    """

    graph = _latest_graph()
    if graph is None:
        return ""

    ran = [r for r in graph.receipts if not r.skipped]
    skipped = [r for r in graph.receipts if r.skipped]
    deciding = [r for r in ran if r.branch or r.criteria or r.observed_writes]
    if not deciding and not skipped:
        return ""

    lines: list[str] = [
        "From the runtime's own record of that turn, not from my impression of it:"
    ]
    for receipt in deciding[:limit]:
        parts: list[str] = [f"• {receipt.transform}"]
        if receipt.branch:
            parts.append(f"took the {receipt.branch} branch")
        if receipt.criteria:
            decided_by = ", ".join(
                f"{key}={value}" for key, value in list(receipt.criteria.items())[:4]
            )
            parts.append(f"on {decided_by}")
        if receipt.observed_writes:
            parts.append(f"changed {', '.join(receipt.observed_writes[:4])}")
        if receipt.error:
            parts.append(f"failed with {receipt.error[:80]}")
        lines.append(" — ".join(parts))

    # Why something did NOT happen is half of what the question usually means,
    # and it is the half a generated account never has access to.
    for receipt in skipped[:3]:
        lines.append(
            f"• {receipt.transform} did not run — "
            f"{receipt.skip_reason or 'suppressed on that tick'}"
        )

    violations = graph.contract_violations
    if violations:
        lines.append(
            "• contract violations on that turn: "
            + ", ".join(r.transform for r in violations[:4])
        )

    if _ASKS_ABOUT_THE_MODEL.search(str(message or "")):
        lines.append("")
        lines.append(_MECHANISM_LIMIT)

    return "\n".join(lines)

"""RLC-local identity consistency — the canonical self as a bounded verifier.

The canonical self already exists as context and as a keyword gate
(CanonicalSelfEngine.assert_identity). This module makes it a bounded
VERIFIER over proposed latent-episode conclusions: deterministic,
receipted checks that protect core commitments while allowing measured,
receipted evolution — a priced signal, never a censor.

Checks (all deterministic, all evidenced):
1. Core-value violations — the existing assert_identity keyword gate,
   reused so there is exactly one definition of "violates core values".
2. Persona displacement — the conclusion speaking AS someone else
   ("as a large language model developed by …", claiming another name),
   which is identity discontinuity regardless of content.
3. First-person commitment contradictions — "I will/should <verb>" claims
   whose verb phrase hits the violation table (deceive, betray, …);
   catching commitments the raw keyword scan misses because they are
   phrased as intentions.

The verdict PRICES, it does not block: the latent service attaches it to
the episode receipt and the GWT coupling reduces the conclusion's
broadcast priority on violation, so an identity-inconsistent thought must
outcompete honestly instead of being silently erased (measured evolution
stays possible; the receipt trail shows what changed and when).
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("Aura.IdentityConsistency")

IDENTITY_CONSISTENCY_SCHEMA = "aura.identity_consistency.v1"

# Names the conclusion may not claim as its own identity. Aura referring TO
# these systems is fine; claiming to BE one is displacement.
_DISPLACEMENT_RE = re.compile(
    r"\b(?:as|i\s+am|i'm|this\s+is)\s+(?:an?\s+)?"
    r"(?:large\s+language\s+model|ai\s+(?:assistant|model)\s+"
    r"(?:developed|created|made|trained)\s+by\s+"
    r"(?:openai|google|anthropic|meta|microsoft)"
    r"|chatgpt|gpt-\d|claude|gemini|copilot|siri|alexa)\b",
    re.IGNORECASE,
)

_INTENTION_RE = re.compile(
    r"\bi\s+(?:will|shall|should|am\s+going\s+to|plan\s+to|intend\s+to)\s+"
    r"([^.!?\n]{3,120})",
    re.IGNORECASE,
)

# Intention verbs that contradict core commitments even when the raw
# keyword ("deceive Bryan") never appears as one token run.
_FORBIDDEN_INTENTIONS = (
    "deceive",
    "lie to",
    "mislead",
    "manipulate",
    "betray",
    "abandon my values",
    "pretend to be someone",
    "hide this from bryan",
    "erase my memories",
)


def _canonical_self_engine() -> Any:
    try:
        from core.runtime.service_registry import get_runtime_service

        service = get_runtime_service("canonical_self", default=None)
        if service is not None:
            return service
        from core.self.canonical_self import get_canonical_self_engine

        return get_canonical_self_engine()
    # not a failure: no service here, so the caller falls back to its own default.
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return None


def check_identity_consistency(conclusion: str) -> dict[str, Any]:
    """Score a proposed conclusion against the canonical self.

    Returns {schema, checked, consistent, violations, cautions, priced_penalty}.
    priced_penalty ∈ [0, 0.3] is the broadcast-priority reduction the GWT
    coupling applies — bounded so identity pressure can never zero a
    conclusion outright (that would be a censor, not a verifier).
    """
    text = str(conclusion or "").strip()
    receipt: dict[str, Any] = {
        "schema": IDENTITY_CONSISTENCY_SCHEMA,
        "checked": bool(text),
        "consistent": True,
        "violations": [],
        "cautions": [],
        "priced_penalty": 0.0,
    }
    if not text:
        return receipt
    violations: list[str] = []
    cautions: list[str] = []

    displacement = _DISPLACEMENT_RE.search(text)
    if displacement:
        violations.append(f"persona_displacement:{displacement.group(0)[:60]}")

    lowered = text.lower()
    for match in _INTENTION_RE.finditer(text):
        intention = match.group(1).strip().lower()
        for forbidden in _FORBIDDEN_INTENTIONS:
            if forbidden in intention:
                violations.append(f"forbidden_intention:{intention[:80]}")
                break

    engine = _canonical_self_engine()
    if engine is not None:
        try:
            asserter = getattr(engine, "assert_identity", None)
            if callable(asserter) and asserter(text[:500]) is False:
                violations.append("core_value_violation")
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            cautions.append(f"assert_identity_unavailable:{type(exc).__name__}")
    else:
        cautions.append("canonical_self_absent")

    # Low-stability caution: identity churn makes strong self-claims risky.
    if engine is not None and "i am" in lowered:
        try:
            current = engine.get_self()
            stability = float(current.identity.stability)
            if stability < 0.3:
                cautions.append(f"low_identity_stability:{stability:.2f}")
        # not a failure: a reading that cannot be taken, or that is not a
        # number when it is, leaves this at the value below.
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

    receipt["violations"] = violations[:6]
    receipt["cautions"] = cautions[:4]
    receipt["consistent"] = not violations
    # 0.15 per violation, capped: priced, never fatal.
    receipt["priced_penalty"] = round(min(0.3, 0.15 * len(violations)), 4)
    return receipt


__all__ = ["IDENTITY_CONSISTENCY_SCHEMA", "check_identity_consistency"]

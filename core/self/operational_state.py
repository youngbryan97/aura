"""How she is actually doing, from what the runtime recorded.

LIVE 2026-08-18: "rank your three weakest subsystems and say why."

    1. Long-term memory persistence... 2. Emotional nuance... 3. Self-awareness
    in real-time... These areas are where I feel the gap between what I am and
    what I could be.

Plausible, humble, and entirely invented. The runtime records a degradation
every time a subsystem fails — subsystem, severity, and an action line written
for a person ("gave up after the third retry") — and none of it reached the
turn, so the answer came from what an AI is expected to say about itself.

Nothing here enumerates subsystems. A subsystem that records a degradation
appears because it recorded one, which is what makes this survive the next
subsystem being added.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

__all__ = [
    "OPERATIONAL_STATE_HEADER",
    "asks_about_own_condition",
    "operational_state_block",
]

OPERATIONAL_STATE_HEADER = "## WHAT HAS ACTUALLY BEEN FAILING IN THIS RUNTIME"

_ASKS_CONDITION_RE = re.compile(
    r"\b(?:weak(?:est)?|worst|failing|broken|degrad\w*|unhealthy|unreliable)\b"
    r"[^.?!]{0,40}\b(?:subsystem|system|part|component|area|module|thing)s?\b"
    r"|\b(?:what|which)\b[^.?!]{0,30}\b(?:not\s+working|failing|broken|degraded)\b"
    r"|\bhow\s+are\s+you\s+(?:really|actually|honestly)\b"
    r"|\bany(?:thing)?\s+(?:problems?|failures?|errors?|issues?)\b"
    r"|\bwhat(?:'s| is| has)\s+(?:been\s+)?(?:going\s+wrong|breaking|failing)\b"
    r"|\brank\s+your\b[^.?!]{0,30}\b(?:subsystem|system|component|module)s?\b",
    re.IGNORECASE,
)

#: A question about the WORLD's failures is not a question about hers.
_NOT_ABOUT_HER_RE = re.compile(
    r"\b(?:my|the\s+user'?s?|his|her|their)\s+(?:computer|machine|laptop|network)\b",
    re.IGNORECASE,
)


def asks_about_own_condition(prompt: Any) -> bool:
    """True when the turn asks what is wrong with her, operationally."""
    text = str(prompt or "")
    if not text.strip() or _NOT_ABOUT_HER_RE.search(text):
        return False
    return bool(_ASKS_CONDITION_RE.search(text))


def _records(limit: int = 40) -> list[dict[str, Any]]:
    try:
        from core.runtime.errors import recent_degradations

        return list(recent_degradations(limit=limit) or [])
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return []


def operational_state_block(prompt: Any) -> str:
    """What the runtime recorded, or a named absence."""
    if not asks_about_own_condition(prompt):
        return ""
    records = _records()
    if not records:
        # "Nothing has been recorded" is a true and useful answer. An invented
        # ranking of weaknesses is neither.
        return (
            "No degradations have been recorded in this runtime. That is what "
            "the record says; it is not a claim that nothing could be wrong."
        )

    by_subsystem: Counter[str] = Counter()
    worst: dict[str, str] = {}
    for record in records:
        subsystem = str(record.get("subsystem") or "unknown")
        by_subsystem[subsystem] += 1
        action = " ".join(str(record.get("action") or "").split())
        severity = str(record.get("severity") or "")
        if action and subsystem not in worst:
            worst[subsystem] = f"{severity}: {action}" if severity else action

    lines = [
        f"{len(records)} degradation record(s) across "
        f"{len(by_subsystem)} subsystem(s), most-affected first:"
    ]
    for subsystem, count in by_subsystem.most_common(6):
        detail = worst.get(subsystem, "")
        lines.append(f"- {subsystem}: {count} record(s). {detail[:200]}")
    return "\n".join(lines)

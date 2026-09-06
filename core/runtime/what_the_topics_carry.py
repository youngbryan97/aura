"""core/runtime/what_the_topics_carry.py — the declarations, one file, read at import.

Where the shapes live. `what_an_event_carries` is the mechanism and this is
the content, kept apart so that adding a declaration is an edit to a list
rather than an edit to a checker.

Declared here rather than beside each producer for one reason: a topic with
two producers has one shape, and a shape declared next to one of them is a
shape the other can disagree with silently. That is the failure this exists to
catch, so putting the declaration where both can see it is the point.
"""

from __future__ import annotations

from core.runtime.what_an_event_carries import WhatATopicCarries, declare

__all__ = ["declare_what_the_topics_carry"]


def declare_what_the_topics_carry() -> int:
    """Say what the topics with more than one consumer carry. Returns how many."""

    said = (
        WhatATopicCarries(
            topic="turn_recorded",
            version="1",
            means="one side of one exchange landed in the conversation store",
            requires={
                "role": str,
                "content": str,
                "session_id": str,
                "turn_id": (str, int),
            },
            allows={
                "origin": (str, type(None)),
                "cid": (str, type(None)),
                "content_chars": int,
            },
        ),
        WhatATopicCarries(
            topic="llm.endpoint_health",
            version="1",
            means="an endpoint changed state; routing may need to move",
            requires={"endpoint": str},
            allows={
                "healthy": bool,
                "reason": str,
                "latency_ms": (int, float),
                "at": (int, float),
                "status": str,
                "detail": object,
            },
        ),
        WhatATopicCarries(
            topic="will.decision",
            version="1",
            means="the will admitted or refused something, with why",
            requires={},
            allows={
                "decision": str,
                "action": str,
                "reason": str,
                "receipt_id": str,
                "allowed": bool,
                "at": (int, float),
                "detail": object,
            },
            closed=False,
        ),
    )
    for one in said:
        declare(one)
    return len(said)


declare_what_the_topics_carry()

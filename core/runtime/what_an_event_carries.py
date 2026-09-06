"""core/runtime/what_an_event_carries.py — the topic is an address, not a schema.

Two peer architectures make the same point. OpenHands has typed event classes
across its runtime; CrewAI dispatches typed events and catches schema mistakes
before dispatch. Aura's bus is ``publish(topic: str, data: Any)``, so a
consumer's only way to know what arrives on a topic is to read every producer,
and a producer that changes a key finds out when something downstream stops
working.

The topic stays a string. It is an address and it is a good one — string
topics are what make the bus cheap to extend and easy to route. What is added
is the other half: a topic may declare what its payload carries, and once it
does, publishing something else on it is a fault rather than a surprise.

Three properties, and the third is the one that makes this worth having:

* Declaring is opt-in. A bus that refused undeclared topics would stop the
  runtime the day it landed, and the topics worth declaring are the ones with
  more than one consumer.
* Validation is not. Once a topic declares, every publish on it is checked,
  and a violation is a recorded degradation rather than an exception — a bus
  that raises inside publish turns a consumer's schema mistake into a
  producer's crash.
* What is undeclared is counted. The number is the work remaining, and it is
  ratcheted downward by a test rather than left as an intention.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.WhatAnEventCarries")

__all__ = [
    "WhatATopicCarries",
    "check",
    "declare",
    "the_declared",
    "the_undeclared",
    "violations",
    "forget_what_was_seen",
]


@dataclass(frozen=True)
class WhatATopicCarries:
    """What a payload on this topic must and may contain."""

    topic: str
    #: Bumped when the shape changes in a way a consumer would notice.
    version: str
    #: Keys that must be present, with the type each must be. A tuple of types
    #: means any of them; None means the key must be present and may be
    #: anything, which is a weaker claim honestly stated.
    requires: Mapping[str, Any] = field(default_factory=dict)
    #: Keys that may be present, same shape. Anything not required and not
    #: allowed is reported — a payload growing a key nobody declared is how a
    #: schema drifts without anyone deciding to change it.
    allows: Mapping[str, Any] = field(default_factory=dict)
    #: One line: what a consumer can do with this.
    means: str = ""
    #: False where the payload is deliberately open. Then only the required
    #: keys are checked and extra keys are not reported.
    closed: bool = True


_LOCK = threading.Lock()
_DECLARED: dict[str, WhatATopicCarries] = {}
_SEEN_UNDECLARED: dict[str, int] = {}
_VIOLATIONS: list[dict[str, Any]] = []

#: How many violations are kept. A ring, because a producer in a loop would
#: otherwise turn one mistake into a memory leak.
HOW_MANY_VIOLATIONS_ARE_KEPT = 64


def declare(spec: WhatATopicCarries) -> WhatATopicCarries:
    """Say what a topic carries. Re-declaring with a new version is allowed."""

    with _LOCK:
        _DECLARED[spec.topic] = spec
    return spec


def the_declared() -> dict[str, WhatATopicCarries]:
    with _LOCK:
        return dict(_DECLARED)


def the_undeclared() -> dict[str, int]:
    """Topics that have been published on and say nothing about their payload."""

    with _LOCK:
        return dict(_SEEN_UNDECLARED)


def violations() -> list[dict[str, Any]]:
    with _LOCK:
        return list(_VIOLATIONS)


def forget_what_was_seen() -> None:
    """For tests. Declarations survive; observations do not."""

    with _LOCK:
        _SEEN_UNDECLARED.clear()
        _VIOLATIONS.clear()


def _wrong(key: str, value: Any, wanted: Any) -> str | None:
    if wanted is None:
        return None
    types = wanted if isinstance(wanted, tuple) else (wanted,)
    if isinstance(value, types):
        return None
    named = "/".join(getattr(one, "__name__", str(one)) for one in types)
    return f"{key} is {type(value).__name__}, declared {named}"


#: Whether the declarations have been read in this process. They live in their
#: own module so that adding one is an edit to a list; nothing imports that
#: module for its own sake, so the first check pulls it in.
_THE_SHAPES_ARE_KNOWN = False


def _make_sure_the_shapes_are_known() -> None:
    global _THE_SHAPES_ARE_KNOWN
    if _THE_SHAPES_ARE_KNOWN:
        return
    _THE_SHAPES_ARE_KNOWN = True
    try:
        import core.runtime.what_the_topics_carry  # noqa: F401  (declares on import)
    except Exception as exc:  # noqa: BLE001 — a missing declaration file is not a fault
        logger.debug("no topic declarations loaded: %s", exc)


def check(topic: str, payload: Any) -> tuple[bool, tuple[str, ...]]:
    """Whether this payload matches what the topic declared.

    True with no reasons for an undeclared topic: silence about a topic is not
    a claim that anything is wrong with it, and saying otherwise would make
    every publish on the bus look like a fault.
    """

    _make_sure_the_shapes_are_known()
    name = str(topic or "")
    with _LOCK:
        spec = _DECLARED.get(name)
        if spec is None:
            _SEEN_UNDECLARED[name] = _SEEN_UNDECLARED.get(name, 0) + 1
            return True, ()
    if not isinstance(payload, Mapping):
        # A declared topic carrying something that is not a mapping is the
        # commonest way a payload contract is broken, and the least visible.
        return False, (f"payload is {type(payload).__name__}, declared a mapping",)
    said: list[str] = []
    for key, wanted in spec.requires.items():
        if key not in payload:
            said.append(f"{key} is missing")
            continue
        wrong = _wrong(key, payload[key], wanted)
        if wrong:
            said.append(wrong)
    for key, value in payload.items():
        if key in spec.requires:
            continue
        if key in spec.allows:
            wrong = _wrong(key, value, spec.allows[key])
            if wrong:
                said.append(wrong)
        elif spec.closed:
            said.append(f"{key} is not declared on this topic")
    if said:
        with _LOCK:
            _VIOLATIONS.append(
                {"topic": name, "version": spec.version, "reasons": list(said)}
            )
            del _VIOLATIONS[:-HOW_MANY_VIOLATIONS_ARE_KEPT]
    return (not said), tuple(said)
